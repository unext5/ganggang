import numpy as np
import pandas as pd
import pickle
import logging
import os
from datetime import datetime
from scipy.stats import entropy
from sklearn.cluster import KMeans
from collections import Counter

# Importiere Basis-HMM-Funktionen
from enhanced_hmm_em_v2 import forward_backward, compute_confidence, validate_current_state

class HMMEnsemble:
    """
    Ensemble von HMM-Modellen für robustere Vorhersagen.
    Kombiniert mehrere HMM-Modelle mit verschiedenen Konfigurationen.
    Verbesserte Version mit dynamischen Gewichten und Kontextadaptivität.
    """
    def __init__(self, models=None, weights=None, ensemble_type="adaptive"):
        """
        Initialisiert das Ensemble.
        
        Args:
            models: Liste von HMM-Modellparametern (dict oder Tuple)
            weights: Optionale Gewichtungen für die Modelle
            ensemble_type: Art des Ensembles (voting, bayes, adaptive)
        """
        self.models = models if models else []
        self.n_models = len(self.models)
        self.ensemble_type = ensemble_type
        self.live_instances = []
        
        # Ensemble-Gewichte
        if weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            self.weights = np.array(weights) / sum(weights)
        
        # Modell-Performance-Tracking
        self.model_performance = {i: {"accuracy": 0.5, "recall_history": []} for i in range(self.n_models)}
        
        # Feature-Spaltennamen, falls in den Modellen vorhanden
        self.feature_cols = None
        if self.n_models > 0 and isinstance(self.models[0], dict) and "feature_cols" in self.models[0]:
            self.feature_cols = self.models[0]["feature_cols"]
        
        # Speichere kontextspezifische Gewichte
        self.context_weights = {}
        
        # Tracking für Modellverhalten in verschiedenen Marktphasen
        self.market_phase_performance = {
            "bullish": {i: {"correct": 0, "total": 0} for i in range(self.n_models)},
            "bearish": {i: {"correct": 0, "total": 0} for i in range(self.n_models)},
            "high_vol": {i: {"correct": 0, "total": 0} for i in range(self.n_models)},
            "low_vol": {i: {"correct": 0, "total": 0} for i in range(self.n_models)}
        }
        
        # Exponentiell gewichteter gleitender Durchschnitt für langfristiges Lernen
        self.ewma_weights = self.weights.copy()
        self.learning_rate = 0.05  # Lernrate für EWMA-Updates
        
        # Logging einrichten
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('hmm_ensemble')
    
    def add_model(self, model, weight=1.0):
        """
        Fügt ein Modell zum Ensemble hinzu.
        
        Args:
            model: HMM-Modellparameter (dict oder Tuple)
            weight: Gewicht des Modells im Ensemble
        """
        self.models.append(model)
        
        # Aktualisiere Gewichte
        old_sum = sum(self.weights)
        self.weights = np.append(self.weights, weight)
        self.weights = self.weights / (old_sum + weight)
        
        # Aktualisiere EWMA-Gewichte
        self.ewma_weights = np.append(self.ewma_weights, weight)
        self.ewma_weights = self.ewma_weights / (old_sum + weight)
        
        # Aktualisiere Anzahl der Modelle
        self.n_models = len(self.models)
        
        # Initialisiere Performance-Tracking für neues Modell
        self.model_performance[self.n_models-1] = {"accuracy": 0.5, "recall_history": []}
        
        # Initialisiere Marktphasen-Performance für neues Modell
        for phase in self.market_phase_performance:
            self.market_phase_performance[phase][self.n_models-1] = {"correct": 0, "total": 0}
    
    def initialize_live_instances(self, EnhancedLiveHMMMt5, dims_egarch=None):
        """
        Initialisiert Live-Instanzen für alle Modelle im Ensemble.
        
        Args:
            EnhancedLiveHMMMt5: Klasse für Live-HMM-Instanzen
            dims_egarch: Dimensionen für EGARCH
        
        Returns:
            List von initialisierten Live-HMM-Instanzen
        """
        self.live_instances = []
        
        for model in self.models:
            if isinstance(model, dict):
                # Modell ist in dict-Format
                instance = EnhancedLiveHMMMt5(
                    model, 
                    dims_egarch=dims_egarch,
                    feature_cols=model.get("feature_cols", self.feature_cols)
                )
            else:
                # Modell ist in Tuple-Format (pi, A, st_list, ll)
                pi, A, st_list, _ = model
                instance = EnhancedLiveHMMMt5(
                    {"K": len(st_list), "pi": pi, "A": A, "st_params": st_list},
                    dims_egarch=dims_egarch,
                    feature_cols=self.feature_cols
                )
            
            self.live_instances.append(instance)
        
        return self.live_instances
    
    def predict(self, feature_tminus1, feature_t, time_info=None, current_price=None):
        """
        Führt eine Ensemble-Vorhersage basierend auf den aktuellen Features durch.
        
        Args:
            feature_tminus1: Feature-Vektor t-1
            feature_t: Feature-Vektor t
            time_info: Zeitstempel
            current_price: Aktueller Preis
        
        Returns:
            dict: Ensemble-Vorhersage mit Konfidenz
        """
        if not self.live_instances:
            raise ValueError("Live-Instanzen nicht initialisiert. Bitte initialize_live_instances() aufrufen.")
        
        # Sammle Vorhersagen aller Modelle
        predictions = []
        
        for i, instance in enumerate(self.live_instances):
            try:
                result = instance.partial_step(feature_tminus1, feature_t, time_info, current_price)
                # Füge Modell-ID hinzu
                result["model_id"] = i
                predictions.append(result)
            except Exception as e:
                logging.warning(f"Fehler bei Modell {i}: {str(e)}")
        
        if not predictions:
            raise RuntimeError("Keine Vorhersagen von Modellen erhalten.")
        
        # Marktkontext extrahieren (für kontextabhängige Gewichtung)
        market_context = self._extract_market_context(feature_t, predictions)
        context_key = market_context["context_key"]
        
        # Ensemble-Methode auswählen
        if self.ensemble_type == "voting":
            ensemble_result = self._voting_ensemble(predictions)
        elif self.ensemble_type == "bayes":
            ensemble_result = self._bayesian_ensemble(predictions)
        elif self.ensemble_type == "adaptive":
            # Verwende kontextspezifische Gewichte falls vorhanden
            if context_key in self.context_weights:
                context_weights = self.context_weights[context_key]
                self.logger.info(f"Verwende kontextspezifische Gewichte für {context_key}")
                ensemble_result = self._adaptive_ensemble(predictions, feature_t, context_weights)
            else:
                ensemble_result = self._adaptive_ensemble(predictions, feature_t)
        else:
            # Fallback auf einfaches Voting
            ensemble_result = self._voting_ensemble(predictions)
        
        # Füge Marktkontext zur Ergebnisstruktur hinzu
        ensemble_result["market_context"] = market_context
        
        # Aktualisiere kontextspezifische Gewichte für zukünftige Vorhersagen
        self._update_context_weights(context_key, predictions, feature_t)
        
        return ensemble_result
    
    def _extract_market_context(self, features, predictions):
        """
        Extrahiert den Marktkontext aus Features und Vorhersagen.
        
        Args:
            features: Aktuelle Feature-Vektoren
            predictions: Modellvorhersagen
        
        Returns:
            dict: Marktkontext-Informationen
        """
        # Bestimme Marktrichtung aus Durchschnitt der Zustandslabels
        direction_count = {"bullish": 0, "bearish": 0, "neutral": 0}
        volatility_count = {"high": 0, "medium": 0, "low": 0}
        
        for pred in predictions:
            # Richtungs-Zählung
            if "Bullish" in pred.get("state_label", "") or "Bull" in pred.get("state_label", ""):
                direction_count["bullish"] += 1
            elif "Bearish" in pred.get("state_label", "") or "Bear" in pred.get("state_label", ""):
                direction_count["bearish"] += 1
            else:
                direction_count["neutral"] += 1
            
            # Volatilitäts-Zählung
            if "High" in pred.get("state_label", ""):
                volatility_count["high"] += 1
            elif "Low" in pred.get("state_label", ""):
                volatility_count["low"] += 1
            else:
                volatility_count["medium"] += 1
        
        # Mehrheitsentscheidung für Marktrichtung
        market_direction = max(direction_count, key=direction_count.get)
        
        # Mehrheitsentscheidung für Volatilitätsniveau
        volatility_level = max(volatility_count, key=volatility_count.get)
        
        # Feature-basierte Kontextanalyse
        feature_context = "normal"
        
        # Analysiere die wichtigsten Indikatoren aus den Features (Returns, Volatilität)
        if len(features) >= 9:  # Sicherstellen, dass genug Features vorhanden sind
            # Annahme: Die ersten 4 Features sind Returns, Features 6-8 enthalten Volatilitätsmetriken
            returns_avg = np.mean(features[:4])
            vol_indicators = np.mean(features[6:9]) if len(features) > 8 else 0
            
            if abs(returns_avg) > 0.005:  # Starke Bewegung
                feature_context = "trend" if returns_avg > 0 else "downtrend"
            elif vol_indicators > 0.02:  # Hohe Volatilität
                feature_context = "volatile"
        
        # Eindeutiger Kontext-Schlüssel für Gewichtungs-Cache
        context_key = f"{market_direction}_{volatility_level}_{feature_context}"
        
        return {
            "direction": market_direction,
            "volatility": volatility_level,
            "feature_context": feature_context,
            "context_key": context_key
        }
    
    def _update_context_weights(self, context_key, predictions, features):
        """
        Aktualisiert kontextspezifische Gewichte basierend auf neuen Daten.
        
        Args:
            context_key: Eindeutiger Kontext-Schlüssel
            predictions: Aktuelle Modellvorhersagen
            features: Aktuelle Features
        """
        # Berechne neue Gewichte basierend auf Konfidenz und vergangener Performance
        dynamic_weights = np.zeros(self.n_models)
        
        for i, pred in enumerate(predictions):
            if i >= self.n_models:
                continue
            
            # Basis-Gewichtung aus EWMA-Gewichten
            dynamic_weights[i] = self.ewma_weights[i]
            
            # Modifiziere basierend auf aktueller Konfidenz
            confidence = pred.get("state_confidence", 0.5)
            dynamic_weights[i] *= (0.5 + 0.5 * confidence)  # Zwischen 50-100% des Originalgewichts
        
        # Normalisiere Gewichte
        if np.sum(dynamic_weights) > 0:
            dynamic_weights = dynamic_weights / np.sum(dynamic_weights)
        else:
            dynamic_weights = self.weights.copy()
        
        # Speichere oder aktualisiere Kontext-Gewichte
        if context_key in self.context_weights:
            # Exponentiell gewichtetes Update
            alpha = 0.1  # Lernrate
            old_weights = self.context_weights[context_key]
            updated_weights = (1 - alpha) * old_weights + alpha * dynamic_weights
            self.context_weights[context_key] = updated_weights
        else:
            # Neuer Kontext
            self.context_weights[context_key] = dynamic_weights
    
    def _voting_ensemble(self, predictions):
        """
        Mehrheitsabstimmung der Modelle im Ensemble.
        
        Args:
            predictions: Liste der Modellvorhersagen
        
        Returns:
            dict: Beste Ensemble-Vorhersage
        """
        # Gewichte berücksichtigen
        state_votes = {}
        
        for i, pred in enumerate(predictions):
            state = pred["state_idx"]
            confidence = pred["state_confidence"]
            weight = self.weights[pred["model_id"]]
            
            # Gewichtetes Voting
            vote_strength = weight * confidence
            
            if state not in state_votes:
                state_votes[state] = {
                    "total_weight": vote_strength,
                    "predictions": [pred],
                    "confidences": [confidence]
                }
            else:
                state_votes[state]["total_weight"] += vote_strength
                state_votes[state]["predictions"].append(pred)
                state_votes[state]["confidences"].append(confidence)
        
        # Bestimme den Gewinner-Zustand
        winner_state = max(state_votes, key=lambda s: state_votes[s]["total_weight"])
        
        # Berechne Ensemble-Statistiken
        winner_info = state_votes[winner_state]
        winner_confidence = np.mean(winner_info["confidences"])
        
        # Wähle die Vorhersage mit der höchsten Konfidenz als Repräsentant
        best_pred_idx = np.argmax(winner_info["confidences"])
        representative_pred = winner_info["predictions"][best_pred_idx].copy()
        
        # Überarbeite die Vorhersage mit Ensemble-Informationen
        representative_pred["ensemble_confidence"] = winner_confidence
        representative_pred["ensemble_vote_weight"] = winner_info["total_weight"]
        representative_pred["ensemble_agreement"] = len(winner_info["predictions"]) / len(predictions)
        representative_pred["ensemble_state"] = winner_state
        representative_pred["ensemble_method"] = "voting"
        
        return representative_pred
    
    def _bayesian_ensemble(self, predictions):
        """
        Bayesianisches Ensemble: Kombiniert Modelle basierend auf Posteriori-Wahrscheinlichkeiten.
        
        Args:
            predictions: Liste der Modellvorhersagen
        
        Returns:
            dict: Bayesianische Ensemble-Vorhersage
        """
        # Sammle alle möglichen Zustände
        all_states = set()
        for pred in predictions:
            all_states.add(pred["state_idx"])
        
        # Initialisiere Prior (gleichverteilt)
        prior = {state: 1.0 / len(all_states) for state in all_states}
        
        # Berechne Posterior mit Bayes' Theorem
        posterior = prior.copy()
        
        for i, pred in enumerate(predictions):
            state = pred["state_idx"]
            confidence = pred["state_confidence"]
            weight = self.weights[pred["model_id"]]
            
            # Likelihood basierend auf Konfidenz und Gewicht
            likelihood = confidence * weight
            
            # Aktualisiere Posterior
            posterior[state] *= likelihood
        
        # Normalisiere Posterior
        total = sum(posterior.values())
        if total > 0:
            for state in posterior:
                posterior[state] /= total
        
        # Finde den Zustand mit dem höchsten Posterior
        best_state = max(posterior, key=posterior.get)
        
        # Wähle ein Modell, das diesen Zustand vorhergesagt hat
        best_preds = [p for p in predictions if p["state_idx"] == best_state]
        if best_preds:
            # Wähle das Modell mit der höchsten Konfidenz
            representative_pred = max(best_preds, key=lambda p: p["state_confidence"]).copy()
        else:
            # Fallback: Modell mit der höchsten gewichteten Konfidenz
            representative_pred = max(predictions, 
                                    key=lambda p: p["state_confidence"] * self.weights[p["model_id"]]).copy()
        
        # Füge Ensemble-Informationen hinzu
        representative_pred["ensemble_confidence"] = posterior[best_state]
        representative_pred["ensemble_posterior"] = posterior
        representative_pred["ensemble_state"] = best_state
        representative_pred["ensemble_method"] = "bayes"
        
        return representative_pred
    
    def _adaptive_ensemble(self, predictions, current_features, context_weights=None):
        """
        Adaptives Ensemble: Wählt Modelle basierend auf aktueller Performance und Kontextähnlichkeit.
        
        Args:
            predictions: Liste der Modellvorhersagen
            current_features: Aktuelle Features für Kontextanalyse
            context_weights: Optionale kontextspezifische Gewichte
        
        Returns:
            dict: Adaptive Ensemble-Vorhersage
        """
        # Feature-Kontext verwenden, um ähnliche historische Situationen zu finden
        
        # Aktualisiere Performance-Metriken für alle Modelle
        for pred in predictions:
            model_id = pred["model_id"]
            # Berechne Feature-Entropy als Proxy für Unsicherheit
            if current_features.size > 0:
                normalized_features = current_features / (np.max(np.abs(current_features)) + 1e-10)
                feature_entropy = entropy(np.abs(normalized_features) + 1e-10)
            else:
                feature_entropy = 0
            
            # Verwende Validitätswert und Konfidenz
            validity = pred.get("state_validity", 0.5)
            confidence = pred["state_confidence"]
            
            # Aktualisiere das Performance-Tracking
            self.model_performance[model_id]["recall_history"].append({
                "validity": validity,
                "confidence": confidence,
                "entropy": feature_entropy
            })
            
            # Behalte nur die letzten 100 Werte
            if len(self.model_performance[model_id]["recall_history"]) > 100:
                self.model_performance[model_id]["recall_history"].pop(0)
        
        # Berechne dynamische Gewichte basierend auf Feature-Kontext
        dynamic_weights = np.zeros(self.n_models)
        
        # Verwende kontextspezifische Gewichte, falls vorhanden
        if context_weights is not None and len(context_weights) == self.n_models:
            dynamic_weights = context_weights.copy()
        else:
            # Berechne Gewichte basierend auf aktueller Performance
            for i in range(self.n_models):
                if i >= len(predictions) or predictions[i]["model_id"] != i:
                    continue
                    
                # Basis-Gewicht
                dynamic_weights[i] = self.ewma_weights[i]
                
                # Performance-Historie abrufen
                history = self.model_performance[i]["recall_history"]
                if not history:
                    continue
                
                # Berechne durchschnittliche Performance in ähnlichen Situationen
                recent_validity = np.mean([h["validity"] for h in history[-10:]])
                
                # Aktualisiere dynamisches Gewicht
                dynamic_weights[i] *= (0.5 + recent_validity)
        
        # Normalisiere dynamische Gewichte
        if np.sum(dynamic_weights) > 0:
            dynamic_weights = dynamic_weights / np.sum(dynamic_weights)
        else:
            dynamic_weights = self.weights.copy()
        
        # Aktualisiere EWMA-Gewichte für langfristiges Lernen
        self.ewma_weights = (1 - self.learning_rate) * self.ewma_weights + self.learning_rate * dynamic_weights
        
        # Gewichtete Abstimmung mit dynamischen Gewichten
        state_votes = {}
        
        for i, pred in enumerate(predictions):
            state = pred["state_idx"]
            confidence = pred["state_confidence"]
            model_id = pred["model_id"]
            weight = dynamic_weights[model_id]
            
            # Gewichtetes Voting
            vote_strength = weight * confidence
            
            if state not in state_votes:
                state_votes[state] = {
                    "total_weight": vote_strength,
                    "predictions": [pred],
                    "confidences": [confidence]
                }
            else:
                state_votes[state]["total_weight"] += vote_strength
                state_votes[state]["predictions"].append(pred)
                state_votes[state]["confidences"].append(confidence)
        
        # Bestimme den Gewinner-Zustand
        winner_state = max(state_votes, key=lambda s: state_votes[s]["total_weight"])
        
        # Berechne Ensemble-Statistiken
        winner_info = state_votes[winner_state]
        winner_confidence = np.mean(winner_info["confidences"])
        
        # Wähle die Vorhersage mit der höchsten Konfidenz als Repräsentant
        best_pred_idx = np.argmax(winner_info["confidences"])
        representative_pred = winner_info["predictions"][best_pred_idx].copy()
        
        # Überarbeite die Vorhersage mit Ensemble-Informationen
        representative_pred["ensemble_confidence"] = winner_confidence
        representative_pred["ensemble_dynamic_weights"] = dynamic_weights.tolist()
        representative_pred["ensemble_state"] = winner_state
        representative_pred["ensemble_method"] = "adaptive"
        
        return representative_pred
    
    def update_weights_from_outcome(self, model_predictions, true_state, market_context=None):
        """
        Aktualisiert Modellgewichte basierend auf dem tatsächlichen Ergebnis.
        
        Args:
            model_predictions: Vorhersagen der einzelnen Modelle
            true_state: Tatsächlicher Zustand (Ground Truth)
            market_context: Optionale Marktkontext-Information
        """
        # Belohnungen für korrekte/falsche Vorhersagen
        reward_correct = 0.1
        penalty_incorrect = 0.05
        
        # Aktualisierungsvektoren
        weight_updates = np.zeros(self.n_models)
        
        # Aktualisiere Gewichte basierend auf Vorhersagegenauigkeit
        for pred in model_predictions:
            model_id = pred["model_id"]
            predicted_state = pred["state_idx"]
            
            # Überprüfe Korrektheit
            is_correct = (predicted_state == true_state)
            
            # Update Marktphasen-Performance, falls Kontext vorhanden
            if market_context:
                direction = market_context.get("direction", "neutral")
                volatility = market_context.get("volatility", "medium")
                
                # Aktualisiere entsprechende Phasen-Statistiken
                if direction in ("bullish", "bearish"):
                    self.market_phase_performance[direction][model_id]["total"] += 1
                    if is_correct:
                        self.market_phase_performance[direction][model_id]["correct"] += 1
                
                # Volatilitäts-basierte Performance
                vol_key = "high_vol" if volatility == "high" else "low_vol" if volatility == "low" else None
                if vol_key:
                    self.market_phase_performance[vol_key][model_id]["total"] += 1
                    if is_correct:
                        self.market_phase_performance[vol_key][model_id]["correct"] += 1
            
            # Berechne Gewichtsänderung
            if is_correct:
                weight_updates[model_id] = reward_correct
            else:
                weight_updates[model_id] = -penalty_incorrect
        
        # Normalisiere Updates um Summe = 0 sicherzustellen
        if np.sum(np.abs(weight_updates)) > 0:
            # Berechne Skalierungsfaktor, um Summe zu erhalten
            scale_factor = np.sum(np.abs(weight_updates)) / 2
            weight_updates = weight_updates / scale_factor
        
        # Aktualisiere Gewichte
        new_weights = self.weights + weight_updates
        
        # Begrenze auf positive Werte und normalisiere
        new_weights = np.maximum(new_weights, 0.01)  # Mindestgewicht 1%
        self.weights = new_weights / np.sum(new_weights)
        
        # Aktualisiere auch EWMA-Gewichte für langfristiges Lernen
        self.ewma_weights = (1 - self.learning_rate) * self.ewma_weights + self.learning_rate * self.weights
        
        # Aktualisiere kontextspezifische Gewichte, falls Kontext vorhanden
        if market_context and "context_key" in market_context:
            context_key = market_context["context_key"]
            if context_key in self.context_weights:
                self.context_weights[context_key] = self.weights.copy()
    
    def update_weights(self, performance_data):
        """
        Aktualisiert die Modellgewichte basierend auf Performance-Daten.
        
        Args:
            performance_data: Dict mit Performance-Metriken pro Modell
        """
        new_weights = np.zeros(self.n_models)
        
        for i in range(self.n_models):
            if i in performance_data:
                # Verwende Win-Rate oder andere Performance-Metriken
                new_weights[i] = performance_data[i].get("win_rate", 0.5)
            else:
                # Behalte das aktuelle Gewicht bei
                new_weights[i] = self.weights[i]
        
        # Normalisiere neue Gewichte
        if np.sum(new_weights) > 0:
            self.weights = new_weights / np.sum(new_weights)
    
    def get_context_specific_weights(self, market_phase=None, volatility_level=None):
        """
        Gibt optimierte Gewichte für spezifischen Marktkontext zurück.
        
        Args:
            market_phase: Marktphase (bullish, bearish, neutral)
            volatility_level: Volatilitätsniveau (high, medium, low)
        
        Returns:
            numpy.array: Optimierte Gewichte
        """
        # Wenn kein spezifischer Kontext angegeben, aktuelle Gewichte zurückgeben
        if market_phase is None and volatility_level is None:
            return self.weights
        
        # Berechne optimierte Gewichte basierend auf historischer Performance
        optimized_weights = np.ones(self.n_models) / self.n_models
        
        # Marktphasen-spezifische Gewichte
        if market_phase in ("bullish", "bearish"):
            phase_data = self.market_phase_performance[market_phase]
            
            for i in range(self.n_models):
                total = phase_data[i]["total"]
                if total > 0:
                    win_rate = phase_data[i]["correct"] / total
                    # Verwende Win-Rate als Basis für Gewicht, aber glätte für Stabilität
                    optimized_weights[i] = 0.3 + 0.7 * win_rate
        
        # Volatilitäts-spezifische Gewichte
        vol_key = None
        if volatility_level == "high":
            vol_key = "high_vol"
        elif volatility_level == "low":
            vol_key = "low_vol"
        
        if vol_key:
            vol_data = self.market_phase_performance[vol_key]
            vol_weights = np.ones(self.n_models) / self.n_models
            
            for i in range(self.n_models):
                total = vol_data[i]["total"]
                if total > 0:
                    win_rate = vol_data[i]["correct"] / total
                    vol_weights[i] = 0.3 + 0.7 * win_rate
            
            # Kombiniere Marktphasen- und Volatilitätsgewichte
            optimized_weights = (optimized_weights + vol_weights) / 2
        
        # Normalisiere
        optimized_weights = optimized_weights / np.sum(optimized_weights)
        
        return optimized_weights
    
    def save(self, filepath="hmm_ensemble.pkl"):
        """
        Speichert das Ensemble-Modell.
        
        Args:
            filepath: Pfad zum Speichern des Modells
        """
        # Live-Instanzen nicht speichern
        save_data = {
            "models": self.models,
            "weights": self.weights,
            "ewma_weights": self.ewma_weights,
            "ensemble_type": self.ensemble_type,
            "model_performance": self.model_performance,
            "market_phase_performance": self.market_phase_performance,
            "context_weights": self.context_weights,
            "feature_cols": self.feature_cols,
            "saved_time": datetime.now().isoformat()
        }
        
        # Sichere Speicherung mit temporärer Datei
        temp_file = f"{filepath}.tmp"
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Bei Erfolg, verschiebe zum endgültigen Ziel
            if os.path.exists(filepath):
                os.replace(filepath, f"{filepath}.bak")  # Backup
            
            os.replace(temp_file, filepath)
            logging.info(f"Ensemble-Modell gespeichert in {filepath}")
        except Exception as e:
            logging.error(f"Fehler beim Speichern des Ensemble-Modells: {str(e)}")
            # Direktes Speichern als Fallback
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(save_data, f)
                logging.info(f"Ensemble-Modell gespeichert in {filepath} (Fallback)")
            except Exception as e2:
                logging.error(f"Kritischer Fehler beim Speichern: {str(e2)}")
    
    @classmethod
    def load(cls, filepath="hmm_ensemble.pkl"):
        """
        Lädt ein gespeichertes Ensemble-Modell.
        
        Args:
            filepath: Pfad zum gespeicherten Modell
        
        Returns:
            HMMEnsemble: Geladenes Ensemble-Modell
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Erstelle neue Instanz
            ensemble = cls(models=data.get("models", []), 
                          weights=data.get("weights"), 
                          ensemble_type=data.get("ensemble_type", "voting"))
            
            # Lade zusätzliche Daten
            ensemble.model_performance = data.get("model_performance", {})
            ensemble.market_phase_performance = data.get("market_phase_performance", {})
            ensemble.context_weights = data.get("context_weights", {})
            ensemble.feature_cols = data.get("feature_cols")
            ensemble.ewma_weights = data.get("ewma_weights", ensemble.weights.copy())
            
            logging.info(f"Ensemble-Modell geladen aus {filepath}, gespeichert am {data.get('saved_time', 'unbekannt')}")
            
            return ensemble
        except Exception as e:
            logging.error(f"Fehler beim Laden des Ensemble-Modells: {str(e)}")
            return None

# Hilfsfunktionen für das Ensemble-Training

def create_ensemble_from_k_values(features, K_values, train_hmm_once, 
                                 use_tdist=True, dims_egarch=None, times=None,
                                 weights_method="equal"):
    """
    Erstellt ein Ensemble von HMM-Modellen mit verschiedenen K-Werten.
    
    Args:
        features: Feature-Matrix
        K_values: Liste von K-Werten (Anzahl der Zustände)
        train_hmm_once: Funktion zum HMM-Training
        use_tdist: T-Verteilung verwenden
        dims_egarch: Dimensionen für EGARCH
        times: Zeitpunkte für zeitvariable Übergänge
        weights_method: Methode zur Gewichtsberechnung (equal, likelihood, crossval)
    
    Returns:
        ensemble: HMMEnsemble mit trainierten Modellen
    """
    models = []
    likelihoods = []
    
    for K in K_values:
        pi, A, st_list, ll = train_hmm_once(
            features, K, n_starts=5, max_iter=20, 
            use_tdist=use_tdist, dims_egarch=dims_egarch, 
            times=times
        )
        
        models.append((pi, A, st_list, ll))
        likelihoods.append(ll)
    
    # Berechne Gewichte
    if weights_method == "likelihood":
        # Normalisiere Log-Likelihoods
        ll_array = np.array(likelihoods)
        ll_min = np.min(ll_array)
        normalized_ll = ll_array - ll_min
        weights = np.exp(normalized_ll)
        weights = weights / np.sum(weights)
    elif weights_method == "crossval":
        # Einfache Kreuzvalidierung (hier: nur simuliert)
        weights = np.linspace(0.5, 1.0, len(K_values))
        weights = weights / np.sum(weights)
    else:
        # Gleichgewichtung
        weights = np.ones(len(K_values)) / len(K_values)
    
    # Erstelle Ensemble
    ensemble = HMMEnsemble(models, weights, ensemble_type="voting")
    
    return ensemble

def create_diversified_ensemble(features, base_K, train_hmm_once, 
                               n_models=3, use_tdist=True, dims_egarch=None, 
                               times=None, use_bayes=False):
    """
    Erstellt ein diversifiziertes Ensemble mit verschiedenen Modellparametern.
    
    Args:
        features: Feature-Matrix
        base_K: Basis-Anzahl der Zustände
        train_hmm_once: Funktion zum HMM-Training
        n_models: Anzahl der Modelle im Ensemble
        use_tdist: T-Verteilung verwenden
        dims_egarch: Dimensionen für EGARCH
        times: Zeitpunkte für zeitvariable Übergänge
        use_bayes: Bayesianisches Ensemble verwenden
    
    Returns:
        ensemble: Diversifiziertes HMMEnsemble
    """
    models = []
    likelihoods = []
    
    # Basis-Modell
    pi, A, st_list, ll = train_hmm_once(
        features, base_K, n_starts=5, max_iter=20, 
        use_tdist=use_tdist, dims_egarch=dims_egarch, times=times
    )
    
    models.append((pi, A, st_list, ll))
    likelihoods.append(ll)
    
    # Diversifizierte Modelle
    for i in range(1, n_models):
        # Variiere Parameter
        varied_K = max(2, base_K + np.random.randint(-1, 2))  # K ± 1
        varied_tdist = np.random.choice([True, False], p=[0.7, 0.3])  # 70% T-Verteilung
        
        # Subsample Features für mehr Diversität
        T, D = features.shape
        feature_mask = np.ones(D, dtype=bool)
        
        # Zufällig einige Features weglassen (max 20%)
        if D > 5:  # Nur wenn genug Features
            n_drop = np.random.randint(0, min(int(D * 0.2) + 1, D-4))
            drop_indices = np.random.choice(D, size=n_drop, replace=False)
            feature_mask[drop_indices] = False
        
        # Trainiere variiertes Modell
        varied_features = features[:, feature_mask]
        
        # Angepasste dims_egarch, falls vorhanden
        varied_dims_egarch = None
        if dims_egarch is not None:
            varied_dims_egarch = [i for i, keep in enumerate(feature_mask) if keep]
            
        pi_var, A_var, st_list_var, ll_var = train_hmm_once(
            varied_features, varied_K, n_starts=3, max_iter=15, 
            use_tdist=varied_tdist, dims_egarch=varied_dims_egarch, times=times
        )
        
        models.append((pi_var, A_var, st_list_var, ll_var))
        likelihoods.append(ll_var)
    
    # Berechne Gewichte
    weights = np.exp(np.array(likelihoods) - min(likelihoods))
    weights = weights / np.sum(weights)
    
    # Erstelle Ensemble
    ensemble_type = "bayes" if use_bayes else "adaptive"
    ensemble = HMMEnsemble(models, weights, ensemble_type=ensemble_type)
    
    return ensemble
