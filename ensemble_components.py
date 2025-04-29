import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from collections import deque
import logging
import pickle
import os
from datetime import datetime
import json
import time
import math

class AdaptiveComponentEnsemble:
    """
    Implementiert ein adaptives Ensemble verschiedener Trading-Komponenten
    (HMM, Hybrid-Modell, Market Memory, Order Book Analyzer).
    
    Die Klasse ermöglicht:
    1. Gewichtete Kombination von Komponenten-Signalen
    2. Dynamische Anpassung der Gewichte basierend auf Performance
    3. Meta-Lernen über die beste Kombination in verschiedenen Marktzuständen
    4. Spezielle Behandlung widersprüchlicher Signale
    """
    def __init__(self, components=None, initial_weights=None, history_size=500,
                 learning_rate=0.15, state_specific=True, model_path="models/ensemble_components.pkl"):
        """
        Initialisiert das Komponenten-Ensemble.
        
        Args:
            components: Liste der verfügbaren Komponenten
            initial_weights: Anfangsgewichte für Komponenten
            history_size: Größe des Signalverlaufs für Performancetracking
            learning_rate: Lernrate für Gewichtsanpassung
            state_specific: Ob zustandsspezifische Gewichte verwendet werden sollen
            model_path: Pfad zum Speichern/Laden des Modells
        """
        # Verfügbare Komponenten
        self.components = components or [
            "hmm", "hybrid_model", "market_memory", "order_book"
        ]
        
        # Anfangsgewichte (falls nicht angegeben, gleiche Gewichte)
        self.initial_weights = initial_weights or {
            comp: 1.0/len(self.components) for comp in self.components
        }
        
        # Gewichte für verschiedene Komponenten
        self.weights = self.initial_weights.copy()
        
        # Zustandsspezifische Gewichte
        self.state_specific = state_specific
        self.state_weights = {}  # {state_label: {component: weight}}
        
        # Konfiguration
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.min_weight = 0.1  # Minimales Gewicht für jede Komponente
        
        # Signalverlauf und Performancemetrik
        self.signal_history = deque(maxlen=history_size)
        self.performance_metrics = {
            comp: {"correct": 0, "total": 0} for comp in self.components
        }
        
        # Konfliktlösungsmatrix
        self.conflict_resolution = self._initialize_conflict_resolution()
        
        # Meta-Lernkomponente
        self.meta_learner = {
            "state_transitions": {},  # {from_state: {to_state: count}}
            "state_performance": {},  # {state: {component: performance}}
            "volatility_impact": {},  # {volatility_level: {component: performance}}
        }
        
        # Zähler für kontinuierliches Lernen
        self.update_counter = 0
        self.last_major_update = datetime.now()
        
        # Logger setup
        logging.basicConfig(level=logging.DEBUG,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('component_ensemble')
        
        # Lade gespeichertes Modell, falls vorhanden
        self.load_model()
    
    def _initialize_conflict_resolution(self):
        """
        Initialisiert die Konfliktlösungsmatrix für gegensätzliche Signale.
        
        Returns:
            dict: Matrix für Konfliktlösung
        """
        # Basisstrategie: Jede Kombination von widersprüchlichen Signalen
        # und wie sie aufgelöst werden sollen
        resolution = {
            ("hmm", "hybrid_model"): {
                "weight": 0.5,  # Anfängliches Gewicht für HMM über Hybrid
                "rule": "weighted_vote"  # Regel zur Konfliktlösung
            },
            ("hmm", "market_memory"): {
                "weight": 0.6,  # Anfängliches Gewicht für HMM über Memory
                "rule": "weighted_vote"
            },
            ("hmm", "order_book"): {
                "weight": 0.7,  # Anfängliches Gewicht für HMM über OB
                "rule": "weighted_vote"
            },
            ("hybrid_model", "market_memory"): {
                "weight": 0.55,  # Anfängliches Gewicht für Hybrid über Memory
                "rule": "weighted_vote"
            },
            ("hybrid_model", "order_book"): {
                "weight": 0.6,  # Anfängliches Gewicht für Hybrid über OB
                "rule": "weighted_vote"
            },
            ("market_memory", "order_book"): {
                "weight": 0.5,  # Anfängliches Gewicht für Memory über OB
                "rule": "weighted_vote"
            }
        }
        
        # Füge umgekehrte Paare hinzu
        reverse_pairs = {}
        for (comp1, comp2), settings in resolution.items():
            reverse_pairs[(comp2, comp1)] = {
                "weight": 1.0 - settings["weight"],
                "rule": settings["rule"]
            }
        
        resolution.update(reverse_pairs)
        
        return resolution
    
    def update_market_state(self, state_label=None, volatility="medium", 
                           trend="neutral", session="unknown"):
        """
        Aktualisiert den aktuellen Marktzustand für zustandsspezifische Gewichte.
        
        Args:
            state_label: HMM-Zustandslabel
            volatility: Volatilitätslevel ('low', 'medium', 'high')
            trend: Markttrend ('bullish', 'bearish', 'neutral')
            session: Marktsession ('asian', 'european', 'us', 'overlap')
            
        Returns:
            dict: Aktuelle Gewichte für diesen Zustand
        """
        # Aktualisiere Zustandsgewichte, falls nötig
        if self.state_specific and state_label:
            # Erstelle Zustandsgewichte, falls noch nicht vorhanden
            if state_label not in self.state_weights:
                self.state_weights[state_label] = self.weights.copy()
                self.logger.info(f"Created new state weights for '{state_label}'")
            
            current_weights = self.state_weights[state_label]
        else:
            current_weights = self.weights
        
        # Passe Gewichte basierend auf zusätzlichen Marktfaktoren an
        adjusted_weights = self._adjust_weights_for_market_conditions(
            current_weights, volatility, trend, session
        )
        
        return adjusted_weights
    
    def _adjust_weights_for_market_conditions(self, weights, volatility, trend, session):
        """
        Passt Komponentengewichte basierend auf aktuellen Marktbedingungen an.
        
        Args:
            weights: Basis-Komponentengewichte
            volatility: Volatilitätslevel
            trend: Markttrend
            session: Marktsession
            
        Returns:
            dict: Angepasste Gewichte
        """
        adjusted = weights.copy()
        
        # Anpassung für Volatilität
        if volatility == "high":
            # HMM und Hybrid tendenziell besser in hoher Volatilität
            if "hmm" in adjusted:
                adjusted["hmm"] *= 1.2
            if "hybrid_model" in adjusted:
                adjusted["hybrid_model"] *= 1.1
            if "order_book" in adjusted:
                adjusted["order_book"] *= 0.8
        elif volatility == "low":
            # Market Memory tendenziell besser in niedriger Volatilität
            if "market_memory" in adjusted:
                adjusted["market_memory"] *= 1.2
            if "order_book" in adjusted:
                adjusted["order_book"] *= 1.1
        
        # Anpassung für Trend
        if trend == "bullish" or trend == "bearish":
            # Klare Trends: HMM und Hybrid bevorzugen
            if "hmm" in adjusted:
                adjusted["hmm"] *= 1.1
            if "hybrid_model" in adjusted:
                adjusted["hybrid_model"] *= 1.1
        else:  # neutral
            # Seitwärtsbewegung: Order Book und Memory bevorzugen
            if "market_memory" in adjusted:
                adjusted["market_memory"] *= 1.1
            if "order_book" in adjusted:
                adjusted["order_book"] *= 1.1
        
        # Anpassung für Marktsession
        if session == "overlap":
            # Überlappungssessions: höhere Liquidiät, Order Book nützlicher
            if "order_book" in adjusted:
                adjusted["order_book"] *= 1.2
        elif session == "asian":
            # Asiatische Session: oft weniger Volatilität
            if "market_memory" in adjusted:
                adjusted["market_memory"] *= 1.1
        
        # Normalisiere die angepassten Gewichte
        weight_sum = sum(adjusted.values())
        for comp in adjusted:
            adjusted[comp] /= weight_sum
        
        return adjusted
    
    def ensemble_signal(self, component_signals, state_label=None, 
                       volatility="medium", trend="neutral", session="unknown",
                       current_price=None, current_time=None):
        """
        Kombiniert Signale von verschiedenen Komponenten zu einem Ensemble-Signal.
        
        Args:
            component_signals: Dictionary mit {'component': signal_info}
            state_label: HMM-Zustandslabel
            volatility: Volatilitätslevel
            trend: Markttrend
            session: Marktsession
            current_price: Aktueller Preis
            current_time: Aktuelle Zeit
            
        Returns:
            dict: Ensemble-Signalinformationen
        """
        # --- NEU: Überprüfung auf Feature-Dimensionsprobleme ---
        # Prüfe, ob Features der richtigen Dimension in den Komponenten sind
        for comp_name, signal_info in component_signals.items():
            if "features" in signal_info:
                features = signal_info["features"]
                if isinstance(features, np.ndarray) and features.shape[1] > 19 and comp_name in ["hmm", "hybrid_model"]:
                    self.logger.info(f"Reduziere Feature-Dimension für {comp_name} von {features.shape[1]} auf 19")
                    # Nur die ersten 19 Features behalten (Standard-HMM-Dimensionen)
                    signal_info["features"] = features[:, :19]
        # --- ENDE der Überprüfung ---
        
        # Prüfe, ob es überhaupt Komponenten-Signale gibt
        if not component_signals:
            self.logger.warning("Ensemble Signal - No component signals provided")
            return {
                "signal": "NEUTRAL",
                "strength": 0.1,
                "component_contributions": {},
                "conflicts": [],
                "weights": {},
                "state": state_label
            }
            
        # Hole aktuelle Gewichte für diesen Zustand
        weights = self.update_market_state(state_label, volatility, trend, session)
        self.logger.debug(f"Ensemble Signal - State: {state_label}, Adjusted Weights: {weights}") # LOG ADDED
        self.logger.debug(f"Ensemble Signal - Input Component Signals: {component_signals}") # LOG ADDED
        
        # Initialisiere Signalwerte - with FIXED non-zero default for NEUTRAL to avoid the 0.2381 issue
        signal_scores = {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.01}
        component_contributions = {}
        
        # Verarbeite Signale von jeder Komponente
        for comp, signal_info in component_signals.items():
            if comp not in self.components:
                continue
            
            # Extrahiere Signaltyp und -stärke (robuster)
            signal_type = signal_info.get("signal", "NEUTRAL")
            
            # Prioritize 'strength', fallback to 'confidence', then default to 1.0
            signal_strength = signal_info.get("strength") 
            if signal_strength is None: # Check if 'strength' key exists and is not None
                signal_strength = signal_info.get("confidence") # Fallback to 'confidence'
            
            # Ensure strength is a valid float
            try:
                signal_strength = float(signal_strength) if signal_strength is not None else None
            except (ValueError, TypeError):
                 self.logger.warning(f"Invalid strength/confidence value for {comp}: {signal_strength}. Using None.")
                 signal_strength = None
                 
            # Handle the signal type with appropriate default strength
            if signal_type == "NEUTRAL":
                # For NEUTRAL signals, use a lower default strength
                if signal_strength is None:
                    signal_strength = 0.15  # Lower default for NEUTRAL
            else:
                # For directional signals (BUY/SELL), use standard default
                if signal_strength is None:
                    signal_strength = 0.8  # Higher default for directional signals
                
            # Ensure strength is non-negative
            if signal_strength < 0:
                 self.logger.warning(f"Negative strength value for {comp}: {signal_strength}. Using absolute value.")
                 signal_strength = abs(signal_strength)
            
            # Cap maximum strength
            if signal_strength > 1.0:
                signal_strength = 1.0
                 
            # Standardisiere Signaltyp
            if signal_type in ["LONG", "UP", "BULLISH"]:
                signal_type = "BUY"
            elif signal_type in ["SHORT", "DOWN", "BEARISH"]:
                signal_type = "SELL"
            else:
                signal_type = "NEUTRAL"
            
            # Gewichteter Beitrag dieser Komponente
            contribution = weights.get(comp, 0) * signal_strength
            
            # Addiere zum entsprechenden Signalwert
            signal_scores[signal_type] += contribution
            
            # LOG ADDED - Log component details
            self.logger.debug(f"  Comp: {comp}, Signal: {signal_type}, Strength: {signal_strength:.4f}, "
                              f"Weight: {weights.get(comp, 0):.4f}, Contribution: {contribution:.4f}")
            
            # Speichere Komponentenbeitrag für Transparenz
            component_contributions[comp] = {
                "signal": signal_type,
                "strength": signal_strength,
                "weight": weights.get(comp, 0),
                "contribution": contribution
            }
        
        # Erkenne mögliche Konflikte
        conflicts = self._identify_conflicts(component_signals)
        
        # Löse Konflikte, falls vorhanden
        if conflicts:
            signal_scores = self._resolve_conflicts(
                signal_scores, conflicts, component_signals, state_label
            )
        
        # Bestimme das finale Signal
        self.logger.debug(f"Ensemble Signal - Scores before conflict resolution: {signal_scores}") # LOG ADDED (might be same as after if no conflict)
        
        # CRITICAL FIX: Determine if we have actual directional signals with strength
        has_directional_signal = signal_scores["BUY"] > 0.1 or signal_scores["SELL"] > 0.1
        
        if has_directional_signal:
            # We have a strong enough directional signal, use it
            if signal_scores["BUY"] > signal_scores["SELL"]:
                signal_type = "BUY"
                signal_strength = signal_scores["BUY"]
            else:
                signal_type = "SELL"
                signal_strength = signal_scores["SELL"]
        else:
            # Use highest score but apply minimum threshold for meaningful signals
            max_score = max(signal_scores.values())
            final_signal = max(signal_scores.items(), key=lambda x: x[1])
            signal_type = final_signal[0]
            
            # Ensure non-zero strength for NEUTRAL (avoid the 0.2381 fixed value)
            if signal_type == "NEUTRAL":
                # Random small variation to avoid exact same value repeatedly
                import random
                signal_strength = 0.1 + random.uniform(0.01, 0.05)
            else:
                signal_strength = final_signal[1]
        
        # self.logger.debug(f"Ensemble Signal - Final Scores after potential conflict resolution: {signal_scores}") # LOG ADDED - Moved timing
        self.logger.debug(f"Ensemble Signal - Final Scores (after potential conflict resolution): {signal_scores}") # LOG ADDED
        self.logger.debug(f"Ensemble Signal - Determined Signal: {signal_type}, Strength: {signal_strength:.4f}") # LOG ADDED
        
        # Prüfe, ob Signal stark genug ist (redundant if max_score <= 0 handled above, but good safety check)
        if signal_strength < 0.1:  # Mindeststärke für ein Signal
            signal_type = "NEUTRAL"
            signal_strength = max(signal_scores["NEUTRAL"], 0.05)  # Adjusted fallback strength slightly
        
        # Erstelle Signaleintrag für die Historie
        signal_entry = {
            "timestamp": current_time or datetime.now().isoformat(),
            "price": current_price,
            "state": state_label,
            "volatility": volatility,
            "trend": trend,
            "session": session,
            "signal": signal_type,
            "strength": signal_strength,
            "component_signals": {comp: signal_info.get("signal", "NEUTRAL") 
                                for comp, signal_info in component_signals.items()},
            "component_contributions": component_contributions,
            "weights": weights,
            "outcome": None  # Wird später aktualisiert
        }
        
        # Füge zur Signalhistorie hinzu
        self.signal_history.append(signal_entry)
        
        # Inkrementiere Update-Zähler
        self.update_counter += 1
        
        # Prüfe, ob ein regelmäßiges Update der Gewichte nötig ist
        if self.update_counter % 50 == 0:
            self._perform_periodic_updates()
        
        # Erstelle Ensembleausgabe
        return {
            "signal": signal_type,
            "strength": signal_strength,
            "component_contributions": component_contributions,
            "conflicts": conflicts,
            "weights": weights,
            "state": state_label
        }
    
    def _identify_conflicts(self, component_signals):
        """
        Identifiziert Konflikte zwischen verschiedenen Komponenten.
        
        Args:
            component_signals: Dictionary mit {'component': signal_info}
            
        Returns:
            list: Liste erkannter Konflikte
        """
        conflicts = []
        
        # Sammle Signale pro Typ
        signal_components = {"BUY": [], "SELL": [], "NEUTRAL": []}
        
        for comp, signal_info in component_signals.items():
            if comp not in self.components:
                continue
            
            # Standardisiere Signaltyp
            signal_type = signal_info.get("signal", "NEUTRAL")
            
            if signal_type in ["LONG", "UP", "BULLISH"]:
                signal_type = "BUY"
            elif signal_type in ["SHORT", "DOWN", "BEARISH"]:
                signal_type = "SELL"
            else:
                signal_type = "NEUTRAL"
            
            # Füge Komponente zur entsprechenden Signalliste hinzu
            signal_components[signal_type].append(comp)
        
        # Prüfe auf direkten Konflikt (BUY vs SELL)
        if signal_components["BUY"] and signal_components["SELL"]:
            for buy_comp in signal_components["BUY"]:
                for sell_comp in signal_components["SELL"]:
                    conflicts.append({
                        "components": (buy_comp, sell_comp),
                        "signals": ("BUY", "SELL"),
                        "type": "direct_conflict"
                    })
        
        return conflicts
    
    def _resolve_conflicts(self, signal_scores, conflicts, component_signals, state_label):
        """
        Löst Konflikte zwischen widersprüchlichen Signalen.
        
        Args:
            signal_scores: Aktuelle Signalwerte
            conflicts: Liste erkannter Konflikte
            component_signals: Dictionary mit {'component': signal_info}
            state_label: HMM-Zustandslabel
            
        Returns:
            dict: Angepasste Signalwerte
        """
        adjusted_scores = signal_scores.copy()
        
        for conflict in conflicts:
            comp_pair = conflict["components"]
            
            # Prüfe, ob Konfliktlösung für dieses Paar definiert ist
            if comp_pair in self.conflict_resolution:
                resolution = self.conflict_resolution[comp_pair]
                
                # Angepasste Gewichte basierend auf zustandsspezifischer Performance
                if state_label and state_label in self.meta_learner["state_performance"]:
                    # Hole zustandsspezifische Komponentenperformance
                    state_perf = self.meta_learner["state_performance"][state_label]
                    
                    if comp_pair[0] in state_perf and comp_pair[1] in state_perf:
                        perf0 = state_perf[comp_pair[0]]
                        perf1 = state_perf[comp_pair[1]]
                        
                        # Anpassung des Konfliktgewichts basierend auf relativer Performance
                        if perf0 + perf1 > 0:
                            adjusted_weight = perf0 / (perf0 + perf1)
                            
                            # Begrenze Anpassung, um nicht zu extreme Werte zu haben
                            adjusted_weight = max(0.2, min(0.8, adjusted_weight))
                            
                            # Aktualisiere Konfliktgewicht
                            resolution_weight = (resolution["weight"] * 0.7 + 
                                               adjusted_weight * 0.3)
                        else:
                            resolution_weight = resolution["weight"]
                    else:
                        resolution_weight = resolution["weight"]
                else:
                    resolution_weight = resolution["weight"]
                
                # Anwenden der Konfliktlösungsregel
                if resolution["rule"] == "weighted_vote":
                    # Gewichtete Abstimmung zwischen den beiden Komponenten
                    comp0 = comp_pair[0]
                    comp1 = comp_pair[1]
                    
                    signal0 = component_signals[comp0].get("signal", "NEUTRAL")
                    signal1 = component_signals[comp1].get("signal", "NEUTRAL")
                    
                    # Standardisiere Signale
                    if signal0 in ["LONG", "UP", "BULLISH"]:
                        signal0 = "BUY"
                    elif signal0 in ["SHORT", "DOWN", "BEARISH"]:
                        signal0 = "SELL"
                    else:
                        signal0 = "NEUTRAL"
                        
                    if signal1 in ["LONG", "UP", "BULLISH"]:
                        signal1 = "BUY"
                    elif signal1 in ["SHORT", "DOWN", "BEARISH"]:
                        signal1 = "SELL"
                    else:
                        signal1 = "NEUTRAL"
                    
                    # Finde alte Beiträge und passe sie an
                    old_contribution0 = adjusted_scores[signal0]
                    old_contribution1 = adjusted_scores[signal1]
                    
                    # Gewichtete Lösung: erste Komponente wird stärker/schwächer
                    adjusted_scores[signal0] = old_contribution0 * resolution_weight
                    adjusted_scores[signal1] = old_contribution1 * (1 - resolution_weight)
                
                elif resolution["rule"] == "hmm_priority":
                    # HMM-Priorität, ignoriere andere Komponente weitgehend
                    # Diese Logik kann je nach Bedarf angepasst werden
                    pass
                
                elif resolution["rule"] == "average":
                    # Durchschnittliches Signal (für numerische Signale)
                    pass
        
        return adjusted_scores
    
    def update_signal_outcome(self, signal_idx, outcome, profit_pips=None):
        """
        Aktualisiert ein früheres Signal mit seinem tatsächlichen Ergebnis.
        
        Args:
            signal_idx: Index des Signals in der Historie (-1 für letztes Signal)
            outcome: "success" oder "failure"
            profit_pips: Gewinn/Verlust in Pips
            
        Returns:
            bool: Erfolg des Updates
        """
        # Validiere den Signal-Index
        if abs(signal_idx) > len(self.signal_history):
            self.logger.warning(f"Invalid signal index: {signal_idx}")
            return False
        
        # Konvertiere negativen Index (relative Position vom Ende)
        if signal_idx < 0:
            signal_idx = len(self.signal_history) + signal_idx
        
        # Hole das Signal
        signal = self.signal_history[signal_idx]
        
        # Aktualisiere Outcome
        signal["outcome"] = outcome
        signal["profit_pips"] = profit_pips
        signal["evaluation_time"] = datetime.now().isoformat()
        
        # Aktualisiere Komponenten-Performance
        for comp, comp_signal in signal["component_signals"].items():
            if comp not in self.components:
                continue
            
            # Prüfe, ob Komponentensignal korrekt war
            was_correct = False
            
            if outcome == "success":
                # Bei Erfolg: Komponentensignal sollte mit Ensemblesignal übereinstimmen
                if comp_signal == signal["signal"]:
                    was_correct = True
            else:
                # Bei Misserfolg: Komponentensignal sollte nicht mit Ensemblesignal übereinstimmen
                if comp_signal != signal["signal"]:
                    was_correct = True
            
            # Aktualisiere Performancemetriken für diese Komponente
            if comp in self.performance_metrics:
                self.performance_metrics[comp]["total"] += 1
                
                if was_correct:
                    self.performance_metrics[comp]["correct"] += 1
        
        # Aktualisiere zustandsspezifische Leistung für Meta-Learner
        state = signal["state"]
        
        if state and state not in self.meta_learner["state_performance"]:
            self.meta_learner["state_performance"][state] = {
                comp: 0.0 for comp in self.components
            }
        
        if state:
            state_perf = self.meta_learner["state_performance"][state]
            
            # Aktualisiere Performance für jede Komponente in diesem Zustand
            alpha = 0.1  # Lernrate für Meta-Learner
            
            for comp, comp_signal in signal["component_signals"].items():
                if comp not in self.components:
                    continue
                
                # Berechne Belohnung/Bestrafung für diese Komponente
                reward = 0.0
                
                if outcome == "success":
                    if comp_signal == signal["signal"]:
                        # Korrekte Vorhersage bei Erfolg
                        reward = 1.0
                        if profit_pips is not None and profit_pips > 0:
                            reward *= min(3.0, 1.0 + profit_pips / 10.0)  # Skalieren mit Gewinn
                    else:
                        # Falsche Vorhersage bei Erfolg
                        reward = -0.5
                else:  # failure
                    if comp_signal != signal["signal"]:
                        # Korrekte Ablehnung bei Misserfolg
                        reward = 0.5
                    else:
                        # Falsche Vorhersage bei Misserfolg
                        reward = -1.0
                        if profit_pips is not None and profit_pips < 0:
                            reward *= min(3.0, 1.0 + abs(profit_pips) / 10.0)  # Skalieren mit Verlust
                
                # Update Performance mit exponentiellem gleitenden Mittel
                if comp in state_perf:
                    state_perf[comp] = (1 - alpha) * state_perf[comp] + alpha * reward
        
        # Aktualisiere Komponentengewichte basierend auf allgemeiner Performance
        self._update_weights()
        
        # Speichere bei jeder 10. Aktualisierung
        if self.update_counter % 10 == 0:
            self.save_model()
        
        return True
    
    def _update_weights(self):
        """
        Aktualisiert Komponentengewichte basierend auf Performance.
        """
        new_weights = {}
        
        # Berechne neue Gewichte basierend auf Performance
        total_performance = 0.0
        
        for comp in self.components:
            if comp in self.performance_metrics:
                metrics = self.performance_metrics[comp]
                
                if metrics["total"] > 0:
                    # Performance = Anteil korrekter Signale
                    performance = metrics["correct"] / metrics["total"]
                else:
                    performance = 0.5  # Neutraler Startwert
                
                # Mindestperformance, um Nullgewichte zu vermeiden
                performance = max(0.1, performance)
                
                new_weights[comp] = performance
                total_performance += performance
        
        # Normalisiere Gewichte
        if total_performance > 0:
            for comp in new_weights:
                new_weights[comp] /= total_performance
                
                # Stelle sicher, dass jede Komponente mindestens Mindestgewicht hat
                new_weights[comp] = max(self.min_weight, new_weights[comp])
            
            # Normalisiere erneut nach Anwendung von Mindestgewichten
            weight_sum = sum(new_weights.values())
            for comp in new_weights:
                new_weights[comp] /= weight_sum
        else:
            # Fallback: Verwende initiale Gewichte
            new_weights = self.initial_weights.copy()
        
        # Aktualisiere globale Gewichte mit exponentiellem gleitenden Mittel
        alpha = self.learning_rate
        
        for comp in self.components:
            if comp in new_weights and comp in self.weights:
                self.weights[comp] = (1 - alpha) * self.weights[comp] + alpha * new_weights[comp]
        
        # Aktualisiere auch Konfliktlösungsmatrix
        self._update_conflict_resolution()
    
    def _update_conflict_resolution(self):
        """
        Aktualisiert die Konfliktlösungsmatrix basierend auf Performance.
        """
        # Für jedes Komponentenpaar
        for (comp1, comp2) in self.conflict_resolution:
            if comp1 in self.performance_metrics and comp2 in self.performance_metrics:
                metrics1 = self.performance_metrics[comp1]
                metrics2 = self.performance_metrics[comp2]
                
                # Berechne Performance-Verhältnis
                perf1 = metrics1["correct"] / metrics1["total"] if metrics1["total"] > 0 else 0.5
                perf2 = metrics2["correct"] / metrics2["total"] if metrics2["total"] > 0 else 0.5
                
                if perf1 + perf2 > 0:
                    # Gewicht basierend auf relativem Erfolg
                    new_weight = perf1 / (perf1 + perf2)
                    
                    # Begrenzen auf vernünftigen Bereich
                    new_weight = max(0.2, min(0.8, new_weight))
                    
                    # Update mit gleitendem Mittel
                    current_weight = self.conflict_resolution[(comp1, comp2)]["weight"]
                    updated_weight = (1 - self.learning_rate) * current_weight + self.learning_rate * new_weight
                    
                    self.conflict_resolution[(comp1, comp2)]["weight"] = updated_weight
                    
                    # Aktualisiere auch umgekehrtes Paar
                    if (comp2, comp1) in self.conflict_resolution:
                        self.conflict_resolution[(comp2, comp1)]["weight"] = 1.0 - updated_weight
    
    def _perform_periodic_updates(self):
        """
        Führt periodische Updates und Wartungsaufgaben durch.
        """
        # Berechne Zeitraum seit letztem Major-Update
        time_since_update = (datetime.now() - self.last_major_update).total_seconds()
        
        # Major Update alle 12 Stunden (43200 Sekunden)
        if time_since_update > 43200:
            self.logger.info("Performing major update of ensemble weights and meta-learner")
            
            # Aktualisiere Gewichte mit höherer Lernrate
            old_lr = self.learning_rate
            self.learning_rate = min(0.3, old_lr * 2)  # Erhöhe Lernrate temporär
            self._update_weights()
            self.learning_rate = old_lr  # Setze Lernrate zurück
            
            # Analyse der Signalhistorie für Meta-Learner
            self._analyze_signal_history()
            
            # Bereinige selten verwendete Zustände aus state_weights
            self._clean_state_weights()
            
            # Aktualisiere Zeitstempel
            self.last_major_update = datetime.now()
            
            # Speichere Modell nach Major-Update
            self.save_model()
    
    def _analyze_signal_history(self):
        """
        Analysiert die Signalhistorie für Meta-Learning.
        """
        # Mindestanzahl von Signalen für verlässliche Analyse
        if len(self.signal_history) < 20:
            return
        
        # Aktualisiere Zustandsübergangsmatrix
        prev_state = None
        
        for signal in self.signal_history:
            state = signal.get("state")
            
            if state and prev_state:
                # Aktualisiere Zustandsübergangsmatrix
                if prev_state not in self.meta_learner["state_transitions"]:
                    self.meta_learner["state_transitions"][prev_state] = {}
                
                if state not in self.meta_learner["state_transitions"][prev_state]:
                    self.meta_learner["state_transitions"][prev_state][state] = 0
                
                self.meta_learner["state_transitions"][prev_state][state] += 1
            
            prev_state = state
        
        # Aktualisiere volatilitätsspezifische Performance
        for volatility in ["low", "medium", "high"]:
            if volatility not in self.meta_learner["volatility_impact"]:
                self.meta_learner["volatility_impact"][volatility] = {
                    comp: 0.0 for comp in self.components
                }
            
            # Zähle erfolgreiche Signale pro Komponente in dieser Volatilität
            vol_signals = [s for s in self.signal_history 
                         if s.get("volatility") == volatility and s.get("outcome") is not None]
            
            if vol_signals:
                vol_perf = self.meta_learner["volatility_impact"][volatility]
                
                for comp in self.components:
                    correct = sum(1 for s in vol_signals 
                                if s["component_signals"].get(comp) == s["signal"] and s["outcome"] == "success")
                    total = sum(1 for s in vol_signals if comp in s["component_signals"])
                    
                    if total > 0:
                        performance = correct / total
                        # Exponentielles gleitendes Mittel
                        vol_perf[comp] = vol_perf[comp] * 0.8 + performance * 0.2
    
    def _clean_state_weights(self):
        """
        Bereinigt selten verwendete Zustände aus state_weights.
        """
        # Zähle Auftreten jedes Zustands in der Historie
        state_counts = {}
        
        for signal in self.signal_history:
            state = signal.get("state")
            if state:
                if state not in state_counts:
                    state_counts[state] = 0
                state_counts[state] += 1
        
        # Entferne selten verwendete Zustände (weniger als 3 Auftreten)
        rare_states = [state for state, count in state_counts.items() if count < 3]
        
        for state in rare_states:
            if state in self.state_weights:
                del self.state_weights[state]
                self.logger.info(f"Removed rare state '{state}' from state weights")
    
    def get_performance_metrics(self):
        """
        Gibt detaillierte Performancemetriken für alle Komponenten zurück.
        
        Returns:
            dict: Performancemetriken
        """
        metrics = {
            "components": {},
            "ensemble": {"correct": 0, "total": 0},
            "states": {},
            "weights": self.weights
        }
        
        # Komponenten-Metriken
        for comp in self.components:
            if comp in self.performance_metrics:
                perf = self.performance_metrics[comp]
                
                if perf["total"] > 0:
                    accuracy = perf["correct"] / perf["total"]
                else:
                    accuracy = 0.0
                
                metrics["components"][comp] = {
                    "accuracy": accuracy,
                    "correct": perf["correct"],
                    "total": perf["total"]
                }
            else:
                metrics["components"][comp] = {
                    "accuracy": 0.0, "correct": 0, "total": 0
                }
        
        # Ensemble-Genauigkeit
        success_count = 0
        total_evaluated = 0
        
        for signal in self.signal_history:
            if signal.get("outcome") is not None:
                total_evaluated += 1
                if signal["outcome"] == "success":
                    success_count += 1
        
        if total_evaluated > 0:
            metrics["ensemble"]["accuracy"] = success_count / total_evaluated
            metrics["ensemble"]["correct"] = success_count
            metrics["ensemble"]["total"] = total_evaluated
        
        # Zustandsspezifische Metriken
        state_outcomes = {}
        
        for signal in self.signal_history:
            state = signal.get("state")
            outcome = signal.get("outcome")
            
            if state and outcome is not None:
                if state not in state_outcomes:
                    state_outcomes[state] = {"success": 0, "failure": 0}
                
                if outcome == "success":
                    state_outcomes[state]["success"] += 1
                else:
                    state_outcomes[state]["failure"] += 1
        
        for state, outcomes in state_outcomes.items():
            total = outcomes["success"] + outcomes["failure"]
            
            if total > 0:
                metrics["states"][state] = {
                    "accuracy": outcomes["success"] / total,
                    "correct": outcomes["success"],
                    "total": total
                }
        
        return metrics
    
    def save_model(self, path=None):
        """
        Speichert das Ensemble-Modell.
        
        Args:
            path: Optionaler alternativer Speicherpfad
            
        Returns:
            bool: Erfolg des Speichervorgangs
        """
        if path is None:
            path = self.model_path
        
        try:
            # Erstelle Verzeichnis falls nicht vorhanden
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Bereite Modelldaten vor
            model_data = {
                "components": self.components,
                "weights": self.weights,
                "state_weights": self.state_weights,
                "performance_metrics": self.performance_metrics,
                "conflict_resolution": self.conflict_resolution,
                "meta_learner": self.meta_learner,
                "learning_rate": self.learning_rate,
                "min_weight": self.min_weight,
                "update_counter": self.update_counter,
                "last_major_update": self.last_major_update.isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Speichere mit Pickle
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Ensemble model saved to {path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ensemble model: {str(e)}")
            return False
    
    def load_model(self, path=None):
        """
        Lädt ein gespeichertes Ensemble-Modell.
        
        Args:
            path: Optionaler alternativer Ladepfad
            
        Returns:
            bool: Erfolg des Ladevorgangs
        """
        if path is None:
            path = self.model_path
        
        if not os.path.exists(path):
            self.logger.info(f"No existing model found at {path}")
            return False
        
        try:
            # Lade mit Pickle
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Aktualisiere Attribute
            if "components" in model_data:
                self.components = model_data["components"]
            
            if "weights" in model_data:
                self.weights = model_data["weights"]
            
            if "state_weights" in model_data:
                self.state_weights = model_data["state_weights"]
            
            if "performance_metrics" in model_data:
                self.performance_metrics = model_data["performance_metrics"]
            
            if "conflict_resolution" in model_data:
                self.conflict_resolution = model_data["conflict_resolution"]
            
            if "meta_learner" in model_data:
                self.meta_learner = model_data["meta_learner"]
            
            if "learning_rate" in model_data:
                self.learning_rate = model_data["learning_rate"]
            
            if "min_weight" in model_data:
                self.min_weight = model_data["min_weight"]
            
            if "update_counter" in model_data:
                self.update_counter = model_data["update_counter"]
            
            if "last_major_update" in model_data:
                try:
                    self.last_major_update = datetime.fromisoformat(model_data["last_major_update"])
                except:
                    self.last_major_update = datetime.now()
            
            self.logger.info(f"Ensemble model loaded from {path}")
            
            # Log Modellinformationen
            if "timestamp" in model_data:
                self.logger.info(f"Model saved on: {model_data['timestamp']}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble model: {str(e)}")
            return False


class FusionBacktester:
    """
    Backtestet Strategien mit Feature-Fusion und Ensemble-Komponenten.
    """
    def __init__(self, feature_fusion=None, ensemble=None):
        """
        Initialisiert den Backtester.
        
        Args:
            feature_fusion: RegularizedFeatureFusion-Instanz
            ensemble: AdaptiveComponentEnsemble-Instanz
        """
        self.feature_fusion = feature_fusion
        self.ensemble = ensemble
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('fusion_backtester')
    
    def backtest_feature_fusion(self, main_features, cross_features, order_features, 
                              price_data, lookahead=10, use_stop_loss=True,
                              take_profit_pips=20, stop_loss_pips=10):
        """
        Backtestet eine Feature-Fusion-Strategie.
        
        Args:
            main_features: Hauptsymbol-Features
            cross_features: Cross-Asset-Features
            order_features: Order Book Features
            price_data: DataFrame mit OHLCV-Preisdaten
            lookahead: Lookahead-Periode für Signale
            use_stop_loss: Ob Stop-Loss verwendet werden soll
            take_profit_pips: Take-Profit-Größe in Pips
            stop_loss_pips: Stop-Loss-Größe in Pips
            
        Returns:
            dict: Backtest-Ergebnisse
        """
        if self.feature_fusion is None:
            self.logger.error("Feature fusion model not initialized")
            return {"error": "feature_fusion_not_initialized"}
        
        if not self.feature_fusion.is_model_trained:
            self.logger.error("Feature fusion model not trained")
            return {"error": "feature_fusion_not_trained"}
        
        # Parameter validieren
        n_samples = len(main_features)
        
        if len(cross_features) != n_samples or len(order_features) != n_samples:
            self.logger.error(f"Feature dimension mismatch: {n_samples}, {len(cross_features)}, {len(order_features)}")
            return {"error": "feature_dimension_mismatch"}
        
        if len(price_data) < n_samples:
            self.logger.error("Insufficient price data for backtest")
            return {"error": "insufficient_price_data"}
        
        # Pip-Größe für Preise bestimmen
        if 'open' in price_data.columns and price_data['open'].iloc[0] > 100:
            pip_size = 0.01  # JPY-Paare
        else:
            pip_size = 0.0001  # Nicht-JPY-Paare
        
        # Ergebnislisten initialisieren
        trades = []
        signals = []
        positions = []
        current_position = None
        equity_curve = [1.0]  # Startkapital normalisiert auf 1.0
        
        # Feature-Korrelationsanalyse
        correlations = self.feature_fusion.analyze_feature_correlations(
            main_features[:min(1000, n_samples)], 
            cross_features[:min(1000, n_samples)], 
            order_features[:min(1000, n_samples)]
        )
        
        # Hauptbacktest-Schleife
        for i in range(n_samples - lookahead):
            # Fusioniere Features für den aktuellen Zeitpunkt
            fused_features = self.feature_fusion.fuse_features(
                main_features[i], cross_features[i], order_features[i]
            )
            
            # Generiere Signal mit fortgeschrittener Klassifikationstechnik
            signal = self._generate_fusion_signal(fused_features, price_data.iloc[i])
            
            # Speichere Signal
            signals.append({
                "index": i,
                "timestamp": price_data.index[i],
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "price": price_data['close'].iloc[i]
            })
            
            # Verarbeite aktive Position, falls vorhanden
            if current_position is not None:
                # Prüfe, ob Exit-Bedingungen erfüllt sind
                exit_price, exit_reason = self._check_exit_conditions(
                    current_position, price_data.iloc[i], 
                    take_profit_pips, stop_loss_pips, pip_size, use_stop_loss,
                    signals=signals
                )
                
                if exit_price is not None:
                    # Schließe die Position
                    profit_pips = self._calculate_profit(
                        current_position, exit_price, pip_size
                    )
                    
                    # Aktualisiere Equity
                    equity_change = profit_pips / 100  # Vereinfachte Performance-Berechnung
                    equity_curve.append(equity_curve[-1] * (1 + equity_change))
                    
                    # Speichere den abgeschlossenen Trade
                    current_position["exit_price"] = exit_price
                    current_position["exit_time"] = price_data.index[i]
                    current_position["exit_reason"] = exit_reason
                    current_position["profit_pips"] = profit_pips
                    current_position["profit_percent"] = equity_change * 100
                    
                    trades.append(current_position)
                    current_position = None
            
            # Prüfe auf neues Handelssignal, wenn keine Position aktiv ist
            if current_position is None and signal["signal"] in ["BUY", "SELL"] and signal["confidence"] > 0.6:
                # Öffne neue Position
                current_position = {
                    "entry_time": price_data.index[i],
                    "entry_price": price_data['close'].iloc[i],
                    "direction": signal["signal"],
                    "confidence": signal["confidence"],
                    "size": 1.0  # Vereinfachte Position Sizing
                }
                
                positions.append(current_position.copy())
        
        # Schließe offene Position am Ende des Backtest
        if current_position is not None:
            exit_price = price_data['close'].iloc[-1]
            profit_pips = self._calculate_profit(current_position, exit_price, pip_size)
            
            # Aktualisiere Equity
            equity_change = profit_pips / 100
            equity_curve.append(equity_curve[-1] * (1 + equity_change))
            
            # Speichere den abgeschlossenen Trade
            current_position["exit_price"] = exit_price
            current_position["exit_time"] = price_data.index[-1]
            current_position["exit_reason"] = "end_of_backtest"
            current_position["profit_pips"] = profit_pips
            current_position["profit_percent"] = equity_change * 100
            
            trades.append(current_position)
        
        # Berechne Performance-Statistiken
        performance = self._calculate_performance_metrics(trades, equity_curve)
        
        # Aktualisiere Feature-Fusion-Gewichte basierend auf Performance
        if self.feature_fusion.adaptive_weights and performance["profit_factor"] > 0:
            self.feature_fusion.update_weights_from_performance(
                performance["profit_factor"], "main"
            )
        
        return {
            "trades": trades,
            "signals": signals,
            "positions": positions,
            "equity_curve": equity_curve,
            "performance": performance,
            "correlations": correlations,
            "feature_weights": self.feature_fusion.get_feature_importance()
        }
    
    def backtest_ensemble(self, component_signals, hmm_states, price_data,
                         use_stop_loss=True, take_profit_pips=20, stop_loss_pips=10,
                         position_sizing='fixed', max_open_positions=1, max_risk_pct=2.0):
        """
        Advanced implementation of ensemble strategy backtesting with sophisticated
        risk management, position sizing, and comprehensive analytics.

        Args:
            component_signals: List of signals per component for each time point
            hmm_states: HMM states for each time point
            price_data: DataFrame with OHLCV price data
            use_stop_loss: Whether to use stop loss
            take_profit_pips: Take profit size in pips
            stop_loss_pips: Stop loss size in pips
            position_sizing: Position sizing method ('fixed', 'risk_pct', 'kelly', 'adaptive')
            max_open_positions: Maximum number of concurrent positions
            max_risk_pct: Maximum risk percentage per trade

        Returns:
            dict: Comprehensive backtest results with rich analytics
        """
        # Initialize performance monitoring
        start_time = time.time()

        if self.ensemble is None:
            self.logger.error("Ensemble component not initialized")
            return {"error": "ensemble_not_initialized", "success": False}

        # Robust dimension validation with detailed diagnostics
        input_diagnostics = {
            "signals_length": len(component_signals) if component_signals else 0,
            "states_length": len(hmm_states) if hmm_states else 0,
            "price_data_length": len(price_data) if price_data is not None else 0,
            "price_data_columns": list(price_data.columns) if price_data is not None else []
        }

        self.logger.info(f"Backtest input dimensions: signals={input_diagnostics['signals_length']}, " +
                        f"states={input_diagnostics['states_length']}, prices={input_diagnostics['price_data_length']}")

        # Validate critical requirements
        if not component_signals or not hmm_states or price_data is None:
            self.logger.error("Missing required input data for backtesting")
            return {
                "error": "missing_input_data",
                "diagnostics": input_diagnostics,
                "success": False
            }

        # Validate price data structure
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in price_data.columns]

        if missing_columns:
            self.logger.error(f"Price data missing required columns: {missing_columns}")
            return {
                "error": "invalid_price_data",
                "missing_columns": missing_columns,
                "success": False
            }

        # Calculate the minimum valid size to ensure consistent dimensions
        min_size = min(
            input_diagnostics['signals_length'],
            input_diagnostics['states_length'],
            input_diagnostics['price_data_length']
        )

        if min_size == 0:
            self.logger.error("No valid overlapping data available for backtest")
            return {"error": "no_valid_data", "diagnostics": input_diagnostics, "success": False}

        # If dimensions don't match, log detailed alignment information
        alignment_info = {}
        truncation_required = False

        if input_diagnostics['signals_length'] != min_size:
            truncation_required = True
            alignment_info["signals_truncation"] = {
                "original": input_diagnostics['signals_length'],
                "truncated": min_size,
                "removed": input_diagnostics['signals_length'] - min_size
            }

        if input_diagnostics['states_length'] != min_size:
            truncation_required = True
            alignment_info["states_truncation"] = {
                "original": input_diagnostics['states_length'],
                "truncated": min_size,
                "removed": input_diagnostics['states_length'] - min_size
            }

        if input_diagnostics['price_data_length'] != min_size:
            truncation_required = True
            alignment_info["price_truncation"] = {
                "original": input_diagnostics['price_data_length'],
                "truncated": min_size,
                "removed": input_diagnostics['price_data_length'] - min_size
            }

        if truncation_required:
            self.logger.warning(f"Data dimensions misaligned - truncating to common length of {min_size}")

            # Log detailed truncation statistics
            for key, info in alignment_info.items():
                if info.get("removed", 0) > 0:
                    self.logger.info(f"{key}: Removed {info['removed']} items " +
                                    f"({100 * info['removed'] / info['original']:.1f}% of original)")

        # Create properly aligned data subsets
        try:
            component_signals = component_signals[:min_size]
            hmm_states = hmm_states[:min_size]
            price_data = price_data.iloc[:min_size].copy()
        except Exception as e:
            self.logger.error(f"Error during data alignment: {str(e)}")
            return {"error": "alignment_failed", "exception": str(e), "success": False}

        # Determine pip size for prices based on symbol characteristics
        if 'open' in price_data.columns:
            first_price = price_data['open'].iloc[0]
            # Intelligent pip size detection
            if first_price > 100:  # JPY pairs
                pip_size = 0.01
            elif first_price > 1:  # Major pairs
                pip_size = 0.0001
            else:  # Crypto or CFDs
                pip_size = 0.0001 * first_price
        else:
            # Default pip size
            pip_size = 0.0001

        self.logger.info(f"Using pip size: {pip_size}")

        # Initialize advanced backtesting variables
        trades = []
        signals = []
        positions = []
        open_positions = []  # For tracking multiple concurrent positions
        equity_curve = [1.0]  # Initial capital normalized to 1.0
        drawdowns = [0.0]     # For drawdown tracking
        exposure = [0.0]      # For tracking market exposure
        cash = 1.0            # Initial cash (normalized)
        peak_equity = 1.0     # For drawdown calculation

        # Initialize enhanced analytics
        analytics = {
            "signal_counts": {
                "BUY": 0, "SELL": 0, "NONE": 0
            },
            "signal_by_state": {},
            "trades_by_hour": {},
            "trades_by_state": {},
            "consecutive_wins": 0,
            "max_consecutive_wins": 0,
            "consecutive_losses": 0,
            "max_consecutive_losses": 0,
            "win_after_win": 0,
            "win_after_loss": 0,
            "loss_after_win": 0,
            "loss_after_loss": 0
        }

        # --- Add this line ---
        last_valid_close_price = None # Store the last known close price

        # Main backtest loop with comprehensive error handling
        try:
            for i in range(min_size):
                # Extract data for current time step
                try:
                    # Initialize price variables for this step
                    open_price, high_price, low_price, close_price = None, None, None, None

                    # Get component signals and state for this time point with proper type checking
                    if isinstance(hmm_states[i], str):
                        state_label = hmm_states[i]
                    elif hasattr(hmm_states[i], "get"):
                        state_label = hmm_states[i].get("state", "unknown")
                    elif isinstance(hmm_states[i], (int, np.integer)):
                        state_label = f"state_{hmm_states[i]}"
                    else:
                        state_label = "unknown"

                    comp_signals = component_signals[i]
                    current_timestamp = price_data.index[i]

                    # Extract price information safely
                    current_price_data = price_data.iloc[i]
                    if 'open' in current_price_data: open_price = current_price_data['open']
                    if 'high' in current_price_data: high_price = current_price_data['high']
                    if 'low' in current_price_data: low_price = current_price_data['low']
                    if 'close' in current_price_data: close_price = current_price_data['close']

                    # Check if essential price data is missing for this step
                    if open_price is None or high_price is None or low_price is None or close_price is None:
                        self.logger.warning(f"Missing essential OHLC data at index {i}. Skipping step.")
                        # Append previous values to maintain curve lengths if possible
                        if equity_curve: equity_curve.append(equity_curve[-1])
                        if drawdowns: drawdowns.append(drawdowns[-1])
                        if exposure: exposure.append(exposure[-1])
                        continue # Skip to the next iteration

                    # --- Add this line ---
                    last_valid_close_price = close_price # Update last valid price

                    # Advanced market condition detection
                    try:
                        # Volatility detection
                        volatility = "medium"
                        current_atr = None # Define current_atr here
                        if 'atr' in price_data.columns or 'atr_30m' in price_data.columns:
                            atr_col = 'atr' if 'atr' in price_data.columns else 'atr_30m'
                            if i >= 20:
                                avg_atr = price_data[atr_col].iloc[i-20:i].mean()
                                current_atr = price_data[atr_col].iloc[i]

                                if current_atr > avg_atr * 1.5:
                                    volatility = "high"
                                elif current_atr < avg_atr * 0.7:
                                    volatility = "low"
                        else:
                            # Price-based volatility estimation using rolling window calculation
                            if i >= 10:
                                ranges = []
                                for j in range(max(0, i-10), i):
                                    if j < len(price_data):
                                        ranges.append(price_data['high'].iloc[j] - price_data['low'].iloc[j])

                                if ranges:
                                    avg_range = sum(ranges) / len(ranges)
                                    current_range = high_price - low_price

                                    if current_range > avg_range * 1.5:
                                        volatility = "high"
                                    elif current_range < avg_range * 0.7:
                                        volatility = "low"

                        # Multi-timeframe trend detection
                        trend = "neutral"
                        if i >= 5:
                            # Short-term price behavior
                            short_term_price = price_data['close'].iloc[max(0, i-5)]
                            short_term_change = (close_price / short_term_price) - 1

                            # Medium-term price behavior (if enough data)
                            medium_term_change = 0
                            if i >= 20:
                                medium_term_price = price_data['close'].iloc[max(0, i-20)]
                                medium_term_change = (close_price / medium_term_price) - 1

                            # Weighted trend assessment
                            weighted_change = short_term_change * 0.7 + medium_term_change * 0.3

                            if weighted_change > 0.005:  # 0.5% change
                                trend = "bullish"
                            elif weighted_change < -0.005:
                                trend = "bearish"

                            # Additionally check momentum indicators if available
                            if 'rsi_30m' in price_data.columns:
                                rsi = price_data['rsi_30m'].iloc[i]
                                if rsi > 70:
                                    trend = "overbought"
                                elif rsi < 30:
                                    trend = "oversold"

                        # Market session detection with global time consideration
                        session = "unknown"
                        if hasattr(current_timestamp, "hour"):
                            hour = current_timestamp.hour
                            # Precise session mapping considering global markets
                            if 0 <= hour < 3:
                                session = "asian_early"
                            elif 3 <= hour < 8:
                                session = "asian_late"
                            elif 8 <= hour < 12:
                                session = "european_open"
                            elif 12 <= hour < 16:
                                session = "us_european_overlap"
                            elif 16 <= hour < 20:
                                session = "us_main"
                            else:
                                session = "us_late"

                        # Market phase detection for more nuanced strategy parameters
                        market_phase = "unknown"
                        if i >= 50 and 'atr_30m' in price_data.columns and 'rsi_30m' in price_data.columns:
                            # Calculate volatility trend
                            recent_vol = price_data['atr_30m'].iloc[max(0, i-10):i].mean()
                            past_vol = price_data['atr_30m'].iloc[max(0, i-50):max(0, i-40)].mean()
                            vol_change = (recent_vol / past_vol) - 1 if past_vol > 0 else 0

                            # Calculate trend strength and consistency
                            returns_window = price_data['close'].pct_change().iloc[max(0, i-20):i].values
                            if len(returns_window) > 0: # Ensure window is not empty
                                pos_returns = np.sum(returns_window > 0)
                                neg_returns = np.sum(returns_window < 0)
                                # Trend consistency measures
                                trend_strength = abs(pos_returns - neg_returns) / len(returns_window)
                            else:
                                trend_strength = 0

                            # Determine market phase
                            if trend_strength > 0.6:
                                market_phase = "trending"
                            elif vol_change > 0.3:
                                market_phase = "expanding_volatility"
                            elif vol_change < -0.3:
                                market_phase = "contracting_volatility"
                            else:
                                market_phase = "ranging"
                    except Exception as e:
                        self.logger.debug(f"Error in market condition detection at index {i}: {str(e)}")
                        volatility, trend, session, market_phase = "medium", "neutral", "unknown", "unknown"

                    # Generate ensemble signal
                    ensemble_result = self.ensemble.ensemble_signal(
                        comp_signals, state_label, volatility, trend, session,
                        current_price=close_price,
                        current_time=str(current_timestamp)
                    )

                    # Update signal analytics
                    signal_type = ensemble_result["signal"]
                    analytics["signal_counts"][signal_type] = analytics["signal_counts"].get(signal_type, 0) + 1

                    if state_label not in analytics["signal_by_state"]:
                        analytics["signal_by_state"][state_label] = {"BUY": 0, "SELL": 0, "NONE": 0}

                    analytics["signal_by_state"][state_label][signal_type] = \
                        analytics["signal_by_state"][state_label].get(signal_type, 0) + 1

                    # Save signal with enriched metadata
                    signal_entry = {
                        "index": i,
                        "timestamp": current_timestamp,
                        "signal": signal_type,
                        "strength": ensemble_result["strength"],
                        "state": state_label,
                        "price": close_price,
                        "volatility": volatility,
                        "trend": trend,
                        "session": session,
                        "market_phase": market_phase,
                        "component_weights": ensemble_result.get("weights", {}),
                        "component_signals": comp_signals
                    }

                    signals.append(signal_entry) # Add signal info to the list

                    # Process active positions with sophisticated exit conditions
                    positions_to_remove = []
                    for pos_idx, position in enumerate(open_positions):
                        # Update position with current market conditions for sophisticated exit logic
                        position["current_market_phase"] = market_phase
                        position["current_volatility"] = volatility
                        position["current_trend"] = trend
                        position["current_session"] = session
                        position["current_timestamp"] = current_timestamp
                        position["current_price"] = close_price

                        # Check if any exit conditions met (with full advanced strategy)
                        exit_price, exit_reason = self._check_exit_conditions(
                            position, current_price_data,
                            take_profit_pips, stop_loss_pips, pip_size, use_stop_loss,
                            signals=signals # Pass the signal history
                        )

                        # Update position tracking
                        position["bars_held"] = position.get("bars_held", 0) + 1

                        if exit_price is not None:
                            # Close the position
                            profit_pips = self._calculate_profit(
                                position, exit_price, pip_size
                            )

                            # Calculate position P&L with sophisticated modeling
                            position_size = position.get("size", 1.0)
                            actual_profit_pct = (profit_pips * pip_size / position["entry_price"]) * 100 if position["entry_price"] != 0 else 0
                            position_pnl = position_size * actual_profit_pct / 100

                            # Update equity and cash
                            equity_change = position_pnl
                            new_equity = equity_curve[-1] * (1 + equity_change)
                            equity_curve.append(new_equity)
                            cash += position_size * (1 + equity_change)

                            # Update peak equity and drawdown tracking
                            if new_equity > peak_equity:
                                peak_equity = new_equity

                            current_drawdown = (peak_equity - new_equity) / peak_equity if peak_equity > 0 else 0
                            drawdowns.append(current_drawdown)

                            # Update exposure - position is closed
                            # Recalculate total size of remaining open positions
                            remaining_open_size = sum(p.get("size", 1.0) for idx, p in enumerate(open_positions) if idx != pos_idx)
                            exposure.append(remaining_open_size / cash if cash > 0 else 0)


                            # Save the completed trade
                            position["exit_price"] = exit_price
                            position["exit_time"] = current_timestamp
                            position["exit_reason"] = exit_reason
                            position["profit_pips"] = profit_pips
                            position["profit_percent"] = actual_profit_pct
                            position["bars_held"] = position.get("bars_held", 0)

                            # Update win/loss streak analytics
                            if profit_pips > 0:
                                analytics["consecutive_wins"] += 1
                                analytics["consecutive_losses"] = 0
                                analytics["max_consecutive_wins"] = max(
                                    analytics["max_consecutive_wins"], analytics["consecutive_wins"])

                                # Win after win/loss tracking
                                if len(trades) > 0:
                                    last_trade = trades[-1]
                                    if last_trade["profit_pips"] > 0:
                                        analytics["win_after_win"] += 1
                                    else:
                                        analytics["win_after_loss"] += 1
                            else:
                                analytics["consecutive_losses"] += 1
                                analytics["consecutive_wins"] = 0
                                analytics["max_consecutive_losses"] = max(
                                    analytics["max_consecutive_losses"], analytics["consecutive_losses"])

                                # Loss after win/loss tracking
                                if len(trades) > 0:
                                    last_trade = trades[-1]
                                    if last_trade["profit_pips"] > 0:
                                        analytics["loss_after_win"] += 1
                                    else:
                                        analytics["loss_after_loss"] += 1

                            # Update trade hour analytics
                            trade_hour = position["entry_time"].hour if hasattr(position["entry_time"], "hour") else 0
                            if trade_hour not in analytics["trades_by_hour"]:
                                analytics["trades_by_hour"][trade_hour] = {"wins": 0, "losses": 0, "total": 0}

                            analytics["trades_by_hour"][trade_hour]["total"] += 1
                            if profit_pips > 0:
                                analytics["trades_by_hour"][trade_hour]["wins"] += 1
                            else:
                                analytics["trades_by_hour"][trade_hour]["losses"] += 1

                            # Update trade state analytics
                            position_state = position.get("state", "unknown")
                            if position_state not in analytics["trades_by_state"]:
                                analytics["trades_by_state"][position_state] = {"wins": 0, "losses": 0, "total": 0}

                            analytics["trades_by_state"][position_state]["total"] += 1
                            if profit_pips > 0:
                                analytics["trades_by_state"][position_state]["wins"] += 1
                            else:
                                analytics["trades_by_state"][position_state]["losses"] += 1

                            # Update component performance in ensemble using the CORRECT signal index
                            # FIX: Use the stored entry signal index
                            entry_signal_index = position.get("entry_signal_index", -1) # Get the stored index
                            if entry_signal_index != -1:
                                # Ensure the index is valid relative to the signals list length
                                # IMPORTANT: Use the original 'signals' list length for validation
                                if 0 <= entry_signal_index < len(signals):
                                    outcome = "success" if profit_pips > 0 else "failure"
                                    self.ensemble.update_signal_outcome(entry_signal_index, outcome, profit_pips)
                                else:
                                     self.logger.warning(f"Invalid entry signal index {entry_signal_index} for trade closed at index {i}. Max index: {len(signals)-1}")
                            else:
                                self.logger.warning(f"Could not find entry signal index for trade closed at index {i}")


                            trades.append(position)
                            positions_to_remove.append(pos_idx) # Mark for removal

                    # Remove closed positions AFTER iterating
                    for pos_idx in sorted(positions_to_remove, reverse=True):
                        if pos_idx < len(open_positions):
                            del open_positions[pos_idx]

                    # Check for new trading signal with sophisticated entry conditions
                    new_position_allowed = len(open_positions) < max_open_positions

                    # --- Logging hinzufügen ---
                    if signal_type in ["BUY", "SELL"]:
                         print(f"Signal: {signal_type}, Strength: {ensemble_result['strength']}")
                    # --- ENDE Logging ---

                    # --- NEU: Temporäre niedrigere Schwelle für den Start ---
                    current_progress_pct = i / min_size if min_size > 0 else 0 # Avoid division by zero
                    # Use lower threshold for the first 15% of the backtest to encourage initial trades
                    entry_threshold = 0.15 if current_progress_pct < 0.15 else 0.35
                    # --- ENDE Temporäre Schwelle ---

                    # Use the dynamically determined entry_threshold
                    if new_position_allowed and signal_type in ["BUY", "SELL"] and ensemble_result["strength"] > entry_threshold:
                        # Advanced position sizing
                        position_size = 1.0  # Default

                        if position_sizing == 'risk_pct':
                            # Risk percentage based sizing
                            current_equity = equity_curve[-1]
                            risk_amount = current_equity * (max_risk_pct / 100)
                            # --- FIX: Check if close_price is valid number before using --- 
                            if isinstance(close_price, (int, float)) and close_price != 0:
                                pip_value = pip_size * (1.0 / close_price)
                            else:
                                # Fallback if close_price is None or 0
                                self.logger.warning(f"Invalid close_price ({close_price}) for pip_value calculation at step {i}. Using default pip size.")
                                pip_value = pip_size 
                            # --- END FIX --- 
                            risk_pips = stop_loss_pips if use_stop_loss else 20 # Default risk
                            if risk_pips * pip_value != 0:
                                position_size = risk_amount / (risk_pips * pip_value)
                            else:
                                position_size = 0.1 # Fallback size

                        elif position_sizing == 'kelly':
                            # Kelly criterion based sizing
                            if len(trades) >= 10:
                                wins = [t for t in trades if t["profit_pips"] > 0]
                                losses = [t for t in trades if t["profit_pips"] <= 0]
                                win_rate = len(wins) / len(trades) if trades else 0.5
                                avg_win = sum(t["profit_pips"] for t in wins) / len(wins) if wins else 10
                                avg_loss = sum(t["profit_pips"] for t in losses) / len(losses) if losses else 10
                                if avg_loss != 0 and abs(avg_loss) != 0:
                                    win_loss_ratio = avg_win / abs(avg_loss)
                                    if win_loss_ratio != 0: # Avoid division by zero
                                        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
                                        position_size = max(0.1, min(1.0, kelly_fraction * 0.5)) # Half-Kelly
                                    else:
                                        position_size = 0.1 # Fallback
                                else:
                                    position_size = 0.1 # Fallback

                        elif position_sizing == 'adaptive':
                            base_size = 1.0
                            vol_factor = 1.0
                            if volatility == "high": vol_factor = 0.7
                            elif volatility == "low": vol_factor = 1.2
                            conf_factor = ensemble_result["strength"]
                            perf_factor = 1.0
                            if analytics["consecutive_wins"] > 2: perf_factor = 1.2
                            elif analytics["consecutive_losses"] > 2: perf_factor = 0.8
                            position_size = base_size * vol_factor * conf_factor * perf_factor
                            position_size = max(0.3, min(1.5, position_size)) # Limit range

                        # Calculate max holding period based on volatility and trend
                        max_bars = 20  # Default
                        if volatility == "high":
                            max_bars = 10
                        elif volatility == "low":
                            max_bars = 30
                        if trend == "neutral":
                            max_bars = int(max_bars * 0.7)

                        # Open new position with rich market context
                        new_position = {
                            "entry_signal_index": i, # FIX: Store the index of the signal
                            "entry_time": current_timestamp,
                            "entry_price": close_price,
                            "direction": signal_type,
                            "strength": ensemble_result["strength"],
                            "state": state_label,
                            "size": position_size,
                            "max_bars": max_bars,
                            "bars_held": 0,
                            "volatility": volatility,
                            "trend": trend,
                            "market_phase": market_phase,
                            "session": session,
                            "entry_signal": ensemble_result,
                            "component_weights": ensemble_result.get("weights", {}),
                            "trailing_stop": None,
                            "atr_at_entry": current_atr if current_atr is not None else None,
                            "rsi_at_entry": price_data['rsi_30m'].iloc[i] if 'rsi_30m' in price_data.columns else None,
                            "last_price": close_price
                        }

                        open_positions.append(new_position)
                        positions.append(new_position.copy())

                        # Update exposure tracking
                        exposure.append(sum(p.get("size", 1.0) for p in open_positions) / cash if cash > 0 else 0)
                    else:
                        # No new position - update equity/drawdown/exposure if lists are not empty
                        if equity_curve: equity_curve.append(equity_curve[-1])
                        if drawdowns: drawdowns.append(drawdowns[-1])
                        if exposure: exposure.append(exposure[-1])


                except Exception as e:
                    # Enhanced error logging with more context
                    error_context = {
                        "index": i,
                        "state_label": state_label if 'state_label' in locals() else 'undefined',
                        "component_signals_available": 'comp_signals' in locals(),
                        "price_data_row_exists": 'current_price_data' in locals(),
                        "last_valid_price": last_valid_close_price # Use last valid price here
                    }
                    self.logger.error(f"Critical error processing time step {i}: {str(e)}", exc_info=True)
                    self.logger.error(f"Error Context: {error_context}")
                    
                    # Graceful continuation: Use last known equity/exposure values
                    if equity_curve: equity_curve.append(equity_curve[-1])
                    if drawdowns: drawdowns.append(drawdowns[-1])
                    if exposure: exposure.append(exposure[-1])
                    
                    # Optional: Consider stopping backtest after too many errors?
                    # if error_count > max_errors: raise RuntimeError("Too many errors during backtest") from e
                    continue # Continue to the next iteration

            # Close any remaining open positions at the end of the backtest
            # Use the last available close_price (or the very last valid one if the loop ended early)
            final_close_price = price_data['close'].iloc[-1] if not price_data.empty else last_valid_close_price
            if final_close_price is None:
                 self.logger.error("Could not determine a final closing price for open positions at the end of backtest.")
                 final_close_price = 0 # Assign a default value to prevent crashes, though results might be inaccurate

            last_timestamp = price_data.index[-1] if not price_data.empty else None

            for position in open_positions:
                # Use final_close_price which is guaranteed to have a value (even if it's the fallback 0)
                exit_price = final_close_price
                profit_pips = self._calculate_profit(position, exit_price, pip_size)

                # Calculate P&L
                position_size = position.get("size", 1.0)
                actual_profit_pct = (profit_pips * pip_size / position["entry_price"]) * 100 if position["entry_price"] != 0 else 0
                position_pnl = position_size * actual_profit_pct / 100

                # Update equity
                equity_curve.append(equity_curve[-1] * (1 + position_pnl))

                # Save the completed trade
                position["exit_price"] = exit_price
                position["exit_time"] = last_timestamp
                position["exit_reason"] = "end_of_backtest"
                position["profit_pips"] = profit_pips
                position["profit_percent"] = actual_profit_pct
                position["bars_held"] = position.get("bars_held", 0)

                # Update component performance in ensemble using the CORRECT signal index
                entry_signal_index = position.get("entry_signal_index", -1)
                if entry_signal_index != -1 and 0 <= entry_signal_index < len(signals):
                     outcome = "success" if profit_pips > 0 else "failure"
                     self.ensemble.update_signal_outcome(entry_signal_index, outcome, profit_pips)
                else:
                     self.logger.warning(f"Could not update outcome for position closed at end_of_backtest, invalid index: {entry_signal_index}")


                trades.append(position)

            # Calculate comprehensive performance statistics
            performance = self._calculate_enhanced_performance_metrics(
                trades, equity_curve, drawdowns, exposure)

            # Get current weights and metrics
            ensemble_metrics = self.ensemble.get_performance_metrics()

            # Track execution time
            execution_time = time.time() - start_time

            # Create comprehensive result object
            backtest_results = {
                "trades": trades,
                "signals": signals,
                "positions": positions,
                "equity_curve": equity_curve,
                "drawdowns": drawdowns,
                "exposure": exposure,
                "performance": performance,
                "ensemble_metrics": ensemble_metrics,
                "analytics": analytics,
                "execution_time": execution_time,
                "success": True,
                "data_alignment": alignment_info,
                "backtest_parameters": {
                    "take_profit_pips": take_profit_pips,
                    "stop_loss_pips": stop_loss_pips,
                    "use_stop_loss": use_stop_loss,
                    "position_sizing": position_sizing,
                    "max_open_positions": max_open_positions,
                    "max_risk_pct": max_risk_pct,
                    "pip_size": pip_size
                }
            }

            self.logger.info(f"Backtest completed successfully in {execution_time:.2f} seconds with {len(trades)} trades")
            return backtest_results

        except Exception as e:
            self.logger.error(f"Critical error during backtesting: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                "error": "backtest_execution_failed",
                "exception": str(e),
                "success": False,
                "partial_trades": trades if 'trades' in locals() else []
            }
    
    def _check_exit_conditions(self, position, price_row, take_profit_pips, 
                             stop_loss_pips, pip_size, use_stop_loss, signals=None):
        """
        Vollständige Exit-Strategie für Handelspositionen mit mehrschichtiger Analyse.
        
        Args:
            position: Aktuelle Position
            price_row: Aktuelle Preisdaten
            take_profit_pips: Take-Profit-Größe in Pips
            stop_loss_pips: Stop-Loss-Größe in Pips
            pip_size: Pip-Größe für das Symbol
            use_stop_loss: Ob Stop-Loss verwendet werden soll
            signals: Liste der bisherigen Signale für Signal-basierte Exits
            
        Returns:
            tuple: (exit_price, exit_reason) oder (None, None)
        """
        # Extrahiere aktuelle Preise und Positionsdaten
        entry_price = position["entry_price"]
        direction = position["direction"]
        
        high = price_row['high']
        low = price_row['low']
        close = price_row['close']
        
        # Positionsalter und Marktphase
        bars_held = position.get("bars_held", 0)
        market_phase = position.get("current_market_phase", "unknown")
        volatility = position.get("current_volatility", "medium")
        trend = position.get("current_trend", "neutral")
        
        # 1. Take-Profit und Stop-Loss Bedingungen (Grundlegende Strategie)
        # Berechne Pip-Werte für Take-Profit und Stop-Loss
        take_profit_value = take_profit_pips * pip_size
        stop_loss_value = stop_loss_pips * pip_size
        
        # Adaptives Take-Profit/Stop-Loss basierend auf Marktphase
        tp_multiplier = 1.0
        sl_multiplier = 1.0
        
        if market_phase == "trending":
            # In Trendphasen: Höheres Take-Profit, engerer Stop-Loss
            tp_multiplier = 1.2
            sl_multiplier = 0.8
        elif market_phase == "ranging":
            # In Range-Phasen: Niedrigeres Take-Profit, weiterer Stop-Loss
            tp_multiplier = 0.8
            sl_multiplier = 1.2
        elif market_phase == "expanding_volatility":
            # Bei steigender Volatilität: Weiteres Take-Profit, engerer Stop-Loss
            tp_multiplier = 1.3
            sl_multiplier = 0.7
            
        # Passe TP/SL basierend auf Volatilität an
        if volatility == "high":
            tp_multiplier *= 1.2
            sl_multiplier *= 0.9
        elif volatility == "low":
            tp_multiplier *= 0.9
            sl_multiplier *= 1.1
        
        # Berechne angepasste TP/SL-Werte
        adjusted_tp_value = take_profit_value * tp_multiplier
        adjusted_sl_value = stop_loss_value * sl_multiplier
        
        # Prüfe auf normalen Take-Profit/Stop-Loss
        if direction == "BUY":
            take_profit_price = entry_price + adjusted_tp_value
            stop_loss_price = entry_price - adjusted_sl_value
            
            if high >= take_profit_price:
                return take_profit_price, "take_profit"
            
            if use_stop_loss and low <= stop_loss_price:
                return stop_loss_price, "stop_loss"
        
        elif direction == "SELL":
            take_profit_price = entry_price - adjusted_tp_value
            stop_loss_price = entry_price + adjusted_sl_value
            
            if low <= take_profit_price:
                return take_profit_price, "take_profit"
            
            if use_stop_loss and high >= stop_loss_price:
                return stop_loss_price, "stop_loss"
        
        # 2. Signalumkehr basierend auf mehreren Faktoren
        if direction == "BUY":
            # A. Konträres Ensemble-Signal mit hoher Konfidenz
            if signals and len(signals) >= 3:
                recent_signals = [s["signal"] for s in signals[-3:] if s.get("strength", 0) > 0.7]
                if recent_signals and recent_signals.count("SELL") >= 2:
                    return close, "signal_reversal"
            
            # B. Technischer Umkehrindikator: RSI-Divergenz
            if 'rsi_at_entry' in position and 'rsi_30m' in price_row:
                position_rsi = position.get('rsi_at_entry', 0)
                current_rsi = price_row['rsi_30m']
                # Bearishe Divergenz: Preis steigt, aber RSI fällt
                if close > entry_price * 1.005 and current_rsi < position_rsi * 0.9:
                    return close, "bearish_divergence"
            
            # C. Volumen-Spike mit Preisabfall
            if 'volume' in price_row and 'volume' in position:
                avg_volume = position.get('avg_volume', price_row['volume'])
                if price_row['volume'] > avg_volume * 2.5 and close < position.get("last_price", entry_price) * 0.997:
                    return close, "volume_reversal"
            
            # D. Dynamischer Trendindikator (z.B. Parabolic SAR)
            if 'sar' in price_row and price_row['sar'] > close:
                return close, "trend_reversal"
        
        elif direction == "SELL":
            # A. Konträres Ensemble-Signal mit hoher Konfidenz
            if signals and len(signals) >= 3:
                recent_signals = [s["signal"] for s in signals[-3:] if s.get("strength", 0) > 0.7]
                if recent_signals and recent_signals.count("BUY") >= 2:
                    return close, "signal_reversal"
            
            # B. Technischer Umkehrindikator: RSI-Divergenz
            if 'rsi_at_entry' in position and 'rsi_30m' in price_row:
                position_rsi = position.get('rsi_at_entry', 0)
                current_rsi = price_row['rsi_30m']
                # Bullishe Divergenz: Preis fällt, aber RSI steigt
                if close < entry_price * 0.995 and current_rsi > position_rsi * 1.1:
                    return close, "bullish_divergence"
            
            # C. Volumen-Spike mit Preisanstieg
            if 'volume' in price_row and 'volume' in position:
                avg_volume = position.get('avg_volume', price_row['volume'])
                if price_row['volume'] > avg_volume * 2.5 and close > position.get("last_price", entry_price) * 1.003:
                    return close, "volume_reversal"
            
            # D. Dynamischer Trendindikator (z.B. Parabolic SAR)
            if 'sar' in price_row and price_row['sar'] < close:
                return close, "trend_reversal"
        
        # 3. Marktphasenbasierte Exit-Logik mit adaptiven Schwellenwerten
        if market_phase == "trending" and bars_held > 5:
            # In Trendphasen: Trailing Stop Loss aktivieren
            trail_pct = 0.3 if volatility == "high" else 0.5
            if direction == "BUY":
                trailing_stop = max(position.get("trailing_stop", position["entry_price"]), close * (1 - trail_pct/100))
                if trailing_stop > position.get("trailing_stop", 0) and low <= trailing_stop:
                    return trailing_stop, "trailing_stop"
                position["trailing_stop"] = trailing_stop
            else:  # SELL
                trailing_stop = min(position.get("trailing_stop", position["entry_price"]), close * (1 + trail_pct/100))
                if trailing_stop < position.get("trailing_stop", float('inf')) and high >= trailing_stop:
                    return trailing_stop, "trailing_stop"
                position["trailing_stop"] = trailing_stop
                
        elif market_phase == "ranging" and bars_held > 3:
            # In Range-Märkten: Aggressivere Take-Profits
            if direction == "BUY" and close >= entry_price * (1 + (take_profit_pips*pip_size*0.7)/entry_price):
                return close, "range_take_profit"
            elif direction == "SELL" and close <= entry_price * (1 - (take_profit_pips*pip_size*0.7)/entry_price):
                return close, "range_take_profit"
                
        elif market_phase == "expanding_volatility" and bars_held > 1:
            # Bei steigender Volatilität: Schneller reagieren
            # Verwende ATR-basierte Stops
            if 'atr_at_entry' in position:
                atr_value = position['atr_at_entry']
                if direction == "BUY" and close < entry_price - atr_value * 1.5:
                    return close, "volatility_stop"
                elif direction == "SELL" and close > entry_price + atr_value * 1.5:
                    return close, "volatility_stop"
        
        # 4. Maximale Haltezeit basierend auf Marktbedingungen
        max_bars = position.get("max_bars", 20)
        if bars_held >= max_bars:
            return close, "max_holding_period"
        
        # 5. Trendwechsel-basierter Exit
        if (direction == "BUY" and trend == "bearish" and volatility == "high") or \
           (direction == "SELL" and trend == "bullish" and volatility == "high"):
            return close, "trend_change"
            
        # 6. Partielle Gewinnmitnahme bei starken Bewegungen
        # Implementierung würde zusätzliche Position-Tracking Logik erfordern
        # Hier nur einfache Version:
        profit_pips = self._calculate_profit(position, close, pip_size)
        if profit_pips > take_profit_pips * 1.5:
            return close, "accelerated_profit_taking"
        
        # Keiner der Exit-Gründe trifft zu
        position["last_price"] = close  # Update den letzten Preis für die nächste Prüfung
        return None, None
    
    def _calculate_enhanced_performance_metrics(self, trades, equity_curve, drawdowns, exposure):
        """
        Calculates comprehensive performance metrics with advanced risk-adjusted statistics.
        
        Args:
            trades: List of completed trades
            equity_curve: List of equity values
            drawdowns: List of drawdown values
            exposure: List of exposure values
            
        Returns:
            dict: Comprehensive performance metrics
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_profit": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "calmar_ratio": 0
            }
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["profit_pips"] > 0]
        losing_trades = [t for t in trades if t["profit_pips"] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Profit statistics
        total_profit = sum(t["profit_pips"] for t in winning_trades)
        total_loss = abs(sum(t["profit_pips"] for t in losing_trades))
        
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate advanced time-based metrics
        if len(trades) >= 2:
            # Average trade duration
            durations = []
            for trade in trades:
                if hasattr(trade["entry_time"], "timestamp") and hasattr(trade["exit_time"], "timestamp"):
                    duration_seconds = (trade["exit_time"].timestamp() - trade["entry_time"].timestamp())
                    durations.append(duration_seconds)
                elif hasattr(trade, "bars_held"):
                    durations.append(trade["bars_held"])
            
            avg_bars_held = sum(t.get("bars_held", 0) for t in trades) / len(trades)
            avg_duration_seconds = sum(durations) / len(durations) if durations else 0
            
            # Trades per day equivalent
            if avg_duration_seconds > 0:
                trades_per_day = (24 * 3600) / avg_duration_seconds
            else:
                trades_per_day = 0
        else:
            avg_bars_held = 0
            avg_duration_seconds = 0
            trades_per_day = 0
        
        # Returns analysis
        if len(equity_curve) > 1:
            # Calculate returns
            returns = [(equity_curve[i] / equity_curve[i-1] - 1) for i in range(1, len(equity_curve))]
            
            # Basic return statistics
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Risk-adjusted metrics
            risk_free_rate = 0.0  # Assuming 0% risk-free rate
            
            # Sharpe Ratio
            if std_return > 0:
                sharpe_ratio = np.sqrt(252) * (avg_return - risk_free_rate) / std_return
            else:
                sharpe_ratio = 0
                
            # Sortino Ratio (penalizes only downside deviation)
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0
            
            if downside_deviation > 0:
                sortino_ratio = np.sqrt(252) * (avg_return - risk_free_rate) / downside_deviation
            else:
                sortino_ratio = 0 if avg_return == 0 else float('inf')
            
            # Maximum Drawdown
            max_drawdown = max(drawdowns) if drawdowns else 0
            
            # Calmar Ratio
            if max_drawdown > 0:
                # Annualize return
                total_return = equity_curve[-1] / equity_curve[0] - 1
                # Approximate number of years
                years = len(equity_curve) / 252  # Assuming 252 trading days per year
                annualized_return = ((1 + total_return) ** (1 / max(1, years))) - 1
                calmar_ratio = annualized_return / max_drawdown
            else:
                calmar_ratio = 0 if avg_return == 0 else float('inf')
            
            # Drawdown statistics
            avg_drawdown = np.mean(drawdowns) if drawdowns else 0
            
            # Market exposure
            avg_exposure = np.mean(exposure) if exposure else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            avg_drawdown = 0
            calmar_ratio = 0
            avg_exposure = 0
        
        # Return comprehensive metrics
        return {
            # Basic metrics
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "avg_profit_pips": avg_profit,
            "avg_loss_pips": avg_loss,
            "total_profit_pips": float(total_profit),
            "total_loss_pips": float(total_loss),
            "net_profit_pips": float(total_profit - total_loss),
            "profit_factor": profit_factor,
            
            # Risk metrics
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            
            # Time metrics
            "avg_bars_held": avg_bars_held,
            "avg_duration_seconds": avg_duration_seconds,
            "trades_per_day": trades_per_day,
            
            # Exposure
            "avg_exposure": avg_exposure,
            
            # Final results
            "final_equity": equity_curve[-1] if equity_curve else 1.0,
            "total_return_pct": (equity_curve[-1] / equity_curve[0] - 1) * 100 if len(equity_curve) > 1 else 0
        }
    
    def _generate_fusion_signal(self, fused_features, price_row):
        """
        Erzeugt ein sophistiziertes Handelssignal basierend auf fusionierten Features
        mit Multi-Faktor-Analyseansatz.
        
        Args:
            fused_features: Fusionierte Features
            price_row: Aktuelle Preisdaten
            
        Returns:
            dict: Generiertes Signal mit umfassenden Metadaten
        """
        # Vollständige Implementierung mit Multi-Faktor-Analyse
        if len(fused_features) < 4:
            return {"signal": "NEUTRAL", "confidence": 0.0, "factors": {}}
        
        # 1. Extrahiere mehrere Faktoren aus den Merkmalen
        # Feature-Komponenten für verschiedene Aspekte der Analyse
        direction_feature = fused_features[0]  # Kurzfristige Richtung
        volatility_feature = abs(fused_features[1])  # Volatilitätseinschätzung
        momentum_feature = fused_features[2] if len(fused_features) > 2 else 0  # Momentum
        mean_reversion_feature = fused_features[3] if len(fused_features) > 3 else 0  # Mean-Reversion
        
        # 2. Feature-spezifische Signalkomponenten berechnen
        # Richtungssignal
        direction_score = np.tanh(direction_feature * 3)  # Skalierung mit tanh für [-1, 1]
        
        # Momentum-Signal
        momentum_score = np.tanh(momentum_feature * 2)
        
        # Mean-Reversion-Signal (umgekehrtes Momentum)
        mean_reversion_score = -np.tanh(mean_reversion_feature * 1.5)
        
        # 3. Kontextinformationen aus Preisdaten extrahieren
        context_factors = {}
        
        # Volatilitätsbasierte Anpassung
        volatility_score = volatility_feature
        context_factors["volatility"] = volatility_score
        
        # Preis-zu-Durchschnitt Verhältnis (falls verfügbar)
        if 'ma20' in price_row and price_row['close'] > 0:
            price_to_ma = price_row['close'] / price_row['ma20']
            # Über 1: Preis über MA, unter 1: Preis unter MA
            context_factors["price_to_ma"] = price_to_ma
            
            # Mean-Reversion-Impuls: Stärker wenn Preis weit vom MA entfernt
            ma_distance = abs(price_to_ma - 1)
            mean_reversion_multiplier = 1.0 + ma_distance * 2
            mean_reversion_score *= mean_reversion_multiplier
        
        # RSI (falls verfügbar)
        if 'rsi_30m' in price_row:
            rsi = price_row['rsi_30m'] 
            context_factors["rsi"] = rsi
            
            # Verstärke Mean-Reversion bei extremen RSI-Werten
            if rsi > 70:
                mean_reversion_score -= 0.3  # Stärkerer Verkaufsdruck bei hohem RSI
            elif rsi < 30:
                mean_reversion_score += 0.3  # Stärkerer Kaufdruck bei niedrigem RSI
        
        # 4. Kombiniere Signalkomponenten mit gewichteter Summe
        # Marktregime-Anpassung der Gewichte
        momentum_weight = 0.4
        mean_reversion_weight = 0.3
        direction_weight = 0.3
        
        # Passe Gewichte an Marktkontext an
        if 'atr_30m' in price_row and 'atr_30m_avg' in price_row:
            current_atr = price_row['atr_30m']
            avg_atr = price_row['atr_30m_avg']
            
            # Bei hoher Volatilität: Mehr Gewicht auf Mean-Reversion
            if current_atr > avg_atr * 1.5:
                mean_reversion_weight += 0.1
                momentum_weight -= 0.1
            # Bei niedriger Volatilität: Mehr Gewicht auf Momentum
            elif current_atr < avg_atr * 0.7:
                momentum_weight += 0.1
                mean_reversion_weight -= 0.1
        
        # Kombiniere Komponenten mit adaptiven Gewichten
        combined_score = (
            direction_score * direction_weight + 
            momentum_score * momentum_weight + 
            mean_reversion_score * mean_reversion_weight
        )
        
        # 5. Signal und Konfidenz bestimmen
        signal_strength = abs(combined_score)
        
        # Konvertiere Score in Signal mit Konfidenz
        if combined_score > 0.2:
            signal = "BUY"
            confidence = min(1.0, signal_strength)
        elif combined_score < -0.2:
            signal = "SELL"
            confidence = min(1.0, signal_strength)
        else:
            signal = "NEUTRAL"
            confidence = max(0.2, 1.0 - signal_strength * 2)  # Höhere Neutral-Konfidenz bei Score nahe 0
        
        # 6. Erstelle umfassende Signaldaten für detaillierte Analyse
        signal_data = {
            "signal": signal,
            "confidence": confidence,
            "combined_score": float(combined_score),
            "components": {
                "direction": float(direction_score),
                "momentum": float(momentum_score),
                "mean_reversion": float(mean_reversion_score)
            },
            "weights": {
                "direction": direction_weight,
                "momentum": momentum_weight,
                "mean_reversion": mean_reversion_weight
            },
            "context_factors": context_factors
        }
        
        return signal_data
    
    def _calculate_profit(self, position, exit_price, pip_size):
        """
        Berechnet den Gewinn/Verlust einer Position in Pips mit präziser Kalkulationstechnik.
        
        Args:
            position: Position
            exit_price: Ausstiegspreis
            pip_size: Pip-Größe für das Symbol
            
        Returns:
            float: Gewinn/Verlust in Pips
        """
        entry_price = position["entry_price"]
        direction = position["direction"]
        position_size = position.get("size", 1.0)
        
        # Berechne absoluten Preisunterschied
        price_diff = exit_price - entry_price
        
        # Direktionale Berechnung basierend auf Positionstyp
        if direction == "BUY":
            pip_profit = price_diff / pip_size
        else:  # SELL
            pip_profit = -price_diff / pip_size
        
        # Anpassung für Position Size
        adjusted_pip_profit = pip_profit * position_size
        
        # Rundung für konsistente Darstellung
        return round(adjusted_pip_profit, 2)
    
    def _calculate_performance_metrics(self, trades, equity_curve):
        """
        Berechnet Performance-Metriken basierend auf abgeschlossenen Trades.
        
        Args:
            trades: Liste abgeschlossener Trades
            equity_curve: Equity-Kurve
            
        Returns:
            dict: Performance-Metriken
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_profit": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
        
        # Grundlegende Trade-Statistiken
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["profit_pips"] > 0]
        losing_trades = [t for t in trades if t["profit_pips"] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Gewinn-Statistiken
        total_profit = sum(t["profit_pips"] for t in winning_trades)
        total_loss = sum(t["profit_pips"] for t in losing_trades)
        
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        # Profit-Faktor
        profit_factor = abs(total_profit / total_loss) if total_loss < 0 else 0
        
        # Drawdown-Berechnung
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Sharpe Ratio (vereinfacht)
        returns = [(equity_curve[i] / equity_curve[i-1] - 1) for i in range(1, len(equity_curve))]
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 1
        
        sharpe_ratio = np.sqrt(252) * avg_return / std_return if std_return > 0 else 0
        
        return {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "total_profit_pips": float(total_profit),
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "final_equity": equity_curve[-1]
        }
    
    def _calculate_max_drawdown(self, equity_curve):
        """
        Berechnet den maximalen Drawdown basierend auf einer Equity-Kurve.
        
        Args:
            equity_curve: Liste mit Equity-Werten
            
        Returns:
            float: Maximaler Drawdown als Dezimalwert
        """
        max_dd = 0
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd