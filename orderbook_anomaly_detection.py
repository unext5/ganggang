import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import deque
import logging
import joblib
import os
import time
from datetime import datetime, timedelta
import traceback

class OrderBookAnomalyDetector:
    """
    Erkennt Anomalien in Order Book Daten, die auf ungewöhnliche Marktbedingungen,
    Manipulationsversuche oder seltene Ereignisse hindeuten können.
    """
    def __init__(self, history_size=500, contamination=0.05, 
                 use_pca=True, n_components=10, model_path="models/ob_anomaly_detector.pkl"):
        """
        Initialisiert den Order Book Anomaliedetektor.
        
        Args:
            history_size: Größe des Historien-Puffers für Trainingsdaten
            contamination: Erwarteter Anteil von Anomalien (für Isolation Forest)
            use_pca: Ob Dimensionsreduktion mit PCA verwendet werden soll
            n_components: Anzahl der PCA-Komponenten
            model_path: Pfad zum Speichern/Laden des trainierten Modells
        """
        self.history = deque(maxlen=history_size)
        self.contamination = contamination
        self.use_pca = use_pca
        self.n_components = n_components
        self.model_path = model_path
        
        # ML-Komponenten
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        # Anomaliehistorie für Trendanalyse
        self.anomaly_history = deque(maxlen=100)
        self.anomaly_score_history = deque(maxlen=100)
        
        # Modellstatus
        self.is_fitted = False
        self.last_fit_time = None
        self.min_samples_for_fit = 50
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('ob_anomaly_detector')
        
        # Versuche, vorhandenes Modell zu laden
        self.load_model()
    
    def add_orderbook_features(self, features):
        """
        Fügt Order Book Features zum Historien-Puffer hinzu.
        
        Args:
            features: Dictionary mit Order Book Features oder Liste von Dictionaries
        """
        if isinstance(features, list):
            for feature in features:
                self.history.append(feature)
        else:
            self.history.append(features)
        
        # Überprüfe, ob genug Daten für ein (erneutes) Training vorhanden sind
        if len(self.history) >= self.min_samples_for_fit and (
            not self.is_fitted or 
            len(self.history) % 100 == 0  # Regelmäßiges Nachtraining
        ):
            # HINWEIS: Wir rufen fit() jetzt explizit im Hauptskript auf,
            #          daher ist der automatische Aufruf hier nicht mehr nötig.
            # self.fit()
            pass # Nur Daten sammeln
    
    def fit(self, data=None):
        """
        Trainiert den Anomaliedetektor.
        Wenn 'data' übergeben wird, wird dieses direkt verwendet,
        andernfalls wird die interne 'history' genutzt.
        """
        X = None
        if data is not None:
            # Direkte Verwendung der übergebenen Daten (muss bereits Matrix sein)
            if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] >= self.min_samples_for_fit:
                X = data
                self.logger.info(f"Verwende übergebene Datenmatrix ({X.shape[0]} Samples) für Training.")
            else:
                self.logger.warning(f"Übergebene Daten sind ungültig oder zu klein ({getattr(data, 'shape', 'N/A')}). Training abgebrochen.")
                return False
        else:
            # Verwendung der internen History
            if len(self.history) < self.min_samples_for_fit:
                self.logger.warning(f"Nicht genug Daten in History für Training ({len(self.history)}/{self.min_samples_for_fit})")
                return False
            try:
                X = self._convert_to_feature_matrix(list(self.history)) # Pass a copy
                self.logger.info(f"Verwende interne History ({X.shape[0]} Samples) für Training.")
            except Exception as e:
                self.logger.error(f"Fehler beim Konvertieren der History: {str(e)}")
                return False

        if X is None or X.shape[0] == 0 or X.shape[1] == 0:
            self.logger.warning("Leere Feature-Matrix nach Datenaufbereitung, Training abgebrochen")
            return False

        try:
            # Skaliere Features
            # WICHTIG: Kein Padding hier! Das Padding geschieht im Training Workflow.
            X_scaled = self.scaler.fit_transform(X)

            # Dimensionsreduktion, falls aktiviert
            if self.use_pca and self.pca is not None:
                X_transformed = self.pca.fit_transform(X_scaled)
                explained_variance = sum(self.pca.explained_variance_ratio_)
                self.logger.info(f"PCA: {self.n_components} Komponenten erklären {explained_variance:.2f} der Varianz")
            else:
                X_transformed = X_scaled

            # Trainiere Isolation Forest
            self.model.fit(X_transformed)

            self.is_fitted = True
            self.last_fit_time = datetime.now()

            self.logger.info(f"Anomaliedetektionsmodell trainiert mit {X.shape[0]} Samples, {X.shape[1]} Features")

            # Speichere trainiertes Modell
            self.save_model()

            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Training des Anomaliedetektors: {str(e)}", exc_info=True)
            return False
    
    def detect_anomalies(self, features, threshold=None):
        """
        Erkennt Anomalien in Order Book Features.
        
        Args:
            features: Dictionary mit Order Book Features oder Liste von Dictionaries
            threshold: Optionaler Schwellenwert für Anomalie-Scores (-1 bis 0, niedrigere Werte = stärkere Anomalien)
            
        Returns:
            dict: Ergebnisse der Anomalieerkennung
        """
        if not self.is_fitted:
            if len(self.history) >= self.min_samples_for_fit:
                self.fit() # Try to fit if not fitted and enough data
            else:
                self.logger.warning("Anomaliedetektor nicht trainiert und nicht genug Daten zum Trainieren.")
                return {"is_anomaly": False, "anomaly_score": 0, "confidence": 0}
            # Check again if fit succeeded
            if not self.is_fitted:
                 self.logger.warning("Anomaliedetektor-Training fehlgeschlagen, Erkennung nicht möglich.")
                 return {"is_anomaly": False, "anomaly_score": 0, "confidence": 0}


        # Standardschwellenwert basierend auf Kontamination
        if threshold is None:
            # Konvertiere Kontamination zu einem Score-Schwellenwert
            # Isolation Forest verwendet negative Scores für Anomalien
            threshold = -0.2  # Standardwert, wenn nicht durch Kontamination bestimmt

        try:
            # Konvertiere in Feature-Matrix
            is_single = not isinstance(features, list)
            features_list = [features] if is_single else features
            X = self._convert_to_feature_matrix(features_list)

            if X.shape[0] == 0:
                return {"is_anomaly": False, "anomaly_score": 0, "confidence": 0}

            # Ensure the feature matrix has exactly the expected number of features
            # This check remains as a safeguard
            current_num_features = X.shape[1]
            
            # Fix: Dynamically handle feature dimension mismatch
            if hasattr(self.scaler, 'n_features_in_'):
                expected_num_features = self.scaler.n_features_in_
                
                # If we have a feature mismatch, try to adapt instead of returning error
                if current_num_features != expected_num_features:
                    self.logger.warning(f"Adapting feature dimensions. Expected {expected_num_features}, got {current_num_features}.")
                    
                    if current_num_features < expected_num_features:
                        # Pad with zeros if we have fewer features
                        padding = np.zeros((X.shape[0], expected_num_features - current_num_features))
                        X = np.hstack((X, padding))
                        self.logger.info(f"Padded features from {current_num_features} to {expected_num_features}")
                    else:
                        # Truncate if we have more features
                        X = X[:, :expected_num_features]
                        self.logger.info(f"Truncated features from {current_num_features} to {expected_num_features}")
            else:
                # If scaler isn't fitted yet, set default expected features
                expected_num_features = 20

            # Skaliere und transformiere Features mit dem *trainierten* scaler/pca
            X_scaled = self.scaler.transform(X) # Use transform, not fit_transform

            if self.use_pca and self.pca is not None:
                X_transformed = self.pca.transform(X_scaled) # Use transform, not fit_transform
            else:
                X_transformed = X_scaled

            # Berechne Anomalie-Scores
            anomaly_scores = self.model.score_samples(X_transformed)

            # Erkenne Anomalien
            is_anomaly = anomaly_scores < threshold

            # Berechne Konfidenz (0 bis 1), wobei 1 = hohe Sicherheit einer Anomalie
            min_score = -0.5  # Typisches Minimum für starke Anomalien
            confidence = np.clip((threshold - anomaly_scores) / (threshold - min_score), 0, 1)

            # Füge zur Anomaliehistorie für Trendanalyse hinzu (nur wenn es einzelne Vorhersage ist)
            # Bei Batch-Vorhersage wird das Hinzufügen extern gehandhabt, um Duplikate zu vermeiden
            if is_single:
                self.anomaly_history.append(bool(is_anomaly[0]))
                self.anomaly_score_history.append(float(anomaly_scores[0]))

            if is_single:
                return {
                    "is_anomaly": bool(is_anomaly[0]),
                    "anomaly_score": float(anomaly_scores[0]),
                    "confidence": float(confidence[0]),
                    "threshold": threshold
                }
            else:
                # Return list of results for batch prediction
                results = []
                for i in range(len(features_list)):
                    results.append({
                        "is_anomaly": bool(is_anomaly[i]),
                        "anomaly_score": float(anomaly_scores[i]),
                        "confidence": float(confidence[i]),
                        "threshold": threshold
                    })
                return results

        except Exception as e:
            self.logger.error(f"Fehler bei der Anomalieerkennung: {str(e)}", exc_info=True) # Add traceback
            # Detailliertere Fehlerinformation zurückgeben
            return {"is_anomaly": False, "anomaly_score": 0, "confidence": 0, "error": str(e), "details": traceback.format_exc()}
    
    def analyze_anomaly_trends(self, window=20):
        """
        Analysiert Trends in der Anomaliehistorie.
        
        Args:
            window: Fenstergröße für die Analyse
            
        Returns:
            dict: Anomalietrend-Analyse
        """
        if len(self.anomaly_history) < window:
            return {
                "recent_anomaly_rate": 0,
                "trend": "insufficient_data",
                "avg_score": 0
            }
        
        # Betrachte die letzten window Datenpunkte
        recent_anomalies = list(self.anomaly_history)[-window:]
        recent_scores = list(self.anomaly_score_history)[-window:]
        
        # Berechne den Anteil der Anomalien
        anomaly_rate = sum(1 for a in recent_anomalies if a) / len(recent_anomalies)
        
        # Berechne durchschnittlichen Score
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Analysiere Trend
        if len(self.anomaly_history) >= window * 2:
            prev_anomalies = list(self.anomaly_history)[-(window*2):-window]
            prev_anomaly_rate = sum(1 for a in prev_anomalies if a) / len(prev_anomalies)
            
            if anomaly_rate > prev_anomaly_rate * 1.5:
                trend = "increasing"
            elif anomaly_rate < prev_anomaly_rate * 0.5:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "recent_anomaly_rate": anomaly_rate,
            "trend": trend,
            "avg_score": avg_score
        }
    
    def classify_anomaly_type(self, features):
        """
        Klassifiziert den Typ einer erkannten Anomalie basierend auf den Feature-Charakteristiken.
        
        Args:
            features: Dictionary mit Order Book Features
            
        Returns:
            dict: Klassifikation des Anomalietyps
        """
        anomaly_result = self.detect_anomalies(features)
        
        if not anomaly_result["is_anomaly"]:
            return {"type": "normal", "confidence": 0}
        
        # Analysiere Feature-Charakteristiken zur Bestimmung des Anomalietyps
        
        # 1. Prüfe auf Liquiditätsanomalien
        liquidity_anomaly = False
        liquidity_confidence = 0.0
        
        if "imbalance" in features and abs(features["imbalance"] - 1.0) > 3.0:
            # Stark unausgeglichenes Order Book
            liquidity_anomaly = True
            liquidity_confidence = min(1.0, abs(features["imbalance"] - 1.0) / 3.0)
        
        # 2. Prüfe auf Manipulationsversuche
        manipulation_anomaly = False
        manipulation_confidence = 0.0
        
        if "bid_wall_count" in features and features["bid_wall_count"] > 2:
            # Viele Liquiditätswände können auf Manipulation hindeuten
            manipulation_anomaly = True
            manipulation_confidence = min(1.0, features["bid_wall_count"] / 3.0)
        
        if "ask_wall_count" in features and features["ask_wall_count"] > 2:
            manipulation_anomaly = True
            manipulation_confidence = max(manipulation_confidence, 
                                        min(1.0, features["ask_wall_count"] / 3.0))
        
        # 3. Prüfe auf Volatilitätsausbrüche
        volatility_anomaly = False
        volatility_confidence = 0.0
        
        if "spread_pips" in features and features["spread_pips"] > 5.0:
            # Ungewöhnlich großer Spread
            volatility_anomaly = True
            volatility_confidence = min(1.0, features["spread_pips"] / 10.0)
        
        # Bestimme den dominanten Anomalietyp
        anomaly_types = [
            ("liquidity", liquidity_confidence),
            ("manipulation", manipulation_confidence),
            ("volatility", volatility_confidence)
        ]
        
        dominant_type = max(anomaly_types, key=lambda x: x[1])
        
        if dominant_type[1] > 0.3:
            anomaly_type = dominant_type[0]
            type_confidence = dominant_type[1]
        else:
            # Wenn keine spezifische Charakteristik dominiert
            anomaly_type = "unknown"
            type_confidence = anomaly_result["confidence"]
        
        return {
            "type": anomaly_type,
            "confidence": type_confidence,
            "details": {
                "liquidity": liquidity_confidence,
                "manipulation": manipulation_confidence,
                "volatility": volatility_confidence
            },
            "anomaly_score": anomaly_result["anomaly_score"]
        }
    
    def get_anomaly_trading_recommendation(self, anomaly_info, current_price=None, 
                                          state_label=None):
        """
        Generiert Handelsempfehlungen basierend auf erkannten Anomalien.
        
        Args:
            anomaly_info: Ergebnis der Anomalieerkennung oder -klassifikation
            current_price: Aktueller Marktpreis
            state_label: HMM-Zustandslabel
            
        Returns:
            dict: Handelsempfehlung für anomale Marktbedingungen
        """
        if "type" not in anomaly_info:
            # Wenn keine Klassifikation vorhanden, klassifiziere zuerst
            if "is_anomaly" in anomaly_info and anomaly_info["is_anomaly"]:
                # Dummy-Features für Klassifikation
                dummy_features = {"anomaly_score": anomaly_info["anomaly_score"]}
                anomaly_info = self.classify_anomaly_type(dummy_features)
            else:
                return {
                    "action": "ignore",
                    "reason": "no_anomaly",
                    "confidence": 0
                }
        
        anomaly_type = anomaly_info["type"]
        confidence = anomaly_info["confidence"]
        
        # Bestimme Markttrend aus State Label
        trend = "unknown"
        if state_label:
            if "Bullish" in state_label:
                trend = "bullish"
            elif "Bearish" in state_label:
                trend = "bearish"
        
        # Handelsempfehlungen basierend auf Anomalietyp
        if anomaly_type == "liquidity":
            if confidence > 0.7:
                # Bei starker Liquiditätsanomalie: Vorsicht mit aktiven Positionen
                if trend == "bullish":
                    return {
                        "action": "reduce_long",
                        "reason": "liquidity_anomaly",
                        "confidence": confidence,
                        "description": "Reduziere Long-Positionen wegen ungewöhnlicher Liquiditätsbedingungen"
                    }
                elif trend == "bearish":
                    return {
                        "action": "reduce_short",
                        "reason": "liquidity_anomaly",
                        "confidence": confidence,
                        "description": "Reduziere Short-Positionen wegen ungewöhnlicher Liquiditätsbedingungen"
                    }
                else: # Hinzugefügt: Fall für trend == "unknown" bei hoher Konfidenz
                    return {
                        "action": "caution",
                        "reason": "liquidity_anomaly",
                        "confidence": confidence,
                        "description": "Hohe Vorsicht (unbekannter Trend) wegen starker Liquiditätsanomalie"
                    }
            else:
                return {
                    "action": "caution",
                    "reason": "liquidity_anomaly",
                    "confidence": confidence,
                    "description": "Erhöhte Vorsicht wegen ungewöhnlicher Liquiditätsbedingungen"
                }
                
        elif anomaly_type == "manipulation":
            # Bei Manipulationsverdacht: Keine neuen Positionen eröffnen
            return {
                "action": "avoid_new_positions",
                "reason": "manipulation_suspected",
                "confidence": confidence,
                "description": "Keine neuen Positionen wegen Manipulationsverdacht"
            }
            
        elif anomaly_type == "volatility":
            if confidence > 0.5:
                # Bei Volatilitätsausbrüchen: Stop-Losses anpassen
                return {
                    "action": "widen_stops",
                    "reason": "volatility_anomaly",
                    "confidence": confidence,
                    "description": "Erweitere Stop-Losses wegen erhöhter Volatilität",
                    "adjustment_factor": 1.0 + confidence  # Stop-Loss-Anpassungsfaktor
                }
            else:
                return {
                    "action": "caution",
                    "reason": "volatility_anomaly",
                    "confidence": confidence,
                    "description": "Erhöhte Vorsicht wegen ungewöhnlicher Volatilität"
                }
        
        else:  # unknown anomaly type
            # Bei unbekannter Anomalie: Generelle Vorsicht
            return {
                "action": "caution",
                "reason": "unknown_anomaly",
                "confidence": confidence,
                "description": "Erhöhte Vorsicht wegen unklarer Anomalie"
            }
    
    def save_model(self, path=None):
        """
        Speichert das trainierte Anomaliedetektionsmodell.
        
        Args:
            path: Optionaler alternativer Speicherpfad
            
        Returns:
            bool: Erfolg des Speichervorgangs
        """
        if not self.is_fitted:
            self.logger.warning("Kein trainiertes Modell zum Speichern vorhanden")
            return False
        
        if path is None:
            path = self.model_path
        
        try:
            # Erstelle Verzeichnis, falls nicht vorhanden
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Speichere Modell und zugehörige Komponenten
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "pca": self.pca if self.use_pca else None,
                "use_pca": self.use_pca,
                "n_components": self.n_components,
                "contamination": self.contamination,
                "timestamp": datetime.now().isoformat()
            }
            
            joblib.dump(model_data, path)
            self.logger.info(f"Anomaliedetektionsmodell gespeichert in {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Modells: {str(e)}")
            return False
    
    def load_model(self, path=None):
        """
        Lädt ein vorhandenes Anomaliedetektionsmodell.
        
        Args:
            path: Optionaler alternativer Ladepfad
            
        Returns:
            bool: Erfolg des Ladevorgangs
        """
        if path is None:
            path = self.model_path
        
        if not os.path.exists(path):
            self.logger.info(f"Kein vorhandenes Modell gefunden unter {path}")
            return False
        
        try:
            model_data = joblib.load(path)
            
            # Überprüfe, ob alle erforderlichen Komponenten vorhanden sind
            if "model" not in model_data:
                self.logger.warning(f"Ungültiges Modellformat in {path}")
                return False
            
            # Lade Modell und Komponenten
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            
            if "pca" in model_data and model_data["pca"] is not None:
                self.pca = model_data["pca"]
                self.use_pca = model_data.get("use_pca", True)
                self.n_components = model_data.get("n_components", 10)
            
            self.contamination = model_data.get("contamination", self.contamination)
            
            self.is_fitted = True
            self.logger.info(f"Anomaliedetektionsmodell geladen aus {path}")
            
            # Log Modellinformationen
            if "timestamp" in model_data:
                self.logger.info(f"Modell trainiert am: {model_data['timestamp']}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Modells: {str(e)}")
            return False
    
    def _convert_to_feature_matrix(self, features_list):
        """
        Konvertiert eine Liste von Feature-Dictionaries in eine Feature-Matrix.
        
        Args:
            features_list: Liste von Feature-Dictionaries
            
        Returns:
            np.array: Feature-Matrix mit Dimensionen [n_samples, n_features]
        """
        if not features_list:
            return np.array([])
        
        # Extrahiere alle Feature-Namen
        all_keys = set()
        for features in features_list:
            # Check if features is a dictionary before using keys()
            if isinstance(features, dict):
                all_keys.update(features.keys())
            elif isinstance(features, np.ndarray):
                # For numpy arrays, we create a dictionary with numeric keys
                features_dict = {i: float(val) for i, val in enumerate(features.flatten())}
                all_keys.update(features_dict.keys())
            else:
                self.logger.warning(f"Unexpected features type: {type(features)}. Skipping.")
        
        # Erstelle Feature-Matrix
        n_samples = len(features_list)
        n_features = len(all_keys)
        X = np.zeros((n_samples, n_features))
        
        # Ordne Feature-Namen Spaltenindizes zu
        key_to_idx = {key: i for i, key in enumerate(sorted(all_keys))}
        
        # Fülle Matrix
        for i, features in enumerate(features_list):
            if isinstance(features, dict):
                for key, value in features.items():
                    if isinstance(value, (int, float)):
                        X[i, key_to_idx[key]] = value
            elif isinstance(features, np.ndarray):
                for j, value in enumerate(features):
                    if j < len(features) and isinstance(value, (int, float)):
                        key = j if j in key_to_idx else next(iter(key_to_idx.keys()))
                        X[i, key_to_idx[key]] = value
        
        return X


class OrderBookChangeDetector:
    """
    Spezialisierte Klasse zur Erkennung von abrupten Änderungen im Order Book,
    die auf bevorstehende Preisbewegungen hindeuten können.
    """
    def __init__(self, history_size=100, window_size=5, threshold=3.0):
        """
        Initialisiert den Change Detector für Order Books.
        
        Args:
            history_size: Größe des Historien-Puffers
            window_size: Fenstergröße für Vergleich von Änderungen
            threshold: Schwellenwert für signifikante Änderungen (in Standardabweichungen)
        """
        self.history = deque(maxlen=history_size)
        self.window_size = window_size
        self.threshold = threshold
        
        # Metriken-Tracking
        self.metrics_history = deque(maxlen=history_size)
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('ob_change_detector')
    
    def add_orderbook(self, order_book):
        """
        Fügt ein Order Book zur Historie hinzu und detektiert Änderungen.
        
        Args:
            order_book: Order Book Daten
            
        Returns:
            dict: Erkannte Änderungen
        """
        # Extrahiere und berechne relevante Metriken
        metrics = self._extract_metrics(order_book)
        
        # Speichere Metriken in Historie
        self.metrics_history.append(metrics)
        
        # Speichere Order Book in Historie
        self.history.append(order_book)
        
        # Erkenne Änderungen, wenn genügend Historie vorhanden
        if len(self.history) >= self.window_size:
            return self.detect_changes()
        
        return {"changes_detected": False}
    
    def detect_changes(self):
        """
        Detektiert signifikante Änderungen in der Order Book Historie.
        
        Returns:
            dict: Erkannte Änderungen mit Details
        """
        if len(self.metrics_history) < self.window_size:
            return {"changes_detected": False, "reason": "insufficient_history"}
        
        # Betrachte die letzten window_size Einträge
        recent_metrics = list(self.metrics_history)[-self.window_size:]
        
        # Berechne Änderungsraten für verschiedene Metriken
        changes = {}
        signals = []
        
        # 1. Imbalance Änderung (Bid/Ask Verhältnis)
        imbalance_values = [m["imbalance"] for m in recent_metrics if "imbalance" in m]
        if len(imbalance_values) >= 3:
            imbalance_change = self._calculate_change_rate(imbalance_values)
            changes["imbalance_change"] = imbalance_change
            
            # Signifikante Imbalance-Änderung
            if abs(imbalance_change) > self.threshold:
                signals.append({
                    "metric": "imbalance",
                    "change": imbalance_change,
                    "direction": "increase" if imbalance_change > 0 else "decrease",
                    "strength": abs(imbalance_change) / self.threshold
                })
        
        # 2. Spread Änderung
        spread_values = [m["spread"] for m in recent_metrics if "spread" in m]
        if len(spread_values) >= 3:
            spread_change = self._calculate_change_rate(spread_values)
            changes["spread_change"] = spread_change
            
            # Signifikante Spread-Änderung
            if abs(spread_change) > self.threshold:
                signals.append({
                    "metric": "spread",
                    "change": spread_change,
                    "direction": "increase" if spread_change > 0 else "decrease",
                    "strength": abs(spread_change) / self.threshold
                })
        
        # 3. Liquiditätsänderung
        liquidity_values = [m["total_liquidity"] for m in recent_metrics if "total_liquidity" in m]
        if len(liquidity_values) >= 3:
            liquidity_change = self._calculate_change_rate(liquidity_values)
            changes["liquidity_change"] = liquidity_change
            
            # Signifikante Liquiditätsänderung
            if abs(liquidity_change) > self.threshold:
                signals.append({
                    "metric": "liquidity",
                    "change": liquidity_change,
                    "direction": "increase" if liquidity_change > 0 else "decrease",
                    "strength": abs(liquidity_change) / self.threshold
                })
        
        # 4. Liquiditätsverteilung (Gini-Koeffizient)
        distribution_values = [m["distribution"] for m in recent_metrics if "distribution" in m]
        if len(distribution_values) >= 3:
            distribution_change = self._calculate_change_rate(distribution_values)
            changes["distribution_change"] = distribution_change
            
            # Signifikante Änderung der Liquiditätsverteilung
            if abs(distribution_change) > self.threshold:
                signals.append({
                    "metric": "distribution",
                    "change": distribution_change,
                    "direction": "increase" if distribution_change > 0 else "decrease",
                    "strength": abs(distribution_change) / self.threshold
                })
        
        # Generiere Gesamtergebnis
        result = {
            "changes_detected": len(signals) > 0,
            "changes": changes,
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
        
        # Spezifische Signale für Handelsentscheidungen
        if len(signals) > 0:
            # Analysiere die Signale im Detail
            if any(s["metric"] == "imbalance" and s["change"] > self.threshold for s in signals):
                # Starker Anstieg des Bid/Ask-Verhältnisses - potentiell bullish
                result["trade_signal"] = {
                    "direction": "bullish",
                    "confidence": min(1.0, max(s["strength"] for s in signals if s["metric"] == "imbalance") / 2),
                    "source": "orderbook_change",
                    "description": "Starker Anstieg des Bid/Ask-Verhältnisses"
                }
            elif any(s["metric"] == "imbalance" and s["change"] < -self.threshold for s in signals):
                # Starker Abfall des Bid/Ask-Verhältnisses - potentiell bearish
                result["trade_signal"] = {
                    "direction": "bearish",
                    "confidence": min(1.0, max(s["strength"] for s in signals if s["metric"] == "imbalance") / 2),
                    "source": "orderbook_change",
                    "description": "Starker Abfall des Bid/Ask-Verhältnisses"
                }
            elif any(s["metric"] == "liquidity" and s["change"] > self.threshold * 1.5 for s in signals):
                # Sehr starker Liquiditätsanstieg - potentiell Marktwende
                result["trade_signal"] = {
                    "direction": "reversal",
                    "confidence": min(1.0, max(s["strength"] for s in signals if s["metric"] == "liquidity") / 3),
                    "source": "orderbook_change",
                    "description": "Starker Liquiditätsanstieg deutet auf mögliche Marktwende hin"
                }
        
        return result
    
    def get_trading_signal(self, changes=None, price_trend=None):
        """
        Generiert ein Handelssignal basierend auf erkannten Änderungen.
        
        Args:
            changes: Erkannte Änderungen oder None, um letzte Erkennung zu verwenden
            price_trend: Optionale Information über aktuellen Preistrend
            
        Returns:
            dict: Handelssignal mit Richtung und Konfidenz
        """
        if changes is None:
            # Führe Erkennung mit aktuellen Daten durch
            changes = self.detect_changes()
        
        if not changes.get("changes_detected", False):
            return {
                "signal": "NONE",
                "confidence": 0,
                "source": "orderbook_change"
            }
        
        # Extrahiere Signale
        signals = changes.get("signals", [])
        
        if not signals:
            return {
                "signal": "NONE",
                "confidence": 0,
                "source": "orderbook_change"
            }
        
        # Analysiere Implikationen der Signale
        bullish_signals = []
        bearish_signals = []
        
        for signal in signals:
            metric = signal["metric"]
            direction = signal["direction"]
            strength = signal["strength"]
            
            if metric == "imbalance":
                if direction == "increase":
                    bullish_signals.append((strength, "imbalance"))
                else:
                    bearish_signals.append((strength, "imbalance"))
            
            elif metric == "liquidity":
                # Liquidity Shock kann in beide Richtungen interpretiert werden
                # Berücksichtige Preistrend, falls verfügbar
                if price_trend:
                    if price_trend == "up" and direction == "decrease":
                        # Fallende Liquidität in Aufwärtstrend: Fortsetzung wahrscheinlich
                        bullish_signals.append((strength * 0.7, "liquidity"))
                    elif price_trend == "down" and direction == "decrease":
                        # Fallende Liquidität in Abwärtstrend: Fortsetzung wahrscheinlich
                        bearish_signals.append((strength * 0.7, "liquidity"))
                    elif price_trend == "up" and direction == "increase":
                        # Steigende Liquidität in Aufwärtstrend: mögliche Umkehr
                        bearish_signals.append((strength * 0.5, "liquidity"))
                    elif price_trend == "down" and direction == "increase":
                        # Steigende Liquidität in Abwärtstrend: mögliche Umkehr
                        bullish_signals.append((strength * 0.5, "liquidity"))
        
        # Berechne Gesamtsignalstärke
        bullish_strength = sum(s[0] for s in bullish_signals) / max(1, len(bullish_signals))
        bearish_strength = sum(s[0] for s in bearish_signals) / max(1, len(bearish_signals))
        
        # Bestimme stärkstes Signal
        if bullish_strength > bearish_strength and bullish_strength > 0.5:
            signal = "LONG"
            confidence = bullish_strength / 2  # Skaliere zu vernünftigem Konfidenzbereich
            sources = [s[1] for s in bullish_signals]
        elif bearish_strength > bullish_strength and bearish_strength > 0.5:
            signal = "SHORT"
            confidence = bearish_strength / 2
            sources = [s[1] for s in bearish_signals]
        else:
            signal = "NONE"
            confidence = 0
            sources = []
        
        return {
            "signal": signal,
            "confidence": min(0.9, confidence),  # Cap bei 0.9, Order Book Signale sollten nie 100% sein
            "source": "orderbook_change",
            "details": {
                "bullish_strength": bullish_strength,
                "bearish_strength": bearish_strength,
                "sources": sources
            }
        }
    
    def _extract_metrics(self, order_book):
        """
        Extrahiert relevante Metriken aus Order Book Daten.
        
        Args:
            order_book: Order Book Dictionary
            
        Returns:
            dict: Extrahierte Metriken
        """
        metrics = {}
        
        if "bids" not in order_book or "asks" not in order_book:
            return metrics
        
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        
        if not bids or not asks:
            return metrics
        
        # 1. Imbalance (Bid/Ask Verhältnis)
        total_bid_volume = sum(bid["volume"] for bid in bids)
        total_ask_volume = sum(ask["volume"] for ask in asks)
        
        if total_ask_volume > 0:
            metrics["imbalance"] = total_bid_volume / total_ask_volume
        else:
            metrics["imbalance"] = 1.0
        
        # 2. Spread
        best_bid = bids[0]["price"] if bids else 0
        best_ask = asks[0]["price"] if asks else 0
        
        if best_bid > 0 and best_ask > best_bid:
            metrics["spread"] = best_ask - best_bid
        else:
            metrics["spread"] = 0
        
        # 3. Gesamtliquidität
        metrics["total_liquidity"] = total_bid_volume + total_ask_volume
        
        # 4. Liquiditätsverteilung (Gini-Koeffizient oder anderes Maß)
        bid_volumes = [bid["volume"] for bid in bids]
        ask_volumes = [ask["volume"] for ask in asks]
        
        # Einfacher Konzentrationswert: Verhältnis von Top-Level zu Gesamtliquidität
        if bids and total_bid_volume > 0:
            top_bid_ratio = bids[0]["volume"] / total_bid_volume
        else:
            top_bid_ratio = 0
            
        if asks and total_ask_volume > 0:
            top_ask_ratio = asks[0]["volume"] / total_ask_volume
        else:
            top_ask_ratio = 0
            
        metrics["distribution"] = (top_bid_ratio + top_ask_ratio) / 2
        
        # 5. Preisebenen
        metrics["levels_count"] = len(bids) + len(asks)
        
        # 6. Wall-Detection
        bid_walls = any(bid["volume"] > 3 * sum(b["volume"] for b in bids) / len(bids) for bid in bids)
        ask_walls = any(ask["volume"] > 3 * sum(a["volume"] for a in asks) / len(asks) for ask in asks)
        
        metrics["has_walls"] = 1 if (bid_walls or ask_walls) else 0
        
        # Zeitstempel hinzufügen
        metrics["timestamp"] = order_book.get("timestamp", datetime.now().isoformat())
        
        return metrics
    
    def _calculate_change_rate(self, values):
        """
        Berechnet die Änderungsrate in einer Sequenz von Werten.
        
        Args:
            values: Liste von numerischen Werten
            
        Returns:
            float: Normalisierte Änderungsrate
        """
        if len(values) < 3:
            return 0
        
        # Berechne prozentuale Änderung vom ersten zum letzten Wert
        first_value = values[0]
        last_value = values[-1]
        
        if abs(first_value) < 1e-10:
            first_value = 1e-10 * (1 if first_value >= 0 else -1)
        
        percentage_change = (last_value - first_value) / abs(first_value)
        
        # Berechne Standardabweichung zur Normalisierung
        std_dev = np.std(values)
        
        if std_dev < 1e-10:
            return 0
        
        # Normalisierte Änderungsrate in Einheiten der Standardabweichung
        normalized_change = percentage_change / std_dev
        
        return normalized_change