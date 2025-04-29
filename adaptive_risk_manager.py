import numpy as np
import pandas as pd
import math
import json
import os
from datetime import datetime, timedelta
from collections import deque
import logging

class AdaptiveRiskManager:
    """
    Erweiterte Klasse für adaptives Risikomanagement basierend auf HMM-Zuständen.
    Berechnet optimale Position Sizing und TP/SL-Level für manuelle Trades.
    """
    def __init__(self, symbol, pip_value=0.01, base_risk_pct=1.0, balance=None,
                account_currency="USD"):
        """
        Initialisiert den Manager.
        
        Args:
            symbol: Handelssymbol (z.B. "GBPJPY")
            pip_value: Wert eines Pips im Symbol
            base_risk_pct: Basis-Risiko in Prozent des Kontos
            balance: Aktueller Kontostand (falls None, muss später gesetzt werden)
            account_currency: Währung des Kontos
        """
        self.symbol = symbol
        self.pip_value = pip_value
        self.base_risk_pct = base_risk_pct
        self.balance = balance
        self.account_currency = account_currency
        
        # Risiko-Profil basierend auf Performance-Metriken
        self.risk_profile = {
            "low": 0.5,      # Risiko-Multiplikator für Niedrig-Risiko-Situationen
            "medium": 1.0,   # Standard-Multiplikator
            "high": 1.5      # Erhöhtes Risiko für Hoch-Konfidenz-Signale
        }
        
        # Performance-Historie für Zustandszuordnungen
        self.state_performance = {}
        
        # Aktuelle Marktvolatilität
        self.current_volatility = {
            "atr": None,     # ATR-Wert
            "level": "medium", # Volatilitätslevel (low, medium, high)
            "percentile": 50 # Volatilität im Vergleich zur Historie (Perzentil)
        }
        
        # Historie der Volatilitätsdaten
        self.volatility_history = deque(maxlen=100)
        
        # Optimale TP/SL-Ratios basierend auf Backtest-Daten
        self.tp_sl_ratios = {
            "high_bull": {"tp_atr": 3.0, "sl_atr": 1.5},
            "medium_bull": {"tp_atr": 2.5, "sl_atr": 1.2},
            "low_bull": {"tp_atr": 2.0, "sl_atr": 1.0},
            "low_bear": {"tp_atr": 2.0, "sl_atr": 1.0},
            "medium_bear": {"tp_atr": 2.5, "sl_atr": 1.2},
            "high_bear": {"tp_atr": 3.0, "sl_atr": 1.5}
        }
        
        # Interne Konfiguration
        self.min_confidence = 0.4  # Minimale Konfidenz für Handelsentscheidungen
        self.min_position_size = 0.01  # Kleinste erlaubte Positionsgröße
        self.max_position_size = 10.0  # Größte erlaubte Positionsgröße
        
        # Logger
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('adaptive_risk_manager')
        
        # Lade bestehende Daten, falls verfügbar
        self._load_state_performance()
    
    def _determine_regime_type(self, state_label):
        """
        Extrahiert den Regime-Typ aus dem Zustandslabel.
        
        Args:
            state_label: Label des Zustands (z.B. "High Bullish")
            
        Returns:
            str: Regime-Typ (z.B. "high_bull")
        """
        # Bestimme Volatilität (high, medium, low)
        if "Very High" in state_label or "High" in state_label:
            vol_level = "high"
        elif "Medium" in state_label:
            vol_level = "medium"
        elif "Very Low" in state_label or "Low" in state_label:
            vol_level = "low"
        else:
            vol_level = "medium"  # Default
        
        # Bestimme Richtung (bull, bear)
        if any(term in state_label for term in ["Bullish", "Bull", "Strong Bullish"]):
            direction = "bull"
        elif any(term in state_label for term in ["Bearish", "Bear", "Strong Bearish"]):
            direction = "bear"
        else:
            direction = "neutral"
        
        # Kombiniere zu Regime-Typ
        if direction == "neutral":
            regime_type = f"{vol_level}_neutral"
        else:
            regime_type = f"{vol_level}_{direction}"
        
        return regime_type
    
    def _load_state_performance(self, filepath="state_performance.json"):
        """
        Lädt gespeicherte Daten zur Zustandsperformance.
        
        Args:
            filepath: Pfad zur Performance-Datei
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    self.state_performance = json.load(file)
                self.logger.info(f"Zustandsperformance aus {filepath} geladen")
        except Exception as e:
            self.logger.warning(f"Konnte Zustandsperformance nicht laden: {str(e)}")
            self.state_performance = {}
    
    def save_state_performance(self, filepath="state_performance.json"):
        """
        Speichert aktuelle Daten zur Zustandsperformance.
        
        Args:
            filepath: Pfad zur Performance-Datei
        """
        try:
            with open(filepath, 'w') as file:
                json.dump(self.state_performance, file, indent=2)
            self.logger.info(f"Zustandsperformance in {filepath} gespeichert")
        except Exception as e:
            self.logger.warning(f"Konnte Zustandsperformance nicht speichern: {str(e)}")
    
    def update_balance(self, balance):
        """
        Aktualisiert den aktuellen Kontostand.
        
        Args:
            balance: Neuer Kontostand
        """
        self.balance = balance
    
    def update_volatility(self, atr_value):
        """
        Aktualisiert die aktuelle Marktvolatilität.
        
        Args:
            atr_value: Aktueller ATR-Wert
        """
        # Speichere neuen ATR-Wert mit Zeitstempel
        self.current_volatility["atr"] = atr_value
        self.volatility_history.append({
            "time": datetime.now().isoformat(),
            "atr": atr_value
        })
        
        # Berechne Volatilitätslevel basierend auf historischen Daten
        if len(self.volatility_history) > 10:
            atr_history = [entry["atr"] for entry in self.volatility_history]
            atr_mean = np.mean(atr_history)
            atr_std = np.std(atr_history)
            
            # Berechne Perzentil
            percentile = sum(1 for x in atr_history if x <= atr_value) / len(atr_history) * 100
            self.current_volatility["percentile"] = percentile
            
            # Definiere Volatilitätslevel
            if atr_value < atr_mean - 0.5 * atr_std:
                self.current_volatility["level"] = "low"
            elif atr_value > atr_mean + 0.5 * atr_std:
                self.current_volatility["level"] = "high"
            else:
                self.current_volatility["level"] = "medium"
        
        self.logger.info(f"Volatilität aktualisiert: {self.current_volatility}")
    
    def calculate_position_size(self, risk_pct, stop_loss_pips):
        """
        Berechnet die optimale Positionsgröße.
        
        Args:
            risk_pct: Risikoprozentsatz
            stop_loss_pips: Stop-Loss in Pips
        
        Returns:
            float: Positionsgröße in Lots
        """
        if self.balance is None:
            self.logger.warning("Kein Kontostand festgelegt!")
            return self.min_position_size
        
        # Risikoberechnung
        risk_amount = self.balance * (risk_pct / 100)
        
        # Wenn Stop-Loss zu eng ist, passe an
        min_sl_pips = 5  # Mindestens 5 Pips SL
        if stop_loss_pips < min_sl_pips:
            stop_loss_pips = min_sl_pips
            self.logger.warning(f"Stop-Loss zu eng, auf {min_sl_pips} Pips angepasst")
        
        # Positionsgröße = Risikobetrag / (Stop-Loss in Pips * Pip-Wert)
        # Pip-Wert ist in der Kontowährung
        position_size = risk_amount / (stop_loss_pips * self.pip_value)
        
        # Begrenze Positionsgröße auf sinnvollen Bereich
        position_size = max(min(position_size, self.max_position_size), self.min_position_size)
        
        # Runde auf die nächste 0.01
        return round(position_size * 100) / 100
    
    def calculate_risk_parameters(self, state_label, state_confidence, current_price, 
                                atr_value, account_balance=None):
        """
        Berechnet Trading-Parameter basierend auf HMM-Zustand.
        
        Args:
            state_label: Label des aktuellen Zustands
            state_confidence: Konfidenz des Zustands (0-1)
            current_price: Aktueller Preis
            atr_value: ATR-Wert für Stop-Loss-Berechnung
            account_balance: Kontostand (falls None, wird der gespeicherte verwendet)
        
        Returns:
            dict: Trading-Parameter
        """
        # Aktualisiere Volatilität und Kontostand
        self.update_volatility(atr_value)
        if account_balance is not None:
            self.update_balance(account_balance)
        
        # Bestimme Regime-Typ für Risiko-Multiplikatoren
        regime_type = self._determine_regime_type(state_label)
        
        # Optimale TP/SL-Ratios basierend auf dem aktuellen Regime
        tp_sl_config = self.tp_sl_ratios.get(regime_type, {"tp_atr": 2.0, "sl_atr": 1.0})
        
        # Confidence-Faktor
        if state_confidence < self.min_confidence:
            self.logger.warning(f"Zustandskonfidenz zu niedrig: {state_confidence:.2f}")
            confidence_factor = 0.5  # Reduziere Risiko bei niedriger Konfidenz
        else:
            # Skaliere zwischen 0.6 und 1.2 basierend auf Konfidenz
            confidence_factor = 0.6 + 0.6 * (state_confidence - self.min_confidence) / (1 - self.min_confidence)
        
        # Berücksichtige Performance-Historie
        if regime_type in self.state_performance:
            win_rate = self.state_performance[regime_type].get("win_rate", 0.5)
            # Überdurchschnittliche Performance erhöht Risiko
            performance_factor = 0.5 + win_rate
        else:
            performance_factor = 1.0
        
        # Volatilitätsanpassung
        vol_factor = 1.0
        if self.current_volatility["level"] == "high":
            vol_factor = 0.8  # Reduziere Position bei hoher Volatilität
        elif self.current_volatility["level"] == "low":
            vol_factor = 1.2  # Erhöhe Position bei niedriger Volatilität
        
        # Kombinierter Risikofaktor
        risk_factor = confidence_factor * performance_factor * vol_factor
        
        # Richtungsbestimmung
        if any(term in state_label for term in ["Bullish", "Bull"]):
            direction = "LONG"
            tp_factor = tp_sl_config["tp_atr"]
            sl_factor = tp_sl_config["sl_atr"]
        else:
            direction = "SHORT"
            tp_factor = tp_sl_config["tp_atr"]
            sl_factor = tp_sl_config["sl_atr"]
        
        # Distanzen für TP und SL in Pips
        # ATR wird in Punkten angegeben, umrechnen in Pips
        atr_in_pips = atr_value / self.pip_value
        
        # Basis-Distanzen
        tp_distance_pips = int(atr_in_pips * tp_factor)
        sl_distance_pips = int(atr_in_pips * sl_factor)
        
        # TP/SL-Preise
        if direction == "LONG":
            tp_price = current_price + tp_distance_pips * self.pip_value
            sl_price = current_price - sl_distance_pips * self.pip_value
        else:
            tp_price = current_price - tp_distance_pips * self.pip_value
            sl_price = current_price + sl_distance_pips * self.pip_value
        
        # Berechne angepasstes Risiko (unter Berücksichtigung aller Faktoren)
        adjusted_risk = self.base_risk_pct * risk_factor
        
        # Positionsgröße
        position_size = self.calculate_position_size(adjusted_risk, sl_distance_pips)
        
        # Risk/Reward-Verhältnis
        risk_reward = tp_distance_pips / sl_distance_pips if sl_distance_pips > 0 else 0
        
        # Runde Preise auf symbolspezifische Dezimalstellen
        tp_price = round(tp_price, 3)
        sl_price = round(sl_price, 3)
        
        return {
            "direction": direction,
            "entry_price": round(current_price, 3),
            "take_profit": tp_price,
            "stop_loss": sl_price,
            "tp_distance_pips": tp_distance_pips,
            "sl_distance_pips": sl_distance_pips,
            "risk_reward": round(risk_reward, 2),
            "position_size": position_size,
            "risk_percentage": round(adjusted_risk, 2),
            "state_confidence": round(state_confidence, 2),
            "volatility_level": self.current_volatility["level"],
            "confidence_factor": round(confidence_factor, 2),
            "performance_factor": round(performance_factor, 2)
        }
    
    def update_state_performance(self, state_label, win=None, profit_pips=None, risk_reward=None):
        """
        Aktualisiert die Performance-Statistik für einen Zustand.
        
        Args:
            state_label: Label des Zustands
            win: True wenn Trade gewonnen, False wenn verloren, None wenn unbekannt
            profit_pips: Gewinn/Verlust in Pips
            risk_reward: Risk-Reward-Verhältnis des Trades
        """
        # Bestimme Regime-Typ
        regime_type = self._determine_regime_type(state_label)
        
        # Initialisiere, falls noch nicht vorhanden
        if regime_type not in self.state_performance:
            self.state_performance[regime_type] = {
                "wins": 0,
                "losses": 0,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_profit_pips": 0.0,
                "total_profit_pips": 0.0,
                "avg_risk_reward": 0.0,
                "last_updated": datetime.now().isoformat()
            }
        
        # Aktualisiere Statistiken
        if win is not None:
            if win:
                self.state_performance[regime_type]["wins"] += 1
            else:
                self.state_performance[regime_type]["losses"] += 1
            
            self.state_performance[regime_type]["total_trades"] += 1
            wins = self.state_performance[regime_type]["wins"]
            total = self.state_performance[regime_type]["total_trades"]
            self.state_performance[regime_type]["win_rate"] = wins / total if total > 0 else 0
        
        if profit_pips is not None:
            # Aktualisiere Profit-Statistiken
            current_total = self.state_performance[regime_type]["total_profit_pips"]
            new_total = current_total + profit_pips
            self.state_performance[regime_type]["total_profit_pips"] = new_total
            
            total_trades = self.state_performance[regime_type]["total_trades"]
            if total_trades > 0:
                self.state_performance[regime_type]["avg_profit_pips"] = new_total / total_trades
        
        if risk_reward is not None:
            # Aktualisiere Risk-Reward-Verhältnis (exponentiell gewichteter Durchschnitt)
            old_rr = self.state_performance[regime_type]["avg_risk_reward"]
            alpha = 0.1  # Gewichtungsfaktor (0.1 = 10% Gewichtung für neuen Wert)
            new_rr = (1 - alpha) * old_rr + alpha * risk_reward
            self.state_performance[regime_type]["avg_risk_reward"] = new_rr
        
        # Aktualisiere Zeitstempel
        self.state_performance[regime_type]["last_updated"] = datetime.now().isoformat()
        
        # Speichere aktualisierten Zustand
        self.save_state_performance()
        
        return self.state_performance[regime_type]
    
    def get_state_performance(self, state_label):
        """
        Ruft die Performance-Statistik für einen Zustand ab.
        
        Args:
            state_label: Label des Zustands
            
        Returns:
            dict: Performance-Statistiken oder None wenn nicht verfügbar
        """
        regime_type = self._determine_regime_type(state_label)
        return self.state_performance.get(regime_type, None)
    
    def get_optimal_parameters(self, state_label, current_price, atr_value=None, custom_factors=None):
        """
        Berechnet optimale Trading-Parameter basierend auf historischer Performance.
        
        Args:
            state_label: Label des Zustands
            current_price: Aktueller Preis
            atr_value: ATR-Wert (falls None, wird Standard-Wert verwendet)
            custom_factors: Benutzerdefinierte Anpassungsfaktoren
            
        Returns:
            dict: Optimale Trading-Parameter
        """
        # Bestimme Regime-Typ
        regime_type = self._determine_regime_type(state_label)
        
        # Hole Performance-Daten
        performance = self.state_performance.get(regime_type, {})
        
        # Default-ATR, falls nicht angegeben
        if atr_value is None:
            atr_value = self.current_volatility.get("atr", 0.01)
        
        # Optimale TP/SL-Werte berechnen
        # Bei guter historischer Performance: höhere TP/SL-Ratio
        win_rate = performance.get("win_rate", 0.5)
        avg_profit = performance.get("avg_profit_pips", 0)
        
        # Basis TP/SL-Ratios
        tp_sl_config = self.tp_sl_ratios.get(regime_type, {"tp_atr": 2.0, "sl_atr": 1.0})
        
        # Anpassung basierend auf historischer Performance
        if win_rate > 0.6:  # Überdurchschnittliche Win-Rate
            tp_factor = tp_sl_config["tp_atr"] * 1.2
            sl_factor = tp_sl_config["sl_atr"] * 0.8
        elif win_rate < 0.4:  # Unterdurchschnittliche Win-Rate
            tp_factor = tp_sl_config["tp_atr"] * 0.8
            sl_factor = tp_sl_config["sl_atr"] * 1.2
        else:  # Durchschnittliche Win-Rate
            tp_factor = tp_sl_config["tp_atr"]
            sl_factor = tp_sl_config["sl_atr"]
        
        # Berücksichtige benutzerdefinierte Faktoren, falls vorhanden
        if custom_factors:
            if "tp_factor" in custom_factors:
                tp_factor *= custom_factors["tp_factor"]
            if "sl_factor" in custom_factors:
                sl_factor *= custom_factors["sl_factor"]
        
        # Berechne TP/SL-Distanzen
        direction = "LONG" if any(term in state_label for term in ["Bullish", "Bull"]) else "SHORT"
        
        # ATR in Pips umrechnen
        atr_pips = atr_value / self.pip_value
        
        tp_pips = int(atr_pips * tp_factor)
        sl_pips = int(atr_pips * sl_factor)
        
        # Berechne TP/SL-Preise
        if direction == "LONG":
            tp_price = current_price + tp_pips * self.pip_value
            sl_price = current_price - sl_pips * self.pip_value
        else:
            tp_price = current_price - tp_pips * self.pip_value
            sl_price = current_price + sl_pips * self.pip_value
        
        # Positionsgröße berechnen
        risk_pct = self.base_risk_pct
        if win_rate > 0.55:
            risk_pct *= (1 + (win_rate - 0.55) * 2)  # Erhöhe Risiko bei hoher Win-Rate
        elif win_rate < 0.45:
            risk_pct *= (1 - (0.45 - win_rate) * 2)  # Reduziere Risiko bei niedriger Win-Rate
        
        position_size = self.calculate_position_size(risk_pct, sl_pips)
        
        return {
            "regime_type": regime_type,
            "direction": direction,
            "entry_price": round(current_price, 3),
            "take_profit": round(tp_price, 3),
            "stop_loss": round(sl_price, 3),
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "tp_factor": round(tp_factor, 2),
            "sl_factor": round(sl_factor, 2),
            "risk_reward": round(tp_pips / sl_pips, 2),
            "position_size": position_size,
            "win_rate": round(win_rate, 2),
            "avg_profit_pips": round(avg_profit, 1),
            "risk_percentage": round(risk_pct, 2)
        }
