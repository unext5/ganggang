import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from collections import deque, defaultdict
import random
import math
import json
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

class SyntheticOrderBookGenerator:
    """
    Generiert synthetische Order Book Daten für das Training von Trading-Modellen.
    Verwendet statistische Eigenschaften von tatsächlichen Märkten, um realistische
    Simulationen zu erzeugen.
    """
    def __init__(self, base_symbol="GBPJPY", volatility=None, pip_size=0.01, 
                 max_levels=10, calibration_data=None):
        """
        Initialisiert den Generator für synthetische Order Book Daten.
        
        Args:
            base_symbol: Haupthandelssymbol (für statistische Eigenschaften)
            volatility: Optionaler Volatilitätswert (falls None, wird aus den Daten abgeleitet)
            pip_size: Größe eines Pips für das Symbol
            max_levels: Maximale Anzahl von Price Levels im Order Book
            calibration_data: Optionale echte Order Book Daten für Kalibrierung
        """
        self.base_symbol = base_symbol
        self.volatility = volatility
        self.pip_size = pip_size
        self.max_levels = max_levels
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('synthetic_orderbook')
        
        # Statistische Parameter für die Generierung
        self.params = {
            # Allgemeine Eigenschaften
            "spread_mean": 2.0,  # Spread in Pips (Mittelwert)
            "spread_std": 0.5,   # Spread-Standardabweichung
            "volume_decay": 1.2,  # Abnahme des Volumens mit der Entfernung vom Spread
            "price_step": 0.1,   # Minimale Preisänderung in Pips
            
            # Volumenverteilung
            "volume_mean": 10.0,  # Mittleres Volumen pro Level
            "volume_std": 5.0,    # Standardabweichung des Volumens
            "volume_skew": 0.5,   # Schiefe der Volumenverteilung
            
            # Marktbedingungsparameter
            "imbalance_range": (0.3, 3.0),  # Bereich für Bid/Ask-Imbalance
            "iceberg_prob": 0.15,  # Wahrscheinlichkeit für versteckte Liquidität
            "institutional_prob": 0.25,  # Wahrscheinlichkeit für institutionelle Aktivität
            
            # Dynamische Eigenschaften
            "refresh_rate": 0.3,  # Rate, mit der Order Book aktualisiert wird
            "price_impact": 0.5,  # Einfluss des Handelsvolumens auf den Preis
            
            # Marktzustandsabhängige Parameter
            "market_states": {
                "trending_bull": {
                    "imbalance_mult": 1.5,  # Multiplikator für Bid/Ask-Imbalance
                    "volume_mult": 1.2,     # Multiplikator für Volumen
                    "spread_mult": 0.8      # Multiplikator für Spread
                },
                "trending_bear": {
                    "imbalance_mult": 0.7,
                    "volume_mult": 1.2,
                    "spread_mult": 0.8
                },
                "ranging": {
                    "imbalance_mult": 1.0,
                    "volume_mult": 1.0,
                    "spread_mult": 1.0
                },
                "high_volatility": {
                    "imbalance_mult": 1.3,
                    "volume_mult": 1.5,
                    "spread_mult": 1.5
                },
                "low_volatility": {
                    "imbalance_mult": 0.9,
                    "volume_mult": 0.8,
                    "spread_mult": 0.7
                }
            }
        }
        
        # Kalibriere Parameter, falls Daten vorhanden
        if calibration_data is not None:
            self.calibrate_with_real_data(calibration_data)
        
        # Letzter generierter Order Book-Zustand
        self.last_book = None
        self.last_mid_price = None
        
        # Modelle für die Anomalieerkennung und Clusteranalyse
        self.anomaly_detector = None
        self.pattern_clusters = None
        
        # Historie für kontinuierliche Kalibrierung
        self.generated_books = deque(maxlen=1000)

    def set_default_calibration(self, atr_pips=10.0, spread_pips=1.0, price_levels=10):
        """
        Setzt Standard-Kalibrierungsparameter, wenn keine Preisdaten verfügbar sind.
    
        Args:
            atr_pips: Standard-ATR in Pips
            spread_pips: Standard-Spread in Pips
            price_levels: Anzahl der Preislevels
        """
        self.atr_pips = atr_pips
        self.spread_pips = spread_pips
        self.max_levels = price_levels
    
        # Pip-Größe basierend auf Symbol
        if self.base_symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            self.pip_size = 0.0001
        elif self.base_symbol in ['GBPJPY', 'EURJPY']:
            self.pip_size = 0.01
    
        # Konvertiere Pips in Preiseinheiten
        self.atr = self.atr_pips * self.pip_size
        self.spread = self.spread_pips * self.pip_size
    
        logging.info(f"Synthetic OB: Standard-Kalibrierung gesetzt (ATR={atr_pips} Pips, Spread={spread_pips} Pips)")
        return True
    
    def calibrate_with_real_data(self, calibration_data):
        """
        Kalibriert die Generator-Parameter mit echten Order Book Daten.
        
        Args:
            calibration_data: Liste von Order Book Snapshots
        """
        if not calibration_data or len(calibration_data) < 10:
            self.logger.warning("Nicht genug Kalibrierungsdaten vorhanden.")
            return
        
        self.logger.info(f"Kalibriere mit {len(calibration_data)} Order Book Snapshots")
        
        # Extrahiere Metriken aus den Daten
        spreads = []
        bid_volumes = []
        ask_volumes = []
        imbalances = []
        
        for book in calibration_data:
            if "bids" not in book or "asks" not in book:
                continue
                
            bids = book["bids"]
            asks = book["asks"]
            
            if not bids or not asks:
                continue
            
            # Spread berechnen
            spread = asks[0]["price"] - bids[0]["price"]
            spreads.append(spread / self.pip_size)  # in Pips
            
            # Volumen aggregieren
            bid_vol = sum(b["volume"] for b in bids)
            ask_vol = sum(a["volume"] for a in asks)
            
            bid_volumes.append(bid_vol)
            ask_volumes.append(ask_vol)
            
            # Imbalance
            if ask_vol > 0:
                imbalances.append(bid_vol / ask_vol)
        
        # Parameter aktualisieren, wenn genügend Daten vorhanden
        if spreads:
            self.params["spread_mean"] = np.mean(spreads)
            self.params["spread_std"] = np.std(spreads)
            self.logger.info(f"Kalibrierter Spread: {self.params['spread_mean']:.2f} ± {self.params['spread_std']:.2f} Pips")
        
        if bid_volumes and ask_volumes:
            # Volumenparameter aktualisieren
            all_volumes = bid_volumes + ask_volumes
            
            self.params["volume_mean"] = np.mean(all_volumes) / self.max_levels
            self.params["volume_std"] = np.std(all_volumes) / self.max_levels
            
            # Berechne Schiefe
            if len(all_volumes) > 2:
                from scipy.stats import skew
                self.params["volume_skew"] = skew(all_volumes)
                
            self.logger.info(f"Kalibriertes Volumen: μ={self.params['volume_mean']:.2f}, σ={self.params['volume_std']:.2f}")
        
        if imbalances:
            # Aktualisiere Imbalance-Bereich
            min_imb = max(0.1, np.percentile(imbalances, 5))
            max_imb = min(10.0, np.percentile(imbalances, 95))
            
            self.params["imbalance_range"] = (min_imb, max_imb)
            self.logger.info(f"Kalibrierter Imbalance-Bereich: {min_imb:.2f} - {max_imb:.2f}")
        
        # Optional: Preisschritt kalibrieren
        price_steps = []
        for book in calibration_data:
            if "bids" in book and len(book["bids"]) > 1:
                # Durchschnittlicher Abstand zwischen Preislevels
                bids = sorted(book["bids"], key=lambda x: x["price"], reverse=True)
                steps = [abs(bids[i]["price"] - bids[i+1]["price"]) for i in range(len(bids)-1)]
                price_steps.extend(steps)
        
        if price_steps:
            # Verwende den Median als robusteren Schätzer
            self.params["price_step"] = np.median(price_steps) / self.pip_size
            self.logger.info(f"Kalibrierter Preisschritt: {self.params['price_step']:.4f} Pips")
    
    def calibrate_with_price_data(self, price_data):
        """
        Kalibriert die Generator-Parameter mit historischen Preisdaten.
        
        Args:
            price_data: DataFrame mit OHLCV-Daten
        """
        # --- FIX: Reduziere Mindestdatenanforderung und verbessere Logging --- 
        MIN_PRICE_DATA_POINTS = 30 # Reduziert von 100
        
        if price_data is None or len(price_data) < MIN_PRICE_DATA_POINTS:
            self.logger.warning(f"Nicht genug Preisdaten für Kalibrierung (benötigt={MIN_PRICE_DATA_POINTS}, vorhanden={len(price_data) if price_data is not None else 0}). Verwende Standard-/vorherige Werte.")
            # Behalte vorherige Werte bei oder setze robuste Defaults, statt abzubrechen.
            # Beispiel: Wenn self.volatility schon existiert, behalte es, sonst Default.
            if not hasattr(self, 'volatility') or self.volatility is None:
                 self.volatility = 5.0 # Robuster Defaultwert (z.B. 5 Pips ATR)
                 self.params["spread_mean"] = 1.0
                 self.params["spread_std"] = 0.2
            # Breche hier nicht mehr ab, damit der Rest des Skripts weiterlaufen kann.
            return
            
        # --- END FIX ---
        
        # Volatilität berechnen
        if 'high' in price_data.columns and 'low' in price_data.columns:
            # Berechne ATR (Average True Range)
            highs = price_data['high'].values
            lows = price_data['low'].values
            closes = price_data['close'].values if 'close' in price_data.columns else None
            
            if closes is not None:
                tr = []
                for i in range(1, len(highs)):
                    tr1 = highs[i] - lows[i]
                    tr2 = abs(highs[i] - closes[i-1])
                    tr3 = abs(lows[i] - closes[i-1])
                    tr.append(max(tr1, tr2, tr3))
                
                atr = np.mean(tr) if tr else None
            else:
                # Einfacherer Ansatz ohne Close-Daten
                ranges = highs - lows
                atr = np.mean(ranges)
            
            if atr is not None:
                # Umrechnung in Pips
                atr_pips = atr / self.pip_size
                self.volatility = atr_pips
                
                # Aktualisiere spreadabhängige Parameter
                self.params["spread_mean"] = max(1.0, atr_pips * 0.05)  # ~5% der ATR
                self.params["spread_std"] = self.params["spread_mean"] * 0.2
                
                # Volumen-Skalierung basierend auf Volatilität
                vol_scale = min(3.0, max(0.5, atr_pips / 10))  # Begrenze Skalierung
                self.params["volume_mean"] *= vol_scale
                self.params["volume_std"] *= vol_scale
                
                self.logger.info(f"Kalibrierung mit Preisdaten: ATR={atr_pips:.2f} Pips, " +
                               f"Spread={self.params['spread_mean']:.2f} Pips")
        
        # Korrelation von Preis und Volumen für Volumenverteilung
        if 'close' in price_data.columns and 'volume' in price_data.columns:
            # Berechne Preisänderungen
            price_data['returns'] = price_data['close'].pct_change()
            
            # Korreliere absolute Returns mit Volumen
            abs_returns = price_data['returns'].abs().dropna()
            volumes = price_data['volume'].iloc[1:].values  # Überspringe ersten Wert wegen pct_change
            
            if len(abs_returns) == len(volumes) and len(abs_returns) > 10:
                correlation = np.corrcoef(abs_returns, volumes)[0, 1]
                
                # Aktualisiere volume_skew basierend auf Korrelation
                if not np.isnan(correlation):
                    # Positive Korrelation = höheres Volumen bei größeren Preisänderungen
                    self.params["volume_skew"] = max(-1.0, min(1.0, correlation))
                    self.logger.info(f"Volume-Return Korrelation: {correlation:.4f}")
        
        # Analyse der Marktbedingungen und Trends
        if 'close' in price_data.columns and len(price_data) > 20:
            # Einfache Trendidentifikation
            short_ma = price_data['close'].rolling(window=5).mean()
            long_ma = price_data['close'].rolling(window=20).mean()
            
            # Aktueller Trend
            if len(long_ma.dropna()) > 0:
                last_short = short_ma.iloc[-1]
                last_long = long_ma.iloc[-1]
                
                trend_strength = abs(last_short - last_long) / last_long
                
                if last_short > last_long:
                    # Aufwärtstrend
                    self.current_market_state = "trending_bull"
                elif last_short < last_long:
                    # Abwärtstrend
                    self.current_market_state = "trending_bear"
                else:
                    # Seitwärtsbewegung
                    self.current_market_state = "ranging"
                
                # Volatilitätsbasierter Zustand
                recent_volatility = price_data['close'].pct_change().abs().rolling(window=10).mean().iloc[-1]
                hist_volatility = price_data['close'].pct_change().abs().mean()
                
                if not np.isnan(recent_volatility) and not np.isnan(hist_volatility):
                    vol_ratio = recent_volatility / hist_volatility
                    
                    if vol_ratio > 1.5:
                        self.current_market_state = "high_volatility"
                    elif vol_ratio < 0.5:
                        self.current_market_state = "low_volatility"
                
                self.logger.info(f"Identifizierter Marktzustand: {self.current_market_state}")
    
    def generate_order_book(self, current_price=None, market_state=None, seed=None):
        """
        Generiert ein synthetisches Order Book basierend auf den konfigurierten Parametern.
        
        Args:
            current_price: Aktueller Preis, um den herum das Order Book generiert wird
            market_state: Optionaler Marktzustand, beeinflusst die Generierungsparameter
            seed: Optionaler Seed für Reproduzierbarkeit
        
        Returns:
            dict: Generiertes Order Book mit bids und asks
        """
        # Setze Seed für Reproduzierbarkeit, wenn angegeben
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Falls kein Preis angegeben, verwende den letzten oder generiere einen
        if current_price is None:
            if self.last_mid_price is not None:
                # Füge kleine zufällige Bewegung hinzu
                volatility = self.volatility if self.volatility is not None else 5.0  # Default 5 Pips
                price_change = np.random.normal(0, volatility * 0.1) * self.pip_size
                current_price = self.last_mid_price + price_change
            else:
                # Standardpreis, falls keiner bekannt ist
                current_price = 150.0  # Beispielpreis für GBPJPY
        
        # Verwende den angegebenen Marktzustand oder den gespeicherten
        if market_state is None and hasattr(self, 'current_market_state'):
            market_state = self.current_market_state
        
        # Hole Zustandsparameter für den aktuellen Marktzustand
        state_params = self.params["market_states"].get(
            market_state, 
            self.params["market_states"]["ranging"]  # Default: ranging
        )
        
        # Angepasste Parameter für diesen Zustand
        spread_mean = self.params["spread_mean"] * state_params["spread_mult"]
        spread_std = self.params["spread_std"] * state_params["spread_mult"]
        volume_mean = self.params["volume_mean"] * state_params["volume_mult"]
        volume_std = self.params["volume_std"] * state_params["volume_mult"]
        
        # Imbalance-Bereich anpassen
        imb_min, imb_max = self.params["imbalance_range"]
        imb_mult = state_params["imbalance_mult"]
        
        if imb_mult > 1.0:
            # Mehr Bid-Imbalance (bullisch)
            adjusted_imbalance = (imb_min, imb_max * imb_mult)
        elif imb_mult < 1.0:
            # Mehr Ask-Imbalance (bärisch)
            adjusted_imbalance = (imb_min * imb_mult, imb_max)
        else:
            adjusted_imbalance = (imb_min, imb_max)
        
        # Generiere Spread
        spread = max(1.0, np.random.normal(spread_mean, spread_std)) * self.pip_size
        
        # Mid-Preis und Best Bid/Ask
        mid_price = current_price
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Generiere Imbalance-Faktor
        imbalance = np.random.uniform(adjusted_imbalance[0], adjusted_imbalance[1])
        
        # Generiere Orders
        bids = []
        asks = []
        
        # Bid-Orders generieren
        current_bid = best_bid
        for i in range(self.max_levels):
            # Preisabnahme (größer für tiefere Levels)
            price_decrease = self.params["price_step"] * (1 + 0.1 * i) * self.pip_size
            current_bid -= price_decrease
            
            # Volumen mit abnehmender Tendenz zu tieferen Levels
            decay_factor = math.exp(-i / self.params["volume_decay"])
            level_volume = max(0.1, np.random.normal(
                volume_mean * decay_factor * imbalance,
                volume_std * decay_factor
            ))
            
            # Volumen-Spikes einfügen (für institutionelle Aktivität)
            if np.random.random() < self.params["institutional_prob"] * (i+1)/self.max_levels:
                level_volume *= np.random.uniform(2.0, 5.0)
            
            bids.append({"price": current_bid, "volume": level_volume})
        
        # Ask-Orders generieren
        current_ask = best_ask
        for i in range(self.max_levels):
            # Preiszunahme
            price_increase = self.params["price_step"] * (1 + 0.1 * i) * self.pip_size
            current_ask += price_increase
            
            # Volumen mit abnehmender Tendenz zu höheren Levels
            decay_factor = math.exp(-i / self.params["volume_decay"])
            level_volume = max(0.1, np.random.normal(
                volume_mean * decay_factor / imbalance,  # Inverse of imbalance for asks
                volume_std * decay_factor
            ))
            
            # Volumen-Spikes einfügen
            if np.random.random() < self.params["institutional_prob"] * (i+1)/self.max_levels:
                level_volume *= np.random.uniform(2.0, 5.0)
            
            asks.append({"price": current_ask, "volume": level_volume})
        
        # "Iceberg Orders" hinzufügen (versteckte Liquidität)
        if np.random.random() < self.params["iceberg_prob"]:
            # Zufälliges Level
            iceberg_idx = np.random.randint(0, min(3, len(bids)))
            bids[iceberg_idx]["iceberg"] = True
            bids[iceberg_idx]["hidden_volume"] = bids[iceberg_idx]["volume"] * np.random.uniform(3.0, 10.0)
        
        if np.random.random() < self.params["iceberg_prob"]:
            iceberg_idx = np.random.randint(0, min(3, len(asks)))
            asks[iceberg_idx]["iceberg"] = True
            asks[iceberg_idx]["hidden_volume"] = asks[iceberg_idx]["volume"] * np.random.uniform(3.0, 10.0)
        
        # Manchmal: Liquiditätswand in eine Richtung (starker Support/Resistance)
        if np.random.random() < 0.15:  # 15% Wahrscheinlichkeit
            is_support = np.random.random() < 0.5
            
            if is_support:
                # Support-Level (Bid-Seite)
                level_idx = np.random.randint(1, min(5, len(bids)))
                bids[level_idx]["volume"] *= np.random.uniform(5.0, 15.0)
                bids[level_idx]["is_support"] = True
            else:
                # Resistance-Level (Ask-Seite)
                level_idx = np.random.randint(1, min(5, len(asks)))
                asks[level_idx]["volume"] *= np.random.uniform(5.0, 15.0)
                asks[level_idx]["is_resistance"] = True
        
        # Order Book zusammenstellen
        order_book = {
            "bids": bids,
            "asks": asks,
            "mid_price": mid_price,
            "spread": spread / self.pip_size,  # in Pips für einfachere Interpretation
            "imbalance": imbalance,
            "timestamp": datetime.now(),
            "market_state": market_state
        }
        
        # Speichern für zukünftige Generationen
        self.last_book = order_book
        self.last_mid_price = mid_price
        
        # Füge zur Historie hinzu
        self.generated_books.append(order_book)
        
        return order_book
    
    def generate_sequence(self, length, base_price=None, price_data=None, 
                         market_state=None, with_price_impact=True):
        """
        Generiert eine Sequenz von Order Books mit realistischen Übergängen.
        
        Args:
            length: Länge der zu generierenden Sequenz
            base_price: Startpreis für die Sequenz
            price_data: Optional, Preisverlauf zum Alignment der Sequenz
            market_state: Marktzustand für die Sequenz
            with_price_impact: Ob Handelsvolumen den Preis beeinflussen soll
        
        Returns:
            list: Sequenz von Order Books
        """
        sequence = []
        current_price = base_price
        
        # Wenn Preisdaten vorhanden, verwende sie zur Orientierung
        prices = None
        if price_data is not None and len(price_data) >= length:
            if 'close' in price_data.columns:
                prices = price_data['close'].values[-length:]
                if base_price is None:
                    current_price = prices[0]
        
        # Generiere das erste Order Book
        book = self.generate_order_book(current_price, market_state)
        sequence.append(book)
        
        # Generiere die restlichen Order Books in der Sequenz
        for i in range(1, length):
            # Aktualisiere den Preis
            if prices is not None:
                # Folge den tatsächlichen Preisdaten
                current_price = prices[i]
            else:
                # Berechne neue Preisänderung
                volatility = self.volatility if self.volatility is not None else 5.0
                random_drift = np.random.normal(0, volatility * 0.05) * self.pip_size
                
                # Optional: Preis-Impact basierend auf dem letzten Order Book
                price_impact = 0
                if with_price_impact and book is not None:
                    # Imbalance beeinflusst die Preisrichtung
                    imbalance = book.get("imbalance", 1.0)
                    impact_strength = self.params["price_impact"]
                    
                    # Imbalance > 1 bedeutet mehr Bids als Asks -> Preis steigt tendenziell
                    imbalance_effect = math.log(imbalance) * impact_strength * self.pip_size
                    price_impact = imbalance_effect
                
                # Gesamte Preisänderung
                price_change = random_drift + price_impact
                current_price = sequence[-1]["mid_price"] + price_change
            
            # Bestimme Marktzustand für dieses Book
            current_state = market_state
            
            # Zufällige Zustandsübergänge, wenn kein fester Zustand vorgegeben
            if current_state is None and i > 0 and i % 10 == 0:  # Alle 10 Schritte
                prev_state = sequence[-1].get("market_state", "ranging")
                
                # Zustandsübergänge mit Wahrscheinlichkeiten
                transition_probs = {
                    "trending_bull": {
                        "trending_bull": 0.8,
                        "ranging": 0.15, 
                        "high_volatility": 0.05
                    },
                    "trending_bear": {
                        "trending_bear": 0.8, 
                        "ranging": 0.15, 
                        "high_volatility": 0.05
                    },
                    "ranging": {
                        "ranging": 0.7,
                        "trending_bull": 0.1,
                        "trending_bear": 0.1,
                        "low_volatility": 0.1
                    },
                    "high_volatility": {
                        "high_volatility": 0.7,
                        "trending_bull": 0.15,
                        "trending_bear": 0.15
                    },
                    "low_volatility": {
                        "low_volatility": 0.8,
                        "ranging": 0.2
                    }
                }
                
                # Gewichtete Zufallsauswahl für nächsten Zustand
                if prev_state in transition_probs:
                    states = list(transition_probs[prev_state].keys())
                    probs = list(transition_probs[prev_state].values())
                    current_state = np.random.choice(states, p=probs)
            
            # Generiere nächstes Order Book
            book = self.generate_order_book(current_price, current_state)
            
            # Progressive Aktualisierung statt kompletter Neugenerierung
            if i > 0 and np.random.random() > self.params["refresh_rate"]:
                # Partiell aktualisieren statt komplett neu generieren
                # Dies erzeugt realistischere Übergänge zwischen snapshots
                prev_book = sequence[-1]
                
                # Wähle zufällige Levels zum Aktualisieren
                for side, levels in [("bids", book["bids"]), ("asks", book["asks"])]:
                    prev_levels = prev_book[side]
                    
                    for j in range(min(len(levels), len(prev_levels))):
                        if np.random.random() < 0.7:  # 70% Chance, ein Level zu behalten
                            # Behalte Volumen, passe nur den Preis an
                            if side == "bids":
                                # Für Bids: neuer Preis = alter Preis + relativer Preisunterschied
                                price_diff = book["mid_price"] - prev_book["mid_price"]
                                levels[j]["price"] = prev_levels[j]["price"] + price_diff
                            else:
                                # Für Asks: ähnlich wie für Bids
                                price_diff = book["mid_price"] - prev_book["mid_price"]
                                levels[j]["price"] = prev_levels[j]["price"] + price_diff
                            
                            # Volumen leicht anpassen
                            volume_change = np.random.normal(0, prev_levels[j]["volume"] * 0.1)
                            new_volume = max(0.1, prev_levels[j]["volume"] + volume_change)
                            levels[j]["volume"] = new_volume
                            
                            # Übertrage spezielle Eigenschaften
                            for key in ["iceberg", "hidden_volume", "is_support", "is_resistance"]:
                                if key in prev_levels[j]:
                                    levels[j][key] = prev_levels[j][key]
            
            sequence.append(book)
        
        return sequence
    
    def create_training_dataset(self, price_data, n_samples=100, sequence_length=20):
        """
        Erstellt einen Trainingsdatensatz mit Order Book Features.
        
        Args:
            price_data: Preisdaten als DataFrame
            n_samples: Anzahl der Sequenzen im Datensatz
            sequence_length: Länge jeder Sequenz
        
        Returns:
            dict: Trainingsdatensatz mit Features und Labels
        """
        # Kalibriere Generator mit Preisdaten
        self.calibrate_with_price_data(price_data)
        
        # Liste zum Speichern der Sequenzen
        all_sequences = []
        
        # Returns aus den Preisdaten berechnen
        if 'close' in price_data.columns:
            price_data['returns'] = price_data['close'].pct_change()
            price_data['volatility'] = price_data['returns'].rolling(window=20).std()
            
            # Marktzustände identifizieren
            price_data['ma_short'] = price_data['close'].rolling(window=5).mean()
            price_data['ma_long'] = price_data['close'].rolling(window=20).mean()
            
            # Entferne NaN-Werte
            price_data = price_data.dropna()
        
        # Samples generieren
        for i in range(n_samples):
            # Zufälligen Startpunkt auswählen
            if len(price_data) > sequence_length:
                start_idx = np.random.randint(0, len(price_data) - sequence_length)
                sample_prices = price_data.iloc[start_idx:start_idx+sequence_length]
                
                # Marktzustand identifizieren
                if 'ma_short' in sample_prices.columns and 'ma_long' in sample_prices.columns:
                    last_ma_short = sample_prices['ma_short'].iloc[-1]
                    last_ma_long = sample_prices['ma_long'].iloc[-1]
                    
                    if last_ma_short > last_ma_long * 1.01:
                        market_state = "trending_bull"
                    elif last_ma_short < last_ma_long * 0.99:
                        market_state = "trending_bear"
                    else:
                        market_state = "ranging"
                    
                    # Volatilität
                    if 'volatility' in sample_prices.columns:
                        recent_vol = sample_prices['volatility'].iloc[-1]
                        
                        if not pd.isna(recent_vol):
                            mean_vol = price_data['volatility'].mean()
                            
                            if recent_vol > mean_vol * 1.5:
                                market_state = "high_volatility"
                            elif recent_vol < mean_vol * 0.5:
                                market_state = "low_volatility"
                else:
                    market_state = None
                
                # Start-Preis
                start_price = sample_prices['close'].iloc[0]
                
                # Order Book Sequenz generieren
                ob_sequence = self.generate_sequence(
                    sequence_length, 
                    base_price=start_price,
                    price_data=sample_prices,
                    market_state=market_state
                )
                
                # Preise als separaten Array speichern
                prices = sample_prices['close'].values
                
                # Features extrahieren
                ob_features = self._extract_features_from_sequence(ob_sequence)
                
                # Label: künftige Preisbewegung
                future_return = 0
                if i < len(price_data) - (start_idx + sequence_length):
                    future_price = price_data['close'].iloc[start_idx + sequence_length]
                    current_price = price_data['close'].iloc[start_idx + sequence_length - 1]
                    future_return = (future_price / current_price - 1) * 100  # Prozentuale Änderung
                
                # Sequenz mit Metadaten speichern
                all_sequences.append({
                    "order_book_sequence": ob_sequence,
                    "features": ob_features,
                    "prices": prices,
                    "future_return": future_return,
                    "market_state": market_state,
                    "start_idx": start_idx
                })
            
            # Log-Fortschritt
            if (i+1) % 10 == 0:
                self.logger.info(f"Generierung: {i+1}/{n_samples} Sequenzen erstellt")
        
        # Extraktion von zusammengeführten Features für Training
        X, y = self._prepare_training_data(all_sequences)
        
        return {
            "sequences": all_sequences,
            "X": X,
            "y": y,
            "feature_names": self._get_feature_names()
        }
    
    def _extract_features_from_sequence(self, ob_sequence):
        """
        Extrahiert trainierbare Features aus einer Order Book Sequenz.
        
        Args:
            ob_sequence: Sequenz von Order Books
        
        Returns:
            np.array: Matrix mit Features [timesteps, features]
        """
        n_steps = len(ob_sequence)
        n_features = len(self._get_feature_names())
        
        # Initialisiere Feature-Matrix
        features = np.zeros((n_steps, n_features))
        
        for t, book in enumerate(ob_sequence):
            feat_idx = 0
            
            # 1. Basis-Metriken
            bids = book["bids"]
            asks = book["asks"]
            
            # Spread
            features[t, feat_idx] = book["spread"]
            feat_idx += 1
            
            # Bid/Ask Imbalance
            features[t, feat_idx] = book["imbalance"]
            feat_idx += 1
            
            # Volumen-Metriken
            total_bid_vol = sum(b["volume"] for b in bids)
            total_ask_vol = sum(a["volume"] for a in asks)
            
            features[t, feat_idx] = total_bid_vol
            feat_idx += 1
            
            features[t, feat_idx] = total_ask_vol
            feat_idx += 1
            
            # Volume Ratio
            features[t, feat_idx] = total_bid_vol / total_ask_vol if total_ask_vol > 0 else 1.0
            feat_idx += 1
            
            # 2. Erweiterte Metriken
            
            # Volumen-gewichtete Preise
            vwap_bid = sum(b["price"] * b["volume"] for b in bids) / total_bid_vol if total_bid_vol > 0 else bids[0]["price"]
            vwap_ask = sum(a["price"] * a["volume"] for a in asks) / total_ask_vol if total_ask_vol > 0 else asks[0]["price"]
            
            features[t, feat_idx] = (vwap_bid - bids[0]["price"]) / self.pip_size  # Abstand in Pips
            feat_idx += 1
            
            features[t, feat_idx] = (asks[0]["price"] - vwap_ask) / self.pip_size  # Abstand in Pips
            feat_idx += 1
            
            # Konzentration des Volumens
            # Gini-Koeffizient für Volumensverteilung
            bid_volumes = [b["volume"] for b in bids]
            ask_volumes = [a["volume"] for a in asks]
            
            features[t, feat_idx] = self._gini_coefficient(bid_volumes)
            feat_idx += 1
            
            features[t, feat_idx] = self._gini_coefficient(ask_volumes)
            feat_idx += 1
            
            # 3. Top-Level Metriken
            
            # Volumen in den Top-3-Levels
            top_bid_vol = sum(b["volume"] for b in bids[:min(3, len(bids))])
            top_ask_vol = sum(a["volume"] for a in asks[:min(3, len(asks))])
            
            features[t, feat_idx] = top_bid_vol / total_bid_vol if total_bid_vol > 0 else 0
            feat_idx += 1
            
            features[t, feat_idx] = top_ask_vol / total_ask_vol if total_ask_vol > 0 else 0
            feat_idx += 1
            
            # Relative Liquiditätsverteilung
            for i in range(min(3, len(bids))):
                features[t, feat_idx] = bids[i]["volume"] / total_bid_vol if total_bid_vol > 0 else 0
                feat_idx += 1
            
            # Fülle mit 0 auf, falls weniger als 3 Levels
            for i in range(len(bids), 3):
                features[t, feat_idx] = 0
                feat_idx += 1
                
            for i in range(min(3, len(asks))):
                features[t, feat_idx] = asks[i]["volume"] / total_ask_vol if total_ask_vol > 0 else 0
                feat_idx += 1
            
            # Fülle mit 0 auf, falls weniger als 3 Levels
            for i in range(len(asks), 3):
                features[t, feat_idx] = 0
                feat_idx += 1
            
            # 4. Institutionelle Aktivität / "Wände"
            
            # Erkenne Wände (große Volumen im Vergleich zu umgebenden Levels)
            bid_walls = self._detect_walls(bids, threshold=3.0)
            ask_walls = self._detect_walls(asks, threshold=3.0)
            
            features[t, feat_idx] = len(bid_walls)
            feat_idx += 1
            
            features[t, feat_idx] = len(ask_walls)
            feat_idx += 1
            
            # 5. Hidden Liquidity Schätzung
            
            # Anzahl der Iceberg-Orders
            iceberg_bids = sum(1 for b in bids if b.get("iceberg", False))
            iceberg_asks = sum(1 for a in asks if a.get("iceberg", False))
            
            features[t, feat_idx] = iceberg_bids
            feat_idx += 1
            
            features[t, feat_idx] = iceberg_asks
            feat_idx += 1
            
            # Hidden Volumen
            hidden_bid_vol = sum(b.get("hidden_volume", 0) for b in bids)
            hidden_ask_vol = sum(a.get("hidden_volume", 0) for a in asks)
            
            features[t, feat_idx] = hidden_bid_vol / total_bid_vol if total_bid_vol > 0 else 0
            feat_idx += 1
            
            features[t, feat_idx] = hidden_ask_vol / total_ask_vol if total_ask_vol > 0 else 0
            feat_idx += 1
            
            # Überprüfe, ob wir alle Features ausgefüllt haben
            assert feat_idx == n_features, f"Feature-Extraktion fehlerhaft: {feat_idx} von {n_features} Features extrahiert"
        
        return features
    
    def _prepare_training_data(self, sequences):
        """
        Bereitet Trainingsdaten aus Order Book Sequenzen vor.
        
        Args:
            sequences: Liste von Order Book Sequenzen mit Features und Labels
        
        Returns:
            tuple: (X, y) Feature-Matrix und Labels
        """
        if not sequences:
            return np.array([]), np.array([])
        
        # Extrahiere Features und Labels
        X_list = []
        y_list = []
        
        for seq in sequences:
            X_list.append(seq["features"])
            
            # Binäres Label: 1 für positive Returns, 0 für negative
            y = 1 if seq["future_return"] > 0 else 0
            y_list.append(y)
        
        # Stapel von 3D-Arrays: [Samples, Timesteps, Features]
        X = np.stack(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def _get_feature_names(self):
        """
        Gibt Namen für die extrahierten Features zurück.
        
        Returns:
            list: Feature-Namen
        """
        features = [
            "spread",                  # Spread in Pips
            "imbalance",               # Bid/Ask Imbalance
            "total_bid_volume",        # Gesamtvolumen auf der Bid-Seite
            "total_ask_volume",        # Gesamtvolumen auf der Ask-Seite
            "volume_ratio",            # Verhältnis Bid/Ask-Volumen
            
            "vwap_bid_distance",       # Abstand des volumengewichteten Bid-Preises vom besten Bid
            "vwap_ask_distance",       # Abstand des volumengewichteten Ask-Preises vom besten Ask
            "bid_volume_concentration", # Gini-Koeffizient der Bid-Volumenverteilung
            "ask_volume_concentration", # Gini-Koeffizient der Ask-Volumenverteilung
            
            "top_bid_concentration",    # Anteil des Volumens in den Top-3-Bid-Levels
            "top_ask_concentration",    # Anteil des Volumens in den Top-3-Ask-Levels
            
            "bid_level_1_ratio",        # Volumensanteil im ersten Bid-Level
            "bid_level_2_ratio",        # Volumensanteil im zweiten Bid-Level
            "bid_level_3_ratio",        # Volumensanteil im dritten Bid-Level
            "ask_level_1_ratio",        # Volumensanteil im ersten Ask-Level
            "ask_level_2_ratio",        # Volumensanteil im zweiten Ask-Level
            "ask_level_3_ratio",        # Volumensanteil im dritten Ask-Level
            
            "bid_walls_count",          # Anzahl erkannter Bid-Wände
            "ask_walls_count",          # Anzahl erkannter Ask-Wände
            
            "iceberg_bids_count",       # Anzahl erkannter Iceberg-Orders auf der Bid-Seite
            "iceberg_asks_count",       # Anzahl erkannter Iceberg-Orders auf der Ask-Seite
            "hidden_bid_volume_ratio",  # Geschätztes verstecktes Volumen auf der Bid-Seite
            "hidden_ask_volume_ratio",  # Geschätztes verstecktes Volumen auf der Ask-Seite
        ]
        
        return features
    
    def calibrate_with_real_and_synthetic(self, real_data, n_synthetic=100, 
                                         learning_rate=0.1, max_iterations=10):
        """
        Kalibriert die Generierungsparameter durch iterativen Vergleich von echten 
        und synthetischen Daten.
        
        Args:
            real_data: Echte Order Book Daten
            n_synthetic: Anzahl zu generierender synthetischer Bücher pro Iteration
            learning_rate: Lernrate für Parameteranpassung
            max_iterations: Maximale Anzahl von Iterationen
        
        Returns:
            dict: Kalibrierungsmetrik vor und nach dem Prozess
        """
        if not real_data or len(real_data) < 10:
            self.logger.warning("Nicht genug echte Daten für Kalibrierung.")
            return {"success": False}
        
        # Extrahiere Metriken aus echten Daten
        real_metrics = self._calculate_metrics_from_books(real_data)
        
        # Initialisiere Tracking
        initial_distance = None
        current_distance = float('inf')
        
        # Iterativer Kalibrierungsprozess
        for iteration in range(max_iterations):
            # Generiere synthetische Daten mit aktuellen Parametern
            synthetic_books = [self.generate_order_book() for _ in range(n_synthetic)]
            
            # Berechne Metriken für synthetische Daten
            synthetic_metrics = self._calculate_metrics_from_books(synthetic_books)
            
            # Berechne Abstand zwischen realen und synthetischen Metriken
            metric_distance = self._calculate_metrics_distance(real_metrics, synthetic_metrics)
            
            # Speichere initialen Abstand
            if initial_distance is None:
                initial_distance = metric_distance
            
            # Update aktuellen Abstand
            current_distance = metric_distance
            
            self.logger.info(f"Iteration {iteration+1}/{max_iterations}: Metrik-Abstand = {metric_distance:.4f}")
            
            # Berechne die optimale Anpassung für Parameter
            param_adjustments = self._calculate_parameter_adjustments(
                real_metrics, synthetic_metrics, learning_rate
            )
            
            # Wende Anpassungen an
            self._apply_parameter_adjustments(param_adjustments)
            
            # Früher Abbruch, wenn die Verbesserung minimal ist
            if iteration > 0 and metric_distance < 0.05:
                self.logger.info("Kalibrierung konvergiert, breche früh ab.")
                break
        
        # Finaler Kalibrierungsbericht
        return {
            "success": True,
            "initial_distance": initial_distance,
            "final_distance": current_distance,
            "improvement": (initial_distance - current_distance) / initial_distance if initial_distance > 0 else 0,
            "iterations": iteration + 1
        }
    
    def _calculate_metrics_from_books(self, books):
        """
        Berechnet zusammenfassende Metriken aus einer Liste von Order Books.
        
        Args:
            books: Liste von Order Books
        
        Returns:
            dict: Zusammenfassende Metriken
        """
        spreads = []
        imbalances = []
        bid_volumes = []
        ask_volumes = []
        bid_concentrations = []
        ask_concentrations = []
        
        for book in books:
            if "bids" not in book or "asks" not in book:
                continue
                
            # Spread
            if "spread" in book:
                spreads.append(book["spread"])
            elif len(book["bids"]) > 0 and len(book["asks"]) > 0:
                spread = (book["asks"][0]["price"] - book["bids"][0]["price"]) / self.pip_size
                spreads.append(spread)
            
            # Imbalance
            if "imbalance" in book:
                imbalances.append(book["imbalance"])
            else:
                total_bid_vol = sum(b["volume"] for b in book["bids"])
                total_ask_vol = sum(a["volume"] for a in book["asks"])
                
                if total_ask_vol > 0:
                    imbalances.append(total_bid_vol / total_ask_vol)
            
            # Volumes
            bid_vol = sum(b["volume"] for b in book["bids"])
            ask_vol = sum(a["volume"] for a in book["asks"])
            
            bid_volumes.append(bid_vol)
            ask_volumes.append(ask_vol)
            
            # Volume concentration
            bid_vol_array = [b["volume"] for b in book["bids"]]
            ask_vol_array = [a["volume"] for a in book["asks"]]
            
            bid_concentrations.append(self._gini_coefficient(bid_vol_array))
            ask_concentrations.append(self._gini_coefficient(ask_vol_array))
        
        # Berechne zusammenfassende Statistiken
        metrics = {}
        
        # Spread-Metriken
        if spreads:
            metrics["spread_mean"] = np.mean(spreads)
            metrics["spread_std"] = np.std(spreads)
            metrics["spread_median"] = np.median(spreads)
            
        # Imbalance-Metriken
        if imbalances:
            metrics["imbalance_mean"] = np.mean(imbalances)
            metrics["imbalance_std"] = np.std(imbalances)
            metrics["imbalance_median"] = np.median(imbalances)
            
        # Volumen-Metriken
        if bid_volumes and ask_volumes:
            metrics["bid_volume_mean"] = np.mean(bid_volumes)
            metrics["ask_volume_mean"] = np.mean(ask_volumes)
            metrics["volume_ratio_mean"] = np.mean([b/a if a > 0 else 1.0 for b, a in zip(bid_volumes, ask_volumes)])
            
        # Konzentration-Metriken
        if bid_concentrations and ask_concentrations:
            metrics["bid_concentration_mean"] = np.mean(bid_concentrations)
            metrics["ask_concentration_mean"] = np.mean(ask_concentrations)
        
        return metrics
    
    def _calculate_metrics_distance(self, real_metrics, synthetic_metrics):
        """
        Berechnet den Abstand zwischen realen und synthetischen Metriken.
        
        Args:
            real_metrics: Metriken aus echten Daten
            synthetic_metrics: Metriken aus synthetischen Daten
        
        Returns:
            float: Abstandsmaß
        """
        # Gemeinsame Metriken finden
        common_keys = set(real_metrics.keys()) & set(synthetic_metrics.keys())
        
        if not common_keys:
            return float('inf')
        
        # Normalisierte Abstandsberechnung
        distances = []
        
        for key in common_keys:
            # Normalisierter Wert für diesen Schlüssel
            real_value = real_metrics[key]
            synth_value = synthetic_metrics[key]
            
            # Vermeide Division durch Null
            if abs(real_value) < 1e-10:
                real_value = 1e-10
            
            # Normalisierter relativer Abstand
            rel_distance = abs(real_value - synth_value) / abs(real_value)
            distances.append(min(rel_distance, 1.0))  # Begrenze auf max. 1.0
        
        # Durchschnittlicher Abstand
        return np.mean(distances) if distances else float('inf')
    
    def _calculate_parameter_adjustments(self, real_metrics, synthetic_metrics, learning_rate):
        """
        Berechnet Parameteranpassungen basierend auf Metrik-Unterschieden.
        
        Args:
            real_metrics: Metriken aus echten Daten
            synthetic_metrics: Metriken aus synthetischen Daten
            learning_rate: Lernrate für Anpassungen
        
        Returns:
            dict: Parameteranpassungen
        """
        adjustments = {}
        
        # Spread-Anpassung
        if "spread_mean" in real_metrics and "spread_mean" in synthetic_metrics:
            real_spread = real_metrics["spread_mean"]
            synth_spread = synthetic_metrics["spread_mean"]
            
            # Verhältnis berechnen: real/synthetisch
            ratio = real_spread / synth_spread if synth_spread > 0 else 1.0
            
            # Begrenze Anpassung
            ratio = max(0.5, min(2.0, ratio))
            
            # Berechne Anpassungsfaktor
            adjustment = 1.0 + (ratio - 1.0) * learning_rate
            
            adjustments["spread_mean"] = adjustment
            adjustments["spread_std"] = adjustment
        
        # Imbalance-Anpassung
        if "imbalance_mean" in real_metrics and "imbalance_mean" in synthetic_metrics:
            real_imb = real_metrics["imbalance_mean"]
            synth_imb = synthetic_metrics["imbalance_mean"]
            
            ratio = real_imb / synth_imb if synth_imb > 0 else 1.0
            ratio = max(0.5, min(2.0, ratio))
            
            adjustment = 1.0 + (ratio - 1.0) * learning_rate
            
            # Imbalance-Bereichsanpassung
            imb_min, imb_max = self.params["imbalance_range"]
            
            if ratio > 1.0:  # Real > Synthetic
                adjustments["imbalance_max"] = adjustment  # Erhöhe das Maximum
            else:  # Real < Synthetic
                adjustments["imbalance_min"] = adjustment  # Verringere das Minimum
        
        # Volumenanpassungen
        if "bid_volume_mean" in real_metrics and "bid_volume_mean" in synthetic_metrics:
            real_vol = real_metrics["bid_volume_mean"]
            synth_vol = synthetic_metrics["bid_volume_mean"]
            
            ratio = real_vol / synth_vol if synth_vol > 0 else 1.0
            ratio = max(0.5, min(2.0, ratio))
            
            adjustment = 1.0 + (ratio - 1.0) * learning_rate
            
            adjustments["volume_mean"] = adjustment
        
        # Volumenkonzentration
        if "bid_concentration_mean" in real_metrics and "bid_concentration_mean" in synthetic_metrics:
            real_conc = real_metrics["bid_concentration_mean"]
            synth_conc = synthetic_metrics["bid_concentration_mean"]
            
            ratio = real_conc / synth_conc if synth_conc > 0 else 1.0
            ratio = max(0.5, min(2.0, ratio))
            
            adjustment = 1.0 + (ratio - 1.0) * learning_rate
            
            adjustments["volume_decay"] = 1.0 / adjustment  # Inverser Effekt
        
        return adjustments
    
    def _apply_parameter_adjustments(self, adjustments):
        """
        Wendet die berechneten Anpassungen auf die Generatorparameter an.
        
        Args:
            adjustments: Parameteranpassungen
        """
        # Spread-Anpassungen
        if "spread_mean" in adjustments:
            self.params["spread_mean"] *= adjustments["spread_mean"]
            self.logger.info(f"Spread angepasst: {self.params['spread_mean']:.4f} Pips")
        
        if "spread_std" in adjustments:
            self.params["spread_std"] *= adjustments["spread_std"]
        
        # Imbalance-Anpassungen
        if "imbalance_min" in adjustments:
            imb_min, imb_max = self.params["imbalance_range"]
            self.params["imbalance_range"] = (imb_min * adjustments["imbalance_min"], imb_max)
        
        if "imbalance_max" in adjustments:
            imb_min, imb_max = self.params["imbalance_range"]
            self.params["imbalance_range"] = (imb_min, imb_max * adjustments["imbalance_max"])
            
        self.logger.info(f"Imbalance-Bereich angepasst: {self.params['imbalance_range']}")
        
        # Volumen-Anpassungen
        if "volume_mean" in adjustments:
            self.params["volume_mean"] *= adjustments["volume_mean"]
            self.logger.info(f"Volumen angepasst: {self.params['volume_mean']:.4f}")
        
        # Volumenverteilung
        if "volume_decay" in adjustments:
            self.params["volume_decay"] *= adjustments["volume_decay"]
            self.logger.info(f"Volumenverteilung angepasst: Decay={self.params['volume_decay']:.4f}")
    
    def train_adapter_model(self, real_data, synthetic_data, n_features=10, debug=False):
        """
        Trainiert ein Adapter-Modell, um synthetische Daten an echte anzupassen.
        Dieser Ansatz verwendet eine Art Domain-Adaptation.
        
        Args:
            real_data: Liste von echten Order Books
            synthetic_data: Liste von synthetischen Order Books
            n_features: Anzahl der Features für das Adapter-Modell
            debug: Debug-Ausgaben aktivieren
        
        Returns:
            object: Trainiertes Adapter-Modell
        """
        if len(real_data) < 10 or len(synthetic_data) < 10:
            self.logger.warning("Nicht genug Daten für Adapter-Training.")
            return None
        
        # Features aus Order Books extrahieren
        real_features = self._extract_adapter_features(real_data)
        synth_features = self._extract_adapter_features(synthetic_data)
        
        if debug:
            self.logger.info(f"Extrahierte Features: Real {real_features.shape}, Synth {synth_features.shape}")
        
        # Scaler für Feature-Normalisierung
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_features)
        synth_scaled = scaler.transform(synth_features)
        
        # Labels erstellen: 0 für synthetisch, 1 für echt
        real_labels = np.ones(len(real_features))
        synth_labels = np.zeros(len(synth_features))
        
        # Kombiniere Daten für Training
        X = np.vstack([real_scaled, synth_scaled])
        y = np.hstack([real_labels, synth_labels])
        
        # Trainiere Gaussian Mixture Model für die Verteilungsschätzung
        n_components = min(5, len(real_scaled) // 10)
        
        # GMM für echte Daten
        real_gmm = GaussianMixture(n_components=n_components, random_state=42)
        real_gmm.fit(real_scaled)
        
        # GMM für synthetische Daten
        synth_gmm = GaussianMixture(n_components=n_components, random_state=42)
        synth_gmm.fit(synth_scaled)
        
        # Adapter-Modell-Konfiguration
        adapter_model = {
            "scaler": scaler,
            "real_gmm": real_gmm,
            "synth_gmm": synth_gmm,
            "feature_stats": {
                "real_mean": np.mean(real_features, axis=0),
                "real_std": np.std(real_features, axis=0),
                "synth_mean": np.mean(synth_features, axis=0),
                "synth_std": np.std(synth_features, axis=0)
            }
        }
        
        if debug:
            # Evaluiere die Anpassungsfähigkeit
            real_log_probs = real_gmm.score_samples(real_scaled)
            synth_log_probs = synth_gmm.score_samples(synth_scaled)
            
            adapted_synth = self._adapt_features(synth_scaled, adapter_model)
            adapted_log_probs = real_gmm.score_samples(adapted_synth)
            
            self.logger.info("Log-Probability Statistics:")
            self.logger.info(f"Real in Real GMM: {np.mean(real_log_probs):.4f} ± {np.std(real_log_probs):.4f}")
            self.logger.info(f"Synth in Synth GMM: {np.mean(synth_log_probs):.4f} ± {np.std(synth_log_probs):.4f}")
            self.logger.info(f"Adapted Synth in Real GMM: {np.mean(adapted_log_probs):.4f} ± {np.std(adapted_log_probs):.4f}")
        
        return adapter_model
    
    def _extract_adapter_features(self, books):
        """
        Extrahiert Features für das Adapter-Modell.
        
        Args:
            books: Liste von Order Books
            
        Returns:
            np.array: Feature-Matrix
        """
        if not books:
            return np.array([])
        
        features = []
        
        for book in books:
            if "bids" not in book or "asks" not in book:
                continue
                
            bids = book["bids"]
            asks = book["asks"]
            
            if not bids or not asks:
                continue
            
            # Feature-Vektor für dieses Book
            feat = []
            
            # 1. Spread
            spread = 0
            if "spread" in book:
                spread = book["spread"]
            elif len(bids) > 0 and len(asks) > 0:
                spread = (asks[0]["price"] - bids[0]["price"]) / self.pip_size
            feat.append(spread)
            
            # 2. Imbalance
            imbalance = 1.0
            if "imbalance" in book:
                imbalance = book["imbalance"]
            else:
                total_bid_vol = sum(b["volume"] for b in bids)
                total_ask_vol = sum(a["volume"] for a in asks)
                
                if total_ask_vol > 0:
                    imbalance = total_bid_vol / total_ask_vol
            feat.append(imbalance)
            
            # 3. Volumen-Metriken
            total_bid_vol = sum(b["volume"] for b in bids)
            total_ask_vol = sum(a["volume"] for a in asks)
            
            feat.append(total_bid_vol)
            feat.append(total_ask_vol)
            
            # 4. Volumenkonzentration
            bid_volumes = [b["volume"] for b in bids]
            ask_volumes = [a["volume"] for a in asks]
            
            bid_gini = self._gini_coefficient(bid_volumes)
            ask_gini = self._gini_coefficient(ask_volumes)
            
            feat.append(bid_gini)
            feat.append(ask_gini)
            
            # 5. Preisabstände
            if len(bids) > 1:
                price_gaps = [abs(bids[i+1]["price"] - bids[i]["price"]) / self.pip_size 
                             for i in range(min(3, len(bids)-1))]
                feat.append(np.mean(price_gaps) if price_gaps else 0)
            else:
                feat.append(0)
                
            if len(asks) > 1:
                price_gaps = [abs(asks[i+1]["price"] - asks[i]["price"]) / self.pip_size 
                             for i in range(min(3, len(asks)-1))]
                feat.append(np.mean(price_gaps) if price_gaps else 0)
            else:
                feat.append(0)
            
            # 6. Level-Verhältnis (top 3 Levels)
            if len(bids) >= 3:
                top_bid_vol = sum(b["volume"] for b in bids[:3])
                feat.append(top_bid_vol / total_bid_vol if total_bid_vol > 0 else 0)
            else:
                feat.append(1.0)  # Alle Volumina in den vorhandenen Levels
                
            if len(asks) >= 3:
                top_ask_vol = sum(a["volume"] for a in asks[:3])
                feat.append(top_ask_vol / total_ask_vol if total_ask_vol > 0 else 0)
            else:
                feat.append(1.0)
            
            features.append(feat)
        
        return np.array(features)
    
    def _adapt_features(self, synth_features, adapter_model):
        """
        Passt synthetische Features an die Verteilung echter Features an.
        
        Args:
            synth_features: Synthetische Features
            adapter_model: Trainiertes Adapter-Modell
            
        Returns:
            np.array: Angepasste Features
        """
        # Extrahiere Komponenten aus dem Modell
        real_gmm = adapter_model["real_gmm"]
        synth_gmm = adapter_model["synth_gmm"]
        
        # Feature-Statistiken
        real_mean = adapter_model["feature_stats"]["real_mean"]
        real_std = adapter_model["feature_stats"]["real_std"]
        synth_mean = adapter_model["feature_stats"]["synth_mean"]
        synth_std = adapter_model["feature_stats"]["synth_std"]
        
        # Generiere angepasste Features
        adapted_features = synth_features.copy()
        
        # Whitening-Transformation: Normalisiere auf Synth-Verteilung
        normalized = (synth_features - synth_mean) / synth_std
        
        # Coloring-Transformation: Transformiere zur Real-Verteilung
        adapted_features = normalized * real_std + real_mean
        
        return adapted_features
    
    def adapt_order_book(self, synth_book, adapter_model):
        """
        Passt ein synthetisches Order Book an die Verteilung echter Bücher an.
        
        Args:
            synth_book: Synthetisches Order Book
            adapter_model: Trainiertes Adapter-Modell
            
        Returns:
            dict: Angepasstes Order Book
        """
        if adapter_model is None:
            return synth_book
        
        # Extrahiere Features vom synthetischen Book
        synth_features = self._extract_adapter_features([synth_book])
        
        if len(synth_features) == 0:
            return synth_book
        
        # Skalieren
        scaler = adapter_model["scaler"]
        synth_scaled = scaler.transform(synth_features)
        
        # Anpassen
        adapted_scaled = self._adapt_features(synth_scaled, adapter_model)
        
        # Zurücktransformieren
        adapted_features = scaler.inverse_transform(adapted_scaled)[0]
        
        # Erstelle angepasstes Order Book
        adapted_book = synth_book.copy()
        
        # Anpassen der Haupteigenschaften
        # 1. Spread anpassen
        spread_factor = adapted_features[0] / synth_features[0][0] if synth_features[0][0] > 0 else 1.0
        spread_factor = max(0.5, min(2.0, spread_factor))  # Begrenzen
        
        # Passe Preise an
        if "bids" in adapted_book and "asks" in adapted_book:
            mid_price = (adapted_book["bids"][0]["price"] + adapted_book["asks"][0]["price"]) / 2
            new_spread = adapted_book["spread"] * spread_factor if "spread" in adapted_book else \
                         (adapted_book["asks"][0]["price"] - adapted_book["bids"][0]["price"]) * spread_factor
            
            # Update Preise
            for i, bid in enumerate(adapted_book["bids"]):
                # Angepasster Abstand vom Mittelpunkt
                distance = mid_price - bid["price"]
                adapted_book["bids"][i]["price"] = mid_price - distance * spread_factor
            
            for i, ask in enumerate(adapted_book["asks"]):
                # Angepasster Abstand vom Mittelpunkt
                distance = ask["price"] - mid_price
                adapted_book["asks"][i]["price"] = mid_price + distance * spread_factor
        
        # 2. Imbalance anpassen
        imbalance_factor = adapted_features[1] / synth_features[0][1] if synth_features[0][1] > 0 else 1.0
        imbalance_factor = max(0.5, min(2.0, imbalance_factor))
        
        # Passe Imbalance an, indem Bid-Volumen angepasst wird
        if "bids" in adapted_book:
            for i, bid in enumerate(adapted_book["bids"]):
                adapted_book["bids"][i]["volume"] *= imbalance_factor
        
        # 3. Konzentrationsanpassung
        concentration_factor = adapted_features[4] / synth_features[0][4] if synth_features[0][4] > 0 else 1.0
        
        if concentration_factor > 1.0:
            # Erhöhe Konzentration: verstärke Volumen an der Spitze, reduziere am Ende
            for i, bid in enumerate(adapted_book["bids"]):
                # Faktor, der abnehmend über die Levels wirkt
                level_factor = 1.0 + (concentration_factor - 1.0) * (1.0 - i / len(adapted_book["bids"]))
                adapted_book["bids"][i]["volume"] *= level_factor
            
            for i, ask in enumerate(adapted_book["asks"]):
                level_factor = 1.0 + (concentration_factor - 1.0) * (1.0 - i / len(adapted_book["asks"]))
                adapted_book["asks"][i]["volume"] *= level_factor
        else:
            # Verringere Konzentration: reduziere Volumen an der Spitze, erhöhe am Ende
            for i, bid in enumerate(adapted_book["bids"]):
                level_factor = 1.0 + (1.0 - concentration_factor) * (i / len(adapted_book["bids"]))
                adapted_book["bids"][i]["volume"] *= level_factor
            
            for i, ask in enumerate(adapted_book["asks"]):
                level_factor = 1.0 + (1.0 - concentration_factor) * (i / len(adapted_book["asks"]))
                adapted_book["asks"][i]["volume"] *= level_factor
        
        return adapted_book
    
    def save_parameters(self, filepath="synthetic_orderbook_params.json"):
        """
        Speichert die aktuellen Generierungsparameter.
        
        Args:
            filepath: Pfad zum Speichern
            
        Returns:
            bool: Erfolg
        """
        try:
            # Erstelle Kopie der Parameter ohne komplexe Objekte
            save_params = self.params.copy()
            
            # In JSON speichern
            with open(filepath, 'w') as f:
                json.dump(save_params, f, indent=2)
            
            self.logger.info(f"Parameter gespeichert in {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Parameter: {str(e)}")
            return False
    
    def load_parameters(self, filepath="synthetic_orderbook_params.json"):
        """
        Lädt Generierungsparameter aus einer Datei.
        
        Args:
            filepath: Pfad zur Parameterdatei
            
        Returns:
            bool: Erfolg
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"Parameterdatei {filepath} nicht gefunden.")
            return False
        
        try:
            with open(filepath, 'r') as f:
                loaded_params = json.load(f)
            
            # Parameter aktualisieren
            self.params.update(loaded_params)
            
            self.logger.info(f"Parameter geladen aus {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Parameter: {str(e)}")
            return False
    
    def _gini_coefficient(self, values):
        """
        Berechnet den Gini-Koeffizient für eine Liste von Werten.
        Dies ist ein Maß für die Ungleichheit der Verteilung (0=völlig gleich, 1=maximale Ungleichheit).
        
        Args:
            values: Liste von numerischen Werten
        
        Returns:
            float: Gini-Koeffizient
        """
        if not values or sum(values) == 0:
            return 0
        
        # Sortiere Werte
        values = sorted(values)
        n = len(values)
        
        # Berechne kumulierte Summe
        cumsum = 0
        for i, value in enumerate(values):
            cumsum += (i + 1) * value
        
        # Gini-Formel
        return (2 * cumsum) / (n * sum(values)) - (n + 1) / n
    
    def _detect_walls(self, orders, threshold=3.0):
        """
        Erkennt "Wände" im Order Book (große Orders im Vergleich zu benachbarten).
        
        Args:
            orders: Liste von Order-Dictionaries mit price und volume
            threshold: Schwellenwert, ab dem ein Orders als Wand gilt
            
        Returns:
            list: Erkannte Wände mit Preis und Volumen
        """
        if not orders or len(orders) < 3:
            return []
        
        # Extrahiere Volumina
        volumes = [order["volume"] for order in orders]
        avg_volume = sum(volumes) / len(volumes)
        
        # Finde Wände
        walls = []
        
        for i, order in enumerate(orders):
            # Vergleiche mit Durchschnitt
            if order["volume"] > threshold * avg_volume:
                # Vergleiche mit benachbarten Orders (falls vorhanden)
                is_wall = True
                
                if i > 0 and order["volume"] < orders[i-1]["volume"] * 1.5:
                    is_wall = False
                
                if i < len(orders) - 1 and order["volume"] < orders[i+1]["volume"] * 1.5:
                    is_wall = False
                
                if is_wall:
                    walls.append({
                        "price": order["price"],
                        "volume": order["volume"],
                        "level": i
                    })
        
        return walls

class OrderBookFeatureProcessor:
    """
    Verarbeitet und transformiert Order Book Features für die Verwendung
    in Modellen. Implementiert speziell Feature-Normalisierung und -Transformation
    für synthetische und echte Daten.
    """
    def __init__(self, pip_size=0.01):
        """
        Initialisiert den Feature-Processor.
        
        Args:
            pip_size: Größe eines Pips für das Handelssymbol
        """
        self.pip_size = pip_size
        self.scalers = {}
        self.feature_stats = {}
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('ob_feature_processor')
    
    def extract_features(self, order_book, include_raw=False):
        """
        Extrahiert Standard-Features aus einem Order Book.
        
        Args:
            order_book: Order Book Daten
            include_raw: Auch Roh-Features zurückgeben
            
        Returns:
            dict: Extrahierte Features
        """
        if not order_book or "bids" not in order_book or "asks" not in order_book:
            return {}
        
        bids = order_book["bids"]
        asks = order_book["asks"]
        
        if not bids or not asks:
            return {}
        
        # 1. Basis Metriken
        
        # Spread
        spread = asks[0]["price"] - bids[0]["price"]
        spread_pips = spread / self.pip_size
        
        # Mid Price
        mid_price = (bids[0]["price"] + asks[0]["price"]) / 2
        
        # Volumen
        total_bid_vol = sum(b["volume"] for b in bids)
        total_ask_vol = sum(a["volume"] for a in asks)
        
        # Imbalance
        imbalance = total_bid_vol / total_ask_vol if total_ask_vol > 0 else 1.0
        
        # 2. Volumen-Verteilung
        
        # Top Level Konzentration
        top_bid_vol = sum(b["volume"] for b in bids[:min(3, len(bids))])
        top_ask_vol = sum(a["volume"] for a in asks[:min(3, len(asks))])
        
        top_bid_ratio = top_bid_vol / total_bid_vol if total_bid_vol > 0 else 0
        top_ask_ratio = top_ask_vol / total_ask_vol if total_ask_vol > 0 else 0
        
        # Volumen Konzentration (Gini)
        bid_volumes = [b["volume"] for b in bids]
        ask_volumes = [a["volume"] for a in asks]
        
        bid_gini = self._gini_coefficient(bid_volumes)
        ask_gini = self._gini_coefficient(ask_volumes)
        
        # 3. Preis-Level Abstände
        
        bid_price_gaps = []
        for i in range(len(bids) - 1):
            gap = (bids[i]["price"] - bids[i+1]["price"]) / self.pip_size
            bid_price_gaps.append(gap)
        
        ask_price_gaps = []
        for i in range(len(asks) - 1):
            gap = (asks[i+1]["price"] - asks[i]["price"]) / self.pip_size
            ask_price_gaps.append(gap)
        
        avg_bid_gap = np.mean(bid_price_gaps) if bid_price_gaps else 0
        avg_ask_gap = np.mean(ask_price_gaps) if ask_price_gaps else 0
        
        # 4. Institutionelle Aktivität
        
        # Erkenne Wände
        bid_walls = self._detect_walls(bids, threshold=3.0)
        ask_walls = self._detect_walls(asks, threshold=3.0)
        
        # 5. Feature Dictionary
        features = {
            # Verhältnis-basierte Features (robust)
            "spread_pips": spread_pips,
            "imbalance": imbalance,
            "bid_concentration": top_bid_ratio,
            "ask_concentration": top_ask_ratio,
            "bid_gini": bid_gini,
            "ask_gini": ask_gini,
            "avg_bid_gap": avg_bid_gap,
            "avg_ask_gap": avg_ask_gap,
            "bid_wall_count": len(bid_walls),
            "ask_wall_count": len(ask_walls),
            
            # Level-spezifische Verhältnisse
            "bid_level_1_ratio": bids[0]["volume"] / total_bid_vol if total_bid_vol > 0 else 0,
            "bid_level_2_ratio": bids[1]["volume"] / total_bid_vol if len(bids) > 1 and total_bid_vol > 0 else 0,
            "bid_level_3_ratio": bids[2]["volume"] / total_bid_vol if len(bids) > 2 and total_bid_vol > 0 else 0,
            "ask_level_1_ratio": asks[0]["volume"] / total_ask_vol if total_ask_vol > 0 else 0,
            "ask_level_2_ratio": asks[1]["volume"] / total_ask_vol if len(asks) > 1 and total_ask_vol > 0 else 0,
            "ask_level_3_ratio": asks[2]["volume"] / total_ask_vol if len(asks) > 2 and total_ask_vol > 0 else 0,
        }
        
        # Füge Rohwerte hinzu, wenn gewünscht
        if include_raw:
            features.update({
                "mid_price": mid_price,
                "total_bid_volume": total_bid_vol,
                "total_ask_volume": total_ask_vol,
                "bid_walls": bid_walls,
                "ask_walls": ask_walls
            })
        
        return features
    
    def compute_relative_features(self, order_book):
        """
        Berechnet relative Features, die robust gegenüber Verteilungsunterschieden sind.
        
        Args:
            order_book: Order Book Daten
            
        Returns:
            dict: Relative Features
        """
        # Basisfeatures extrahieren
        features = self.extract_features(order_book)
        
        # Transformierte Features
        rel_features = {}
        
        # 1. Normalisierte Imbalance
        imb = features.get("imbalance", 1.0)
        # Logarithmische Transformation für symmetrischere Verteilung um 0
        rel_features["log_imbalance"] = np.log(imb) if imb > 0 else 0
        
        # 2. Relative Konzentration (Bid vs Ask)
        b_conc = features.get("bid_concentration", 0)
        a_conc = features.get("ask_concentration", 0)
        
        if a_conc > 0:
            rel_features["concentration_ratio"] = b_conc / a_conc
        else:
            rel_features["concentration_ratio"] = 1.0
            
        # 3. Gini-Differenz
        rel_features["gini_diff"] = features.get("bid_gini", 0) - features.get("ask_gini", 0)
        
        # 4. Gap-Verhältnis
        b_gap = features.get("avg_bid_gap", 0)
        a_gap = features.get("avg_ask_gap", 0)
        
        if a_gap > 0:
            rel_features["gap_ratio"] = b_gap / a_gap
        else:
            rel_features["gap_ratio"] = 1.0
        
        # 5. Wall-Imbalance
        b_walls = features.get("bid_wall_count", 0)
        a_walls = features.get("ask_wall_count", 0)
        
        # +1 für numerische Stabilität
        rel_features["wall_imbalance"] = (b_walls + 1) / (a_walls + 1)
        
        # 6. Level-Ratio Aggregationen
        
        # Bid Level Verhältnis (L1/avg(L2,L3))
        l1 = features.get("bid_level_1_ratio", 0)
        l2 = features.get("bid_level_2_ratio", 0)
        l3 = features.get("bid_level_3_ratio", 0)
        
        if (l2 + l3) > 0:
            rel_features["bid_top_ratio"] = l1 / ((l2 + l3) / 2)
        else:
            rel_features["bid_top_ratio"] = 1.0
        
        # Ask Level Verhältnis
        l1 = features.get("ask_level_1_ratio", 0)
        l2 = features.get("ask_level_2_ratio", 0)
        l3 = features.get("ask_level_3_ratio", 0)
        
        if (l2 + l3) > 0:
            rel_features["ask_top_ratio"] = l1 / ((l2 + l3) / 2)
        else:
            rel_features["ask_top_ratio"] = 1.0
        
        # 7. Spread Normalisierung
        # Normalisieren auf durchschnittlichen Gap
        avg_gap = (features.get("avg_bid_gap", 0) + features.get("avg_ask_gap", 0)) / 2
        if avg_gap > 0:
            rel_features["relative_spread"] = features.get("spread_pips", 0) / avg_gap
        else:
            rel_features["relative_spread"] = features.get("spread_pips", 0)
        
        return rel_features
    
    def fit_scalers(self, real_books, synth_books=None):
        """
        Passt Scaler für Feature-Transformation an.
        
        Args:
            real_books: Liste von echten Order Books
            synth_books: Optional, Liste von synthetischen Order Books
        """
        # Extrahiere Features
        real_features = []
        for book in real_books:
            features = self.extract_features(book)
            if features:
                real_features.append(features)
        
        if not real_features:
            self.logger.warning("Keine gültigen echten Order Books für Scaler-Anpassung.")
            return
        
        # Erstelle DataFrame
        real_df = pd.DataFrame(real_features)
        
        # Statistiken für echte Daten
        self.feature_stats["real"] = {
            "mean": real_df.mean().to_dict(),
            "std": real_df.std().to_dict(),
            "median": real_df.median().to_dict(),
            "min": real_df.min().to_dict(),
            "max": real_df.max().to_dict()
        }
        
        # Fit Scaler für echte Daten
        self.scalers["real"] = {}
        for col in real_df.columns:
            self.scalers["real"][col] = StandardScaler()
            self.scalers["real"][col].fit(real_df[[col]])
        
        # Wenn synthetische Daten vorhanden sind
        if synth_books:
            # Extrahiere Features
            synth_features = []
            for book in synth_books:
                features = self.extract_features(book)
                if features:
                    synth_features.append(features)
            
            if synth_features:
                # Erstelle DataFrame
                synth_df = pd.DataFrame(synth_features)
                
                # Statistiken für synthetische Daten
                self.feature_stats["synth"] = {
                    "mean": synth_df.mean().to_dict(),
                    "std": synth_df.std().to_dict(),
                    "median": synth_df.median().to_dict(),
                    "min": synth_df.min().to_dict(),
                    "max": synth_df.max().to_dict()
                }
                
                # Fit Scaler für synthetische Daten
                self.scalers["synth"] = {}
                for col in synth_df.columns:
                    if col in real_df.columns:
                        self.scalers["synth"][col] = StandardScaler()
                        self.scalers["synth"][col].fit(synth_df[[col]])
        
        self.logger.info(f"Scaler für {len(self.scalers.get('real', {}))} Features angepasst")
    
    def transform_features(self, features, source_type="synth", target_type="real"):
        """
        Transformiert Features von einem Verteilungstyp zu einem anderen.
        
        Args:
            features: Feature-Dictionary oder Liste von Dictionaries
            source_type: Quelltyp der Features ('synth' oder 'real')
            target_type: Zieltyp der Transformation ('synth' oder 'real')
            
        Returns:
            dict: Transformierte Features
        """
        if source_type == target_type:
            return features  # Keine Transformation nötig
        
        if not self.scalers or source_type not in self.scalers or target_type not in self.scalers:
            self.logger.warning(f"Scaler nicht verfügbar für {source_type}->{target_type} Transformation")
            return features
        
        # Liste von Dictionaries oder einzelnes Dictionary
        is_list = isinstance(features, list)
        features_list = features if is_list else [features]
        
        # Transformierte Features
        transformed_list = []
        
        for feat_dict in features_list:
            transformed = {}
            
            for key, value in feat_dict.items():
                if key in self.scalers[source_type] and key in self.scalers[target_type]:
                    # Zuerst auf Quellverteilung normalisieren
                    source_scaler = self.scalers[source_type][key]
                    target_scaler = self.scalers[target_type][key]
                    
                    # Transformiere zu Standardnormalverteilung
                    value_standardized = source_scaler.transform([[value]])[0][0]
                    
                    # Transformiere zur Zielverteilung
                    transformed[key] = target_scaler.inverse_transform([[value_standardized]])[0][0]
                else:
                    # Behalte Originalwert bei, wenn keine Scaler verfügbar
                    transformed[key] = value
            
            transformed_list.append(transformed)
        
        return transformed_list if is_list else transformed_list[0]
    
    def _gini_coefficient(self, values):
        """
        Berechnet den Gini-Koeffizient für eine Liste von Werten.
        Dies ist ein Maß für die Ungleichheit der Verteilung (0=völlig gleich, 1=maximale Ungleichheit).
        
        Args:
            values: Liste von numerischen Werten
        
        Returns:
            float: Gini-Koeffizient
        """
        if not values or sum(values) == 0:
            return 0
        
        # Sortiere Werte
        values = sorted(values)
        n = len(values)
        
        # Berechne kumulierte Summe
        cumsum = 0
        for i, value in enumerate(values):
            cumsum += (i + 1) * value
        
        # Gini-Formel
        return (2 * cumsum) / (n * sum(values)) - (n + 1) / n
    
    def _detect_walls(self, orders, threshold=3.0):
        """
        Erkennt "Wände" im Order Book (große Orders im Vergleich zu benachbarten).
        
        Args:
            orders: Liste von Order-Dictionaries mit price und volume
            threshold: Schwellenwert, ab dem ein Orders als Wand gilt
            
        Returns:
            list: Erkannte Wände mit Preis und Volumen
        """
        if not orders or len(orders) < 3:
            return []
        
        # Extrahiere Volumina
        volumes = [order["volume"] for order in orders]
        avg_volume = sum(volumes) / len(volumes)
        
        # Finde Wände
        walls = []
        
        for i, order in enumerate(orders):
            # Vergleiche mit Durchschnitt
            if order["volume"] > threshold * avg_volume:
                # Vergleiche mit benachbarten Orders (falls vorhanden)
                is_wall = True
                
                if i > 0 and order["volume"] < orders[i-1]["volume"] * 1.5:
                    is_wall = False
                
                if i < len(orders) - 1 and order["volume"] < orders[i+1]["volume"] * 1.5:
                    is_wall = False
                
                if is_wall:
                    walls.append({
                        "price": order["price"],
                        "volume": order["volume"],
                        "level": i
                    })
        
        return walls