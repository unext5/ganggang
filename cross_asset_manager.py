import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import time
import logging
import os
import pickle
import json
from collections import deque
import threading
import math

class CrossAssetManager:
    """
    Verwaltet die Integration von Cross-Asset-Daten für das Trading.
    
    Diese Klasse übernimmt:
    1. Abrufen und Speichern von Daten für verwandte Assets
    2. Berechnung von Korrelationen und Lead-Lag-Beziehungen
    3. Extraktion von Cross-Asset-Features für das Hauptsymbol
    4. Kontinuierliche Aktualisierung der Daten und Beziehungen
    """
    def __init__(self, main_symbol="GBPJPY", cross_assets=None, 
                timeframes=None, correlation_window=100, 
                lead_lag_max=10, update_correlation_interval=24,
                data_dir="data/cross_assets"):
        """
        Initialisiert den Cross-Asset-Manager.
        
        Args:
            main_symbol: Haupthandelssymbol
            cross_assets: Liste von Cross-Assets (oder None für automatische Auswahl)
            timeframes: Liste von Timeframes für Daten (oder None für Standard)
            correlation_window: Fenstergröße für Korrelationsberechnung
            lead_lag_max: Maximale Lead/Lag-Verschiebung in Bars
            update_correlation_interval: Intervall für Korrelationsupdates in Stunden
            data_dir: Verzeichnis zum Speichern der Daten
        """
        self.main_symbol = main_symbol
        
        # Standardliste von Cross-Assets basierend auf Hauptsymbol
        if cross_assets is None:
            self.cross_assets = self._get_default_cross_assets(main_symbol)
        else:
            self.cross_assets = cross_assets
        
        # Timeframes für Daten
        if timeframes is None:
            self.timeframes = [
                mt5.TIMEFRAME_M30,  # 30 Minuten
                mt5.TIMEFRAME_H1,   # 1 Stunde
                mt5.TIMEFRAME_H4    # 4 Stunden
            ]
        else:
            self.timeframes = timeframes
        
        # Konfiguration
        self.correlation_window = correlation_window
        self.lead_lag_max = lead_lag_max
        self.update_correlation_interval = update_correlation_interval * 3600  # in Sekunden
        self.data_dir = data_dir
        
        # Daten-Cache
        self.data_cache = {}  # {symbol: {timeframe: pd.DataFrame}}
        self.last_updates = {}  # {symbol: {timeframe: datetime}}
        
        # Korrelationsmatrix und Lead-Lag-Beziehungen
        self.correlations = {}  # {symbol: {timeframe: correlation}}
        self.lead_lag = {}  # {symbol: {timeframe: lead_lag_bars}}
        self.correlation_weights = {}  # {symbol: gewicht basierend auf Korrelationsstärke}
        
        # Cross-Asset-Features
        self.current_features = {}  # Aktuelle Features für jeden Zeitpunkt
        self.feature_history = deque(maxlen=500)  # Historie für kontinuierliches Lernen
        
        # Dynamische Symbolauswahl
        self.active_symbols = set()  # Aktive Symbole nach Korrelation
        self.min_correlation = 0.2  # Minimale Korrelation für aktive Symbole
        
        # Status und Wartung
        self.last_correlation_update = datetime.now() - timedelta(hours=update_correlation_interval)
        self.is_initialized = False
        
        # Threading
        self.lock = threading.RLock()
        self.update_thread = None
        self.stop_thread = False
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('cross_asset_manager')
        
        # Verzeichnis erstellen falls nicht vorhanden
        os.makedirs(self.data_dir, exist_ok=True)
    
    def initialize(self, lookback_days=30, force_refresh=False, auto_select=True,
                  min_correlation=0.2):
        """
        Initialisiert den Manager mit historischen Daten.
        
        Args:
            lookback_days: Anzahl der Tage für historische Daten
            force_refresh: Ob Daten neu abgerufen werden sollen
            auto_select: Ob Cross-Assets automatisch ausgewählt werden sollen
            min_correlation: Minimale Korrelation für automatische Auswahl
            
        Returns:
            bool: Erfolg der Initialisierung
        """
        self.min_correlation = min_correlation
        
        # Stelle sicher, dass MetaTrader 5 verbunden ist
        if not mt5.initialize():
            self.logger.error("MetaTrader5 konnte nicht initialisiert werden")
            return False
        
        self.logger.info(f"Initialisiere Cross-Asset-Manager für {self.main_symbol}")
        
        # Lade Daten für Hauptsymbol
        main_data = {}
        
        for timeframe in self.timeframes:
            data = self._fetch_data(self.main_symbol, timeframe, lookback_days, force_refresh)
            if data is not None:
                main_data[timeframe] = data
        
        if not main_data:
            self.logger.error(f"Konnte keine Daten für Hauptsymbol {self.main_symbol} abrufen")
            return False
        
        # Lade Cross-Asset-Daten und berechne Korrelationen
        with self.lock:
            for symbol in self.cross_assets:
                self.data_cache[symbol] = {}
                self.correlations[symbol] = {}
                self.lead_lag[symbol] = {}
                
                for timeframe in self.timeframes:
                    if timeframe not in main_data:
                        continue
                    
                    data = self._fetch_data(symbol, timeframe, lookback_days, force_refresh)
                    
                    if data is not None:
                        self.data_cache[symbol][timeframe] = data
                        
                        # Berechne Korrelation und Lead-Lag
                        corr, lead_lag_bars = self._calculate_correlation_and_lead_lag(
                            main_data[timeframe], data, self.correlation_window, self.lead_lag_max
                        )
                        
                        self.correlations[symbol][timeframe] = corr
                        self.lead_lag[symbol][timeframe] = lead_lag_bars
                        
                        self.logger.info(f"Korrelation {symbol}/{self.main_symbol} ({self._timeframe_to_string(timeframe)}): "
                                      f"{corr:.3f} (Lead/Lag: {lead_lag_bars})")
        
        # Automatische Symbolauswahl basierend auf Korrelation
        if auto_select:
            self._select_active_symbols(min_correlation)
        
        # Berechne Korrelationsgewichte
        self._calculate_correlation_weights()
        
        # Markiere als initialisiert
        self.is_initialized = True
        
        # Speichere Status
        self._save_status()
        
        # Starte Update-Thread
        self._start_update_thread()
        
        self.logger.info(f"Cross-Asset-Manager initialisiert mit {len(self.active_symbols)} aktiven Assets")
        
        return True
    
    def get_cross_asset_features(self, timestamp=None):
        """
        Extrahiert Cross-Asset-Features für einen bestimmten Zeitpunkt.
        
        Args:
            timestamp: Zeitstempel für die Features (None = aktuell)
            
        Returns:
            dict: Cross-Asset-Features
        """
        if not self.is_initialized:
            self.logger.warning("Cross-Asset-Manager ist nicht initialisiert")
            return {}
        
        # Aktuelles Datum verwenden, falls kein Zeitstempel angegeben
        if timestamp is None:
            timestamp = datetime.now()
        
        # Feature-Dictionary initialisieren
        features = {}
        
        # MT5-Zeitzone
        mt5_timezone = pytz.timezone("Etc/UTC")
        
        # Konvertiere zu aware datetime
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except:
                timestamp = datetime.now()
        
        # Stelle sicher, dass der Zeitstempel ein timezone-aware datetime ist
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=mt5_timezone)
        
        with self.lock:
            # Für jedes aktive Symbol Features extrahieren
            for symbol in self.active_symbols:
                if symbol not in self.data_cache:
                    continue
                
                # Für jeden Timeframe
                for timeframe in self.timeframes:
                    if timeframe not in self.data_cache[symbol]:
                        continue
                    
                    # Hole Daten für dieses Symbol und Timeframe
                    data = self.data_cache[symbol][timeframe]
                    
                    if data is None or len(data) == 0:
                        continue
                    
                    # Finde nächsten Zeitpunkt
                    idx = self._find_nearest_timestamp(data, timestamp)
                    
                    if idx is None:
                        continue
                    
                    # Lead-Lag-Anpassung
                    lead_lag_bars = self.lead_lag.get(symbol, {}).get(timeframe, 0)
                    adjusted_idx = idx - lead_lag_bars
                    
                    if adjusted_idx < 0 or adjusted_idx >= len(data):
                        adjusted_idx = idx  # Fallback zum ursprünglichen Index
                    
                    # Extrahiere Features
                    row = data.iloc[adjusted_idx]
                    
                    # Berechne Features (normalisierte Preisbewegungen, Volatilität etc.)
                    symbol_tf = f"{symbol}_{self._timeframe_to_string(timeframe)}"
                    
                    features[f"{symbol_tf}_close_rel"] = self._calculate_relative_price(row, 'close', data)
                    features[f"{symbol_tf}_high_rel"] = self._calculate_relative_price(row, 'high', data)
                    features[f"{symbol_tf}_low_rel"] = self._calculate_relative_price(row, 'low', data)
                    
                    # Volatilität (normalisierte ATR)
                    if 'atr' in row:
                        features[f"{symbol_tf}_atr"] = row['atr']
                    elif 'high' in row and 'low' in row and adjusted_idx > 0:
                        prev_row = data.iloc[adjusted_idx - 1]
                        features[f"{symbol_tf}_atr"] = (row['high'] - row['low']) / prev_row['close'] if prev_row['close'] > 0 else 0
                    
                    # Korrelationsgewicht für dieses Symbol und Timeframe
                    corr_weight = self.correlations.get(symbol, {}).get(timeframe, 0)
                    features[f"{symbol_tf}_corr_weight"] = corr_weight
                    
                    # Volumen (falls verfügbar)
                    if 'tick_volume' in row:
                        features[f"{symbol_tf}_volume_rel"] = self._calculate_relative_volume(row, data)
            
            # Speichere Features für kontinuierliches Lernen
            if features:
                self.current_features = features.copy()
                features_entry = {
                    "timestamp": timestamp.isoformat(),
                    "features": features
                }
                self.feature_history.append(features_entry)
        
        return features
    
    def update_correlations(self, force=False):
        """
        Aktualisiert Korrelationen und Lead-Lag-Beziehungen.
        
        Args:
            force: Ob Update erzwungen werden soll
            
        Returns:
            bool: Erfolg des Updates
        """
        # Prüfe, ob Update nötig ist
        now = datetime.now()
        time_since_update = (now - self.last_correlation_update).total_seconds()
        
        if not force and time_since_update < self.update_correlation_interval:
            self.logger.debug(f"Zu früh für Korrelationsupdate, {(self.update_correlation_interval - time_since_update) / 3600:.1f}h verbleibend")
            return False
        
        self.logger.info("Aktualisiere Cross-Asset-Korrelationen")
        
        # Stelle sicher, dass MetaTrader 5 verbunden ist
        if not mt5.initialize():
            self.logger.error("MetaTrader5 konnte nicht initialisiert werden")
            return False
        
        # Lade aktuelle Daten für Hauptsymbol
        main_data = {}
        
        for timeframe in self.timeframes:
            data = self._fetch_data(self.main_symbol, timeframe, lookback_days=30, force_refresh=True)
            if data is not None:
                main_data[timeframe] = data
        
        if not main_data:
            self.logger.error(f"Konnte keine aktuellen Daten für Hauptsymbol {self.main_symbol} abrufen")
            return False
        
        # Aktualisiere Cross-Asset-Daten und Korrelationen
        with self.lock:
            for symbol in self.cross_assets:
                if symbol not in self.correlations:
                    self.correlations[symbol] = {}
                
                if symbol not in self.lead_lag:
                    self.lead_lag[symbol] = {}
                
                for timeframe in self.timeframes:
                    if timeframe not in main_data:
                        continue
                    
                    data = self._fetch_data(symbol, timeframe, lookback_days=30, force_refresh=True)
                    
                    if data is not None:
                        self.data_cache[symbol][timeframe] = data
                        
                        # Berechne Korrelation und Lead-Lag
                        corr, lead_lag_bars = self._calculate_correlation_and_lead_lag(
                            main_data[timeframe], data, self.correlation_window, self.lead_lag_max
                        )
                        
                        old_corr = self.correlations[symbol].get(timeframe, 0)
                        old_lead_lag = self.lead_lag[symbol].get(timeframe, 0)
                        
                        # Exponentiell gewichtetes gleitendes Mittel für stabile Werte
                        alpha = 0.3  # Gewichtungsfaktor für neue Werte
                        smoothed_corr = old_corr * (1 - alpha) + corr * alpha
                        
                        # Lead-Lag nur aktualisieren, wenn signifikante Korrelation
                        if abs(corr) > 0.3:
                            # Gewichtetes Mittel für Lead-Lag
                            smoothed_lead_lag = int(old_lead_lag * (1 - alpha) + lead_lag_bars * alpha)
                        else:
                            smoothed_lead_lag = old_lead_lag
                        
                        self.correlations[symbol][timeframe] = smoothed_corr
                        self.lead_lag[symbol][timeframe] = smoothed_lead_lag
                        
                        self.logger.info(f"Aktualisierte Korrelation {symbol}/{self.main_symbol} "
                                      f"({self._timeframe_to_string(timeframe)}): "
                                      f"{smoothed_corr:.3f} (Lead/Lag: {smoothed_lead_lag})")
        
        # Aktualisiere aktive Symbole
        self._select_active_symbols(self.min_correlation)
        
        # Aktualisiere Korrelationsgewichte
        self._calculate_correlation_weights()
        
        # Aktualisiere Zeitstempel
        self.last_correlation_update = now
        
        # Speichere aktualisierten Status
        self._save_status()
        
        self.logger.info(f"Korrelationsupdate abgeschlossen mit {len(self.active_symbols)} aktiven Assets")
        
        return True
    
    def add_custom_assets(self, assets):
        """
        Fügt zusätzliche Assets zur Überwachung hinzu.
        
        Args:
            assets: Liste von Assets
            
        Returns:
            bool: Erfolg der Aktion
        """
        if not assets:
            return False
        
        for asset in assets:
            if asset not in self.cross_assets:
                self.cross_assets.append(asset)
                self.logger.info(f"Asset hinzugefügt: {asset}")
        
        # Initialisiere neue Assets
        for asset in assets:
            for timeframe in self.timeframes:
                data = self._fetch_data(asset, timeframe, lookback_days=30)
                
                if data is not None and self.main_symbol in self.data_cache:
                    self.data_cache[asset] = self.data_cache.get(asset, {})
                    self.data_cache[asset][timeframe] = data
                    
                    main_data = self.data_cache[self.main_symbol].get(timeframe)
                    
                    if main_data is not None:
                        # Berechne Korrelation und Lead-Lag
                        corr, lead_lag_bars = self._calculate_correlation_and_lead_lag(
                            main_data, data, self.correlation_window, self.lead_lag_max
                        )
                        
                        self.correlations[asset] = self.correlations.get(asset, {})
                        self.lead_lag[asset] = self.lead_lag.get(asset, {})
                        
                        self.correlations[asset][timeframe] = corr
                        self.lead_lag[asset][timeframe] = lead_lag_bars
                        
                        self.logger.info(f"Korrelation {asset}/{self.main_symbol} ({self._timeframe_to_string(timeframe)}): "
                                      f"{corr:.3f} (Lead/Lag: {lead_lag_bars})")
        
        # Aktualisiere aktive Symbole
        self._select_active_symbols(self.min_correlation)
        
        # Aktualisiere Korrelationsgewichte
        self._calculate_correlation_weights()
        
        # Speichere aktualisierten Status
        self._save_status()
        
        return True
    
    def remove_assets(self, assets):
        """
        Entfernt Assets aus der Überwachung.
        
        Args:
            assets: Liste von Assets
            
        Returns:
            bool: Erfolg der Aktion
        """
        if not assets:
            return False
        
        for asset in assets:
            if asset in self.cross_assets:
                self.cross_assets.remove(asset)
                
                # Entferne aus aktiven Symbolen
                if asset in self.active_symbols:
                    self.active_symbols.remove(asset)
                
                # Entferne aus Daten-Cache
                if asset in self.data_cache:
                    del self.data_cache[asset]
                
                # Entferne aus Korrelationen
                if asset in self.correlations:
                    del self.correlations[asset]
                
                # Entferne aus Lead-Lag
                if asset in self.lead_lag:
                    del self.lead_lag[asset]
                
                # Entferne aus Korrelationsgewichten
                if asset in self.correlation_weights:
                    del self.correlation_weights[asset]
                
                self.logger.info(f"Asset entfernt: {asset}")
        
        # Aktualisiere Korrelationsgewichte
        self._calculate_correlation_weights()
        
        # Speichere aktualisierten Status
        self._save_status()
        
        return True
    
    def get_correlation_report(self, detailed=False):
        """
        Erstellt einen Bericht über Korrelationen und Lead-Lag-Beziehungen.
        
        Args:
            detailed: Ob ein detaillierter Bericht erstellt werden soll
            
        Returns:
            dict: Korrelationsbericht
        """
        if not self.is_initialized:
            return {"error": "Manager nicht initialisiert"}
        
        with self.lock:
            report = {
                "main_symbol": self.main_symbol,
                "active_symbols": list(self.active_symbols),
                "last_update": self.last_correlation_update.isoformat(),
                "correlations": {},
                "lead_lag": {},
                "weights": {}
            }
            
            # Füge Korrelationen hinzu
            for symbol in self.correlations:
                symbol_corrs = {}
                
                for timeframe in self.correlations[symbol]:
                    symbol_corrs[self._timeframe_to_string(timeframe)] = self.correlations[symbol][timeframe]
                
                report["correlations"][symbol] = symbol_corrs
            
            # Füge Lead-Lag-Werte hinzu
            for symbol in self.lead_lag:
                symbol_ll = {}
                
                for timeframe in self.lead_lag[symbol]:
                    symbol_ll[self._timeframe_to_string(timeframe)] = self.lead_lag[symbol][timeframe]
                
                report["lead_lag"][symbol] = symbol_ll
            
            # Füge Gewichte hinzu
            for symbol in self.correlation_weights:
                report["weights"][symbol] = self.correlation_weights[symbol]
            
            # Detaillierter Bericht
            if detailed:
                # Aktuelle Preisbewegungen
                report["price_movements"] = {}
                
                for symbol in self.active_symbols:
                    if symbol in self.data_cache:
                        movements = {}
                        
                        for timeframe in self.data_cache[symbol]:
                            data = self.data_cache[symbol][timeframe]
                            
                            if data is not None and len(data) >= 2:
                                last = data.iloc[-1]
                                prev = data.iloc[-2]
                                
                                pct_change = (last['close'] - prev['close']) / prev['close'] * 100 if prev['close'] > 0 else 0
                                movements[self._timeframe_to_string(timeframe)] = pct_change
                        
                        report["price_movements"][symbol] = movements
                
                # Füge aktuelle Features hinzu
                report["current_features"] = self.current_features
        
        return report
    
    def visualize_correlations(self, save_path=None):
        """
        Visualisiert Korrelationen zwischen Assets.
        
        Args:
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            bool: Erfolg der Visualisierung
        """
        if not self.is_initialized:
            self.logger.error("Manager nicht initialisiert")
            return False
        
        try:
            # Bereite Daten vor
            symbols = list(self.active_symbols)
            if not symbols:
                self.logger.warning("Keine aktiven Symbole für Visualisierung")
                return False
            
            # Durchschnittliche Korrelation pro Symbol
            avg_correlations = {}
            
            for symbol in symbols:
                corrs = [abs(self.correlations[symbol][tf]) for tf in self.correlations[symbol]]
                avg_correlations[symbol] = sum(corrs) / len(corrs) if corrs else 0
            
            # Sortiere Symbole nach Korrelationsstärke
            sorted_symbols = sorted(avg_correlations.items(), key=lambda x: x[1], reverse=True)
            sorted_symbols = [item[0] for item in sorted_symbols]
            
            # Erstelle Matrix
            correlation_matrix = np.zeros((len(sorted_symbols), len(self.timeframes)))
            
            for i, symbol in enumerate(sorted_symbols):
                for j, tf in enumerate(self.timeframes):
                    if tf in self.correlations.get(symbol, {}):
                        correlation_matrix[i, j] = self.correlations[symbol][tf]
            
            # Erstelle Plot
            plt.figure(figsize=(10, 8))
            plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Beschriftungen
            plt.colorbar(label='Korrelation')
            plt.title(f'Cross-Asset-Korrelationen mit {self.main_symbol}')
            plt.xlabel('Timeframe')
            plt.ylabel('Symbol')
            
            plt.xticks(range(len(self.timeframes)), 
                     [self._timeframe_to_string(tf) for tf in self.timeframes])
            plt.yticks(range(len(sorted_symbols)), sorted_symbols)
            
            # Füge Korrelationswerte zum Plot hinzu
            for i in range(len(sorted_symbols)):
                for j in range(len(self.timeframes)):
                    color = 'black' if abs(correlation_matrix[i, j]) < 0.5 else 'white'
                    plt.text(j, i, f"{correlation_matrix[i, j]:.2f}", 
                           ha="center", va="center", color=color)
            
            plt.tight_layout()
            
            # Speichere oder zeige Plot
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Korrelationsvisualisierung gespeichert unter {save_path}")
            else:
                plt.show()
            
            plt.close()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Fehler bei Korrelationsvisualisierung: {str(e)}")
            return False
    
    def _get_default_cross_assets(self, main_symbol):
        """
        Gibt eine Standardliste von Cross-Assets basierend auf dem Hauptsymbol zurück.
        
        Args:
            main_symbol: Haupthandelssymbol
            
        Returns:
            list: Standardliste von Cross-Assets
        """
        # Extrahiere Währungen aus dem Symbol
        currencies = []
        
        if len(main_symbol) >= 6:
            base_currency = main_symbol[:3]
            quote_currency = main_symbol[3:6]
            currencies = [base_currency, quote_currency]
        
        # Erstelle Liste von verwandten Symbolen
        related_symbols = []
        
        # Forex Majors
        forex_majors = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", 
            "AUDUSD", "NZDUSD", "USDCAD"
        ]
        
        # Indizes
        indices = {
            "USD": "US30",      # Dow Jones
            "EUR": "DE30",      # DAX
            "GBP": "UK100.cash",     # FTSE 100
            "JPY": "JP225.cash",     # Nikkei 225
            "AUD": "AUS200",    # ASX 200
            "CHF": "SWISS20",   # Swiss Market Index
            "CAD": "CAC40"      # Canada Index
        }
        
        # Commodities
        commodities = ["XAUUSD", "XAGUSD", "XBRUSD"]  # Gold, Silber, Öl
        
        # Füge verwandte Währungspaare hinzu
        for currency in currencies:
            for major in forex_majors:
                if currency in major and major != main_symbol:
                    related_symbols.append(major)
        
        # Füge verwandte Indizes hinzu
        for currency in currencies:
            if currency in indices:
                related_symbols.append(indices[currency])
        
        # Füge Commodities hinzu
        related_symbols.extend(commodities)
        
        # Entferne Duplikate
        related_symbols = list(set(related_symbols))
        
        # Symbole für spezifische Währungspaare
        if main_symbol == "EURUSD":
            specific_symbols = ["GBPUSD", "USDCHF", "EURGBP", "EURCHF", "US30", "DE30"]
            related_symbols.extend(specific_symbols)
        elif main_symbol == "GBPJPY":
            specific_symbols = ["GBPUSD", "USDJPY", "EURJPY", "GBPCHF", "UK100.cash", "JP225.cash"]
            related_symbols.extend(specific_symbols)
        
        # Entferne Duplikate und das Hauptsymbol selbst
        related_symbols = [s for s in list(set(related_symbols)) if s != main_symbol]
        
        return related_symbols
    
    def _fetch_data(self, symbol, timeframe, lookback_days=30, force_refresh=False):
        """
        Holt historische Daten für ein Symbol und Timeframe.
        
        Args:
            symbol: Handelssymbol
            timeframe: MT5-Timeframe (z.B. mt5.TIMEFRAME_H1)
            lookback_days: Anzahl der Tage für historische Daten
            force_refresh: Ob Daten neu abgerufen werden sollen
            
        Returns:
            pd.DataFrame: Historische Daten oder None bei Fehler
        """
        # Prüfe Daten-Cache
        if not force_refresh and symbol in self.data_cache and timeframe in self.data_cache[symbol]:
            last_update = self.last_updates.get(symbol, {}).get(timeframe)
            
            if last_update and (datetime.now() - last_update).total_seconds() < 3600:  # 1 Stunde
                return self.data_cache[symbol][timeframe]
        
        # Prüfe, ob Datei existiert
        file_path = self._get_data_file_path(symbol, timeframe)
        
        if not force_refresh and os.path.exists(file_path):
            # Prüfe Alter der Datei
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if (datetime.now() - file_time).total_seconds() < 3600:  # 1 Stunde
                try:
                    # Lade aus Datei
                    data = pd.read_pickle(file_path)
                    
                    if data is not None and len(data) > 0:
                        # Aktualisiere Cache
                        if symbol not in self.data_cache:
                            self.data_cache[symbol] = {}
                        
                        self.data_cache[symbol][timeframe] = data
                        
                        if symbol not in self.last_updates:
                            self.last_updates[symbol] = {}
                        
                        self.last_updates[symbol][timeframe] = datetime.now()
                        
                        return data
                except Exception as e:
                    self.logger.warning(f"Fehler beim Laden von {symbol} ({self._timeframe_to_string(timeframe)}): {str(e)}")
        
        # Berechne Zeiträume
        from_date = datetime.now() - timedelta(days=lookback_days)
        to_date = datetime.now()
        
        # MT5-Zeitzone
        mt5_timezone = pytz.timezone("Etc/UTC")
        
        # Konvertiere zu MT5-Zeitzone
        from_date = from_date.replace(tzinfo=pytz.UTC)
        to_date = to_date.replace(tzinfo=pytz.UTC)
        
        # Hole Daten von MT5
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"Keine Daten für {symbol} ({self._timeframe_to_string(timeframe)})")
                return None
            
            # Konvertiere zu DataFrame
            data = pd.DataFrame(rates)
            
            # Konvertiere Zeit
            data['time'] = pd.to_datetime(data['time'], unit='s')
            
            # Berechne technische Indikatoren
            data = self._add_technical_indicators(data)
            
            # Speichere in Cache
            if symbol not in self.data_cache:
                self.data_cache[symbol] = {}
            
            self.data_cache[symbol][timeframe] = data
            
            if symbol not in self.last_updates:
                self.last_updates[symbol] = {}
            
            self.last_updates[symbol][timeframe] = datetime.now()
            
            # Speichere in Datei
            data.to_pickle(file_path)
            
            self.logger.debug(f"Daten für {symbol} ({self._timeframe_to_string(timeframe)}) abgerufen: {len(data)} Bars")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Daten für {symbol} ({self._timeframe_to_string(timeframe)}): {str(e)}")
            return None
    
    def _calculate_correlation_and_lead_lag(self, main_data, cross_data, window=100, max_shift=10):
        """
        Berechnet Korrelation und Lead-Lag-Beziehung zwischen zwei Zeitreihen.
        
        Args:
            main_data: DataFrame mit Hauptsymbol-Daten
            cross_data: DataFrame mit Cross-Asset-Daten
            window: Fenstergröße für Korrelationsberechnung
            max_shift: Maximale Verschiebung für Lead-Lag-Suche
            
        Returns:
            tuple: (Korrelation, Lead-Lag in Bars)
        """
        if main_data is None or cross_data is None:
            return 0, 0
        
        if len(main_data) < window or len(cross_data) < window:
            return 0, 0
        
        try:
            # Bereite Preisdaten vor
            main_close = main_data['close'].values[-window:]
            cross_close = cross_data['close'].values[-window:]
            
            # Berechne prozentuale Änderungen
            main_returns = np.diff(main_close) / main_close[:-1]
            cross_returns = np.diff(cross_close) / cross_close[:-1]
            
            # Finde optimale Verschiebung (Lead-Lag)
            best_corr = 0
            best_shift = 0
            
            for shift in range(-max_shift, max_shift + 1):
                if shift < 0:
                    # Cross Asset führt (negativ)
                    corr = np.corrcoef(
                        main_returns[-shift:],
                        cross_returns[:shift]
                    )[0, 1]
                elif shift > 0:
                    # Main Asset führt (positiv)
                    corr = np.corrcoef(
                        main_returns[:-shift],
                        cross_returns[shift:]
                    )[0, 1]
                else:
                    # Keine Verschiebung
                    corr = np.corrcoef(main_returns, cross_returns)[0, 1]
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_shift = shift
            
            # Negative Verschiebung bedeutet, dass das Cross-Asset führt
            return best_corr, best_shift
            
        except Exception as e:
            self.logger.error(f"Fehler bei Korrelationsberechnung: {str(e)}")
            return 0, 0
    
    def _add_technical_indicators(self, data):
        """
        Fügt technische Indikatoren zu den Daten hinzu.
        
        Args:
            data: DataFrame mit Preis-Daten
            
        Returns:
            pd.DataFrame: Daten mit Indikatoren
        """
        if data is None or len(data) == 0:
            return data
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        data['ma5'] = data['close'].rolling(window=5).mean()
        data['ma10'] = data['close'].rolling(window=10).mean()
        data['ma20'] = data['close'].rolling(window=20).mean()
        
        # Bollinger Bands
        data['ma20_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['ma20'] + 2 * data['ma20_std']
        data['bb_lower'] = data['ma20'] - 2 * data['ma20_std']
        
        # ATR
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        data['atr'] = true_range.rolling(window=14).mean()
        
        # MACD
        exp12 = data['close'].ewm(span=12, adjust=False).mean()
        exp26 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp12 - exp26
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        
        return data
    
    def _find_nearest_timestamp(self, data, timestamp):
        """
        Findet den nächsten Zeitstempel in den Daten.
        
        Args:
            data: DataFrame mit Zeitstempel-Index
            timestamp: Zu suchender Zeitstempel
            
        Returns:
            int: Index des nächsten Zeitstempels oder None
        """
        if data is None or len(data) == 0:
            return None
        
        # Konvertiere Zeitzone wenn nötig
        if timestamp.tzinfo is not None and data['time'].iloc[0].tzinfo is None:
            timestamp = timestamp.replace(tzinfo=None)
        
        try:
            # Versuche exakte Übereinstimmung zu finden
            exact_match = data.index[data['time'] == timestamp]
            if len(exact_match) > 0:
                return exact_match[0]
            
            # Finde nächstgelegenen Zeitpunkt
            time_diff = abs(data['time'] - timestamp)
            return time_diff.idxmin()
        except:
            # Alternativ: manuelle Suche
            min_diff = float('inf')
            min_idx = None
            
            for i, row_time in enumerate(data['time']):
                diff = abs((row_time - timestamp).total_seconds())
                
                if diff < min_diff:
                    min_diff = diff
                    min_idx = i
            
            return min_idx
    
    def _calculate_relative_price(self, row, price_col, data, lookback=5):
        """
        Berechnet den relativen Preis im Vergleich zum vorherigen Zeitraum.
        
        Args:
            row: Aktuelle Zeile
            price_col: Preisspalte (z.B. 'close')
            data: Vollständiger DataFrame
            lookback: Lookback-Periode für Normalisierung
            
        Returns:
            float: Relativer Preis
        """
        if price_col not in row or price_col not in data.columns:
            return 0
        
        try:
            price = row[price_col]
            idx = data.index[data['time'] == row['time']]
            
            if len(idx) == 0 or idx[0] < lookback:
                return 0
            
            prev_idx = idx[0] - lookback
            prev_price = data.iloc[prev_idx][price_col]
            
            return (price / prev_price - 1) if prev_price > 0 else 0
        except:
            return 0
    
    def _calculate_relative_volume(self, row, data, lookback=20):
        """
        Berechnet das relative Volumen im Vergleich zum Durchschnitt.
        
        Args:
            row: Aktuelle Zeile
            data: Vollständiger DataFrame
            lookback: Lookback-Periode für Durchschnitt
            
        Returns:
            float: Relatives Volumen
        """
        if 'tick_volume' not in row or 'tick_volume' not in data.columns:
            return 0
        
        try:
            volume = row['tick_volume']
            idx = data.index[data['time'] == row['time']]
            
            if len(idx) == 0 or idx[0] < lookback:
                return 0
            
            prev_idx = idx[0] - lookback
            avg_volume = data.iloc[prev_idx:idx[0]]['tick_volume'].mean()
            
            return (volume / avg_volume) if avg_volume > 0 else 0
        except:
            return 0
    
    def _select_active_symbols(self, min_correlation=0.2):
        """
        Wählt aktive Symbole basierend auf Korrelationsstärke aus.
        
        Args:
            min_correlation: Minimale absolute Korrelation für aktive Symbole
        """
        self.active_symbols = set()
        
        for symbol in self.correlations:
            # Durchschnittliche absolute Korrelation über alle Timeframes
            abs_correlations = [abs(self.correlations[symbol][tf]) for tf in self.correlations[symbol]]
            
            if abs_correlations and max(abs_correlations) >= min_correlation:
                self.active_symbols.add(symbol)
    
    def _calculate_correlation_weights(self):
        """
        Berechnet Gewichte für jedes Symbol basierend auf Korrelationsstärke.
        """
        weights = {}
        
        # Berechne durchschnittliche absolute Korrelation pro Symbol
        avg_abs_correlations = {}
        
        for symbol in self.active_symbols:
            abs_corrs = [abs(self.correlations[symbol][tf]) for tf in self.correlations[symbol]]
            avg_abs_correlations[symbol] = sum(abs_corrs) / len(abs_corrs) if abs_corrs else 0
        
        # Normalisiere zu Summe 1
        total = sum(avg_abs_correlations.values())
        
        if total > 0:
            for symbol in avg_abs_correlations:
                weights[symbol] = avg_abs_correlations[symbol] / total
        
        self.correlation_weights = weights
    
    def _get_data_file_path(self, symbol, timeframe):
        """
        Gibt den Dateipfad für gespeicherte Daten zurück.
        
        Args:
            symbol: Handelssymbol
            timeframe: MT5-Timeframe
            
        Returns:
            str: Dateipfad
        """
        tf_str = self._timeframe_to_string(timeframe)
        return os.path.join(self.data_dir, f"{symbol}_{tf_str}.pkl")
    
    def _timeframe_to_string(self, timeframe):
        """
        Konvertiert MT5-Timeframe zu String.
        
        Args:
            timeframe: MT5-Timeframe
            
        Returns:
            str: Timeframe als String
        """
        if timeframe == mt5.TIMEFRAME_M1:
            return "M1"
        elif timeframe == mt5.TIMEFRAME_M5:
            return "M5"
        elif timeframe == mt5.TIMEFRAME_M15:
            return "M15"
        elif timeframe == mt5.TIMEFRAME_M30:
            return "M30"
        elif timeframe == mt5.TIMEFRAME_H1:
            return "H1"
        elif timeframe == mt5.TIMEFRAME_H4:
            return "H4"
        elif timeframe == mt5.TIMEFRAME_D1:
            return "D1"
        elif timeframe == mt5.TIMEFRAME_W1:
            return "W1"
        elif timeframe == mt5.TIMEFRAME_MN1:
            return "MN1"
        else:
            return f"TF{timeframe}"
    
    def _save_status(self):
        """
        Speichert den aktuellen Status des Managers.
        
        Returns:
            bool: Erfolg des Speicherns
        """
        try:
            status = {
                "main_symbol": self.main_symbol,
                "cross_assets": self.cross_assets,
                "active_symbols": list(self.active_symbols),
                "correlations": self.correlations,
                "lead_lag": self.lead_lag,
                "correlation_weights": self.correlation_weights,
                "last_correlation_update": self.last_correlation_update.isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Speichere Status in JSON-Datei
            status_file = os.path.join(self.data_dir, "cross_asset_status.json")
            
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            self.logger.debug(f"Status gespeichert in {status_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Status: {str(e)}")
            return False
    
    def _load_status(self):
        """
        Lädt den zuletzt gespeicherten Status.
        
        Returns:
            bool: Erfolg des Ladens
        """
        status_file = os.path.join(self.data_dir, "cross_asset_status.json")
        
        if not os.path.exists(status_file):
            self.logger.debug("Keine Status-Datei gefunden")
            return False
        
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            if status["main_symbol"] != self.main_symbol:
                self.logger.warning(f"Hauptsymbol in Status-Datei ({status['main_symbol']}) unterscheidet sich vom aktuellen ({self.main_symbol})")
                return False
            
            self.cross_assets = status.get("cross_assets", self.cross_assets)
            self.active_symbols = set(status.get("active_symbols", []))
            self.correlations = status.get("correlations", {})
            self.lead_lag = status.get("lead_lag", {})
            self.correlation_weights = status.get("correlation_weights", {})
            
            if "last_correlation_update" in status:
                try:
                    self.last_correlation_update = datetime.fromisoformat(status["last_correlation_update"])
                except:
                    self.last_correlation_update = datetime.now() - timedelta(hours=self.update_correlation_interval/3600)
            
            self.logger.info(f"Status geladen aus {status_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Status: {str(e)}")
            return False
    
    def _start_update_thread(self):
        """
        Startet einen Thread für regelmäßige Updates.
        """
        if self.update_thread is not None and self.update_thread.is_alive():
            self.logger.debug("Update-Thread läuft bereits")
            return
        
        self.stop_thread = False
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.debug("Update-Thread gestartet")
    
    def _stop_update_thread(self):
        """
        Stoppt den Update-Thread.
        """
        if self.update_thread is None or not self.update_thread.is_alive():
            return
        
        self.stop_thread = True
        self.update_thread.join(timeout=5)
        
        self.logger.debug("Update-Thread gestoppt")
    
    def _update_loop(self):
        """
        Hauptschleife für regelmäßige Updates.
        """
        while not self.stop_thread:
            # Prüfe, ob Update nötig ist
            now = datetime.now()
            time_since_update = (now - self.last_correlation_update).total_seconds()
            
            if time_since_update >= self.update_correlation_interval:
                try:
                    self.update_correlations()
                except Exception as e:
                    self.logger.error(f"Fehler bei automatischem Update: {str(e)}")
            
            # Warte für nächste Prüfung (alle 5 Minuten)
            for _ in range(300):  # 5 Minuten in Sekunden
                if self.stop_thread:
                    break
                time.sleep(1)
    
    def __del__(self):
        """
        Aufräumen beim Zerstören der Instanz.
        """
        self._stop_update_thread()
        
        # Speichere finalen Status
        if self.is_initialized:
            self._save_status()

class CrossAssetBacktester:
    """
    Backtestet die Vorhersagekraft von Cross-Asset-Signalen.
    """
    def __init__(self, main_symbol="GBPJPY", cross_assets=None, 
                lookback_days=100, test_days=30):
        """
        Initialisiert den Cross-Asset-Backtester.
        
        Args:
            main_symbol: Haupthandelssymbol
            cross_assets: Liste von Cross-Assets
            lookback_days: Anzahl der Tage für Training
            test_days: Anzahl der Tage für Test
        """
        self.main_symbol = main_symbol
        self.cross_assets = cross_assets
        self.lookback_days = lookback_days
        self.test_days = test_days
        
        # Manager für Cross-Asset-Daten
        self.manager = CrossAssetManager(
            main_symbol=main_symbol,
            cross_assets=cross_assets
        )
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('cross_asset_backtester')
    
    def run_backtest(self, prediction_horizons=[1, 5, 10], correlation_threshold=0.2):
        """
        Führt einen Backtest durch, um die Vorhersagekraft von Cross-Assets zu prüfen.
        
        Args:
            prediction_horizons: Liste von Vorhersagehorizonten in Bars
            correlation_threshold: Minimale Korrelation für Berücksichtigung
            
        Returns:
            dict: Backtest-Ergebnisse
        """
        self.logger.info(f"Starte Cross-Asset-Backtest für {self.main_symbol}")
        
        # Initialisiere Manager
        if not self.manager.initialize(lookback_days=self.lookback_days + self.test_days):
            self.logger.error("Fehler bei Initialisierung des Managers")
            return {"error": "Initialisierungsfehler"}
        
        # Hole Daten für Hauptsymbol (für alle Timeframes)
        main_data = {}
        
        for timeframe in self.manager.timeframes:
            data = self._fetch_test_data(self.main_symbol, timeframe)
            if data is not None:
                main_data[timeframe] = data
        
        if not main_data:
            self.logger.error("Keine Testdaten für Hauptsymbol verfügbar")
            return {"error": "Keine Testdaten"}
        
        # Bereite Ergebnisstruktur vor
        results = {
            "main_symbol": self.main_symbol,
            "test_period": f"{self.test_days} Tage",
            "prediction_horizons": prediction_horizons,
            "assets": {},
            "metrics": {},
            "best_predictors": {}
        }
        
        # Teste jedes Asset und Timeframe
        for symbol in self.manager.active_symbols:
            symbol_results = {}
            
            for timeframe in main_data:
                if timeframe not in self.manager.correlations.get(symbol, {}):
                    continue
                
                # Prüfe Korrelation
                correlation = self.manager.correlations[symbol][timeframe]
                
                if abs(correlation) < correlation_threshold:
                    continue
                
                # Hole Testdaten für dieses Symbol
                asset_data = self._fetch_test_data(symbol, timeframe)
                
                if asset_data is None or len(asset_data) < 20:
                    continue
                
                # Lead-Lag berücksichtigen
                lead_lag = self.manager.lead_lag.get(symbol, {}).get(timeframe, 0)
                
                # Teste Vorhersagekraft für verschiedene Horizonte
                horizon_results = {}
                
                for horizon in prediction_horizons:
                    prediction_metrics = self._test_prediction_power(
                        main_data[timeframe], asset_data, horizon, lead_lag
                    )
                    
                    horizon_results[horizon] = prediction_metrics
                
                # Speichere Ergebnisse für dieses Symbol und Timeframe
                tf_str = self.manager._timeframe_to_string(timeframe)
                symbol_results[tf_str] = {
                    "correlation": correlation,
                    "lead_lag": lead_lag,
                    "horizons": horizon_results
                }
            
            if symbol_results:
                results["assets"][symbol] = symbol_results
        
        # Berechne zusammenfassende Metriken
        for horizon in prediction_horizons:
            # Beste Prädiktoren für diesen Horizont
            best_predictors = []
            
            for symbol in results["assets"]:
                for tf_str in results["assets"][symbol]:
                    if horizon in results["assets"][symbol][tf_str]["horizons"]:
                        metrics = results["assets"][symbol][tf_str]["horizons"][horizon]
                        
                        if metrics["directional_accuracy"] > 0.55:  # Mindestgenauigkeit
                            best_predictors.append({
                                "symbol": symbol,
                                "timeframe": tf_str,
                                "accuracy": metrics["directional_accuracy"],
                                "lead_lag": results["assets"][symbol][tf_str]["lead_lag"],
                                "correlation": results["assets"][symbol][tf_str]["correlation"]
                            })
            
            # Sortiere nach Genauigkeit
            best_predictors = sorted(best_predictors, key=lambda x: x["accuracy"], reverse=True)
            
            # Speichere besten Prädiktoren
            results["best_predictors"][horizon] = best_predictors[:5]  # Top 5
        
        # Berechne durchschnittliche Metriken
        avg_metrics = {
            "directional_accuracy": {},
            "correlation": {},
            "prediction_strength": {}
        }
        
        for horizon in prediction_horizons:
            accuracies = []
            correlations = []
            strengths = []
            
            for symbol in results["assets"]:
                for tf_str in results["assets"][symbol]:
                    if horizon in results["assets"][symbol][tf_str]["horizons"]:
                        metrics = results["assets"][symbol][tf_str]["horizons"][horizon]
                        accuracies.append(metrics["directional_accuracy"])
                        correlations.append(abs(results["assets"][symbol][tf_str]["correlation"]))
                        strengths.append(metrics["prediction_strength"])
            
            avg_metrics["directional_accuracy"][horizon] = np.mean(accuracies) if accuracies else 0
            avg_metrics["correlation"][horizon] = np.mean(correlations) if correlations else 0
            avg_metrics["prediction_strength"][horizon] = np.mean(strengths) if strengths else 0
        
        results["metrics"] = avg_metrics
        
        self.logger.info(f"Cross-Asset-Backtest abgeschlossen mit {len(results['assets'])} Assets")
        
        return results
    
    def _fetch_test_data(self, symbol, timeframe):
        """
        Holt Testdaten für ein Symbol und Timeframe.
        
        Args:
            symbol: Handelssymbol
            timeframe: MT5-Timeframe
            
        Returns:
            pd.DataFrame: Testdaten
        """
        if symbol not in self.manager.data_cache or timeframe not in self.manager.data_cache[symbol]:
            return None
        
        data = self.manager.data_cache[symbol][timeframe]
        
        if data is None or len(data) < self.lookback_days:
            return None
        
        # Beschränke auf Testperiode
        test_data = data.iloc[-int(self.test_days * 24 / self._timeframe_hours(timeframe)):]
        
        return test_data if len(test_data) > 0 else None
    
    def _test_prediction_power(self, main_data, asset_data, horizon, lead_lag=0):
        """
        Testet die Vorhersagekraft eines Assets für das Hauptsymbol.
        
        Args:
            main_data: DataFrame mit Hauptsymbol-Daten
            asset_data: DataFrame mit Asset-Daten
            horizon: Vorhersagehorizont in Bars
            lead_lag: Lead-Lag-Beziehung in Bars
            
        Returns:
            dict: Vorhersagemetriken
        """
        if main_data is None or asset_data is None:
            return {"directional_accuracy": 0, "prediction_strength": 0}
        
        # Beschränke auf gemeinsamen Zeitraum
        common_range = set(main_data['time']).intersection(set(asset_data['time']))
        
        if not common_range:
            return {"directional_accuracy": 0, "prediction_strength": 0}
        
        # Erstelle Time-Series mit Prozent-Änderungen
        main_pct = main_data.set_index('time')['close'].pct_change()
        asset_pct = asset_data.set_index('time')['close'].pct_change()
        
        # Filtere NaN-Werte
        main_pct = main_pct.dropna()
        asset_pct = asset_pct.dropna()
        
        # Gemeinsame Zeitpunkte
        common_idx = main_pct.index.intersection(asset_pct.index)
        
        if len(common_idx) < horizon + 5:
            return {"directional_accuracy": 0, "prediction_strength": 0}
        
        # Sortiere nach Zeit
        main_pct = main_pct.loc[common_idx].sort_index()
        asset_pct = asset_pct.loc[common_idx].sort_index()
        
        # Anwende Lead-Lag-Verschiebung
        if lead_lag != 0:
            asset_pct = asset_pct.shift(lead_lag)
            
            # Filtere NaN-Werte nach Verschiebung
            valid_idx = main_pct.index.intersection(asset_pct.dropna().index)
            main_pct = main_pct.loc[valid_idx]
            asset_pct = asset_pct.loc[valid_idx]
        
        # Berechne zukünftige Preisänderungen für Hauptsymbol
        future_returns = main_pct.shift(-horizon)
        
        # Entferne NaN-Werte
        valid_idx = asset_pct.dropna().index.intersection(future_returns.dropna().index)
        
        if len(valid_idx) < 20:
            return {"directional_accuracy": 0, "prediction_strength": 0}
        
        asset_pct = asset_pct.loc[valid_idx]
        future_returns = future_returns.loc[valid_idx]
        
        # Berechne Vorhersagegenauigkeit
        asset_direction = np.sign(asset_pct)
        future_direction = np.sign(future_returns)
        
        # Berechne direktionale Genauigkeit
        direction_match = (asset_direction == future_direction)
        directional_accuracy = direction_match.mean()
        
        # Berechne Vorhersagestärke (Korrelation)
        prediction_strength = np.corrcoef(asset_pct, future_returns)[0, 1]
        
        return {
            "directional_accuracy": float(directional_accuracy),
            "prediction_strength": float(prediction_strength) if not np.isnan(prediction_strength) else 0.0,
            "sample_size": len(valid_idx)
        }
    
    def _timeframe_hours(self, timeframe):
        """
        Konvertiert MT5-Timeframe zu Stunden.
        
        Args:
            timeframe: MT5-Timeframe
            
        Returns:
            float: Timeframe in Stunden
        """
        if timeframe == mt5.TIMEFRAME_M1:
            return 1/60
        elif timeframe == mt5.TIMEFRAME_M5:
            return 5/60
        elif timeframe == mt5.TIMEFRAME_M15:
            return 15/60
        elif timeframe == mt5.TIMEFRAME_M30:
            return 30/60
        elif timeframe == mt5.TIMEFRAME_H1:
            return 1
        elif timeframe == mt5.TIMEFRAME_H4:
            return 4
        elif timeframe == mt5.TIMEFRAME_D1:
            return 24
        elif timeframe == mt5.TIMEFRAME_W1:
            return 24*7
        elif timeframe == mt5.TIMEFRAME_MN1:
            return 24*30
        else:
            return 1  # Standardwert