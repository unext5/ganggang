import pandas as pd
import numpy as np
import ta  # pip install ta
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(path_30min_csv, path_5min_csv, path_1h_csv=None, path_4h_csv=None):
    """
    Lädt die verschiedenen Timeframe-Daten und führt Basis-Sortierung durch.
    """
    # 30min Daten laden
    df_30 = pd.read_csv(path_30min_csv, parse_dates=['time'])
    df_30.sort_values('time', inplace=True)
    df_30.reset_index(drop=True, inplace=True)
    
    # 5min Daten laden
    df_5 = pd.read_csv(path_5min_csv, parse_dates=['time'])
    df_5.sort_values('time', inplace=True)
    df_5.reset_index(drop=True, inplace=True)
    
    # 1h Daten laden (falls vorhanden)
    df_1h = None
    if path_1h_csv:
        df_1h = pd.read_csv(path_1h_csv, parse_dates=['time'])
        df_1h.sort_values('time', inplace=True)
        df_1h.reset_index(drop=True, inplace=True)
    
    # 4h Daten laden (falls vorhanden)
    df_4h = None
    if path_4h_csv:
        df_4h = pd.read_csv(path_4h_csv, parse_dates=['time'])
        df_4h.sort_values('time', inplace=True)
        df_4h.reset_index(drop=True, inplace=True)
    
    return df_30, df_5, df_1h, df_4h

def add_session_flags(df):
    """
    Fügt Flags für Handelssessions hinzu:
    - Asia: 00:00-08:00 UTC
    - Europe: 07:00-16:00 UTC
    - US: 13:00-22:00 UTC
    - Overlap: Überschneidungen der Sessions
    """
    # Zeitzonen in UTC umrechnen
    df['hour_utc'] = df['time'].dt.hour
    
    # Session Flags
    df['session_asia'] = ((df['hour_utc'] >= 0) & (df['hour_utc'] < 8)).astype(int)
    df['session_europe'] = ((df['hour_utc'] >= 7) & (df['hour_utc'] < 16)).astype(int)
    df['session_us'] = ((df['hour_utc'] >= 13) & (df['hour_utc'] < 22)).astype(int)
    
    # Overlap Flags
    df['session_overlap'] = (
        ((df['hour_utc'] >= 7) & (df['hour_utc'] < 8)) |  # Asia-Europe
        ((df['hour_utc'] >= 13) & (df['hour_utc'] < 16))   # Europe-US
    ).astype(int)
    
    return df

def add_weekday_flags(df):
    """
    Fügt Wochentag-Flags hinzu (Montag bis Freitag)
    """
    # Wochentage (0 = Montag, 4 = Freitag)
    day_of_week = df['time'].dt.dayofweek
    
    # One-hot encoding für die Wochentage
    df['day_mon'] = (day_of_week == 0).astype(int)
    df['day_tue'] = (day_of_week == 1).astype(int)
    df['day_wed'] = (day_of_week == 2).astype(int)
    df['day_thu'] = (day_of_week == 3).astype(int)
    df['day_fri'] = (day_of_week == 4).astype(int)
    
    return df

def optimize_features(features, feature_cols, variance_threshold=0.95, n_components=None):
    """
    PCA zur Dimensionsreduktion und Feature-Selektion
    
    Args:
        features: Numpy-Array mit Eingabefeatures
        feature_cols: Liste der Feature-Namen
        variance_threshold: Anteil der zu erhaltenden Varianz (wenn n_components=None)
        n_components: Direkte Angabe der zu verwendenden Komponenten (überschreibt variance_threshold)
    
    Returns:
        reduced_features: PCA-reduzierte Features
        pca_model: Trainiertes PCA-Modell
        feature_importance: Wichtigkeit jedes ursprünglichen Features
        scaler: Trainierter Standardisierungs-Scaler
    """
    # Standardisierung der Daten
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # PCA durchführen
    if n_components is None:
        # Vollständige PCA zuerst
        pca_full = PCA()
        pca_full.fit(scaled_features)
        
        # Erklärte Varianz berechnen
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Anzahl Komponenten für gewünschte erklärte Varianz
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Reduzierte PCA berechnen
    pca_reduced = PCA(n_components=n_components)
    reduced_features = pca_reduced.fit_transform(scaled_features)
    
    # Feature Importance
    # Nimm Absolutwerte der Komponenten und summiere über alle Komponenten
    feature_importance = np.abs(pca_reduced.components_).sum(axis=0)
    # Normalisieren
    feature_importance = feature_importance / feature_importance.sum()
    
    feature_importance_dict = {feature_cols[i]: feature_importance[i] for i in range(len(feature_cols))}
    
    return reduced_features, pca_reduced, feature_importance_dict, scaler

def load_cross_asset_data(base_df, related_assets_files):
    """
    Lädt korrelierte Asset-Daten und bereitet sie für Cross-Asset-Training vor
    
    Args:
        base_df: DataFrame mit Basisdaten (GBPJPY)
        related_assets_files: Dictionary mit Asset-Namen und Dateipfaden
    
    Returns:
        DataFrame mit hinzugefügten korrelierten Asset-Features
    """
    result_df = base_df.copy()
    
    # Für jedes korrelierte Asset
    for asset_name, filepath in related_assets_files.items():
        try:
            # Asset-Daten laden
            asset_df = pd.read_csv(filepath, parse_dates=['time'])
            asset_df.sort_values('time', inplace=True)
            
            # Synchronisiere Zeitreihen
            merged_df = pd.merge_asof(
                result_df[['time']].sort_values('time'),
                asset_df[['time', 'close']].sort_values('time'),
                on='time', direction='backward'
            )
            
            # Berechne Returns
            merged_df[f'close_{asset_name}'] = merged_df['close']
            merged_df[f'log_return_{asset_name}'] = np.log(
                merged_df[f'close_{asset_name}'] / 
                merged_df[f'close_{asset_name}'].shift(1)
            ).fillna(0)
            
            # Berechne Korrelation mit Hauptpaar
            if len(result_df) > 100 and 'log_return30' in result_df.columns:
                recent_base_returns = result_df['log_return30'].iloc[-100:].values
                recent_asset_returns = merged_df[f'log_return_{asset_name}'].iloc[-100:].values
                
                if len(recent_base_returns) == len(recent_asset_returns) and len(recent_base_returns) > 0:
                    correlation = np.corrcoef(recent_base_returns, recent_asset_returns)[0, 1]
                    print(f"Korrelation mit {asset_name}: {correlation:.4f}")
                else:
                    correlation = 0
                    print(f"Warnung: Kann Korrelation für {asset_name} nicht berechnen")
            else:
                correlation = 1  # Default: Immer hinzufügen, wenn nicht genug Daten
            
            # Füge Asset-Features hinzu (bei starker Korrelation oder immer zu Beginn)
            if abs(correlation) > 0.3 or len(result_df) < 100:
                # Merge zurück zum Hauptdatensatz
                result_df = pd.merge(
                    result_df, 
                    merged_df[['time', f'log_return_{asset_name}']], 
                    on='time', how='left'
                )
                result_df[f'log_return_{asset_name}'].fillna(0, inplace=True)
                print(f"Asset {asset_name} hinzugefügt")
            else:
                print(f"Asset {asset_name} übersprungen (zu geringe Korrelation)")
                
        except Exception as e:
            print(f"Fehler beim Laden von {asset_name}: {str(e)}")
    
    return result_df

def compute_all_features(path_30min_csv, path_5min_csv, path_1h_csv=None, path_4h_csv=None, 
                        related_assets=None, apply_pca=False, pca_threshold=0.95):
    """
    Erweiterte Feature-Berechnung mit zusätzlichen Features und optionaler Dimensionsreduktion:
      1-4) log_return(30min, 5min, 1h, 4h)
      5-9) Indikatoren (RSI 30m/1h, ATR 30m/4h, MACD 30m)
      10-14) Session & Volume (session_asia/europe/us/overlap, log_volume)
      15-19) Wochentage (day_mon, day_tue, day_wed, day_thu, day_fri)
      20+) Optional: Korrelierte Assets
    
    Args:
        path_30min_csv: Pfad zur 30min-CSV-Datei
        path_5min_csv: Pfad zur 5min-CSV-Datei
        path_1h_csv: Pfad zur 1h-CSV-Datei (optional)
        path_4h_csv: Pfad zur 4h-CSV-Datei (optional)
        related_assets: Dictionary mit Pfaden zu korrelierten Assets (optional)
        apply_pca: Ob PCA angewendet werden soll
        pca_threshold: Anteil der beizubehaltenden Varianz bei PCA
    
    Returns:
        DataFrame mit allen Features oder
        (DataFrame, PCA-Modell, Feature-Importance, Scaler) wenn apply_pca=True
    """
    # 1) Daten laden
    df_30, df_5, df_1h, df_4h = load_data(path_30min_csv, path_5min_csv, path_1h_csv, path_4h_csv)
    
    # 2) Basis-Features für 30min
    # 30min Returns
    df_30['log_return30'] = np.log(df_30['close']/df_30['close'].shift(1))
    df_30['log_return30'].fillna(0, inplace=True)
    
    # Log Volume - mit Unterstützung für tick_volume oder volume
    if 'tick_volume' in df_30.columns:
        df_30['log_volume'] = np.log1p(df_30['tick_volume'])
    elif 'volume' in df_30.columns:
        df_30['log_volume'] = np.log1p(df_30['volume'])
    else:
        # Falls keine Volume-Daten verfügbar, verwenden wir einen Dummy-Wert
        df_30['log_volume'] = 0
    
    # RSI (30min)
    df_30['rsi_30m'] = ta.momentum.rsi(df_30['close'], window=14)
    
    # ATR (30min)
    df_30['atr_30m'] = ta.volatility.average_true_range(
        df_30['high'], df_30['low'], df_30['close'], window=14
    )
    
    # MACD-Diff (30min)
    df_30['macd_30m'] = ta.trend.macd_diff(
        df_30['close'], window_slow=26, window_fast=12, window_sign=9
    )
    
    # 3) Session Flags und Wochentag-Flags
    df_30 = add_session_flags(df_30)
    df_30 = add_weekday_flags(df_30)
    
    # 4) Merge von 5min Daten für log_return5
    df_5['log_return5'] = np.log(df_5['close']/df_5['close'].shift(1))
    df_5['log_return5'].fillna(0, inplace=True)
    
    df_30 = pd.merge_asof(
        df_30.sort_values('time'),
        df_5[['time','log_return5']].sort_values('time'),
        on='time', direction='backward'
    )
    
    # 5) 1h Features hinzufügen, falls 1h Daten vorhanden
    if df_1h is not None:
        # 1h Returns
        df_1h['log_return1h'] = np.log(df_1h['close']/df_1h['close'].shift(1))
        df_1h['log_return1h'].fillna(0, inplace=True)
        
        # RSI (1h)
        df_1h['rsi_1h'] = ta.momentum.rsi(df_1h['close'], window=14)
        
        # Merge mit 30min Daten
        df_30 = pd.merge_asof(
            df_30.sort_values('time'),
            df_1h[['time','log_return1h','rsi_1h']].sort_values('time'),
            on='time', direction='backward'
        )
    else:
        # Fallback: Dummy-Werte setzen
        df_30['log_return1h'] = df_30['log_return30']
        df_30['rsi_1h'] = df_30['rsi_30m']
    
    # 6) 4h Features hinzufügen, falls 4h Daten vorhanden
    if df_4h is not None:
        # 4h Returns
        df_4h['log_return4h'] = np.log(df_4h['close']/df_4h['close'].shift(1))
        df_4h['log_return4h'].fillna(0, inplace=True)
        
        # ATR (4h)
        df_4h['atr_4h'] = ta.volatility.average_true_range(
            df_4h['high'], df_4h['low'], df_4h['close'], window=14
        )
        
        # Merge mit 30min Daten
        df_30 = pd.merge_asof(
            df_30.sort_values('time'),
            df_4h[['time','log_return4h','atr_4h']].sort_values('time'),
            on='time', direction='backward'
        )
    else:
        # Fallback: Dummy-Werte setzen
        df_30['log_return4h'] = df_30['log_return30']
        df_30['atr_4h'] = df_30['atr_30m']
    
    # 7) Optional: Cross-Asset-Daten hinzufügen
    if related_assets:
        df_30 = load_cross_asset_data(df_30, related_assets)
    
    # 8) Aufräumen & Finale Vorbereitung
    # Liste aller benötigten Feature-Spalten (Basis)
    feature_cols = [
        'log_return30', 'log_return5', 'log_return1h', 'log_return4h',
        'rsi_30m', 'rsi_1h', 'atr_30m', 'atr_4h', 'macd_30m',
        'session_asia', 'session_europe', 'session_us', 'session_overlap',
        'log_volume',
        'day_mon', 'day_tue', 'day_wed', 'day_thu', 'day_fri'
    ]
    
    # Füge Cross-Asset-Features hinzu, falls vorhanden
    cross_asset_features = [col for col in df_30.columns if col.startswith('log_return_')]
    feature_cols.extend(cross_asset_features)
    
    # NaN-Werte in Features bereinigen
    for col in feature_cols:
        if col in df_30.columns:
            if col.startswith('log_return') or col.startswith('rsi') or col.startswith('atr') or col.startswith('macd'):
                df_30[col].fillna(0, inplace=True)
            else:
                df_30[col].fillna(0, inplace=True)
    
    # Sicherstellen, dass alle Feature-Spalten existieren
    for col in feature_cols:
        if col not in df_30.columns:
            df_30[col] = 0
    
    # Finale Bereinigung
    df_30.dropna(subset=feature_cols, inplace=True)
    df_30.reset_index(drop=True, inplace=True)
    
    # 9) Optional: PCA für Dimensionsreduktion
    if apply_pca:
        features_matrix = df_30[feature_cols].values
        reduced_features, pca_model, feature_importance, scaler = optimize_features(
            features_matrix, feature_cols, pca_threshold
        )
        
        # PCA-Features zum DataFrame hinzufügen
        for i in range(reduced_features.shape[1]):
            df_30[f'pca_feature_{i}'] = reduced_features[:, i]
        
        return df_30, pca_model, feature_importance, scaler
    
    return df_30

def compute_features_from_mt5(rates_30m, rates_5m, rates_1h=None, rates_4h=None, 
                             apply_pca=False, pca_model=None, scaler=None):
    """
    Erstellt Features aus MT5-Daten direkt (ohne CSV-Import)
    Mit optionaler PCA-Transformation
    """
    # Konvertieren zu DataFrames
    df_30 = pd.DataFrame(rates_30m)
    df_5 = pd.DataFrame(rates_5m)
    
    # Zeit konvertieren
    df_30['time'] = pd.to_datetime(df_30['time'], unit='s')
    df_5['time'] = pd.to_datetime(df_5['time'], unit='s')
    
    # Sortieren
    df_30.sort_values('time', inplace=True)
    df_5.sort_values('time', inplace=True)
    df_30.reset_index(drop=True, inplace=True)
    df_5.reset_index(drop=True, inplace=True)
    
    # 1h und 4h dataframes, falls vorhanden
    df_1h = None
    if rates_1h is not None:
        df_1h = pd.DataFrame(rates_1h)
        df_1h['time'] = pd.to_datetime(df_1h['time'], unit='s')
        df_1h.sort_values('time', inplace=True)
        df_1h.reset_index(drop=True, inplace=True)
    
    df_4h = None
    if rates_4h is not None:
        df_4h = pd.DataFrame(rates_4h)
        df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s')
        df_4h.sort_values('time', inplace=True)
        df_4h.reset_index(drop=True, inplace=True)
    
    # 30min Features
    df_30['log_return30'] = np.log(df_30['close']/df_30['close'].shift(1))
    df_30['log_return30'].fillna(0, inplace=True)
    
    # Log Volume - mit Unterstützung für tick_volume
    if 'tick_volume' in df_30.columns:
        df_30['log_volume'] = np.log1p(df_30['tick_volume'])
    elif 'volume' in df_30.columns:
        df_30['log_volume'] = np.log1p(df_30['volume'])
    else:
        df_30['log_volume'] = 0
    
    # RSI (30min)
    df_30['rsi_30m'] = ta.momentum.rsi(df_30['close'], window=14)
    
    # ATR (30min)
    df_30['atr_30m'] = ta.volatility.average_true_range(
        df_30['high'], df_30['low'], df_30['close'], window=14
    )
    
    # MACD-Diff (30min)
    df_30['macd_30m'] = ta.trend.macd_diff(
        df_30['close'], window_slow=26, window_fast=12, window_sign=9
    )
    
    # Session und Wochentag Flags
    df_30 = add_session_flags(df_30)
    df_30 = add_weekday_flags(df_30)
    
    # 5min Features
    df_5['log_return5'] = np.log(df_5['close']/df_5['close'].shift(1))
    df_5['log_return5'].fillna(0, inplace=True)
    
    # Merge 5min zu 30min
    df_30 = pd.merge_asof(
        df_30.sort_values('time'),
        df_5[['time','log_return5']].sort_values('time'),
        on='time', direction='backward'
    )
    
    # 1h Features, falls vorhanden
    if df_1h is not None:
        df_1h['log_return1h'] = np.log(df_1h['close']/df_1h['close'].shift(1))
        df_1h['log_return1h'].fillna(0, inplace=True)
        df_1h['rsi_1h'] = ta.momentum.rsi(df_1h['close'], window=14)
        
        df_30 = pd.merge_asof(
            df_30.sort_values('time'),
            df_1h[['time','log_return1h','rsi_1h']].sort_values('time'),
            on='time', direction='backward'
        )
    else:
        df_30['log_return1h'] = df_30['log_return30']
        df_30['rsi_1h'] = df_30['rsi_30m']
    
    # 4h Features, falls vorhanden
    if df_4h is not None:
        df_4h['log_return4h'] = np.log(df_4h['close']/df_4h['close'].shift(1))
        df_4h['log_return4h'].fillna(0, inplace=True)
        df_4h['atr_4h'] = ta.volatility.average_true_range(
            df_4h['high'], df_4h['low'], df_4h['close'], window=14
        )
        
        df_30 = pd.merge_asof(
            df_30.sort_values('time'),
            df_4h[['time','log_return4h','atr_4h']].sort_values('time'),
            on='time', direction='backward'
        )
    else:
        df_30['log_return4h'] = df_30['log_return30']
        df_30['atr_4h'] = df_30['atr_30m']
    
    # Feature-Spalten definieren
    feature_cols = [
        'log_return30', 'log_return5', 'log_return1h', 'log_return4h',
        'rsi_30m', 'rsi_1h', 'atr_30m', 'atr_4h', 'macd_30m',
        'session_asia', 'session_europe', 'session_us', 'session_overlap',
        'log_volume',
        'day_mon', 'day_tue', 'day_wed', 'day_thu', 'day_fri'
    ]
    
    # NaN-Werte bereinigen
    for col in feature_cols:
        if col in df_30.columns:
            df_30[col].fillna(0, inplace=True)
        else:
            df_30[col] = 0
    
    df_30.dropna(subset=feature_cols, inplace=True)
    df_30.reset_index(drop=True, inplace=True)
    
    # Optional: PCA anwenden, wenn Modell bereitgestellt
    if apply_pca and pca_model is not None and scaler is not None:
        # Features extrahieren
        features_matrix = df_30[feature_cols].values
        
        # Skalieren
        scaled_features = scaler.transform(features_matrix)
        
        # PCA anwenden
        reduced_features = pca_model.transform(scaled_features)
        
        # PCA-Features zum DataFrame hinzufügen
        for i in range(reduced_features.shape[1]):
            df_30[f'pca_feature_{i}'] = reduced_features[:, i]
    
    return df_30
