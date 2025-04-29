###############################################################################
# enhanced_live_inference_mt5_v3.py
#
# Significantly Enhanced Version:
#  - Integration with hybrid NN+HMM model for improved predictions
#  - Dynamic feature selection to adapt to changing market conditions
#  - Market memory pattern matching for historical precedents
#  - Order book analysis for microstructure insights
#  - Cross-asset correlation and feature integration
#  - Comprehensive monitoring and visualization capabilities
###############################################################################

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import pickle
import time
import logging
import json
import matplotlib.pyplot as plt
import ta
import threading
from datetime import datetime, timedelta
from collections import deque
import warnings
import shutil
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# Add the parent directory to the path to find modules like enhanced_features_v2
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Custom JSON Encoder für NumPy-Datentypen
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

# Import enhanced components
try:
    from enhanced_features_v2 import compute_all_features
except ImportError:
    logging.warning("enhanced_features_v2 could not be imported")

try:
    from enhanced_vol_models_v2 import t_pdf_diag_multidim, egarch_update_1step_multi
except ImportError:
    logging.error("Critical error: enhanced_vol_models_v2 could not be imported!")

try:
    from enhanced_hmm_em_v2 import compute_confidence, validate_current_state, forward_backward
except ImportError:
    logging.error("Critical error: enhanced_hmm_em_v2 could not be imported!")

# New component imports
try:
    from enhanced_hybrid_model import HybridModel
except ImportError:
    logging.warning("enhanced_hybrid_model could not be imported")
    HybridModel = None

try:
    from market_memory import MarketMemory
except ImportError:
    logging.warning("market_memory could not be imported")
    MarketMemory = None

try:
    from order_book_analyzer import OrderBookAnalyzer
except ImportError:
    logging.warning("order_book_analyzer could not be imported")
    OrderBookAnalyzer = None

try:
    from dynamic_feature_selection import DynamicFeatureSelector
except ImportError:
    logging.warning("dynamic_feature_selection could not be imported")
    DynamicFeatureSelector = None

# Optional dashboard for monitoring (will be launched in a separate thread if available)
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    DASH_AVAILABLE = True
except ImportError:
    logging.warning("Dash not available. Real-time dashboard will be disabled.")
    DASH_AVAILABLE = False

# Existing component imports
try:
    from ensemble_hmm_impl import HMMEnsemble  # Hier geändert
except ImportError:
    logging.warning("enhanced_ensemble could not be imported")
    HMMEnsemble = None

try:
    from cross_asset_manager import CrossAssetManager, CrossAssetBacktester
    CROSS_ASSET_AVAILABLE = True
except ImportError:
    logging.warning("cross_asset_manager could not be imported")
    CrossAssetManager = None
    CrossAssetBacktester = None
    CROSS_ASSET_AVAILABLE = False

try:
    from adaptive_risk_manager import AdaptiveRiskManager
except ImportError:
    logging.warning("adaptive_risk_manager could not be imported")
    AdaptiveRiskManager = None

# Order Book Modul-Importe
try:
    from orderbook_anomaly_detection import OrderBookAnomalyDetector, OrderBookChangeDetector
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    logging.warning("orderbook_anomaly_detection could not be imported")
    OrderBookAnomalyDetector = None
    OrderBookChangeDetector = None
    ANOMALY_DETECTION_AVAILABLE = False

try:
    from synthetic_order_book import SyntheticOrderBookGenerator, OrderBookFeatureProcessor
    ORDER_BOOK_AVAILABLE = True
except ImportError:
    logging.warning("synthetic_order_book could not be imported")
    SyntheticOrderBookGenerator = None
    OrderBookFeatureProcessor = None
    ORDER_BOOK_AVAILABLE = False

# Adaptive Signal Weighting
try:
    from adaptive_signal_weighting import AdaptiveSignalWeighting, AdaptiveSignalProcessor, DomainAdaptationLayer
    SIGNAL_WEIGHTING_AVAILABLE = True
except ImportError:
    logging.warning("adaptive_signal_weighting could not be imported")
    AdaptiveSignalWeighting = None
    AdaptiveSignalProcessor = None
    DomainAdaptationLayer = None
    SIGNAL_WEIGHTING_AVAILABLE = False

# Ensemble Components
try:
    from ensemble_components import AdaptiveComponentEnsemble, FusionBacktester
    ENSEMBLE_AVAILABLE = True
except ImportError:
    logging.warning("ensemble_components could not be imported")
    AdaptiveComponentEnsemble = None
    FusionBacktester = None
    ENSEMBLE_AVAILABLE = False

# Feature Fusion Modul-Import
try:
    from regularized_feature_fusion import RegularizedFeatureFusion, EnhancedFeatureFusionEnsemble
    FEATURE_FUSION_AVAILABLE = True
except ImportError:
    logging.warning("regularized_feature_fusion could not be imported")
    RegularizedFeatureFusion = None
    EnhancedFeatureFusionEnsemble = None
    FEATURE_FUSION_AVAILABLE = False

# GPU-Beschleunigung aktivieren
try:
    from gpu_accelerator import accelerate_hmm_functions, check_gpu_acceleration
    
    # GPU-Status prüfen und anzeigen
    gpu_info = check_gpu_acceleration()
    logging.info(f"GPU-Beschleunigung: {'Verfügbar' if gpu_info['gpu_available'] else 'Nicht verfügbar'}")
    
    # HMM-Funktionen durch GPU-beschleunigte Versionen ersetzen
    if gpu_info["gpu_available"]:
        accelerate_hmm_functions()
        logging.info("HMM-Funktionen mit GPU-Beschleunigung aktiviert")
except ImportError:
    logging.warning("GPU-Beschleunigungsmodul nicht gefunden. CPU wird verwendet.")

###############################################################################
# CONFIGURATION
###############################################################################

# MT5 Configuration
MT5_LOGIN    = 12345678            # Optional
MT5_PASSWORD = "YourPassword"      # Optional
MT5_SERVER   = "YourBroker-Server" # Optional

# Trading Configuration
SYMBOL       = "GBPJPY"
TIMEFRAME_30M= mt5.TIMEFRAME_M30
TIMEFRAME_5M = mt5.TIMEFRAME_M5
TIMEFRAME_1H = mt5.TIMEFRAME_H1
TIMEFRAME_4H = mt5.TIMEFRAME_H4
TIMEFRAME_1M = mt5.TIMEFRAME_M1  # For price action entries

# Program Configuration
CHECK_INTERVAL_SEC = 30
MAX_CONNECTION_ERRORS = 5  # Maximum connection errors before termination
DASH_PORT = 8050          # Port for the Dash dashboard

# File Paths
LAST_CANDLE_FILE = "last_candle_time.txt"
MODEL_FILE = "enhanced_model_full/enhanced_hmm_model.pkl"
MODEL_FALLBACK_FILE = "enhanced_model_full/enhanced_hmm_model.pkl"
PERFORMANCE_FILE = "hmm_performance_tracker.json"
TRADE_HISTORY_FILE = "trade_history.json"
STATE_HISTORY_FILE = "state_history.json"
REPORT_OUTPUT_DIR = "hmm_reports"  # Directory for regular reports
LOG_DIR = "logs"

# Historical Candles per Timeframe
COUNT_30M_LOOP = 50
COUNT_5M_LOOP  = 150
COUNT_1H_LOOP  = 30
COUNT_4H_LOOP  = 20
COUNT_1M_LOOP  = 200  # For price action analysis

# Model Configuration
DIMS_EGARCH = [0, 1, 2, 3]  # Indices of dimensions using EGARCH
STATE_HISTORY_WINDOW = 100  # Window for state evaluation

# Advanced Features Configuration
USE_MARKET_MEMORY = True       # Use pattern matching from market memory
ORDER_BOOK_MAX_LEVELS = 10 
USE_ORDER_BOOK = True          # Use order book analysis
USE_HYBRID_MODEL = True        # Use hybrid HMM+NN model
USE_FEATURE_SELECTION = True   # Use dynamic feature selection
USE_CROSS_ASSETS = True        # Use cross-asset features
CROSS_ASSET_CORR_WINDOW = 100       # Correlation window size 
CROSS_ASSET_LEAD_LAG_MAX = 10       # Maximum lead/lag shift in bars
CROSS_ASSET_UPDATE_INTERVAL = 6     # Hours between correlation updates
CROSS_ASSET_DATA_DIR = "data/cross_assets"
HYBRID_CONTRIBUTION_WEIGHT = 0.3  # How much the hybrid model contributes to decisions
# Order Book Konfiguration
USE_ORDER_BOOK = True          # Order Book-Analyse aktivieren
USE_SYNTHETIC_ORDER_BOOK = True  # Synthetisches Order Book bei fehlendem echten Order Book verwenden
ORDER_BOOK_MAX_LEVELS = 10     # Maximale Anzahl von Preislevels im Order Book
ORDER_BOOK_UPDATE_INTERVAL = 15  # Aktualisierungsintervall für Order Book in Sekunden
ORDER_BOOK_MODEL_PATH = "models/ob_anomaly_detector.pkl"  # Pfad zum gespeicherten Anomalie-Detektor
ORDER_BOOK_CALIBRATION_SIZE = 100  # Anzahl der Order Books für die Kalibrierung
ORDER_BOOK_HISTORY_SIZE = 100  # Größe des Order Book Historie-Puffers
USE_FEATURE_FUSION = True      # Use feature fusion for integrated features

# Feature Fusion Configuration
FEATURE_FUSION_METHOD = 'attention'  # 'concat', 'attention', 'weighted', 'autoencoder'
FEATURE_FUSION_REGULARIZATION = 'elastic'  # 'l1', 'l2', 'elastic'
FEATURE_FUSION_ADAPTIVE = True  # Adaptive weights
FEATURE_FUSION_MODEL_PATH = "enhanced_model_full/feature_fusion.pkl"
FEATURE_FUSION_RETRAIN_INTERVAL = 86400  # 24 hours in seconds

# Signal Weighting Configuration
USE_SIGNAL_WEIGHTING = True    # Use adaptive signal weighting
SIGNAL_WEIGHTING_LEARNING_RATE = 0.05
SIGNAL_WEIGHTING_STATE_SPECIFIC = True
SIGNAL_WEIGHTING_MIN_WEIGHT = 0.1
SIGNAL_WEIGHTING_FILE = "models/adaptive_weights.json"

# Ensemble Configuration
USE_ENSEMBLE = True            # Use ensemble modeling
ENSEMBLE_COMPONENTS = ['hmm', 'hybrid_model', 'market_memory', 'order_book']
ENSEMBLE_HISTORY_SIZE = 500
ENSEMBLE_MODEL_PATH = "enhanced_model_full/ensemble_components.pkl"

# Set up logging with file rotation
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/enhanced_hmm_live_v3_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Ensure report directory exists
if not os.path.exists(REPORT_OUTPUT_DIR):
    os.makedirs(REPORT_OUTPUT_DIR)

###############################################################################
# Data Management and MT5 Helper Functions
###############################################################################

def get_mt5_data_with_retry(symbol, timeframe, count, max_retries=3, delay_seconds=2):
    """
    Robust MT5 data query with retry on errors.
    
    Args:
        symbol: Symbol to query data for
        timeframe: Timeframe (e.g. mt5.TIMEFRAME_30M)
        count: Number of candles to request
        max_retries: Maximum number of retry attempts
        delay_seconds: Delay between retry attempts in seconds
    
    Returns:
        numpy.array with candle data or None on error
    """
    for attempt in range(max_retries):
        try:
            data = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if data is not None and len(data) >= count * 0.9:  # At least 90% of requested data
                return data
            logging.warning(f"Attempt {attempt+1}/{max_retries}: Incomplete data for {symbol}, {len(data) if data is not None else 0}/{count} candles")
            time.sleep(delay_seconds)
        except Exception as e:
            logging.error(f"MT5 error fetching {symbol}: {str(e)}")
            time.sleep(delay_seconds)
    
    # Last attempt - accept incomplete data
    try:
        return mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    except Exception as e:
        logging.error(f"Final MT5 error: {str(e)}")
        return None

def get_mt5_rates_range_with_retry(symbol, timeframe, start_date, end_date, max_retries=3, delay_seconds=2):
    """
    Robust MT5 data query for a time range with retry on errors.
    
    Args:
        symbol: Symbol to query data for
        timeframe: Timeframe (e.g. mt5.TIMEFRAME_30M)
        start_date: Start date (datetime)
        end_date: End date (datetime)
        max_retries: Maximum number of retry attempts
        delay_seconds: Delay between retry attempts in seconds
    
    Returns:
        numpy.array with candle data or None on error
    """
    for attempt in range(max_retries):
        try:
            data = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            if data is not None and len(data) > 0:  # At least some data
                return data
            logging.warning(f"Attempt {attempt+1}/{max_retries}: Incomplete data for {symbol} from {start_date} to {end_date}")
            time.sleep(delay_seconds)
        except Exception as e:
            logging.error(f"MT5 error fetching {symbol}: {str(e)}")
            time.sleep(delay_seconds)
    
    # Last attempt
    try:
        return mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    except Exception as e:
        logging.error(f"Final MT5 error: {str(e)}")
        return None

def check_mt5_connection():
    """
    Check and try to reestablish MT5 connection.
    
    Returns:
        bool: True if connected, False otherwise
    """
    if not mt5.terminal_info():
        logging.warning("MT5 connection lost. Attempting reconnection...")
        mt5.shutdown()
        time.sleep(1)
        if not mt5.initialize():
            logging.error("MT5 reconnection failed.")
            return False
        logging.info("MT5 successfully reconnected.")
    return True

def get_last_sunday_23h():
    """
    Returns timestamp of last Sunday at 23:00.
    """
    now = datetime.now()
    offset_days = (now.weekday() - 6) % 7
    last_sunday = now - timedelta(days=offset_days)
    last_sunday_23 = datetime(last_sunday.year, last_sunday.month, last_sunday.day, 23, 0)
    if last_sunday_23 > now:
        last_sunday_23 -= timedelta(days=7)
    return last_sunday_23

def get_order_book_data(symbol=SYMBOL, max_depth=ORDER_BOOK_MAX_LEVELS):
    """
    Erweiterte Funktion zum Abrufen von Order Book Daten von MT5.
    
    Args:
        symbol: Handelssymbol
        max_depth: Maximale Tiefe des Order Books
    
    Returns:
        dict: Order Book Daten oder None bei Fehler
    """
    try:
        # Get order book from MT5
        book = mt5.market_book_get(symbol)
        
        if book is None or len(book) < 2:
            logging.warning(f"No order book data available for {symbol}")
            return None
        
        # Konvertiere zu besserem Format
        bids = []
        asks = []
        
        for item in book:
            if item.type == mt5.BOOK_TYPE_SELL:
                asks.append({"price": item.price, "volume": item.volume})
            elif item.type == mt5.BOOK_TYPE_BUY:
                bids.append({"price": item.price, "volume": item.volume})
        
        # Sortiere Bids (absteigend) und Asks (aufsteigend)
        bids.sort(key=lambda x: x["price"], reverse=True)
        asks.sort(key=lambda x: x["price"])
        
        # Begrenze auf angegebene Tiefe
        bids = bids[:max_depth]
        asks = asks[:max_depth]
        
        # Erweiterte Metriken berechnen
        spread = asks[0]["price"] - bids[0]["price"] if bids and asks else 0
        total_bid_volume = sum(bid["volume"] for bid in bids) if bids else 0
        total_ask_volume = sum(ask["volume"] for ask in asks) if asks else 0
        imbalance = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1.0
        
        return {
            "bids": bids,
            "asks": asks,
            "mid_price": (bids[0]["price"] + asks[0]["price"]) / 2 if bids and asks else None,
            "spread": spread,
            "spread_pips": spread / 0.01,  # Umrechnung in Pips, angepasst an Symbol
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "imbalance": imbalance,
            "timestamp": datetime.now(),
            "symbol": symbol
        }
    except Exception as e:
        logging.error(f"Error getting order book data: {str(e)}")
        return None

def generate_synthetic_order_book(generator, current_price, market_state=None):
    """
    Generiert ein synthetisches Order Book, wenn echtes Order Book nicht verfügbar ist.
    
    Args:
        generator: SyntheticOrderBookGenerator-Instanz
        current_price: Aktueller Preis
        market_state: Optionaler Marktzustand (bullish, bearish, etc.)
    
    Returns:
        dict: Synthetisches Order Book oder None bei Fehler
    """
    if generator is None:
        return None
        
    try:
        # Check if the generator has the expected method
        if not hasattr(generator, 'generate_order_book'):
            logging.error("SyntheticOrderBookGenerator instance has no generate_order_book method")
            return None
            
        # Get the method signature to check for accepted parameters
        import inspect
        params = inspect.signature(generator.generate_order_book).parameters
        
        # Create kwargs based on available parameters
        kwargs = {'current_price': current_price}
        
        if 'market_state' in params and market_state is not None:
            kwargs['market_state'] = market_state
            
        # Generiere synthetisches Order Book with supported parameters only
        synthetic_book = generator.generate_order_book(**kwargs)
        
        # Markiere als synthetisch
        if synthetic_book:
            synthetic_book["is_synthetic"] = True
        
        return synthetic_book
    except Exception as e:
        logging.error(f"Error generating synthetic order book: {str(e)}")
        return None

###############################################################################
# Feature Calculation and Indicators
###############################################################################

def compute_rsi(series, window=14):
    """Calculate Relative Strength Index"""
    return ta.momentum.rsi(series, window=window)

def compute_atr(df, window=14):
    """Calculate Average True Range"""
    return ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=window)

def compute_macd_diff(series, fast=12, slow=26, signal=9):
    """Calculate MACD difference line"""
    return ta.trend.macd_diff(series, window_slow=slow, window_fast=fast, window_sign=signal)

def add_session_flags(df):
    """
    Add flags for trading sessions:
    - Asia: 00:00-08:00 UTC
    - Europe: 07:00-16:00 UTC
    - US: 13:00-22:00 UTC
    - Overlap: Session overlaps
    """
    # Convert timezones to UTC
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
    
    # Remove hour_utc column (no longer needed)
    df.drop('hour_utc', axis=1, inplace=True)
    
    return df

def add_weekday_flags(df):
    """
    Add weekday flags (Monday to Friday)
    """
    # Weekdays (0 = Monday, 4 = Friday)
    day_of_week = df['time'].dt.dayofweek
    
    # One-hot encoding for weekdays
    df['day_mon'] = (day_of_week == 0).astype(int)
    df['day_tue'] = (day_of_week == 1).astype(int)
    df['day_wed'] = (day_of_week == 2).astype(int)
    df['day_thu'] = (day_of_week == 3).astype(int)
    df['day_fri'] = (day_of_week == 4).astype(int)
    
    return df

def compute_features_from_mt5_data(rates_30m, rates_5m, rates_1h=None, rates_4h=None,
                                 apply_pca=False, pca_model=None, scaler=None):
    """
    Calculate enhanced features from MT5 data with optional PCA transformation.
    """
    # Convert to DataFrames
    df_30 = pd.DataFrame(rates_30m)
    df_5 = pd.DataFrame(rates_5m)
    
    # Convert time
    df_30['time'] = pd.to_datetime(df_30['time'], unit='s')
    df_5['time'] = pd.to_datetime(df_5['time'], unit='s')
    
    # Sort
    df_30.sort_values('time', inplace=True)
    df_5.sort_values('time', inplace=True)
    df_30.reset_index(drop=True, inplace=True)
    df_5.reset_index(drop=True, inplace=True)
    
    # 1h and 4h DataFrames, if available
    df_1h = None
    if rates_1h is not None and len(rates_1h) > 0:
        df_1h = pd.DataFrame(rates_1h)
        df_1h['time'] = pd.to_datetime(df_1h['time'], unit='s')
        df_1h.sort_values('time', inplace=True)
        df_1h.reset_index(drop=True, inplace=True)
    
    df_4h = None
    if rates_4h is not None and len(rates_4h) > 0:
        df_4h = pd.DataFrame(rates_4h)
        df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s')
        df_4h.sort_values('time', inplace=True)
        df_4h.reset_index(drop=True, inplace=True)
    
    # 30min Features
    df_30['log_return30'] = np.log(df_30['close']/df_30['close'].shift(1))
    df_30['log_return30'].fillna(0, inplace=True)
    
    # Log Volume - with support for tick_volume
    if 'tick_volume' in df_30.columns:
        df_30['log_volume'] = np.log1p(df_30['tick_volume'])
    elif 'volume' in df_30.columns:
        df_30['log_volume'] = np.log1p(df_30['volume'])
    else:
        df_30['log_volume'] = 0
    
    # RSI (30min)
    df_30['rsi_30m'] = compute_rsi(df_30['close'], window=14)
    
    # ATR (30min)
    df_30['atr_30m'] = compute_atr(df_30, window=14)
    
    # MACD-Diff (30min)
    df_30['macd_30m'] = compute_macd_diff(df_30['close'], fast=12, slow=26, signal=9)
    
    # Session and weekday flags
    df_30 = add_session_flags(df_30)
    df_30 = add_weekday_flags(df_30)
    
    # 5min Features
    df_5['log_return5'] = np.log(df_5['close']/df_5['close'].shift(1))
    df_5['log_return5'].fillna(0, inplace=True)
    
    # Merge 5min to 30min
    df_30 = pd.merge_asof(
        df_30.sort_values('time'),
        df_5[['time','log_return5']].sort_values('time'),
        on='time', direction='backward'
    )
    
    # 1h Features, if available
    if df_1h is not None:
        df_1h['log_return1h'] = np.log(df_1h['close']/df_1h['close'].shift(1))
        df_1h['log_return1h'].fillna(0, inplace=True)
        df_1h['rsi_1h'] = compute_rsi(df_1h['close'], window=14)
        
        df_30 = pd.merge_asof(
            df_30.sort_values('time'),
            df_1h[['time','log_return1h','rsi_1h']].sort_values('time'),
            on='time', direction='backward'
        )
    else:
        df_30['log_return1h'] = df_30['log_return30']
        df_30['rsi_1h'] = df_30['rsi_30m']
    
    # 4h Features, if available
    if df_4h is not None:
        df_4h['log_return4h'] = np.log(df_4h['close']/df_4h['close'].shift(1))
        df_4h['log_return4h'].fillna(0, inplace=True)
        df_4h['atr_4h'] = compute_atr(df_4h, window=14)
        
        df_30 = pd.merge_asof(
            df_30.sort_values('time'),
            df_4h[['time','log_return4h','atr_4h']].sort_values('time'),
            on='time', direction='backward'
        )
    else:
        df_30['log_return4h'] = df_30['log_return30']
        df_30['atr_4h'] = df_30['atr_30m']
    
    # Define feature columns
    feature_cols = [
        'log_return30', 'log_return5', 'log_return1h', 'log_return4h',
        'rsi_30m', 'rsi_1h', 'atr_30m', 'atr_4h', 'macd_30m',
        'session_asia', 'session_europe', 'session_us', 'session_overlap',
        'log_volume',
        'day_mon', 'day_tue', 'day_wed', 'day_thu', 'day_fri'
    ]
    
    # Clean NaN values
    for col in feature_cols:
        if col in df_30.columns:
            df_30[col].fillna(0, inplace=True)
        else:
            df_30[col] = 0
    
    df_30.dropna(subset=feature_cols, inplace=True)
    df_30.reset_index(drop=True, inplace=True)
    
    # Optional: Apply PCA if model provided
    if apply_pca and pca_model is not None and scaler is not None:
        try:
            # Extract features
            features_matrix = df_30[feature_cols].values
            
            # Scale features
            scaled_features = scaler.transform(features_matrix)
            
            # Apply PCA
            reduced_features = pca_model.transform(scaled_features)
            
            # Add PCA features to DataFrame
            for i in range(reduced_features.shape[1]):
                df_30[f'pca_feature_{i}'] = reduced_features[:, i]
                
            logging.info(f"PCA successfully applied: {reduced_features.shape[1]} components")
        except Exception as e:
            logging.error(f"Error applying PCA: {str(e)}")

    # Stelle finale Dimensionskonsistenz sicher
    if feature_cols:
        expected_dimension = len(feature_cols)
        for col in feature_cols:
            if col not in df_30.columns:
                df_30[col] = 0
    
        # Prüfe, ob alle erforderlichen Feature-Spalten existieren
        missing_cols = [col for col in feature_cols if col not in df_30.columns]
        if missing_cols:
            logging.warning(f"Fehlende Feature-Spalten in compute_features_from_mt5_data: {missing_cols}")
    
    return df_30

def analyze_price_action_1min(rates_1m, state_label, min_lookback=30):
    """
    Analyzes 1-minute chart for advanced price action patterns.
    
    Args:
        rates_1m: 1-minute candles
        state_label: Label of current HMM state
        min_lookback: Minimum number of candles for analysis
    
    Returns:
        dict: Identified price action patterns and potential entry signals
    """
    if rates_1m is None or len(rates_1m) < min_lookback:
        return {"signals": [], "patterns": []}
    
    # Convert to DataFrame
    df_1m = pd.DataFrame(rates_1m)
    df_1m['time'] = pd.to_datetime(df_1m['time'], unit='s')
    
    # Sort
    df_1m.sort_values('time', inplace=True)
    df_1m.reset_index(drop=True, inplace=True)
    
    # Technical indicators
    df_1m['rsi'] = ta.momentum.rsi(df_1m['close'], window=14)
    df_1m['atr'] = ta.volatility.average_true_range(df_1m['high'], df_1m['low'], df_1m['close'], window=14)
    
    # Moving averages
    df_1m['sma20'] = df_1m['close'].rolling(window=20).mean()
    df_1m['sma50'] = df_1m['close'].rolling(window=50).mean()
    df_1m['ema9'] = ta.trend.ema_indicator(df_1m['close'], window=9)
    
    # Bollinger Bands
    indicator_bb = ta.volatility.BollingerBands(df_1m['close'], window=20, window_dev=2)
    df_1m['bb_upper'] = indicator_bb.bollinger_hband()
    df_1m['bb_lower'] = indicator_bb.bollinger_lband()
    df_1m['bb_mid'] = indicator_bb.bollinger_mavg()
    
    # MACD
    df_1m['macd'] = ta.trend.macd(df_1m['close'])
    df_1m['macd_signal'] = ta.trend.macd_signal(df_1m['close'])
    df_1m['macd_diff'] = ta.trend.macd_diff(df_1m['close'])
    
    # Calculate candle properties
    df_1m['body_size'] = abs(df_1m['close'] - df_1m['open'])
    df_1m['upper_wick'] = df_1m.apply(lambda x: x['high'] - max(x['open'], x['close']), axis=1)
    df_1m['lower_wick'] = df_1m.apply(lambda x: min(x['open'], x['close']) - x['low'], axis=1)
    df_1m['range'] = df_1m['high'] - df_1m['low']
    df_1m['body_to_range'] = df_1m['body_size'] / df_1m['range'].where(df_1m['range'] != 0, 1)
    
    # Calculate body position relative to range
    df_1m['body_position'] = df_1m.apply(
        lambda x: ((max(x['open'], x['close']) + min(x['open'], x['close'])) / 2 - x['low']) / 
                 (x['high'] - x['low']) if (x['high'] - x['low']) > 0 else 0.5, 
        axis=1
    )
    
    # Candle directions
    df_1m['is_bullish'] = (df_1m['close'] > df_1m['open']).astype(int)
    df_1m['is_bearish'] = (df_1m['close'] < df_1m['open']).astype(int)
    
    # Market direction from state label
    is_bullish = "Bullish" in state_label
    
    # Lists for detected patterns
    patterns = []
    signals = []
    
    # Advanced Pattern Detection
    for i in range(min_lookback, len(df_1m)):
        current = df_1m.iloc[i]
        prev1 = df_1m.iloc[i-1]
        prev2 = df_1m.iloc[i-2]
        prev3 = df_1m.iloc[i-3] if i >= 3 else None
        
        # 1. ADVANCED BULLISH PATTERNS
        
        # 1.1 Bullish Engulfing with Volume Confirmation
        if (is_bullish and 
            prev1['is_bearish'] == 1 and  # Previous candle bearish
            current['is_bullish'] == 1 and  # Current candle bullish
            current['open'] < prev1['close'] and  # Opens below previous close
            current['close'] > prev1['open'] and  # Closes above previous open
            current['tick_volume'] > prev1['tick_volume'] * 1.2):  # 20% more volume
            
            patterns.append({
                "type": "Bullish Engulfing with Volume",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.85  # Higher confidence with volume confirmation
            })
            
            if is_bullish:  # Only generate signal if HMM also bullish
                signals.append({
                    "type": "LONG",
                    "pattern": "Bullish Engulfing with Volume",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.85
                })
        
        # 1.2 Morning Star (Three-candle bullish reversal)
        if (is_bullish and prev3 is not None and
            prev3['is_bearish'] == 1 and  # First candle bearish
            abs(prev2['body_size']) < prev3['body_size'] * 0.5 and  # Second candle small
            current['is_bullish'] == 1 and  # Third candle bullish
            current['close'] > (prev3['open'] + prev3['close']) / 2):  # Closes above midpoint of first candle
            
            patterns.append({
                "type": "Morning Star",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.9  # Strong pattern
            })
            
            if is_bullish:
                signals.append({
                    "type": "LONG",
                    "pattern": "Morning Star",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.9
                })
        
        # 1.3 Double Bottom with RSI Divergence
        if (is_bullish and prev3 is not None and
            abs(prev3['low'] - prev1['low']) / prev1['low'] < 0.001 and  # Similar lows
            prev2['low'] > prev1['low'] and  # Higher low between
            current['close'] > prev1['high'] and  # Breakout
            df_1m.iloc[i-3]['rsi'] < 30 and  # First bottom RSI oversold
            df_1m.iloc[i-1]['rsi'] > df_1m.iloc[i-3]['rsi']):  # RSI making higher low
            
            patterns.append({
                "type": "Double Bottom with RSI Divergence",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.95  # Very strong pattern
            })
            
            if is_bullish:
                signals.append({
                    "type": "LONG",
                    "pattern": "Double Bottom with RSI Divergence",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.95
                })
        
        # 1.4 Bullish MACD Crossover with EMA Support
        if (is_bullish and 
            prev1['macd'] < prev1['macd_signal'] and  # Previous MACD below signal
            current['macd'] > current['macd_signal'] and  # Current MACD crossed above signal
            current['close'] > current['ema9'] and  # Price above EMA9
            current['ema9'] > current['sma20']):  # EMA9 above SMA20
            
            patterns.append({
                "type": "Bullish MACD Crossover with EMA Support",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.8
            })
            
            if is_bullish:
                signals.append({
                    "type": "LONG",
                    "pattern": "Bullish MACD Crossover with EMA Support",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.8
                })
                
        # 1.5 Bollinger Band Bounce from Lower Band
        if (is_bullish and
            prev1['low'] < prev1['bb_lower'] and  # Previous candle touched lower band
            current['close'] > current['open'] and  # Current candle bullish
            current['close'] > prev1['close'] and  # Higher close
            current['rsi'] > 30):  # RSI recovering from oversold
            
            patterns.append({
                "type": "Bollinger Band Bounce",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.85
            })
            
            if is_bullish:
                signals.append({
                    "type": "LONG",
                    "pattern": "Bollinger Band Bounce",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.85
                })
        
        # 2. ADVANCED BEARISH PATTERNS
        
        # 2.1 Bearish Engulfing with Volume Confirmation
        if (not is_bullish and 
            prev1['is_bullish'] == 1 and  # Previous candle bullish
            current['is_bearish'] == 1 and  # Current candle bearish
            current['open'] > prev1['close'] and  # Opens above previous close
            current['close'] < prev1['open'] and  # Closes below previous open
            current['tick_volume'] > prev1['tick_volume'] * 1.2):  # 20% more volume
            
            patterns.append({
                "type": "Bearish Engulfing with Volume",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.85
            })
            
            if not is_bullish:  # Only generate signal if HMM also bearish
                signals.append({
                    "type": "SHORT",
                    "pattern": "Bearish Engulfing with Volume",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.85
                })
        
        # 2.2 Evening Star (Three-candle bearish reversal)
        if (not is_bullish and prev3 is not None and
            prev3['is_bullish'] == 1 and  # First candle bullish
            abs(prev2['body_size']) < prev3['body_size'] * 0.5 and  # Second candle small
            current['is_bearish'] == 1 and  # Third candle bearish
            current['close'] < (prev3['open'] + prev3['close']) / 2):  # Closes below midpoint of first candle
            
            patterns.append({
                "type": "Evening Star",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.9
            })
            
            if not is_bullish:
                signals.append({
                    "type": "SHORT",
                    "pattern": "Evening Star",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.9
                })
        
        # 2.3 Double Top with RSI Divergence
        if (not is_bullish and prev3 is not None and
            abs(prev3['high'] - prev1['high']) / prev1['high'] < 0.001 and  # Similar highs
            prev2['high'] < prev1['high'] and  # Lower high between
            current['close'] < prev1['low'] and  # Breakdown
            df_1m.iloc[i-3]['rsi'] > 70 and  # First top RSI overbought
            df_1m.iloc[i-1]['rsi'] < df_1m.iloc[i-3]['rsi']):  # RSI making lower high
            
            patterns.append({
                "type": "Double Top with RSI Divergence",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.95
            })
            
            if not is_bullish:
                signals.append({
                    "type": "SHORT",
                    "pattern": "Double Top with RSI Divergence",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.95
                })
        
        # 2.4 Bearish MACD Crossover below EMA
        if (not is_bullish and 
            prev1['macd'] > prev1['macd_signal'] and  # Previous MACD above signal
            current['macd'] < current['macd_signal'] and  # Current MACD crossed below signal
            current['close'] < current['ema9'] and  # Price below EMA9
            current['ema9'] < current['sma20']):  # EMA9 below SMA20
            
            patterns.append({
                "type": "Bearish MACD Crossover below EMA",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.8
            })
            
            if not is_bullish:
                signals.append({
                    "type": "SHORT",
                    "pattern": "Bearish MACD Crossover below EMA",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.8
                })
        
        # 2.5 Bollinger Band Bounce from Upper Band
        if (not is_bullish and
            prev1['high'] > prev1['bb_upper'] and  # Previous candle touched upper band
            current['close'] < current['open'] and  # Current candle bearish
            current['close'] < prev1['close'] and  # Lower close
            current['rsi'] < 70):  # RSI coming from overbought
            
            patterns.append({
                "type": "Bollinger Band Upper Rejection",
                "time": current['time'].isoformat(),
                "price": current['close'],
                "strength": 0.85
            })
            
            if not is_bullish:
                signals.append({
                    "type": "SHORT",
                    "pattern": "Bollinger Band Upper Rejection",
                    "time": current['time'].isoformat(),
                    "price": current['close'],
                    "strength": 0.85
                })
                
        # 3. MULTI-TIMEFRAME CONFLUENCE PATTERNS
        
        # Add volume profile analysis
        # We'll focus on the recent volume distribution to identify high-volume nodes
        recent_df = df_1m.iloc[max(0, i-100):i+1]  # Look at last 100 candles
        
        # Create simplified volume profile
        if 'tick_volume' in recent_df.columns:
            # Create price bins
            price_range = recent_df['high'].max() - recent_df['low'].min()
            bin_size = price_range / 10  # 10 bins
            
            # Group by price levels and sum volumes
            recent_df['price_bin'] = ((recent_df['close'] - recent_df['low'].min()) / bin_size).astype(int)
            volume_profile = recent_df.groupby('price_bin')['tick_volume'].sum()
            
            # Find high volume nodes (HVN)
            if not volume_profile.empty:
                max_volume_bin = volume_profile.idxmax()
                hvn_price_level = recent_df['low'].min() + (max_volume_bin + 0.5) * bin_size
                
                # Check if current price is near HVN
                if abs(current['close'] - hvn_price_level) / current['close'] < 0.001:
                    # Price at high volume node
                    patterns.append({
                        "type": "High Volume Node",
                        "time": current['time'].isoformat(),
                        "price": current['close'],
                        "strength": 0.75,
                        "hvn_level": hvn_price_level
                    })
                    
                    # If price is bouncing from HVN in the direction of our state
                    bounce_up = current['close'] > current['open'] and is_bullish
                    bounce_down = current['close'] < current['open'] and not is_bullish
                    
                    if bounce_up or bounce_down:
                        signal_type = "LONG" if bounce_up else "SHORT"
                        signals.append({
                            "type": signal_type,
                            "pattern": "HVN Bounce",
                            "time": current['time'].isoformat(),
                            "price": current['close'],
                            "strength": 0.8
                        })
    
    return {
        "signals": signals,
        "patterns": patterns
    }

def ensure_feature_dimensions(features, expected_dim):
    """
    Ensures that feature vectors have consistent dimensions.
    
    Args:
        features: Feature vector or matrix
        expected_dim: Expected number of dimensions
        
    Returns:
        numpy.ndarray: Feature vector with consistent dimensions
    """
    import numpy as np
    
    # Handle None input
    if features is None:
        return np.zeros(expected_dim)
    
    # Convert to numpy array if not already
    if not isinstance(features, np.ndarray):
        try:
            features = np.array(features)
        except:
            logging.error("Could not convert features to numpy array")
            return np.zeros(expected_dim)
    
    # Ensure proper shape
    if len(features.shape) == 0:  # Scalar
        return np.zeros(expected_dim)
    
    if len(features.shape) == 1:  # Vector
        if features.shape[0] > expected_dim:
            # Too many features - truncate
            return features[:expected_dim]
        elif features.shape[0] < expected_dim:
            # Too few features - pad with zeros
            result = np.zeros(expected_dim)
            result[:features.shape[0]] = features
            return result
        else:
            # Correct dimension
            return features
    
    if len(features.shape) == 2:  # Matrix
        # If single sample in matrix form, convert to vector
        if features.shape[0] == 1:
            return ensure_feature_dimensions(features[0], expected_dim)
        
        # If multiple samples, ensure each has correct dimension
        result = np.zeros((features.shape[0], expected_dim))
        for i in range(features.shape[0]):
            sample = features[i]
            if len(sample) > expected_dim:
                result[i, :] = sample[:expected_dim]
            else:
                result[i, :len(sample)] = sample
        return result
    
    # Higher dimensions not supported
    logging.error(f"Unsupported feature shape: {features.shape}")
    return np.zeros(expected_dim)

###############################################################################
# Performance Tracking Classes
###############################################################################

class PerformanceTracker:
    """
    Tracks and analyzes HMM performance across different states.
    """
    def __init__(self, K, load_file=None):
        """
        Initialize the tracker.
        
        Args:
            K: Number of states
            load_file: Optional, file to load existing data from
        """
        self.K = K
        self.state_trades = {i: [] for i in range(K)}
        self.state_metrics = {i: {} for i in range(K)}
        self.state_transitions = {}  # Stores statistics for state transitions
        self.validation_history = []  # Stores state validity values over time
        self.created_timestamp = datetime.now().isoformat()
        self.last_updated = self.created_timestamp
        
        # Load existing data if available
        if load_file and os.path.exists(load_file):
            self.load_from_file(load_file)
    
    def add_trade(self, state, entry_price, exit_price, direction, duration, 
                  profit_pips, entry_time=None, exit_time=None):
        """
        Add a trade to performance tracking.
        """
        pnl = exit_price - entry_price if direction == "LONG" else entry_price - exit_price
        pnl_pct = pnl / entry_price
        
        trade_data = {
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "direction": direction,
            "duration": duration,
            "profit_pips": profit_pips,
            "entry_price": entry_price,
            "exit_price": exit_price
        }
        
        if entry_time:
            trade_data["entry_time"] = entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time
        
        if exit_time:
            trade_data["exit_time"] = exit_time.isoformat() if isinstance(exit_time, datetime) else exit_time
        
        self.state_trades[state].append(trade_data)
        self.last_updated = datetime.now().isoformat()
        
        # Update metrics after each new trade
        self.update_metrics()
    
    def add_validation_score(self, state, score, time_point=None):
        """
        Add a validity score for a state.
        """
        entry = {
            "state": state,
            "score": score,
            "time": datetime.now().isoformat() if time_point is None else time_point.isoformat() if isinstance(time_point, datetime) else time_point
        }
        self.validation_history.append(entry)
        self.last_updated = datetime.now().isoformat()
        
        # Limit history to a reasonable size
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
    
    def log_state_transition(self, from_state, to_state, transition_prob, time_point=None):
        """
        Log a state transition for analysis.
        """
        key = f"{from_state}->{to_state}"
        time_str = datetime.now().isoformat() if time_point is None else time_point.isoformat() if isinstance(time_point, datetime) else time_point
        
        if key not in self.state_transitions:
            self.state_transitions[key] = {
                "count": 0,
                "probs": [],
                "times": []
            }
        
        self.state_transitions[key]["count"] += 1
        self.state_transitions[key]["probs"].append(transition_prob)
        self.state_transitions[key]["times"].append(time_str)
        self.last_updated = datetime.now().isoformat()
        
        # Limit history sizes
        max_history = 500
        if len(self.state_transitions[key]["probs"]) > max_history:
            self.state_transitions[key]["probs"] = self.state_transitions[key]["probs"][-max_history:]
            self.state_transitions[key]["times"] = self.state_transitions[key]["times"][-max_history:]
    
    def update_metrics(self):
        """
        Update all performance metrics.
        """
        for state in range(self.K):
            trades = self.state_trades[state]
            if not trades:
                continue
            
            pnls = [t["pnl"] for t in trades]
            pnl_pcts = [t["pnl_pct"] for t in trades]
            durations = [t["duration"] for t in trades]
            profit_pips = [t.get("profit_pips", 0) for t in trades]
            
            win_count = sum(1 for pnl in pnls if pnl > 0)
            loss_count = len(pnls) - win_count
            
            # Average wins/losses
            avg_win = np.mean([pnl for pnl in pnls if pnl > 0]) if win_count > 0 else 0
            avg_loss = np.mean([abs(pnl) for pnl in pnls if pnl <= 0]) if loss_count > 0 else 0
            
            self.state_metrics[state] = {
                "win_rate": win_count / len(trades) if trades else 0,
                "avg_pnl": np.mean(pnls) if pnls else 0,
                "avg_pnl_pct": np.mean(pnl_pcts) if pnl_pcts else 0,
                "sharpe": np.mean(pnl_pcts) / np.std(pnl_pcts) if pnl_pcts and np.std(pnl_pcts) > 0 else 0,
                "avg_duration": np.mean(durations) if durations else 0,
                "trade_count": len(trades),
                "profit_factor": avg_win / avg_loss if avg_loss > 0 and avg_win > 0 else 0,
                "avg_profit_pips": np.mean(profit_pips) if profit_pips else 0,
                "total_profit_pips": sum(profit_pips) if profit_pips else 0
            }
        
        self.last_updated = datetime.now().isoformat()
    
    def get_top_transitions(self, top_n=3):
        """
        Returns the most frequent and profitable state transitions.
        """
        # Sort by frequency
        by_freq = sorted(self.state_transitions.items(), 
                        key=lambda x: x[1]["count"], reverse=True)
        
        return by_freq[:top_n]
    
    def get_state_summary(self, state):
        """
        Returns a summary of performance for a specific state.
        """
        if state not in self.state_metrics:
            return {"error": f"State {state} not found"}
        
        return self.state_metrics[state]
    
    def get_metrics_all_states(self):
        """
        Returns all metrics for all states.
        """
        return self.state_metrics
    
    def save_to_file(self, filename=PERFORMANCE_FILE):
        """
        Save performance metrics to file.
        
        Args:
            filename: Output filename
        """
        try:
            # Corrected data structure to save only existing attributes with correct names
            data = {
                "state_metrics": self.state_metrics,
                # "global_metrics": self.global_metrics, # Removed, does not exist
                # "transitions": {str(k): v for k, v in self.transitions.items()}, # Removed, use state_transitions
                "state_transitions": {str(k): v for k, v in self.state_transitions.items()}, # Correct attribute name
                # "validation_scores": {str(k): v for k, v in self.validation_scores.items()}, # Removed, use validation_history
                "validation_history": self.validation_history, # Correct attribute name (list of dicts)
                # "regime_performance": {str(k): v for k, v in self.regime_performance.items()} # Removed, does not exist
                "created_timestamp": self.created_timestamp, # Added for completeness
                "last_updated": self.last_updated # Added for completeness
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
            logging.info(f"Performance data saved to {filename}")
            
        except Exception as e:
            logging.error(f"Error saving performance tracking data: {str(e)}")
            # logging.error(f"Critical error saving performance data: {str(e)}") # Avoid duplicate logging
    
    def load_from_file(self, filename=PERFORMANCE_FILE):
        """
        Load tracking data from a file.
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Convert state_trades keys back to integers
            if "state_trades" in data:
                self.state_trades = {int(k): v for k, v in data["state_trades"].items()}
            
            # Convert state_metrics keys back to integers
            if "state_metrics" in data:
                self.state_metrics = {int(k): v for k, v in data["state_metrics"].items()}
            
            if "state_transitions" in data:
                self.state_transitions = data["state_transitions"]
            
            if "validation_history" in data:
                self.validation_history = data["validation_history"]
            
            if "created" in data:
                self.created_timestamp = data["created"]
                
            if "last_updated" in data:
                self.last_updated = data["last_updated"]
            
            logging.info(f"Performance tracking data loaded from {filename}")
        except Exception as e:
            logging.error(f"Error loading tracking data: {str(e)}")

class StateHistory:
    """
    Stores and analyzes HMM state history.
    """
    def __init__(self, max_size=STATE_HISTORY_WINDOW, load_file=None):
        self.max_size = max_size
        self.states = deque(maxlen=max_size)
        self.confidences = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.prices = deque(maxlen=max_size)
        self.created = datetime.now().isoformat()
        self.last_updated = self.created
        
        if load_file and os.path.exists(load_file):
            self.load_from_file(load_file)
    
    def add_state(self, state, confidence, timestamp=None, price=None):
        """
        Add a state to the history.
        """
        self.states.append(state)
        self.confidences.append(confidence)
        
        if timestamp is None:
            timestamp = datetime.now()
        self.timestamps.append(timestamp if isinstance(timestamp, str) else timestamp.isoformat())
        
        if price is not None:
            self.prices.append(price)
        
        self.last_updated = datetime.now().isoformat()
    
    def get_state_sequence(self, n=None):
        """
        Returns the last n states.
        """
        if n is None:
            return list(self.states)
        return list(self.states)[-n:]
    
    def get_state_frequency(self):
        """
        Returns the frequency of each state.
        """
        states_list = list(self.states)
        unique_states = set(states_list)
        
        freq = {}
        for state in unique_states:
            freq[state] = states_list.count(state) / len(states_list)
        
        return freq
    
    def detect_trends(self, window=20):
        """
        Detects trends in the states.
        """
        if len(self.states) < window:
            return None
        
        recent_states = list(self.states)[-window:]
        
        # Check state frequency
        state_counts = {}
        for state in recent_states:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        dominant_state = max(state_counts, key=state_counts.get)
        dominance = state_counts[dominant_state] / window
        
        # Check stability (fewer transitions = more stable)
        transitions = sum(1 for i in range(1, len(recent_states)) 
                         if recent_states[i] != recent_states[i-1])
        stability = 1 - (transitions / (window - 1))
        
        return {
            "dominant_state": dominant_state,
            "dominance": dominance,
            "stability": stability
        }
    
    def save_to_file(self, filename=STATE_HISTORY_FILE):
        """
        Save state history to file.
        
        Args:
            filename: Output filename
        """
        try:
            # Correctly build data structure from deques
            # Ensure all deques have the same length or handle potential mismatches
            max_len = len(self.states) 
            data = {
                "states": [int(s) for s in list(self.states)],
                "timestamps": list(self.timestamps),
                "confidences": [float(c) for c in list(self.confidences)],
                "prices": [float(p) if p is not None else None for p in list(self.prices)],
                "created": self.created, # Added for completeness
                "last_updated": self.last_updated # Added for completeness
            }
            
            # Verify lengths match (simple check)
            if not (len(data['states']) == len(data['timestamps']) == len(data['confidences']) == len(data['prices'])):
                 logging.warning(f"StateHistory save_to_file: Deque lengths mismatch! States:{len(data['states'])}, Timestamps:{len(data['timestamps'])}, Confidences:{len(data['confidences'])}, Prices:{len(data['prices'])}. Saving truncated data.")
                 min_len = min(len(data['states']), len(data['timestamps']), len(data['confidences']), len(data['prices']))
                 data['states'] = data['states'][:min_len]
                 data['timestamps'] = data['timestamps'][:min_len]
                 data['confidences'] = data['confidences'][:min_len]
                 data['prices'] = data['prices'][:min_len]


            with open(filename, 'w') as f:
                json.dump(data, f, cls=NumpyEncoder)
            logging.info(f"State history saved to {filename}")
            
        except Exception as e:
            logging.error(f"Error saving state history data: {str(e)}")
            # logging.error(f"Critical error saving state history data: {str(e)}") # Avoid duplicate logging
    
    def load_from_file(self, filename=STATE_HISTORY_FILE):
        """
        Load state history from a file.
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Load data into the deques
            self.states = deque(data.get("states", []), maxlen=self.max_size)
            self.confidences = deque(data.get("confidences", []), maxlen=self.max_size)
            self.timestamps = deque(data.get("timestamps", []), maxlen=self.max_size)
            self.prices = deque(data.get("prices", []), maxlen=self.max_size)
            
            if "created" in data:
                self.created = data["created"]
                
            if "last_updated" in data:
                self.last_updated = data["last_updated"]
                
            logging.info(f"State history data loaded from {filename}")
        except Exception as e:
            logging.error(f"Error loading state history: {str(e)}")

class TradeManager:
    """
    Manages trading recommendations and position sizing based on HMM state.
    """
    def __init__(self, symbol=SYMBOL, pip_value=0.01, base_risk_pct=1.0):
        self.symbol = symbol
        self.pip_value = pip_value  # Value of one pip in the symbol
        self.base_risk_pct = base_risk_pct  # Base risk percentage of account
        self.active_trades = []
        self.trade_history = []
        self.created = datetime.now().isoformat()
        self.last_updated = self.created
        
        # Load existing trade history if available
        if os.path.exists(TRADE_HISTORY_FILE):
            self.load_trade_history()
    
    def calculate_position_size(self, account_balance, risk_pct, stop_loss_pips):
        """
        Calculate the optimal position size.
        
        Args:
            account_balance: Account balance
            risk_pct: Risk percentage
            stop_loss_pips: Stop-Loss in pips
        
        Returns:
            position_size: Number of lots
        """
        # Risk calculation
        risk_amount = account_balance * (risk_pct / 100)
        
        # Position size = Risk amount / (Stop-Loss in pips * pip value)
        position_size = risk_amount / (stop_loss_pips * self.pip_value)
        
        # Limit position size to reasonable range (0.01 to 10 lots)
        position_size = min(max(position_size, 0.01), 10.0)
        
        # Round to nearest 0.01
        return round(position_size * 100) / 100
    
    def calculate_risk_parameters(self, state_label, state_confidence, current_price, 
                                 atr_value, account_balance):
        """
        Calculate trading parameters based on the market state.
        
        Args:
            state_label: Label of current state
            state_confidence: Confidence of state (0-1)
            current_price: Current price
            atr_value: ATR value for volatility adjustment
            account_balance: Account balance for position sizing
        
        Returns:
            dict: Trading parameters
        """
        # Base values for different regime types
        risk_multipliers = {
            "High Bullish": (3.0, 1.5, 1.2),     # (TP_Multi, SL_Multi, Risk_Multi)
            "Medium Bullish": (2.5, 1.2, 1.0),
            "Low Bullish": (2.0, 1.0, 0.8),
            "Low Bearish": (2.0, 1.0, 0.8),
            "Medium Bearish": (2.5, 1.2, 1.0),
            "High Bearish": (3.0, 1.5, 1.2)
        }
        
        # Adjustment based on confidence
        confidence_factor = 0.5 + 0.5 * state_confidence  # between 0.5 and 1.0
        
        # Default values in case label is not recognized
        tp_multi, sl_multi, risk_multi = 2.0, 1.0, 1.0
        
        # Get values from mapping if available
        for key in risk_multipliers:
            if key in state_label:
                tp_multi, sl_multi, risk_multi = risk_multipliers[key]
                break
        
        # Distances for TP and SL
        tp_distance_pips = int(atr_value * tp_multi * confidence_factor / self.pip_value)
        sl_distance_pips = int(atr_value * sl_multi * confidence_factor / self.pip_value)
        
        # Ensure TP/SL distances are reasonable
        min_sl_pips = 5  # Minimum SL for safety
        sl_distance_pips = max(sl_distance_pips, min_sl_pips)
        tp_distance_pips = max(tp_distance_pips, sl_distance_pips)  # TP should be larger than SL
        
        # Price calculations
        if "Bullish" in state_label:
            tp_price = current_price + tp_distance_pips * self.pip_value
            sl_price = current_price - sl_distance_pips * self.pip_value
            direction = "LONG"
        else:
            tp_price = current_price - tp_distance_pips * self.pip_value
            sl_price = current_price + sl_distance_pips * self.pip_value
            direction = "SHORT"
        
        # Adjusted risk
        adjusted_risk = self.base_risk_pct * risk_multi * confidence_factor
        
        # Position size
        position_size = self.calculate_position_size(
            account_balance, adjusted_risk, sl_distance_pips
        )
        
        # Risk/Reward ratio
        risk_reward = tp_distance_pips / sl_distance_pips if sl_distance_pips > 0 else 0
        
        return {
            "direction": direction,
            "entry_price": round(current_price, 3),
            "take_profit": round(tp_price, 3),
            "stop_loss": round(sl_price, 3),
            "tp_distance_pips": tp_distance_pips,
            "sl_distance_pips": sl_distance_pips,
            "risk_reward": round(risk_reward, 2),
            "position_size": position_size,
            "risk_percentage": round(adjusted_risk, 2),
            "state_confidence": round(state_confidence, 2)
        }
    
    def add_trade_recommendation(self, trade_params, state, state_label, timestamp=None):
        """
        Add a trading recommendation.
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        recommendation = {
            **trade_params,
            "state": state,
            "state_label": state_label,
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            "status": "recommended"
        }
        
        self.active_trades.append(recommendation)
        self.last_updated = datetime.now().isoformat()
        self.save_trade_history()
        
        return recommendation
    
    def get_active_recommendations(self):
        """
        Get active trading recommendations.
        """
        return self.active_trades
    
    def mark_trade_executed(self, trade_idx, execution_price=None):
        """
        Mark a recommendation as executed.
        """
        if 0 <= trade_idx < len(self.active_trades):
            self.active_trades[trade_idx]["status"] = "executed"
            
            if execution_price:
                self.active_trades[trade_idx]["actual_entry"] = execution_price
            
            self.active_trades[trade_idx]["execution_time"] = datetime.now().isoformat()
            self.last_updated = datetime.now().isoformat()
            self.save_trade_history()
            
            return self.active_trades[trade_idx]
        
        return None
    
    def mark_trade_closed(self, trade_idx, exit_price, exit_reason="manual", timestamp=None):
        """
        Mark an executed recommendation as closed.
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if 0 <= trade_idx < len(self.active_trades):
            trade = self.active_trades[trade_idx]
            
            if trade["status"] != "executed":
                return None
            
            # Calculations for the closed trade
            entry_price = trade.get("actual_entry", trade["entry_price"])
            direction = trade["direction"]
            
            # Calculate profit/loss
            if direction == "LONG":
                profit_pips = int((exit_price - entry_price) / self.pip_value)
            else:  # SHORT
                profit_pips = int((entry_price - exit_price) / self.pip_value)
            
            # Calculate duration
            entry_time = datetime.fromisoformat(trade["execution_time"])
            duration_minutes = (timestamp - entry_time).total_seconds() / 60
            
            # Update trade
            trade["status"] = "closed"
            trade["exit_price"] = exit_price
            trade["exit_time"] = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
            trade["profit_pips"] = profit_pips
            trade["duration_minutes"] = duration_minutes
            trade["exit_reason"] = exit_reason
            
            # Move to history
            self.trade_history.append(trade)
            self.active_trades.pop(trade_idx)
            
            self.last_updated = datetime.now().isoformat()
            self.save_trade_history()
            
            return trade
        
        return None
    
    def get_trade_history(self):
        """
        Get trade history.
        """
        return self.trade_history
    
    def save_trade_history(self):
        """
        Save trade history to a file.
        """
        data = {
            "active_trades": self.active_trades,
            "trade_history": self.trade_history,
            "created": self.created,
            "last_updated": self.last_updated
        }
        
        # Safe saving
        try:
            temp_file = f"{TRADE_HISTORY_FILE}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Create backup if file already exists
            if os.path.exists(TRADE_HISTORY_FILE):
                backup_file = f"{TRADE_HISTORY_FILE}.bak"
                try:
                    os.replace(TRADE_HISTORY_FILE, backup_file)
                except Exception:
                    pass  # Ignore backup errors
            
            # Move temporary file to actual file
            os.replace(temp_file, TRADE_HISTORY_FILE)
            logging.info(f"Trade history saved to {TRADE_HISTORY_FILE}")
        except Exception as e:
            logging.error(f"Error saving trade history: {str(e)}")
            # Try direct saving as fallback
            try:
                with open(TRADE_HISTORY_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e2:
                logging.error(f"Critical error saving trade history: {str(e2)}")
    
    def load_trade_history(self):
        """
        Load trade history from a file.
        """
        try:
            with open(TRADE_HISTORY_FILE, 'r') as f:
                data = json.load(f)
            
            self.active_trades = data.get("active_trades", [])
            self.trade_history = data.get("trade_history", [])
            
            if "created" in data:
                self.created = data["created"]
                
            if "last_updated" in data:
                self.last_updated = data["last_updated"]
                
            logging.info(f"Trade history loaded from {TRADE_HISTORY_FILE}")
        except Exception as e:
            logging.error(f"Error loading trade history: {str(e)}")

###############################################################################
# Enhanced HMM Class
###############################################################################

class EnhancedLiveHMMMt5:
    def __init__(self, model_params, dims_egarch=None, feature_cols=None, 
                performance_tracker=None, state_history=None, trade_manager=None,
                market_memory=None, hybrid_model=None, feature_selector=None,
                order_book_analyzer=None, hybrid_wrapper=None,
                order_book_anomaly_detector=None, order_book_change_detector=None,
                order_book_processor=None, synthetic_ob_generator=None):
        """
        Initialize the enhanced live HMM with additional tracking functions.
        """
        # Grundlegende Parameter
        self.model_params = model_params
        self.K = model_params["K"]
        self.pi = model_params["pi"]
        self.A = model_params["A"]
        self.st_params = model_params["st_params"]
        self.use_tdist = model_params.get("use_tdist", True)
        self.dims_egarch = dims_egarch if dims_egarch is not None else model_params.get("dims_egarch")
        self.feature_cols = feature_cols if feature_cols is not None else model_params.get("feature_cols")
        
        # Feature Dimensionen ermitteln
        self.num_features = 0
        if self.feature_cols:
            self.num_features = len(self.feature_cols)
        elif self.st_params and isinstance(self.st_params, list) and len(self.st_params) > 0:
            # Versuche, aus den State-Parametern abzuleiten
            try:
                # Prüfe, ob 'mu' existiert und ein numpy array oder eine Liste ist
                if 'mu' in self.st_params[0] and hasattr(self.st_params[0]['mu'], '__len__'):
                     self.num_features = len(self.st_params[0]['mu'])
                # Fallback, falls mu nicht existiert oder nicht die richtige Struktur hat
                elif 'cov' in self.st_params[0] and hasattr(self.st_params[0]['cov'], 'shape'):
                     self.num_features = self.st_params[0]['cov'].shape[0]
            except Exception as e:
                logging.warning(f"Konnte Feature-Dimension nicht automatisch ermitteln: {e}")
        
        if self.num_features == 0:
            logging.warning("Anzahl der Features konnte nicht bestimmt werden. Setze auf 0.")
            # raise ValueError("Anzahl der Features konnte nicht bestimmt werden.")

        # Initialisierung des internen Zustands
        self.alpha = self.pi  # Startwahrscheinlichkeiten als erster Zustand
        self.prev_alpha = None  # Hinzugefügt: Vorheriger Alpha-Wert für validate_current_state
        self.current_state = np.argmax(self.alpha)
        self.prev_state = None
        self.last_update_time = None
        self.last_feature_vector = None
        self.t = -1  # Wichtig: Zeit-Counter für partial_step
        self.scaling = None  # Skalierungsfaktor für numerische Stabilität
        
        # Initialize sigma values for all states and dimensions
        self.sigma_id = np.ones((self.K, self.num_features), dtype=float) * 0.01
    
        # Extract volatility model parameters for all states
        self.vol_models = []
        for i in range(self.K):
            mu_i = self.st_params[i]["mu"]
            (om, al, ga, be) = self.st_params[i]["params_vol"]
            df_ = self.st_params[i].get("df_t", 10)
        
            # Optional parameters
            skew_params = self.st_params[i].get("skew_params", None)
            mixture_weight = self.st_params[i].get("mixture_weight", 0.5)
        
            params = {
                "mu": mu_i,
                "omega": om, "alpha": al, "gamma": ga, "beta": be,
                "df": df_
            }
        
            if skew_params is not None:
                params["skew_params"] = skew_params
        
            if mixture_weight != 0.5:
                params["mixture_weight"] = mixture_weight
        
            self.vol_models.append(params)
    
        # Performance tracking
        self.performance_tracker = performance_tracker
        self.state_history = state_history
        self.trade_manager = trade_manager
    
        # New components
        self.market_memory = market_memory
        self.hybrid_model = hybrid_model
        self.feature_selector = feature_selector
        self.order_book_analyzer = order_book_analyzer
        self.hybrid_wrapper = hybrid_wrapper
    
        # Order Book Komponenten
        self.order_book_anomaly_detector = order_book_anomaly_detector
        self.order_book_change_detector = order_book_change_detector
        self.order_book_processor = order_book_processor
        self.synthetic_ob_generator = synthetic_ob_generator
        self.last_ob_analysis = None
        self.ob_signal_strength = 0.0
    
        # For hybrid model integration
        self.last_prediction = None
        self.hybrid_contribution = 0.0

        # Neue Komponenten für Signal Weighting und Ensemble
        self.signal_weighting = None  # Wird bei Komponenten-Initialisierung gesetzt
        self.signal_processor = None  # Wird bei Komponenten-Initialisierung gesetzt
        self.domain_adapter = None    # Wird bei Komponenten-Initialisierung gesetzt
        self.ensemble = None          # Wird bei Komponenten-Initialisierung gesetzt
        self.fusion_backtester = None # Wird bei Komponenten-Initialisierung gesetzt
    
        # Aktuelle Features speichern für spätere Nutzung
        self.current_features = None

        # Feature Fusion Komponente
        self.feature_fusion = None  # Wird bei Komponenten-Initialisierung gesetzt

    
    def partial_init(self, feat_0):
        """
        Initialize the HMM with the first feature vector.
        """
        self.t = 0
        # Equilibrium initialization for sigma
        for i in range(self.K):
            om = self.vol_models[i]["omega"]
            be = self.vol_models[i]["beta"]
            den = (1 - be)
            if abs(den) < 1e-6:
                den = 1e-6 if den >= 0 else -1e-6
            log_sig2_0 = om / den
            log_sig2_0 = max(min(log_sig2_0, 50.0), -50.0)
            s0 = math.exp(0.5 * log_sig2_0)
            if s0 < 1e-12:
                s0 = 1e-12
            for d_ in range(self.num_features):
                if d_ in self.dims_egarch:
                    self.sigma_id[i, d_] = s0
                else:
                    self.sigma_id[i, d_] = 1.0
        
        # Calculate emission for the first time step
        B = np.zeros(self.K)
        for i in range(self.K):
            volm = self.vol_models[i]
            x_ = np.zeros(self.num_features)
            for d_ in range(self.num_features):
                x_[d_] = (feat_0[d_] - volm["mu"][d_]) / self.sigma_id[i, d_]
            val_pdf = t_pdf_diag_multidim(x_, volm["df"])
            B[i] = val_pdf
        
        # Initialize Alpha vector
        alpha_unnorm = self.pi * B
        sum_ = alpha_unnorm.sum()
        if sum_ < 1e-300:
            sum_ = 1e-300
        self.alpha = alpha_unnorm / sum_
        self.scaling = sum_
    
    def partial_step(self, feat_tminus1, feat_t, time_info=None, current_price=None, feature_weights=None, order_book_data=None):
        """
        One step of the Forward algorithm with EGARCH update and enhanced tracking.

        Args:
            feat_tminus1: Feature vector at time t-1
            feat_t: Feature vector at time t
            time_info: Timestamp (for tracking)
            current_price: Current price (for tracking)
            feature_weights: Optional weights for dynamic feature selection
            order_book_data: Order book data for microstructure analysis

        Returns:
            Tuple with state information and tracking data
        """
        # QUALITÄTSVERBESSERUNG: Dimensionsvalidierung zur Vermeidung von Gamma-Array-Problemen
        if feat_tminus1 is None or feat_t is None:
            logging.error("Feature-Vektoren dürfen nicht None sein")
            # Fallback für extreme Fehler
            return {
                "state_idx": 0,
                "state_label": "Error - Invalid Features",
                "market_phase": "Unknown",
                "volatility_level": "Unknown",
                "state_probability": 0.0,
                "state_confidence": 0.0,
                "state_validity": 0.0,
                "needs_retraining": True,
                "transition_occurred": False,
                "transition_type": None,
                "transition_probability": 0.0
            }

        # Normalisiere Feature-Dimensionen für konsistente Verarbeitung
        # QUALITÄTSVERBESSERUNG: Stelle sicher, dass beide Vektoren eindimensional sind
        if len(feat_tminus1.shape) > 1 and feat_tminus1.shape[0] == 1:
            feat_tminus1 = feat_tminus1.flatten()

        if len(feat_t.shape) > 1 and feat_t.shape[0] == 1:
            feat_t = feat_t.flatten()

        # Sichere Dimensionsanpassung mit einheitlicher Funktion
        expected_dimension = len(self.feature_cols)
        feat_tminus1 = ensure_feature_dimensions(feat_tminus1, expected_dimension)
        feat_t = ensure_feature_dimensions(feat_t, expected_dimension)

        self.t += 1
        self.current_features = feat_t.copy()  # Speichere für spätere Verwendung

        # Store previous state
        if self.current_state is not None:
            self.prev_state = self.current_state

        # 1) Update sigma for all states
        for i in range(self.K):
            volm = self.vol_models[i]
            self.sigma_id[i, :] = egarch_update_1step_multi(
                feat_tminus1,
                volm["mu"],
                self.sigma_id[i, :],
                volm["omega"], volm["alpha"], volm["gamma"], volm["beta"],
                self.dims_egarch
            )
    
        # Apply dynamic feature selection if enabled and weights provided
        if feature_weights is None and self.feature_selector and USE_FEATURE_SELECTION:
            try:
                # Get attention weights for current features
                feature_weights = self.feature_selector.get_attention_weights(
                    feat_t, 
                    self._interpret_state(np.argmax(self.alpha))[0]
                )
                logging.debug(f"Dynamic feature weights applied: {feature_weights}")
            except Exception as e:
                logging.error(f"Error getting feature weights: {str(e)}")
                feature_weights = None
    
        # 2) Calculate emission B for current time step
        B = np.zeros(self.K)
        for i in range(self.K):
            volm = self.vol_models[i]
            x_ = np.zeros(self.num_features)
            for d_ in range(self.num_features):
                # Apply feature weights if provided
                weight = 1.0
                if feature_weights is not None and d_ < len(feature_weights):
                    weight = max(0.1, feature_weights[d_])  # Ensure non-zero weight
                x_[d_] = (feat_t[d_] - volm["mu"][d_]) / (self.sigma_id[i, d_] * weight)
            val_pdf = t_pdf_diag_multidim(x_, volm["df"])
            B[i] = val_pdf
    
        # 3) Update Alpha vector
        alpha_new = np.zeros(self.K)
        for j in range(self.K):
            alpha_new[j] = np.sum(self.alpha * self.A[:, j]) * B[j]
        sum_ = alpha_new.sum()
        if sum_ < 1e-300:
            sum_ = 1e-300
        alpha_new /= sum_
    
        self.alpha = alpha_new
        self.scaling = sum_
    
        # Determine most likely state
        i_star = np.argmax(self.alpha)
        label, market_phase, vol_level = self._interpret_state(i_star)
        prob_star = self.alpha[i_star]
    
        # Update current state information
        self.current_state = i_star
    
        # Check if a state transition occurred
        transition_type = None
        transition_prob = 0.0
    
        if self.prev_state is not None and self.prev_state != self.current_state:
            transition_prob = self.A[self.prev_state, self.current_state]
            prev_label, prev_phase, prev_vol = self._interpret_state(self.prev_state)
            transition_type = f"{prev_phase}→{market_phase}"
        
            # Log transition if tracking enabled
            if self.performance_tracker:
                self.performance_tracker.log_state_transition(
                    self.prev_state, self.current_state, transition_prob, time_info
                )
    
        # Calculate state confidence
        state_confidence = self.calculate_state_confidence()
    
        # Log state in history
        if self.state_history:
            self.state_history.add_state(i_star, state_confidence, time_info, current_price)
    
        # Check validity of current state
        validity_score = 1.0
        needs_retraining = False
    
        # --- HMM Dimension Fix Start ---
        # Check validity only if we have current and previous features and gammas
        validity_result = None # Initialize
        if len(feat_t) > 0 and len(feat_tminus1) > 0 and self.prev_alpha is not None:
            features_check = np.vstack([feat_tminus1, feat_t]) # Shape (2, D)
            gamma_check = np.vstack([self.prev_alpha, self.alpha]) # Shape (2, K)
            
            # Ensure dimensions match before calling validate_current_state
            if features_check.shape[0] == gamma_check.shape[0]:
                 # --- FIX: Correctly handle dictionary return value --- 
                 validation_result = validate_current_state(
                     gamma_check,
                     features_check,
                     self.current_state,
                     lookback=2
                 )
            else:
                logging.warning(f"Dimension mismatch before validate_current_state: Features {features_check.shape[0]}, Gamma {gamma_check.shape[0]}")
                # Fallback to single point validation if mismatch occurs
                features_check_t = feat_t.reshape(1, -1)
                gamma_check_t = self.alpha.reshape(1, -1)
                # --- FIX: Correctly handle dictionary return value --- 
                validation_result = validate_current_state(
                     gamma_check_t, features_check_t, self.current_state, lookback=1
                 )

        elif len(feat_t) > 0: # Fallback for the very first step where prev_alpha is None
            features_check_t = feat_t.reshape(1, -1)
            gamma_check_t = self.alpha.reshape(1, -1)
            # --- FIX: Correctly handle dictionary return value --- 
            validation_result = validate_current_state(
                 gamma_check_t, features_check_t, self.current_state, lookback=1
             )
             
        # --- FIX: Extract values from the dictionary --- 
        validity_score = 0.5 # Default if validation failed
        needs_retraining = True # Default if validation failed
        validation_reason = "Validation skipped or failed"
        if isinstance(validation_result, dict):
             if validation_result.get('valid', False):
                 validity_score = validation_result.get('probability', 0.5)
                 # Determine needs_retraining based on score (example logic)
                 needs_retraining = validity_score < 0.5 
             else:
                 validity_score = 0.1 # Low score if validation failed
                 needs_retraining = True
             validation_reason = validation_result.get('reason', 'Unknown')
        # --- END FIX --- 
        # --- HMM Dimension Fix End ---
            
        # Log validity score
        if self.performance_tracker:
            # Log the numeric score, not the dictionary/tuple
            self.performance_tracker.add_validation_score(i_star, validity_score, time_info)
            
        # Integrate with market memory if available
        memory_prediction = None
        if self.market_memory and USE_MARKET_MEMORY:
            try:
                # Look for similar historical patterns
                features_seq = np.vstack([feat_tminus1, feat_t])
                memory_result = self.market_memory.find_similar_patterns(features_seq, n_neighbors=5)
            
                if memory_result:
                    # Get outcome prediction
                    memory_prediction = self.market_memory.predict_outcome(
                        features_seq, label, n_neighbors=5
                    )
                    logging.info(f"Market memory prediction: {memory_prediction.get('prediction', 'unknown')} "
                                f"(confidence: {memory_prediction.get('confidence', 0):.2f})")
                
                    # Store pattern in memory for future reference
                    self.market_memory.add_pattern(features_seq, label)
            except Exception as e:
                logging.error(f"Error using market memory: {str(e)}")
    
        # Integrate with order book analysis
        ob_analysis = None
        ob_signal = None

        if order_book_data:
            # Process order book data
            if self.order_book_processor:
                try:
                    # Extract features from order book
                    ob_features = self.order_book_processor.extract_features(order_book_data)
        
                    # Add to anomaly detector if available
                    if self.order_book_anomaly_detector:
                        self.order_book_anomaly_detector.add_orderbook_features(ob_features)
            
                        # Check for anomalies
                        anomaly_result = self.order_book_anomaly_detector.detect_anomalies(ob_features)
            
                        if anomaly_result["is_anomaly"]:
                            # Classify anomaly type
                            anomaly_info = self.order_book_anomaly_detector.classify_anomaly_type(ob_features)
                
                            # Get trading recommendation
                            direction = "buy" if "Bullish" in label else "sell"
                            trading_rec = self.order_book_anomaly_detector.get_anomaly_trading_recommendation(
                                anomaly_info,
                                current_price=current_price,
                                state_label=label
                            )
                
                            logging.info(f"Order book anomaly detected: {anomaly_info['type']} (conf: {anomaly_info['confidence']:.2f})")
                            logging.info(f"Trading recommendation: {trading_rec['action']} - {trading_rec['description']}")
                
                            ob_analysis = {
                                "anomaly_detected": True,
                                "anomaly_type": anomaly_info['type'],
                                "confidence": anomaly_info['confidence'],
                                "recommendation": trading_rec
                            }
        
                    # Check for significant changes if change detector available
                    if self.order_book_change_detector:
                        changes = self.order_book_change_detector.add_orderbook(order_book_data)
            
                        if changes.get("changes_detected", False):
                            # Get trading signal from changes
                            direction = "up" if "Bullish" in label else "down" if "Bearish" in label else None
                            ob_signal = self.order_book_change_detector.get_trading_signal(changes, price_trend=direction)
                
                            if ob_signal["signal"] != "NONE":
                                logging.info(f"Order book change signal: {ob_signal['signal']} (conf: {ob_signal['confidence']:.2f})")
                    
                                # Calculate signal strength
                                self.ob_signal_strength = ob_signal['confidence'] * 0.8  # Scale slightly
        
                    # Store for future reference
                    self.last_ob_analysis = {
                        "features": ob_features,
                        "anomaly": ob_analysis,
                        "signal": ob_signal,
                        "timestamp": datetime.now()
                    }
            
                except Exception as e:
                    logging.error(f"Error analyzing order book: {str(e)}")
        elif self.synthetic_ob_generator and current_price is not None:
            # Generate and analyze synthetic order book if real data not available
            try:
                # Determine market state based on current label
                market_state = None
                if "Bullish" in label:
                    market_state = "trending_bull"
                elif "Bearish" in label:
                    market_state = "trending_bear"
                else:
                    market_state = "ranging"
        
                # Generate synthetic order book
                synthetic_book = self.synthetic_ob_generator.generate_order_book(
                    current_price=current_price,
                    market_state=market_state
                )
    
                # Process synthetic book (with lower confidence)
                if synthetic_book and self.order_book_processor:
                    ob_features = self.order_book_processor.extract_features(synthetic_book)
        
                    # Store with synthetic flag
                    self.last_ob_analysis = {
                        "features": ob_features,
                        "is_synthetic": True,
                        "timestamp": datetime.now()
                    }
        
                    # Calculate baseline signal strength (lower for synthetic data)
                    self.ob_signal_strength = 0.3  # Lower baseline for synthetic data
        
                    logging.debug("Using synthetic order book for analysis")
            except Exception as e:
                logging.error(f"Error with synthetic order book: {str(e)}")
    
        # Integrate with hybrid model if available
        hybrid_prediction = None
        if self.hybrid_model and USE_HYBRID_MODEL:
            try:
                # --- FIX: Prepare inputs correctly for HybridModel predict_* methods ---
                # 1. Prepare Feature Sequence -> Shape: (1, seq_length, D)
                seq_length = self.hybrid_model.sequence_length if hasattr(self.hybrid_model, 'sequence_length') else 10
                num_features = self.num_features
                feature_seq_for_model = None

                # We need a sequence ending at time t. Requires history.
                # Simplification: Tile current feature if no history buffer available in this context.
                # A more robust solution would involve maintaining a feature buffer.
                current_features = feat_t # Assuming feat_t is the latest vector
                tiled_features = np.tile(current_features, (seq_length, 1))
                feature_seq_for_model = tiled_features.reshape(1, seq_length, num_features)
                
                # Ensure final feature shape
                if feature_seq_for_model.shape != (1, seq_length, num_features):
                    raise ValueError(f"Incorrect feature shape: {feature_seq_for_model.shape}")

                # 2. Prepare HMM State -> Shape: (1, K)
                hmm_state_onehot = np.zeros(self.K)
                hmm_state_onehot[i_star] = 1.0
                hmm_state_input = hmm_state_onehot.reshape(1, self.K)

                # Ensure final HMM shape
                if hmm_state_input.shape != (1, self.K):
                     raise ValueError(f"Incorrect HMM state shape: {hmm_state_input.shape}")
                     
                # Prepare context (dummy for now if not available)
                # TODO: Implement proper context feature generation here if needed by get_action_recommendation
                context_features = np.zeros(12) # Assuming 12 context features as per model definition
                # --- END FIX --- 
                
                # Make predictions using the correctly shaped inputs
                direction_pred = self.hybrid_model.predict_market_direction(feature_seq_for_model, hmm_state_input)
                volatility_pred = self.hybrid_model.predict_market_volatility(feature_seq_for_model, hmm_state_input)
                action_rec = self.hybrid_model.get_action_recommendation(feature_seq_for_model, hmm_state_input, context_features.reshape(1, -1)) # Add batch dim for context
                
                # Combine with HMM prediction
                hybrid_prediction = self.hybrid_model.combine_predictions(
                    {"state_idx": i_star, "state_label": label, "state_confidence": state_confidence},
                    direction_pred,
                    volatility_pred,
                    action_rec
                )
                
                # Store for later use
                self.last_prediction = hybrid_prediction
                
                # Calculate hybrid contribution
                if "signal_strength" in hybrid_prediction:
                    self.hybrid_contribution = hybrid_prediction["signal_strength"] * HYBRID_CONTRIBUTION_WEIGHT
                
                logging.info(f"Hybrid model signal: {hybrid_prediction.get('signal', 'NONE')} "
                            f"(strength: {hybrid_prediction.get('signal_strength', 0):.2f})")
            except Exception as e:
                logging.error(f"Error using hybrid model: {str(e)}")
        
        # Update feature selector if available
        if self.feature_selector and USE_FEATURE_SELECTION:
            try:
                # Update importance based on prediction success
                # --- FIX: Use the extracted numeric validity_score --- 
                prediction_success = validity_score > 0.7 # Now this comparison should work
                # --- END FIX ---
                self.feature_selector.update_importance(
                    np.vstack([feat_tminus1, feat_t]), 
                    label, 
                    prediction_success
                )
            except Exception as e:
                # --- FIX: Log full traceback for TypeError --- 
                import traceback
                tb_str = traceback.format_exc()
                logging.error(f"Error updating feature selection: {str(e)}\nTraceback:\n{tb_str}")
                # --- END FIX ---
        
        # Wrapper for return values
        result = {
            "state_idx": i_star,
            "state_label": label,
            "market_phase": market_phase,
            "volatility_level": vol_level,
            "state_probability": prob_star,
            "state_confidence": state_confidence,
            "state_validity": validity_score,
            "needs_retraining": needs_retraining,
            "transition_occurred": transition_type is not None,
            "transition_type": transition_type,
            "transition_probability": transition_prob,
            "memory_prediction": memory_prediction,
            "hybrid_prediction": hybrid_prediction,
            "ob_analysis": ob_analysis,
            "hybrid_contribution": self.hybrid_contribution,
            "ob_signal_strength": getattr(self, 'ob_signal_strength', 0.0)
        }
        
        return result
    
    def calculate_state_confidence(self):
        """
        Calculate the confidence of the current state
        Higher values = clearer state assignment
        """
        sorted_alpha = np.sort(self.alpha)[::-1]  # Sort descending
        if len(sorted_alpha) > 1:
            # Difference between highest and second highest value
            confidence = sorted_alpha[0] - sorted_alpha[1]
        else:
            confidence = 1.0
        return min(confidence * 5, 1.0)  # Scale to max 1.0
    
    def calculate_risk_parameters(self, atr_value, account_balance):
        """
        Calculate risk parameters for trading based on current state.
        
        Args:
            atr_value: ATR value for volatility adjustment
            account_balance: Account balance for position sizing
        
        Returns:
            dict: Trading parameters
        """
        if self.current_state is None or self.trade_manager is None:
            return None
        
        # Get state parameters
        _, market_phase, vol_level = self._interpret_state(self.current_state)
        state_label = f"{vol_level} {market_phase}"
        state_confidence = self.calculate_state_confidence()
        
        # Get current price
        price_info = mt5.symbol_info_tick(SYMBOL)
        if price_info is None:
            logging.error(f"Error getting current price for {SYMBOL}")
            return None
        
        current_price = price_info.ask  # Ask price for long positions
        
        # Calculate combined confidence from different components
        combined_confidence = state_confidence
        signal_strength_factors = []
        override_direction = None
        hybrid_label_modifier = ""
        
        # Add hybrid model contribution if available
        if hasattr(self, 'hybrid_contribution') and self.hybrid_contribution > 0:
            signal_strength_factors.append(("Hybrid", self.hybrid_contribution))
            
            # Use hybrid model signal if available
            if self.last_prediction and "signal" in self.last_prediction:
                if self.last_prediction["signal"] != "NONE":
                    hybrid_signal = self.last_prediction["signal"]
                    hybrid_label_modifier = f" (Hybrid {hybrid_signal})"
                    
                    # Check if hybrid model contradicts HMM state
                    if hybrid_signal == "LONG" and "Bearish" in state_label:
                        if self.hybrid_contribution > 0.7:  # Only override if strong signal
                            override_direction = "LONG"
                            logging.info("Strong hybrid model signal overriding HMM state direction to LONG")
                    elif hybrid_signal == "SHORT" and "Bullish" in state_label:
                        if self.hybrid_contribution > 0.7:  # Only override if strong signal
                            override_direction = "SHORT"
                            logging.info("Strong hybrid model signal overriding HMM state direction to SHORT")
        
        # Add order book analysis contribution if available
        if hasattr(self, 'ob_signal_strength') and self.ob_signal_strength > 0:
            signal_strength_factors.append(("Order Book", self.ob_signal_strength * 0.5))  # Weight OB less than hybrid
        
        # Add market memory contribution if available
        if hasattr(self, 'market_memory') and USE_MARKET_MEMORY:
            memory_confidence = 0.0
            try:
                # Get prediction from most recent analysis
                features_seq = np.vstack([np.zeros(self.num_features), np.zeros(self.num_features)])  # Dummy for this example
                memory_prediction = self.market_memory.predict_outcome(features_seq, state_label)
                
                if memory_prediction and "confidence" in memory_prediction:
                    memory_confidence = memory_prediction["confidence"] * 0.4  # Weight memory less than hybrid
                    signal_strength_factors.append(("Memory", memory_confidence))
            except Exception as e:
                logging.error(f"Error getting memory prediction: {str(e)}")
        
        # Calculate weighted average of all signal strengths
        if signal_strength_factors:
            total_weight = sum(weight for _, weight in signal_strength_factors)
            for source, weight in signal_strength_factors:
                combined_confidence += weight
            
            # Normalize to max 1.0
            combined_confidence = min(combined_confidence, 1.0)
            
            # Log contribution breakdown
            logging.info(f"Signal strength contributors: {signal_strength_factors}")
            logging.info(f"Combined confidence: {combined_confidence:.2f}")
        
        # Apply direction override if necessary
        modified_state_label = state_label
        if override_direction:
            if override_direction == "LONG":
                modified_state_label = modified_state_label.replace("Bearish", "Hybrid-Bullish")
            elif override_direction == "SHORT":
                modified_state_label = modified_state_label.replace("Bullish", "Hybrid-Bearish")
            
            logging.info(f"Direction override applied: {modified_state_label}")
        
        # Add hybrid label modifier
        if hybrid_label_modifier:
            modified_state_label += hybrid_label_modifier
        
        # Calculate risk parameters with combined confidence
        risk_params = self.trade_manager.calculate_risk_parameters(
            modified_state_label, combined_confidence, current_price, atr_value, account_balance
        )
        
        return risk_params
    
    def _interpret_state(self, i_star):
        """
        Extended interpretation of current state:
        - Market phase (bullish/bearish) based on all returns
        - Volatility level based on all return sigmas
        - Daytime context from session flags
        """
        volm = self.vol_models[i_star]
        mu_vec = volm["mu"]
        
        # Feature indices for interpretation
        # These can be different depending on feature list
        # This is a standard list that should be adapted
        indices = {
            'ret30': 0, 'ret5': 1, 'ret1h': 2, 'ret4h': 3,
            'rsi_30m': 4, 'rsi_1h': 5, 'atr_30m': 6, 'atr_4h': 7, 'macd_30m': 8,
            'session_asia': 9, 'session_europe': 10, 'session_us': 11, 'session_overlap': 12,
            'log_volume': 13,
            'day_mon': 14, 'day_tue': 15, 'day_wed': 16, 'day_thu': 17, 'day_fri': 18
        }
        
        # If feature names available, use them for correct indices
        if self.feature_cols:
            indices = {col: i for i, col in enumerate(self.feature_cols)}
        
        # Market phase based on all returns
        return_indices = [
            indices.get('log_return30', 0),
            indices.get('log_return5', 1),
            indices.get('log_return1h', 2) if 'log_return1h' in indices else -1,
            indices.get('log_return4h', 3) if 'log_return4h' in indices else -1
        ]
        return_indices = [idx for idx in return_indices if idx >= 0 and idx < len(mu_vec)]
        
        # Weighted sum of all returns
        weights = [0.25, 0.15, 0.3, 0.3]  # Adjust according to importance
        weights = weights[:len(return_indices)]
        weights = [w/sum(weights) for w in weights]  # Normalize
        
        # Weighted average of returns
        mean_return = sum(mu_vec[idx] * weights[i] for i, idx in enumerate(return_indices))
        
        # Determine market phase (extended for more granularity)
        if mean_return > 0.0015:
            market_phase = "Strong Bullish"
        elif mean_return > 0.0005:
            market_phase = "Bullish"
        elif mean_return > 0:
            market_phase = "Weak Bullish"
        elif mean_return > -0.0005:
            market_phase = "Weak Bearish"
        elif mean_return > -0.0015:
            market_phase = "Bearish"
        else:
            market_phase = "Strong Bearish"
        
        # Volatility level based on sigmas of all returns
        vol_value = 0
        for i, idx in enumerate(return_indices):
            vol_value += self.sigma_id[i_star, idx] * weights[i]
        
        if vol_value > 0.02:
            vol_level = "Very High"
        elif vol_value > 0.015:
            vol_level = "High"
        elif vol_value > 0.01:
            vol_level = "Medium"
        elif vol_value > 0.005:
            vol_level = "Low"
        else:
            vol_level = "Very Low"
        
        # Session context
        session_context = ""
        session_indices = [
            indices.get('session_asia', -1),
            indices.get('session_europe', -1),
            indices.get('session_us', -1),
            indices.get('session_overlap', -1)
        ]
        
        # Check which session is active (based on feature vector)
        active_sessions = []
        for i, name in enumerate(['Asia', 'Europe', 'US', 'Overlap']):
            idx = session_indices[i]
            if idx >= 0 and idx < len(mu_vec) and mu_vec[idx] > 0.5:  # If session is active
                active_sessions.append(name)
        
        if active_sessions:
            session_context = f" ({'/'.join(active_sessions)})"
        
        # Combined label
        label = f"{vol_level} {market_phase}{session_context}"
        
        return label, market_phase, vol_level
    
    def get_composite_signal(self, current_price=None, atr_value=None, account_balance=None, prev_feature=None):
        """
        Generate a composite trading signal integrating HMM state with hybrid model,
        market memory and order book analysis.
        
        Args:
            current_price: Current market price
            atr_value: ATR value for volatility adjustment
            account_balance: Account balance for position sizing
            prev_feature: Previous feature vector for sequence analysis
            
        Returns:
            dict: Integrated trading signal with confidence and parameters
        """
        if self.current_state is None:
            return {"signal": "NONE", "confidence": 0.0, "explanation": "No state information available"}
        
        # Get current state info
        state_idx = self.current_state
        state_label, market_phase, vol_level = self._interpret_state(state_idx)
        state_confidence = self.calculate_state_confidence()

        # Wenn Signal Processor und Ensemble vorhanden sind, diese für die Signalgenerierung nutzen
        if hasattr(self, 'signal_processor') and self.signal_processor and hasattr(self, 'ensemble') and self.ensemble:
            try:
                # Sammle Signale von allen Komponenten
                component_signals = {}
                expected_dim = self.num_features # Erwartete Dimension für alle Komponenten
            
                # 1. HMM Signal
                hmm_signal = {
                    "signal": "LONG" if "Bullish" in market_phase else "SHORT" if "Bearish" in market_phase else "NONE",
                    "strength": state_confidence,
                    "state_idx": state_idx
                }
                component_signals["hmm"] = hmm_signal
            
                # 2. Hybrid Model Signal
                if hasattr(self, 'hybrid_model') and self.hybrid_model:
                    # --- FIX: Prepare inputs correctly for HybridModel predict_* methods --- 
                    if prev_feature is not None and self.current_features is not None:
                        # 1. Prepare Feature Sequence -> Shape: (1, seq_length, D)
                        seq_length = self.hybrid_model.sequence_length if hasattr(self.hybrid_model, 'sequence_length') else 10
                        num_features = self.num_features
                        feature_seq_for_model = None
                        
                        # Use history (prev_feature, current_features) if available
                        # We need seq_length features ending with current_features
                        # This part requires access to a feature history buffer which is not directly available here.
                        # Simplification: Use the logic from partial_step (tiling current feature)
                        # TODO: Implement proper sequence buffering for get_composite_signal
                        current_features_hmm = ensure_feature_dimensions(self.current_features, num_features)
                        tiled_features = np.tile(current_features_hmm, (seq_length, 1))
                        feature_seq_for_model = tiled_features.reshape(1, seq_length, num_features)
                        
                        # Ensure final feature shape
                        if feature_seq_for_model.shape != (1, seq_length, num_features):
                             raise ValueError(f"Incorrect feature shape in get_composite_signal: {feature_seq_for_model.shape}")

                        # 2. Prepare HMM State -> Shape: (1, K)
                        hmm_state_onehot = np.zeros(self.K)
                        hmm_state_onehot[state_idx] = 1.0
                        hmm_state_input = hmm_state_onehot.reshape(1, self.K)

                        # Ensure final HMM shape
                        if hmm_state_input.shape != (1, self.K):
                            raise ValueError(f"Incorrect HMM state shape in get_composite_signal: {hmm_state_input.shape}")
                        # --- END FIX --- 
                        
                        # Hole Vorhersagen vom Hybrid-Modell
                        try:
                            direction_pred = self.hybrid_model.predict_market_direction(feature_seq_for_model, hmm_state_input)
                            volatility_pred = self.hybrid_model.predict_market_volatility(feature_seq_for_model, hmm_state_input)
                            
                            # ... (rest of signal combination) ...
                            signal_type = "NONE"
                            if direction_pred is not None:
                                if isinstance(direction_pred, dict) and "signal" in direction_pred:
                                    signal_type = direction_pred["signal"]
                                elif isinstance(direction_pred, str):
                                    signal_type = direction_pred
                        
                            signal_strength = 0.5  # Default
                            if direction_pred is not None and isinstance(direction_pred, dict) and "confidence" in direction_pred:
                                signal_strength = direction_pred["confidence"]
                        
                            hybrid_signal = {
                                "signal": signal_type,
                                "strength": signal_strength
                            }
                            component_signals["hybrid_model"] = hybrid_signal
                        except Exception as e:
                            logging.error(f"Error getting hybrid model prediction: {str(e)}")
                            component_signals["hybrid_model"] = {"signal": "NONE", "strength": 0.0}
                    else:
                        component_signals["hybrid_model"] = {"signal": "NONE", "strength": 0.0}
            
                # 3. Market Memory Signal
                if hasattr(self, 'market_memory') and self.market_memory:
                    try:
                        if prev_feature is not None and self.current_features is not None:
                            # --- FIX: Ensure features have expected dimension --- 
                            prev_feature_mem = ensure_feature_dimensions(prev_feature, expected_dim)
                            current_features_mem = ensure_feature_dimensions(self.current_features, expected_dim)
                            features_seq = np.vstack([prev_feature_mem, current_features_mem])
                            # --- END FIX ---
                            memory_prediction = self.market_memory.predict_outcome(features_seq, state_label)
                        
                            if memory_prediction and "prediction" in memory_prediction:
                                prediction = memory_prediction["prediction"]
                                confidence = memory_prediction.get("confidence", 0.5)
                             
                                # Einfaches Mapping von Memory-Vorhersage zu Signal
                                signal_type = "NONE"
                                if prediction == "profitable" and "Bull" in state_label:
                                    signal_type = "LONG"
                                elif prediction == "profitable" and "Bear" in state_label:
                                    signal_type = "SHORT"
                            
                                memory_signal = {
                                    "signal": signal_type,
                                    "strength": confidence
                                }
                                component_signals["market_memory"] = memory_signal
                            else:
                                component_signals["market_memory"] = {"signal": "NONE", "strength": 0.0}
                        else:
                            component_signals["market_memory"] = {"signal": "NONE", "strength": 0.0}
                    except Exception as e:
                        logging.error(f"Error getting market memory prediction: {str(e)}")
                        component_signals["market_memory"] = {"signal": "NONE", "strength": 0.0}
            
                # 4. Order Book Signal (enthält normalerweise keine Features, die angepasst werden müssen)
                if hasattr(self, 'last_ob_analysis') and self.last_ob_analysis and 'signal' in self.last_ob_analysis:
                    ob_info = self.last_ob_analysis['signal']
                    if ob_info and 'signal' in ob_info and 'confidence' in ob_info:
                        component_signals["order_book"] = {
                            "signal": ob_info['signal'],
                            "strength": ob_info['confidence']
                        }
                    else:
                        component_signals["order_book"] = {"signal": "NONE", "strength": 0.0}
                else:
                    component_signals["order_book"] = {"signal": "NONE", "strength": 0.0}
            
                # Volatilität und Trend aus dem State-Label extrahieren
                volatility = "medium"
                if "High" in vol_level or "Very High" in vol_level:
                    volatility = "high"
                elif "Low" in vol_level or "Very Low" in vol_level:
                    volatility = "low"
                
                trend = "neutral"
                if "Bullish" in market_phase or "Bull" in market_phase:
                    trend = "bullish"
                elif "Bearish" in market_phase or "Bear" in market_phase:
                    trend = "bearish"
                
                # Session aus State-Label extrahieren (falls vorhanden)
                session = "unknown"
                if "Asia" in state_label:
                    session = "asian"
                elif "Europe" in state_label:
                    session = "european"
                elif "US" in state_label:
                    session = "us"
                elif "Overlap" in state_label:
                    session = "overlap"
            
                # Ensemble Signal generieren
                ensemble_result = self.ensemble.ensemble_signal(
                    component_signals=component_signals,
                    state_label=state_label,
                    volatility=volatility,
                    trend=trend,
                    session=session,
                    current_price=current_price,
                    current_time=datetime.now().isoformat()
                )
            
                if ensemble_result:
                    # Erstelle ein erweitertes Ergebnis mit Komponenten-Details
                    result = {
                        "signal": ensemble_result["signal"],
                        "confidence": ensemble_result["strength"],
                        "component_contributions": ensemble_result["component_contributions"],
                        "state_idx": state_idx,
                        "state_label": state_label,
                        "weights": ensemble_result["weights"],
                        "explanation": [f"Ensemble signal based on weighted components: {ensemble_result['signal']} ({ensemble_result['strength']:.2f})"]
                    }
                
                    # Füge Komponenten-Details hinzu
                    for comp, contrib in ensemble_result["component_contributions"].items():
                        result["explanation"].append(f"{comp}: {contrib['signal']} (weight: {contrib['weight']:.2f}, contribution: {contrib['contribution']:.2f})")
                
                    # Get risk parameters if a signal exists
                    if result["signal"] != "NONE" and result["confidence"] > 0.5 and self.trade_manager:
                        if atr_value is None:
                            # Use default ATR value or estimate from EGARCH
                            atr_value = max(0.001, self.sigma_id[state_idx, 0] * 100)  # Simple estimate
                        
                        if account_balance is None:
                            # Try to get account balance from MT5
                            try:
                                account_info = mt5.account_info()
                                account_balance = account_info.balance if account_info else 10000
                            except:
                                account_balance = 10000  # Default value
                    
                        # Get trading parameters
                        risk_params = self.trade_manager.calculate_risk_parameters(
                            state_label,
                            result["confidence"],
                            current_price,
                            atr_value,
                            account_balance
                        )
                    
                        if risk_params:
                            # Override direction if needed
                            risk_params["direction"] = result["signal"]
                            result["risk_params"] = risk_params
                
                    return result
            except Exception as e:
                logging.error(f"Error generating ensemble signal: {str(e)}")
                # Fall through to standard method if ensemble fails
 
        if hasattr(self, 'hybrid_wrapper') and self.hybrid_wrapper is not None and prev_feature is not None:
            try:
                # --- FIX START: Explicit dimension handling before wrapper call ---
                # Ensure prev_feature and current_features are 1D arrays of the correct HMM dimension
                # Use self.num_features which should hold the correct dimension based on HMM params
                expected_hmm_dim = self.num_features 
                prev_feature_1d = ensure_feature_dimensions(prev_feature, expected_hmm_dim)
                # Ensure current_features is accessed correctly (it's stored on self)
                current_features_1d = ensure_feature_dimensions(self.current_features, expected_hmm_dim)

                # Check if dimensions are correct after ensuring
                if prev_feature_1d.shape != (expected_hmm_dim,) or current_features_1d.shape != (expected_hmm_dim,):
                     logging.error(f"Dimension mismatch AFTER ensure_feature_dimensions: prev={prev_feature_1d.shape}, curr={current_features_1d.shape}, expected=({expected_hmm_dim},)")
                     # Handle error appropriately, maybe skip wrapper call
                     raise ValueError("Feature dimensions could not be reconciled before vstack.")

                # Prepare sequence for wrapper - should be shape (2, expected_hmm_dim)
                features_seq = np.vstack([prev_feature_1d, current_features_1d])
                # --- FIX END ---
            
                # Get prediction from wrapper using the sequence with ensured dimensions
                # The wrapper internally handles TF dimension adaptation
                wrapper_pred = self.hybrid_wrapper(features_seq)
                # Removed the redundant dimension reduction logic here, as the wrapper handles it.
            
                if wrapper_pred:
                    logging.info(f"Using hybrid wrapper prediction: {wrapper_pred['signal']} (strength: {wrapper_pred.get('signal_strength', 0):.2f})")
                
                    # Combine wrapper prediction with additional context
                    wrapper_pred.update({
                        "hmm_state_idx": state_idx,
                        "hmm_state_label": state_label,
                        "hmm_state_confidence": state_confidence,
                        "timestamp": datetime.now().isoformat(),
                        "hybrid_wrapper_used": True
                    })
                
                    # Calculate risk parameters if a signal exists
                    if wrapper_pred["signal"] != "NONE" and wrapper_pred.get("signal_strength", 0) > 0.5 and self.trade_manager:
                        if atr_value is None:
                            # Use default ATR value or estimate from EGARCH
                            atr_value = max(0.001, self.sigma_id[state_idx, 0] * 100)  # Simple estimate
                        
                        if account_balance is None:
                            # Try to get account balance from MT5
                            try:
                                account_info = mt5.account_info()
                                account_balance = account_info.balance if account_info else 10000
                            except:
                                account_balance = 10000  # Default value
                    
                        # Get trading parameters
                        risk_params = self.trade_manager.calculate_risk_parameters(
                            state_label,
                            wrapper_pred.get("combined_confidence", wrapper_pred.get("signal_strength", 0.5)),
                            current_price,
                            atr_value,
                            account_balance
                        )
                    
                        if risk_params:
                            # Override direction if needed
                            risk_params["direction"] = wrapper_pred["signal"]
                            wrapper_pred["risk_params"] = risk_params
                
                    # --- FIX START: Standardize return dictionary --- 
                    # Ensure the returned dictionary has the 'confidence' key
                    wrapper_pred['confidence'] = wrapper_pred.get('combined_confidence', wrapper_pred.get('signal_strength', 0.0))
                    # Ensure other standard keys are present for consistency
                    wrapper_pred['explanation'] = [f"Signal from Hybrid Wrapper: {wrapper_pred['signal']} (strength: {wrapper_pred.get('signal_strength', 0):.2f})"]
                    wrapper_pred['sources'] = [("HybridWrapper", wrapper_pred.get('signal_strength', 0.0))]
                    # --- FIX END ---
                    
                    return wrapper_pred 
            except Exception as e:
                logging.error(f"Error using hybrid wrapper: {str(e)}")
                # Fall back to standard method if wrapper fails
        
        # Base signal from HMM state
        signal = "LONG" if "Bullish" in market_phase else "SHORT" if "Bearish" in market_phase else "NONE"
        signal_confidence = state_confidence
        signal_sources = [("HMM", state_confidence)]
        signal_explanation = [f"HMM State {state_idx}: {state_label} (conf: {state_confidence:.2f})"]
        
        # If no clear market direction, set to NONE
        if "Weak" in market_phase and state_confidence < 0.7:
            signal = "NONE"
            signal_explanation.append("Market direction too weak, no clear signal")
        
        # Integrate hybrid model if available
        hybrid_signal = "NONE"
        hybrid_confidence = 0.0
        if self.hybrid_model and hasattr(self, 'last_prediction') and self.last_prediction:
            hybrid_signal = self.last_prediction.get("signal", "NONE")
            hybrid_confidence = self.last_prediction.get("signal_strength", 0.0)
            
            if hybrid_signal != "NONE" and hybrid_confidence > 0.6:
                # Hybrid model has a strong signal
                signal_sources.append(("Hybrid", hybrid_confidence))
                signal_explanation.append(f"Hybrid model: {hybrid_signal} (conf: {hybrid_confidence:.2f})")
                
                # If hybrid strongly contradicts HMM and has high confidence
                if hybrid_signal != signal and hybrid_confidence > 0.8 and state_confidence < 0.7:
                    signal = hybrid_signal
                    signal_explanation.append(f"Hybrid model overriding HMM state direction to {hybrid_signal}")
        
        # Integrate market memory if available
        memory_signal = "NONE"
        memory_confidence = 0.0
        if self.market_memory and hasattr(self, 'memory_prediction') and self.memory_prediction:
            prediction = self.memory_prediction.get("prediction", "unknown")
            memory_confidence = self.memory_prediction.get("confidence", 0.0)
            
            if prediction == "profitable" and signal != "NONE":
                # Memory confirms signal direction is profitable
                memory_signal = signal
                signal_sources.append(("Memory", memory_confidence))
                signal_explanation.append(f"Market memory confirms {signal} (conf: {memory_confidence:.2f})")
            elif prediction == "loss" and memory_confidence > 0.7:
                # Memory strongly indicates signal would be a loss
                memory_signal = "NONE"
                signal_explanation.append(f"Market memory indicates potential loss, suggesting no trade")
                signal = "NONE"  # Override to no signal if memory strongly indicates loss

        # Integrate order book analysis if available
        ob_signal = "NONE"
        ob_confidence = 0.0
        if hasattr(self, 'last_ob_analysis') and self.last_ob_analysis:
            # Check if we have a direct signal from order book change detection
            if 'signal' in self.last_ob_analysis and self.last_ob_analysis['signal']:
                ob_info = self.last_ob_analysis['signal']
                if 'signal' in ob_info and ob_info['signal'] != "NONE":
                    ob_signal = ob_info['signal']
                    ob_confidence = ob_info.get('confidence', 0.0)
            
                    # Adjust confidence if it's from synthetic data
                    if self.last_ob_analysis.get('is_synthetic', False):
                        ob_confidence *= 0.7  # Lower confidence for synthetic data
                
                    signal_sources.append(("OrderBook", ob_confidence))
                    signal_explanation.append(f"Order book signal: {ob_signal} (conf: {ob_confidence:.2f})")
    
            # Check for anomalies that might impact trading
            if 'anomaly' in self.last_ob_analysis and self.last_ob_analysis['anomaly']:
                anomaly_info = self.last_ob_analysis['anomaly']
                if anomaly_info.get('recommendation', {}).get('action') in ['reduce_long', 'reduce_short', 'avoid_new_positions']:
                    signal_explanation.append(f"Warning: {anomaly_info.get('anomaly_type', 'unknown')} anomaly detected")
                    # Reduce confidence if anomaly suggests caution
                    signal_confidence *= 0.8

        # Integrate cross-asset signals if available
        cross_asset_signal = "NONE"
        cross_asset_confidence = 0.0
        if hasattr(self, 'components') and self.components.get("cross_asset_mgr") and hasattr(self.components["cross_asset_mgr"], 'get_correlation_report'):
            try:
                # Get correlation report
                corr_report = self.components["cross_asset_mgr"].get_correlation_report(detailed=True)
        
                # Check for strong price movements in correlated assets
                if "price_movements" in corr_report:
                    bullish_count = 0
                    bearish_count = 0
                    total_weight = 0
            
                    for symbol, movements in corr_report["price_movements"].items():
                        # Get correlation weight for this symbol
                        weight = corr_report.get("weights", {}).get(symbol, 0.1)
                
                        # Check recent price movements
                        for tf, pct_change in movements.items():
                            if pct_change > 0.1:  # Strong bullish movement
                                bullish_count += weight
                            elif pct_change < -0.1:  # Strong bearish movement
                                bearish_count += weight
                    
                            total_weight += weight
            
                # Generate signal if there's a clear direction
                if total_weight > 0:
                    bullish_ratio = bullish_count / total_weight
                    bearish_ratio = bearish_count / total_weight
                
                    if bullish_ratio > 0.6 and bullish_ratio > bearish_ratio * 2:
                        cross_asset_signal = "LONG"
                        cross_asset_confidence = bullish_ratio
                    elif bearish_ratio > 0.6 and bearish_ratio > bullish_ratio * 2:
                        cross_asset_signal = "SHORT"
                        cross_asset_confidence = bearish_ratio
                    
                    # Add to signal sources
                    if cross_asset_signal != "NONE":
                        signal_sources.append(("CrossAsset", cross_asset_confidence * 0.5))  # Half weight
                        signal_explanation.append(f"Cross-asset signal: {cross_asset_signal} (conf: {cross_asset_confidence:.2f})")
            except Exception as e:
                logging.error(f"Error generating cross-asset signal: {str(e)}")
        
        # Calculate combined confidence
        if len(signal_sources) > 1:
            # Weighted average of all signal confidences
            total_weight = sum(weight for _, weight in signal_sources)
            signal_confidence = sum(conf * weight for source, conf in signal_sources) / total_weight
        
        # Get risk parameters if a signal exists
        risk_params = None
        if signal != "NONE" and signal_confidence > 0.5 and self.trade_manager:
            if atr_value is None:
                # Use default ATR value or estimate from EGARCH
                atr_value = max(0.001, self.sigma_id[state_idx, 0] * 100)  # Simple estimate
                
            if account_balance is None:
                # Try to get account balance from MT5
                try:
                    account_info = mt5.account_info()
                    account_balance = account_info.balance if account_info else 10000
                except:
                    account_balance = 10000  # Default value
            
            # Get trading parameters
            risk_params = self.trade_manager.calculate_risk_parameters(
                state_label,
                signal_confidence,
                current_price,
                atr_value,
                account_balance
            )
            
            if risk_params:
                # Override direction if needed
                risk_params["direction"] = signal
        
        return {
            "signal": signal,
            "confidence": signal_confidence,
            "state_idx": state_idx,
            "state_label": state_label,
            "sources": signal_sources,
            "explanation": signal_explanation,
            "risk_params": risk_params,
            "hybrid_signal": hybrid_signal,
            "memory_signal": memory_signal,
            "ob_signal": ob_signal,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_performance(self, trade_result, signal_info=None):
        """
        Update performance statistics based on trade results.
        
        Args:
            trade_result: Dict with trade outcome information
            signal_info: Original signal information that generated the trade
            
        Returns:
            dict: Updated performance metrics
        """
        if self.performance_tracker is None:
            return None
        
        # Extract trade information
        state = trade_result.get("state", self.current_state)
        profit_pips = trade_result.get("profit_pips", 0)
        win = profit_pips > 0
        
        # Update state performance
        updated_metrics = self.performance_tracker.add_trade(
            state,
            trade_result.get("entry_price", 0),
            trade_result.get("exit_price", 0),
            trade_result.get("direction", "LONG"),
            trade_result.get("duration", 0),
            profit_pips,
            trade_result.get("entry_time"),
            trade_result.get("exit_time")
        )
        
        # Update component-specific performance
        # 1. Update market memory if available
        if self.market_memory and hasattr(self, 'memory_prediction') and signal_info:
            state_label = signal_info.get("state_label", "")
            self.market_memory.update_state_performance(state_label, win, profit_pips)
        
        # 2. Update feature selector if available
        if self.feature_selector and signal_info and "features" in signal_info:
            features = signal_info.get("features")
            state_label = signal_info.get("state_label", "")
            self.feature_selector.update_importance(features, state_label, win)
        
        # 3. Save performance statistics periodically
        self.performance_tracker.save_to_file()
        if self.state_history:
            self.state_history.save_to_file()
        
        return updated_metrics
    
    def analyze_current_market(self, features_tminus1, features_t, current_price, market_data=None):
        """
        Comprehensive market analysis combining all components.
        
        Args:
            features_tminus1: Features at t-1
            features_t: Current features
            current_price: Current market price
            market_data: Additional market data (optional)
            
        Returns:
            dict: Comprehensive market analysis
        """
        # 1. Update HMM state
        state_info = self.partial_step(features_tminus1, features_t, 
                                       datetime.now(), current_price)
        
        # 2. Generate trading signal
        signal = self.get_composite_signal(
            current_price=current_price,
            atr_value=features_t[6] if len(features_t) > 6 else None,  # Assuming atr_30m is at index 6
            prev_feature=features_tminus1
        )
        
        # 3. Analyze order book if available
        ob_analysis = None
        if self.order_book_analyzer and market_data and "order_book" in market_data:
            ob_analysis = self.order_book_analyzer.calculate_liquidity(market_data["order_book"])
        
        # 4. Get similar historical patterns if market memory available
        similar_patterns = []
        if self.market_memory:
            # Combine t-1 and t features for sequence
            feature_seq = np.vstack([features_tminus1, features_t])
            similar_patterns = self.market_memory.find_similar_patterns(feature_seq, n_neighbors=3)
        
        # 5. Get market trends from state history
        market_trends = None
        if self.state_history:
            market_trends = self.state_history.detect_trends(window=20)
        
        # 6. Assemble comprehensive analysis
        analysis = {
            "state": state_info,
            "signal": signal,
            "order_book": ob_analysis,
            "similar_patterns": similar_patterns,
            "market_trends": market_trends,
            "time": datetime.now().isoformat(),
            "price": current_price
        }
        
        # 7. Generate human-readable summary
        summary_lines = [
            f"State: {state_info['state_label']} (conf: {state_info['state_confidence']:.2f})",
            f"Signal: {signal['signal']} (conf: {signal['confidence']:.2f})",
        ]
        
        if "risk_params" in signal and signal["risk_params"]:
            risk = signal["risk_params"]
            summary_lines.append(f"Position size: {risk.get('position_size', 0)} lots")
            summary_lines.append(f"Stop loss: {risk.get('stop_loss', 0)}, Take profit: {risk.get('take_profit', 0)}")
            summary_lines.append(f"Risk/Reward: {risk.get('risk_reward', 0)}")
        
        if market_trends:
            summary_lines.append(f"Dominant state: {market_trends.get('dominant_state', 'Unknown')}")
            summary_lines.append(f"Market stability: {market_trends.get('stability', 0):.2f}")
        
        analysis["summary"] = "\n".join(summary_lines)
        
        return analysis

###############################################################################
# Configuration Constants
###############################################################################

# Feature selection & component configuration
USE_FEATURE_SELECTION = True  # Enable dynamic feature selection
USE_MARKET_MEMORY = True      # Enable market memory for pattern matching
USE_HYBRID_MODEL = True       # Enable hybrid neural network model
USE_ORDER_BOOK = True         # Enable order book analysis

# Weights for hybrid model contribution
HYBRID_CONTRIBUTION_WEIGHT = 0.5

# Time intervals
SIGNAL_UPDATE_INTERVAL = 60    # Generate new signals every 60 seconds
REPORT_INTERVAL = 3600         # Generate reports every hour
DATA_UPDATE_INTERVAL = 5       # Update market data every 5 seconds
OB_UPDATE_INTERVAL = 15        # Update order book every 15 seconds

# Data storage paths
REPORT_DIR = "reports"
SIGNAL_HISTORY_FILE = "signal_history.json"
VISUALIZATION_DIR = "visualizations"

# Ensure directories exist
for directory in [REPORT_DIR, VISUALIZATION_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

###############################################################################
# Advanced Market Analysis Functions
###############################################################################

def analyze_regime_characteristics(state_history, performance_tracker):
    """
    Analyzes the characteristics of each market regime (state).
    
    Args:
        state_history: StateHistory object
        performance_tracker: PerformanceTracker object
        
    Returns:
        dict: Regime analysis results
    """
    if not state_history or not state_history.states:
        return {}
    
    # Get state frequency
    state_freq = state_history.get_state_frequency()
    
    # Get state performance
    state_metrics = performance_tracker.get_metrics_all_states() if performance_tracker else {}
    
    # Analyze each state/regime
    regimes = {}
    
    for state, freq in state_freq.items():
        state_idx = int(state)
        
        # Get performance metrics for this state
        metrics = state_metrics.get(state_idx, {})
        
        # Calculate regime characteristics
        characteristics = {
            "frequency": freq,
            "win_rate": metrics.get("win_rate", 0),
            "avg_profit_pips": metrics.get("avg_profit_pips", 0),
            "trade_count": metrics.get("trade_count", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "avg_duration": metrics.get("avg_duration", 0),
        }
        
        # Add to regimes
        regimes[state_idx] = characteristics
    
    # Calculate transition probabilities
    transitions = {}
    states_list = list(state_history.states)
    
    for i in range(1, len(states_list)):
        from_state = states_list[i-1]
        to_state = states_list[i]
        key = f"{from_state}->{to_state}"
        
        if key not in transitions:
            transitions[key] = 0
        transitions[key] += 1
    
    # Normalize transitions
    for key in transitions:
        from_state = key.split("->")[0]
        from_count = states_list.count(int(from_state))
        if from_count > 0:
            transitions[key] = transitions[key] / from_count
    
    return {
        "regimes": regimes,
        "transitions": transitions
    }

def analyze_market_conditions(df_features, live_hmm, current_state_idx):
    """
    Analyzes current market conditions relative to historical patterns.
    
    Args:
        df_features: DataFrame with current features
        live_hmm: EnhancedLiveHMMMt5 instance
        current_state_idx: Current state index
        
    Returns:
        dict: Market condition analysis
    """
    if df_features.empty:
        return {}
    
    # Get current feature values
    current_row = df_features.iloc[-1]
    
    # Analyze volatility
    current_atr = current_row.get('atr_30m', 0)
    atr_percentile = 50  # Default middle percentile
    
    if 'atr_30m' in df_features.columns:
        atr_history = df_features['atr_30m'].dropna()
        if len(atr_history) > 10:
            atr_percentile = int(percentileofscore(atr_history, current_atr))
    
    # Analyze returns
    current_returns = []
    return_percentiles = {}
    
    for col in ['log_return30', 'log_return5', 'log_return1h', 'log_return4h']:
        if col in df_features.columns:
            value = current_row.get(col, 0)
            current_returns.append(value)
            
            history = df_features[col].dropna()
            if len(history) > 10:
                percentile = int(percentileofscore(history, value))
                return_percentiles[col] = percentile
    
    # Get current session
    current_session = "Unknown"
    for session in ['session_asia', 'session_europe', 'session_us', 'session_overlap']:
        if session in df_features.columns and current_row.get(session, 0) > 0.5:
            current_session = session.replace('session_', '')
    
    # Get state interpretation
    state_label, market_phase, vol_level = live_hmm._interpret_state(current_state_idx)
    
    return {
        "volatility": {
            "current_atr": current_atr,
            "percentile": atr_percentile,
            "level": vol_level
        },
        "returns": {
            "current": current_returns,
            "percentiles": return_percentiles
        },
        "session": current_session,
        "market_phase": market_phase,
        "state_label": state_label
    }

def analyze_signal_quality(signal, state_info, performance_tracker=None):
    """
    Analyzes the quality and reliability of a trading signal.
    
    Args:
        signal: Signal information from get_composite_signal
        state_info: State information from partial_step
        performance_tracker: PerformanceTracker instance
    
    Returns:
        dict: Signal quality analysis
    """
    if not signal or not state_info:
        return {"quality": 0.0, "explanation": ["No signal or state information"]}
    
    # Base quality on confidence
    quality = signal.get("confidence", 0.0)
    
    # Factors affecting signal quality
    quality_factors = []
    
    # 1. State validity
    state_validity = state_info.get("state_validity", 1.0)
    if state_validity < 0.5:
        quality_factors.append((-0.2, f"Low state validity ({state_validity:.2f})"))
    
    # 2. State confidence
    state_confidence = state_info.get("state_confidence", 0.0)
    if state_confidence > 0.8:
        quality_factors.append((0.1, f"High state confidence ({state_confidence:.2f})"))
    
    # 3. Component agreement
    signal_direction = signal.get("signal", "NONE")
    hybrid_signal = signal.get("hybrid_signal", "NONE")
    memory_signal = signal.get("memory_signal", "NONE")
    
    if signal_direction != "NONE":
        if signal_direction == hybrid_signal and hybrid_signal != "NONE":
            quality_factors.append((0.15, "Hybrid model confirms signal direction"))
        elif hybrid_signal != "NONE" and hybrid_signal != signal_direction:
            quality_factors.append((-0.1, "Hybrid model contradicts signal direction"))
        
        if signal_direction == memory_signal and memory_signal != "NONE":
            quality_factors.append((0.1, "Market memory confirms signal direction"))
        elif memory_signal != "NONE" and memory_signal != signal_direction:
            quality_factors.append((-0.15, "Market memory contradicts signal direction"))
    
    # 4. Historical performance of this state
    if performance_tracker:
        state_idx = state_info.get("state_idx", -1)
        metrics = performance_tracker.get_state_summary(state_idx)
        
        if metrics:
            win_rate = metrics.get("win_rate", 0.5)
            
            if win_rate > 0.6:
                quality_factors.append((0.1, f"State has good historical performance (WR: {win_rate:.2f})"))
            elif win_rate < 0.4:
                quality_factors.append((-0.15, f"State has poor historical performance (WR: {win_rate:.2f})"))
    
    # Apply all quality factors
    base_quality = quality
    for factor, explanation in quality_factors:
        quality += factor
    
    # Ensure quality is between 0 and 1
    quality = max(0.0, min(1.0, quality))
    
    # Generate explanation
    explanations = [f"Base quality: {base_quality:.2f} (signal confidence)"]
    for factor, explanation in quality_factors:
        explanations.append(f"{'+' if factor > 0 else ''}{factor:.2f}: {explanation}")
    
    return {
        "quality": quality,
        "factors": quality_factors,
        "explanation": explanations,
        "base_quality": base_quality
    }

def prepare_feature_fusion_data(df_features, components, main_features):
    """
    Bereitet Daten für die Feature-Fusion vor, inkl. Cross-Asset und Order Book Features.
    Stellt die Dimensionskonsistenz für maximale Qualität der Feature-Fusion sicher.
    
    Args:
        df_features: DataFrame mit Hauptfeatures
        components: Dictionary mit initialisierten Komponenten
        main_features: Hauptfeatures-Matrix
    
    Returns:
        tuple: (cross_features, order_book_features) mit standardisierten Dimensionen
    """
    # Überprüfe Eingabevalidität mit intelligenter Fehlerbehandlung
    if main_features is None or len(main_features) == 0:
        logging.warning("Leere Hauptfeatures - standardisierte Null-Features werden zurückgegeben")
        # Gib leere Arrays mit korrekter Form zurück statt None
        return np.zeros((0, 10)), np.zeros((0, 20))

    # Sichere Dimensionskonsistenz der Hauptfeatures
    expected_feature_dim = len(df_features.columns) if isinstance(df_features, pd.DataFrame) else main_features.shape[1]
    main_features = ensure_feature_dimensions(main_features, expected_feature_dim)
    
    # Bestimme Anzahl der Samples für konsistente Dimensions-Rückgabe
    num_samples = len(main_features)
    
    # Initialisiere mit leeren Arrays der korrekten Form
    cross_features = np.zeros((num_samples, 10))  # Standarddimension für Cross-Asset
    order_features = np.zeros((num_samples, 20))  # Standarddimension für Order Book
    
    # Extrahiere Cross-Asset-Features, falls verfügbar
    if 'cross_asset_mgr' in components and hasattr(components['cross_asset_mgr'], 'get_cross_asset_features'):
        try:
            # Für jeden Zeitpunkt Features extrahieren
            temp_cross_features = []
            
            for i, row in df_features.iterrows():
                timestamp = row['time'] if 'time' in row else None
                cross_asset_feats = components['cross_asset_mgr'].get_cross_asset_features(timestamp)
                
                # Konvertiere Dict zu Liste mit fester Reihenfolge
                cross_feat_list = []
                if isinstance(cross_asset_feats, dict):
                    for key in sorted(cross_asset_feats.keys()):
                        cross_feat_list.append(cross_asset_feats[key])
                else:
                    # Fallback für unerwartete Typen
                    logging.warning(f"Unerwarteter Cross-Asset-Feature-Typ: {type(cross_asset_feats)}")
                    cross_feat_list = [0.0] * 10  # Standardwerte
                
                # QUALITÄTSVERBESSERUNG: Standardisiere die Feature-Dimension
                if len(cross_feat_list) > 10:
                    # Zu viele Features - wähle die wichtigsten (ersten 10) aus
                    logging.debug(f"Reduziere Cross-Asset-Features von {len(cross_feat_list)} auf 10")
                    cross_feat_list = cross_feat_list[:10]
                elif len(cross_feat_list) < 10:
                    # Zu wenige Features - erweitere mit Nullen
                    padding = [0.0] * (10 - len(cross_feat_list))
                    cross_feat_list.extend(padding)
                
                temp_cross_features.append(cross_feat_list)
            
            if temp_cross_features:
                cross_features = np.array(temp_cross_features)
                
                # QUALITÄTSVERBESSERUNG: Verifiziere die endgültige Form explizit
                if cross_features.shape[0] != num_samples:
                    # Dimensionskonflikt - korrigiere auf die richtige Länge
                    logging.warning(f"Cross-Asset-Feature-Länge ({cross_features.shape[0]}) stimmt nicht mit der Hauptfeature-Länge ({num_samples}) überein")
                    if cross_features.shape[0] > num_samples:
                        cross_features = cross_features[:num_samples]
                    else:
                        # Erweitere mit Nullen für fehlende Samples
                        padding = np.zeros((num_samples - cross_features.shape[0], 10))
                        cross_features = np.vstack([cross_features, padding])
                
                # Stelle sicher, dass die zweite Dimension immer 10 ist
                if cross_features.shape[1] != 10:
                    logging.warning(f"Cross-Asset-Feature-Breite ({cross_features.shape[1]}) auf 10 angepasst")
                    temp = np.zeros((cross_features.shape[0], 10))
                    # Kopiere so viele Daten wie möglich
                    copy_cols = min(cross_features.shape[1], 10)
                    temp[:, :copy_cols] = cross_features[:, :copy_cols]
                    cross_features = temp
        except Exception as e:
            logging.error(f"Fehler beim Extrahieren von Cross-Asset-Features: {str(e)}")
            # Behalte die Standard-Nullmatrix bei
    
    # Generiere Order Book Features, falls aktiviert
    if 'synthetic_ob_generator' in components and components['synthetic_ob_generator']:
        try:
            # Erstelle eine intelligente Sammlung von Order-Book-Features
            temp_order_features = []
            
            # Kalibriere Generator mit Preisdaten wenn möglich
            if len(df_features) > 30:  # Mindestens 30 Datenpunkte für eine sinnvolle Kalibrierung
                try:
                    components['synthetic_ob_generator'].calibrate_with_price_data(df_features)
                    logging.debug("Order-Book-Generator erfolgreich kalibriert")
                except Exception as e:
                    logging.warning(f"OB-Generator-Kalibrierung fehlgeschlagen: {str(e)}")
            
            # Generiere für jeden Zeitpunkt ein Order Book
            for i, row in df_features.iterrows():
                # Aktuelle Marktbedingungen bestimmen
                volatility = "medium"  # Standardwert
                trend = "neutral"      # Standardwert
                
                if 'rsi_30m' in row:
                    volatility = "high" if row['rsi_30m'] > 70 or row['rsi_30m'] < 30 else "medium"
                    
                if 'log_return30' in row:
                    trend = "bullish" if row['log_return30'] > 0.001 else "bearish" if row['log_return30'] < -0.001 else "neutral"
                
                # Stelle sicher, dass wir einen gültigen Preis haben
                current_price = None
                for price_col in ['close', 'open', 'price']:
                    if price_col in row and not pd.isna(row[price_col]):
                        current_price = row[price_col]
                        break
                
                if current_price is not None:
                    # Synthetisches Order Book generieren
                    ob = components['synthetic_ob_generator'].generate_order_book(
                        current_price=current_price,
                        market_state=trend
                    )
                    
                    # Extrahiere Features aus Order Book
                    if 'order_book_processor' in components and components['order_book_processor']:
                        ob_features = components['order_book_processor'].extract_features(ob)
                        
                        # Konvertiere zu Liste mit fester Reihenfolge
                        ob_feat_list = []
                        if isinstance(ob_features, dict):
                            for key in sorted(ob_features.keys()):
                                ob_feat_list.append(ob_features[key])
                        else:
                            # Fallback für unerwartete Typen
                            ob_feat_list = [0.0] * 20  # Standardwerte
                        
                        # QUALITÄTSVERBESSERUNG: Standardisiere Feature-Dimension
                        if len(ob_feat_list) > 20:
                            # Zu viele Features - wähle die wichtigsten aus
                            logging.debug(f"Reduziere Order-Book-Features von {len(ob_feat_list)} auf 20")
                            ob_feat_list = ob_feat_list[:20]
                        elif len(ob_feat_list) < 20:
                            # Zu wenige Features - erweitere mit Nullen
                            padding = [0.0] * (20 - len(ob_feat_list))
                            ob_feat_list.extend(padding)
                        
                        temp_order_features.append(ob_feat_list)
                    else:
                        # Kein Prozessor verfügbar - verwende Standardnullen
                        temp_order_features.append([0.0] * 20)
                else:
                    # Kein Preis verfügbar - verwende Standardnullen
                    temp_order_features.append([0.0] * 20)
            
            if temp_order_features:
                order_features = np.array(temp_order_features)
                
                # QUALITÄTSVERBESSERUNG: Verifiziere die endgültige Form explizit
                if order_features.shape[0] != num_samples:
                    # Dimensionskonflikt - korrigiere auf die richtige Länge
                    logging.warning(f"Order-Book-Feature-Länge ({order_features.shape[0]}) stimmt nicht mit der Hauptfeature-Länge ({num_samples}) überein")
                    if order_features.shape[0] > num_samples:
                        order_features = order_features[:num_samples]
                    else:
                        # Erweitere mit Nullen für fehlende Samples
                        padding = np.zeros((num_samples - order_features.shape[0], 20))
                        order_features = np.vstack([order_features, padding])
                
                # Stelle sicher, dass die zweite Dimension immer 20 ist
                if order_features.shape[1] != 20:
                    logging.warning(f"Order-Book-Feature-Breite ({order_features.shape[1]}) auf 20 angepasst")
                    temp = np.zeros((order_features.shape[0], 20))
                    # Kopiere so viele Daten wie möglich
                    copy_cols = min(order_features.shape[1], 20)
                    temp[:, :copy_cols] = order_features[:, :copy_cols]
                    order_features = temp
        except Exception as e:
            logging.error(f"Fehler bei der Generierung von Order-Book-Features: {str(e)}")
            # Behalte die Standard-Nullmatrix bei
    
    # Endgültige Dimensionsprüfung
    logging.debug(f"Feature-Dimensionen: main={main_features.shape}, cross={cross_features.shape}, order={order_features.shape}")
    
    return cross_features, order_features

def train_feature_fusion_model(components, main_features, cross_features, order_features, df_features):
    """
    Trainiert das Feature-Fusion-Modell.
    
    Args:
        components: Dictionary mit initialisierten Komponenten
        main_features: Hauptfeatures-Matrix
        cross_features: Cross-Asset-Features-Matrix
        order_features: Order-Book-Features-Matrix
        df_features: DataFrame mit Hauptfeatures
    
    Returns:
        bool: Erfolg des Trainings
    """
    if 'feature_fusion' not in components or not components['feature_fusion']:
        return False
    
    logging.info("Training Feature Fusion model...")
    
    # Erstelle Target-Features (einfach: zukünftige Returns)
    future_returns = None
    if 'log_return30' in df_features.columns:
        future_returns = df_features['log_return30'].shift(-1).values[:-1]
        
        # Beschränke Features auf gleiche Länge
        main_subset = main_features[:-1]
        cross_subset = cross_features[:-1] if len(cross_features) > 0 else cross_features
        order_subset = order_features[:-1] if len(order_features) > 0 else order_features
    else:
        main_subset = main_features
        cross_subset = cross_features
        order_subset = order_features
    
    # Trainiere das Modell
    result = components['feature_fusion'].fit(
        main_features=main_subset,
        cross_asset_features=cross_subset,
        order_book_features=order_subset,
        target_features=future_returns,
        epochs=50,
        batch_size=32,
        validation_split=0.2
    )
    
    if result.get("success", False):
        logging.info("Feature Fusion model successfully trained")
        
        # Speichere das Modell
        components['feature_fusion'].save_model()
        return True
    else:
        logging.warning(f"Feature Fusion training failed: {result.get('error', 'unknown error')}")
        return False

###############################################################################
# Robust Data Collection Functions
###############################################################################

def collect_all_timeframe_data(symbol, connection_retry=3):
    """
    Collects data for all timeframes with robust error handling.
    
    Args:
        symbol: Trading symbol
        connection_retry: Number of retries for connection issues
        
    Returns:
        dict: Data for all timeframes
    """
    for attempt in range(connection_retry):
        try:
            # Check MT5 connection
            if not check_mt5_connection():
                logging.warning(f"MT5 connection check failed (attempt {attempt+1}/{connection_retry})")
                time.sleep(2)
                continue
            
            # Collect data for all timeframes
            rates30m = get_mt5_data_with_retry(symbol, TIMEFRAME_30M, COUNT_30M_LOOP)
            rates5m = get_mt5_data_with_retry(symbol, TIMEFRAME_5M, COUNT_5M_LOOP)
            rates1h = get_mt5_data_with_retry(symbol, TIMEFRAME_1H, COUNT_1H_LOOP)
            rates4h = get_mt5_data_with_retry(symbol, TIMEFRAME_4H, COUNT_4H_LOOP)
            rates1m = get_mt5_data_with_retry(symbol, TIMEFRAME_1M, COUNT_1M_LOOP)
            
            # Verify we have essential data
            if rates30m is None or len(rates30m) < 10 or rates5m is None or len(rates5m) < 10:
                logging.warning(f"Essential data missing (attempt {attempt+1}/{connection_retry})")
                time.sleep(2)
                continue
            
            # Get order book if enabled
            order_book = None
            if USE_ORDER_BOOK:
                try:
                    book = mt5.market_book_get(symbol)
                    if book:
                        order_book = book
                except Exception as e:
                    logging.warning(f"Error getting order book: {str(e)}")
            
            # Get current price
            current_price = None
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    current_price = (tick.bid + tick.ask) / 2
            except Exception as e:
                logging.warning(f"Error getting current price: {str(e)}")
            
            return {
                "30m": rates30m,
                "5m": rates5m,
                "1h": rates1h,
                "4h": rates4h,
                "1m": rates1m,
                "order_book": order_book,
                "current_price": current_price,
                "timestamp": datetime.now()
            }
                
        except Exception as e:
            logging.error(f"Error collecting data (attempt {attempt+1}/{connection_retry}): {str(e)}")
            time.sleep(2)
    
    # If we get here, all attempts failed
    logging.error(f"Failed to collect data after {connection_retry} attempts")
    return None

def prepare_features(market_data, cross_asset_mgr=None, feature_cols=None, apply_pca=False):
    """
    Prepares features from market data with enhanced cross-asset integration and
    robust error recovery.
    
    Args:
        market_data: Market data from collect_all_timeframe_data
        cross_asset_mgr: Optional CrossAssetManager
        feature_cols: Feature column names
        apply_pca: Whether to apply PCA
        
    Returns:
        tuple: (DataFrame with features, feature columns)
    """
    # Early validation with meaningful error handling
    if not market_data:
        logging.warning("No market data available for feature preparation")
        return None, feature_cols
    
    # Extract required data with explicit existence checks
    rates30m = market_data.get("30m")
    rates5m = market_data.get("5m")
    rates1h = market_data.get("1h")
    rates4h = market_data.get("4h")
    
    # Validate essential data
    if rates30m is None or rates5m is None:
        logging.warning("Missing essential timeframe data (30m or 5m)")
        return None, feature_cols
    
    # Quality improvement: Validate minimum data points for meaningful features
    if len(rates30m) < 5 or len(rates5m) < 5:
        logging.warning(f"Insufficient data points for feature calculation: 30m={len(rates30m)}, 5m={len(rates5m)}")
        return None, feature_cols
    
    # Initialize variables
    cross_asset_features = None
    pca_model = None
    scaler = None
    df_features = None  # Initialize explicitly to avoid reference errors

    # Compute features first
    df_features = compute_features_from_mt5_data(
        rates30m, rates5m, rates1h, rates4h,
        apply_pca=False  # Apply PCA later if needed
    )
    
    # Validate feature computation was successful
    if df_features is None or df_features.empty:
        logging.warning("Feature computation failed or returned empty DataFrame")
        return None, feature_cols
    
    # Now handle cross-asset features with robust error handling
    if cross_asset_mgr and hasattr(cross_asset_mgr, 'get_cross_asset_features'):
        try:
            # Timestamp safety: Now we have df_features guaranteed
            timestamp = datetime.now()
            
            # Use the most recent timestamp from data if available
            if 'time' in df_features.columns:
                # Take the latest time that's not in the future
                current_time = datetime.now()
                latest_time = df_features['time'].iloc[-1]
                
                # Ensure the timestamp is valid (not in future)
                if isinstance(latest_time, pd.Timestamp) and latest_time <= current_time:
                    timestamp = latest_time
            
            # Extract cross-asset features with timeout protection
            cross_asset_timeout = 5  # seconds
            start_time = time.time()
            
            cross_asset_features = cross_asset_mgr.get_cross_asset_features(timestamp)
            
            # Timeout check to prevent hanging
            if time.time() - start_time > cross_asset_timeout:
                logging.warning(f"Cross-asset feature extraction took too long ({time.time() - start_time:.1f}s)")
            
            # Log active cross-assets for diagnostics
            if hasattr(cross_asset_mgr, 'active_symbols'):
                active_symbols = list(cross_asset_mgr.active_symbols)
                logging.debug(f"Active cross-assets: {active_symbols}")
        
            # Check for PCA model components
            if apply_pca:
                if hasattr(cross_asset_mgr, 'pca_model') and hasattr(cross_asset_mgr, 'scaler'):
                    pca_model = cross_asset_mgr.pca_model
                    scaler = cross_asset_mgr.scaler
                elif hasattr(cross_asset_mgr, 'models') and 'pca_model' in cross_asset_mgr.models:
                    pca_model = cross_asset_mgr.models['pca_model']
                    scaler = cross_asset_mgr.models.get('scaler')
                
                # Apply PCA if available
                if pca_model is not None and scaler is not None:
                    try:
                        # Extract features
                        features_matrix = df_features[feature_cols].values
                        
                        # Scale features
                        scaled_features = scaler.transform(features_matrix)
                        
                        # Apply PCA
                        reduced_features = pca_model.transform(scaled_features)
                        
                        # Add PCA features to DataFrame
                        for i in range(reduced_features.shape[1]):
                            df_features[f'pca_feature_{i}'] = reduced_features[:, i]
                            
                        logging.info(f"PCA successfully applied: {reduced_features.shape[1]} components")
                    except Exception as e:
                        logging.error(f"Error applying PCA: {str(e)}")
                        
        except Exception as e:
            logging.error(f"Error getting cross-asset features: {str(e)}")
            # Continue without cross-asset features
    
    # Add cross-asset features if available
    if cross_asset_features is not None:
        try:
            # Convert dictionary to DataFrame if needed
            if isinstance(cross_asset_features, dict):
                # Create DataFrame from dictionary
                time_col = pd.Series([timestamp])
                ca_df = pd.DataFrame([cross_asset_features])
            
                if not ca_df.empty:
                    ca_df['time'] = time_col
                    cross_asset_features = ca_df
        
            # Continue only if we have valid DataFrame
            if isinstance(cross_asset_features, pd.DataFrame) and not cross_asset_features.empty:
                # Get columns to merge, expanded for new cross-asset manager
                cross_cols = []
                for col in cross_asset_features.columns:
                    if col != 'time' and (
                        col.startswith('log_return_') or 
                        col.startswith('volatility_') or
                        col.startswith('momentum_') or
                        '_close_rel' in col or
                        '_corr_weight' in col or
                        '_atr' in col
                    ):
                        cross_cols.append(col)
            
                if cross_cols:
                    try:
                        # Ensure time column exists in both DataFrames
                        if 'time' in cross_asset_features.columns and 'time' in df_features.columns:
                            # Merge cross-asset features with main features
                            df_features = pd.merge_asof(
                                df_features.sort_values('time'),
                                cross_asset_features[['time'] + cross_cols].sort_values('time'),
                                on='time', direction='backward'
                            )
                        else:
                            # Alternative approach if time column is missing
                            # Add cross-asset features as additional columns
                            for col in cross_cols:
                                if len(cross_asset_features[col]) > 0:
                                    df_features[col] = cross_asset_features[col].iloc[0]
                    
                        # Add cross-asset columns to feature_cols
                        if feature_cols:
                            new_features = [col for col in cross_cols if col not in feature_cols]
                            if new_features:
                                feature_cols = feature_cols + new_features
                                logging.info(f"Added {len(new_features)} cross-asset features")
                    except Exception as e:
                        logging.error(f"Error merging cross-asset features: {str(e)}")
                        # Continue without these features
        except Exception as e:
            logging.error(f"Error processing cross-asset features: {str(e)}")
    
    # Ensure all feature columns exist
    if feature_cols:
        for col in feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0
    
    return df_features, feature_cols

###############################################################################
# Visualization and Reporting Functions
###############################################################################

def generate_signal_visualization(signal_history, state_history, save_path=None):
    """
    Generates a visualization of signal history and states.
    
    Args:
        signal_history: List of signal dictionaries
        state_history: StateHistory instance
        save_path: Path to save the visualization
        
    Returns:
        Path to saved visualization or None
    """
    if not signal_history or not state_history or not state_history.states:
        return None
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: State history
    ax1 = plt.subplot(3, 1, 1)
    states = list(state_history.states)
    confidences = list(state_history.confidences)
    timestamps = [datetime.fromisoformat(ts) for ts in state_history.timestamps]
    
    ax1.plot(timestamps, states, 'o-', markersize=4)
    ax1.set_ylabel('State')
    ax1.set_title('HMM State History')
    ax1.grid(True)
    
    # Plot 2: Signal history
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    
    # Extract signal data
    signal_times = []
    signal_values = []
    signal_confidences = []
    
    for signal in signal_history:
        if "timestamp" not in signal or "signal" not in signal:
            continue
            
        try:
            timestamp = datetime.fromisoformat(signal["timestamp"])
        except:
            continue
            
        signal_times.append(timestamp)
        
        # Convert signal to numeric value
        if signal["signal"] == "LONG":
            signal_values.append(1)
        elif signal["signal"] == "SHORT":
            signal_values.append(-1)
        else:
            signal_values.append(0)
            
        signal_confidences.append(signal.get("confidence", 0.5))
    
    if signal_times:
        # Plot signals
        ax2.plot(signal_times, signal_values, 'o-', color='blue')
        ax2.set_ylabel('Signal')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['SHORT', 'NONE', 'LONG'])
        ax2.grid(True)
        
        # Plot signal confidence
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(signal_times, signal_confidences, 'r-')
        ax3.set_ylabel('Confidence')
        ax3.set_ylim(0, 1)
        ax3.grid(True)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def generate_performance_summary(performance_tracker, symbol):
    """
    Generates a summary of trading performance.
    
    Args:
        performance_tracker: PerformanceTracker instance
        symbol: Trading symbol
        
    Returns:
        str: Performance summary
    """
    if not performance_tracker:
        return "No performance data available."
    
    # Update metrics
    performance_tracker.update_metrics()
    all_metrics = performance_tracker.get_metrics_all_states()
    
    # Summary
    summary = f"# Performance Summary for {symbol}\n\n"
    summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Overall statistics
    total_trades = sum(metrics.get('trade_count', 0) for metrics in all_metrics.values())
    total_profit = sum(metrics.get('total_profit_pips', 0) for metrics in all_metrics.values())
    
    if total_trades > 0:
        avg_profit = total_profit / total_trades if total_trades != 0 else 0 # Avoid division by zero
        win_counts = sum(metrics.get('win_rate', 0) * metrics.get('trade_count', 0) 
                        for metrics in all_metrics.values())
        overall_win_rate = win_counts / total_trades if total_trades > 0 else 0
        
        summary += f"## Overall Performance\n\n"
        summary += f"- Total trades: {total_trades}\n"
        summary += f"- Total profit (pips): {total_profit:.1f}\n"
        summary += f"- Average profit per trade (pips): {avg_profit:.1f}\n"
        summary += f"- Overall win rate: {overall_win_rate:.2f}\n\\n"
    else:
        summary += "No trades recorded yet.\\n\\n"
    
    # State performance
    if all_metrics:
        summary += f"## Performance by State\\n\\n"
        
        for state, metrics in all_metrics.items():
            if metrics.get('trade_count', 0) == 0:
                continue
                
            summary += f"### State {state}\\n\\n"
            summary += f"- Trades: {metrics.get('trade_count', 0)}\\n"
            summary += f"- Win rate: {metrics.get('win_rate', 0):.2f}\\n"
            summary += f"- Average profit (pips): {metrics.get('avg_profit_pips', 0):.1f}\\n"
            summary += f"- Total profit (pips): {metrics.get('total_profit_pips', 0):.1f}\\n"
            summary += f"- Profit factor: {metrics.get('profit_factor', 0):.2f}\\n\\n"
    
    # Recent signals
    top_transitions = performance_tracker.get_top_transitions(top_n=3)
    if top_transitions:
        summary += "## Top State Transitions\\n\\n"
        for transition, data in top_transitions:
            # Check if 'probs' exists and is not empty before calculating mean
            avg_prob_str = f"Avg. probability: {np.mean(data['probs']):.3f}" if data.get('probs') else "Avg. probability: N/A"
            summary += f"- {transition}: {data['count']} times, {avg_prob_str}\\n"
    
    # Removed the global 'ensemble' check here
    # If ensemble metrics are needed, they should be passed as an argument explicitly

    return summary

def save_signal_history(signal_history, filename=SIGNAL_HISTORY_FILE):
    """
    Save signal history to a file.
    
    Args:
        signal_history: List of signal dictionaries
        filename: Output filename
    """
    # Ensure the directory exists - but only if there's a directory part
    dirname = os.path.dirname(filename)
    if dirname:  # Nur ausführen, wenn ein Verzeichnisname existiert
        os.makedirs(dirname, exist_ok=True)
    
    try:
        with open(filename, 'w') as f:
            json.dump(signal_history, f, indent=2, cls=NumpyEncoder)
    except Exception as e:
        logging.error(f"Error saving signal history: {str(e)}")

def load_signal_history(filename=SIGNAL_HISTORY_FILE, max_signals=500):
    """
    Loads signal history from a file.
    
    Args:
        filename: Input file
        max_signals: Maximum number of signals to keep
        
    Returns:
        list: Signal history
    """
    if not os.path.exists(filename):
        return []
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Limit the number of signals
        if len(data) > max_signals:
            data = data[-max_signals:]
            
        return data
    except Exception as e:
        logging.error(f"Error loading signal history: {str(e)}")
        return []

def load_hybrid_wrapper(output_dir="enhanced_model_full"):
    """
    Verbesserte Version: Lädt den Hybrid-Wrapper mit TensorFlow-Kompatibilität.
    
    Args:
        output_dir: Hauptverzeichnis mit dem Wrapper-Pfad
        
    Returns:
        object: HybridWrapper-Objekt für Vorhersagen oder None bei Fehler
    """
    import os
    import logging
    import pickle
    import json
    
    wrapper_path = os.path.join(output_dir, 'hybrid_wrapper.pkl')
    
    if not os.path.exists(wrapper_path):
        logging.warning(f"Hybrid wrapper not found at {wrapper_path}")
        
        # Versuchen, einen Wrapper dynamisch zu erzeugen, falls Model-Dateien existieren
        model_dir = os.path.join(output_dir, "hybrid_model")
        config_path = os.path.join(model_dir, "hybrid_config.json")
        
        if os.path.exists(config_path):
            try:
                # Lade Konfiguration
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                
                # Lade HMM-Modellparameter
                model_file = os.path.join(output_dir, "enhanced_hmm_model.pkl")
                if os.path.exists(model_file):
                    with open(model_file, "rb") as f:
                        model_data = pickle.load(f)
                    
                    # Extrahiere HMM-Parameter
                    if "model" in model_data:
                        model_params = model_data["model"]
                        feature_cols = model_data.get("feature_cols", model_config.get("feature_cols", []))
                    else:
                        model_params = model_data
                        feature_cols = model_params.get("feature_cols", model_config.get("feature_cols", []))
                    
                    # Stelle sicher, dass model_params alle nötigen Schlüssel hat
                    for key in ["K", "pi", "A", "st_params"]:
                        if key not in model_params:
                            if key == "K" and "st_params" in model_params:
                                model_params["K"] = len(model_params["st_params"])
                            else:
                                logging.error(f"Missing required key in model_params: {key}")
                                return None
                    
                    # Erzeuge einen neuen Wrapper
                    wrapper = HybridWrapper(
                        model_params=model_params,
                        model_paths=model_config.get("model_paths", {}),
                        feature_cols=feature_cols
                    )
                    
                    logging.info(f"Successfully created hybrid wrapper from config at {config_path}")
                    return wrapper
                else:
                    logging.error(f"HMM model file not found at {model_file}")
                    return None
                    
            except Exception as e:
                logging.error(f"Error creating hybrid wrapper from config: {str(e)}")
                return None
    
    # Normal das gespeicherte Wrapper-Objekt laden, wenn es existiert
    try:
        with open(wrapper_path, 'rb') as f:
            wrapper = pickle.load(f)
        
        logging.info(f"Hybrid wrapper loaded from {wrapper_path}")
            
        return wrapper
    except Exception as e:
        logging.error(f"Error loading hybrid wrapper: {str(e)}")
        return None

###############################################################################
# Main Loop and System Functions
###############################################################################

def _create_fallback_hybrid_model(feature_cols, K):
    """
    Erstellt ein Fallback-Hybrid-Modell mit minimaler Funktionalität,
    wenn das echte HybridModel nicht verfügbar ist.
    
    Args:
        feature_cols: Liste der Feature-Namen
        K: Anzahl der HMM-Zustände
        
    Returns:
        object: Fallback-Hybrid-Modell mit benötigter API
    """
    class FallbackHybridModel:
        """Minimale Implementierung des Hybrid-Modells für Fallback-Betrieb"""
        
        def __init__(self, feature_cols, hmm_states):
            self.feature_cols = feature_cols
            self.hmm_states = hmm_states
            self.sequence_length = 10  # Standard
            self.input_dim = len(feature_cols)
            logging.info(f"Fallback hybrid model created with {self.input_dim} features and {hmm_states} states")
        
        def predict_direction(self, features, hmm_state):
            """Einfache Richtungsvorhersage basierend auf historischen Daten"""
            # Einfache Heuristik: Verwende erste Feature (normalerweise ein Return)
            if isinstance(features, np.ndarray) and features.size > 0:
                if len(features.shape) > 1 and features.shape[0] > 0 and features.shape[1] > 0:
                    # Nehme Durchschnitt des ersten Features (normalerweise Returns)
                    avg_return = np.mean(features[:, 0])
                    if avg_return > 0.0003:  # Kleiner positiver Schwellenwert
                        return {"signal": "LONG", "confidence": min(0.5, abs(avg_return * 200))}
                    elif avg_return < -0.0003:  # Kleiner negativer Schwellenwert
                        return {"signal": "SHORT", "confidence": min(0.5, abs(avg_return * 200))}
            
            # Standard: Kein Signal
            return {"signal": "NONE", "confidence": 0.0}
            
        def predict_volatility(self, features, hmm_state):
            """Einfache Volatilitätsvorhersage basierend auf historischen Daten"""
            # Bestimme ob Volatilität hoch, normal oder niedrig ist
            if isinstance(features, np.ndarray) and features.size > 0:
                if len(features.shape) > 1 and features.shape[0] > 0 and features.shape[1] > 3:
                    # Verwende Rückgabe-Features für Volatilitätsschätzung
                    returns = features[:, :4]  # Erste 4 Features sollten Returns sein
                    volatility = np.std(returns)
                    
                    # Einfache Volatilitätsklassifikation
                    is_increasing = False
                    if features.shape[0] > 5:
                        # Vergleiche erste und zweite Hälfte der Sequenz
                        vol_first = np.std(returns[:features.shape[0]//2])
                        vol_second = np.std(returns[features.shape[0]//2:])
                        is_increasing = vol_second > vol_first * 1.2  # 20% Anstieg
                    
                    # Return als Format: [decreasing, stable, increasing]
                    probs = [0.1, 0.8, 0.1]  # Standardwerte
                    
                    if is_increasing:
                        probs = [0.1, 0.3, 0.6]  # Steigende Volatilität
                    elif volatility > 0.002:  # Hohe Volatilität
                        probs = [0.2, 0.3, 0.5]  # Wahrscheinlicher weiter steigend
                    elif volatility < 0.0005:  # Niedrige Volatilität
                        probs = [0.3, 0.6, 0.1]  # Wahrscheinlich stabil/sinkend
                    
                    return probs
            
            # Standard: Stabile Volatilität
            return [0.2, 0.6, 0.2]
        
        def get_action_recommendation(self, features, hmm_state, context=None):
            """Einfache Aktionsempfehlung basierend auf Vorhersagen"""
            direction = self.predict_direction(features, hmm_state)
            return direction
        
        def combine_predictions(self, hmm_prediction, direction_pred, volatility_pred, action_rec=None):
            """Kombiniert verschiedene Vorhersagen zu einem Gesamtergebnis"""
            # Verwende einfach die Richtungsvorhersage als Hauptsignal
            result = {
                "signal": direction_pred.get("signal", "NONE"),
                "signal_strength": direction_pred.get("confidence", 0.0),
                "combined_confidence": direction_pred.get("confidence", 0.0) * 0.8,  # Reduziere Konfidenz
                "hmm_state_idx": hmm_prediction.get("state_idx", 0),
                "hmm_state_label": hmm_prediction.get("state_label", "Unknown")
            }
            
            return result
        
        def load_models(self, path):
            """Simuliert das Laden von Modellen"""
            logging.info(f"Fallback hybrid model simulating model loading from {path}")
            return False
            
        def build_models(self):
            """Simuliert den Aufbau von Modellen"""
            logging.info("Fallback hybrid model simulating model building")
            return True
    
    # Erstelle und gib das Fallback-Modell zurück
    return FallbackHybridModel(feature_cols, K)

def initialize_components(model_params, feature_cols, dims_egarch=None):
    global USE_HYBRID_MODEL  # Fügen Sie diese Zeile hinzu
    """
    Initializes all system components.
    
    Args:
        model_params: HMM model parameters
        feature_cols: Feature column names
        dims_egarch: EGARCH dimensions
        
    Returns:
        dict: Initialized components
    """
    components = {}
    
    # Get state count
    K = model_params["K"]
    
    # Initialize performance tracker
    components["performance_tracker"] = PerformanceTracker(K, load_file=PERFORMANCE_FILE)
    
    # Initialize state history
    components["state_history"] = StateHistory(load_file=STATE_HISTORY_FILE)
    
    # Initialize trade manager
    components["trade_manager"] = TradeManager(symbol=SYMBOL)
    
    # Initialize risk manager
    try:
        if AdaptiveRiskManager:
            try:
                account_info = mt5.account_info()
                account_balance = account_info.balance if account_info else 10000
            except:
                account_balance = 10000
                
            components["risk_manager"] = AdaptiveRiskManager(
                symbol=SYMBOL,
                pip_value=0.01,
                base_risk_pct=1.0,
                balance=account_balance,
                account_currency="USD"
            )
        else:
            components["risk_manager"] = components["trade_manager"]
    except Exception as e:
        logging.error(f"Error initializing risk manager: {str(e)}")
        components["risk_manager"] = components["trade_manager"]

    # Lade den Hybrid-Wrapper, falls verfügbar
    hybrid_wrapper = load_hybrid_wrapper(output_dir="enhanced_model_full")
    if hybrid_wrapper:
        logging.info("Hybrid wrapper loaded successfully")
        components["hybrid_wrapper"] = hybrid_wrapper
    else:
        components["hybrid_wrapper"] = None
    
    # Initialize market memory if enabled
    if USE_MARKET_MEMORY:
        try:
            components["market_memory"] = MarketMemory(
                max_patterns=1000,
                similarity_window=50,
                filename="market_memory.pkl"
            )
            logging.info("Market memory initialized")
        except Exception as e:
            logging.error(f"Error initializing market memory: {str(e)}")
            components["market_memory"] = None
    
    # Initialize hybrid model if enabled with robust error handling and fallback
    if USE_HYBRID_MODEL:
        # Qualitätsverbesserung: Überprüfe explizit, ob HybridModel importiert wurde
        if 'HybridModel' in globals() and globals()['HybridModel'] is not None:
            hybrid_model_config = {}
            hybrid_config_path = os.path.join("enhanced_model_full", "hybrid_model", "hybrid_config.json")
            try:
                # Lade die spezifische Konfiguration für das Hybridmodell
                if os.path.exists(hybrid_config_path):
                    with open(hybrid_config_path, 'r') as f:
                        hybrid_model_config = json.load(f)
                    logging.info(f"Loaded hybrid model configuration from {hybrid_config_path}")
                else:
                    logging.warning(f"Hybrid model config file not found at {hybrid_config_path}. Using defaults.")

                # Extrahiere Parameter oder verwende Defaults
                lstm_units = hybrid_model_config.get("lstm_units", 64)
                dense_units = hybrid_model_config.get("dense_units", 32)
                sequence_length = hybrid_model_config.get("sequence_length", 10)
                dropout_rate = hybrid_model_config.get("dropout_rate", 0.2)
                learning_rate = hybrid_model_config.get("learning_rate", 0.001)

                # Erstelle eine Instanz des Hybrid-Modells mit geladenen Parametern
                hybrid_model = HybridModel(
                    input_dim=len(feature_cols),
                    hmm_states=K,
                    lstm_units=lstm_units,
                    dense_units=dense_units,
                    sequence_length=sequence_length,
                    learning_rate=learning_rate
                )
            
                # Try to load existing models with expanded search
                # Entferne den Pfad, der 'CONFIG' benötigt
                model_paths = [
                    os.path.join("enhanced_model_full", "hybrid_model"),  # Direkter Pfad zum Trainingsoutput
                    "hybrid_models",                                     # Standard-Fallback
                ]
            
                hybrid_model_loaded = False
                for path in model_paths:
                    if os.path.exists(path):
                        try:
                            # Versuche Modelle zu laden
                            if hybrid_model.load_models(path):
                                components["hybrid_model"] = hybrid_model
                                logging.info(f"Hybrid model loaded from {path}")
                                hybrid_model_loaded = True
                                break
                        except Exception as e:
                            logging.warning(f"Failed to load hybrid model from {path}: {str(e)}")
            
                # Wenn kein Modell geladen wurde, baue ein neues
                if not hybrid_model_loaded:
                    # Erstelle Modelle in Memory, ohne TensorFlow-Fehler
                    try:
                        hybrid_model.build_models()
                        components["hybrid_model"] = hybrid_model
                        logging.info("New hybrid model initialized with in-memory models")
                    except Exception as e:
                        logging.error(f"Error building hybrid models: {str(e)}")
                        components["hybrid_model"] = None
            
                # Qualitätsverbesserung: Überprüfe, ob die benötigten Methoden verfügbar sind
                required_methods = ['predict_direction', 'predict_volatility']
                if components["hybrid_model"] is not None:
                    for method in required_methods:
                        if not hasattr(components["hybrid_model"], method):
                            # Methode fehlt - erstelle einen Fallback
                            setattr(components["hybrid_model"], method, 
                                   lambda *args, **kwargs: {"signal": "NONE", "confidence": 0.0})
                            logging.warning(f"Created fallback implementation for missing method: {method}")
        
            except Exception as e:
                logging.error(f"Error initializing hybrid model: {str(e)}")
                # Qualitätsverbesserung: Erstelle ein voll funktionsfähiges Fallback-Modell
                components["hybrid_model"] = _create_fallback_hybrid_model(feature_cols, K)
                logging.info("Using fallback hybrid model implementation")
        else:
            # HybridModel wurde nicht importiert oder ist None
            logging.warning("HybridModel not available - disabling hybrid model functionality")
            # Deaktiviere für diesen Lauf
            USE_HYBRID_MODEL = False
            components["hybrid_model"] = None
    
    # Initialize feature selector if enabled
    if USE_FEATURE_SELECTION:
        try:
            components["feature_selector"] = DynamicFeatureSelector(
                feature_cols,
                history_length=100
            )
            logging.info("Feature selector initialized")
        except Exception as e:
            logging.error(f"Error initializing feature selector: {str(e)}")
            components["feature_selector"] = None
    
    # Initialize order book analyzer if enabled
    if USE_ORDER_BOOK:
        try:
            components["order_book_analyzer"] = OrderBookAnalyzer(
                symbol=SYMBOL,
                depth=10,
                history_length=100
            )
            logging.info("Order book analyzer initialized")
        except Exception as e:
            logging.error(f"Error initializing order book analyzer: {str(e)}")
            components["order_book_analyzer"] = None

    # Initialize order book analyzer if enabled
    if USE_ORDER_BOOK:
        try:
            if ANOMALY_DETECTION_AVAILABLE:
                # Initialize anomaly detector
                components["order_book_anomaly_detector"] = OrderBookAnomalyDetector(
                    history_size=ORDER_BOOK_HISTORY_SIZE,
                    contamination=0.05,
                    use_pca=True,
                    n_components=10,  # Added explicit PCA component setting
                    model_path=ORDER_BOOK_MODEL_PATH
                )
                
                # Initialize the detector with some basic data to ensure it's properly fitted
                if components["synthetic_ob_generator"] is not None:
                    try:
                        # Generate some synthetic order books for initial training
                        initial_data = []
                        for i in range(50):  # Generate 50 samples
                            price = 150.0 + (i / 100.0)  # Sample price range
                            ob = components["synthetic_ob_generator"].generate_order_book(
                                current_price=price,
                                market_state="neutral"
                            )
                            if components["order_book_processor"] is not None:
                                features = components["order_book_processor"].extract_features(ob)
                                if features:
                                    initial_data.append(features)
                        
                        # Add this data to the anomaly detector
                        if initial_data:
                            components["order_book_anomaly_detector"].add_orderbook_features(initial_data)
                            components["order_book_anomaly_detector"].fit()
                            logging.info(f"Order book anomaly detector initialized with {len(initial_data)} synthetic samples")
                    except Exception as e:
                        logging.warning(f"Could not initialize anomaly detector with synthetic data: {str(e)}")
            
                # Initialize change detector
                components["order_book_change_detector"] = OrderBookChangeDetector(
                    history_size=ORDER_BOOK_HISTORY_SIZE,
                    window_size=5
                )
            
                # Try to load existing model
                if components["order_book_anomaly_detector"].load_model():
                    logging.info("Order book anomaly detection model loaded successfully")
                else:
                    logging.info("No pre-trained order book anomaly model found, will train on collected data")
        
            if ORDER_BOOK_AVAILABLE:
                # Initialize feature processor
                components["order_book_processor"] = OrderBookFeatureProcessor(
                    pip_size=0.01  # Default for GBPJPY, adjust for other symbols
                )
            
                # Initialize synthetic generator if enabled
                if USE_SYNTHETIC_ORDER_BOOK:
                    components["synthetic_ob_generator"] = SyntheticOrderBookGenerator(
                        base_symbol=SYMBOL,
                        pip_size=0.01,  # Default for GBPJPY
                        max_levels=ORDER_BOOK_MAX_LEVELS
                    )
                    logging.info("Synthetic order book generator initialized")
        
            logging.info("Order book analysis components initialized")
        except Exception as e:
            logging.error(f"Error initializing order book components: {str(e)}")
            components["order_book_anomaly_detector"] = None
            components["order_book_change_detector"] = None
            components["order_book_processor"] = None
            components["synthetic_ob_generator"] = None
    
    # Initialize cross-asset manager with enhanced version
    try:
        if USE_CROSS_ASSETS and CROSS_ASSET_AVAILABLE:
            # Define timeframes for cross-asset analysis
            cross_timeframes = [
                TIMEFRAME_30M,
                TIMEFRAME_1H,
                TIMEFRAME_4H
            ]
        
            # Initialize the manager with proper configuration 
            cross_asset_mgr = CrossAssetManager(
                main_symbol=SYMBOL,
                cross_assets=None,  # Auto-select based on main symbol
                timeframes=cross_timeframes,
                correlation_window=CROSS_ASSET_CORR_WINDOW,
                lead_lag_max=CROSS_ASSET_LEAD_LAG_MAX,
                update_correlation_interval=CROSS_ASSET_UPDATE_INTERVAL,
                data_dir=CROSS_ASSET_DATA_DIR
            )
        
            # Initialize data and correlations
            if cross_asset_mgr.initialize(lookback_days=7):
                components["cross_asset_mgr"] = cross_asset_mgr
                logging.info(f"Enhanced cross-asset manager initialized with {len(cross_asset_mgr.active_symbols)} active assets")
            
                # Generate visualization if directory exists
                if os.path.exists(REPORT_OUTPUT_DIR):
                    viz_path = os.path.join(REPORT_OUTPUT_DIR, "cross_asset_correlation.png")
                    if cross_asset_mgr.visualize_correlations(save_path=viz_path):
                        logging.info(f"Cross-asset correlation visualization saved to {viz_path}")
            else:
                components["cross_asset_mgr"] = None
                logging.warning("Cross-asset manager initialization failed")
        else:
            components["cross_asset_mgr"] = None
            if USE_CROSS_ASSETS and not CROSS_ASSET_AVAILABLE:
                logging.warning("Cross-asset features requested but not available")
    except Exception as e:
        logging.error(f"Error initializing cross-asset manager: {str(e)}")
        components["cross_asset_mgr"] = None

    # Initialize feature fusion if enabled
    if USE_FEATURE_FUSION and FEATURE_FUSION_AVAILABLE:
        try:
            # Determine feature sizes based on available data
            main_feature_size = len(feature_cols)
            cross_asset_feature_size = 10  # Default size, will be adjusted when cross assets are loaded
            order_book_feature_size = 20   # Default size for order book features
        
            components["feature_fusion"] = RegularizedFeatureFusion(
                main_feature_size=main_feature_size,
                cross_asset_feature_size=cross_asset_feature_size,
                order_book_feature_size=order_book_feature_size,
                output_feature_size=main_feature_size,  # Output same size as main features
                regularization=FEATURE_FUSION_REGULARIZATION,
                fusion_method=FEATURE_FUSION_METHOD,
                adaptive_weights=FEATURE_FUSION_ADAPTIVE,
                model_path=FEATURE_FUSION_MODEL_PATH
            )
        
            # Try to load existing model
            if os.path.exists(FEATURE_FUSION_MODEL_PATH):
                components["feature_fusion"].load_model()
                logging.info("Feature fusion model loaded successfully")
            else:
                logging.info("No existing feature fusion model found, will train on collected data")
            
        except Exception as e:
            logging.error(f"Error initializing feature fusion: {str(e)}")
            components["feature_fusion"] = None

    # NEUER CODE: Initialize signal weighting if enabled
    if USE_SIGNAL_WEIGHTING and SIGNAL_WEIGHTING_AVAILABLE:
        try:
            # Initialize with adjusted base weights to favor valid signals
            component_weights = {
                "hmm": 0.3,           # Reduced from 0.4 
                "hybrid_model": 0.4,   # Increased from 0.3
                "market_memory": 0.2,  # Kept the same
                "order_book": 0.1      # Kept the same
            }
            
            components["signal_weighting"] = AdaptiveSignalWeighting(
                components=ENSEMBLE_COMPONENTS,
                base_weights=component_weights,  # Using our explicit weights that favor hybrid model
                history_window=ENSEMBLE_HISTORY_SIZE,
                learning_rate=SIGNAL_WEIGHTING_LEARNING_RATE,
                state_specific=SIGNAL_WEIGHTING_STATE_SPECIFIC
            )
            
            # Add signal processor with reduced confidence threshold to generate more trades
            components["signal_processor"] = AdaptiveSignalProcessor(
                weighting_manager=components["signal_weighting"],
                confidence_threshold=0.3,  # Reduced from 0.5 to generate more trades
                history_window=ENSEMBLE_HISTORY_SIZE,
                consistency_window=1  # Reduced from default to make it easier to generate consistent signals
            )
            
            # Add domain adaptation layer for bridging synthetic and real data
            components["domain_adapter"] = DomainAdaptationLayer(
                feature_processor=None,  # Will be set when order book processor is initialized
                max_history=500
            )
            
            # Link domain adapter with order book processor if available
            if "order_book_processor" in components and components["order_book_processor"]:
                components["domain_adapter"].feature_processor = components["order_book_processor"]
            
            logging.info("Signal weighting components initialized")
        except Exception as e:
            logging.error(f"Error initializing signal weighting: {str(e)}")
            components["signal_weighting"] = None
            components["signal_processor"] = None
            components["domain_adapter"] = None
    
    # Initialize ensemble if enabled
    if USE_ENSEMBLE and ENSEMBLE_AVAILABLE:
        try:
            components["ensemble"] = AdaptiveComponentEnsemble(
                components=ENSEMBLE_COMPONENTS,
                initial_weights=None,  # Will use default equal weights
                history_size=ENSEMBLE_HISTORY_SIZE,
                learning_rate=SIGNAL_WEIGHTING_LEARNING_RATE,
                state_specific=SIGNAL_WEIGHTING_STATE_SPECIFIC,
                model_path=ENSEMBLE_MODEL_PATH
            )
            
            # Initialize backtester if feature fusion is available
            if "feature_fusion" in components and components["feature_fusion"]:
                components["fusion_backtester"] = FusionBacktester(
                    feature_fusion=components["feature_fusion"],
                    ensemble=components["ensemble"]
                )
            
            logging.info("Ensemble components initialized")
        except Exception as e:
            logging.error(f"Error initializing ensemble: {str(e)}")
            components["ensemble"] = None
            components["fusion_backtester"] = None
    
    # Initialisiere EnhancedLiveHMMMt5 mit allen Komponenten
    components["live_hmm"] = EnhancedLiveHMMMt5(
        model_params,
        dims_egarch=dims_egarch,
        feature_cols=feature_cols,
        performance_tracker=components["performance_tracker"],
        state_history=components["state_history"],
        trade_manager=components["risk_manager"],
        market_memory=components.get("market_memory"),
        hybrid_model=components.get("hybrid_model"),
        feature_selector=components.get("feature_selector"),
        order_book_analyzer=components.get("order_book_analyzer"),
        hybrid_wrapper=components.get("hybrid_wrapper"),
        order_book_anomaly_detector=components.get("order_book_anomaly_detector"),
        order_book_change_detector=components.get("order_book_change_detector"),
        order_book_processor=components.get("order_book_processor"),
        synthetic_ob_generator=components.get("synthetic_ob_generator")
    )
    
    # Setze feature_fusion direkt (falls verfügbar)
    if "feature_fusion" in components and components["feature_fusion"]:
        components["live_hmm"].feature_fusion = components["feature_fusion"]

    # Setze signal_weighting, signal_processor und ensemble direkt (falls verfügbar)
    if "signal_weighting" in components and components["signal_weighting"]:
        components["live_hmm"].signal_weighting = components["signal_weighting"]
    
    if "signal_processor" in components and components["signal_processor"]:
        components["live_hmm"].signal_processor = components["signal_processor"]
    
    if "domain_adapter" in components and components["domain_adapter"]:
        components["live_hmm"].domain_adapter = components["domain_adapter"]
    
    if "ensemble" in components and components["ensemble"]:
        components["live_hmm"].ensemble = components["ensemble"]
    
    if "fusion_backtester" in components and components["fusion_backtester"]:
        components["live_hmm"].fusion_backtester = components["fusion_backtester"]
    
    if "feature_fusion" in components and components["feature_fusion"]:
        components["live_hmm"].feature_fusion = components["feature_fusion"]

    return components

def catchup_offline(live_hmm, start_time, components=None):
    """
    Process historical data to bring the model up to date.
    Loads additional history before start_time to ensure indicator calculation.
    
    Args:
        live_hmm: EnhancedLiveHMMMt5 instance
        start_time: Starting time for catchup
        components: Optional component dictionary
        
    Returns:
        datetime: Timestamp of last processed candle
    """
    now_ = datetime.now()
    MIN_HISTORY_BARS = 50 # Number of extra bars to load for indicator history

    # Calculate start times including history buffer
    try:
        # Ensure start_time is timezone-naive or consistent with datetime.timedelta usage
        # If start_time is timezone-aware, careful conversion might be needed
        # Assuming start_time is naive UTC for simplicity here
        hist_start_time_m30 = start_time - timedelta(minutes=30 * MIN_HISTORY_BARS)
        hist_start_time_m5 = start_time - timedelta(minutes=5 * MIN_HISTORY_BARS)
        hist_start_time_h1 = start_time - timedelta(hours=1 * MIN_HISTORY_BARS)
        hist_start_time_h4 = start_time - timedelta(hours=4 * MIN_HISTORY_BARS)
        logging.info(f"Calculated history start times: M30={hist_start_time_m30}, M5={hist_start_time_m5}, H1={hist_start_time_h1}, H4={hist_start_time_h4}")
    except TypeError as e:
        logging.error(f"Error calculating history start times (likely timezone issue): {e}")
        logging.warning("Proceeding without history buffer due to time calculation error.")
        hist_start_time_m30 = start_time
        hist_start_time_m5 = start_time
        hist_start_time_h1 = start_time
        hist_start_time_h4 = start_time


    # Robust data retrieval with retries, using history start times
    logging.info(f"Fetching M30 data from {hist_start_time_m30} to {now_}")
    rates30_cu = get_mt5_rates_range_with_retry(SYMBOL, TIMEFRAME_30M, hist_start_time_m30, now_)
    logging.info(f"Fetching M5 data from {hist_start_time_m5} to {now_}")
    rates5_cu = get_mt5_rates_range_with_retry(SYMBOL, TIMEFRAME_5M, hist_start_time_m5, now_)
    logging.info(f"Fetching H1 data from {hist_start_time_h1} to {now_}")
    rates1h_cu = get_mt5_rates_range_with_retry(SYMBOL, TIMEFRAME_1H, hist_start_time_h1, now_)
    logging.info(f"Fetching H4 data from {hist_start_time_h4} to {now_}")
    rates4h_cu = get_mt5_rates_range_with_retry(SYMBOL, TIMEFRAME_4H, hist_start_time_h4, now_)
    
    # --- Rest of the function remains the same ---
    # Check if enough data was retrieved (focus on M30 and M5 as primary)
    # A minimum length check is still useful, e.g., at least MIN_HISTORY_BARS / 2
    min_len_check = MIN_HISTORY_BARS // 2 
    if rates30_cu is None or len(rates30_cu) < min_len_check or rates5_cu is None or len(rates5_cu) < min_len_check:
        logging.error(f"Insufficient primary data ({len(rates30_cu)} M30, {len(rates5_cu)} M5) even after requesting history buffer (min needed: {min_len_check}). Cannot proceed with catchup.")
        return None
    
    logging.info(f"Retrieved data points: M30={len(rates30_cu) if rates30_cu is not None else 0}, M5={len(rates5_cu) if rates5_cu is not None else 0}, H1={len(rates1h_cu) if rates1h_cu is not None else 0}, H4={len(rates4h_cu) if rates4h_cu is not None else 0}")

    # 1h and 4h are optional, check if they have *any* data
    if rates1h_cu is None or len(rates1h_cu) == 0:
        logging.info("No 1h candles available for catchup, using fallbacks.")
        rates1h_cu = None
    
    if rates4h_cu is None or len(rates4h_cu) == 0:
        logging.info("No 4h candles available for catchup, using fallbacks.")
        rates4h_cu = None

    # ... existing code ...

def main():
    """Main loop and system functions"""
    # Hier alle globalen Variablen deklarieren, die in der Funktion geändert werden sollen
    global USE_HYBRID_MODEL
    global ORDER_BOOK_MAX_LEVELS
    global REPORT_DIR
    global SIGNAL_HISTORY_FILE
    global VISUALIZATION_DIR
    
    logging.info("=== Starting Enhanced HMM Live Trading System ===")

    # Globales Berichtsverzeichnis erstellen
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    global_report_dir = os.path.join(REPORT_DIR, session_timestamp + "_session")
    if not os.path.exists(global_report_dir):
        os.makedirs(global_report_dir)
    logging.info(f"Erstelle globales Berichtsverzeichnis: {global_report_dir}")
    
    # Initialize MT5
    if not mt5.initialize():
        logging.error("MT5 Initialize failed.")
        return
    
    try:
        # Load HMM model
        try:
            model_file = MODEL_FILE
            if not os.path.exists(model_file):
                model_file = MODEL_FALLBACK_FILE
                logging.warning(f"Primary model file not found, using fallback: {model_file}")
            
            if not os.path.exists(model_file):
                logging.error(f"Neither primary nor fallback model file found. Aborting.")
                mt5.shutdown()
                return
            
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)
            
            # Extract model parameters
            if "model" in model_data:
                model_params = model_data["model"]
                feature_cols = model_data.get("feature_cols")
                dims_egarch = model_data.get("dims_egarch")
            else:
                # Older format
                model_params = model_data
                feature_cols = model_params.get("feature_cols")
                dims_egarch = DIMS_EGARCH  # Use default EGARCH dimensions

            # Fallback für Order Book Komponenten bei Training ohne diese
            if "components" in model_data and "order_book" in model_data["components"]:
                # Übernehme relevante Konfiguration aus Trainingsmodell
                if model_data["components"]["order_book"].get("enabled", False):
                    logging.info("Order book analysis was enabled during training")

                    # Falls Parameter gespeichert wurden, verwende diese
                    ob_params = model_data["components"]["order_book"].get("params", {})
                    if ob_params:
                        # Aktualisiere Konfiguration basierend auf Trainingsparametern
                        if "max_levels" in ob_params:
                            ORDER_BOOK_MAX_LEVELS = ob_params["max_levels"]
                            logging.info(f"Setting ORDER_BOOK_MAX_LEVELS to {ORDER_BOOK_MAX_LEVELS} from model")
            else:
                logging.info("No order book configuration found in model, using defaults")
            
            K = model_params.get("n_components", 4)
            
            if feature_cols:
                logging.info(f"Loaded HMM model K={K} with {len(feature_cols)} dimensions")
                logging.info(f"Features: {feature_cols}")
            else:
                logging.info(f"Loaded HMM model K={K}, but no feature_cols found")
                feature_cols = [
                    'log_return30', 'log_return5', 'log_return1h', 'log_return4h',
                    'rsi_30m', 'rsi_1h', 'atr_30m', 'atr_4h', 'macd_30m',
                    'session_asia', 'session_europe', 'session_us', 'session_overlap',
                    'log_volume',
                    'day_mon', 'day_tue', 'day_wed', 'day_thu', 'day_fri'
                ]
                
            # Use explicit DIMS_EGARCH if available
            if dims_egarch:
                logging.info(f"Using EGARCH dimensions from model: {dims_egarch}")
            else:
                dims_egarch = DIMS_EGARCH
                logging.info(f"Using default EGARCH dimensions: {dims_egarch}")
        except Exception as e:
            logging.exception(f"Error loading HMM model: {str(e)}")
            mt5.shutdown()
            return
        
        # Initialize components
        components = initialize_components(model_params, feature_cols, dims_egarch)
        live_hmm = components["live_hmm"]
        
        # Load previous signal history
        signal_history = load_signal_history()
        logging.info(f"Loaded {len(signal_history)} previous signals")
        
        # Check last candle time
        if os.path.exists(LAST_CANDLE_FILE):
            try:
                with open(LAST_CANDLE_FILE, "r") as f:
                    last_candle_str = f.read().strip()
                
                last_candle_time = datetime.fromisoformat(last_candle_str)
                logging.info(f"Last candle time found: {last_candle_time}")
            except Exception as e:
                logging.exception(f"Error loading last candle time: {str(e)}")
                last_candle_time = get_last_sunday_23h()
                logging.info(f"Fallback to Sunday 23:00: {last_candle_time}")
        else:
            logging.info("No last_candle_time file => fallback to Sunday 23:00.")
            last_candle_time = get_last_sunday_23h()
        
        # Offline catchup
        logging.info(f"Starting offline catchup from {last_candle_time} to now...")
        new_t = catchup_offline(live_hmm, last_candle_time, components)
        if new_t is not None:
            with open(LAST_CANDLE_FILE, "w") as f:
                f.write(new_t.isoformat())
            logging.info(f"Catchup done until {new_t}")
        else:
            logging.info("Catchup => no new or insufficient data => partial or none.")
        
        # --- Added HMM Initialization Check --- 
        if live_hmm.alpha is None:
            logging.warning("HMM state (alpha) is None after catchup. Attempting manual initialization.")
            try:
                # Fetch recent data specifically for initialization
                init_rates30 = get_mt5_data_with_retry(SYMBOL, TIMEFRAME_30M, 50)
                init_rates5 = get_mt5_data_with_retry(SYMBOL, TIMEFRAME_5M, 50)
                init_rates1h = get_mt5_data_with_retry(SYMBOL, TIMEFRAME_1H, 50)
                init_rates4h = get_mt5_data_with_retry(SYMBOL, TIMEFRAME_4H, 50)

                if init_rates30 is not None and len(init_rates30) > 1 and init_rates5 is not None and len(init_rates5) > 1:
                    # Use prepare_features to get a DataFrame and the correct feature columns
                    df_init, init_feature_cols = prepare_features(
                        {"30m": init_rates30, "5m": init_rates5, "1h": init_rates1h, "4h": init_rates4h},
                        cross_asset_mgr=components.get("cross_asset_mgr"),
                        feature_cols=feature_cols, # Pass original feature_cols
                        apply_pca=True # Or False, depending on config
                    )

                    if df_init is not None and not df_init.empty:
                        # Use the identified feature columns from prepare_features
                        feature_cols_to_use_init = init_feature_cols if init_feature_cols else feature_cols
                        expected_dim_init = len(feature_cols_to_use_init)
                        
                        # Ensure all columns exist in df_init
                        missing_cols_init = [col for col in feature_cols_to_use_init if col not in df_init.columns]
                        if missing_cols_init:
                            logging.warning(f"Missing features in init DataFrame: {missing_cols_init}")
                            for col in missing_cols_init:
                                df_init[col] = 0
                        
                        # Extract the first feature vector for initialization
                        row0_init = df_init.iloc[0]
                        feat0_init = np.array([row0_init[col] for col in feature_cols_to_use_init])
                        feat0_init_ensured = ensure_feature_dimensions(feat0_init, expected_dim_init)
                        
                        # Call partial_init
                        live_hmm.partial_init(feat0_init_ensured)
                        logging.info("HMM state manually initialized using recent historical data.")
                    else:
                        logging.error("Failed to compute features for manual HMM initialization.")
                else:
                    logging.error("Insufficient historical data for manual HMM initialization.")
            except Exception as init_e:
                logging.exception(f"Error during manual HMM initialization: {init_e}")
        # --- End Added HMM Initialization Check ---

        # Main loop variables
        last_time = None
        last_signal_time = datetime.now()
        last_report_time = datetime.now()
        last_visualization_time = datetime.now()
        last_check_time = datetime.now()
        last_ob_update_time = datetime.now()
        connection_error_count = 0
        current_features = None
        prev_features = None
        
        logging.info("Starting main loop for enhanced market analysis and signal generation")
        
        while True:
            try:
                # Check MT5 connection periodically
                current_time = datetime.now()
                if (current_time - last_check_time).total_seconds() > 300:  # Every 5 minutes
                    if not check_mt5_connection():
                        connection_error_count += 1
                        if connection_error_count >= MAX_CONNECTION_ERRORS:
                            logging.error(f"Too many connection errors ({connection_error_count}). Exiting.")
                            break
                    else:
                        connection_error_count = 0  # Reset on successful connection
                    last_check_time = current_time
                
                # Collect market data
                market_data = collect_all_timeframe_data(SYMBOL)

                if not market_data:
                    logging.warning("Failed to collect market data. Waiting.")
                    time.sleep(DATA_UPDATE_INTERVAL)
                    continue

                # Update order book if needed
                order_book_data = None
                if USE_ORDER_BOOK and (datetime.now() - last_ob_update_time).total_seconds() >= ORDER_BOOK_UPDATE_INTERVAL:
                    order_book_data = get_order_book_data(SYMBOL, max_depth=ORDER_BOOK_MAX_LEVELS)
    
                    # Generate synthetic order book if real one not available and synthetic generator initialized
                    if order_book_data is None and USE_SYNTHETIC_ORDER_BOOK and "synthetic_ob_generator" in components:
                        current_price = market_data.get("current_price")
                        if current_price is not None:
                            # Determine market state from recent HMM states if available
                            market_state = None
                            if components["state_history"] and len(components["state_history"].states) > 0:
                                recent_state = components["state_history"].states[-1]
                                state_label = live_hmm._interpret_state(recent_state)[0]
                
                                if "Bullish" in state_label:
                                    market_state = "trending_bull"
                                elif "Bearish" in state_label:
                                    market_state = "trending_bear"
                                else:
                                    market_state = "ranging"
            
                            order_book_data = generate_synthetic_order_book(
                                components["synthetic_ob_generator"],
                                current_price,
                                market_state
                            )
                            if order_book_data:
                                logging.debug("Using synthetic order book")
    
                    # Set last update time
                    last_ob_update_time = datetime.now()
                    market_data["order_book"] = order_book_data
                
                # Prepare features
                df_features, updated_feature_cols = prepare_features(
                    market_data, 
                    cross_asset_mgr=components.get("cross_asset_mgr"),
                    feature_cols=feature_cols,
                    apply_pca=True
                )
                
                if df_features is None or len(df_features) < 2:
                    logging.warning("Insufficient feature data. Waiting.")
                    time.sleep(DATA_UPDATE_INTERVAL)
                    continue
                
                # Get current and previous feature rows
                row_cur = df_features.iloc[-1]
                row_prev = df_features.iloc[-2]
                time_cur = row_cur['time']

                # Apply feature fusion if enabled
                if USE_FEATURE_FUSION and "feature_fusion" in components and components["feature_fusion"]:
                    try:
                        # Extract main features
                        main_features = df_features[feature_cols].values
        
                        # Prepare cross-asset and order book features
                        cross_features, order_features = prepare_feature_fusion_data(
                            df_features, components, main_features)
        
                        # Apply feature fusion
                        fused_features = components["feature_fusion"].fuse_features(
                            main_features[-1:], 
                            cross_features[-1:] if len(cross_features) > 0 else np.zeros((1, 10)),
                            order_features[-1:] if len(order_features) > 0 else np.zeros((1, 20))
                        )
        
                        # Replace current features with fused features
                        if len(fused_features.shape) > 1 and fused_features.shape[0] > 0:
                            for i, col in enumerate(feature_cols):
                                if i < fused_features.shape[1]:
                                    df_features.loc[df_features.index[-1], col] = fused_features[0, i]
        
                        logging.info("Feature fusion applied successfully")
                    except Exception as e:
                        logging.error(f"Error applying feature fusion: {str(e)}")
                
                # Check if we have a new candle
                if last_time is not None and time_cur == last_time:
                    # No new candle, just update order book if needed
                    if USE_ORDER_BOOK and components.get("order_book_analyzer") and market_data.get("order_book"):
                        if (datetime.now() - last_ob_update_time).total_seconds() >= OB_UPDATE_INTERVAL:
                            try:
                                components["order_book_analyzer"].get_order_book()
                                last_ob_update_time = datetime.now()
                            except Exception as e:
                                logging.error(f"Error updating order book: {str(e)}")
                    
                    time.sleep(DATA_UPDATE_INTERVAL)
                    continue
                
                # Update last candle time
                last_time = time_cur
                
                # Extract feature vectors
                feature_cols_to_use = updated_feature_cols if updated_feature_cols else feature_cols
                expected_dim = len(feature_cols_to_use) # Original expected dimension
                
                # Check for missing columns
                missing_features = [col for col in feature_cols_to_use if col not in df_features.columns]
                if missing_features:
                    logging.warning(f"Missing features in DataFrame: {missing_features}")
                    # Add missing features with 0 values
                    for col in missing_features:
                        df_features[col] = 0
                
                # Extract only needed features
                available_features = [col for col in feature_cols_to_use if col in df_features.columns]
                
                prev_features_full = np.array([row_prev[col] for col in available_features])
                current_features_full = np.array([row_cur[col] for col in available_features])

                # Ensure features match the expected dimension for HMM, especially after fusion
                prev_features = ensure_feature_dimensions(prev_features_full, expected_dim)
                current_features = ensure_feature_dimensions(current_features_full, expected_dim)
                
                # Get current price
                current_price = market_data.get("current_price")
                if current_price is None:
                    current_price = row_cur['close']
                
                # Update HMM state
                state_info = live_hmm.partial_step(
                    prev_features, 
                    current_features, 
                    time_info=time_cur, 
                    current_price=current_price,
                    order_book_data=market_data.get("order_book")
                )

                # Kalibriere synthetischen Order Book Generator, wenn genug Daten vorhanden
                if USE_ORDER_BOOK and USE_SYNTHETIC_ORDER_BOOK and "synthetic_ob_generator" in components:
                    if order_book_data is not None:
                        # Sammle Order Book Daten für Kalibrierung
                        if not hasattr(components, 'ob_data_buffer'):
                            components['ob_data_buffer'] = []
                            components['ob_calibration_needed'] = True
            
                        components['ob_data_buffer'].append(order_book_data)
        
                        # Kalibriere, wenn genug Daten vorhanden
                        if components.get('ob_calibration_needed', True) and len(components['ob_data_buffer']) >= ORDER_BOOK_CALIBRATION_SIZE:
                            logging.info(f"Calibrating synthetic order book generator with {len(components['ob_data_buffer'])} samples")
                            try:
                                components["synthetic_ob_generator"].calibrate_with_real_data(components['ob_data_buffer'])
                                # Kalibriere auch mit Preisdaten
                                try:
                                    # Prüfe, ob genügend Daten für die Kalibrierung vorhanden sind
                                    if df_features is not None and len(df_features) >= 20:
                                        # Finde und validiere Preisdaten
                                        price_cols = ['close', 'open', 'high', 'low']
                                        valid_price_cols = [col for col in price_cols if col in df_features.columns 
                                                          and not df_features[col].isna().all()]
        
                                        if valid_price_cols:
                                            # Kalibriere mit ausreichenden Daten
                                            components["synthetic_ob_generator"].calibrate_with_price_data(df_features)
                                            logging.info(f"Order Book Generator mit {len(df_features)} Datenpunkten kalibriert")
                                        else:
                                            # Fallback auf Standard-ATR und Spread
                                            if hasattr(components["synthetic_ob_generator"], 'set_default_calibration'):
                                                components["synthetic_ob_generator"].set_default_calibration(
                                                    atr_pips=15.0,  # Standardwert für GBPJPY
                                                    spread_pips=1.5  # Standardwert für GBPJPY
                                                )
                                                logging.info("Order Book Generator mit Standardwerten kalibriert (keine Preisdaten)")
                                            else:
                                                # Manuelle Setzen von Standardwerten, falls Methode nicht existiert
                                                components["synthetic_ob_generator"].atr_pips = 15.0
                                                components["synthetic_ob_generator"].spread_pips = 1.5
                                                logging.info("Order Book Generator mit Standardwerten kalibriert (manuell)")
                                    elif components["synthetic_ob_generator"] is not None:
                                        # Zu wenig Daten, verwende Standardwerte
                                        logging.warning(f"Nicht genug Preisdaten für OB-Kalibrierung ({len(df_features) if df_features is not None else 0} < 20)")
        
                                        # Setze Standardwerte je nach Verfügbarkeit der Methode
                                        if hasattr(components["synthetic_ob_generator"], 'set_default_calibration'):
                                            components["synthetic_ob_generator"].set_default_calibration(
                                                atr_pips=15.0,  # Standardwert für GBPJPY
                                                spread_pips=1.5  # Standardwert für GBPJPY
                                            )
                                        else:
                                            # Manuelle Setzen von Standardwerten
                                            components["synthetic_ob_generator"].atr_pips = 15.0
                                            components["synthetic_ob_generator"].spread_pips = 1.5
            
                                        logging.info("Order Book Generator mit Standardwerten kalibriert (zu wenig Daten)")
                                except Exception as e:
                                    logging.error(f"Fehler bei der Kalibrierung des Order Book Generators: {str(e)}")
                                    # Versuche dennoch, Standardwerte zu setzen
                                    try:
                                        if components["synthetic_ob_generator"] is not None:
                                            components["synthetic_ob_generator"].atr_pips = 15.0
                                            components["synthetic_ob_generator"].spread_pips = 1.5
                                            logging.info("Order Book Generator mit Notfall-Standardwerten kalibriert")
                                    except:
                                        pass  # Stillschweigende Fehler bei extremen Fällen
                
                                components['ob_calibration_needed'] = False
                                logging.info("Synthetic order book generator calibrated successfully")
                            except Exception as e:
                                logging.error(f"Error calibrating synthetic order book generator: {str(e)}")
                
                # Log basic state information
                msg = f"[{time_cur}] => State={state_info['state_idx']}, " \
                      f"Label={state_info['state_label']}, " \
                      f"Phase={state_info['market_phase']}, " \
                      f"Vol={state_info['volatility_level']}, " \
                      f"Confidence={state_info['state_confidence']:.2f}"
                
                print(msg)
                logging.info(msg)

                # Update cross-asset correlations periodically
                if USE_CROSS_ASSETS and "cross_asset_mgr" in components and components["cross_asset_mgr"]:
                    try:
                        # Check if update is needed (every 6 hours by default)
                        if hasattr(components["cross_asset_mgr"], 'update_correlations'):
                            # Update correlations automatically based on interval
                            components["cross_asset_mgr"].update_correlations()
                    except Exception as e:
                        logging.error(f"Error updating cross-asset correlations: {str(e)}")
                
                # Generate signals periodically
                if (datetime.now() - last_signal_time).total_seconds() >= SIGNAL_UPDATE_INTERVAL:
                    # Get account balance for position sizing
                    account_balance = 10000  # Default
                    try:
                        account_info = mt5.account_info()
                        if account_info:
                            account_balance = account_info.balance
                    except Exception as e:
                        logging.error(f"Error getting account balance: {str(e)}")
                    
                    # Update risk manager
                    if "risk_manager" in components:
                        components["risk_manager"].update_balance(account_balance)
                    
                    # Get ATR value for volatility adjustment
                    atr_value = row_cur.get('atr_30m', 0.001)
                    
                    # Generate integrated trading signal
                    signal = live_hmm.get_composite_signal(
                        current_price=current_price,
                        atr_value=atr_value,
                        account_balance=account_balance,
                        prev_feature=prev_features
                    )

                    # Wenn Ensemble aktiviert ist, resultate auswerten und Signalperformance tracken
                    if "ensemble" in components and components["ensemble"]:
                        # Prüfe auf abgeschlossene Trades
                        if "trade_manager" in components and hasattr(components["trade_manager"], "get_active_recommendations"):
                            active_trades = components["trade_manager"].get_active_recommendations()
        
                            for trade_idx, trade in enumerate(active_trades):
                                if trade.get("status") == "executed":
                                    # Prüfe, ob Trade geschlossen werden sollte basierend auf Ensemble-Signal
                                    if signal["signal"] != trade["direction"] and signal["confidence"] > 0.7:
                                        # Signal hat sich umgedreht mit hoher Konvidenz - Trade schließen
                                        exit_price = current_price
                                        profit_pips = None
                    
                                        if exit_price and trade.get("actual_entry") or trade.get("entry_price"):
                                            entry_price = trade.get("actual_entry", trade.get("entry_price"))
                                            pip_value = 0.01 if entry_price > 100 else 0.0001  # Näherung für Pip-Größe
                        
                                            if trade["direction"] == "LONG":
                                                profit_pips = (exit_price - entry_price) / pip_value
                                            else:  # SHORT
                                                profit_pips = (entry_price - exit_price) / pip_value
                    
                                        # Trade als geschlossen markieren
                                        closed_trade = components["trade_manager"].mark_trade_closed(
                                            trade_idx, exit_price, "ensemble_signal_reversal", datetime.now()
                                        )
                    
                                        if closed_trade and "ensemble" in components and components["ensemble"]:
                                            # Ensemble über den Trade-Ausgang informieren
                                            outcome = "success" if profit_pips > 0 else "failure"
                                            signal_idx = -1  # Wir nehmen an, dass das letzte Signal verwendet wurde
                        
                                            components["ensemble"].update_signal_outcome(signal_idx, outcome, profit_pips)
                        
                                            logging.info(f"Trade closed based on ensemble signal reversal: {profit_pips} pips ({outcome})")
                    
                    # Add timestamp and features to signal
                    signal["timestamp"] = datetime.now().isoformat()
                    
                    # Add signal to history
                    signal_history.append(signal)
                    
                    # Trim signal history if needed
                    if len(signal_history) > 500:
                        signal_history = signal_history[-500:]
                    
                    # Save signal history periodically
                    save_signal_history(signal_history)
                    
                    # Log signal information
                    signal_msg = f"Signal: {signal['signal']} (conf: {signal['confidence']:.2f})"
                    if "explanation" in signal:
                        signal_msg += f"\nExplanation: {signal['explanation']}"
                    
                    print(signal_msg)
                    logging.info(signal_msg)
                    
                    # Analyze signal quality
                    quality = analyze_signal_quality(signal, state_info, components["performance_tracker"])
                    logging.info(f"Signal quality: {quality['quality']:.2f}")
                    for explanation in quality.get('explanation', []):
                        logging.info(f"  {explanation}")
                    
                    # Show trading parameters if available
                    if "risk_params" in signal and signal["risk_params"]:
                        risk = signal["risk_params"]
                        risk_msg = f"Trade parameters: {risk.get('direction', 'NONE')} at {risk.get('entry_price', 0)}, " \
                                  f"TP: {risk.get('take_price', 0)}, SL: {risk.get('stop_loss', 0)}, " \
                                  f"R:R = {risk.get('risk_reward', 0)}, " \
                                  f"Size: {risk.get('position_size', 0)} lots, " \
                                  f"Risk: {risk.get('risk_percentage', 0)}%"
                        
                        print(risk_msg)
                        logging.info(risk_msg)

                    if "hybrid_wrapper_used" in signal and signal["hybrid_wrapper_used"]:
                        logging.info("This signal was generated using the hybrid wrapper") 
                    
                    last_signal_time = datetime.now()

                # Analyse des aktuellen Order Books
                if "order_book_processor" in components and market_data and "order_book" in market_data:
                    order_book = market_data["order_book"]
    
                    if order_book:
                        # Extract features
                        features = components["order_book_processor"].extract_features(order_book)
        
                        # Print key metrics
                        logging.info(f"Order Book Metrics: Spread={features.get('spread_pips', 0):.1f} pips, " +
                                    f"Imbalance={features.get('imbalance', 1.0):.2f}")
        
                        # Check for anomalies
                        if "order_book_anomaly_detector" in components:
                            anomaly_result = components["order_book_anomaly_detector"].detect_anomalies(features)
                            if anomaly_result["is_anomaly"]:
                                anomaly_type = components["order_book_anomaly_detector"].classify_anomaly_type(features)
                                logging.warning(f"Order Book Anomaly: {anomaly_type['type']} " +
                                              f"(confidence: {anomaly_type['confidence']:.2f})")
                
                # Generate periodic reports
                if (datetime.now() - last_report_time).total_seconds() >= REPORT_INTERVAL:
                    # Save all tracking data
                    components["performance_tracker"].save_to_file()
                    components["state_history"].save_to_file()
    
                    if hasattr(components["risk_manager"], 'save_state_performance'):
                        components["risk_manager"].save_state_performance()
    
                    # ÄNDERE DIESEN TEIL: Erstelle ein Update-spezifisches Berichtsverzeichnis
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_dir = os.path.join(global_report_dir, f"update_{timestamp_str}")
    
                    if not os.path.exists(report_dir):
                        os.makedirs(report_dir)
    
                    # Aktualisiere das "Letztes Update"-Verzeichnis für einfachen Zugriff
                    latest_dir = os.path.join(global_report_dir, "latest")
                    if os.path.exists(latest_dir) and os.path.islink(latest_dir):
                        try: # Added try-except for robustness
                           os.remove(latest_dir)  # Entferne den vorherigen symbolischen Link
                        except OSError as e:
                           logging.warning(f"Could not remove existing symlink {latest_dir}: {e}")
                    elif os.path.exists(latest_dir):
                        # Falls es ein reguläres Verzeichnis ist, umbenennen oder löschen
                        try:
                            shutil.rmtree(latest_dir)
                        except Exception as e: # Broader exception catch
                            logging.warning(f"Could not remove existing latest directory {latest_dir}: {e}")
                            old_latest = latest_dir + "_old_" + timestamp_str # Make backup name unique
                            if os.path.exists(old_latest):
                                try:
                                   shutil.rmtree(old_latest)
                                except Exception as e_old:
                                    logging.warning(f"Could not remove old backup {old_latest}: {e_old}")
                            try:
                               os.rename(latest_dir, old_latest)
                            except Exception as e_rename:
                                logging.warning(f"Could not rename latest directory {latest_dir} to {old_latest}: {e_rename}")

                    # Erstelle symbolischen Link auf neuestes Update-Verzeichnis
                    try:
                        # Ensure target directory exists before creating link
                        if os.path.exists(report_dir):
                           os.symlink(report_dir, latest_dir, target_is_directory=True)
                        else:
                           logging.warning(f"Target directory {report_dir} does not exist for symlink.")
                    except Exception as e: # Broader exception catch for symlink/copy
                        logging.warning(f"Could not create symlink {latest_dir} -> {report_dir}: {e}. Attempting copy.")
                        # Fallback für Systeme ohne Symlink-Unterstützung
                        try:
                            if os.path.exists(report_dir):
                               shutil.copytree(report_dir, latest_dir)
                            else:
                                logging.warning(f"Target directory {report_dir} does not exist for copy fallback.")
                        except Exception as e_copy:
                             logging.error(f"Failed to create latest directory link/copy: {e_copy}")

                    try:
                        # Generate visualization
                        vis_path = os.path.join(report_dir, "signal_state_history.png")
                        generate_signal_visualization(
                            signal_history,
                            components["state_history"],
                            save_path=vis_path
                        )
        
                        # Generate performance summary - corrected call
                        summary = generate_performance_summary(
                            components["performance_tracker"],
                            SYMBOL
                            # Removed: ensemble=components.get("ensemble")
                        )
        
                        with open(os.path.join(report_dir, "performance_summary.md"), "w") as f:
                            f.write(summary)
        
                        # Generate regime analysis
                        regime_analysis = analyze_regime_characteristics(
                            components["state_history"],
                            components["performance_tracker"]
                        )
        
                        with open(os.path.join(report_dir, "regime_analysis.json"), "w") as f:
                            json.dump(regime_analysis, f, indent=2, cls=NumpyEncoder) # Added NumpyEncoder
        
                        logging.info(f"Periodic reports generated and saved to {report_dir}")
                    except Exception as e:
                        logging.error(f"Error generating reports: {str(e)}") # Keep this error log
    
                    # ... (rest of the reporting section) ...

                    last_report_time = datetime.now()

                # ... (rest of the main loop) ...

                # Retrain feature fusion model periodically
                if USE_FEATURE_FUSION and "feature_fusion" in components and components["feature_fusion"]:
                    current_time = datetime.now()
                    if not hasattr(components, 'last_feature_fusion_train_time') or \
                       (current_time - components.get('last_feature_fusion_train_time', current_time - timedelta(days=2))).total_seconds() >= FEATURE_FUSION_RETRAIN_INTERVAL:
                        try:
                            logging.info("Scheduling feature fusion model retraining...")
            
                            # Collect enough data for training
                            historical_data = collect_all_timeframe_data(SYMBOL)
                            if historical_data:
                                # Prepare features
                                df_hist, _ = prepare_features(historical_data, components.get("cross_asset_mgr"), feature_cols)
                
                                if df_hist is not None and len(df_hist) > 50:  # Ensure enough data
                                    main_features = df_hist[feature_cols].values
                    
                                    # Prepare cross-asset and order book features
                                    cross_features, order_features = prepare_feature_fusion_data(
                                        df_hist, components, main_features)
                    
                                    # Train model
                                    success = train_feature_fusion_model(
                                        components, main_features, cross_features, order_features, df_hist)
                    
                                    if success:
                                        logging.info("Feature fusion model retrained successfully")
                    
                                    components['last_feature_fusion_train_time'] = current_time
                    
                                    # Save component timestamp
                                    with open(os.path.join(LOG_DIR, "last_feature_fusion_train.txt"), "w") as f:
                                        f.write(current_time.isoformat())
                        except Exception as e:
                            logging.error(f"Error retraining feature fusion model: {str(e)}")
                
                # Save last candle time
                with open(LAST_CANDLE_FILE, "w") as f:
                    f.write(time_cur.isoformat())
                
                # Delay between updates
                time.sleep(DATA_UPDATE_INTERVAL)
            
            except KeyboardInterrupt:
                logging.info("User interrupted execution")
                break
            except Exception as e:
                logging.exception(f"Exception in main loop: {str(e)}")
                time.sleep(DATA_UPDATE_INTERVAL)
                continue
        
        # Save all data before exiting
        try:
            # Save performance tracking data
            components["performance_tracker"].save_to_file()
            components["state_history"].save_to_file()
            components["trade_manager"].save_trade_history()
            
            if hasattr(components["risk_manager"], 'save_state_performance'):
                components["risk_manager"].save_state_performance()
                
            # Save cross-asset data if available
            if "cross_asset_mgr" in components and components["cross_asset_mgr"]:
                if hasattr(components["cross_asset_mgr"], 'save_models'):
                    components["cross_asset_mgr"].save_models()
                    
            # Save market memory if available
            if "market_memory" in components and components["market_memory"]:
                components["market_memory"].save_memory()

            # Save cross-asset data before exiting
            if "cross_asset_mgr" in components and components["cross_asset_mgr"]:
                try:
                    # Save final status
                    if hasattr(components["cross_asset_mgr"], '_save_status'):
                        components["cross_asset_mgr"]._save_status()
                        logging.info("Cross-asset manager status saved")
                except Exception as e:
                    logging.error(f"Error saving cross-asset manager status: {str(e)}")
                
            # Save signal history
            save_signal_history(signal_history)
                
            logging.info("All data saved before exit")
        except Exception as e:
            logging.error(f"Error saving data before exit: {str(e)}")
        
    except Exception as e:
        logging.exception(f"Critical error in main process: {str(e)}")
    finally:
        mt5.shutdown()
        logging.info("=== Enhanced HMM Live Trading System stopped ===")

if __name__ == "__main__":
    main()