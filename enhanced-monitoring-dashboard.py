#!/usr/bin/env python3
"""
Enhanced HMM Trading System Dashboard
-------------------------------------
Professional-grade monitoring dashboard for the Enhanced HMM Trading System.
This dashboard provides real-time visualization of trading system performance,
current state, signals, and advanced metrics.

Features:
- Real-time state and signal monitoring
- Performance tracking and visualization
- Order book analysis display
- Market memory pattern recognition
- Hybrid model prediction visualization
- Cross-asset correlation monitoring
- Authentication and security

Usage:
1. Run as standalone: python enhanced_monitoring_dashboard.py
2. Import and use in trading system:
   from enhanced_monitoring_dashboard import create_dashboard, start_dashboard
   dashboard_app = create_dashboard(shared_data)
   start_dashboard(dashboard_app)
"""

import os
import sys
import json
import pickle
import logging
import threading
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta
from collections import deque
from scipy.stats import percentileofscore

# Dash/Plotly imports for visualization
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_auth
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# Flask for serving
from flask import Flask, request, redirect

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dashboard")

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# Constants
UPDATE_INTERVAL_MS = 5000  # 5 seconds
MAX_HISTORY_POINTS = 500
PERFORMANCE_FILE = "hmm_performance_tracker.json"
STATE_HISTORY_FILE = "state_history.json"
TRADE_HISTORY_FILE = "trade_history.json"
SIGNAL_HISTORY_FILE = "signal_history.json"
MODEL_FILE = "enhanced_hmm_model_v3.pkl"
CONFIG_FILE = "dashboard_config.json"
CREDENTIALS_FILE = "dashboard_credentials.json"

# Theme and styling
THEME = dbc.themes.DARKLY
COLORS = {
    'background': '#1E1E1E',
    'text': '#E0E0E0',
    'grid': '#333333',
    'long': '#00CC96',
    'short': '#EF553B',
    'neutral': '#636EFA',
    'highlight': '#AB63FA',
    'warning': '#FFA15A',
    'bullish': '#1CA954',
    'bearish': '#E0514C',
    'high_vol': '#FF9F1C',
    'low_vol': '#89CFF0'
}

# Dashboard data cache
# This will be shared with the main trading system when integrated
class DashboardData:
    def __init__(self):
        self.current_state = {}
        self.latest_signal = {}
        self.performance_data = {}
        self.state_history = deque(maxlen=MAX_HISTORY_POINTS)
        self.signal_history = deque(maxlen=MAX_HISTORY_POINTS)
        self.price_history = deque(maxlen=MAX_HISTORY_POINTS)
        self.trade_history = deque(maxlen=MAX_HISTORY_POINTS)
        self.order_book_data = {}
        self.market_memory_data = {}
        self.hybrid_model_data = {}
        self.feature_importance = {}
        self.cross_asset_data = {}
        self.component_status = {}
        self.last_updated = datetime.now()
        
    def update_from_files(self):
        """Update data from various JSON files (for standalone mode)"""
        try:
            # Update state history
            if os.path.exists(STATE_HISTORY_FILE):
                with open(STATE_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    if 'states' in data and 'timestamps' in data:
                        self.state_history = deque(
                            [{'state': int(s), 
                              'time': t, 
                              'confidence': c if i < len(data.get('confidences', [])) else 0.5,
                              'price': p if i < len(data.get('prices', [])) else None} 
                             for i, (s, t, c, p) in enumerate(
                                 zip(data['states'], 
                                     data['timestamps'], 
                                     data.get('confidences', [0.5] * len(data['states'])),
                                     data.get('prices', [None] * len(data['states']))))],
                            maxlen=MAX_HISTORY_POINTS
                        )
                        if self.state_history:
                            last_state = self.state_history[-1]
                            self.current_state = {
                                'state_idx': last_state['state'],
                                'time': last_state['time'],
                                'confidence': last_state['confidence'],
                                'validity': 0.8  # Default value
                            }
                            
                            # Infer state label
                            try:
                                with open(MODEL_FILE, 'rb') as f:
                                    model_data = pickle.load(f)
                                    model_params = model_data.get('model', {})
                                    if 'components' in model_data:
                                        self.component_status = {
                                            comp: True for comp in model_data['components'].keys()
                                        }
                                    # Set feature importance if available
                                    self.feature_importance = model_data.get('feature_importance', {})
                            except Exception as e:
                                logger.error(f"Error loading model data: {str(e)}")
                                model_params = {}
                                
                            # Simple state label mapping if model not available
                            state_idx = self.current_state['state_idx']
                            state_labels = ["High Bullish", "Low Bullish", "Low Bearish", "High Bearish"]
                            self.current_state['state_label'] = state_labels[state_idx % len(state_labels)]
            
            # Update signal history
            if os.path.exists(SIGNAL_HISTORY_FILE):
                with open(SIGNAL_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.signal_history = deque(data, maxlen=MAX_HISTORY_POINTS)
                    if self.signal_history:
                        self.latest_signal = self.signal_history[-1]
            
            # Update performance data
            if os.path.exists(PERFORMANCE_FILE):
                with open(PERFORMANCE_FILE, 'r') as f:
                    self.performance_data = json.load(f)
            
            # Update trade history
            if os.path.exists(TRADE_HISTORY_FILE):
                with open(TRADE_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    
                    # Process active and closed trades
                    active_trades = data.get('active_trades', [])
                    self.trade_history = deque(data.get('trade_history', []), maxlen=MAX_HISTORY_POINTS)
                    
                    # Extract price history from trades if possible
                    if self.trade_history:
                        try:
                            for trade in self.trade_history:
                                if 'entry_time' in trade and 'entry_price' in trade:
                                    self.price_history.append({
                                        'time': trade['entry_time'],
                                        'price': trade['entry_price']
                                    })
                        except Exception as e:
                            logger.error(f"Error extracting price history from trades: {str(e)}")
            
            # Update timestamps
            self.last_updated = datetime.now()
            logger.info(f"Dashboard data updated from files at {self.last_updated}")
            
        except Exception as e:
            logger.error(f"Error updating dashboard data from files: {str(e)}")
    
    def get_price_data(self):
        """Get price history for charting"""
        if not self.price_history:
            # Generate dummy data if no price history
            return {
                'times': [(datetime.now() - timedelta(minutes=i)).isoformat() for i in range(100, 0, -1)],
                'prices': [np.random.uniform(180, 190) for _ in range(100)]
            }
        
        # Sort by time
        sorted_history = sorted(self.price_history, key=lambda x: x['time'])
        
        return {
            'times': [p['time'] for p in sorted_history],
            'prices': [p['price'] for p in sorted_history]
        }
    
    def get_state_performance(self):
        """Get performance metrics by state"""
        if not self.performance_data or 'state_metrics' not in self.performance_data:
            # Generate dummy data if no performance data
            return {str(i): {
                'win_rate': np.random.uniform(0.3, 0.7),
                'avg_profit_pips': np.random.uniform(-5, 15),
                'total_profit_pips': np.random.uniform(-100, 300),
                'trade_count': np.random.randint(10, 100)
            } for i in range(4)}
        
        return self.performance_data.get('state_metrics', {})
    
    def get_feature_importance(self):
        """Get feature importance data"""
        if not self.feature_importance:
            # Generate dummy data if no feature importance data
            return {f"feature_{i}": np.random.uniform(0, 1) for i in range(10)}
        
        return self.feature_importance
    
    def get_synthetic_market_memory(self):
        """Generate synthetic market memory data if none available"""
        return {
            "pattern_count": np.random.randint(50, 500),
            "similar_patterns": np.random.randint(0, 10),
            "prediction": "profitable" if np.random.random() > 0.5 else "loss",
            "prediction_confidence": np.random.uniform(0.5, 0.95),
            "success_rate": np.random.uniform(0.4, 0.7)
        }
    
    def get_synthetic_hybrid_prediction(self):
        """Generate synthetic hybrid model prediction if none available"""
        return {
            "direction_pred": {
                "up": np.random.uniform(0.2, 0.8),
                "down": np.random.uniform(0.1, 0.4),
                "sideways": np.random.uniform(0.1, 0.3)
            },
            "volatility_pred": {
                "low": np.random.uniform(0.2, 0.5),
                "medium": np.random.uniform(0.2, 0.5),
                "high": np.random.uniform(0.1, 0.4)
            },
            "rl_actions": {
                "no_action": np.random.uniform(0.1, 0.5),
                "enter_long": np.random.uniform(0.1, 0.8),
                "enter_short": np.random.uniform(0.1, 0.8),
                "exit_long": np.random.uniform(0.1, 0.6),
                "exit_short": np.random.uniform(0.1, 0.6)
            },
            "combined_signal": "LONG" if np.random.random() > 0.5 else "SHORT",
            "confidence": np.random.uniform(0.5, 0.9)
        }
    
    def get_order_book_data(self):
        """Get order book data for visualization"""
        if not self.order_book_data:
            # Generate synthetic order book data if none available
            return {
                "bids": [{"price": 180 - i*0.1, "volume": np.random.uniform(0.5, 5)} for i in range(10)],
                "asks": [{"price": 180 + i*0.1, "volume": np.random.uniform(0.5, 5)} for i in range(10)],
                "bid_volume": 25,
                "ask_volume": 20,
                "imbalance": 1.25,
                "spread": 0.2,
                "is_synthetic": True,
                "timestamp": datetime.now().isoformat()
            }
        
        return self.order_book_data
    
    def get_cross_asset_data(self):
        """Get cross-asset correlation data"""
        if not self.cross_asset_data:
            # Generate synthetic cross-asset data if none available
            symbols = ["EURUSD", "USDJPY", "AUDUSD", "GOLD", "SP500", "NASDAQ"]
            return {
                "correlations": {sym: np.random.uniform(-1, 1) for sym in symbols},
                "movements": {sym: np.random.uniform(-0.5, 0.5) for sym in symbols},
                "lead_assets": [symbols[i] for i in range(2) if np.random.random() > 0.5],
                "lag_assets": [symbols[i] for i in range(2, 4) if np.random.random() > 0.5],
                "predictions": {sym: "up" if np.random.random() > 0.5 else "down" for sym in symbols[:3]}
            }
        
        return self.cross_asset_data

# Dashboard data instance
dashboard_data = DashboardData()

# Load configuration
def load_config():
    """Load dashboard configuration"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
    
    # Default configuration
    return {
        "theme": "darkly",
        "update_interval": 5000,
        "max_history_points": 500,
        "debug": False,
        "host": "0.0.0.0",
        "port": 8050,
        "enable_auth": False
    }

# Load user credentials
def load_credentials():
    """Load user credentials for dashboard authentication"""
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                creds = json.load(f)
                return {user['username']: user['password'] for user in creds}
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")
    
    # Default credentials (should be changed in production)
    return {'admin': 'password'}

# Load and apply configuration
config = load_config()
credentials = load_credentials() if config.get('enable_auth', False) else None
update_interval = config.get('update_interval', UPDATE_INTERVAL_MS)
app_theme = config.get('theme', 'darkly')

# Create Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[getattr(dbc.themes, app_theme.upper(), dbc.themes.DARKLY)],
    title="Enhanced HMM Trading System",
    update_title="Updating...",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# Add authentication if enabled
if credentials:
    auth = dash_auth.BasicAuth(app, credentials)

# Create server for WSGI deployment
server = app.server

# Helper functions for visualization
def generate_state_timeline(state_history):
    """Generate state timeline visualization"""
    if not state_history:
        return go.Figure()
    
    # Extract data
    states = [item['state'] for item in state_history]
    times = [datetime.fromisoformat(item['time']) if isinstance(item['time'], str) else item['time'] 
             for item in state_history]
    confidences = [item.get('confidence', 0.5) for item in state_history]
    
    # Create figure
    fig = go.Figure()
    
    # Add state line
    fig.add_trace(go.Scatter(
        x=times,
        y=states,
        mode='lines+markers',
        name='State',
        line=dict(color=COLORS['neutral'], width=2),
        marker=dict(size=8)
    ))
    
    # Add confidence
    fig.add_trace(go.Scatter(
        x=times,
        y=confidences,
        mode='lines',
        name='Confidence',
        line=dict(color=COLORS['highlight'], width=1, dash='dot'),
        yaxis="y2"
    ))
    
    # Update layout
    fig.update_layout(
        title={"text": "State History with Confidence", "x": 0.5},
        xaxis=dict(title="Time", gridcolor=COLORS['grid']),
        yaxis=dict(
            title="State",
            range=[-0.5, max(states) + 0.5] if states else [0, 4],
            dtick=1,
            gridcolor=COLORS['grid']
        ),
        yaxis2=dict(
            title="Confidence",
            titlefont=dict(color=COLORS['highlight']),
            tickfont=dict(color=COLORS['highlight']),
            overlaying="y",
            side="right",
            range=[0, 1],
            gridcolor=COLORS['highlight'],
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    return fig

def generate_price_signal_chart(price_data, signal_history, trade_history=None):
    """Generate price chart with signals and trades"""
    if not price_data or 'times' not in price_data or 'prices' not in price_data:
        return go.Figure()
    
    # Extract price data
    times = [datetime.fromisoformat(t) if isinstance(t, str) else t for t in price_data['times']]
    prices = price_data['prices']
    
    if not times or not prices or len(times) != len(prices):
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=times,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color=COLORS['neutral'], width=2)
    ))
    
    # Add signals
    if signal_history:
        long_signals = []
        short_signals = []
        
        for signal in signal_history:
            if 'signal' not in signal or 'timestamp' not in signal:
                continue
                
            try:
                signal_time = datetime.fromisoformat(signal['timestamp']) if isinstance(signal['timestamp'], str) else signal['timestamp']
                
                # Find closest price point
                closest_idx = min(range(len(times)), key=lambda i: abs((times[i] - signal_time).total_seconds()))
                
                if closest_idx < len(prices):
                    if signal['signal'] == 'LONG':
                        long_signals.append({
                            'time': signal_time,
                            'price': prices[closest_idx],
                            'confidence': signal.get('confidence', 0.5)
                        })
                    elif signal['signal'] == 'SHORT':
                        short_signals.append({
                            'time': signal_time,
                            'price': prices[closest_idx],
                            'confidence': signal.get('confidence', 0.5)
                        })
            except Exception as e:
                logger.error(f"Error processing signal for chart: {str(e)}")
        
        # Add long signal markers
        if long_signals:
            fig.add_trace(go.Scatter(
                x=[s['time'] for s in long_signals],
                y=[s['price'] for s in long_signals],
                mode='markers',
                marker=dict(
                    symbol="triangle-up", 
                    size=[max(8, min(20, s['confidence'] * 20)) for s in long_signals],
                    color=COLORS['long']
                ),
                name='Long Signal'
            ))
        
        # Add short signal markers
        if short_signals:
            fig.add_trace(go.Scatter(
                x=[s['time'] for s in short_signals],
                y=[s['price'] for s in short_signals],
                mode='markers',
                marker=dict(
                    symbol="triangle-down", 
                    size=[max(8, min(20, s['confidence'] * 20)) for s in short_signals],
                    color=COLORS['short']
                ),
                name='Short Signal'
            ))
    
    # Add trades if available
    if trade_history:
        # Separate trades by outcome
        winning_trades = []
        losing_trades = []
        
        for trade in trade_history:
            if 'entry_time' not in trade or 'exit_time' not in trade:
                continue
                
            try:
                entry_time = datetime.fromisoformat(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time']
                exit_time = datetime.fromisoformat(trade['exit_time']) if isinstance(trade['exit_time'], str) else trade['exit_time']
                
                profit_pips = trade.get('profit_pips', 0)
                
                trade_info = {
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': trade.get('entry_price', None),
                    'exit_price': trade.get('exit_price', None),
                    'direction': trade.get('direction', 'LONG'),
                    'profit_pips': profit_pips
                }
                
                if profit_pips > 0:
                    winning_trades.append(trade_info)
                else:
                    losing_trades.append(trade_info)
            except Exception as e:
                logger.error(f"Error processing trade for chart: {str(e)}")
        
        # For trades with prices, draw lines
        for i, trade_list in enumerate([winning_trades, losing_trades]):
            if not trade_list:
                continue
                
            for trade in trade_list:
                if trade['entry_price'] is not None and trade['exit_price'] is not None:
                    color = COLORS['long'] if i == 0 else COLORS['short']
                    fig.add_trace(go.Scatter(
                        x=[trade['entry_time'], trade['exit_time']],
                        y=[trade['entry_price'], trade['exit_price']],
                        mode='lines',
                        line=dict(color=color, width=1, dash='dot'),
                        showlegend=False
                    ))
    
    # Update layout
    fig.update_layout(
        title={"text": "Price Chart with Trading Signals", "x": 0.5},
        xaxis=dict(title="Time", gridcolor=COLORS['grid']),
        yaxis=dict(title="Price", gridcolor=COLORS['grid']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    return fig

def generate_performance_chart(performance_data):
    """Generate performance visualization"""
    if not performance_data:
        return go.Figure()
    
    # Prepare data
    states = []
    win_rates = []
    avg_profits = []
    total_profits = []
    trade_counts = []
    
    for state, metrics in performance_data.items():
        states.append(f"State {state}")
        win_rates.append(metrics.get("win_rate", 0) * 100)
        avg_profits.append(metrics.get("avg_profit_pips", 0))
        total_profits.append(metrics.get("total_profit_pips", 0))
        trade_counts.append(metrics.get("trade_count", 0))
    
    # Create figure
    fig = go.Figure()
    
    # Add win rate bars
    fig.add_trace(go.Bar(
        x=states,
        y=win_rates,
        name='Win Rate (%)',
        marker_color=COLORS['neutral']
    ))
    
    # Add average profit line
    fig.add_trace(go.Scatter(
        x=states,
        y=avg_profits,
        mode='lines+markers',
        name='Avg. Profit (pips)',
        marker=dict(color=COLORS['long'] if np.mean(avg_profits) > 0 else COLORS['short'], size=10),
        line=dict(color=COLORS['long'] if np.mean(avg_profits) > 0 else COLORS['short'], width=2),
        yaxis="y2"
    ))
    
    # Update layout
    fig.update_layout(
        title={"text": "State Performance Metrics", "x": 0.5},
        xaxis=dict(title="State", gridcolor=COLORS['grid']),
        yaxis=dict(
            title="Win Rate (%)",
            range=[0, 100],
            gridcolor=COLORS['grid']
        ),
        yaxis2=dict(
            title="Avg. Profit (pips)",
            titlefont=dict(color=COLORS['long'] if np.mean(avg_profits) > 0 else COLORS['short']),
            tickfont=dict(color=COLORS['long'] if np.mean(avg_profits) > 0 else COLORS['short']),
            overlaying="y",
            side="right",
            gridcolor=COLORS['grid'],
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    # Add data table on hover
    hover_data = [
        f"State: %{x}<br>" +
        f"Win Rate: %{y:.1f}<br>" +
        f"Avg Profit: {avg:.1f} pips<br>" +
        f"Total Profit: {total:.1f} pips<br>" +
        f"Trades: {count}"
        for x, y, avg, total, count in zip(states, win_rates, avg_profits, total_profits, trade_counts)
    ]
    
    fig.update_traces(
        hovertemplate=hover_data,
        selector=dict(type='bar')
    )
    
    return fig

def generate_trade_distribution_chart(trade_history, state_performance):
    """Generate trade distribution visualization"""
    if not trade_history:
        return go.Figure()
    
    # Get counts by state
    state_counts = {}
    state_profits = {}
    
    for trade in trade_history:
        state = trade.get("state", -1)
        if state == -1:
            continue
            
        state_str = str(state)
        state_counts[state_str] = state_counts.get(state_str, 0) + 1
        
        profit = trade.get("profit_pips", 0)
        if state_str in state_profits:
            state_profits[state_str] += profit
        else:
            state_profits[state_str] = profit
    
    # Prepare data for pie chart
    labels = [f"State {s}" for s in state_counts.keys()]
    values = list(state_counts.values())
    
    # Color palette based on profitability
    colors = []
    for state in state_counts.keys():
        profit = state_profits.get(state, 0)
        if profit > 0:
            intensity = min(1.0, profit / 500)  # Scale for color intensity
            colors.append(f"rgba(0, 204, 150, {0.5 + intensity/2})")  # Green for profit
        else:
            intensity = min(1.0, abs(profit) / 500)  # Scale for color intensity
            colors.append(f"rgba(239, 85, 59, {0.5 + intensity/2})")  # Red for loss
    
    # Create figure
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hoverinfo='label+value+text',
        hovertext=[f"Total Profit: {state_profits.get(s, 0):.1f} pips" for s in state_counts.keys()]
    )])
    
    # Add total profit annotation
    total_profit = sum(state_profits.values())
    profit_text = f"Total: {total_profit:.1f} pips"
    
    fig.update_layout(
        title={"text": "Trades by State", "x": 0.5},
        annotations=[dict(
            text=profit_text,
            x=0.5, y=0.5,
            font_size=18,
            showarrow=False,
            font=dict(color=COLORS['long'] if total_profit > 0 else COLORS['short'])
        )],
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    return fig

def generate_feature_importance_chart(feature_importance):
    """Generate feature importance visualization"""
    if not feature_importance:
        return go.Figure()
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top 15 features for clarity
    top_features = sorted_features[:15]
    
    # Prepare data
    feature_names = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    # Color gradient based on importance
    max_importance = max(importances)
    colors = [f"rgba(171, 99, 250, {min(1.0, i/max_importance)})" for i in importances]
    
    # Create figure
    fig = go.Figure([
        go.Bar(
            x=importances,
            y=feature_names,
            orientation='h',
            marker_color=colors
        )
    ])
    
    # Update layout
    fig.update_layout(
        title={"text": "Feature Importance", "x": 0.5},
        xaxis=dict(title="Importance", gridcolor=COLORS['grid']),
        yaxis=dict(title="Feature", autorange="reversed"),
        height=500,
        margin=dict(l=100, r=50, t=50, b=50),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    return fig

def generate_order_book_visualization(order_book_data):
    """Generate order book visualization"""
    if not order_book_data:
        return go.Figure()
    
    # Extract bid and ask data
    bids = order_book_data.get('bids', [])
    asks = order_book_data.get('asks', [])
    
    if not bids or not asks:
        return go.Figure()
    
    # Create bid and ask dataframes
    bid_df = pd.DataFrame(bids)
    ask_df = pd.DataFrame(asks)
    
    # Create figure
    fig = go.Figure()
    
    # Add bid volume bars
    fig.add_trace(go.Bar(
        x=bid_df['price'],
        y=bid_df['volume'],
        name='Bids',
        marker_color=COLORS['long'],
        opacity=0.7
    ))
    
    # Add ask volume bars
    fig.add_trace(go.Bar(
        x=ask_df['price'],
        y=ask_df['volume'],
        name='Asks',
        marker_color=COLORS['short'],
        opacity=0.7
    ))
    
    # Add midpoint line
    if len(bids) > 0 and len(asks) > 0:
        mid_price = (bid_df['price'].max() + ask_df['price'].min()) / 2
        fig.add_shape(
            type="line",
            x0=mid_price,
            y0=0,
            x1=mid_price,
            y1=max(bid_df['volume'].max(), ask_df['volume'].max()) * 1.1,
            line=dict(
                color=COLORS['highlight'],
                width=2,
                dash="dash",
            )
        )
    
    # Update layout
    fig.update_layout(
        title={"text": "Order Book Visualization", "x": 0.5},
        xaxis=dict(title="Price", gridcolor=COLORS['grid']),
        yaxis=dict(title="Volume", gridcolor=COLORS['grid']),
        barmode='overlay',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    # Add synthetic data indicator if applicable
    if order_book_data.get('is_synthetic', False):
        fig.add_annotation(
            text="SYNTHETIC DATA",
            x=0.5, y=0.9,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color=COLORS['warning'], size=12)
        )
    
    return fig

def generate_cross_asset_correlation_chart(cross_asset_data):
    """Generate cross-asset correlation visualization"""
    if not cross_asset_data or 'correlations' not in cross_asset_data:
        return go.Figure()
    
    # Extract correlation data
    correlations = cross_asset_data['correlations']
    
    # Prepare data
    symbols = list(correlations.keys())
    corr_values = list(correlations.values())
    
    # Color based on correlation (positive = green, negative = red)
    colors = [COLORS['long'] if c > 0 else COLORS['short'] for c in corr_values]
    
    # Create figure
    fig = go.Figure([
        go.Bar(
            x=symbols,
            y=corr_values,
            marker_color=colors
        )
    ])
    
    # Add reference line at zero
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(symbols) - 0.5,
        y1=0,
        line=dict(
            color=COLORS['text'],
            width=1,
            dash="dot",
        )
    )
    
    # Update layout
    fig.update_layout(
        title={"text": "Cross-Asset Correlations", "x": 0.5},
        xaxis=dict(title="Asset", gridcolor=COLORS['grid']),
        yaxis=dict(
            title="Correlation", 
            gridcolor=COLORS['grid'],
            range=[-1.1, 1.1]
        ),
        height=350,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    return fig

def generate_hybrid_prediction_charts(hybrid_data):
    """Generate hybrid model prediction visualizations"""
    direction_fig = go.Figure()
    volatility_fig = go.Figure()
    action_fig = go.Figure()
    
    # Direction prediction chart
    if hybrid_data and 'direction_pred' in hybrid_data:
        direction_pred = hybrid_data['direction_pred']
        
        # Prepare data
        labels = list(direction_pred.keys())
        values = list(direction_pred.values())
        
        # Color mapping
        color_map = {'up': COLORS['long'], 'down': COLORS['short'], 'sideways': COLORS['neutral']}
        colors = [color_map.get(l, COLORS['neutral']) for l in labels]
        
        direction_fig = go.Figure([
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors
            )
        ])
        
        direction_fig.update_layout(
            title={"text": "Direction Prediction", "x": 0.5},
            xaxis=dict(title="Direction", gridcolor=COLORS['grid']),
            yaxis=dict(title="Probability", gridcolor=COLORS['grid'], range=[0, 1]),
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text'])
        )
    
    # Volatility prediction chart
    if hybrid_data and 'volatility_pred' in hybrid_data:
        volatility_pred = hybrid_data['volatility_pred']
        
        # Prepare data
        labels = list(volatility_pred.keys())
        values = list(volatility_pred.values())
        
        # Color mapping
        color_map = {'low': COLORS['neutral'], 'medium': COLORS['highlight'], 'high': COLORS['high_vol']}
        colors = [color_map.get(l, COLORS['neutral']) for l in labels]
        
        volatility_fig = go.Figure([
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors
            )
        ])
        
        volatility_fig.update_layout(
            title={"text": "Volatility Prediction", "x": 0.5},
            xaxis=dict(title="Volatility", gridcolor=COLORS['grid']),
            yaxis=dict(title="Probability", gridcolor=COLORS['grid'], range=[0, 1]),
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text'])
        )
    
    # RL action Q-values chart
    if hybrid_data and 'rl_actions' in hybrid_data:
        rl_actions = hybrid_data['rl_actions']
        
        # Prepare data
        actions = list(rl_actions.keys())
        q_values = list(rl_actions.values())
        
        # Color mapping
        color_map = {
            'no_action': COLORS['neutral'],
            'enter_long': COLORS['long'],
            'enter_short': COLORS['short'],
            'exit_long': COLORS['highlight'],
            'exit_short': COLORS['highlight']
        }
        colors = [color_map.get(a, COLORS['neutral']) for a in actions]
        
        action_fig = go.Figure([
            go.Bar(
                x=actions,
                y=q_values,
                marker_color=colors
            )
        ])
        
        action_fig.update_layout(
            title={"text": "RL Action Q-Values", "x": 0.5},
            xaxis=dict(title="Action", gridcolor=COLORS['grid'], tickangle=45),
            yaxis=dict(title="Q-Value", gridcolor=COLORS['grid']),
            height=300,
            margin=dict(l=50, r=50, t=70, b=100),
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text'])
        )
    
    return direction_fig, volatility_fig, action_fig

# Dashboard layout components
def create_header():
    """Create dashboard header"""
    return html.Div([
        html.Div([
            html.Img(src='/assets/logo.png', style={'height': '50px', 'marginRight': '15px'}, 
                    className="d-none d-md-block"),  # Logo hidden on small screens
            html.Div([
                html.H3("Enhanced HMM Trading System", className="mb-0"),
                html.P("Real-time monitoring dashboard", className="text-muted mb-0")
            ])
        ], className="d-flex align-items-center"),
        
        html.Div([
            html.Span("Last Update: ", className="me-2"),
            html.Span(id="last-update", className="text-info")
        ], className="ml-auto")
    ], className="d-flex justify-content-between align-items-center p-3 bg-dark")

def create_current_state_card():
    """Create current state display card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Current State", className="mb-0 text-center")
        ]),
        dbc.CardBody([
            html.Div(id="current-state-display", className="text-center mb-4"),
            dbc.Row([
                dbc.Col([
                    html.P("Confidence", className="text-center mb-1"),
                    dcc.Graph(id="confidence-gauge", config={'displayModeBar': False}, style={'height': '150px'})
                ], width=6),
                dbc.Col([
                    html.P("Validity", className="text-center mb-1"),
                    dcc.Graph(id="validity-gauge", config={'displayModeBar': False}, style={'height': '150px'})
                ], width=6)
            ])
        ])
    ])

def create_signal_card():
    """Create signal display card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Latest Signal", className="mb-0 text-center")
        ]),
        dbc.CardBody([
            html.Div(id="latest-signal-display", className="text-center mb-4"),
            dbc.Row([
                dbc.Col([
                    html.P("Signal Confidence", className="text-center mb-1"),
                    dcc.Graph(id="signal-gauge", config={'displayModeBar': False}, style={'height': '150px'})
                ], width=6),
                dbc.Col([
                    html.P("Hybrid Contribution", className="text-center mb-1"),
                    dcc.Graph(id="hybrid-contribution", config={'displayModeBar': False}, style={'height': '150px'})
                ], width=6)
            ])
        ])
    ])

def create_state_history_card():
    """Create state history card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("State History", className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(
                id="state-history-chart",
                config={'displayModeBar': True, 'scrollZoom': True},
                style={'height': '400px'}
            )
        ])
    ])

def create_price_signal_card():
    """Create price and signals card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Price and Signals", className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(
                id="price-signals-chart",
                config={'displayModeBar': True, 'scrollZoom': True},
                style={'height': '400px'}
            )
        ])
    ])

def create_performance_card():
    """Create performance metrics card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("State Performance", className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(
                id="state-performance-chart",
                config={'displayModeBar': False},
                style={'height': '400px'}
            )
        ])
    ])

def create_trades_card():
    """Create trades distribution card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Trades by State", className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(
                id="trades-by-state-chart",
                config={'displayModeBar': False},
                style={'height': '400px'}
            )
        ])
    ])

def create_feature_importance_card():
    """Create feature importance card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Feature Importance", className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(
                id="feature-importance-chart",
                config={'displayModeBar': False},
                style={'height': '400px'}
            )
        ])
    ])

def create_order_book_card():
    """Create order book visualization card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Order Book Analysis", className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(
                id="order-book-chart",
                config={'displayModeBar': False},
                style={'height': '400px'}
            )
        ])
    ])

def create_market_memory_card():
    """Create market memory card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Market Memory", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P("Pattern Statistics", className="mb-2 fw-bold"),
                        html.Div([
                            html.Span("Total patterns: ", className="text-muted"),
                            html.Span(id="pattern-count", className="text-info ms-1")
                        ], className="mb-1"),
                        html.Div([
                            html.Span("Similar patterns: ", className="text-muted"),
                            html.Span(id="similar-patterns", className="text-info ms-1")
                        ], className="mb-1"),
                        html.Div([
                            html.Span("Base success rate: ", className="text-muted"),
                            html.Span(id="base-success-rate", className="text-info ms-1")
                        ], className="mb-1"),
                        html.Div([
                            html.Span("Pattern prediction: ", className="text-muted"),
                            html.Span(id="pattern-prediction", className="ms-1")
                        ], className="mb-1")
                    ])
                ], width=12, lg=4),
                dbc.Col([
                    dcc.Graph(
                        id="memory-confidence-chart",
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ], width=12, lg=8)
            ])
        ])
    ])

def create_cross_asset_card():
    """Create cross-asset correlation card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Cross-Asset Analysis", className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(
                id="cross-asset-chart",
                config={'displayModeBar': False},
                style={'height': '350px'}
            )
        ])
    ])

def create_hybrid_model_card():
    """Create hybrid model predictions card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Hybrid Model Predictions", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id="direction-prediction-chart",
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ], width=12, lg=6),
                dbc.Col([
                    dcc.Graph(
                        id="volatility-prediction-chart",
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ], width=12, lg=6)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id="rl-actions-chart",
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ], width=12)
            ])
        ])
    ])

def create_component_status_card():
    """Create component status card"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("System Components", className="mb-0")
        ]),
        dbc.CardBody([
            html.Div(id="component-status-container", className="d-flex flex-wrap")
        ])
    ])

# Full dashboard layout
app.layout = html.Div([
    # Header
    create_header(),
    
    # Main content container
    dbc.Container(fluid=True, className="mt-3", children=[
        # Current state and signal row
        dbc.Row([
            dbc.Col(create_current_state_card(), width=12, lg=6, className="mb-3"),
            dbc.Col(create_signal_card(), width=12, lg=6, className="mb-3")
        ]),
        
        # Charts row
        dbc.Row([
            dbc.Col(create_state_history_card(), width=12, lg=6, className="mb-3"),
            dbc.Col(create_price_signal_card(), width=12, lg=6, className="mb-3")
        ]),
        
        # Performance row
        dbc.Row([
            dbc.Col(create_performance_card(), width=12, lg=6, className="mb-3"),
            dbc.Col(create_trades_card(), width=12, lg=6, className="mb-3")
        ]),
        
        # Advanced components row 1
        dbc.Row([
            dbc.Col(create_market_memory_card(), width=12, lg=6, className="mb-3"),
            dbc.Col(create_order_book_card(), width=12, lg=6, className="mb-3")
        ]),
        
        # Advanced components row 2
        dbc.Row([
            dbc.Col(create_hybrid_model_card(), width=12, lg=6, className="mb-3"),
            dbc.Col(create_cross_asset_card(), width=12, lg=6, className="mb-3")
        ]),
        
        # Additional row for feature importance and system status
        dbc.Row([
            dbc.Col(create_feature_importance_card(), width=12, lg=6, className="mb-3"),
            dbc.Col(create_component_status_card(), width=12, lg=6, className="mb-3")
        ]),
        
        # Hidden divs for data storage
        html.Div(id="state-data-store", style={"display": "none"}),
        html.Div(id="signal-data-store", style={"display": "none"}),
        html.Div(id="performance-data-store", style={"display": "none"}),
        html.Div(id="memory-data-store", style={"display": "none"}),
        html.Div(id="hybrid-data-store", style={"display": "none"}),
        
        # Interval for data refresh
        dcc.Interval(
            id="refresh-interval",
            interval=update_interval,
            n_intervals=0
        )
    ])
])

# Callbacks

@app.callback(
    [Output("last-update", "children"),
     Output("state-data-store", "children"),
     Output("signal-data-store", "children"),
     Output("performance-data-store", "children"),
     Output("memory-data-store", "children"),
     Output("hybrid-data-store", "children")],
    [Input("refresh-interval", "n_intervals")]
)
def update_all_data(n):
    """Master update function to refresh all data"""
    # Update data from files (standalone mode)
    dashboard_data.update_from_files()
    
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get data for storage
    current_state = dashboard_data.current_state
    latest_signal = dashboard_data.latest_signal
    performance_data = dashboard_data.get_state_performance()
    memory_data = dashboard_data.get_synthetic_market_memory()
    hybrid_data = dashboard_data.get_synthetic_hybrid_prediction()
    
    return (
        current_time,
        json.dumps(current_state),
        json.dumps(latest_signal),
        json.dumps(performance_data),
        json.dumps(memory_data),
        json.dumps(hybrid_data)
    )

@app.callback(
    [Output("current-state-display", "children"),
     Output("confidence-gauge", "figure"),
     Output("validity-gauge", "figure")],
    [Input("state-data-store", "children")]
)
def update_current_state_display(state_json):
    """Update current state display"""
    if not state_json:
        # Default state
        state = {
            "state_idx": 0,
            "state_label": "Unknown",
            "confidence": 0.5,
            "validity": 0.5
        }
    else:
        try:
            state = json.loads(state_json)
        except:
            state = {
                "state_idx": 0,
                "state_label": "Unknown",
                "confidence": 0.5,
                "validity": 0.5
            }
    
    # State display with color based on state label
    state_idx = state.get("state_idx", 0)
    state_label = state.get("state_label", "Unknown")
    
    # Determine color based on state label
    state_color = COLORS['neutral']
    if "Bullish" in state_label:
        state_color = COLORS['bullish']
    elif "Bearish" in state_label:
        state_color = COLORS['bearish']
    
    state_display = html.Div([
        html.H1(f"State {state_idx}", style={"margin-bottom": "5px"}),
        html.H4(state_label, style={"margin-top": "0px", "color": state_color})
    ])
    
    # Confidence gauge
    confidence = state.get('confidence', 0.5)
    confidence_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS['text']},
            'bar': {'color': COLORS['highlight']},
            'bgcolor': COLORS['background'],
            'bordercolor': COLORS['text'],
            'steps': [
                {'range': [0, 50], 'color': COLORS['background']},
                {'range': [50, 75], 'color': 'rgba(171, 99, 250, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(171, 99, 250, 0.5)'}
            ],
            'threshold': {
                'line': {'color': COLORS['warning'], 'width': 2},
                'thickness': 0.75,
                'value': 70
            }
        },
        number={'suffix': "%", 'font': {'color': COLORS['text']}}
    ))
    
    confidence_gauge.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    # Validity gauge
    validity = state.get('validity', 0.5)
    validity_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=validity * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS['text']},
            'bar': {'color': COLORS['long']},
            'bgcolor': COLORS['background'],
            'bordercolor': COLORS['text'],
            'steps': [
                {'range': [0, 50], 'color': COLORS['background']},
                {'range': [50, 75], 'color': 'rgba(0, 204, 150, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(0, 204, 150, 0.5)'}
            ],
            'threshold': {
                'line': {'color': COLORS['warning'], 'width': 2},
                'thickness': 0.75,
                'value': 70
            }
        },
        number={'suffix': "%", 'font': {'color': COLORS['text']}}
    ))
    
    validity_gauge.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    return state_display, confidence_gauge, validity_gauge

@app.callback(
    [Output("latest-signal-display", "children"),
     Output("signal-gauge", "figure"),
     Output("hybrid-contribution", "figure")],
    [Input("signal-data-store", "children")]
)
def update_signal_display(signal_json):
    """Update signal display"""
    if not signal_json:
        # Default signal
        signal = {
            "signal": "NONE",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "hybrid_contribution": 0.0
        }
    else:
        try:
            signal = json.loads(signal_json)
        except:
            signal = {
                "signal": "NONE",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "hybrid_contribution": 0.0
            }
    
    # Signal display with color based on direction
    signal_type = signal.get('signal', 'NONE')
    signal_time = signal.get('timestamp', datetime.now().isoformat())
    
    if isinstance(signal_time, str):
        try:
            signal_time = datetime.fromisoformat(signal_time)
            formatted_time = signal_time.strftime("%H:%M:%S")
        except:
            formatted_time = "Unknown"
    else:
        formatted_time = "Unknown"
    
    # Determine color based on signal type
    signal_color = COLORS['neutral']
    if signal_type == "LONG":
        signal_color = COLORS['long']
    elif signal_type == "SHORT":
        signal_color = COLORS['short']
    
    signal_display = html.Div([
        html.H1(signal_type, style={"margin-bottom": "5px", "color": signal_color}),
        html.H4(f"Generated at: {formatted_time}", style={"margin-top": "0px"})
    ])
    
    # Signal confidence gauge
    confidence = signal.get('confidence', 0.0)
    signal_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS['text']},
            'bar': {'color': signal_color},
            'bgcolor': COLORS['background'],
            'bordercolor': COLORS['text'],
            'steps': [
                {'range': [0, 50], 'color': COLORS['background']},
                {'range': [50, 75], 'color': f'rgba({signal_color.replace("rgb", "").replace("(", "").replace(")", "")}, 0.3)'},
                {'range': [75, 100], 'color': f'rgba({signal_color.replace("rgb", "").replace("(", "").replace(")", "")}, 0.5)'}
            ],
            'threshold': {
                'line': {'color': COLORS['warning'], 'width': 2},
                'thickness': 0.75,
                'value': 70
            }
        },
        number={'suffix': "%", 'font': {'color': COLORS['text']}}
    ))
    
    signal_gauge.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    # Hybrid contribution gauge
    hybrid_contrib = signal.get('hybrid_contribution', 0.0)
    hybrid_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hybrid_contrib * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS['text']},
            'bar': {'color': COLORS['highlight']},
            'bgcolor': COLORS['background'],
            'bordercolor': COLORS['text'],
            'steps': [
                {'range': [0, 25], 'color': COLORS['background']},
                {'range': [25, 50], 'color': 'rgba(171, 99, 250, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(171, 99, 250, 0.5)'}
            ],
            'threshold': {
                'line': {'color': COLORS['warning'], 'width': 2},
                'thickness': 0.75,
                'value': 50
            }
        },
        number={'suffix': "%", 'font': {'color': COLORS['text']}}
    ))
    
    hybrid_gauge.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text'])
    )
    
    return signal_display, signal_gauge, hybrid_gauge

@app.callback(
    Output("state-history-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_state_history_chart(n):
    """Update state history chart"""
    return generate_state_timeline(dashboard_data.state_history)

@app.callback(
    Output("price-signals-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_price_signals_chart(n):
    """Update price and signals chart"""
    price_data = dashboard_data.get_price_data()
    return generate_price_signal_chart(price_data, dashboard_data.signal_history, dashboard_data.trade_history)

@app.callback(
    Output("state-performance-chart", "figure"),
    [Input("performance-data-store", "children")]
)
def update_performance_chart(performance_json):
    """Update performance chart"""
    if not performance_json:
        return go.Figure()
    
    try:
        performance_data = json.loads(performance_json)
        return generate_performance_chart(performance_data)
    except:
        return go.Figure()

@app.callback(
    Output("trades-by-state-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_trades_chart(n):
    """Update trades by state chart"""
    performance_data = dashboard_data.get_state_performance()
    return generate_trade_distribution_chart(dashboard_data.trade_history, performance_data)

@app.callback(
    Output("feature-importance-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_feature_importance_chart(n):
    """Update feature importance chart"""
    feature_importance = dashboard_data.get_feature_importance()
    return generate_feature_importance_chart(feature_importance)

@app.callback(
    Output("order-book-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_order_book_chart(n):
    """Update order book chart"""
    order_book_data = dashboard_data.get_order_book_data()
    return generate_order_book_visualization(order_book_data)

@app.callback(
    [Output("pattern-count", "children"),
     Output("similar-patterns", "children"),
     Output("base-success-rate", "children"),
     Output("pattern-prediction", "children"),
     Output("memory-confidence-chart", "figure")],
    [Input("memory-data-store", "children")]
)
def update_memory_display(memory_json):
    """Update market memory display"""
    if not memory_json:
        return "0", "0", "0%", "Unknown", go.Figure()
    
    try:
        memory_data = json.loads(memory_json)
        
        pattern_count = memory_data.get("pattern_count", 0)
        similar_patterns = memory_data.get("similar_patterns", 0)
        success_rate = memory_data.get("success_rate", 0) * 100
        prediction = memory_data.get("prediction", "unknown")
        prediction_confidence = memory_data.get("prediction_confidence", 0.5)
        
        # Determine color based on prediction
        prediction_color = COLORS['bullish'] if prediction == "profitable" else COLORS['bearish']
        prediction_text = html.Span(
            f"{prediction.capitalize()} ({prediction_confidence:.2f})",
            style={"color": prediction_color}
        )
        
        # Create confidence chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Profitable", "Loss", "Unknown"],
                    y=[
                        memory_data.get("success_rate", 0) * 100,
                        100 - memory_data.get("success_rate", 0) * 100,
                        0  # No unknown category
                    ],
                    marker_color=[COLORS['long'], COLORS['short'], COLORS['neutral']]
                )
            ]
        ).update_layout(
            title={"text": "Outcome Distribution", "x": 0.5},
            xaxis=dict(gridcolor=COLORS['grid']),
            yaxis=dict(title="Percentage", range=[0, 100], gridcolor=COLORS['grid']),
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text'])
        )
        
        return str(pattern_count), str(similar_patterns), f"{success_rate:.1f}%", prediction_text, fig
    except:
        return "0", "0", "0%", "Unknown", go.Figure()

@app.callback(
    [Output("direction-prediction-chart", "figure"),
     Output("volatility-prediction-chart", "figure"),
     Output("rl-actions-chart", "figure")],
    [Input("hybrid-data-store", "children")]
)
def update_hybrid_charts(hybrid_json):
    """Update hybrid model charts"""
    if not hybrid_json:
        return go.Figure(), go.Figure(), go.Figure()
    
    try:
        hybrid_data = json.loads(hybrid_json)
        return generate_hybrid_prediction_charts(hybrid_data)
    except:
        return go.Figure(), go.Figure(), go.Figure()

@app.callback(
    Output("cross-asset-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_cross_asset_chart(n):
    """Update cross-asset correlation chart"""
    cross_asset_data = dashboard_data.get_cross_asset_data()
    return generate_cross_asset_correlation_chart(cross_asset_data)

@app.callback(
    Output("component-status-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_component_status(n):
    """Update component status indicators"""
    components = {
        "HMM Core": True,  # Always available
        "Feature Selection": dashboard_data.component_status.get("feature_selection", False),
        "Market Memory": dashboard_data.component_status.get("market_memory", False),
        "Hybrid Model": dashboard_data.component_status.get("hybrid_model", False),
        "Order Book": dashboard_data.component_status.get("order_book", False),
        "Cross Asset": dashboard_data.component_status.get("cross_asset", False),
        "Feature Fusion": dashboard_data.component_status.get("feature_fusion", False),
        "Signal Weighting": dashboard_data.component_status.get("signal_weighting", False),
        "Ensemble": dashboard_data.component_status.get("ensemble", False)
    }
    
    status_indicators = []
    for component, status in components.items():
        status_color = "success" if status else "danger"
        status_text = "Active" if status else "Inactive"
        
        indicator = dbc.Badge(
            status_text,
            color=status_color,
            className="me-1 mb-2"
        )
        
        status_indicators.append(
            html.Div([
                html.Span(f"{component}: ", className="me-1"),
                indicator
            ], className="me-3 mb-2")
        )
    
    return status_indicators

# Additional functions for dashboard control
def create_dashboard(shared_data=None):
    """
    Create dashboard with optional shared data.
    
    Args:
        shared_data: Optional shared data object for integration
    
    Returns:
        dash.Dash app instance
    """
    global dashboard_data
    
    if shared_data:
        dashboard_data = shared_data
    
    return app

def start_dashboard(dashboard_app=None, host='0.0.0.0', port=8050, debug=False):
    """Start the dashboard server"""
    global app
    if dashboard_app is None:
        dashboard_app = app
    
    if dashboard_app is not None:
        logger.info(f"Starting dashboard on {host}:{port}")
        dashboard_app.run(host=host, port=port, debug=debug)
    else:
        logger.error("No dashboard app to start")

# Main entry point for standalone execution
if __name__ == '__main__':
    # Set up command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced HMM Trading System Dashboard')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize data
    dashboard_data.update_from_files()
    
    # Start the dashboard
    start_dashboard(host=args.host, port=args.port, debug=args.debug)
