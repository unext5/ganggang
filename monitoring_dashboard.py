import dash
from dash import dcc, html
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import pickle
from collections import deque

# Import our component modules
from market_memory import MarketMemory
from hybrid_model import HybridModel

# Constants
UPDATE_INTERVAL = 60000  # Update every 60 seconds
MAX_POINTS = 500  # Maximum points to display on real-time charts
MODEL_PATH = "enhanced_model/enhanced_hmm_model.pkl"
PERFORMANCE_FILE = "hmm_performance_tracker.json"
STATE_HISTORY_FILE = "state_history.json"
TRADE_HISTORY_FILE = "trade_history.json"

# Initialize dashboard
app = dash.Dash(__name__, 
               title="HMM Trading Dashboard",
               meta_tags=[{"name": "viewport", 
                          "content": "width=device-width, initial-scale=1"}])

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("GBPJPY HMM Trading System Dashboard", 
                style={"color": "white", "margin-bottom": "0px"}),
        html.P("Real-time monitoring of the enhanced HMM trading system", 
              style={"color": "white", "margin-top": "0px"}),
        html.Div([
            html.Span("Last Update: ", style={"color": "white"}),
            html.Span(id="last-update", style={"color": "white"})
        ], style={"float": "right", "margin-top": "-40px"})
    ], style={"background-color": "#2c3e50", "padding": "20px", "margin-bottom": "20px"}),
    
    # Current State Information
    html.Div([
        html.Div([
            html.H3("Current State", style={"text-align": "center"}),
            html.Div(id="current-state-display", className="state-box"),
            html.Div([
                html.Div([
                    html.H5("Confidence"),
                    html.Div(id="confidence-gauge", style={"margin": "auto"})
                ], className="six columns"),
                html.Div([
                    html.H5("Validity"),
                    html.Div(id="validity-gauge", style={"margin": "auto"})
                ], className="six columns"),
            ], className="row", style={"margin-top": "20px"})
        ], className="six columns", style={"background-color": "#ecf0f1", "padding": "15px", "border-radius": "5px"}),
        
        html.Div([
            html.H3("Latest Signal", style={"text-align": "center"}),
            html.Div(id="latest-signal-display", className="signal-box"),
            html.Div([
                html.Div([
                    html.H5("Signal Strength"),
                    html.Div(id="signal-gauge", style={"margin": "auto"})
                ], className="six columns"),
                html.Div([
                    html.H5("Hybrid Contribution"),
                    html.Div(id="hybrid-contribution", style={"margin": "auto"})
                ], className="six columns"),
            ], className="row", style={"margin-top": "20px"})
        ], className="six columns", style={"background-color": "#ecf0f1", "padding": "15px", "border-radius": "5px"})
    ], className="row", style={"margin-bottom": "20px"}),
    
    # Charts Row
    html.Div([
        html.Div([
            html.H3("State History", style={"text-align": "center"}),
            dcc.Graph(id="state-history-chart")
        ], className="six columns", style={"background-color": "#ecf0f1", "padding": "15px", "border-radius": "5px"}),
        
        html.Div([
            html.H3("Price and Signals", style={"text-align": "center"}),
            dcc.Graph(id="price-signals-chart")
        ], className="six columns", style={"background-color": "#ecf0f1", "padding": "15px", "border-radius": "5px"})
    ], className="row", style={"margin-bottom": "20px"}),
    
    # Performance & Statistics
    html.Div([
        html.Div([
            html.H3("State Performance", style={"text-align": "center"}),
            dcc.Graph(id="state-performance-chart")
        ], className="six columns", style={"background-color": "#ecf0f1", "padding": "15px", "border-radius": "5px"}),
        
        html.Div([
            html.H3("Trades by State", style={"text-align": "center"}),
            dcc.Graph(id="trades-by-state-chart")
        ], className="six columns", style={"background-color": "#ecf0f1", "padding": "15px", "border-radius": "5px"})
    ], className="row", style={"margin-bottom": "20px"}),
    
    # Advanced Components
    html.Div([
        html.Div([
            html.H3("Market Memory", style={"text-align": "center"}),
            html.Div([
                html.P("Similar patterns found: ", style={"display": "inline-block"}),
                html.Span(id="similar-patterns", style={"font-weight": "bold"})
            ]),
            html.Div([
                html.P("Base success rate: ", style={"display": "inline-block"}),
                html.Span(id="base-success-rate", style={"font-weight": "bold"})
            ]),
            html.Div([
                html.P("Pattern prediction: ", style={"display": "inline-block"}),
                html.Span(id="pattern-prediction", style={"font-weight": "bold"})
            ]),
            dcc.Graph(id="memory-confidence-chart")
        ], className="six columns", style={"background-color": "#ecf0f1", "padding": "15px", "border-radius": "5px"}),
        
        html.Div([
            html.H3("Hybrid Model Predictions", style={"text-align": "center"}),
            html.Div([
                html.Div([
                    html.H5("Direction Prediction"),
                    dcc.Graph(id="direction-prediction-chart")
                ], className="six columns"),
                html.Div([
                    html.H5("Volatility Prediction"),
                    dcc.Graph(id="volatility-prediction-chart")
                ], className="six columns")
            ], className="row"),
            html.H5("RL Action Q-Values"),
            dcc.Graph(id="rl-qvalues-chart")
        ], className="six columns", style={"background-color": "#ecf0f1", "padding": "15px", "border-radius": "5px"})
    ], className="row", style={"margin-bottom": "20px"}),
    
    # Hidden divs for storing data
    html.Div(id="state-data", style={"display": "none"}),
    html.Div(id="signal-data", style={"display": "none"}),
    html.Div(id="performance-data", style={"display": "none"}),
    html.Div(id="memory-data", style={"display": "none"}),
    html.Div(id="hybrid-data", style={"display": "none"}),
    
    # Interval for data refresh
    dcc.Interval(
        id="interval-component",
        interval=UPDATE_INTERVAL,
        n_intervals=0
    )
], className="container", style={"max-width": "95%", "margin": "auto"})

# Helper functions
def load_json_file(filename, default=None):
    """Load data from JSON file with fallback to default"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
    return default

def load_state_history():
    """Load state history data"""
    data = load_json_file(STATE_HISTORY_FILE, {})
    
    if not data:
        # Generate dummy data if file not found
        return {
            "states": list(range(4)) * 25,
            "confidences": [np.random.uniform(0.5, 1.0) for _ in range(100)],
            "timestamps": [(datetime.now() - timedelta(minutes=i)).isoformat() 
                          for i in range(100, 0, -1)],
            "prices": [np.random.uniform(180, 190) for _ in range(100)]
        }
    
    return data

def load_performance_data():
    """Load performance tracking data"""
    data = load_json_file(PERFORMANCE_FILE, {})
    
    if not data or "state_metrics" not in data:
        # Generate dummy data if file not found
        return {
            "state_metrics": {
                str(i): {
                    "win_rate": np.random.uniform(0.3, 0.7),
                    "avg_profit_pips": np.random.uniform(-5, 15),
                    "total_profit_pips": np.random.uniform(-100, 300),
                    "trade_count": np.random.randint(10, 100)
                } for i in range(4)
            }
        }
    
    return data

def load_trade_history():
    """Load trade history data"""
    data = load_json_file(TRADE_HISTORY_FILE, {})
    
    if not data:
        # Generate dummy data if file not found
        return {
            "active_trades": [],
            "trade_history": [
                {
                    "state": i % 4,
                    "direction": "LONG" if i % 2 == 0 else "SHORT",
                    "profit_pips": np.random.uniform(-20, 30),
                    "entry_time": (datetime.now() - timedelta(hours=i)).isoformat(),
                    "exit_time": (datetime.now() - timedelta(hours=i-1)).isoformat()
                } for i in range(50)
            ]
        }
    
    return data

def load_memory_data():
    """Load or initialize market memory data"""
    try:
        memory = MarketMemory()
        if memory.load_memory():
            stats = memory.get_pattern_statistics()
            return {
                "statistics": stats,
                "loaded": True
            }
    except Exception as e:
        print(f"Error loading market memory: {str(e)}")
    
    # Return dummy data if loading fails
    return {
        "statistics": {
            "count": 0,
            "outcome_distribution": {
                "profitable": 0.6,
                "loss": 0.3,
                "unknown": 0.1
            }
        },
        "loaded": False
    }

def load_hybrid_data():
    """Load or initialize hybrid model data"""
    try:
        model_dir = "hybrid_models"
        hybrid_model = HybridModel(input_dim=19, hmm_states=4)
        if hybrid_model.load_models(model_dir):
            # Return basic model info
            return {
                "loaded": True,
                "direction_accuracy": 0.68,  # These would come from actual model stats
                "volatility_accuracy": 0.72,
                "rl_contribution": 0.55
            }
    except Exception as e:
        print(f"Error loading hybrid model: {str(e)}")
    
    # Return dummy data if loading fails
    return {
        "loaded": False,
        "direction_accuracy": 0.5,
        "volatility_accuracy": 0.5,
        "rl_contribution": 0.25
    }

def get_current_state():
    """Get current state information"""
    # This would normally come from the running HMM system
    # Here we'll use the most recent state from history
    state_history = load_state_history()
    
    if not state_history or not state_history.get("states"):
        return {
            "state_idx": 0,
            "state_label": "Unknown",
            "confidence": 0.5,
            "validity": 0.5
        }
    
    states = state_history["states"]
    confidences = state_history["confidences"]
    
    # Get the latest state
    latest_state = int(states[-1])
    latest_confidence = float(confidences[-1]) if confidences else 0.5
    
    # Map state index to a meaningful label
    state_labels = ["High Bullish", "Low Bullish", "Low Bearish", "High Bearish"]
    state_label = state_labels[latest_state % len(state_labels)]
    
    # In a real system, validity would come from the HMM validation function
    validity = np.random.uniform(0.5, 1.0)
    
    return {
        "state_idx": latest_state,
        "state_label": state_label,
        "confidence": latest_confidence,
        "validity": validity
    }

def get_latest_signal():
    """Get latest trading signal information"""
    # This would normally come from the running HMM system
    # Here we'll generate a signal based on recent states
    state_history = load_state_history()
    
    if not state_history or not state_history.get("states"):
        return {
            "signal": "NONE",
            "strength": 0.0,
            "time": datetime.now().isoformat(),
            "hybrid_contribution": 0.0
        }
    
    states = state_history["states"]
    
    # Simple logic: if last two states are bullish, generate LONG signal
    if len(states) >= 2:
        latest_states = [int(s) for s in states[-2:]]
        
        if all(s in [0, 1] for s in latest_states):
            signal = "LONG"
            strength = np.random.uniform(0.7, 0.95)
        elif all(s in [2, 3] for s in latest_states):
            signal = "SHORT"
            strength = np.random.uniform(0.7, 0.95)
        else:
            signal = "NONE"
            strength = 0.0
    else:
        signal = "NONE"
        strength = 0.0
    
    # Randomly assign hybrid contribution
    hybrid_contribution = np.random.uniform(0.2, 0.8) if signal != "NONE" else 0.0
    
    return {
        "signal": signal,
        "strength": strength,
        "time": datetime.now().isoformat(),
        "hybrid_contribution": hybrid_contribution
    }

# Callbacks
@app.callback(
    [Output("last-update", "children"),
     Output("state-data", "children"),
     Output("signal-data", "children"),
     Output("performance-data", "children"),
     Output("memory-data", "children"),
     Output("hybrid-data", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_data(n):
    """Update all data sources"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get current state
    current_state = get_current_state()
    
    # Get latest signal
    latest_signal = get_latest_signal()
    
    # Load performance data
    performance_data = load_performance_data()
    
    # Load memory data
    memory_data = load_memory_data()
    
    # Load hybrid model data
    hybrid_data = load_hybrid_data()
    
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
     Output("confidence-gauge", "children"),
     Output("validity-gauge", "children")],
    [Input("state-data", "children")]
)
def update_current_state(state_json):
    """Update current state display"""
    if not state_json:
        return "No data", "", ""
    
    state = json.loads(state_json)
    
    # Create state display
    state_display = html.Div([
        html.H2(f"State {state['state_idx']}", style={"margin-bottom": "5px"}),
        html.H4(state['state_label'], style={"margin-top": "0px", "color": "#3498db"})
    ], style={"text-align": "center"})
    
    # Create confidence gauge
    confidence = state.get('confidence', 0.5)
    confidence_gauge = dcc.Graph(
        figure=go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        )),
        config={'displayModeBar': False},
        style={'height': 200}
    )
    
    # Create validity gauge
    validity = state.get('validity', 0.5)
    validity_gauge = dcc.Graph(
        figure=go.Figure(go.Indicator(
            mode="gauge+number",
            value=validity * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Validity"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        )),
        config={'displayModeBar': False},
        style={'height': 200}
    )
    
    return state_display, confidence_gauge, validity_gauge

@app.callback(
    [Output("latest-signal-display", "children"),
     Output("signal-gauge", "children"),
     Output("hybrid-contribution", "children")],
    [Input("signal-data", "children")]
)
def update_signal_display(signal_json):
    """Update signal display"""
    if not signal_json:
        return "No data", "", ""
    
    signal = json.loads(signal_json)
    
    # Set signal color based on direction
    signal_color = "#27ae60" if signal['signal'] == "LONG" else (
                  "#e74c3c" if signal['signal'] == "SHORT" else "#7f8c8d")
    
    # Create signal display
    signal_display = html.Div([
        html.H2(signal['signal'], style={"margin-bottom": "5px", "color": signal_color}),
        html.H4(f"Generated at: {signal['time'][-8:]}", style={"margin-top": "0px"})
    ], style={"text-align": "center"})
    
    # Create signal strength gauge
    strength = signal.get('strength', 0)
    signal_gauge = dcc.Graph(
        figure=go.Figure(go.Indicator(
            mode="gauge+number",
            value=strength * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Strength"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': signal_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        )),
        config={'displayModeBar': False},
        style={'height': 200}
    )
    
    # Create hybrid contribution gauge
    hybrid_contrib = signal.get('hybrid_contribution', 0)
    hybrid_gauge = dcc.Graph(
        figure=go.Figure(go.Indicator(
            mode="gauge+number",
            value=hybrid_contrib * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Hybrid"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 100], 'color': "#d2b4de"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        )),
        config={'displayModeBar': False},
        style={'height': 200}
    )
    
    return signal_display, signal_gauge, hybrid_gauge

@app.callback(
    Output("state-history-chart", "figure"),
    [Input("state-data", "children")]
)
def update_state_history_chart(state_json):
    """Update state history chart"""
    # Load state history
    state_history = load_state_history()
    
    if not state_history or not state_history.get("states"):
        return go.Figure()
    
    # Prepare data
    states = [int(s) for s in state_history["states"]]
    confidences = [float(c) for c in state_history["confidences"]]
    timestamps = [datetime.fromisoformat(ts) for ts in state_history["timestamps"]]
    
    # Create figure
    fig = go.Figure()
    
    # Add state line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=states,
        mode='lines+markers',
        name='State',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add confidence
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=confidences,
        mode='lines',
        name='Confidence',
        line=dict(color='green', width=1, dash='dot'),
        yaxis="y2"
    ))
    
    # Update layout
    fig.update_layout(
        title="State History with Confidence",
        xaxis=dict(title="Time"),
        yaxis=dict(
            title="State",
            range=[-0.5, max(states) + 0.5],
            dtick=1,
            gridcolor='lightgray'
        ),
        yaxis2=dict(
            title="Confidence",
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            overlaying="y",
            side="right",
            range=[0, 1],
            gridcolor='lightgreen',
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

@app.callback(
    Output("price-signals-chart", "figure"),
    [Input("state-data", "children"), Input("signal-data", "children")]
)
def update_price_signals_chart(state_json, signal_json):
    """Update price and signals chart"""
    # Load state history for price data
    state_history = load_state_history()
    
    if not state_history or not state_history.get("prices"):
        return go.Figure()
    
    # Prepare price data
    prices = [float(p) for p in state_history["prices"]]
    timestamps = [datetime.fromisoformat(ts) for ts in state_history["timestamps"]]
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    # Load trade history for signals
    trade_history = load_trade_history()
    
    # Add entry signals
    entries = []
    for trade in trade_history["trade_history"]:
        if "entry_time" in trade:
            try:
                entry_time = datetime.fromisoformat(trade["entry_time"])
                # Find closest price point
                closest_idx = min(range(len(timestamps)), 
                                key=lambda i: abs(timestamps[i] - entry_time))
                
                if closest_idx < len(prices):
                    is_long = trade.get("direction") == "LONG"
                    entries.append({
                        "time": entry_time,
                        "price": prices[closest_idx],
                        "direction": "LONG" if is_long else "SHORT",
                        "color": "green" if is_long else "red",
                        "symbol": "triangle-up" if is_long else "triangle-down"
                    })
            except:
                pass
    
    # Add entry markers
    if entries:
        long_entries = [e for e in entries if e["direction"] == "LONG"]
        short_entries = [e for e in entries if e["direction"] == "SHORT"]
        
        if long_entries:
            fig.add_trace(go.Scatter(
                x=[e["time"] for e in long_entries],
                y=[e["price"] for e in long_entries],
                mode='markers',
                marker=dict(symbol="triangle-up", size=12, color="green"),
                name='Long Entry'
            ))
        
        if short_entries:
            fig.add_trace(go.Scatter(
                x=[e["time"] for e in short_entries],
                y=[e["price"] for e in short_entries],
                mode='markers',
                marker=dict(symbol="triangle-down", size=12, color="red"),
                name='Short Entry'
            ))
    
    # Update layout
    fig.update_layout(
        title="Price Chart with Trading Signals",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Price"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

@app.callback(
    Output("state-performance-chart", "figure"),
    [Input("performance-data", "children")]
)
def update_state_performance_chart(performance_json):
    """Update state performance chart"""
    if not performance_json:
        return go.Figure()
    
    performance = json.loads(performance_json)
    
    if "state_metrics" not in performance:
        return go.Figure()
    
    # Prepare data
    states = []
    win_rates = []
    avg_profits = []
    
    for state, metrics in performance["state_metrics"].items():
        states.append(f"State {state}")
        win_rates.append(metrics.get("win_rate", 0) * 100)
        avg_profits.append(metrics.get("avg_profit_pips", 0))
    
    # Create figure
    fig = go.Figure()
    
    # Add win rate bars
    fig.add_trace(go.Bar(
        x=states,
        y=win_rates,
        name='Win Rate (%)',
        marker_color='blue'
    ))
    
    # Add average profit line
    fig.add_trace(go.Scatter(
        x=states,
        y=avg_profits,
        mode='lines+markers',
        name='Avg. Profit (pips)',
        marker=dict(color='green', size=10),
        line=dict(color='green', width=2),
        yaxis="y2"
    ))
    
    # Update layout
    fig.update_layout(
        title="State Performance Metrics",
        xaxis=dict(title="State"),
        yaxis=dict(
            title="Win Rate (%)",
            range=[0, 100],
            gridcolor='lightgray'
        ),
        yaxis2=dict(
            title="Avg. Profit (pips)",
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            overlaying="y",
            side="right",
            gridcolor='lightgreen',
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

@app.callback(
    Output("trades-by-state-chart", "figure"),
    [Input("performance-data", "children")]
)
def update_trades_by_state_chart(performance_json):
    """Update trades by state chart"""
    if not performance_json:
        return go.Figure()
    
    performance = json.loads(performance_json)
    
    if "state_metrics" not in performance:
        return go.Figure()
    
    # Load trade history
    trade_history = load_trade_history()
    
    # Get counts by state
    state_counts = {}
    state_profits = {}
    
    for trade in trade_history.get("trade_history", []):
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
            colors.append(f"rgba(0, 128, 0, {0.5 + intensity/2})")  # Green for profit
        else:
            intensity = min(1.0, abs(profit) / 500)  # Scale for color intensity
            colors.append(f"rgba(255, 0, 0, {0.5 + intensity/2})")  # Red for loss
    
    # Create figure
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=colors)
    )])
    
    # Add total profit annotation
    total_profit = sum(state_profits.values())
    profit_text = f"Total: {total_profit:.1f} pips"
    
    fig.update_layout(
        title="Trades by State",
        annotations=[dict(
            text=profit_text,
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False
        )],
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

@app.callback(
    [Output("similar-patterns", "children"),
     Output("base-success-rate", "children"),
     Output("pattern-prediction", "children"),
     Output("memory-confidence-chart", "children")],
    [Input("memory-data", "children")]
)
def update_memory_display(memory_json):
    """Update market memory information"""
    if not memory_json:
        return "0", "0%", "Unknown", ""
    
    memory_data = json.loads(memory_json)
    
    # Get statistics
    stats = memory_data.get("statistics", {})
    pattern_count = stats.get("count", 0)
    
    # Get outcome distribution
    outcome_dist = stats.get("outcome_distribution", {})
    profitable_rate = outcome_dist.get("profitable", 0) * 100
    
    # In a real system, this would come from actual memory prediction
    prediction = "profitable" if profitable_rate > 50 else "loss"
    prediction_confidence = np.random.uniform(0.5, 0.9)
    
    # Color code prediction
    prediction_color = "green" if prediction == "profitable" else "red"
    prediction_text = html.Span(
        f"{prediction.capitalize()} ({prediction_confidence:.2f})",
        style={"color": prediction_color}
    )
    
    # Create confidence chart
    confidence_chart = dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=["Profitable", "Loss", "Unknown"],
                    y=[
                        outcome_dist.get("profitable", 0) * 100,
                        outcome_dist.get("loss", 0) * 100,
                        outcome_dist.get("unknown", 0) * 100
                    ],
                    marker_color=["green", "red", "gray"]
                )
            ]
        ).update_layout(
            title="Outcome Distribution",
            yaxis=dict(title="Percentage", range=[0, 100]),
            height=300
        ),
        config={'displayModeBar': False}
    )
    
    return str(pattern_count), f"{profitable_rate:.1f}%", prediction_text, confidence_chart

@app.callback(
    [Output("direction-prediction-chart", "figure"),
     Output("volatility-prediction-chart", "figure"),
     Output("rl-qvalues-chart", "figure")],
    [Input("hybrid-data", "children"), Input("state-data", "children")]
)
def update_hybrid_charts(hybrid_json, state_json):
    """Update hybrid model visualization charts"""
    if not hybrid_json or not state_json:
        return go.Figure(), go.Figure(), go.Figure()
    
    hybrid_data = json.loads(hybrid_json)
    state_data = json.loads(state_json)
    
    # Direction prediction chart
    # In a real system, these would come from actual model predictions
    direction_probs = {
        "up": 0.7 if "Bullish" in state_data.get("state_label", "") else 0.2,
        "down": 0.7 if "Bearish" in state_data.get("state_label", "") else 0.2,
        "sideways": 0.1
    }
    
    # Make sure probabilities sum to 1
    total_prob = sum(direction_probs.values())
    if total_prob > 0:
        direction_probs = {k: v/total_prob for k, v in direction_probs.items()}
    
    direction_fig = go.Figure([
        go.Bar(
            x=list(direction_probs.keys()),
            y=list(direction_probs.values()),
            marker_color=['green', 'red', 'blue']
        )
    ])
    
    direction_fig.update_layout(
        title="Direction Prediction",
        yaxis=dict(title="Probability", range=[0, 1]),
        height=250
    )
    
    # Volatility prediction chart
    # In a real system, these would come from actual model predictions
    volatility_probs = {
        "low": 0.7 if "Low" in state_data.get("state_label", "") else 0.2,
        "medium": 0.4 if "Medium" in state_data.get("state_label", "") else 0.3,
        "high": 0.7 if "High" in state_data.get("state_label", "") else 0.2
    }
    
    # Make sure probabilities sum to 1
    total_prob = sum(volatility_probs.values())
    if total_prob > 0:
        volatility_probs = {k: v/total_prob for k, v in volatility_probs.items()}
    
    volatility_fig = go.Figure([
        go.Bar(
            x=list(volatility_probs.keys()),
            y=list(volatility_probs.values()),
            marker_color=['blue', 'purple', 'red']
        )
    ])
    
    volatility_fig.update_layout(
        title="Volatility Prediction",
        yaxis=dict(title="Probability", range=[0, 1]),
        height=250
    )
    
    # RL Q-values chart
    # In a real system, these would come from actual model predictions
    actions = ["no_action", "enter_long", "enter_short", "exit_long", "exit_short"]
    
    # Generate Q-values based on state
    q_values = [0.0] * len(actions)
    
    if "Bullish" in state_data.get("state_label", ""):
        q_values[1] = np.random.uniform(0.6, 0.9)  # enter_long
    elif "Bearish" in state_data.get("state_label", ""):
        q_values[2] = np.random.uniform(0.6, 0.9)  # enter_short
    else:
        q_values[0] = np.random.uniform(0.6, 0.9)  # no_action
    
    # Add some random values to other actions
    for i in range(len(q_values)):
        if q_values[i] == 0:
            q_values[i] = np.random.uniform(0.1, 0.4)
    
    rl_fig = go.Figure([
        go.Bar(
            x=actions,
            y=q_values,
            marker_color=['gray', 'green', 'red', 'lightgreen', 'lightcoral']
        )
    ])
    
    rl_fig.update_layout(
        title="RL Action Q-Values",
        xaxis=dict(tickangle=45),
        yaxis=dict(title="Q-Value"),
        height=300
    )
    
    return direction_fig, volatility_fig, rl_fig

# Order Book Visualization
@app.callback(
    Output('order-book-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_order_book_graph(n):
    if 'order_book_processor' not in components or not hasattr(live_hmm, 'last_ob_analysis'):
        return go.Figure()
    
    # Get latest order book analysis
    ob_analysis = live_hmm.last_ob_analysis
    
    if not ob_analysis or 'features' not in ob_analysis:
        return go.Figure()
    
    # Create order book visualization
    fig = go.Figure()
    
    # Add bid volume bars
    if 'bid_level_1_ratio' in ob_analysis['features']:
        bid_ratios = [
            ob_analysis['features'].get('bid_level_1_ratio', 0),
            ob_analysis['features'].get('bid_level_2_ratio', 0),
            ob_analysis['features'].get('bid_level_3_ratio', 0)
        ]
        
        fig.add_trace(go.Bar(
            x=['Level 1', 'Level 2', 'Level 3'],
            y=bid_ratios,
            name='Bid Volume',
            marker_color='green'
        ))
    
    # Add ask volume bars
    if 'ask_level_1_ratio' in ob_analysis['features']:
        ask_ratios = [
            ob_analysis['features'].get('ask_level_1_ratio', 0),
            ob_analysis['features'].get('ask_level_2_ratio', 0),
            ob_analysis['features'].get('ask_level_3_ratio', 0)
        ]
        
        fig.add_trace(go.Bar(
            x=['Level 1', 'Level 2', 'Level 3'],
            y=ask_ratios,
            name='Ask Volume',
            marker_color='red'
        ))
    
    # Add imbalance indicator
    if 'imbalance' in ob_analysis['features']:
        imbalance = ob_analysis['features']['imbalance']
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=imbalance,
            title={'text': "Bid/Ask Imbalance"},
            domain={'x': [0.5, 1], 'y': [0, 0.3]},
            gauge={
                'axis': {'range': [0, 3]},
                'bar': {'color': "darkblue"},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1
                }
            }
        ))
    
    # Add anomaly indicator if detected
    if 'anomaly' in ob_analysis and ob_analysis['anomaly']:
        fig.add_annotation(
            text=f"ANOMALY: {ob_analysis['anomaly'].get('anomaly_type', 'unknown')}",
            x=0.5, y=1,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="red", size=16)
        )
    
    # Add synthetic data indicator
    if ob_analysis.get('is_synthetic', False):
        fig.add_annotation(
            text="SYNTHETIC DATA",
            x=0.5, y=0.9,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="orange", size=12)
        )
    
    fig.update_layout(
        title="Order Book Analysis",
        xaxis_title="Price Levels",
        yaxis_title="Volume Ratio",
        barmode='group'
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)