import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import logging

class DynamicFeatureSelector:
    def __init__(self, feature_cols, history_length=100):
        """Initialize the feature selector with enhanced data tracking capability"""
        self.feature_cols = feature_cols
        self.feature_importance = {feature: 1.0/len(feature_cols) for feature in feature_cols}
        self.history = []
        self.history_length = history_length
        self.state_specific_importance = {}
        
        # Enhanced data storage for consistent DataFrame access
        self._feature_matrix = None
        self._feature_df = None
        self._feature_times = None
        
        # Logger setup
        self.logger = logging.getLogger('dynamic_feature_selector')
        
    def store_feature_data(self, features, times=None):
        """
        Store full feature matrix and optionally time information
        to ensure DataFrame availability for all components.
        
        Args:
            features: Full feature matrix [samples, features]
            times: Optional time values for each sample
        """
        try:
            if isinstance(features, np.ndarray):
                self._feature_matrix = features.copy()
            else:
                self._feature_matrix = np.array(features)
                
            # Store time information if provided
            if times is not None:
                self._feature_times = times
                
            # Generate DataFrame immediately
            self._generate_feature_df()
            
            self.logger.info(f"Stored feature matrix with shape {self._feature_matrix.shape}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing feature data: {str(e)}")
            return False
    
    def _generate_feature_df(self):
        """
        Generate feature DataFrame from stored matrix data
        with advanced error handling and recovery mechanisms.
        """
        try:
            # Verify we have data to work with
            if self._feature_matrix is None or len(self._feature_matrix) == 0:
                self.logger.warning("No feature matrix available for DataFrame generation")
                self._feature_df = pd.DataFrame(columns=self.feature_cols)
                return
                
            # Verify dimension compatibility
            if len(self._feature_matrix.shape) < 2:
                # Single sample case - reshape
                data = self._feature_matrix.reshape(1, -1)
            else:
                data = self._feature_matrix
                
            # Handle dimension mismatch
            if data.shape[1] != len(self.feature_cols):
                self.logger.warning(f"Feature dimension mismatch: {data.shape[1]} vs {len(self.feature_cols)}")
                
                # Adjust to match feature_cols length
                if data.shape[1] > len(self.feature_cols):
                    # Too many features - truncate
                    data = data[:, :len(self.feature_cols)]
                else:
                    # Too few features - pad with zeros
                    padding = np.zeros((data.shape[0], len(self.feature_cols) - data.shape[1]))
                    data = np.hstack((data, padding))
            
            # Create DataFrame with feature names
            df = pd.DataFrame(data, columns=self.feature_cols)
            
            # Add time information if available
            if self._feature_times is not None:
                if len(self._feature_times) == len(df):
                    df['time'] = self._feature_times
                else:
                    self.logger.warning(f"Time dimension mismatch: {len(self._feature_times)} vs {len(df)}")
                    # Add time for as many rows as possible
                    min_len = min(len(self._feature_times), len(df))
                    df['time'] = pd.Series([self._feature_times[i] if i < min_len else None 
                                          for i in range(len(df))])
            
            # Add derived trading features for advanced modules
            self._add_derived_features(df)
            
            # Store the DataFrame
            self._feature_df = df
            self.logger.info(f"Generated feature DataFrame with shape {df.shape}")
            
        except Exception as e:
            self.logger.error(f"Error generating feature DataFrame: {str(e)}")
            # Create minimal DataFrame as fallback
            self._create_fallback_df()
    
    def _add_derived_features(self, df):
        """
        Add derived features needed by advanced components.
        
        Args:
            df: Feature DataFrame to enhance
        """
        try:
            # Add basic price columns expected by other modules
            if 'close' not in df.columns:
                # Simple heuristic: use first return column to approximate price
                return_cols = [col for col in df.columns if 'return' in col.lower() or 'log_return' in col.lower()]
                
                if return_cols and len(df) > 1:
                    # Reconstruct approximate price from return series
                    start_price = 100.0  # Arbitrary starting price
                    price_series = [start_price]
                    
                    # Calculate cumulative product of (1 + return)
                    for i in range(1, len(df)):
                        if 'log_return' in return_cols[0]:
                            # Convert log return to simple return
                            ret = np.exp(df[return_cols[0]].iloc[i]) - 1
                        else:
                            ret = df[return_cols[0]].iloc[i]
                        
                        next_price = price_series[-1] * (1 + ret)
                        price_series.append(next_price)
                    
                    df['close'] = price_series
                    
                    # Add other price columns
                    df['open'] = df['close'].shift(1).fillna(df['close'])
                    df['high'] = df['close'] * 1.001  # Approximation
                    df['low'] = df['close'] * 0.999   # Approximation
                else:
                    # No return information - add placeholder columns
                    df['close'] = 100.0
                    df['open'] = 100.0
                    df['high'] = 100.0
                    df['low'] = 100.0
            
            # Add ATR if it doesn't exist but can be calculated
            if 'atr_30m' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close']):
                # Simple ATR calculation (14-period)
                period = 14
                tr_list = []
                
                for i in range(len(df)):
                    if i == 0:
                        tr = df['high'].iloc[i] - df['low'].iloc[i]
                    else:
                        hl = df['high'].iloc[i] - df['low'].iloc[i]
                        hc = abs(df['high'].iloc[i] - df['close'].iloc[i-1])
                        lc = abs(df['low'].iloc[i] - df['close'].iloc[i-1])
                        tr = max(hl, hc, lc)
                    
                    tr_list.append(tr)
                
                # Calculate ATR
                df['atr_30m'] = pd.Series(tr_list).rolling(period).mean().fillna(method='bfill')
                
                # Add ATR average for volatility calculations
                df['atr_30m_avg'] = df['atr_30m'].rolling(50).mean().fillna(df['atr_30m'])
            
            # Add RSI if needed and possible
            if 'rsi_30m' not in df.columns and any(col in df.columns for col in ['close', 'open']):
                price_col = 'close' if 'close' in df.columns else 'open'
                # Simple RSI calculation (14-period)
                delta = df[price_col].diff().fillna(0)
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(14).mean().fillna(0)
                avg_loss = loss.rolling(14).mean().fillna(0)
                
                rs = avg_gain / avg_loss.replace(0, 1e-7)  # Avoid division by zero
                df['rsi_30m'] = 100 - (100 / (1 + rs))
            
            # Add state column if we have state information in history
            if self.history and all('state_type' in entry for entry in self.history):
                # Use the last state for all rows as simple approximation
                latest_state = self.history[-1]['state_type']
                df['state'] = latest_state
                
        except Exception as e:
            self.logger.warning(f"Error adding derived features: {str(e)}")
            # Continue without derived features
            
    def _create_fallback_df(self):
        """Create a minimal fallback DataFrame with expected structure"""
        df = pd.DataFrame(columns=self.feature_cols)
        
        # Add standard columns expected by other modules
        df['time'] = pd.Series(dtype='datetime64[ns]')
        df['close'] = pd.Series(dtype='float64')
        df['open'] = pd.Series(dtype='float64')
        df['high'] = pd.Series(dtype='float64')
        df['low'] = pd.Series(dtype='float64')
        df['volume'] = pd.Series(dtype='float64')
        df['state'] = pd.Series(dtype='object')
        
        # Store this empty DataFrame as fallback
        self._feature_df = df
        self.logger.warning("Created fallback feature DataFrame with 0 rows")
        
    def update_importance(self, features, state_label, prediction_success):
        """
        Updates feature importance based on prediction success
        """
        # --- Feature Selection Fix Start ---
        # Ensure we have enough samples for operations requiring neighbors
        # Determine the minimum number of samples needed (e.g., for potential KNN inside)
        MIN_SAMPLES_REQUIRED = 4 # Based on the error message n_neighbors=4
        
        if features.shape[0] < MIN_SAMPLES_REQUIRED:
            self.logger.warning(f"Skipping feature importance update: Need at least {MIN_SAMPLES_REQUIRED} samples, got {features.shape[0]}")
            return # Exit the function early if not enough samples
        # --- Feature Selection Fix End ---
        
        # Extract state type from label (e.g., "High Bullish" -> "high_bull")
        state_type = self._extract_state_type(state_label)
        
        # Initialize if this state hasn't been seen before
        if state_type not in self.state_specific_importance:
            self.state_specific_importance[state_type] = {
                feature: 1.0/len(self.feature_cols) for feature in self.feature_cols
            }
        
        # Calculate feature contributions using mutual information
        if isinstance(features, pd.DataFrame):
            X = features[self.feature_cols].values
        else:
            X = features
            
        # Create target based on prediction success (1=success, 0=failure)
        y = np.ones(X.shape[0]) if prediction_success else np.zeros(X.shape[0])
        
        # Calculate mutual information between features and success
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        mi_scores = mutual_info_regression(X_scaled, y)
        
        # Normalize scores
        mi_scores = mi_scores / (np.sum(mi_scores) + 1e-10)
        
        # Update state-specific importance with exponential moving average
        alpha = 0.1  # Learning rate
        for i, feature in enumerate(self.feature_cols):
            old_value = self.state_specific_importance[state_type][feature]
            new_value = (1-alpha) * old_value + alpha * mi_scores[i]
            self.state_specific_importance[state_type][feature] = new_value
        
        # Store history for attention mechanism
        self.history.append({
            'state_type': state_type,
            'features': X[-1].copy(),  # Store last feature vector
            'success': prediction_success
        })
        
        # Trim history if needed
        if len(self.history) > self.history_length:
            self.history.pop(0)
    
    def get_initial_weights(self, features):
        """
        Calculate initial feature weights based on training data only.
        
        Args:
            features: Feature matrix [samples, features] for training
        
        Returns:
            numpy array of feature weights
        """
        try:
            # Store feature data for later use
            self.store_feature_data(features)
            
            if len(features) < 10:
                # Not enough data for meaningful weights
                return np.ones(len(self.feature_cols)) / len(self.feature_cols)
            
            # Initialize uniform weights
            weights = np.ones(len(self.feature_cols))
            
            # Method 1: Use variance to determine feature importance
            feature_variance = np.var(features, axis=0)
            normalized_variance = feature_variance / np.sum(feature_variance)
            
            # Method 2: Use mutual information with price movement
            # We don't have labels here, so we'll create synthetic ones from sequential returns
            # This is a heuristic method but can help identify predictive features
            synthetic_targets = np.zeros(len(features) - 1)
            for i in range(len(features) - 1):
                # Assume first few columns are return-related
                if features.shape[1] > 4:
                    future_return = np.mean(features[i+1, :4])  # Average of first 4 features (returns)
                    synthetic_targets[i] = 1 if future_return > 0 else 0
            
            # Calculate mutual information if we have enough samples
            if len(synthetic_targets) > 20:
                # Use only training data for mutual info calculation
                features_subset = features[:-1, :]  # Match the synthetic_targets length
                
                # Scale features for better MI estimation
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(features_subset)
                
                # Calculate mutual information
                mi_scores = mutual_info_regression(X_scaled, synthetic_targets)
                mi_scores = np.nan_to_num(mi_scores)  # Replace NaN with 0
                
                # Normalize if sum is not zero
                if np.sum(mi_scores) > 0:
                    normalized_mi = mi_scores / np.sum(mi_scores)
                    
                    # Combine variance and mutual information (70% MI, 30% variance)
                    weights = 0.7 * normalized_mi + 0.3 * normalized_variance
                else:
                    weights = normalized_variance
            else:
                # Just use variance-based weights
                weights = normalized_variance
            
            # Ensure no zero weights (minimum weight is 0.1)
            weights = 0.1 + 0.9 * weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating initial weights: {str(e)}")
            # Return uniform weights as fallback
            return np.ones(len(self.feature_cols)) / len(self.feature_cols)
            
    def get_attention_weights(self, current_features, state_label):
        """
        Generates attention weights for features with enhanced robustness and quality.
        Adapts intelligently to available data quantity with progressive methods.

        Args:
            current_features: Current feature vector or sequence
            state_label: Current HMM state label

        Returns:
            numpy array: Feature weights optimized for current market context
        """
        # --- FIX: Ensure current_features is a valid numeric numpy array --- 
        try:
            # Ensure it's a numpy array first
            if not isinstance(current_features, np.ndarray):
                current_features = np.array(current_features)
            
            # Attempt conversion to float64, handling potential errors
            if current_features.dtype != np.float64:
                current_features = current_features.astype(np.float64)
                
        except (ValueError, TypeError) as e:
             self.logger.error(f"Could not convert current_features to float64 in get_attention_weights: {e}. Features: {current_features}")
             # Fallback to uniform weights if conversion fails
             return np.ones(len(self.feature_cols)) / len(self.feature_cols)
        
        # Check for NaN/Inf after conversion
        if np.any(np.isnan(current_features)) or np.any(np.isinf(current_features)):
            self.logger.warning("NaN or Inf detected in current_features in get_attention_weights. Using uniform weights.")
            return np.ones(len(self.feature_cols)) / len(self.feature_cols)
        # --- END FIX --- 
            
        # Stelle sicher, dass current_features die richtige Dimension hat
        if isinstance(current_features, np.ndarray):
            # Dimensionsprüfung und -anpassung
            expected_dim = len(self.feature_cols)
    
            # Vektorform sicherstellen
            if len(current_features.shape) > 1 and current_features.shape[0] == 1:
                current_features = current_features.flatten()
    
            # Dimension anpassen wenn nötig
            if len(current_features) != expected_dim:
                logging.debug(f"Feature-Dimensionen angepasst: {len(current_features)} -> {expected_dim}")
        
                if len(current_features) > expected_dim:
                    # Zu viele Features - abschneiden
                    current_features = current_features[:expected_dim]
                else:
                    # Zu wenige Features - mit Nullen auffüllen
                    temp = np.zeros(expected_dim)
                    temp[:len(current_features)] = current_features
                    current_features = temp

        # Extract state type from label (e.g., "High Bullish" -> "high_bull")
        state_type = self._extract_state_type(state_label)

        # If no history for this state, use uniform weights with heuristic insights
        if state_type not in self.state_specific_importance:
            # Even with no history, we can use heuristics based on market phases
            if "bull" in state_type:
                # In bullish markets, slightly favor momentum indicators
                weights = np.ones(len(self.feature_cols))
                for i, col in enumerate(self.feature_cols):
                    if any(term in col.lower() for term in ["rsi", "macd", "momentum"]):
                        weights[i] *= 1.2  # Boost momentum indicators by 20%
                return weights / np.sum(weights)
            elif "bear" in state_type:
                # In bearish markets, slightly favor volatility indicators
                weights = np.ones(len(self.feature_cols))
                for i, col in enumerate(self.feature_cols):
                    if any(term in col.lower() for term in ["atr", "vol", "range"]):
                        weights[i] *= 1.2  # Boost volatility indicators by 20%
                return weights / np.sum(weights)
        
            # No special treatment - use uniform weights
            return np.ones(len(self.feature_cols)) / len(self.feature_cols)

        # Get base importance from state-specific learning
        base_weights = np.array([self.state_specific_importance[state_type][f] 
                              for f in self.feature_cols])

        # For attention mechanism, we need sufficient history
        if len(self.history) < 5:  # Reduziert von 10 auf 5 für bessere Frühphase-Anpassung
            # Even with minimal history, we can apply simple enhancement
            self.logger.debug(f"Using base weights only due to limited history ({len(self.history)})")
            return base_weights

        # Create attention mechanism using TensorFlow with robust error handling
        try:
            # Filter history for current state type and ensure numeric type
            # --- FIX: Ensure history_vectors are numeric --- 
            history_vectors_raw = [h['features'] for h in self.history 
                                   if h['state_type'] == state_type and 'features' in h]
            
            # Convert and validate each vector in history
            history_vectors_list = []
            for vec in history_vectors_raw:
                try:
                    if not isinstance(vec, np.ndarray):
                        vec = np.array(vec)
                    if vec.dtype != np.float64:
                        vec = vec.astype(np.float64)
                    if not np.any(np.isnan(vec)) and not np.any(np.isinf(vec)):
                         history_vectors_list.append(vec)
                except (ValueError, TypeError):
                     self.logger.debug(f"Skipping invalid vector in history: {vec}")
            # --- END FIX --- 
            
            # ... rest of the checks for history length ...
            if len(history_vectors_list) < 2:
                 self.logger.debug(f"Insufficient valid numeric history for state {state_type} (only {len(history_vectors_list)} samples)")
                 return base_weights
                 
            # Convert to numpy array for consistency
            history_vectors = np.array(history_vectors_list)
            
            # Ensure history_vectors has the same number of features as current_features
            if history_vectors.shape[1] != current_features.shape[0]:
                self.logger.warning(f"Feature dimension mismatch between current ({current_features.shape[0]}) and history ({history_vectors.shape[1]}). Using base weights.")
                return base_weights

            # QUALITÄTSVERBESSERUNG: Adaptive attention mechanism based on data quantity
            if len(history_vectors) < 8:
                # Simple but effective method for small datasets
                # Compute similarity directly using Euclidean distance
                # --- FIX: Ensure query is float64 --- 
                query = current_features.astype(np.float64).reshape(1, -1)
                # history_vectors is already float64
                # --- END FIX --- 
                distances = np.linalg.norm(history_vectors - query, axis=1)
                similarity = 1.0 / (1.0 + distances)  # Convert distance to similarity
            
                # Sicherstellen, dass wir nicht mehr Nachbarn anfordern als verfügbar
                max_neighbors = min(4, len(history_vectors))
            
                # Finde die top N ähnlichen Vektoren
                top_indices = np.argsort(distances)[:max_neighbors]
                top_weights = similarity[top_indices]
                top_weights = top_weights / np.sum(top_weights)  # Normalisieren
            
                # Generate feature importance from similarity-weighted history
                weighted_avg = np.zeros(len(self.feature_cols))
                for i, idx in enumerate(top_indices):
                    weighted_avg += top_weights[i] * np.abs(history_vectors[idx])
            
                attention_importance = weighted_avg / (np.sum(weighted_avg) + 1e-10)
            else:
                # Full TensorFlow attention for larger datasets
                # --- FIX: Ensure tensors are float32 --- 
                query = tf.convert_to_tensor(current_features.reshape(1, -1), dtype=tf.float32)
                keys = tf.convert_to_tensor(history_vectors, dtype=tf.float32)
                # --- END FIX --- 
                # Scale dot-product for better numerical stability (similar to Transformer attention)
                scale_factor = 1.0 / np.sqrt(keys.shape[1])
                attention_scores = tf.matmul(query, keys, transpose_b=True) * scale_factor
                attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
                # Weighted combination of history vectors
                weighted_history = tf.matmul(attention_weights, keys)
        
                # Generate feature importance from attention mechanism
                attention_importance = np.abs(weighted_history.numpy()[0])
        
                # Normalize but avoid division by zero
                sum_importance = np.sum(attention_importance)
                if sum_importance > 1e-10:
                    attention_importance = attention_importance / sum_importance
                else:
                    attention_importance = np.ones_like(attention_importance) / len(attention_importance)
    
            # QUALITÄTSVERBESSERUNG: Adaptive mixing ratio based on history size
            # More data = more weight to attention mechanism
            history_confidence = min(0.8, len(history_vectors) / 20)  # Cap at 0.8
    
            # Combine base weights with attention weights using adaptive ratio
            combined_weights = (1 - history_confidence) * base_weights + history_confidence * attention_importance
    
            # Ensure no zeros by adding small epsilon and re-normalizing
            combined_weights = combined_weights + 1e-5
            return combined_weights / np.sum(combined_weights)
    
        except Exception as e:
            # Detailed error logging for future improvements
            self.logger.warning(f"Error in attention mechanism: {str(e)}")
            self.logger.debug(f"Feature shapes: current={np.shape(current_features)}, history={len(self.history)}")
    
            # Return reliable base weights instead of failing
            return base_weights

    def get_weights(self):
        """
        Returns current feature weights as a numpy array in the same order as feature_cols.
    
        Returns:
            numpy array: Feature weights in the same order as self.feature_cols
        """
        # Convert dictionary of feature importance to numpy array
        weights = np.array([self.feature_importance.get(feature, 1.0/len(self.feature_cols)) 
                          for feature in self.feature_cols])
    
        # Normalize weights to sum to number of features (for weighted average)
        weights = weights * len(weights) / np.sum(weights)
    
        return weights
    
    def _extract_state_type(self, state_label):
        """Extract state type from label"""
        if state_label is None:
            return "unknown"
            
        # Convert numeric state to string if needed
        if isinstance(state_label, (int, float)):
            return f"state_{int(state_label)}"
        
        # Extract volatility
        volatility = "medium"
        if "High" in str(state_label):
            volatility = "high"
        elif "Low" in str(state_label):
            volatility = "low"
            
        # Extract direction
        direction = "neutral"
        if any(term in str(state_label) for term in ["Bullish", "Bull"]):
            direction = "bull"
        elif any(term in str(state_label) for term in ["Bearish", "Bear"]):
            direction = "bear"
            
        return f"{volatility}_{direction}"

    def get_feature_df(self):
        """
        Returns the feature DataFrame with all necessary data for advanced modules.
        
        This enhanced implementation ensures that DataFrame is always available
        with properly structured data, even if historical data is limited.
        
        Returns:
            pandas.DataFrame: DataFrame containing features and metadata
        """
        # If we already have a generated DataFrame, return it
        if self._feature_df is not None and len(self._feature_df) > 0:
            return self._feature_df
            
        # If we have the feature matrix but no DataFrame, generate it
        if self._feature_matrix is not None and self._feature_matrix.shape[0] > 0:
            self._generate_feature_df()
            return self._feature_df
            
        # If we have history data, try to create a DataFrame from it
        if self.history and len(self.history) > 0:
            try:
                # Extract features from history
                features_list = [entry['features'] for entry in self.history if 'features' in entry]
                if features_list:
                    # Create DataFrame with feature names as columns
                    df = pd.DataFrame(features_list, columns=self.feature_cols)
                
                    # Add state information if available
                    if all('state_type' in entry for entry in self.history):
                        df['state_type'] = [entry['state_type'] for entry in self.history]
                
                    # Add success information if available
                    if all('success' in entry for entry in self.history):
                        df['prediction_success'] = [entry['success'] for entry in self.history]
                
                    # Add derived features
                    self._add_derived_features(df)
                    
                    # Save and return the DataFrame
                    self._feature_df = df
                    self.logger.info(f"Created DataFrame from history with {len(df)} rows")
                    return df
            except Exception as e:
                self.logger.error(f"Error creating DataFrame from history: {str(e)}")
        
        # No valid data available - create fallback DataFrame
        self._create_fallback_df()
        return self._feature_df