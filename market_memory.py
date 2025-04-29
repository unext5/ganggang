import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging
from collections import deque
import time
import json
from dtw import dtw

class MarketMemory:
    """
    Advanced market memory component that records market conditions and finds
    similar historical patterns to improve prediction accuracy.
    """
    def __init__(self, max_patterns=1000, similarity_window=50, filename="market_memory.pkl"):
        """
        Initializes the market memory component.
        
        Args:
            max_patterns: Maximum number of patterns to store
            similarity_window: Window size for pattern matching
            filename: File to save/load memory data
        """
        self.max_patterns = max_patterns
        self.similarity_window = similarity_window
        self.filename = filename
        self.patterns = deque(maxlen=max_patterns)
        self.scaler = StandardScaler()
        self.knn_model = None
        self.scaler_fitted = False
        self.last_update = None
        
        # Pattern outcome statistics
        self.outcome_stats = {
            "bullish": {"count": 0, "success": 0},
            "bearish": {"count": 0, "success": 0},
            "high_vol": {"count": 0, "success": 0},
            "low_vol": {"count": 0, "success": 0}
        }
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('market_memory')

        # Überprüfe DTW-Bibliotheksversion und setze passende Funktionen
        self._setup_dtw_functions()
        
        # Load existing memory if available
        self.load_memory()

    def _setup_dtw_functions(self):
        """
        Erkennt die installierte DTW-Version und richtet die passenden Funktionen ein.
        Verbesserte Version mit umfassender Fehlerbehandlung.
        """
        try:
            # Import DTW library safely
            from dtw import dtw
        
            # Define a standard distance function with enhanced type handling
            def euclidean_dist(x, y):
                """Sichere Euklidische Distanzfunktion für Skalare oder 1D-Arrays."""
                try:
                    # Convert inputs to numpy arrays of float64
                    x_arr = np.asarray(x, dtype=np.float64)
                    y_arr = np.asarray(y, dtype=np.float64)

                    # Check for NaN/Inf values
                    if np.any(np.isnan(x_arr)) or np.any(np.isinf(x_arr)) or \
                       np.any(np.isnan(y_arr)) or np.any(np.isinf(y_arr)):
                        #self.logger.debug("NaN/Inf detected in euclidean_dist input")
                        return 9999.0 # Return high distance for invalid inputs

                    # Ensure arrays are 1D for norm calculation
                    if x_arr.ndim > 1 or y_arr.ndim > 1:
                         #self.logger.debug(f"Unexpected dimensions in euclidean_dist: {x_arr.shape}, {y_arr.shape}")
                         # Flatten or handle multi-dimensional case if needed, otherwise return high distance
                         # Assuming DTW provides 1D vectors (features at one time step)
                         x_arr = x_arr.flatten()
                         y_arr = y_arr.flatten()

                    # Check if shapes match after potential flattening
                    if x_arr.shape != y_arr.shape:
                        #self.logger.debug(f"Shape mismatch in euclidean_dist: {x_arr.shape} vs {y_arr.shape}")
                        # Decide how to handle mismatch (e.g., pad, truncate, or return high distance)
                        # Returning high distance for now
                        return 9999.0
                        
                    # Calculate Euclidean distance
                    dist = np.linalg.norm(x_arr - y_arr)
                    return dist if np.isfinite(dist) else 9999.0

                except (ValueError, TypeError) as e:
                    #self.logger.debug(f"Error calculating euclidean distance: {e}")
                    return 9999.0 # Return high distance on error
        
            # Test different API versions with proper exception handling
            try:
                # Zuerst mit dem dist-Parameter probieren
                test_result = dtw(np.array([1, 2, 3]), np.array([1, 2, 3]), dist=euclidean_dist)
                self.dtw_version = "dist_param"
                self.logger.info("Using DTW API with dist parameter")
            
                # --- FIX: Extract only distance from result ---
                # dtw with dist often returns a tuple (distance, cost_matrix, acc_cost_matrix, path)
                self.dtw_function = lambda seq1, seq2: dtw(seq1, seq2, dist=euclidean_dist)[0] if isinstance(dtw(seq1, seq2, dist=euclidean_dist), tuple) else dtw(seq1, seq2, dist=euclidean_dist)

            except (TypeError, ValueError) as e:
                try:
                    # Try the new API with distance_method
                    test_result = dtw(np.array([1, 2, 3]), np.array([1, 2, 3]), distance_method="euclidean")
                    self.dtw_version = "new"
                    self.logger.info("Using new DTW API with distance_method parameter")
                
                    # --- FIX: Extract only distance from result ---
                    # This version might also return a result object with attributes or just the distance
                    def dtw_wrapper_new(seq1, seq2):
                        res = dtw(seq1, seq2, distance_method="euclidean")
                        if hasattr(res, 'distance'): return res.distance
                        if hasattr(res, 'normalizedDistance'): return res.normalizedDistance # Alternative attribute
                        if isinstance(res, (float, int)): return res
                        if isinstance(res, tuple) and len(res) > 0: return res[0] # Fallback for tuple
                        self.logger.warning(f"Unexpected DTW result type (new API): {type(res)}")
                        return np.inf
                    self.dtw_function = dtw_wrapper_new

                except (TypeError, ValueError) as e2:
                    # Fallback-Implementierungen probieren
                    try:
                        # Try legacy API
                        test_result = dtw(np.array([1, 2, 3]), np.array([1, 2, 3]), distance_only=True)
                        self.dtw_version = "legacy"
                        self.logger.info("Using legacy DTW API with distance_only parameter")
                    
                        # --- FIX: Ensure distance_only=True returns scalar ---
                        # This should already return only the distance
                        self.dtw_function = lambda seq1, seq2: dtw(seq1, seq2, distance_only=True)
                    except Exception as e3:
                        # Custom DTW implementation as last resort
                        self.dtw_version = "custom"
                        self.logger.info("Using custom DTW implementation")
                    
                        def simple_dtw_distance(seq1, seq2):
                            """Eigene DTW-Implementierung als Fallback"""
                            try:
                                import numpy as np
                                # Konvertiere zu float64 numpy-Arrays
                                seq1 = np.asarray(seq1, dtype=np.float64)
                                seq2 = np.asarray(seq2, dtype=np.float64)
                                
                                # Überprüfe Dimensionen
                                if len(seq1.shape) > 1 and len(seq2.shape) > 1:
                                    # Beide Sequenzen sind 2D - Vergleiche jede Zeile
                                    n, m = seq1.shape[0], seq2.shape[0]
                                    dtw_matrix = np.zeros((n+1, m+1))
                                    dtw_matrix[0, :] = np.inf
                                    dtw_matrix[:, 0] = np.inf
                                    dtw_matrix[0, 0] = 0
                                    
                                    for i in range(1, n+1):
                                        for j in range(1, m+1):
                                            # Verwende Euklidische Distanz für die Zeilen
                                            cost = np.sqrt(np.sum((seq1[i-1] - seq2[j-1]) ** 2))
                                            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], 
                                                                    dtw_matrix[i, j-1], 
                                                                    dtw_matrix[i-1, j-1])
                                else:
                                    # Mindestens eine Sequenz ist 1D
                                    n, m = len(seq1), len(seq2)
                                    dtw_matrix = np.zeros((n+1, m+1))
                                    dtw_matrix[0, :] = np.inf
                                    dtw_matrix[:, 0] = np.inf
                                    dtw_matrix[0, 0] = 0
                                
                                    for i in range(1, n+1):
                                        for j in range(1, m+1):
                                            cost = np.abs(seq1[i-1] - seq2[j-1])
                                            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], 
                                                                    dtw_matrix[i, j-1], 
                                                                    dtw_matrix[i-1, j-1])
                                
                                return float(dtw_matrix[n, m])
                            except Exception as e:
                                self.logger.warning(f"Error in simple_dtw_distance: {str(e)}")
                                return 9999.0  # Große Distanz bei Fehler zurückgeben
                    
                        self.dtw_function = simple_dtw_distance
        
        except ImportError:
            # DTW library not installed
            self.dtw_version = "fallback"
            self.logger.warning("DTW library not found - using Euclidean distance fallback")
        
            # Use Euclidean distance as fallback
            def euclidean_distance(seq1, seq2):
                import numpy as np
                try:
                    # Ensure sequences are the same length by trimming
                    seq1 = np.asarray(seq1, dtype=np.float64)
                    seq2 = np.asarray(seq2, dtype=np.float64)
                    
                    if len(seq1.shape) != len(seq2.shape):
                        # Unterschiedliche Dimensionen - handle speziell
                        self.logger.warning(f"Dimension mismatch in euclidean_distance: {seq1.shape} vs {seq2.shape}")
                        return 9999.0
                    
                    if len(seq1.shape) > 1:
                        # Für Matrizen, verwende die minimale Zeilenanzahl
                        min_rows = min(seq1.shape[0], seq2.shape[0])
                        min_cols = min(seq1.shape[1], seq2.shape[1])
                        a = seq1[:min_rows, :min_cols]
                        b = seq2[:min_rows, :min_cols]
                        # Berechne Euklidische Distanz zwischen allen Zeilen
                        return float(np.mean([np.sqrt(np.sum((a[i] - b[i]) ** 2)) for i in range(min_rows)]))
                    else:
                        # Für Vektoren, verwende die minimale Länge
                        min_len = min(len(seq1), len(seq2))
                        a = seq1[:min_len]
                        b = seq2[:min_len]
                        return float(np.sqrt(np.sum((a - b) ** 2)))
                except Exception as e:
                    self.logger.warning(f"Error in euclidean_distance fallback: {str(e)}")
                    return 9999.0
        
            self.dtw_function = euclidean_distance
    
    def add_pattern(self, features, state_label, outcome=None, price_sequence=None):
        """
        Adds a new market pattern to memory.
        
        Args:
            features: Feature sequence (numpy array or list of arrays)
            state_label: Label of the state identified by HMM
            outcome: Optional outcome information (e.g., "profitable", "loss")
            price_sequence: Optional price data corresponding to features
        """
        # Ensure features is a numpy array
        if isinstance(features, list):
            if all(isinstance(f, np.ndarray) for f in features):
                features = np.vstack(features)
            else:
                features = np.array(features)
        
        # Extract basic characteristic from state label
        characteristics = self._extract_characteristics(state_label)
        
        # Create pattern object
        pattern = {
            "features": features.copy(),
            "state_label": state_label,
            "characteristics": characteristics,
            "timestamp": datetime.now(),
            "outcome": outcome,
            "price_sequence": price_sequence
        }
        
        # Add to collection
        self.patterns.append(pattern)
        
        # If outcome provided, update statistics
        if outcome:
            self._update_outcome_statistics(characteristics, outcome)
        
        # Update KNN model if needed (periodically)
        self.last_update = datetime.now()
        
        # Save memory periodically
        if len(self.patterns) % 50 == 0:
            self.save_memory()
    
    def find_similar_patterns(self, current_features, n_neighbors=5, method="knn"):
        """
        Finds patterns similar to the current market conditions.
        
        Args:
            current_features: Current feature vector or sequence
            n_neighbors: Number of similar patterns to return
            method: Similarity method ("knn" or "dtw")
        
        Returns:
            list: Similar patterns with similarity scores
        """
        if len(self.patterns) < n_neighbors:
            self.logger.warning("Not enough patterns in memory for reliable matching")
            return []
        
        # Ensure current_features is a numpy array
        if isinstance(current_features, list):
            current_features = np.array(current_features)
        
        # Single feature vector vs. sequence
        is_sequence = len(current_features.shape) > 1 and current_features.shape[0] > 1
        
        if method == "knn" and not is_sequence:
            return self._find_similar_knn(current_features, n_neighbors)
        elif method == "dtw" and is_sequence:
            return self._find_similar_dtw(current_features, n_neighbors)
        else:
            # Default to appropriate method based on input
            if is_sequence:
                return self._find_similar_dtw(current_features, n_neighbors)
            else:
                return self._find_similar_knn(current_features, n_neighbors)
    
    def _find_similar_knn(self, current_features, n_neighbors):
        """
        Uses K-nearest neighbors to find similar patterns.
        """
        # Prepare feature vectors from all patterns
        if len(self.patterns) == 0:
            return []
        
        # Build or update KNN model
        self._update_knn_model()
        
        # Scale the query vector
        if self.scaler_fitted:
            current_scaled = self.scaler.transform(current_features.reshape(1, -1))
        else:
            current_scaled = current_features.reshape(1, -1)
        
        # Find nearest neighbors
        try:
            distances, indices = self.knn_model.kneighbors(
                current_scaled, n_neighbors=min(n_neighbors, len(self.patterns))
            )
            
            # Prepare results
            similar_patterns = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                pattern = list(self.patterns)[idx]
                similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                
                similar_patterns.append({
                    "pattern": pattern,
                    "similarity": similarity_score,
                    "rank": i+1
                })
            
            return similar_patterns
        
        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {str(e)}")
            return []
    
    def _find_similar_dtw(self, current_sequence, n_neighbors):
        """
        Uses Dynamic Time Warping to find similar time series patterns.
        """
        if len(self.patterns) == 0:
            return []
    
        # Get sequence length
        seq_len = current_sequence.shape[0]
    
        # Calculate DTW distances to all patterns
        distances = []
        valid_patterns = []
    
        pattern_list = list(self.patterns) # Copy to avoid modification issues

        for i, pattern_data in enumerate(pattern_list):
            # Skip patterns without sequence data of sufficient length
            if "features" not in pattern_data or not isinstance(pattern_data["features"], np.ndarray):
                continue
                
            if len(pattern_data["features"].shape) < 2 or pattern_data["features"].shape[0] < seq_len / 2:
                continue
        
            # Use only last part of longer sequences
            pattern_seq = pattern_data["features"][-seq_len:] if pattern_data["features"].shape[0] > seq_len else pattern_data["features"]
        
            # Ensure sequences are C-contiguous arrays of type float64
            try:
                seq1 = np.ascontiguousarray(current_sequence, dtype=np.float64)
                seq2 = np.ascontiguousarray(pattern_seq, dtype=np.float64) # Use original pattern_seq for conversion check
                 
                # Additional check for NaN/Inf AFTER conversion attempt
                if np.any(np.isnan(seq1)) or np.any(np.isinf(seq1)) or \
                   np.any(np.isnan(seq2)) or np.any(np.isinf(seq2)):
                    self.logger.warning(f"NaN or Inf found in sequences for pattern {i} after conversion. Skipping DTW.")
                    continue
            except (ValueError, TypeError) as ve:
                 self.logger.warning(f"Could not convert sequences to float64 for pattern {i}: {ve}. Skipping DTW.")
                 continue

            # --- START: Inner Try-Except for DTW call ---    
            try:
                # --- DTW CALCULATION --- 
                # Use the selected DTW function
                alignment = self.dtw_function(seq1, seq2)

                # --- FIX START: Robust distance extraction --- 
                distance_value = np.inf # Default to infinite distance if extraction fails
                
                # Try extracting based on common attributes of dtw result objects
                if hasattr(alignment, 'distance'):
                    distance_value = alignment.distance
                elif hasattr(alignment, 'normalizedDistance'): 
                    distance_value = alignment.normalizedDistance 
                elif isinstance(alignment, (int, float)): # Handle simple return value
                    distance_value = alignment
                else:
                    # Fallback if the object structure is unknown
                    self.logger.warning(f"Could not extract distance from DTW result object of type {type(alignment)} for pattern {i}. Using inf.")
                    
                # Ensure distance_value is a scalar float
                if not isinstance(distance_value, (int, float)):
                    self.logger.warning(f"Extracted DTW distance is not a scalar ({type(distance_value)}) for pattern {i}. Using inf.")
                    distance_value = np.inf
                elif np.isnan(distance_value) or np.isinf(distance_value):
                     # Handle potential NaN/Inf values explicitly
                     distance_value = np.inf
                # --- FIX END ---
                 
                # Only append if distance is valid
                if distance_value != np.inf:
                    distances.append(float(distance_value))
                    valid_patterns.append(i)
                else:
                    self.logger.debug(f"Skipping pattern {i} due to invalid DTW distance.")
            
            except Exception as e:
                # Log specific error from self.dtw_function call
                self.logger.error(f"Error during DTW calculation for pattern {i}: {str(e)}. "
                                   f"Current shape: {seq1.shape}, Pattern shape: {seq2.shape}")
                # Skip this pattern if DTW calculation fails
                continue 
            # --- END: Inner Try-Except for DTW call ---
        
        if not distances:
            return []
        
        # Find n_neighbors smallest distances
        neighbor_indices = np.argsort(distances)[:n_neighbors]
        
        # Prepare results
        similar_patterns = []
        
        for i, idx in enumerate(neighbor_indices):
            pattern_idx = valid_patterns[idx]
            pattern = list(self.patterns)[pattern_idx]
            distance = distances[idx]
            similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
            
            similar_patterns.append({
                "pattern": pattern,
                "similarity": similarity_score,
                "rank": i+1
            })
        
        return similar_patterns
    
    def predict_outcome(self, current_features, state_label, n_neighbors=5):
        """
        Predicts likely outcome based on similar historical patterns.
        MODIFIED: Distinguishes between bullish and bearish profitable outcomes.
        
        Args:
            current_features: Current feature vector or sequence
            state_label: Current state label
            n_neighbors: Number of similar patterns to consider
        
        Returns:
            dict: Prediction with signal ('BUY', 'SELL', 'NEUTRAL'), strength, and supporting data
        """
        # Get similar patterns
        similar_patterns = self.find_similar_patterns(current_features, n_neighbors)
        
        if not similar_patterns:
            return {
                "signal": "NONE", # Use NONE instead of prediction
                "strength": 0.0,
                "similar_patterns_count": 0,
                "confidence": 0.0,
                "reason": "insufficient_data"
            }
        
        # Count outcomes and characteristics from similar patterns
        profitable_bullish = 0
        profitable_bearish = 0
        loss_bullish = 0
        loss_bearish = 0
        neutral_outcomes = 0
        unknown_outcomes = 0

        weighted_bullish_profit = 0.0
        weighted_bearish_profit = 0.0
        weighted_bullish_loss = 0.0
        weighted_bearish_loss = 0.0
        total_similarity = sum(p['similarity'] for p in similar_patterns)
        
        for pattern_info in similar_patterns:
            pattern = pattern_info["pattern"]
            similarity = pattern_info["similarity"]
            pattern_chars = pattern.get("characteristics", [])
            outcome = pattern.get("outcome")

            is_bullish_pattern = any(c in pattern_chars for c in ["bullish", "trending_bull"])
            is_bearish_pattern = any(c in pattern_chars for c in ["bearish", "trending_bear"])

            if outcome == "profitable":
                if is_bullish_pattern:
                    profitable_bullish += 1
                    weighted_bullish_profit += similarity
                elif is_bearish_pattern:
                    profitable_bearish += 1
                    weighted_bearish_profit += similarity
                else: # Profitable but neither clearly bullish nor bearish
                    neutral_outcomes += 1
            elif outcome == "loss":
                if is_bullish_pattern:
                    loss_bullish += 1
                    weighted_bullish_loss += similarity
                elif is_bearish_pattern:
                    loss_bearish += 1
                    weighted_bearish_loss += similarity
                else: # Loss but neither clearly bullish nor bearish
                    neutral_outcomes += 1
            elif outcome == "neutral":
                 neutral_outcomes += 1
            else:
                unknown_outcomes += 1

        # Determine dominant signal based on weighted profitable outcomes
        buy_score = weighted_bullish_profit - weighted_bullish_loss # Net bullish confidence
        sell_score = weighted_bearish_profit - weighted_bearish_loss # Net bearish confidence

        # Normalize scores (optional, but good for strength interpretation)
        # Max possible score if all neighbors are perfect matches (similarity=1)
        max_possible_score = total_similarity if total_similarity > 0 else 1.0
        
        buy_strength = buy_score / max_possible_score if max_possible_score > 0 else 0.0
        sell_strength = sell_score / max_possible_score if max_possible_score > 0 else 0.0

        final_signal = "NONE"
        final_strength = 0.0
        confidence = 0.0

        # --- Determine Final Signal and Strength --- 
        # Simple logic: choose the direction with the higher net score
        # Add a minimum threshold to avoid weak signals
        min_net_strength_threshold = 0.05 # Needs tuning

        if buy_strength > sell_strength and buy_strength > min_net_strength_threshold:
            final_signal = "BUY"
            final_strength = buy_strength
            # Confidence can be related to the difference or consistency
            confidence = (profitable_bullish / (profitable_bullish + loss_bullish)) if (profitable_bullish + loss_bullish) > 0 else 0.0
        elif sell_strength > buy_strength and sell_strength > min_net_strength_threshold:
            final_signal = "SELL"
            final_strength = sell_strength # Strength is positive, signal indicates direction
            confidence = (profitable_bearish / (profitable_bearish + loss_bearish)) if (profitable_bearish + loss_bearish) > 0 else 0.0
        else:
            # If scores are close or below threshold, signal is NEUTRAL
            final_signal = "NONE"
            final_strength = 0.0
            confidence = 0.0

        # Clamp strength and confidence between 0 and 1
        final_strength = max(0.0, min(1.0, final_strength))
        confidence = max(0.0, min(1.0, confidence))

        # Return signal in the expected format
        return {
            "signal": final_signal,
            "strength": final_strength,
            "similar_patterns_count": len(similar_patterns),
            "confidence": confidence, # Use confidence for weighting?
            "details": {
                "profitable_bullish": profitable_bullish,
                "profitable_bearish": profitable_bearish,
                "loss_bullish": loss_bullish,
                "loss_bearish": loss_bearish
            }
        }
    
    def get_pattern_statistics(self):
        """
        Returns statistics about stored patterns.
        
        Returns:
            dict: Pattern statistics
        """
        if not self.patterns:
            return {
                "count": 0,
                "state_distribution": {},
                "outcome_distribution": {}
            }
        
        # Count patterns by state and outcome
        state_counts = {}
        outcome_counts = {"profitable": 0, "loss": 0, "neutral": 0, "unknown": 0}
        
        for pattern in self.patterns:
            # State counts
            state = pattern["state_label"]
            if state in state_counts:
                state_counts[state] += 1
            else:
                state_counts[state] = 1
            
            # Outcome counts
            outcome = pattern.get("outcome", "unknown")
            if outcome in outcome_counts:
                outcome_counts[outcome] += 1
            else:
                outcome_counts["unknown"] += 1
        
        # Calculate percentages
        total = len(self.patterns)
        state_distribution = {state: count/total for state, count in state_counts.items()}
        outcome_distribution = {outcome: count/total for outcome, count in outcome_counts.items()}
        
        return {
            "count": total,
            "state_distribution": state_distribution,
            "outcome_distribution": outcome_distribution,
            "outcome_stats": self.outcome_stats
        }
    
    def save_memory(self, filepath=None):
        """
        Saves the market memory to disk.
    
        Args:
            filepath: Optional path to save the file, defaults to self.filename
        """
        try:
            # Use provided filepath or default
            save_file = filepath if filepath is not None else self.filename
        
            # Prepare data to save
            save_data = {
                "patterns": list(self.patterns),
                "outcome_stats": self.outcome_stats,
                "timestamp": datetime.now().isoformat()
            }
        
            # Use temporary file for safe saving
            temp_file = f"{save_file}.tmp"
        
            with open(temp_file, 'wb') as f:
                pickle.dump(save_data, f)
        
            # If successful, replace the original file
            if os.path.exists(save_file):
                os.replace(save_file, f"{save_file}.bak")  # Create backup
        
            os.replace(temp_file, save_file)
            self.logger.info(f"Successfully saved market memory with {len(self.patterns)} patterns")
        
            # Also save statistics in readable format
            stats = self.get_pattern_statistics()
            stats_file = save_file.replace('.pkl', '_stats.json')
        
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving market memory: {str(e)}")
            return False
    
    def load_memory(self):
        """
        Loads the market memory from disk.
        
        Returns:
            bool: Success status
        """
        if not os.path.exists(self.filename):
            self.logger.info(f"No market memory file found at {self.filename}")
            return False
        
        try:
            with open(self.filename, 'rb') as f:
                data = pickle.load(f)
            
            # Load patterns
            if "patterns" in data and isinstance(data["patterns"], list):
                self.patterns = deque(data["patterns"], maxlen=self.max_patterns)
                self.logger.info(f"Loaded {len(self.patterns)} patterns from memory")
            
            # Load outcome statistics
            if "outcome_stats" in data:
                self.outcome_stats = data["outcome_stats"]
            
            # Rebuild the KNN model
            if len(self.patterns) > 10:
                self._update_knn_model(force=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading market memory: {str(e)}")
            return False
    
    def prune_memory(self, max_age_days=90):
        """
        Removes old patterns to maintain memory quality.
        
        Args:
            max_age_days: Maximum age of patterns to keep
        """
        if not self.patterns:
            return
        
        current_time = datetime.now()
        max_age = timedelta(days=max_age_days)
        
        # Count before pruning
        before_count = len(self.patterns)
        
        # Filter patterns by age
        recent_patterns = deque(maxlen=self.max_patterns)
        
        for pattern in self.patterns:
            if "timestamp" in pattern:
                # Parse timestamp if it's a string
                if isinstance(pattern["timestamp"], str):
                    try:
                        pattern_time = datetime.fromisoformat(pattern["timestamp"])
                    except:
                        pattern_time = current_time  # Default to current time if parsing fails
                else:
                    pattern_time = pattern["timestamp"]
                
                # Keep if recent enough
                if current_time - pattern_time <= max_age:
                    recent_patterns.append(pattern)
            else:
                # No timestamp, keep it
                recent_patterns.append(pattern)
        
        # Update patterns
        self.patterns = recent_patterns
        
        # Rebuild KNN model if substantial change
        if before_count - len(self.patterns) > 10:
            self._update_knn_model(force=True)
        
        self.logger.info(f"Pruned memory: {before_count} -> {len(self.patterns)} patterns")
    
    def _update_knn_model(self, force=False):
        """
        Updates the KNN model for efficient similarity search.
        """
        # Skip if model is recent and not forced
        if self.knn_model is not None and not force and len(self.patterns) > 0:
            if self.last_update and (datetime.now() - self.last_update).total_seconds() < 3600:
                return
        
        # Need enough patterns to build model
        if len(self.patterns) < 5:
            return
        
        try:
            # Extract feature vectors from all patterns
            feature_vecs = []
            
            for pattern in self.patterns:
                if "features" in pattern:
                    if len(pattern["features"].shape) > 1:
                        # For sequences, use the last vector
                        feature_vecs.append(pattern["features"][-1])
                    else:
                        feature_vecs.append(pattern["features"])
            
            if not feature_vecs:
                return
            
            X = np.vstack(feature_vecs)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.scaler_fitted = True
            
            # Create KNN model
            self.knn_model = NearestNeighbors(n_neighbors=min(10, len(feature_vecs)), 
                                            algorithm='auto', metric='euclidean')
            self.knn_model.fit(X_scaled)
            
            self.logger.info(f"KNN model updated with {len(feature_vecs)} patterns")
            
        except Exception as e:
            self.logger.error(f"Error updating KNN model: {str(e)}")
    
    def _extract_characteristics(self, state_label):
        """
        Extracts market characteristics from state label.
        """
        characteristics = []
        
        # Trend
        if "Bullish" in state_label or "Bull" in state_label:
            characteristics.append("bullish")
        elif "Bearish" in state_label or "Bear" in state_label:
            characteristics.append("bearish")
        
        # Volatility
        if "High" in state_label:
            characteristics.append("high_vol")
        elif "Low" in state_label:
            characteristics.append("low_vol")
        
        return characteristics
    
    def _update_outcome_statistics(self, characteristics, outcome):
        """
        Updates outcome statistics for market characteristics.
        """
        for characteristic in characteristics:
            if characteristic in self.outcome_stats:
                self.outcome_stats[characteristic]["count"] += 1
                if outcome == "profitable":
                    self.outcome_stats[characteristic]["success"] += 1
    
    def _get_base_rate(self, characteristics):
        """
        Gets the base success rate for a given market characteristic.
        """
        if not characteristics:
            # Overall success rate
            total_success = sum(stats["success"] for stats in self.outcome_stats.values())
            total_count = sum(stats["count"] for stats in self.outcome_stats.values())
            
            return total_success / total_count if total_count > 0 else 0.5
        
        # Weighted average of all applicable characteristics
        total_success = 0
        total_count = 0
        
        for characteristic in characteristics:
            if characteristic in self.outcome_stats:
                stats = self.outcome_stats[characteristic]
                total_success += stats["success"]
                total_count += stats["count"]
        
        return total_success / total_count if total_count > 0 else 0.5