import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from collections import deque
import logging
import json
import os
import pickle
from datetime import datetime
import math

class RegularizedFeatureFusion:
    """
    Implements feature fusion for integrating main features with cross-asset 
    and order book features with enhanced error handling and dimension management.
    """
    def __init__(self, main_feature_size=19, cross_asset_feature_size=10, 
                 order_book_feature_size=23, output_feature_size=None,
                 regularization='elastic', fusion_method='attention',
                 adaptive_weights=True, model_path="models/feature_fusion.pkl"):
        """
        Initializes the Feature-Fusion.
        
        Args:
            main_feature_size: Main feature dimension
            cross_asset_feature_size: Cross-asset feature dimension
            order_book_feature_size: Order book feature dimension
            output_feature_size: Output feature dimension (None = auto)
            regularization: Regularization method ('l1', 'l2', 'elastic')
            fusion_method: Fusion method ('concat', 'attention', 'weighted', 'autoencoder')
            adaptive_weights: Whether to use adaptive weight learning
            model_path: Path for saving/loading the model
        """
        self.main_feature_size = main_feature_size
        self.cross_asset_feature_size = cross_asset_feature_size
        self.order_book_feature_size = order_book_feature_size
        
        # Output size - if not specified, use main feature size
        self.output_feature_size = output_feature_size or main_feature_size
        
        self.regularization = regularization
        self.fusion_method = fusion_method
        self.adaptive_weights = adaptive_weights
        self.model_path = model_path
        
        # Feature group weights
        self.feature_weights = {
            "main": 1.0,
            "cross_asset": 0.5,
            "order_book": 0.3
        }
        
        # Regularization strength
        self.alpha = 0.01  # L1/L2 strength
        self.l1_ratio = 0.5  # Mixing ratio for Elastic Net
        
        # Performance tracking
        self.performance_history = []
        
        # Feature importance
        self.feature_importance = {
            "main": {},
            "cross_asset": {},
            "order_book": {}
        }
        
        # TensorFlow model for complex fusion methods
        self.model = None
        self.is_model_trained = False
        
        # Scalers for feature groups
        self.scalers = {
            "main": StandardScaler(),
            "cross_asset": StandardScaler(),
            "order_book": StandardScaler()
        }
        
        # PCA models (optional)
        self.pca_models = {
            "cross_asset": None,
            "order_book": None
        }
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('feature_fusion')
        
        # Statistics
        self.stats = {
            "fusion_calls": 0,
            "last_updated": None,
            "feature_correlations": {},
            "training_history": []
        }
        
        # Validation stats
        self.validation_scores = {
            "consistency": [],
            "prediction_power": []
        }
        
        # Initialize model if needed
        if fusion_method in ['attention', 'autoencoder']:
            self._initialize_model()
        
        # Load existing model
        self.load_model()
    
    def fit(self, main_features, cross_asset_features, order_book_features, 
           target_features=None, epochs=50, batch_size=32, validation_split=0.2,
           update_weights=True):
        """
        Trains the feature fusion model with enhanced dimension management
        and robust error handling.
        
        Args:
            main_features: Main features [samples, features]
            cross_asset_features: Cross-asset features [samples, features]
            order_book_features: Order book features [samples, features]
            target_features: Target features (optional)
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            update_weights: Whether to update feature weights
            
        Returns:
            dict: Training results
        """
        # Input validation
        if main_features is None or len(main_features) == 0:
            self.logger.error("Empty main features provided")
            return {"success": False, "error": "empty_main_features"}
        
        # For cross/order features, ensure they exist with proper shapes
        cross_asset_features = self._ensure_features(
            cross_asset_features, main_features.shape[0], self.cross_asset_feature_size, "cross_asset")
        
        order_book_features = self._ensure_features(
            order_book_features, main_features.shape[0], self.order_book_feature_size, "order_book")
        
        # Verify sample counts
        n_samples = len(main_features)
        if len(cross_asset_features) != n_samples or len(order_book_features) != n_samples:
            self.logger.error(f"Sample count mismatch: main={n_samples}, "
                           f"cross={len(cross_asset_features)}, order={len(order_book_features)}")
            
            # Try to fix dimensions
            cross_asset_features = self._adjust_dimension(
                cross_asset_features, n_samples, "cross_asset")
            
            order_book_features = self._adjust_dimension(
                order_book_features, n_samples, "order_book")
        
        # Convert to numpy arrays
        main_features = np.array(main_features)
        cross_asset_features = np.array(cross_asset_features)
        order_book_features = np.array(order_book_features)
        
        # Check for feature dimension mismatches
        if main_features.shape[1] != self.main_feature_size:
            self.logger.warning(f"Main feature size mismatch: {main_features.shape[1]} vs expected {self.main_feature_size}")
            # Update model size
            self.main_feature_size = main_features.shape[1]
            # Reinitialize model if needed
            if self.fusion_method in ['attention', 'autoencoder']:
                self._initialize_model()
        
        # Similar checks for cross/order features (already handled during creation)
        
        # Handle NaN/Inf values
        main_features = self._clean_features(main_features, "main")
        cross_asset_features = self._clean_features(cross_asset_features, "cross_asset")
        order_book_features = self._clean_features(order_book_features, "order_book")
        
        # Scale features
        try:
            main_scaled = self.scalers["main"].fit_transform(main_features)
        except Exception as e:
            # If scaling fails, try a simple reshaping
            self.logger.warning(f"Error scaling main features: {str(e)}. Using standardized data.")
            main_scaled = self._standardize_fallback(main_features)
        
        try:
            cross_scaled = self.scalers["cross_asset"].fit_transform(cross_asset_features)
        except Exception as e:
            self.logger.warning(f"Error scaling cross features: {str(e)}. Using standardized data.")
            cross_scaled = self._standardize_fallback(cross_asset_features)
        
        try:
            order_scaled = self.scalers["order_book"].fit_transform(order_book_features)
        except Exception as e:
            self.logger.warning(f"Error scaling order features: {str(e)}. Using standardized data.")
            order_scaled = self._standardize_fallback(order_book_features)
        
        # Determine target
        if target_features is not None:
            # Make sure targets match samples
            if len(target_features) != n_samples:
                self.logger.warning(f"Target feature count mismatch: {len(target_features)} vs {n_samples}")
                
                # Adjust target dimension
                if len(target_features) > n_samples:
                    y = target_features[:n_samples]
                else:
                    # Pad with repeated values
                    padding = np.zeros((n_samples - len(target_features),) + target_features.shape[1:])
                    for i in range(len(padding)):
                        idx = i % len(target_features)  # Cycle through available targets
                        padding[i] = target_features[idx]
                    y = np.concatenate([target_features, padding])
            else:
                y = target_features
        else:
            # If no target specified, use main features (autoencoder style)
            y = main_scaled
        
        # Training based on fusion method
        if self.fusion_method in ['attention', 'autoencoder']:
            # Complex methods with TensorFlow
            if self.fusion_method == 'attention':
                # Use attention model
                try:
                    history = self.model.fit(
                        [main_scaled, cross_scaled, order_scaled], 
                        y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=0
                    )
                    
                    # Store training history
                    self.stats["training_history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "loss": float(history.history['loss'][-1]),
                        "val_loss": float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
                        "epochs": epochs
                    })
                    
                    # Update weights based on attention levels
                    if update_weights:
                        self._update_attention_weights()
                except Exception as e:
                    self.logger.error(f"Error training attention model: {str(e)}")
                    return {"success": False, "error": f"attention_training_failed: {str(e)}"}
                
            elif self.fusion_method == 'autoencoder':
                # Use autoencoder model with reconstruction loss
                try:
                    history = self.autoencoder.fit(
                        [main_scaled, cross_scaled, order_scaled], 
                        [main_scaled, cross_scaled, order_scaled],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=0
                    )
                    
                    # Store training history
                    self.stats["training_history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "loss": float(history.history['loss'][-1]),
                        "val_loss": float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
                        "epochs": epochs
                    })
                except Exception as e:
                    self.logger.error(f"Error training autoencoder model: {str(e)}")
                    return {"success": False, "error": f"autoencoder_training_failed: {str(e)}"}
        
        else:
            # Simpler methods
            if self.fusion_method == 'weighted':
                try:
                    # Weighted fusion with regularized linear regression
                    
                    # Combine features with current weights
                    weighted_features = self._weighted_combine(main_scaled, cross_scaled, order_scaled)
                    
                    # Train regularized model
                    if self.regularization == 'l1':
                        model = Lasso(alpha=self.alpha)
                    elif self.regularization == 'l2':
                        model = Ridge(alpha=self.alpha)
                    else:  # 'elastic'
                        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
                    
                    model.fit(weighted_features, y)
                    
                    # Store coefficients as feature importance
                    if hasattr(model, 'coef_'):
                        coef = model.coef_
                        if coef.ndim > 1:  # For multivariate output
                            coef = np.mean(np.abs(coef), axis=0)
                        
                        # Normalize coefficients
                        if np.sum(np.abs(coef)) > 0:
                            coef = np.abs(coef) / np.sum(np.abs(coef))
                        
                        # Distribute coefficients to feature groups
                        main_end = self.main_feature_size
                        cross_end = main_end + self.cross_asset_feature_size
                        
                        main_importance = coef[:main_end]
                        cross_importance = coef[main_end:cross_end] if cross_end <= len(coef) else []
                        order_importance = coef[cross_end:] if cross_end < len(coef) else []
                        
                        # Update feature importances
                        for i, imp in enumerate(main_importance):
                            self.feature_importance["main"][i] = float(imp)
                        
                        for i, imp in enumerate(cross_importance):
                            self.feature_importance["cross_asset"][i] = float(imp)
                        
                        for i, imp in enumerate(order_importance):
                            self.feature_importance["order_book"][i] = float(imp)
                        
                        # Update group weights based on average importance
                        if update_weights:
                            self.feature_weights["main"] = float(np.mean(main_importance)) if len(main_importance) > 0 else 1.0
                            self.feature_weights["cross_asset"] = float(np.mean(cross_importance)) if len(cross_importance) > 0 else 0.5
                            self.feature_weights["order_book"] = float(np.mean(order_importance)) if len(order_importance) > 0 else 0.3
                            
                            # Normalize weights
                            total_weight = sum(self.feature_weights.values())
                            if total_weight > 0:
                                for key in self.feature_weights:
                                    self.feature_weights[key] /= total_weight
                    
                    self.stats["training_history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "method": self.regularization,
                        "main_weight": self.feature_weights["main"],
                        "cross_asset_weight": self.feature_weights["cross_asset"],
                        "order_book_weight": self.feature_weights["order_book"]
                    })
                except Exception as e:
                    self.logger.error(f"Error training weighted model: {str(e)}")
                    return {"success": False, "error": f"weighted_training_failed: {str(e)}"}
            
            elif self.fusion_method == 'concat':
                try:
                    # Simple concatenation with regularized linear regression
                    
                    # Concatenate features
                    concatenated = np.concatenate([main_scaled, cross_scaled, order_scaled], axis=1)
                    
                    # Train regularized model
                    if self.regularization == 'l1':
                        model = Lasso(alpha=self.alpha)
                    elif self.regularization == 'l2':
                        model = Ridge(alpha=self.alpha)
                    else:  # 'elastic'
                        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
                    
                    model.fit(concatenated, y)
                    
                    # Similar feature importance calculation
                    # (Similar code to the 'weighted' method above)
                except Exception as e:
                    self.logger.error(f"Error training concat model: {str(e)}")
                    return {"success": False, "error": f"concat_training_failed: {str(e)}"}
        
        # Model is now trained
        self.is_model_trained = True
        self.stats["last_updated"] = datetime.now().isoformat()
        
        # Validation checks
        try:
            # Validation dataset
            val_size = int(n_samples * validation_split)
            X_val = main_scaled[:val_size]
            y_val = y[:val_size]
            
            # Check consistency and prediction power
            fused_val = self.fuse_features(
                main_scaled[:val_size],
                cross_scaled[:val_size],
                order_scaled[:val_size]
            )
            
            # Consistency: correlation between fused features and main features
            consistency = 0.0
            try:
                correlations = []
                for i in range(min(fused_val.shape[1], X_val.shape[1])):
                    r = np.corrcoef(fused_val[:, i], X_val[:, i])[0, 1]
                    if not np.isnan(r):
                        correlations.append(r)
                
                if correlations:
                    consistency = np.mean(correlations)
            except Exception as inner_e:
                self.logger.warning(f"Error calculating consistency: {str(inner_e)}")
            
            # Store validation results
            self.validation_scores["consistency"].append(consistency)
            self.validation_scores["prediction_power"].append(1.0)  # Dummy value
        except Exception as e:
            self.logger.warning(f"Error during validation: {str(e)}")
        
        # Save model
        self.save_model()
        
        return {
            "success": True,
            "message": f"Model trained with {n_samples} samples",
            "consistency": consistency if 'consistency' in locals() else 0.0
        }
    
    def _initialize_model(self):
        """
        Initialisiert das TensorFlow-Modell für komplexe Fusionsmethoden.
        """
        if self.fusion_method == 'attention':
            self._initialize_attention_model()
        elif self.fusion_method == 'autoencoder':
            self._initialize_autoencoder_model()
        else:
            # Für einfachere Methoden ist kein spezielles Modell erforderlich
            pass
    
    def _initialize_attention_model(self):
        """
        Initialisiert ein Attention-basiertes Modell für die Feature-Fusion.
        """
        # Input Layers
        main_input = Input(shape=(self.main_feature_size,), name='main_input')
        cross_asset_input = Input(shape=(self.cross_asset_feature_size,), name='cross_asset_input')
        order_book_input = Input(shape=(self.order_book_feature_size,), name='order_book_input')
        
        # Attention Layers (vereinfachtes Attention-Modell)
        # 1. Dense Layer für jeden Feature-Typ
        main_dense = Dense(32, activation='relu')(main_input)
        cross_dense = Dense(32, activation='relu')(cross_asset_input)
        order_dense = Dense(32, activation='relu')(order_book_input)
        
        # 2. Attention Weights
        main_attention = Dense(1, activation='sigmoid', name='main_attention')(main_dense)
        cross_attention = Dense(1, activation='sigmoid', name='cross_attention')(cross_dense)
        order_attention = Dense(1, activation='sigmoid', name='order_attention')(order_dense)
        
        # 3. Normalisierung der Attention Weights
        attention_concat = Concatenate()([main_attention, cross_attention, order_attention])
        attention_softmax = tf.keras.layers.Softmax()(attention_concat)
        
        # Split normalisierte Gewichte
        main_weight = tf.keras.layers.Lambda(lambda x: x[:, 0:1])(attention_softmax)
        cross_weight = tf.keras.layers.Lambda(lambda x: x[:, 1:2])(attention_softmax)
        order_weight = tf.keras.layers.Lambda(lambda x: x[:, 2:3])(attention_softmax)
        
        # 4. Anwendung der Attention Weights
        main_weighted = tf.keras.layers.Multiply()([main_dense, main_weight])
        cross_weighted = tf.keras.layers.Multiply()([cross_dense, cross_weight])
        order_weighted = tf.keras.layers.Multiply()([order_dense, order_weight])
        
        # 5. Fusion
        fusion = Concatenate()([main_weighted, cross_weighted, order_weighted])
        
        # 6. Output Layer
        if self.regularization == 'l1':
            regularizer = tf.keras.regularizers.l1(self.alpha)
        elif self.regularization == 'l2':
            regularizer = tf.keras.regularizers.l2(self.alpha)
        else:  # elastic net
            regularizer = tf.keras.regularizers.l1_l2(l1=self.alpha * self.l1_ratio,
                                                   l2=self.alpha * (1 - self.l1_ratio))
        
        fusion_dense = Dense(64, activation='relu', 
                           kernel_regularizer=regularizer)(fusion)
        output = Dense(self.output_feature_size, 
                     kernel_regularizer=regularizer)(fusion_dense)
        
        # Create Model
        self.model = Model(inputs=[main_input, cross_asset_input, order_book_input], 
                          outputs=output)
        
        # Compile
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.logger.info(f"Attention-basiertes Fusionsmodell initialisiert")
    
    def _initialize_autoencoder_model(self):
        """
        Initialisiert ein Autoencoder-basiertes Modell für die Feature-Fusion.
        """
        # Input Layers
        main_input = Input(shape=(self.main_feature_size,), name='main_input')
        cross_asset_input = Input(shape=(self.cross_asset_feature_size,), name='cross_asset_input')
        order_book_input = Input(shape=(self.order_book_feature_size,), name='order_book_input')
        
        # Encoding Layers
        if self.regularization == 'l1':
            regularizer = tf.keras.regularizers.l1(self.alpha)
        elif self.regularization == 'l2':
            regularizer = tf.keras.regularizers.l2(self.alpha)
        else:  # elastic net
            regularizer = tf.keras.regularizers.l1_l2(l1=self.alpha * self.l1_ratio,
                                                   l2=self.alpha * (1 - self.l1_ratio))
        
        main_encoded = Dense(16, activation='relu', kernel_regularizer=regularizer)(main_input)
        cross_encoded = Dense(8, activation='relu', kernel_regularizer=regularizer)(cross_asset_input)
        order_encoded = Dense(8, activation='relu', kernel_regularizer=regularizer)(order_book_input)
        
        # Concatenate encoded features
        encoded = Concatenate()([main_encoded, cross_encoded, order_encoded])
        
        # Bottleneck layer (Compressed representation)
        bottleneck = Dense(self.output_feature_size, activation='relu',
                         kernel_regularizer=regularizer,
                         name='bottleneck')(encoded)
        
        # Decoding Layers (optional - für Reconstruction Loss)
        decoded = Dense(32, activation='relu')(bottleneck)
        
        # Separate decoders for each input
        main_decoded = Dense(self.main_feature_size, activation='linear',
                           name='main_output')(decoded)
        cross_decoded = Dense(self.cross_asset_feature_size, activation='linear',
                            name='cross_output')(decoded)
        order_decoded = Dense(self.order_book_feature_size, activation='linear',
                            name='order_output')(decoded)
        
        # Create Models
        # 1. Encoder (für Feature-Fusion)
        self.model = Model(inputs=[main_input, cross_asset_input, order_book_input], 
                          outputs=bottleneck)
        
        # 2. Full Autoencoder (für Training)
        self.autoencoder = Model(inputs=[main_input, cross_asset_input, order_book_input], 
                               outputs=[main_decoded, cross_decoded, order_decoded])
        
        # Compile
        self.autoencoder.compile(optimizer='adam', 
                               loss=['mse', 'mse', 'mse'],
                               loss_weights=[1.0, 0.5, 0.3])
        
        self.logger.info(f"Autoencoder-basiertes Fusionsmodell initialisiert")
        
    def fuse_features(self, main_features, cross_asset_features, order_book_features):
        """
        Performs feature fusion with robust error handling and dimension management.
        
        Args:
            main_features: Main features
            cross_asset_features: Cross-asset features
            order_book_features: Order book features
            
        Returns:
            np.array: Fused features
        """
        self.stats["fusion_calls"] += 1
        
        # Input validation
        if main_features is None or (isinstance(main_features, np.ndarray) and main_features.size == 0):
            self.logger.error("Empty main features provided")
            # Return a dummy feature vector of the right size
            if isinstance(main_features, np.ndarray) and main_features.ndim > 1:
                return np.zeros((main_features.shape[0], self.output_feature_size))
            else:
                return np.zeros(self.output_feature_size)
        
        # Check if single sample or batch
        is_batch = False
        if isinstance(main_features, np.ndarray) and main_features.ndim > 1:
            is_batch = True
            n_samples = main_features.shape[0]
        elif isinstance(main_features, list) and len(main_features) > 0 and hasattr(main_features[0], '__len__'):
            is_batch = True
            n_samples = len(main_features)
        else:
            n_samples = 1
        
        # Convert to numpy arrays
        if not isinstance(main_features, np.ndarray):
            main_features = np.array(main_features)
        
        # For cross/order features, ensure they exist with proper shapes
        cross_asset_features = self._ensure_features(
            cross_asset_features, n_samples, self.cross_asset_feature_size, "cross_asset")
        
        order_book_features = self._ensure_features(
            order_book_features, n_samples, self.order_book_feature_size, "order_book")
        
        # Reshape for single samples
        if not is_batch:
            main_features = main_features.reshape(1, -1)
            cross_asset_features = cross_asset_features.reshape(1, -1)
            order_book_features = order_book_features.reshape(1, -1)
        
        # Check feature dimensions and adjust if needed
        if main_features.shape[1] != self.main_feature_size:
            self.logger.warning(f"Main features dimension mismatch: got {main_features.shape[1]}, expected {self.main_feature_size}")
            main_features = self._adjust_feature_dim(main_features, self.main_feature_size)
        
        # Similar adjustments for cross_asset_features and order_book_features
        # (Already handled by _ensure_features)
        
        # Clean features (handle NaN/Inf)
        main_features = self._clean_features(main_features, "main")
        cross_asset_features = self._clean_features(cross_asset_features, "cross_asset")
        order_book_features = self._clean_features(order_book_features, "order_book")
        
        # Scale features with error handling
        try:
            main_scaled = self.scalers["main"].transform(main_features)
        except Exception as e:
            self.logger.warning(f"Error scaling main features: {str(e)}. Using fallback.")
            # Fallback: refit scaler or use standardization
            try:
                self.scalers["main"].fit(main_features)
                main_scaled = self.scalers["main"].transform(main_features)
            except:
                main_scaled = self._standardize_fallback(main_features)
        
        try:
            cross_scaled = self.scalers["cross_asset"].transform(cross_asset_features)
        except Exception as e:
            self.logger.warning(f"Error scaling cross features: {str(e)}. Using fallback.")
            try:
                self.scalers["cross_asset"].fit(cross_asset_features)
                cross_scaled = self.scalers["cross_asset"].transform(cross_asset_features)
            except:
                cross_scaled = self._standardize_fallback(cross_asset_features)
        
        try:
            order_scaled = self.scalers["order_book"].transform(order_book_features)
        except Exception as e:
            self.logger.warning(f"Error scaling order features: {str(e)}. Using fallback.")
            try:
                self.scalers["order_book"].fit(order_book_features)
                order_scaled = self.scalers["order_book"].transform(order_book_features)
            except:
                order_scaled = self._standardize_fallback(order_book_features)
        
        # Perform fusion based on selected method
        result = None
        try:
            if self.fusion_method in ['attention', 'autoencoder']:
                # Complex methods with TensorFlow model
                if not self.is_model_trained:
                    self.logger.warning("Model not trained, using fallback weighted fusion")
                    result = self._weighted_combine(main_scaled, cross_scaled, order_scaled)
                else:
                    # Use trained model
                    result = self.model.predict([main_scaled, cross_scaled, order_scaled])
            
            elif self.fusion_method == 'weighted':
                # Weighted fusion
                result = self._weighted_combine(main_scaled, cross_scaled, order_scaled)
            
            elif self.fusion_method == 'concat':
                # Simple concatenation
                concat = np.concatenate([main_scaled, cross_scaled, order_scaled], axis=1)
                
                # Optional: Dimension reduction
                if concat.shape[1] > self.output_feature_size:
                    if not hasattr(self, 'concat_pca') or self.concat_pca is None:
                        self.concat_pca = PCA(n_components=self.output_feature_size)
                        self.concat_pca.fit(concat)
                    
                    result = self.concat_pca.transform(concat)
                else:
                    result = concat
            else:
                self.logger.error(f"Unknown fusion method: {self.fusion_method}")
                # Fallback to simple concatenation
                result = np.concatenate([main_scaled, cross_scaled, order_scaled], axis=1)
                
                # Truncate if needed
                if result.shape[1] > self.output_feature_size:
                    result = result[:, :self.output_feature_size]
        except Exception as e:
            self.logger.error(f"Error during feature fusion: {str(e)}")
            # Provide a useful fallback
            result = main_scaled.copy()
            if result.shape[1] != self.output_feature_size:
                result = self._adjust_feature_dim(result, self.output_feature_size)
        
        # If single sample, remove batch dimension
        if not is_batch and hasattr(result, 'shape') and len(result.shape) > 1:
            result = result.flatten()
        
        return result
    
    def _ensure_features(self, features, n_samples, feature_size, feature_type):
        """
        Ensures features are available with the correct shape.
        
        Args:
            features: Input features
            n_samples: Required number of samples
            feature_size: Required feature size
            feature_type: Type of feature (for logging)
            
        Returns:
            np.array: Features with the correct shape
        """
        if features is None:
            self.logger.info(f"No {feature_type} features provided, using zeros")
            return np.zeros((n_samples, feature_size))
        
        # Convert to numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Check if dimensions match
        if features.size == 0:
            self.logger.info(f"Empty {feature_type} features, using zeros")
            return np.zeros((n_samples, feature_size))
        
        # Handle single vector vs. matrix
        if features.ndim == 1:
            # Single vector - repeat for all samples
            if len(features) != feature_size:
                self.logger.warning(f"{feature_type} feature size mismatch: {len(features)} vs {feature_size}")
                features = self._adjust_feature_dim(features.reshape(1, -1), feature_size)[0]
            
            return np.tile(features, (n_samples, 1))
        
        # Matrix case - check both dimensions
        if features.shape[0] != n_samples:
            self.logger.warning(f"{feature_type} sample count mismatch: {features.shape[0]} vs {n_samples}")
            features = self._adjust_dimension(features, n_samples, feature_type)
        
        if features.shape[1] != feature_size:
            self.logger.warning(f"{feature_type} feature size mismatch: {features.shape[1]} vs {feature_size}")
            features = self._adjust_feature_dim(features, feature_size)
        
        return features
    
    def _adjust_dimension(self, features, target_samples, feature_type):
        """
        Adjusts the number of samples in a feature matrix.
        
        Args:
            features: Feature matrix
            target_samples: Target number of samples
            feature_type: Type of feature (for logging)
            
        Returns:
            np.array: Adjusted feature matrix
        """
        if len(features) == target_samples:
            return features
            
        if len(features) > target_samples:
            # Too many samples - truncate
            self.logger.info(f"Truncating {feature_type} features from {len(features)} to {target_samples}")
            return features[:target_samples]
        else:
            # Too few samples - pad with repeated values
            self.logger.info(f"Padding {feature_type} features from {len(features)} to {target_samples}")
            
            # Create padding with cyclic repetition of existing features
            feature_size = features.shape[1]
            padding = np.zeros((target_samples - len(features), feature_size))
            
            for i in range(len(padding)):
                idx = i % len(features)  # Cycle through existing features
                padding[i] = features[idx]
            
            return np.vstack([features, padding])
    
    def _adjust_feature_dim(self, features, target_size):
        """
        Adjusts the feature dimension of a matrix.
        
        Args:
            features: Feature matrix
            target_size: Target feature dimension
            
        Returns:
            np.array: Adjusted feature matrix
        """
        current_size = features.shape[1]
        if current_size == target_size:
            return features
        
        if current_size > target_size:
            # Too many features - truncate
            return features[:, :target_size]
        else:
            # Too few features - pad with zeros
            padding = np.zeros((features.shape[0], target_size - current_size))
            return np.hstack([features, padding])
    
    def _clean_features(self, features, feature_type):
        """
        Cleans features by handling NaN/Inf values.
        
        Args:
            features: Feature matrix
            feature_type: Type of feature (for logging)
            
        Returns:
            np.array: Cleaned feature matrix
        """
        # Check for NaN/Inf
        has_nan = np.isnan(features).any()
        has_inf = np.isinf(features).any()
        
        if has_nan or has_inf:
            self.logger.warning(f"Found NaN/Inf in {feature_type} features. Cleaning...")
            
            # Replace with column median
            col_medians = np.nanmedian(features, axis=0)
            
            # Fill any lingering NaN in medians with 0
            col_medians = np.nan_to_num(col_medians)
            
            # Copy to avoid modifying original
            features_clean = features.copy()
            
            # Replace NaN/Inf with medians
            for i in range(features.shape[1]):
                mask = np.logical_or(np.isnan(features[:, i]), np.isinf(features[:, i]))
                features_clean[mask, i] = col_medians[i]
            
            return features_clean
        
        return features

    def _standardize_fallback(self, features):
        """
        Fallback standardization when StandardScaler fails.
        
        Args:
            features: Feature matrix
            
        Returns:
            np.array: Standardized features
        """
        # Simple standardization: (x - mean) / std
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        
        # Avoid division by zero
        stds[stds == 0] = 1.0
        
        return (features - means) / stds
    
    def _weighted_combine(self, main_features, cross_features, order_features):
        """
        Combines features with group weights.
        
        Args:
            main_features: Main features
            cross_features: Cross-asset features
            order_features: Order book features
            
        Returns:
            np.array: Weighted combined features
        """
        # Determine target dimension
        if self.output_feature_size == self.main_feature_size:
            # Simplest case: keep main feature dimensionality
            result = main_features.copy()
            
            # Project cross-asset features
            if cross_features.shape[1] > 0:
                cross_projection = self._project_features(
                    cross_features, 
                    self.cross_asset_feature_size, 
                    self.main_feature_size
                )
                result += cross_projection * self.feature_weights["cross_asset"]
            
            # Project order book features
            if order_features.shape[1] > 0:
                order_projection = self._project_features(
                    order_features, 
                    self.order_book_feature_size, 
                    self.main_feature_size
                )
                result += order_projection * self.feature_weights["order_book"]
            
            # Apply main weight
            result *= self.feature_weights["main"]
            
        else:
            # More complex case: project all to target dimension
            main_projection = self._project_features(
                main_features, 
                self.main_feature_size, 
                self.output_feature_size
            )
            
            cross_projection = self._project_features(
                cross_features, 
                self.cross_asset_feature_size, 
                self.output_feature_size
            )
            
            order_projection = self._project_features(
                order_features, 
                self.order_book_feature_size, 
                self.output_feature_size
            )
            
            # Weighted combination
            result = (main_projection * self.feature_weights["main"] +
                     cross_projection * self.feature_weights["cross_asset"] +
                     order_projection * self.feature_weights["order_book"])
        
        return result
    
    def _project_features(self, features, input_dim, output_dim):
        """
        Projects features from one dimension to another.
        
        Args:
            features: Input features
            input_dim: Input dimension
            output_dim: Output dimension
            
        Returns:
            np.array: Projected features
        """
        if input_dim == output_dim:
            return features
        
        # Handle empty features
        if features.size == 0:
            return np.zeros((features.shape[0], output_dim))
        
        if input_dim < output_dim:
            # Upscaling: copy features multiple times or interpolate
            scale_factor = output_dim / input_dim
            
            if scale_factor.is_integer():
                # Simple repetition
                return np.repeat(features, int(scale_factor), axis=1)
            else:
                # Linear interpolation
                projected = np.zeros((features.shape[0], output_dim))
                
                for i in range(output_dim):
                    src_idx = int(i / scale_factor)
                    src_idx = min(src_idx, input_dim - 1)
                    projected[:, i] = features[:, src_idx]
                
                return projected
        
        else:  # input_dim > output_dim
            # Downscaling: use PCA or average pooling
            
            # Average pooling for smaller dimensions
            if input_dim <= 10 and output_dim <= 5:
                pool_size = input_dim // output_dim
                remainder = input_dim % output_dim
                
                projected = np.zeros((features.shape[0], output_dim))
                
                src_idx = 0
                for i in range(output_dim):
                    # Determine pool size for this output index
                    curr_pool_size = pool_size + (1 if i < remainder else 0)
                    projected[:, i] = np.mean(features[:, src_idx:src_idx+curr_pool_size], axis=1)
                    src_idx += curr_pool_size
                
                return projected
            
            # PCA for larger dimensions
            else:
                feature_type = None
                if input_dim == self.cross_asset_feature_size:
                    feature_type = "cross_asset"
                elif input_dim == self.order_book_feature_size:
                    feature_type = "order_book"
                
                if feature_type and feature_type in self.pca_models and self.pca_models[feature_type] is not None:
                    # Use existing PCA model
                    try:
                        return self.pca_models[feature_type].transform(features)
                    except Exception as e:
                        self.logger.warning(f"Error using PCA model: {str(e)}. Recreating model.")
                        # Fall through to create new model
                
                # Create new PCA model
                try:
                    pca_model = PCA(n_components=output_dim)
                    projected = pca_model.fit_transform(features)
                    
                    # Store PCA model for later use
                    if feature_type:
                        self.pca_models[feature_type] = pca_model
                    
                    return projected
                except Exception as e:
                    self.logger.warning(f"Error creating PCA model: {str(e)}. Using simple averaging.")
                    
                    # Fallback to simple averaging
                    factor = input_dim // output_dim
                    projected = np.zeros((features.shape[0], output_dim))
                    
                    for i in range(output_dim):
                        start_idx = i * factor
                        end_idx = min(start_idx + factor, input_dim)
                        projected[:, i] = np.mean(features[:, start_idx:end_idx], axis=1)
                    
                    return projected
    
    def _update_attention_weights(self):
        """
        Aktualisiert die Feature-Gruppengewichte basierend auf den Attention-Layern.
        """
        if self.model is None or not self.is_model_trained:
            return
        
        # Erstelle Dummy-Input für die Aufmerksamkeitsvorhersage
        dummy_main = np.zeros((1, self.main_feature_size))
        dummy_cross = np.zeros((1, self.cross_asset_feature_size))
        dummy_order = np.zeros((1, self.order_book_feature_size))
        
        # Verwende einen Attention-Layer, um Gewichte vorherzusagen
        attention_layer_main = self.model.get_layer('main_attention')
        attention_layer_cross = self.model.get_layer('cross_attention')
        attention_layer_order = self.model.get_layer('order_attention')
        
        # Extrahiere Attention-Gewichte (vereinfachte Version)
        dummy_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[attention_layer_main.output, 
                   attention_layer_cross.output, 
                   attention_layer_order.output]
        )
        
        attention_weights = dummy_model.predict([dummy_main, dummy_cross, dummy_order])
        
        # Normalisiere Gewichte
        weights_sum = attention_weights[0] + attention_weights[1] + attention_weights[2]
        
        self.feature_weights["main"] = float(attention_weights[0] / weights_sum)
        self.feature_weights["cross_asset"] = float(attention_weights[1] / weights_sum)
        self.feature_weights["order_book"] = float(attention_weights[2] / weights_sum)
        
        self.logger.info(f"Updated feature weights from attention: "
                       f"main={self.feature_weights['main']:.3f}, "
                       f"cross={self.feature_weights['cross_asset']:.3f}, "
                       f"ob={self.feature_weights['order_book']:.3f}")
    
    def analyze_feature_correlations(self, main_features, cross_features, order_features, lookahead=None):
        """
        Analysiert die Korrelationen zwischen verschiedenen Feature-Gruppen und zukünftigen Preisänderungen.
        
        Args:
            main_features: Hauptsymbol-Features
            cross_features: Cross-Asset-Features
            order_features: Order Book Features
            lookahead: Zukünftige Preisänderungen (falls verfügbar)
            
        Returns:
            dict: Korrelationsanalyse
        """
        # Konvertiere zu NumPy-Arrays falls nötig
        main_features = np.array(main_features)
        cross_features = np.array(cross_features)
        order_features = np.array(order_features)
        
        if lookahead is not None:
            lookahead = np.array(lookahead)
        
        # Nur eine Teilmenge der Daten verwenden, falls zu viele Datenpunkte
        max_samples = 1000
        if len(main_features) > max_samples:
            indices = np.random.choice(len(main_features), max_samples, replace=False)
            main_features = main_features[indices]
            cross_features = cross_features[indices]
            order_features = order_features[indices]
            if lookahead is not None:
                lookahead = lookahead[indices]
        
        # Skaliere Features
        main_scaled = self.scalers["main"].transform(main_features)
        cross_scaled = self.scalers["cross_asset"].transform(cross_features)
        order_scaled = self.scalers["order_book"].transform(order_features)
        
        # Berechne Korrelationen zwischen Feature-Gruppen
        correlations = {}
        
        # 1. Korrelation zwischen Hauptfeatures und Cross-Asset-Features
        main_cross_corr = self._calculate_group_correlation(main_scaled, cross_scaled)
        correlations["main_cross"] = float(main_cross_corr)
        
        # 2. Korrelation zwischen Hauptfeatures und Order-Book-Features
        main_order_corr = self._calculate_group_correlation(main_scaled, order_scaled)
        correlations["main_order"] = float(main_order_corr)
        
        # 3. Korrelation zwischen Cross-Asset und Order-Book-Features
        cross_order_corr = self._calculate_group_correlation(cross_scaled, order_scaled)
        correlations["cross_order"] = float(cross_order_corr)
        
        # 4. Korrelation mit Lookahead (falls vorhanden)
        if lookahead is not None:
            # Korrelation zwischen jeder Feature-Gruppe und Lookahead
            main_future_corr = self._calculate_future_correlation(main_scaled, lookahead)
            cross_future_corr = self._calculate_future_correlation(cross_scaled, lookahead)
            order_future_corr = self._calculate_future_correlation(order_scaled, lookahead)
            
            correlations["main_future"] = float(main_future_corr)
            correlations["cross_future"] = float(cross_future_corr)
            correlations["order_future"] = float(order_future_corr)
            
            # Finde die am besten korrelierte Feature-Gruppe
            best_group = max([
                ("main", main_future_corr),
                ("cross_asset", cross_future_corr),
                ("order_book", order_future_corr)
            ], key=lambda x: abs(x[1]))
            
            correlations["best_group"] = best_group[0]
            correlations["best_correlation"] = float(best_group[1])
            
            # Aktualisiere Feature-Gewichte basierend auf Prädiktionskraft
            if self.adaptive_weights:
                # Normalisiere absolute Korrelationen zu Gewichten
                weights = {
                    "main": abs(main_future_corr),
                    "cross_asset": abs(cross_future_corr),
                    "order_book": abs(order_future_corr)
                }
                
                # Stelle sicher, dass Gewichte positiv sind und Mindestwerte haben
                min_weight = 0.1
                for key in weights:
                    weights[key] = max(weights[key], min_weight)
                
                # Normalisiere zu Summe 1
                weight_sum = sum(weights.values())
                for key in weights:
                    weights[key] /= weight_sum
                
                # Exponentiell gewichtetes gleitendes Mittel für stabile Gewichte
                alpha = 0.2  # Gewichtungsfaktor für neue Beobachtungen
                for key in self.feature_weights:
                    self.feature_weights[key] = (1 - alpha) * self.feature_weights[key] + alpha * weights[key]
                
                # Erneut normalisieren
                weight_sum = sum(self.feature_weights.values())
                for key in self.feature_weights:
                    self.feature_weights[key] /= weight_sum
                
                correlations["updated_weights"] = {k: float(v) for k, v in self.feature_weights.items()}
        
        # Speichere Korrelationsstatistiken
        self.stats["feature_correlations"] = correlations
        
        return correlations
    
    def _calculate_group_correlation(self, features1, features2):
        """
        Berechnet die Korrelation zwischen zwei Feature-Gruppen.
        
        Args:
            features1: Erste Feature-Gruppe
            features2: Zweite Feature-Gruppe
            
        Returns:
            float: Durchschnittliche Korrelation
        """
        if features1.shape[1] == 0 or features2.shape[1] == 0:
            return 0
        
        # PCA zur Dimensionsreduktion falls nötig
        max_dim = 10  # Maximum Dimension für Korrelationsberechnung
        
        if features1.shape[1] > max_dim:
            pca1 = PCA(n_components=max_dim)
            features1_reduced = pca1.fit_transform(features1)
        else:
            features1_reduced = features1
        
        if features2.shape[1] > max_dim:
            pca2 = PCA(n_components=max_dim)
            features2_reduced = pca2.fit_transform(features2)
        else:
            features2_reduced = features2
        
        # Berechne paarweise Korrelationen
        corr_values = []
        
        for i in range(features1_reduced.shape[1]):
            for j in range(features2_reduced.shape[1]):
                corr = np.corrcoef(features1_reduced[:, i], features2_reduced[:, j])[0, 1]
                if not np.isnan(corr):
                    corr_values.append(abs(corr))  # Verwende Absolutwert der Korrelation
        
        if corr_values:
            return np.mean(corr_values)
        else:
            return 0
    
    def _calculate_future_correlation(self, features, lookahead):
        """
        Berechnet die Korrelation zwischen Features und zukünftigen Preisänderungen.
        
        Args:
            features: Feature-Matrix
            lookahead: Zukünftige Preisänderungen
            
        Returns:
            float: Durchschnittliche Korrelation
        """
        if features.shape[1] == 0 or lookahead.ndim == 0:
            return 0
        
        # Flache Lookahead, falls mehrdimensional
        if lookahead.ndim > 1 and lookahead.shape[1] > 1:
            lookahead = np.mean(lookahead, axis=1)
        
        # Berechne Korrelation für jede Feature-Dimension
        corr_values = []
        
        for i in range(features.shape[1]):
            corr = np.corrcoef(features[:, i], lookahead)[0, 1]
            if not np.isnan(corr):
                corr_values.append(corr)
        
        if corr_values:
            # Rückgabe des Durchschnitts der besten Korrelationen
            top_k = min(5, len(corr_values))
            top_corrs = sorted(corr_values, key=abs, reverse=True)[:top_k]
            return np.mean(top_corrs)
        else:
            return 0
    
    def save_model(self, path=None):
        """
        Speichert das Feature-Fusionsmodell und zugehörige Komponenten.
        
        Args:
            path: Optionaler alternativer Speicherpfad
            
        Returns:
            bool: Erfolg des Speichervorgangs
        """
        if path is None:
            path = self.model_path
        
        try:
            # Erstelle Verzeichnis falls nicht vorhanden
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Speichere TensorFlow-Modell separat, wenn vorhanden
            if self.model is not None and self.is_model_trained:
                model_path = os.path.splitext(path)[0] + "_tf_model"
                weights_path = os.path.splitext(path)[0] + "_weights.h5"
                self.model.save_weights(weights_path)
            
            # Bereite Modelldaten vor
            model_data = {
                "feature_sizes": {
                    "main": self.main_feature_size,
                    "cross_asset": self.cross_asset_feature_size,
                    "order_book": self.order_book_feature_size,
                    "output": self.output_feature_size
                },
                "feature_weights": self.feature_weights,
                "feature_importance": self.feature_importance,
                "fusion_method": self.fusion_method,
                "regularization": self.regularization,
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "scalers": self.scalers,
                "pca_models": self.pca_models,
                "stats": self.stats,
                "validation_scores": self.validation_scores,
                "is_model_trained": self.is_model_trained,
                "timestamp": datetime.now().isoformat()
            }
            
            # Speichere Modelldaten mit Pickle
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Feature fusion model saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving feature fusion model: {str(e)}")
            return False
    
    def load_model(self, path=None):
        """
        Lädt ein gespeichertes Feature-Fusionsmodell.
        
        Args:
            path: Optionaler alternativer Ladepfad
            
        Returns:
            bool: Erfolg des Ladevorgangs
        """
        if path is None:
            path = self.model_path
        
        if not os.path.exists(path):
            self.logger.info(f"No existing model found at {path}")
            return False
        
        try:
            # Lade Modelldaten mit Pickle
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Lade Feature-Größen
            if "feature_sizes" in model_data:
                self.main_feature_size = model_data["feature_sizes"]["main"]
                self.cross_asset_feature_size = model_data["feature_sizes"]["cross_asset"]
                self.order_book_feature_size = model_data["feature_sizes"]["order_book"]
                self.output_feature_size = model_data["feature_sizes"]["output"]
            
            # Lade Feature-Gewichte und -Importanzen
            if "feature_weights" in model_data:
                self.feature_weights = model_data["feature_weights"]
            
            if "feature_importance" in model_data:
                self.feature_importance = model_data["feature_importance"]
            
            # Lade Konfiguration
            if "fusion_method" in model_data:
                self.fusion_method = model_data["fusion_method"]
            
            if "regularization" in model_data:
                self.regularization = model_data["regularization"]
            
            if "alpha" in model_data:
                self.alpha = model_data["alpha"]
            
            if "l1_ratio" in model_data:
                self.l1_ratio = model_data["l1_ratio"]
            
            # Lade Scaler und PCA-Modelle
            if "scalers" in model_data:
                self.scalers = model_data["scalers"]
            
            if "pca_models" in model_data:
                self.pca_models = model_data["pca_models"]
            
            # Lade Statistiken
            if "stats" in model_data:
                self.stats = model_data["stats"]
            
            if "validation_scores" in model_data:
                self.validation_scores = model_data["validation_scores"]
            
            if "is_model_trained" in model_data:
                self.is_model_trained = model_data["is_model_trained"]
            
            # Initialisiere TensorFlow-Modell
            if self.fusion_method in ['attention', 'autoencoder']:
                self._initialize_model()
                
                # Lade Modellgewichte, falls vorhanden
                weights_path = os.path.splitext(path)[0] + "_weights.h5"
                if os.path.exists(weights_path):
                    try:
                        self.model.load_weights(weights_path)
                    except Exception as e:
                        self.logger.warning(f"Could not load model weights: {str(e)}")
            
            self.logger.info(f"Feature fusion model loaded from {path}")
            
            # Log Modellinformationen
            if "timestamp" in model_data:
                self.logger.info(f"Model trained on: {model_data['timestamp']}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading feature fusion model: {str(e)}")
            return False
    
    def get_feature_importance(self):
        """
        Gibt die Feature-Importanzen für alle Gruppen zurück.
        
        Returns:
            dict: Feature-Importanzen
        """
        result = {
            "group_weights": {k: float(v) for k, v in self.feature_weights.items()},
            "feature_importance": self.feature_importance,
            "validation": {
                "consistency": float(np.mean(self.validation_scores["consistency"])) if self.validation_scores["consistency"] else 0,
                "prediction_power": float(np.mean(self.validation_scores["prediction_power"])) if self.validation_scores["prediction_power"] else 0
            }
        }
        
        return result
    
    def update_weights_from_performance(self, performance_metric, component="main"):
        """
        Aktualisiert die Feature-Gewichte basierend auf Leistungsmetriken.
        
        Args:
            performance_metric: Leistungsmetrik (höher = besser)
            component: Feature-Komponente, die aktualisiert werden soll
            
        Returns:
            dict: Aktualisierte Gewichte
        """
        if component not in self.feature_weights:
            self.logger.warning(f"Unknown component: {component}")
            return self.feature_weights
        
        # Speichere Performancemetrik
        self.performance_history.append({
            "component": component,
            "metric": float(performance_metric),
            "timestamp": datetime.now().isoformat()
        })
        
        # Behalte nur die letzten 100 Einträge
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Berechne durchschnittliche Performance pro Komponente
        component_metrics = {}
        
        for entry in self.performance_history:
            comp = entry["component"]
            metric = entry["metric"]
            
            if comp not in component_metrics:
                component_metrics[comp] = []
            
            component_metrics[comp].append(metric)
        
        avg_metrics = {}
        for comp, metrics in component_metrics.items():
            avg_metrics[comp] = np.mean(metrics)
        
        # Aktualisiere Gewichte nur, wenn genügend Daten für alle Komponenten vorhanden sind
        if len(avg_metrics) >= len(self.feature_weights):
            # Normalisiere zu Summe 1
            metric_sum = sum(avg_metrics.values())
            
            if metric_sum > 0:
                for comp in self.feature_weights:
                    if comp in avg_metrics:
                        # Exponentielles gleitendes Mittel
                        alpha = 0.1  # Gewichtungsfaktor
                        self.feature_weights[comp] = (1 - alpha) * self.feature_weights[comp] + alpha * (avg_metrics[comp] / metric_sum)
                
                # Normalisiere erneut
                weight_sum = sum(self.feature_weights.values())
                for comp in self.feature_weights:
                    self.feature_weights[comp] /= weight_sum
        
        return {k: float(v) for k, v in self.feature_weights.items()}


class EnhancedFeatureFusionEnsemble:
    """
    Implementiert ein Ensemble mehrerer Feature-Fusion-Modelle mit verschiedenen
    Regularisierungs- und Fusionsmethoden für robustere Ergebnisse.
    """
    def __init__(self, num_models=3, base_config=None, model_path="models/fusion_ensemble"):
        """
        Initialisiert das Feature-Fusion-Ensemble.
        
        Args:
            num_models: Anzahl der Modelle im Ensemble
            base_config: Basiskonfiguration für alle Modelle
            model_path: Pfad zum Speichern/Laden des Ensembles
        """
        self.num_models = num_models
        self.model_path = model_path
        
        # Standardkonfiguration, falls keine angegeben
        self.base_config = base_config or {
            "main_feature_size": 19,
            "cross_asset_feature_size": 10,
            "order_book_feature_size": 23,
            "output_feature_size": 19
        }
        
        # Ensemble von Modellen mit verschiedenen Konfigurationen
        self.models = []
        
        # Gewichte für jedes Modell
        self.model_weights = np.ones(num_models) / num_models
        
        # Performance-Tracking für Modellgewichte
        self.performance_history = []
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('fusion_ensemble')
        
        # Initialisiere Modelle
        self._initialize_models()
        
        # Lade gespeicherte Modelle, falls vorhanden
        self.load_ensemble()
    
    def _initialize_models(self):
        """
        Initialisiert verschiedene Feature-Fusion-Modelle für das Ensemble.
        """
        # Verschiedene Konfigurationen für Vielfalt im Ensemble
        configurations = [
            # L1-regularisierte gewichtete Fusion
            {
                "regularization": "l1",
                "fusion_method": "weighted",
                "alpha": 0.01,
                "l1_ratio": 1.0
            },
            # L2-regularisierte gewichtete Fusion
            {
                "regularization": "l2",
                "fusion_method": "weighted",
                "alpha": 0.01,
                "l1_ratio": 0.0
            },
            # Elastic Net mit Aufmerksamkeitsmechanismus
            {
                "regularization": "elastic",
                "fusion_method": "attention",
                "alpha": 0.01,
                "l1_ratio": 0.5
            },
            # Autoencoder-basierte Fusion
            {
                "regularization": "l2",
                "fusion_method": "autoencoder",
                "alpha": 0.01,
                "l1_ratio": 0.0
            },
            # Einfache Konkatenation mit L1-Regularisierung
            {
                "regularization": "l1",
                "fusion_method": "concat",
                "alpha": 0.01,
                "l1_ratio": 1.0
            }
        ]
        
        # Stelle sicher, dass wir nicht mehr Konfigurationen als Modelle haben
        configurations = configurations[:self.num_models]
        
        # Füge weitere Konfigurationen hinzu, falls nötig
        while len(configurations) < self.num_models:
            # Zufällige Auswahl aus vorhandenen Konfigurationen
            idx = np.random.randint(0, len(configurations))
            config = configurations[idx].copy()
            
            # Leichte Variation der Parameter
            if config["regularization"] == "elastic":
                config["l1_ratio"] = np.random.uniform(0.2, 0.8)
            
            config["alpha"] = config["alpha"] * np.random.uniform(0.5, 2.0)
            
            configurations.append(config)
        
        # Erstelle Modelle
        for i, config in enumerate(configurations):
            # Kombiniere Basiskonfiguration mit spezifischer Konfiguration
            full_config = {**self.base_config, **config}
            
            # Erstelle eindeutigen Modellpfad
            model_path = f"{self.model_path}_model_{i}.pkl"
            full_config["model_path"] = model_path
            
            # Erstelle und speichere Modell
            model = RegularizedFeatureFusion(**full_config)
            self.models.append(model)
            
            self.logger.info(f"Model {i} initialized: {config['fusion_method']} with {config['regularization']} regularization")
    
    def fit(self, main_features, cross_asset_features, order_book_features, 
           target_features=None, epochs=50, batch_size=32, validation_split=0.2):
        """
        Trainiert alle Modelle im Ensemble.
        
        Args:
            main_features: Hauptsymbol-Features
            cross_asset_features: Cross-Asset-Features
            order_book_features: Order Book Features
            target_features: Ziel-Features für das Training (optional)
            epochs: Trainings-Epochen
            batch_size: Batch-Größe
            validation_split: Anteil der Validierungsdaten
            
        Returns:
            dict: Trainingsergebnisse
        """
        results = []
        
        for i, model in enumerate(self.models):
            self.logger.info(f"Training model {i+1}/{len(self.models)}...")
            
            result = model.fit(
                main_features, 
                cross_asset_features, 
                order_book_features,
                target_features=target_features,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split
            )
            
            results.append(result)
            
            if result.get("success", False):
                self.logger.info(f"Model {i} trained successfully")
            else:
                self.logger.warning(f"Model {i} training failed: {result.get('error', 'unknown error')}")
        
        # Speichere das trainierte Ensemble
        self.save_ensemble()
        
        return {
            "success": any(r.get("success", False) for r in results),
            "model_results": results
        }
    
    def fuse_features(self, main_features, cross_asset_features, order_book_features):
        """
        Führt Feature-Fusion mit allen Modellen durch und kombiniert die Ergebnisse.
        
        Args:
            main_features: Hauptsymbol-Features
            cross_asset_features: Cross-Asset-Features
            order_book_features: Order Book Features
            
        Returns:
            np.array: Fusionierte Features
        """
        # Prüfe, ob zumindest ein Modell trainiert wurde
        if not any(model.is_model_trained for model in self.models):
            self.logger.warning("No models trained in the ensemble, using first model as fallback")
            return self.models[0].fuse_features(main_features, cross_asset_features, order_book_features)
        
        # Sammle Ergebnisse von allen trainierten Modellen
        fused_features = []
        weights = []
        
        for i, model in enumerate(self.models):
            if model.is_model_trained:
                try:
                    result = model.fuse_features(main_features, cross_asset_features, order_book_features)
                    fused_features.append(result)
                    weights.append(self.model_weights[i])
                except Exception as e:
                    self.logger.warning(f"Model {i} fusion failed: {str(e)}")
        
        if not fused_features:
            self.logger.error("All model fusions failed, using first model as fallback")
            return self.models[0].fuse_features(main_features, cross_asset_features, order_book_features)
        
        # Normalisiere Gewichte
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Gewichtete Kombination der Ergebnisse
        # Zuerst sicherstellen, dass alle Ergebnisse die gleiche Dimension haben
        result_shape = fused_features[0].shape
        aligned_features = []
        
        for features in fused_features:
            if features.shape != result_shape:
                # Einfache Anpassung durch Mitteln oder Truncation/Padding
                if len(result_shape) == 1:  # Einzelner Vektor
                    if len(features) > result_shape[0]:
                        # Truncate
                        aligned_features.append(features[:result_shape[0]])
                    elif len(features) < result_shape[0]:
                        # Pad mit Nullen
                        padded = np.zeros(result_shape)
                        padded[:len(features)] = features
                        aligned_features.append(padded)
                    else:
                        aligned_features.append(features)
                else:  # Batch von Vektoren
                    if features.shape[1] > result_shape[1]:
                        # Truncate
                        aligned_features.append(features[:, :result_shape[1]])
                    elif features.shape[1] < result_shape[1]:
                        # Pad mit Nullen
                        padded = np.zeros(result_shape)
                        padded[:, :features.shape[1]] = features
                        aligned_features.append(padded)
                    else:
                        aligned_features.append(features)
            else:
                aligned_features.append(features)
        
        # Gewichtete Kombination
        combined = np.zeros(result_shape)
        for i, features in enumerate(aligned_features):
            combined += features * weights[i]
        
        return combined
    
    def update_weights_from_performance(self, performance_metrics):
        """
        Aktualisiert die Modellgewichte basierend auf Leistungsmetriken.
        
        Args:
            performance_metrics: Liste von Leistungsmetriken für jedes Modell
            
        Returns:
            np.array: Aktualisierte Modellgewichte
        """
        if len(performance_metrics) != len(self.models):
            self.logger.warning(f"Performance metrics count mismatch: {len(performance_metrics)} vs {len(self.models)}")
            return self.model_weights
        
        # Speichere Performancemetriken
        self.performance_history.append({
            "metrics": performance_metrics,
            "timestamp": datetime.now().isoformat()
        })
        
        # Behalte nur die letzten 50 Einträge
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        
        # Berechne durchschnittliche Performance über die Historie
        if len(self.performance_history) < 5:
            # Zu wenige Daten für zuverlässige Aktualisierung
            return self.model_weights
        
        avg_metrics = np.zeros(len(self.models))
        count = 0
        
        for entry in self.performance_history[-10:]:  # Verwende nur die letzten 10 Einträge
            metrics = entry["metrics"]
            avg_metrics += np.array(metrics)
            count += 1
        
        if count > 0:
            avg_metrics /= count
        
        # Konvertiere zu Gewichten (höhere Metrik = höheres Gewicht)
        min_metric = np.min(avg_metrics)
        if min_metric < 0:
            # Verschiebe alle Metriken ins Positive
            avg_metrics = avg_metrics - min_metric + 1e-6
        
        # Stelle sicher, dass alle Gewichte positiv sind
        avg_metrics = np.maximum(avg_metrics, 1e-6)
        
        # Normalisiere zu Summe 1
        new_weights = avg_metrics / np.sum(avg_metrics)
        
        # Exponentielles gleitendes Mittel für stabilere Gewichte
        alpha = 0.2  # Gewichtungsfaktor für neue Beobachtungen
        self.model_weights = (1 - alpha) * self.model_weights + alpha * new_weights
        
        # Normalisiere erneut
        self.model_weights = self.model_weights / np.sum(self.model_weights)
        
        self.logger.info(f"Model weights updated: {self.model_weights}")
        
        # Speichere aktualisiertes Ensemble
        self.save_ensemble()
        
        return self.model_weights
    
    def analyze_ensemble_diversity(self):
        """
        Analysiert die Diversität der Modelle im Ensemble.
        
        Returns:
            dict: Diversitätsanalyse
        """
        if not self.models:
            return {"diversity": 0, "detail": "No models in ensemble"}
        
        # Erstelle zufällige Test-Features
        main_dim = self.base_config.get("main_feature_size", 19)
        cross_dim = self.base_config.get("cross_asset_feature_size", 10)
        order_dim = self.base_config.get("order_book_feature_size", 23)
        
        n_test = 10
        
        main_features = np.random.normal(0, 1, (n_test, main_dim))
        cross_features = np.random.normal(0, 1, (n_test, cross_dim))
        order_features = np.random.normal(0, 1, (n_test, order_dim))
        
        # Sammle Ergebnisse von allen Modellen
        model_outputs = []
        
        for model in self.models:
            if model.is_model_trained:
                try:
                    result = model.fuse_features(main_features, cross_features, order_features)
                    model_outputs.append(result)
                except Exception:
                    continue
        
        if len(model_outputs) < 2:
            return {"diversity": 0, "detail": "Not enough trained models for diversity analysis"}
        
        # Berechne paarweise Korrelationen zwischen Modellausgaben
        correlations = []
        
        for i in range(len(model_outputs)):
            for j in range(i+1, len(model_outputs)):
                if model_outputs[i].shape == model_outputs[j].shape:
                    # Berechne die durchschnittliche Korrelation über alle Dimensionen
                    dims = min(model_outputs[i].shape[1], model_outputs[j].shape[1]) if len(model_outputs[i].shape) > 1 else 1
                    
                    if dims > 1:
                        corrs = []
                        for d in range(dims):
                            corr = np.corrcoef(model_outputs[i][:, d], model_outputs[j][:, d])[0, 1]
                            if not np.isnan(corr):
                                corrs.append(abs(corr))
                        
                        if corrs:
                            avg_corr = np.mean(corrs)
                            correlations.append(avg_corr)
                    else:
                        corr = np.corrcoef(model_outputs[i].flatten(), model_outputs[j].flatten())[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
        
        if not correlations:
            return {"diversity": 0, "detail": "Could not compute correlations"}
        
        # Diversität = 1 - durchschnittliche Korrelation
        avg_correlation = np.mean(correlations)
        diversity = 1.0 - avg_correlation
        
        return {
            "diversity": float(diversity),
            "avg_correlation": float(avg_correlation),
            "correlations": [float(c) for c in correlations],
            "weights": self.model_weights.tolist()
        }
    
    def save_ensemble(self, path=None):
        """
        Speichert das Feature-Fusion-Ensemble.
        
        Args:
            path: Optionaler alternativer Speicherpfad
            
        Returns:
            bool: Erfolg des Speichervorgangs
        """
        if path is None:
            path = f"{self.model_path}_ensemble.pkl"
        
        try:
            # Erstelle Verzeichnis falls nicht vorhanden
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Speichere Ensemble-Metadaten und Gewichte
            ensemble_data = {
                "num_models": self.num_models,
                "base_config": self.base_config,
                "model_weights": self.model_weights.tolist(),
                "performance_history": self.performance_history,
                "timestamp": datetime.now().isoformat()
            }
            
            # Speichere Metadata mit Pickle
            with open(path, 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            # Speichere jedes Modell einzeln
            for i, model in enumerate(self.models):
                model_path = f"{self.model_path}_model_{i}.pkl"
                model.save_model(model_path)
            
            self.logger.info(f"Ensemble saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ensemble: {str(e)}")
            return False
    
    def load_ensemble(self, path=None):
        """
        Lädt ein gespeichertes Feature-Fusion-Ensemble.
        
        Args:
            path: Optionaler alternativer Ladepfad
            
        Returns:
            bool: Erfolg des Ladevorgangs
        """
        if path is None:
            path = f"{self.model_path}_ensemble.pkl"
        
        if not os.path.exists(path):
            self.logger.info(f"No existing ensemble found at {path}")
            return False
        
        try:
            # Lade Ensemble-Metadaten und Gewichte
            with open(path, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            # Aktualisiere Konfiguration
            if "base_config" in ensemble_data:
                self.base_config = ensemble_data["base_config"]
            
            if "num_models" in ensemble_data:
                self.num_models = ensemble_data["num_models"]
            
            if "model_weights" in ensemble_data:
                self.model_weights = np.array(ensemble_data["model_weights"])
            
            if "performance_history" in ensemble_data:
                self.performance_history = ensemble_data["performance_history"]
            
            # Lade jedes Modell einzeln
            loaded_models = []
            
            for i in range(self.num_models):
                model_path = f"{self.model_path}_model_{i}.pkl"
                
                if os.path.exists(model_path):
                    # Erstelle neues Modell mit Basiskonfiguration
                    model = RegularizedFeatureFusion(**self.base_config)
                    
                    # Lade Modellparameter
                    success = model.load_model(model_path)
                    
                    if success:
                        loaded_models.append(model)
                        self.logger.info(f"Model {i} loaded successfully")
                    else:
                        self.logger.warning(f"Failed to load model {i}")
                        # Füge Standardmodell hinzu
                        loaded_models.append(self.models[i] if i < len(self.models) else RegularizedFeatureFusion(**self.base_config))
                else:
                    self.logger.warning(f"Model file not found: {model_path}")
                    # Füge Standardmodell hinzu
                    loaded_models.append(self.models[i] if i < len(self.models) else RegularizedFeatureFusion(**self.base_config))
            
            # Update models list if we have enough loaded models
            if len(loaded_models) == self.num_models:
                self.models = loaded_models
            else:
                self.logger.warning(f"Loaded {len(loaded_models)}/{self.num_models} models, some models will be initialized")
                # Use loaded models where available, keep existing models for the rest
                for i in range(min(len(loaded_models), len(self.models))):
                    self.models[i] = loaded_models[i]
            
            self.logger.info(f"Ensemble loaded from {path}")
            
            # Log Ensemble-Informationen
            if "timestamp" in ensemble_data:
                self.logger.info(f"Ensemble saved on: {ensemble_data['timestamp']}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble: {str(e)}")
            return False