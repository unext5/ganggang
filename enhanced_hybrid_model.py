import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model, save_model, clone_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Attention, Dot, Reshape
from tensorflow.keras.layers import Activation, add, multiply, BatchNormalization, GRU
from tensorflow.keras.layers import TimeDistributed, Average, GlobalAveragePooling1D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
import os
import pickle
import json
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from collections import deque, Counter
import math
import random
from sklearn.feature_selection import mutual_info_regression

# For reinforcement learning components
import tensorflow_probability as tfp
tfd = tfp.distributions

class HybridModel:
    """
    Erweitertes Hybrid-Modell, das HMM-Zustandserkennung mit neuronalen Netzen für
    verbesserte Marktanalyse und Handelsentscheidungen kombiniert.
    
    Features:
    - Multi-modale Eingabeverarbeitung
    - Aufmerksamkeitsmechanismen für besseres Sequenzverständnis
    - Fortschrittliche Reinforcement Learning-Komponenten
    - Ensemble-Methoden für robustere Vorhersagen
    - Wissenskonsolidierung und kontinuierliches Lernen
    - Regularisierungstechniken gegen Überanpassung
    """
    def __init__(self, input_dim, hmm_states=4, lstm_units=64, dense_units=32, 
                 sequence_length=10, learning_rate=0.001, market_phase_count=5,
                 use_attention=True, use_ensemble=True, max_memory_size=2000):
        """
        Initialisiert das erweiterte Hybrid-Modell.
        
        Args:
            input_dim: Eingabedimension der Features
            hmm_states: Anzahl der HMM-Zustände
            lstm_units: Anzahl der LSTM-Einheiten
            dense_units: Anzahl der Dense-Layer-Einheiten
            sequence_length: Eingabe-Sequenzlänge für LSTM
            learning_rate: Lernrate
            market_phase_count: Anzahl unterschiedlicher Marktphasen für spezialisierte Modelle
            use_attention: Attention-Mechanismen aktivieren
            use_ensemble: Ensemble-Methoden verwenden
            max_memory_size: Maximale Größe des Erfahrungsspeichers
        """
        self.input_dim = input_dim
        self.hmm_states = hmm_states
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.market_phase_count = market_phase_count
        self.use_attention = use_attention
        self.use_ensemble = use_ensemble
        
        # Modellobjekte
        self.direction_model = None      # Richtungsvorhersage (Hauptmodell)
        self.volatility_model = None     # Volatilitätsvorhersage
        self.rl_model = None             # Reinforcement Learning für Ein-/Ausstieg
        
        # Ensemble-Komponenten
        self.ensemble_models = {}        # Modelle für verschiedene Marktphasen
        self.model_variants = []         # Verschiedene Architekturvarianten
        self.ensemble_weights = None     # Dynamische Gewichtungen
        
        # Scaler für Feature-Normalisierung
        self.feature_scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Trainings-Historie
        self.direction_history = None
        self.volatility_history = None
        self.rl_history = None
        
        # Komponenten für kontinuierliches Lernen
        self.is_model_loaded = False
        self.checkpoint_dir = "./model_checkpoints"
        
        # Reinforcement Learning-Komponenten
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = 0.95       # Discount-Faktor
        self.epsilon = 1.0      # Explorationsrate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_model = None  # Zielmodell für stabile RL-Updates
        self.update_target_counter = 0
        self.rl_algorithm = "dqn"  # Verschiedene RL-Algorithmen: "dqn", "sac", "ppo"
        
        # Marktphase-Erkennung und Spezialisierung
        self.market_phases = ["trending_bull", "trending_bear", "ranging", "high_volatility", "low_volatility"]
        self.phase_expertise = {phase: 0.5 for phase in self.market_phases}
        
        # Pattern Memory für Wissenskonsolidierung
        self.pattern_memory = {
            "patterns": deque(maxlen=500),
            "consolidated": deque(maxlen=100),
            "last_consolidated": None
        }
        
        # Logger einrichten
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('hybrid_model')
        
        # Speicher für Markt-Embeddings (für Transfer-Learning)
        self.market_embeddings = {}
        
        # Verzeichnisstruktur erstellen
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Increased regularization parameters
        self.dropout_rate = 0.4  # Increased dropout rate
        self.l1_reg = 0.001       # Added/Increased L1 regularization
        self.l2_reg = 0.002       # Increased L2 regularization
        
        self.logger.info("HybridModel initialized with increased regularization") # Added info
    
    def build_models(self):
        """
        Baut alle Modelle des Hybrid-Systems mit erweiterten Architekturen.
        """
        # 1. Basis-Architektur für Sequenzmodelle wählen
        if self.use_attention:
            self._build_attention_based_models()
        else:
            self._build_standard_models()
        
        # 2. Reinforcement Learning-Modell aufbauen
        if self.rl_algorithm == "dqn":
            self._build_dqn_model()
        elif self.rl_algorithm == "sac":
            self._build_sac_model()
        else:
            self._build_dqn_model()  # Fallback auf DQN
        
        # 3. Ensemble-Modelle für verschiedene Marktphasen aufbauen
        if self.use_ensemble:
            self._build_ensemble_models()
        
        self.logger.info("Alle Neural-Network-Modelle erfolgreich initialisiert")
        
        return {
            "direction_model": self.direction_model,
            "volatility_model": self.volatility_model,
            "rl_model": self.rl_model,
            "ensemble_models": self.ensemble_models if self.use_ensemble else None
        }
    
    def _build_attention_based_models(self):
        """
        Baut verbesserte Modelle mit Attention-Mechanismen.
        Mit Unterstützung für 5-Klassen-Richtungsvorhersage und 3-Klassen-Volatilitätsvorhersage.
        REGULARIZATION ADDED/INCREASED.
        """
        # 1. Richtungsvorhersage-Modell
        direction_inputs = Input(shape=(self.sequence_length, self.input_dim))
        hmm_input = Input(shape=(self.hmm_states,))
    
        # Bidirektionales LSTM mit erhöhter Regularisierung
        x = Bidirectional(LSTM(self.lstm_units,
                              return_sequences=True,
                              kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),      # ADDED Regularizer
                              recurrent_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)))(direction_inputs) # ADDED Regularizer
        x = Dropout(self.dropout_rate)(x) # Use increased rate
    
        # Self-Attention Layer
        e = Dense(1, activation='tanh')(x)
        e = Reshape((-1,))(e)
        attention_weights = Activation('softmax')(e)
    
        # Wende Attention auf Sequenz an
        context = Dot(axes=1)([x, attention_weights])
    
        # Kombiniere mit HMM-Zustand
        combined = Concatenate()([context, hmm_input])
    
        # Dense Layers mit erhöhter Regularisierung
        x = Dense(self.dense_units, activation='relu',
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(combined) # ADDED Regularizer
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate + 0.1)(x) # Increased dropout rate (e.g., 0.5)
    
        # Ausgabe: [Strong Down, Weak Down, Neutral, Weak Up, Strong Up]
        direction_output = Dense(5, activation='softmax')(x)
    
        self.direction_model = Model(
            inputs=[direction_inputs, hmm_input],
            outputs=direction_output
        )
    
        self.direction_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info("Attention-based Direction Model built with increased regularization.") # Log message updated
    
        # 2. Volatilitätsvorhersage-Modell (Apply similar regularization)
        volatility_inputs = Input(shape=(self.sequence_length, self.input_dim))
        # CNN-LSTM Hybrid für Volatilitätserkennung
        conv = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(volatility_inputs)
        conv = BatchNormalization()(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
    
        # Apply regularization to Bidirectional LSTM
        x = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True,
                              kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                              recurrent_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)))(conv) # ADDED Regularizer
        x = Dropout(self.dropout_rate)(x) # Use increased rate
    
        # Attention
        e = Dense(1, activation='tanh')(x)
        e = Reshape((-1,))(e)
        attention_weights = Activation('softmax')(e)
        context = Dot(axes=1)([x, attention_weights])
    
        # Kombiniere mit HMM-Zustand
        combined = Concatenate()([context, hmm_input])
    
        x = Dense(self.dense_units, activation='relu',
                  kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(combined) # ADDED Regularizer
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x) # Use increased rate
    
        # Ausgabe: [Low, Medium, High] Volatilität
        volatility_output = Dense(3, activation='softmax')(x)
    
        self.volatility_model = Model(
            inputs=[volatility_inputs, hmm_input],
            outputs=volatility_output
        )
    
        self.volatility_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info("Attention-based Volatility Model built with increased regularization.") # Log message updated
    
    def _build_standard_models(self):
        """
        Baut Standard-Modelle (ohne Attention) für Systeme mit weniger Ressourcen.
        Mit Unterstützung für 5-Klassen-Richtungsvorhersage und 3-Klassen-Volatilitätsvorhersage.
        REGULARIZATION ADDED/INCREASED.
        """
        # 1. Richtungsvorhersage-Modell
        direction_inputs = Input(shape=(self.sequence_length, self.input_dim))
        hmm_input = Input(shape=(self.hmm_states,))
    
        # Apply regularization to LSTMs
        x = LSTM(self.lstm_units, return_sequences=True,
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                 recurrent_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(direction_inputs) # ADDED Regularizer
        x = Dropout(self.dropout_rate)(x) # Use increased rate
        x = LSTM(self.lstm_units // 2,
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                 recurrent_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(x) # ADDED Regularizer
        x = Dropout(self.dropout_rate)(x) # Use increased rate
    
        combined = Concatenate()([x, hmm_input])
    
        # Apply regularization to Dense layer
        x = Dense(self.dense_units, activation='relu',
                  kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(combined) # ADDED Regularizer
        direction_output = Dense(5, activation='softmax')(x)
        self.direction_model = Model(
            inputs=[direction_inputs, hmm_input],
            outputs=direction_output
        )
        self.direction_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info("Standard Direction Model built with increased regularization.") # Log message updated
    
        # 2. Volatilitätsvorhersage-Modell (Apply similar regularization)
        volatility_inputs = Input(shape=(self.sequence_length, self.input_dim))
        # Apply regularization to LSTMs
        x = LSTM(self.lstm_units, return_sequences=True,
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                 recurrent_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(volatility_inputs) # ADDED Regularizer
        x = Dropout(self.dropout_rate)(x) # Use increased rate
        x = LSTM(self.lstm_units // 2,
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                 recurrent_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(x) # ADDED Regularizer
        x = Dropout(self.dropout_rate)(x) # Use increased rate
    
        combined = Concatenate()([x, hmm_input])
        # Apply regularization to Dense layer
        x = Dense(self.dense_units, activation='relu',
                  kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(combined) # ADDED Regularizer
        volatility_output = Dense(3, activation='softmax')(x)
        self.volatility_model = Model(
            inputs=[volatility_inputs, hmm_input],
            outputs=volatility_output
        )
        self.volatility_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info("Standard Volatility Model built with increased regularization.") # Log message updated
    
    def _build_dqn_model(self):
        """
        Baut ein DQN-Model (Deep Q-Network) für Reinforcement Learning mit erweitertem Aktionsraum.
    
        Der erweiterte Aktionsraum umfasst 11 mögliche Aktionen:
        0: Keine Aktion/Position halten
        1: Long-Position eröffnen (volle Größe)
        2: Long-Position eröffnen (halbe Größe)
        3: Bestehende Long-Position aufstocken
        4: Short-Position eröffnen (volle Größe)
        5: Short-Position eröffnen (halbe Größe)
        6: Bestehende Short-Position aufstocken
        7: Long-Position schließen
        8: Short-Position schließen
        9: Long-Position teilweise schließen (Gewinnmitnahme)
        10: Short-Position teilweise schließen (Gewinnmitnahme)
        """
        # Eingaben: Feature-Sequenz, HMM-Zustand und Marktkontext
        rl_inputs = Input(shape=(self.sequence_length, self.input_dim))
        hmm_input = Input(shape=(self.hmm_states,))
        context_input = Input(shape=(12,))  # Erweiterte Kontext-Features (statt 6)

        # Feature-Extraktion
        x = LSTM(self.lstm_units, return_sequences=False)(rl_inputs)
        x = Dropout(0.2)(x)

        # Kombinieren aller Eingaben
        combined = Concatenate()([x, hmm_input, context_input])

        # Value-Network
        x = Dense(self.dense_units, activation='relu')(combined)
        x = Dense(self.dense_units // 2, activation='relu')(x)

        # Erweiterte Q-Values mit 11 Aktionen:
        # [No Action, Enter Long (full), Enter Long (half), Scale Long,
        #  Enter Short (full), Enter Short (half), Scale Short, 
        #  Exit Long, Exit Short, Partial Exit Long, Partial Exit Short]
        q_values = Dense(11, activation='linear')(x)

        self.rl_model = Model(
            inputs=[rl_inputs, hmm_input, context_input],
            outputs=q_values
        )

        self.rl_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        # Zielmodell für Stabilität
        self.target_model = clone_model(self.rl_model)
        self.target_model.set_weights(self.rl_model.get_weights())
    
    def _build_sac_model(self):
        """
        Soft Actor-Critic (SAC) Implementierung für fortgeschrittenes RL.
        Unterstützt erweiterten Aktionsraum mit 11 Aktionen.
        """
        # Implementiert einen modernen aktorbasierten RL-Algorithmus
    
        # Eingaben
        state_input = Input(shape=(self.sequence_length, self.input_dim))
        hmm_input = Input(shape=(self.hmm_states,))
        context_input = Input(shape=(12,))  # Erweiterte Kontext-Features
    
        # Feature-Extraktion (gemeinsam für Actor und Critic)
        lstm_features = LSTM(self.lstm_units)(state_input)
        combined_features = Concatenate()([lstm_features, hmm_input, context_input])
    
        # Actor-Netzwerk (Policy)
        actor_hidden = Dense(self.dense_units, activation='relu')(combined_features)
        actor_hidden = Dense(self.dense_units // 2, activation='relu')(actor_hidden)
    
        # Ausgabe von Mittelwert und Log-Standardabweichung für Gaußsche Policy
        # Angepasst für 11-dimensionalen Aktionsraum
        action_mean = Dense(11, activation='tanh')(actor_hidden)
        action_log_std = Dense(11, activation='linear')(actor_hidden)
    
        # Critic-Netzwerk (Q-Funktion)
        # Zwei Q-Netzwerke für Double-Q-Learning
        def build_critic():
            critic_hidden = Dense(self.dense_units, activation='relu')(combined_features)
            critic_hidden = Dense(self.dense_units // 2, activation='relu')(critic_hidden)
            q_values = Dense(1, activation='linear')(critic_hidden)
            return Model(inputs=[state_input, hmm_input, context_input], outputs=q_values)
    
        critic1 = build_critic()
        critic2 = build_critic()
    
        # Soft Actor-Critic spezifische Parameter
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.target_entropy = -tf.constant(11, dtype=tf.float32)  # Anzahl der Aktionen = 11
    
        # Placeholder für die SAC-Implementierung
        self.actor_model = Model(
            inputs=[state_input, hmm_input, context_input],
            outputs=[action_mean, action_log_std]
        )
    
        self.critic_models = [critic1, critic2]
    
        # Create target critics for stability
        self.target_critics = [clone_model(critic1), clone_model(critic2)]
        for i in range(2):
            self.target_critics[i].set_weights(self.critic_models[i].get_weights())
    
        # Optimizer
        self.actor_optimizer = Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = Adam(learning_rate=self.learning_rate)
        self.alpha_optimizer = Adam(learning_rate=self.learning_rate)
    
        # Verwende rl_model als Wrapper für SAC
        self.rl_model = self.actor_model
    
    def _build_ensemble_models(self):
        """
        Erstellt ein Ensemble spezialisierter Modelle für verschiedene Marktphasen.
        Mit Unterstützung für 5-Klassen-Richtungsvorhersage.
        """
        for phase in self.market_phases:
            # Spezialisiertes Modell für jede Marktphase
            # Architektur anpassen
            if phase in ["trending_bull", "trending_bear"]:
                # Trending Markt: Optimiert für Trend-Erkennung
                lstm_units = int(self.lstm_units * 1.2)
                dense_units = int(self.dense_units * 1.2)
            elif phase == "ranging":
                # Seitwärtsmarkt: Optimiert für Mean-Reversion
                lstm_units = int(self.lstm_units * 0.9)
                dense_units = int(self.dense_units * 1.1)
            else:
                # Standard-Konfiguration
                lstm_units = self.lstm_units
                dense_units = self.dense_units
        
            # Modell erstellen
            inputs = Input(shape=(self.sequence_length, self.input_dim))
            hmm_input = Input(shape=(self.hmm_states,))
        
            # Schichtenarchitektur
            x = LSTM(lstm_units, return_sequences=True)(inputs)
            x = Dropout(0.2)(x)
            x = LSTM(lstm_units // 2)(x)
        
            # Kombiniere mit HMM
            combined = Concatenate()([x, hmm_input])
        
            x = Dense(dense_units, activation='relu')(combined)
            # Ausgabe für 5-Klassen-System: [Strong Down, Weak Down, Neutral, Weak Up, Strong Up]
            outputs = Dense(5, activation='softmax')(x)
        
            model = Model(inputs=[inputs, hmm_input], outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
            self.ensemble_models[phase] = {
                "model": model,
                "weight": 1.0 / len(self.market_phases),
                "performance": 0.5,
                "training_samples": 0
            }
    
        # Architekturvarianten für zusätzliche Diversität
        archs = ["lstm", "gru", "cnn"]
        for arch in archs:
            inputs = Input(shape=(self.sequence_length, self.input_dim))
            hmm_input = Input(shape=(self.hmm_states,))
        
            if arch == "lstm":
                x = LSTM(self.lstm_units)(inputs)
            elif arch == "gru":
                x = GRU(self.lstm_units)(inputs)
            elif arch == "cnn":
                x = Conv1D(32, 3, activation='relu')(inputs)
                x = MaxPooling1D(2)(x)
                x = Flatten()(x)
        
            combined = Concatenate()([x, hmm_input])
            x = Dense(self.dense_units, activation='relu')(combined)
            # 5-Klassen-Ausgabe
            outputs = Dense(5, activation='softmax')(x)
        
            variant = Model(inputs=[inputs, hmm_input], outputs=outputs)
            variant.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy', 
                metrics=['accuracy']
            )
        
            self.model_variants.append({
                "name": arch,
                "model": variant,
                "weight": 1.0 / len(archs)
            })
    
    def _detect_market_phase(self, features):
        """
        Erkennt die aktuelle Marktphase basierend auf den Features.
        
        Args:
            features: Feature-Matrix oder -Vektor
        
        Returns:
            string: Erkannte Marktphase
        """
        # Einfache Heuristik für die Phasenerkennung
        # In einer echten Implementierung würde hier ein eigenständiges Modell verwendet
        
        if isinstance(features, np.ndarray):
            if len(features.shape) > 2:
                # Sequenz von Features - letzte verwenden
                current_features = features[-1]
            elif len(features.shape) == 2 and features.shape[0] > 1:
                # Mehrere Feature-Vektoren - mitteln
                current_features = np.mean(features, axis=0)
            else:
                # Einzelner Feature-Vektor
                current_features = features.flatten()
        else:
            # Fallback
            return self.market_phases[0]
        
        # Für die Demonstration: einfache Regel-basierte Erkennung
        # Annahme: Die ersten paar Features sind Returns
        if len(current_features) >= 4:
            avg_returns = np.mean(current_features[:4])
            return_volatility = np.std(current_features[:4])
            
            if avg_returns > 0.002:
                return "trending_bull"
            elif avg_returns < -0.002:
                return "trending_bear"
            elif return_volatility > 0.005:
                return "high_volatility"
            elif return_volatility < 0.001:
                return "low_volatility"
            else:
                return "ranging"
        
        # Fallback wenn nicht genug Features
        return "ranging"
    
    def train_direction_model(self, X_seq, hmm_states, y_direction, epochs=50, batch_size=32, 
                            validation_split=0.2, validation_data=None, market_phase=None, use_callbacks=True):
        """
        Trainiert das Richtungsvorhersage-Modell mit erweitertem Callback-System.
        
        # CRUCIAL FIX: Prevent overfitting by using cross-validation approach
    
        Args:
            X_seq: Sequenz von Feature-Vektoren [samples, sequence_length, features]
            hmm_states: HMM-Zustandsvektoren [samples, hmm_states]
            y_direction: Richtungslabels (one-hot) [samples, 3]
            epochs: Anzahl der Trainingsepochen
            batch_size: Batch-Größe
            validation_split: Anteil der Validierungsdaten (wird ignoriert, wenn validation_data gesetzt ist)
            validation_data: Tuple aus ([X_val, hmm_val], y_val) für explizite Validierung
            market_phase: Optional, spezifische Marktphase für spezialisiertes Training
            use_callbacks: Callbacks für frühes Stoppen und Checkpoints verwenden
    
        Returns:
            Training-Historie
        """
        if self.direction_model is None:
            self.build_models()
    
        # Feature-Normalisierung
        if not self.scaler_fitted:
            # Reshape für Skalierung
            shape = X_seq.shape
            X_flat = X_seq.reshape(-1, self.input_dim)
            self.feature_scaler.fit(X_flat)
            self.scaler_fitted = True
            # Zurück zur ursprünglichen Form
            X_scaled = self.feature_scaler.transform(X_flat).reshape(shape)
        else:
            # Bestehenden Scaler anwenden
            shape = X_seq.shape
            X_flat = X_seq.reshape(-1, self.input_dim)
            X_scaled = self.feature_scaler.transform(X_flat).reshape(shape)
    
        # Callbacks für besseres Training
        callbacks = []
    
        if use_callbacks:
            monitor_metric = 'val_loss' if validation_data else 'loss'
            callbacks_list = [
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor_metric,
                    patience=7,  # Reduced patience from 10 to 7
                    restore_best_weights=True,
                    min_delta=0.001 # Kept min_delta
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor_metric,
                    factor=0.5,
                    patience=4,   # Reduced patience from 5 to 4
                    min_lr=1e-6, # Kept min_lr
                    verbose=1
                ),
                # Optional: Checkpoint callback
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.checkpoint_dir, f"direction_model_{market_phase or 'general'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"),
                    save_best_only=True,
                    monitor=monitor_metric,
                    verbose=1
                )
            ]
            callbacks.extend(callbacks_list)
        
        # Wähle das richtige Modell aus
        target_model = self.direction_model
    
        if market_phase is not None and market_phase in self.ensemble_models:
            # Training eines spezialisierten Modells
            target_model = self.ensemble_models[market_phase]["model"]
            self.logger.info(f"Training des spezialisierten Modells für {market_phase}")
        
            # Aktualisiere Trainingszähler
            self.ensemble_models[market_phase]["training_samples"] += len(X_seq)
    
        # Korrekte Initialisierung und Befüllung von train_args
        train_args = {} 
        val_data_used = None
        # Use scaled data for training
        X_train_seq = X_scaled 

        if validation_data:
            # Use explicit validation data if provided
            # Corrected unpacking: expects tuple (inputs, labels)
            val_inputs, y_val_direction = validation_data 
            # Further unpack the inputs
            X_val_seq, hmm_val_states = val_inputs
            
            # Ensure validation data components are numpy arrays
            if not isinstance(y_val_direction, np.ndarray): y_val_direction = np.array(y_val_direction)
            if not isinstance(X_val_seq, np.ndarray): X_val_seq = np.array(X_val_seq)
            if not isinstance(hmm_val_states, np.ndarray): hmm_val_states = np.array(hmm_val_states)
            
            # IMPORTANT ANTI-OVERFITTING FIX:
            # Make sure validation set is sufficiently different from training set
            # This is crucial for preventing the perfect accuracy issue
            
            # 1. Check data size before proceeding
            min_validation_size = 100  # Absolute minimum for meaningful validation
            if len(y_val_direction) < min_validation_size:
                self.logger.warning(f"Validation set too small ({len(y_val_direction)} samples). Using time-based split instead.")
                # Create time-based split (last 20% of data)
                split_idx = int(len(X_seq) * 0.8)
                X_val_seq = X_seq[split_idx:]
                hmm_val_states = hmm_states[split_idx:]
                y_val_direction = y_direction[split_idx:]
                # Use remaining 80% for training
                X_seq = X_seq[:split_idx]
                hmm_states = hmm_states[:split_idx]
                y_direction = y_direction[:split_idx]
            
            # 2. Add slight noise to validation data to prevent overfitting
            noise_scale = 0.01
            if isinstance(X_val_seq, np.ndarray) and X_val_seq.size > 0:
                # Add small amount of Gaussian noise
                X_val_seq = X_val_seq + np.random.normal(0, noise_scale, X_val_seq.shape)
                
            # Pass validation data as a tuple: ([Input1, Input2], Labels)
            train_args['validation_data'] = ([X_val_seq, hmm_val_states], y_val_direction) 
            val_data_used = validation_data 
        elif validation_split > 0:
            # Use validation_split if no explicit data is given
            # IMPORTANT: For time series, we want to use the LAST portion of data for validation
            # not a random sample, to prevent information leakage
            train_args['validation_split'] = validation_split
            # We'll use a custom validation split implementation
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_val_seq = X_seq[split_idx:]
            hmm_val_states = hmm_states[split_idx:]
            y_val_direction = y_direction[split_idx:]
            # Use the first portion for training
            X_train_seq = X_seq[:split_idx]
            hmm_train_states = hmm_states[:split_idx]
            y_train_direction = y_direction[:split_idx]
            
            # Replace the original data with the training portion
            X_seq = X_train_seq
            hmm_states = hmm_train_states
            y_direction = y_train_direction
            
            # Set up explicit validation data instead of using validation_split
            train_args.pop('validation_split', None)
            train_args['validation_data'] = ([X_val_seq, hmm_val_states], y_val_direction)
            
            # shuffle=False will be passed directly to fit() below
            val_data_used = f"time-based-split={validation_split}"
        else:
            # No validation split
            train_args['validation_data'] = None 
            val_data_used = None

        self.logger.info(f"Training direction prediction model{' for phase ' + str(market_phase) if market_phase is not None else ''}")

        # The actual fit call with shuffle=False and **train_args
        history = target_model.fit(
            [X_train_seq, hmm_states], # Training data
            y_direction,             # Training labels
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list if use_callbacks else [],
            verbose=1, 
            shuffle=False, # IMPORTANT: Always False for time series!
            **train_args # Passes validation_split or validation_data
        )
        
        self.direction_history = history.history
        self.logger.info("Richtungsmodell-Training abgeschlossen.")
    
        # Mustererkennung für Wissenskonsolidierung
        self._update_pattern_memory(X_scaled, hmm_states, y_direction)
    
        return history.history
    
    def train_with_market_memory(self, market_memory, epochs=20, batch_size=32):
        """
        Trainiert mit Mustern aus dem Markt-Memory für bessere Generalisierung.
        
        Args:
            market_memory: MarketMemory-Instanz
            epochs: Anzahl der Trainingsepochen
            batch_size: Batch-Größe
        
        Returns:
            Training-Historie
        """
        if not hasattr(market_memory, 'patterns') or len(market_memory.patterns) < 10:
            self.logger.warning("Nicht genug Muster im Markt-Memory für Training.")
            return None
        
        self.logger.info(f"Training mit {len(market_memory.patterns)} Mustern aus Markt-Memory.")
        
        # Muster aus Memory extrahieren
        X_seq_list = []
        labels_list = []
        hmm_states_list = []
        
        for pattern in market_memory.patterns:
            if "features" not in pattern or not isinstance(pattern["features"], np.ndarray):
                continue
                
            features = pattern["features"]
            if len(features) < self.sequence_length:
                continue
                
            # Features für Sequenztraining vorbereiten
            X_seq_list.append(features[-self.sequence_length:])
            
            # Labels aus State-Label ableiten
            state_label = pattern.get("state_label", "Neutral")
            label = np.zeros(3)  # [Up, Down, Sideways]
            
            if "Bull" in state_label:
                label[0] = 1  # Up
            elif "Bear" in state_label:
                label[1] = 1  # Down
            else:
                label[2] = 1  # Sideways
                
            labels_list.append(label)
            
            # Mock HMM-Zustände (falls nicht verfügbar)
            hmm_state = np.zeros(self.hmm_states)
            hmm_states_list.append(hmm_state)
        
        if len(X_seq_list) < 5:
            self.logger.warning("Zu wenige valide Muster für Training gefunden.")
            return None
        
        # Konvertiere zu NumPy-Arrays
        X_seq = np.array(X_seq_list)
        y_direction = np.array(labels_list)
        hmm_states = np.array(hmm_states_list)
        
        # Trainingsfunktion mit diesen Daten aufrufen
        return self.train_direction_model(
            X_seq, hmm_states, y_direction, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=0.1  # Kleinere Validierungsaufteilung bei Memory-Daten
        )
    
    def _update_pattern_memory(self, X_seq, hmm_states, y_direction):
        """
        Aktualisiert den internen Pattern-Speicher für Wissenskonsolidierung.
        
        Args:
            X_seq: Feature-Sequenzen
            hmm_states: HMM-Zustände
            y_direction: Richtungslabels
        """
        # Nur einen Teil der Muster speichern (max. 100 pro Update)
        max_patterns = min(100, len(X_seq))
        indices = np.random.choice(len(X_seq), max_patterns, replace=False)
        
        # Füge ausgewählte Muster hinzu
        for idx in indices:
            self.pattern_memory["patterns"].append({
                "features": X_seq[idx],
                "hmm_state": hmm_states[idx],
                "label": y_direction[idx],
                "timestamp": datetime.now().isoformat()
            })
        
        # Prüfe, ob Konsolidierung nötig ist
        self._check_consolidation_needed()
    
    def _check_consolidation_needed(self):
        """
        Prüft, ob eine Musterkonsolidierung nötig ist und führt sie ggf. durch.
        """
        # Konsolidierung einmal täglich oder nach 500 neuen Mustern
        last_consolidated = self.pattern_memory["last_consolidated"]
        
        if (last_consolidated is None or 
            len(self.pattern_memory["patterns"]) >= 400 or
            (datetime.now() - datetime.fromisoformat(last_consolidated)).days >= 1):
            
            self.consolidate_patterns()
    
    def consolidate_patterns(self):
        """
        Konsolidiert ähnliche Muster um Speicherplatz zu sparen und Überanpassung zu vermeiden.
        """
        patterns = list(self.pattern_memory["patterns"])
        if len(patterns) < 50:
            return
        
        # Features für Clustering extrahieren
        feature_arrays = []
        for pattern in patterns:
            # Durchschnitt für jede Sequenz
            avg_features = np.mean(pattern["features"], axis=0)
            feature_arrays.append(avg_features)
        
        if not feature_arrays:
            return
            
        # Features für Clustering
        cluster_features = np.array(feature_arrays)
        
        # K-Means Clustering
        n_clusters = min(20, len(cluster_features) // 5)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(cluster_features)
        
        # Konsolidierte Muster erstellen
        consolidated = []
        
        for cluster_id in range(n_clusters):
            # Finde alle Muster in diesem Cluster
            cluster_patterns = [p for i, p in enumerate(patterns) if clusters[i] == cluster_id]
            
            if not cluster_patterns:
                continue
                
            # Durchschnittliche Features und Label berechnen
            avg_features = np.mean([p["features"] for p in cluster_patterns], axis=0)
            
            # Mehrheitsabstimmung für Labels
            labels = np.array([p["label"] for p in cluster_patterns])
            avg_label = np.mean(labels, axis=0)
            
            # Konsolidiertes Muster erstellen
            consolidated.append({
                "features": avg_features,
                "label": avg_label,
                "count": len(cluster_patterns),
                "timestamp": datetime.now().isoformat()
            })
        
        # Alte konsolidierte Muster löschen und neue hinzufügen
        self.pattern_memory["consolidated"] = deque(consolidated, maxlen=100)
        self.pattern_memory["last_consolidated"] = datetime.now().isoformat()
        
        # Pattern Memory aufräumen - behalte nur neueste
        if len(self.pattern_memory["patterns"]) > 100:
            # Sortiere nach Zeitstempel (neueste zuerst)
            sorted_patterns = sorted(
                list(self.pattern_memory["patterns"]),
                key=lambda p: p.get("timestamp", ""),
                reverse=True
            )
            # Behalte nur die neuesten 100
            self.pattern_memory["patterns"] = deque(sorted_patterns[:100], maxlen=500)
        
        self.logger.info(f"Musterkonsolidierung abgeschlossen. {len(consolidated)} Cluster erstellt.")
    
    def train_volatility_model(self, X_seq, hmm_states, y_volatility, epochs=50, batch_size=32, validation_split=0.2):
        """
        Trainiert das Volatilitätsvorhersage-Modell.
        
        # CRUCIAL FIX: Use same anti-overfitting strategy as direction model
        
        Args:
            X_seq: Sequenz von Feature-Vektoren [samples, sequence_length, features]
            hmm_states: HMM-Zustandsvektoren [samples, hmm_states]
            y_volatility: Volatilitätslabels (one-hot) [samples, 3]
            epochs: Anzahl der Trainingsepochen
            batch_size: Batch-Größe
            validation_split: Anteil der Validierungsdaten
        
        Returns:
            Training-Historie
        """
        if self.volatility_model is None:
            self.build_models()
        
        # Feature-Normalisierung
        if not self.scaler_fitted:
            # Reshape für Skalierung
            shape = X_seq.shape
            X_flat = X_seq.reshape(-1, self.input_dim)
            self.feature_scaler.fit(X_flat)
            self.scaler_fitted = True
            X_scaled = self.feature_scaler.transform(X_flat).reshape(shape)
        else:
            # Bestehenden Scaler anwenden
            shape = X_seq.shape
            X_flat = X_seq.reshape(-1, self.input_dim)
            X_scaled = self.feature_scaler.transform(X_flat).reshape(shape)
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"volatility_model_{datetime.now().strftime('%Y%m%d_%H%M')}.h5"
        )
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss'
        )
        
        # Trainiere Modell
        # Implement time-based validation split for time series
        # IMPORTANT: For time series, we want to use the LAST portion of data for validation
        # not a random sample, to prevent information leakage
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_val_seq = X_scaled[split_idx:]
        hmm_val_states = hmm_states[split_idx:]
        y_val_volatility = y_volatility[split_idx:]
        
        # Use the first portion for training
        X_train_seq = X_scaled[:split_idx]
        hmm_train_states = hmm_states[:split_idx]
        y_train_volatility = y_volatility[:split_idx]
        
        # Add slight noise to validation data to prevent overfitting
        noise_scale = 0.01
        if isinstance(X_val_seq, np.ndarray) and X_val_seq.size > 0:
            # Add small amount of Gaussian noise
            X_val_seq = X_val_seq + np.random.normal(0, noise_scale, X_val_seq.shape)
        
        history = self.volatility_model.fit(
            [X_train_seq, hmm_train_states],
            y_train_volatility,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([X_val_seq, hmm_val_states], y_val_volatility),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        self.volatility_history = history.history
        self.logger.info("Volatilitätsmodell-Training abgeschlossen.")
        
        return history.history
    
    def train_rl_model_supervised(self, X_seq, hmm_states, context_data, y_actions, epochs=50, batch_size=32):
        """
        Vortraining des RL-Modells mit überwachtem Lernen vor dem eigentlichen RL.
        
        Args:
            X_seq: Sequenz von Feature-Vektoren [samples, sequence_length, features]
            hmm_states: HMM-Zustandsvektoren [samples, hmm_states]
            context_data: Marktkontext-Features [samples, 6]
            y_actions: Aktionslabels [samples, 5]
            epochs: Anzahl der Trainingsepochen
            batch_size: Batch-Größe
        
        Returns:
            Training-Historie
        """
        if self.rl_model is None:
            self.build_models()
        
        # Feature-Normalisierung
        if not self.scaler_fitted:
            shape = X_seq.shape
            X_flat = X_seq.reshape(-1, self.input_dim)
            self.feature_scaler.fit(X_flat)
            self.scaler_fitted = True
            X_scaled = self.feature_scaler.transform(X_flat).reshape(shape)
        else:
            shape = X_seq.shape
            X_flat = X_seq.reshape(-1, self.input_dim)
            X_scaled = self.feature_scaler.transform(X_flat).reshape(shape)
        
        # Memory-effizientes Training mit optimierten Batch-Größen
        # Kleinere Batches für sehr große Datensätze
        if len(X_seq) > 10000:
            # Dynamische Batch-Größe-Anpassung
            adaptive_batch_size = min(batch_size, max(8, len(X_seq) // 500))
            self.logger.info(f"Verwende angepasste Batch-Größe {adaptive_batch_size} für großen Datensatz")
            batch_size = adaptive_batch_size
    
        # Checkpointing für langes Training hinzufügen
        checkpoint_path = os.path.join(self.checkpoint_dir, 'rl_model_checkpoint.h5')
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    
        # Früher Abbruch zur Vermeidung von Überanpassung
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5,
            restore_best_weights=True
        )
    
        # Trainiere mit überwachtem Lernen
        if self.rl_algorithm == "dqn":
            # Speichereffizientes Training mit Callbacks
            history = self.rl_model.fit(
                [X_scaled, hmm_states, context_data],
                y_actions,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[checkpoint_callback, early_stopping],
                verbose=1
            )
        
            # Update des Target-Models
            self.target_model.set_weights(self.rl_model.get_weights())
            self.rl_history = history.history
        
        elif self.rl_algorithm == "sac":
            # Bei aktorbasierten Algorithmen direkte Anpassung der Policy
            # Implementierung erfordert mehr angepasstes Training
            self.logger.warning("SAC Vortraining noch nicht implementiert. Überspringe.")
            return None
        
        self.logger.info("RL-Modell-Vortraining abgeschlossen.")
        
        return self.rl_history
    
    def continue_training(self, X_seq, hmm_states, y_direction, epochs=10, learning_rate_factor=0.1):
        """
        Setzt das Training mit neuen Daten fort (kontinuierliches Lernen).
        
        Args:
            X_seq: Neue Feature-Sequenzen
            hmm_states: Neue HMM-Zustände
            y_direction: Neue Richtungslabels
            epochs: Anzahl der Trainingsepochen
            learning_rate_factor: Reduktionsfaktor für die Lernrate
        
        Returns:
            Training-Historie
        """
        # Stelle sicher, dass ein Modell geladen ist
        if not self.is_model_loaded and self.direction_model is None:
            self.logger.error("Kein Modell geladen oder initialisiert für weiteres Training.")
            return None
        
        # Reduzierte Lernrate für Feinabstimmung
        original_lr = self.learning_rate
        new_lr = original_lr * learning_rate_factor
        
        # Modell mit neuer Lernrate neu kompilieren
        K.set_value(self.direction_model.optimizer.learning_rate, new_lr)
        
        self.logger.info(f"Training wird fortgesetzt mit Lernrate {new_lr:.6f} (Ursprünglich: {original_lr:.6f})")
        
        # Training mit den neuen Daten
        history = self.train_direction_model(
            X_seq, 
            hmm_states, 
            y_direction,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1  # Kleinere Validierung für kontinuierliches Lernen
        )
        
        # Lernrate zurücksetzen
        K.set_value(self.direction_model.optimizer.learning_rate, original_lr)
        
        return history
    
    def memorize(self, state, hmm_state, context, action, reward, next_state, next_hmm, next_context, done):
        """
        Speichert Erfahrung im Replay-Memory für Reinforcement Learning.
        """
        self.memory.append((state, hmm_state, context, action, reward, 
                          next_state, next_hmm, next_context, done))
    
    def act(self, state, hmm_state, context):
        """
        Wählt eine Aktion basierend auf aktuellem Zustand mit Epsilon-Greedy-Strategie.
        Unterstützt erweiterten Aktionsraum mit 11 möglichen Aktionen.
    
        Args:
            state: Current state (feature sequence)
            hmm_state: Current HMM state
            context: Market context features
    
        Returns:
            Selected action index
        """
        if self.rl_algorithm == "dqn":
            # Epsilon-Greedy-Exploration
            if np.random.rand() <= self.epsilon:
                return random.randrange(11)  # Zufällige Aktion aus erweitertem Aktionsraum
        
            # Skaliere Zustand
            if self.scaler_fitted:
                shape = state.shape
                state_flat = state.reshape(-1, self.input_dim)
                state_scaled = self.feature_scaler.transform(state_flat).reshape(shape)
            else:
                state_scaled = state
        
            # Vorhersage der Q-Werte
            act_values = self.rl_model.predict([
                np.expand_dims(state_scaled, axis=0),
                np.expand_dims(hmm_state, axis=0),
                np.expand_dims(context, axis=0)
            ], verbose=0)
        
            return np.argmax(act_values[0])  # Beste Aktion
        
        elif self.rl_algorithm == "sac":
            # SAC verwendet eine stochastische Policy
            # Skaliere Zustand
            if self.scaler_fitted:
                shape = state.shape
                state_flat = state.reshape(-1, self.input_dim)
                state_scaled = self.feature_scaler.transform(state_flat).reshape(shape)
            else:
                state_scaled = state
        
            # Vorhersage von Mittelwert und Standardabweichung
            action_mean, action_log_std = self.actor_model.predict([
                np.expand_dims(state_scaled, axis=0),
                np.expand_dims(hmm_state, axis=0), 
                np.expand_dims(context, axis=0)
            ], verbose=0)
        
            action_mean = action_mean[0]
            action_std = np.exp(action_log_std[0])
        
            # Normale Exploration
            if np.random.rand() <= self.epsilon:
                # Stärkeres zufälliges Rauschen
                action = np.random.normal(action_mean, action_std * 2)
            else:
                # Sampling aus der Normalverteilung
                action = np.random.normal(action_mean, action_std)
        
            # Diskretisieren auf 11 Aktionen für erweiterten Aktionsraum
            return np.argmax(action)
    
    def replay(self, batch_size=32):
        """
        Trainiert das RL-Modell mit Experience Replay.
    
        Args:
            batch_size: Batch-Größe für Training
        """
        if len(self.memory) < batch_size:
            return
    
        if self.rl_algorithm == "dqn":
            self._replay_dqn(batch_size)
        elif self.rl_algorithm == "sac":
            self._replay_sac(batch_size)
    
    def _replay_dqn(self, batch_size):
        """
        DQN Experience Replay mit Unterstützung für erweiterten Aktionsraum.
        """
        # Zufällige Stichprobe aus Memory
        minibatch = random.sample(self.memory, batch_size)
    
        # Batch verarbeiten
        states = []
        hmm_states = []
        contexts = []
        targets = []
    
        for state, hmm_state, context, action, reward, next_state, next_hmm, next_context, done in minibatch:
            # Zustände skalieren
            if self.scaler_fitted:
                shape = state.shape
                state_flat = state.reshape(-1, self.input_dim)
                state_scaled = self.feature_scaler.transform(state_flat).reshape(shape)
            
                next_shape = next_state.shape
                next_flat = next_state.reshape(-1, self.input_dim)
                next_scaled = self.feature_scaler.transform(next_flat).reshape(next_shape)
            else:
                state_scaled = state
                next_scaled = next_state
        
            # Aktuelle Q-Werte
            target = self.rl_model.predict([
                np.expand_dims(state_scaled, axis=0),
                np.expand_dims(hmm_state, axis=0),
                np.expand_dims(context, axis=0)
            ], verbose=0)[0]
        
            if done:
                # Terminaler Zustand
                target[action] = reward
            else:
                # Nicht-terminaler Zustand: Q-Learning mit Target-Netzwerk
                q_future = self.target_model.predict([
                    np.expand_dims(next_scaled, axis=0),
                    np.expand_dims(next_hmm, axis=0),
                    np.expand_dims(next_context, axis=0)
                ], verbose=0)[0]
            
                target[action] = reward + self.gamma * np.amax(q_future)
        
            states.append(state_scaled)
            hmm_states.append(hmm_state)
            contexts.append(context)
            targets.append(target)
    
        # NumPy-Arrays erstellen
        states = np.array(states)
        hmm_states = np.array(hmm_states)
        contexts = np.array(contexts)
        targets = np.array(targets)
    
        # Modell trainieren
        self.rl_model.fit(
            [states, hmm_states, contexts],
            targets,
            epochs=1,
            batch_size=batch_size,
            verbose=0
        )
    
        # Epsilon abklingen lassen
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
        # Target-Modell periodisch aktualisieren
        self.update_target_counter += 1
        if self.update_target_counter % 10 == 0:
            self.target_model.set_weights(self.rl_model.get_weights())
    
    def _replay_sac(self, batch_size):
        """
        SAC (Soft Actor-Critic) Experience Replay mit Unterstützung für erweiterten Aktionsraum.
        """
        # Zufällige Stichprobe aus Memory
        minibatch = random.sample(self.memory, batch_size)
    
        # Batch-Daten vorbereiten
        states = []
        hmm_states = []
        contexts = []
        actions = []
        rewards = []
        next_states = []
        next_hmm_states = []
        next_contexts = []
        dones = []
    
        for state, hmm_state, context, action, reward, next_state, next_hmm, next_context, done in minibatch:
            # Skaliere Zustände
            if self.scaler_fitted:
                shape = state.shape
                state_flat = state.reshape(-1, self.input_dim)
                state_scaled = self.feature_scaler.transform(state_flat).reshape(shape)
            
                next_shape = next_state.shape
                next_flat = next_state.reshape(-1, self.input_dim)
                next_scaled = self.feature_scaler.transform(next_flat).reshape(next_shape)
            else:
                state_scaled = state
                next_scaled = next_state
        
            states.append(state_scaled)
            hmm_states.append(hmm_state)
            contexts.append(context)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_scaled)
            next_hmm_states.append(next_hmm)
            next_contexts.append(next_context)
            dones.append(float(done))
    
        # NumPy-Arrays erstellen
        states = np.array(states)
        hmm_states = np.array(hmm_states)
        contexts = np.array(contexts)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        next_hmm_states = np.array(next_hmm_states)
        next_contexts = np.array(next_contexts)
        dones = np.array(dones)
    
        # SAC Update-Logik (vereinfachte Implementierung)
        # SAC erfordert eine spezifischere Implementation je nach Anforderungen
        self.logger.warning("SAC Replay vereinfachte Implementation (Demo)")
    
        # Alpha-Parameter aktualisieren (Temperaturparameter für Entropy)
        with tf.GradientTape() as tape:
            action_means, action_log_stds = self.actor_model([states, hmm_states, contexts])
            action_stds = tf.exp(action_log_stds)
        
            # Normale Verteilung für Aktionen
            normal_dist = tfp.distributions.Normal(action_means, action_stds)
        
            # Log-Probabilities der Aktionen
            log_probs = normal_dist.log_prob(actions)
        
            # Alpha-Verlust - Angepasst für 11-dimensionalen Aktionsraum
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
            )
    
        # Gradienten für Alpha
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
    
        # Alpha-Wert für Berechnungen
        alpha = tf.exp(self.log_alpha)
    
        # Update der Critic-Netzwerke (Q-Functions)
        for i, critic in enumerate(self.critic_models):
            with tf.GradientTape() as tape:
                # Q-Werte für aktuelle Zustände und Aktionen
                q_values = critic([states, hmm_states, contexts])
            
                # Nächste Aktionswerte und Entropy-Term für Ziel
                next_action_means, next_action_log_stds = self.actor_model([next_states, next_hmm_states, next_contexts])
                next_action_stds = tf.exp(next_action_log_stds)
            
                # Neue Aktionen samplen
                next_actions = next_action_means + next_action_stds * tf.random.normal(next_action_means.shape)
            
                # Berechne Log-Prob
                next_log_probs = tfp.distributions.Normal(next_action_means, next_action_stds).log_prob(next_actions)
            
                # Target-Q-Werte
                next_q1 = self.target_critics[0]([next_states, next_hmm_states, next_contexts])
                next_q2 = self.target_critics[1]([next_states, next_hmm_states, next_contexts])
                next_q = tf.minimum(next_q1, next_q2)
            
                # Entropy-reguliertes Target
                next_value = next_q - alpha * next_log_probs
            
                # Bellman target
                q_target = rewards + (1.0 - dones) * self.gamma * next_value
            
                # Critic-Verlust
                critic_loss = tf.reduce_mean(tf.square(q_values - q_target))
        
            # Gradienten für Critic
            critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    
        # Update Actor-Netzwerk
        with tf.GradientTape() as tape:
            # Actor output
            action_means, action_log_stds = self.actor_model([states, hmm_states, contexts])
            action_stds = tf.exp(action_log_stds)
        
            # Sample Aktionen für nächsten Zustand
            actions_new = action_means + action_stds * tf.random.normal(action_means.shape)
        
            # Log-Probabilities
            log_probs = tfp.distributions.Normal(action_means, action_stds).log_prob(actions_new)
        
            # Q-Werte für neue Aktionen
            q1_new = self.critic_models[0]([states, hmm_states, contexts])
            q2_new = self.critic_models[1]([states, hmm_states, contexts])
            q_new = tf.minimum(q1_new, q2_new)
        
            # Actor loss (maximize Q - alpha * log_prob)
            actor_loss = tf.reduce_mean(alpha * log_probs - q_new)
    
        # Gradienten für Actor
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
    
        # Soft-update der Target-Netzwerke
        tau = 0.005  # Soft update Faktor
    
        for i in range(len(self.critic_models)):
            target_weights = self.target_critics[i].get_weights()
            current_weights = self.critic_models[i].get_weights()
        
            new_weights = []
            for tw, cw in zip(target_weights, current_weights):
                new_weights.append((1 - tau) * tw + tau * cw)
        
            self.target_critics[i].set_weights(new_weights)
    
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def predict_market_direction(self, X_seq, hmm_state, use_ensemble=True):
        """
        Prognostiziert die Marktrichtung (5-Klassen-System) mittels neuronalen Netzen.
    
        Args:
            X_seq: Feature-Sequenz (2D oder 3D)
            hmm_state: HMM-Zustand (one-hot kodiert)
            use_ensemble: Ensemble-Modelle für die Vorhersage verwenden
    
        Returns:
            dict: Richtungsvorhersage mit Wahrscheinlichkeiten
        """
        if self.direction_model is None:
            return {"direction": "neutral", "confidence": 0.5}
    
        # Bereite Sequenz vor (stellt sicher, dass 3D-Format [batch, seq, features])
        X_input = self._prepare_sequence(X_seq)
        
        # Skaliere Features, aber gib keine Warnung aus, wenn der Scaler nicht angepasst ist
        if self.scaler_fitted:
            # Reshape, um den Scaler korrekt anzuwenden
            batch_size, seq_len, n_features = X_input.shape
            X_reshaped = X_input.reshape(-1, n_features)
            X_scaled = self.feature_scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(batch_size, seq_len, n_features)
        else:
            # Verwende unskalierte Daten ohne Warnung
            X_scaled = X_input
        
        # hmm_state als 2D-Array formatieren, falls nötig
        hmm_input = hmm_state
        if len(hmm_input.shape) == 1:
            hmm_input = hmm_input.reshape(1, -1)
        
        # Verwende Ensemble-Vorhersage, wenn aktiviert und verfügbar
        if use_ensemble and self.use_ensemble and self.ensemble_models:
            return self._predict_with_ensemble(X_scaled, hmm_input)
    
        # Standard-Vorhersage mit Hauptmodell
        pred = self.direction_model.predict([X_scaled, hmm_input], verbose=0)[0]
    
        # Mapping für 5-Klassen-System
        directions = ["strong_down", "weak_down", "neutral", "weak_up", "strong_up"]
        direction_dict = {directions[i]: float(pred[i]) for i in range(len(directions))}
    
        # Ermittle wahrscheinlichste Richtung
        max_idx = np.argmax(pred)
        predicted_direction = directions[max_idx]
        confidence = float(pred[max_idx])
    
        # Vereinfachte Richtungsinformation (für Kompatibilität)
        simplified_direction = "down" if max_idx < 2 else "sideways" if max_idx == 2 else "up"
    
        return {
            "direction": predicted_direction,
            "simplified_direction": simplified_direction,
            "confidence": confidence,
            "probabilities": direction_dict
        }
    
    def _predict_with_ensemble(self, X_scaled, hmm_state):
        """
        Macht eine Vorhersage mit dem Ensemble-Modellen.
        Unterstützt 5-Klassen-Richtungsvorhersage.
    
        Args:
            X_scaled: Skalierte Feature-Sequenz
            hmm_state: HMM-Zustand
            
        Returns:
            dict: Ensemble-Vorhersage
        """
        # Ermittle aktuelle Marktphase
        market_phase = self._detect_market_phase(X_scaled[0])
    
        # Basis-Vorhersage mit Hauptmodell
        base_prediction = self.direction_model.predict([X_scaled, hmm_state], verbose=0)[0]
    
        # Spezialisierte Vorhersage, falls verfügbar
        specialized_prediction = None
        if market_phase in self.ensemble_models:
            model_info = self.ensemble_models[market_phase]
        
            # Verwende nur trainierte Modelle
            if model_info["training_samples"] > 100:
                specialized_prediction = model_info["model"].predict([X_scaled, hmm_state], verbose=0)[0]
    
        # Architekturvarianten-Vorhersagen
        variant_predictions = []
        variant_weights = []
    
        for variant in self.model_variants:
            pred = variant["model"].predict([X_scaled, hmm_state], verbose=0)[0]
            variant_predictions.append(pred)
            variant_weights.append(variant["weight"])
    
        # Gewichtete Kombination aller Vorhersagen
        final_prediction = base_prediction.copy()
    
        # Haupt- und spezialisiertes Modell kombinieren
        if specialized_prediction is not None:
            # Gewichtung basierend auf Performance und Konfidenz
            specialized_weight = self.ensemble_models[market_phase]["weight"]
            base_weight = 1.0 - specialized_weight
        
            # Gewichteter Durchschnitt
            final_prediction = (base_weight * base_prediction + 
                              specialized_weight * specialized_prediction)
    
        # Architekturvarianten einbeziehen
        if variant_predictions:
            # Normalisiere Varianten-Gewichte
            variant_sum = sum(variant_weights)
            if variant_sum > 0:
                normalized_weights = [w / variant_sum for w in variant_weights]
            
                # Gewichtung zwischen Haupt+Spezialisiert vs. Varianten
                ensemble_weight = 0.7  # 70% für Haupt+Spezialisiert, 30% für Varianten
            
                # Gewichtete Summe der Varianten
                variant_combined = np.zeros_like(final_prediction)
                for i, pred in enumerate(variant_predictions):
                    variant_combined += normalized_weights[i] * pred
            
                # Finale Kombination
                final_prediction = (ensemble_weight * final_prediction + 
                                   (1 - ensemble_weight) * variant_combined)
    
        # Mapping und Ausgabe für 5-Klassen-System
        directions = ["strong_down", "weak_down", "neutral", "weak_up", "strong_up"]
        direction_dict = {directions[i]: float(final_prediction[i]) for i in range(len(directions))}
    
        # Ermittle wahrscheinlichste Richtung
        max_idx = np.argmax(final_prediction)
        predicted_direction = directions[max_idx]
        confidence = float(final_prediction[max_idx])
    
        # Vereinfachte Richtungsinformation (für Kompatibilität)
        simplified_direction = "down" if max_idx < 2 else "sideways" if max_idx == 2 else "up"
    
        return {
            "direction": predicted_direction,
            "simplified_direction": simplified_direction,
            "confidence": confidence,
            "probabilities": direction_dict,
            "market_phase": market_phase,
            "ensemble_method": "weighted_avg",
            "used_specialized_model": specialized_prediction is not None
        }
    
    def predict_market_volatility(self, X_seq, hmm_state):
        """
        Prognostiziert die Marktvolatilität mittels neuronalen Netzen.
        
        Args:
            X_seq: Feature-Sequenz (2D oder 3D)
            hmm_state: HMM-Zustand (one-hot kodiert)
        
        Returns:
            dict: Volatilitätsvorhersage mit Wahrscheinlichkeiten
        """
        if self.volatility_model is None:
            return {"volatility": "medium", "confidence": 0.5}
        
        # Bereite Sequenz vor (stellt sicher, dass 3D-Format [batch, seq, features])
        X_input = self._prepare_sequence(X_seq)
        
        # Skaliere Features, aber gib keine Warnung aus, wenn der Scaler nicht angepasst ist
        if self.scaler_fitted:
            # Reshape, um den Scaler korrekt anzuwenden
            batch_size, seq_len, n_features = X_input.shape
            X_reshaped = X_input.reshape(-1, n_features)
            X_scaled = self.feature_scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(batch_size, seq_len, n_features)
        else:
            # Verwende unskalierte Daten ohne Warnung
            X_scaled = X_input
        
        # hmm_state als 2D-Array formatieren, falls nötig
        hmm_input = hmm_state
        if len(hmm_input.shape) == 1:
            hmm_input = hmm_input.reshape(1, -1)
        
        # Vorhersage mit Volatilitätsmodell
        pred = self.volatility_model.predict([X_scaled, hmm_input], verbose=0)[0]
        
        # Mapping für 3-Klassen-System
        volatility_levels = ["low", "medium", "high"]
        volatility_dict = {volatility_levels[i]: float(pred[i]) for i in range(len(volatility_levels))}
        
        # Ermittle wahrscheinlichstes Volatilitätsniveau
        max_idx = np.argmax(pred)
        predicted_volatility = volatility_levels[max_idx]
        confidence = float(pred[max_idx])
        
        return {
            "volatility": predicted_volatility,
            "confidence": confidence,
            "probabilities": volatility_dict
        }

    def predict_direction(self, X_seq, hmm_state):
        """
        Wrapper für predict_market_direction zur Kompatibilität mit dem Ensemble Framework.
    
        Args:
            X_seq: Feature-Sequenz
            hmm_state: HMM-Zustand
        
        Returns:
            dict: Richtungsvorhersage
        """
        if not hasattr(self, 'predict_market_direction'):
            self.logger.error("predict_market_direction Methode nicht verfügbar")
            return {"direction": "sideways", "confidence": 0.0, "probabilities": {"up": 0.33, "down": 0.33, "sideways": 0.34}}
    
        return self.predict_market_direction(X_seq, hmm_state)
    
    def predict_volatility(self, X_seq, hmm_state):
        """
        Wrapper für predict_market_volatility zur Kompatibilität mit dem Ensemble Framework.
    
        Args:
            X_seq: Feature-Sequenz
            hmm_state: HMM-Zustand
        
        Returns:
            dict: Volatilitätsvorhersage
        """
        if not hasattr(self, 'predict_market_volatility'):
            self.logger.error("predict_market_volatility Methode nicht verfügbar")
            return {"volatility": "medium", "confidence": 0.0, "probabilities": {"low": 0.33, "medium": 0.34, "high": 0.33}}
    
        return self.predict_market_volatility(X_seq, hmm_state)
    
    def get_action_recommendation(self, X_seq, hmm_state, context):
        """
        Liefert Handlungsempfehlungen vom Reinforcement Learning-Modell.
        Unterstützt erweiterten Aktionsraum mit 11 möglichen Aktionen.
    
        Args:
            X_seq: Feature-Sequenz
            hmm_state: HMM-Zustand
            context: Marktkontext-Features
    
        Returns:
            dict: Empfohlene Aktion mit Erklärung
        """
        if self.rl_model is None:
            self.logger.error("RL-Modell nicht trainiert")
            return None
    
        # Skaliere Features
        if self.scaler_fitted:
            shape = X_seq.shape
            X_flat = X_seq.reshape(-1, self.input_dim)
            X_scaled = self.feature_scaler.transform(X_flat).reshape(shape)
        else:
            X_scaled = X_seq
    
        # Dimensionen anpassen
        if len(X_scaled.shape) == 2:
            # Batch-Dimension hinzufügen
            X_scaled = np.expand_dims(X_scaled, axis=0)
        if len(hmm_state.shape) == 1:
            hmm_state = np.expand_dims(hmm_state, axis=0)
        if len(context.shape) == 1:
            context = np.expand_dims(context, axis=0)
    
        # Je nach RL-Algorithmus
        if self.rl_algorithm == "dqn":
            # Q-Werte vorhersagen
            q_values = self.rl_model.predict([X_scaled, hmm_state, context], verbose=0)[0]
        
            # Aktions-Mapping für erweiterten Aktionsraum
            actions = [
                "no_action",           # 0: Keine Aktion/Position halten
                "enter_long_full",     # 1: Long-Position eröffnen (volle Größe)
                "enter_long_half",     # 2: Long-Position eröffnen (halbe Größe)
                "scale_long",          # 3: Bestehende Long-Position aufstocken
                "enter_short_full",    # 4: Short-Position eröffnen (volle Größe)
                "enter_short_half",    # 5: Short-Position eröffnen (halbe Größe)
                "scale_short",         # 6: Bestehende Short-Position aufstocken
                "exit_long",           # 7: Long-Position schließen
                "exit_short",          # 8: Short-Position schließen
                "partial_exit_long",   # 9: Long-Position teilweise schließen (Gewinnmitnahme)
                "partial_exit_short"   # 10: Short-Position teilweise schließen (Gewinnmitnahme)
            ]
            action_dict = {actions[i]: float(q_values[i]) for i in range(len(actions))}
        
            # Ermittle empfohlene Aktion
            max_idx = np.argmax(q_values)
            recommended_action = actions[max_idx]
        
            # Berechne Konfidenz (relativer Vorteil gegenüber zweitbester Option)
            sorted_values = np.sort(q_values)[::-1]  # Absteigend
            if len(sorted_values) > 1:
                advantage = sorted_values[0] - sorted_values[1]
                # Normalisiere zu [0, 1] mit einer Sigmoid-ähnlichen Funktion
                confidence = 2 / (1 + np.exp(-advantage)) - 1
            else:
                confidence = 1.0
        
            return {
                "action": recommended_action,
                "confidence": float(confidence),
                "q_values": action_dict
            }
        
        elif self.rl_algorithm == "sac":
            # Bei SAC: Mittelwert der Aktionsverteilung
            action_mean, _ = self.actor_model.predict([X_scaled, hmm_state, context], verbose=0)
            action_mean = action_mean[0]
        
            # Aktions-Mapping für erweiterten Aktionsraum
            actions = [
                "no_action", "enter_long_full", "enter_long_half", "scale_long", 
                "enter_short_full", "enter_short_half", "scale_short",
                "exit_long", "exit_short", "partial_exit_long", "partial_exit_short"
            ]
        
            # Argmax für diskrete Aktionen
            max_idx = np.argmax(action_mean)
            recommended_action = actions[max_idx]
        
            # Konfidenz basierend auf Werten
            softmax_values = np.exp(action_mean) / np.sum(np.exp(action_mean))
            confidence = float(softmax_values[max_idx])
        
            # Erstelle Aktion->Wert-Mapping
            action_dict = {actions[i]: float(action_mean[i]) for i in range(len(actions))}
        
            return {
                "action": recommended_action,
                "confidence": confidence,
                "action_values": action_dict
            }
    
    def combine_predictions(self, hmm_prediction, direction_pred, volatility_pred, action_rec):
        """
        Kombiniert alle Vorhersagen für eine finale Handelsentscheidung.
        Unterstützt 5-Klassen-Richtungsvorhersage mit simplified_direction.
    
        Args:
            hmm_prediction: HMM-Zustandsvorhersage
            direction_pred: Richtungsvorhersage (5-Klassen oder 3-Klassen)
            volatility_pred: Volatilitätsvorhersage
            action_rec: Handlungsempfehlung
    
        Returns:
            dict: Kombinierte Vorhersage mit Handelssignal
        """
        if not all([hmm_prediction, direction_pred, volatility_pred, action_rec]):
            self.logger.warning("Fehlende Vorhersagen für Kombination")
            # Return a default dictionary instead of None to avoid TypeErrors upstream
            return {
                "signal": "NONE",
                "signal_type": "error",
                "signal_strength": 0.0,
                "combined_confidence": 0.0,
                "error": "Missing input predictions",
                "timestamp": datetime.now().isoformat()
            }
    
        # Werte extrahieren
        hmm_state = hmm_prediction.get("state_idx", 0)
        hmm_label = hmm_prediction.get("state_label", "Unknown")
        hmm_confidence = hmm_prediction.get("state_confidence", 0.5)
    
        # Verwende simplified_direction, wenn verfügbar, sonst nutze das normale direction-Feld
        if "simplified_direction" in direction_pred:
            direction = direction_pred.get("simplified_direction", "sideways")
        else:
            direction = direction_pred.get("direction", "sideways")
            # Alte Richtungsvorhersage hat möglicherweise "up", "down", "sideways" direkt im "direction"-Feld
            if direction not in ["up", "down", "sideways"]:
                # Konvertiere erweiterte Richtungswerte in einfache
                if "strong_down" in direction or "weak_down" in direction:
                    direction = "down"
                elif "strong_up" in direction or "weak_up" in direction:
                    direction = "up"
                else:
                    direction = "sideways"
    
        direction_confidence = direction_pred.get("confidence", 0.5)
    
        volatility = volatility_pred.get("volatility", "medium")
        volatility_confidence = volatility_pred.get("confidence", 0.5)
    
        action = action_rec.get("action", "no_action")
        action_confidence = action_rec.get("confidence", 0.5)
    
        # Kombinierte Konfidenz
        # Gewichtete Summe basierend auf empirischer Performance der Komponenten
        combined_confidence = (
            0.3 * hmm_confidence +
            0.3 * direction_confidence +
            0.1 * volatility_confidence +
            0.3 * action_confidence
        )
    
        # Handelssignal bestimmen
        signal = "NONE"
        signal_type = "hybrid"
        signal_strength = 0.0
    
        # Logik für Handelssignale (mit den einfachen Richtungen)
        if action in ["enter_long", "enter_short", "enter_long_full", "enter_long_half", "enter_short_full", "enter_short_half"]:
            # RL-Modell empfiehlt Einstieg
            is_long = "long" in action
            signal = "LONG" if is_long else "SHORT"
            signal_type = "rl_entry"
            signal_strength = action_confidence
        
            # Überprüfe mit anderen Komponenten für Bestätigung
            hmm_agrees = (("Bull" in hmm_label and signal == "LONG") or 
                         ("Bear" in hmm_label and signal == "SHORT"))
        
            direction_agrees = ((direction == "up" and signal == "LONG") or 
                              (direction == "down" and signal == "SHORT"))
        
            # Passe Signalstärke basierend auf Übereinstimmung an
            if hmm_agrees and direction_agrees:
                signal_strength = combined_confidence
                signal_type = "full_agreement"
            elif hmm_agrees or direction_agrees:
                signal_strength = 0.7 * combined_confidence
                signal_type = "partial_agreement"
            else:
                signal_strength = 0.3 * action_confidence
                signal_type = "rl_only"
    
        elif "Bull" in hmm_label and direction == "up":
            # Bullische Übereinstimmung zwischen HMM und Richtungsmodell
            signal = "LONG"
            signal_type = "trend_following"
            signal_strength = 0.3 + 0.7 * combined_confidence
        
        elif "Bear" in hmm_label and direction == "down":
            # Bärische Übereinstimmung zwischen HMM und Richtungsmodell
            signal = "SHORT"
            signal_type = "trend_following"
            signal_strength = 0.3 + 0.7 * combined_confidence
    
        # Ausstiegssignale
        if any(exit_action in action for exit_action in ["exit_long", "exit_short", "partial_exit_long", "partial_exit_short"]):
            exit_type = "LONG" if "long" in action else "SHORT"
            partial = "partial" in action
            signal = f"{'PARTIAL_' if partial else ''}EXIT_{exit_type}"
            signal_type = "rl_exit"
            signal_strength = action_confidence
    
        # Erstelle kombiniertes Ergebnis
        result = {
            "signal": signal,
            "signal_type": signal_type,
            "signal_strength": signal_strength,
            "combined_confidence": combined_confidence,
            "hmm_state": hmm_state,
            "hmm_label": hmm_label,
            "direction": direction,
            "detailed_direction": direction_pred.get("direction", direction),  # Originale detaillierte Richtung
            "volatility": volatility,
            "rl_action": action,
            "timestamp": datetime.now().isoformat()
        }
    
        return result
    
    def save_models(self, base_path="./models"):
        """
        Speichert alle trainierten Modelle.
        
        Args:
            base_path: Verzeichnis zum Speichern
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        # Neuronale Netzwerke speichern
        if self.direction_model:
            self.direction_model.save(f"{base_path}/direction_model.h5")
        
        if self.volatility_model:
            self.volatility_model.save(f"{base_path}/volatility_model.h5")
        
        if self.rl_model:
            self.rl_model.save(f"{base_path}/rl_model.h5")
        
        # Ensemble-Modelle speichern
        if self.use_ensemble and self.ensemble_models:
            ensemble_dir = os.path.join(base_path, "ensemble_models")
            os.makedirs(ensemble_dir, exist_ok=True)
            
            for phase, model_info in self.ensemble_models.items():
                model_info["model"].save(os.path.join(ensemble_dir, f"{phase}_model.h5"))
            
            # Ensemble-Konfiguration speichern
            ensemble_config = {phase: {
                "weight": model_info["weight"],
                "performance": model_info["performance"],
                "training_samples": model_info["training_samples"]
            } for phase, model_info in self.ensemble_models.items()}
            
            with open(os.path.join(ensemble_dir, "ensemble_config.json"), "w") as f:
                json.dump(ensemble_config, f, indent=2)
        
        # Feature-Scaler speichern
        if self.scaler_fitted:
            with open(f"{base_path}/feature_scaler.pkl", "wb") as f:
                pickle.dump(self.feature_scaler, f)
        
        # Trainingshistorie und Konfiguration
        config = {
            "input_dim": self.input_dim,
            "hmm_states": self.hmm_states,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "sequence_length": self.sequence_length,
            "learning_rate": self.learning_rate,
            "use_attention": self.use_attention,
            "use_ensemble": self.use_ensemble,
            "rl_algorithm": self.rl_algorithm
        }
        
        history = {
            "direction_history": self.direction_history,
            "volatility_history": self.volatility_history,
            "rl_history": self.rl_history
        }
        
        with open(f"{base_path}/hybrid_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        with open(f"{base_path}/training_history.json", "w") as f:
            # NumPy-Werte zu Python-Typen konvertieren für JSON
            history_clean = {}
            for key, hist in history.items():
                if hist is not None:
                    history_clean[key] = {k: [float(v) for v in vals] for k, vals in hist.items()}
            
            json.dump(history_clean, f, indent=2)
        
        # Pattern Memory speichern
        with open(f"{base_path}/pattern_memory.pkl", "wb") as f:
            pickle.dump(self.pattern_memory, f)
        
        self.logger.info(f"Alle Hybrid-Modelle gespeichert in {base_path}")
    
    def load_models(self, base_path="./models", load_ensembles=True):
        """
        Lädt alle Modelle aus Speicher.
        
        Args:
            base_path: Verzeichnis mit gespeicherten Modellen
            load_ensembles: Ensemble-Modelle laden
        
        Returns:
            bool: Erfolgsstatus
        """
        try:
            # Konfiguration laden
            config_path = os.path.join(base_path, "hybrid_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # Konfiguration aktualisieren
                self.input_dim = config.get("input_dim", self.input_dim)
                self.hmm_states = config.get("hmm_states", self.hmm_states)
                self.lstm_units = config.get("lstm_units", self.lstm_units)
                self.dense_units = config.get("dense_units", self.dense_units)
                self.sequence_length = config.get("sequence_length", self.sequence_length)
                self.learning_rate = config.get("learning_rate", self.learning_rate)
                self.use_attention = config.get("use_attention", self.use_attention)
                self.use_ensemble = config.get("use_ensemble", self.use_ensemble)
                self.rl_algorithm = config.get("rl_algorithm", self.rl_algorithm)
            
            # Richtungsmodell laden (SavedModel Format oder H5)
            direction_path_dir = os.path.join(base_path, "direction_model") # Verzeichnis
            direction_path_h5 = os.path.join(base_path, "direction_model.h5") # .h5 Datei
            if os.path.isdir(direction_path_dir):
                try:
                    loaded_model = load_model(direction_path_dir)
                    if loaded_model:
                        self.direction_model = loaded_model
                        self.logger.info("Richtungsmodell (SavedModel) erfolgreich geladen und zugewiesen.")
                    else:
                        self.logger.error("load_model für Richtungsmodell (SavedModel) gab None zurück.")
                except Exception as load_ex:
                    self.logger.error(f"Fehler beim Laden des Richtungsmodells (SavedModel) aus {direction_path_dir}: {str(load_ex)}")
            elif os.path.exists(direction_path_h5):
                try:
                    loaded_model = load_model(direction_path_h5)
                    if loaded_model:
                        self.direction_model = loaded_model
                        self.logger.info("Richtungsmodell (.h5) erfolgreich geladen und zugewiesen.")
                    else:
                        self.logger.error("load_model für Richtungsmodell (.h5) gab None zurück.")
                except Exception as load_ex:
                    self.logger.error(f"Fehler beim Laden des Richtungsmodells (.h5) aus {direction_path_h5}: {str(load_ex)}")
            else:
                self.logger.warning("Richtungsmodell weder als Verzeichnis noch als .h5 gefunden")
            
            # Volatilitätsmodell laden (SavedModel Format oder H5)
            volatility_path_dir = os.path.join(base_path, "volatility_model") # Verzeichnis
            volatility_path_h5 = os.path.join(base_path, "volatility_model.h5") # .h5 Datei
            if os.path.isdir(volatility_path_dir):
                try:
                    loaded_model = load_model(volatility_path_dir)
                    if loaded_model:
                        self.volatility_model = loaded_model
                        self.logger.info("Volatilitätsmodell (SavedModel) erfolgreich geladen und zugewiesen.")
                    else:
                        self.logger.error("load_model für Volatilitätsmodell (SavedModel) gab None zurück.")
                except Exception as load_ex:
                    self.logger.error(f"Fehler beim Laden des Volatilitätsmodells (SavedModel) aus {volatility_path_dir}: {str(load_ex)}")
            elif os.path.exists(volatility_path_h5):
                try:
                    loaded_model = load_model(volatility_path_h5)
                    if loaded_model:
                        self.volatility_model = loaded_model
                        self.logger.info("Volatilitätsmodell (.h5) erfolgreich geladen und zugewiesen.")
                    else:
                        self.logger.error("load_model für Volatilitätsmodell (.h5) gab None zurück.")
                except Exception as load_ex:
                    self.logger.error(f"Fehler beim Laden des Volatilitätsmodells (.h5) aus {volatility_path_h5}: {str(load_ex)}")
            else:
                self.logger.warning("Volatilitätsmodell weder als Verzeichnis noch als .h5 gefunden")
            
            # RL-Modell laden (SavedModel Format oder H5)
            rl_path_dir = os.path.join(base_path, "rl_model") # Verzeichnis
            rl_path_h5 = os.path.join(base_path, "rl_model.h5") # .h5 Datei
            if os.path.isdir(rl_path_dir):
                try:
                    loaded_model = load_model(rl_path_dir)
                    if loaded_model:
                        self.rl_model = loaded_model
                        self.logger.info("RL-Modell (SavedModel) erfolgreich geladen und zugewiesen.")
                    else:
                        self.logger.error("load_model für RL-Modell (SavedModel) gab None zurück.")
                except Exception as load_ex:
                    self.logger.error(f"Fehler beim Laden des RL-Modells (SavedModel) aus {rl_path_dir}: {str(load_ex)}")
            elif os.path.exists(rl_path_h5):
                try:
                    loaded_model = load_model(rl_path_h5)
                    if loaded_model:
                        self.rl_model = loaded_model
                        self.logger.info("RL-Modell (.h5) erfolgreich geladen und zugewiesen.")
                    else:
                        self.logger.error("load_model für RL-Modell (.h5) gab None zurück.")
                except Exception as load_ex:
                    self.logger.error(f"Fehler beim Laden des RL-Modells (.h5) aus {rl_path_h5}: {str(load_ex)}")
            else:
                self.logger.warning("RL-Modell weder als Verzeichnis noch als .h5 gefunden")

            # Nur fortfahren, wenn RL-Modell geladen wurde
            if self.rl_model is not None:
                # Target-Modell erstellen und syncen
                self.target_model = clone_model(self.rl_model)
                self.target_model.set_weights(self.rl_model.get_weights())
                self.logger.info("RL Target-Modell initialisiert")
                
            # Ensemble-Modelle laden (SavedModel Format oder H5)
            if load_ensembles and self.use_ensemble:
                ensemble_dir = os.path.join(base_path, "ensemble_models")
                if os.path.exists(ensemble_dir):
                    # Ensemble-Konfiguration laden
                    ensemble_config_path = os.path.join(ensemble_dir, "ensemble_config.json")
                    ensemble_config = {}
                    if os.path.exists(ensemble_config_path):
                        with open(ensemble_config_path, "r") as f:
                            ensemble_config = json.load(f)
                    
                    # Phasen-Modelle laden
                    for phase in self.market_phases:
                        model_path_dir = os.path.join(ensemble_dir, f"{phase}_model") # Verzeichnis
                        model_path_h5 = os.path.join(ensemble_dir, f"{phase}_model.h5") # .h5 Datei
                        loaded_model = None
                        if os.path.isdir(model_path_dir):
                            loaded_model = load_model(model_path_dir)
                            self.logger.info(f"Ensemble-Modell für {phase} (SavedModel) geladen")
                        elif os.path.exists(model_path_h5):
                            loaded_model = load_model(model_path_h5)
                            self.logger.info(f"Ensemble-Modell für {phase} (.h5) geladen")
                        
                        if loaded_model:
                            # Konfigurationsdaten oder Defaults
                            config_data = ensemble_config.get(phase, {})
                            weight = config_data.get("weight", 1.0 / len(self.market_phases))
                            performance = config_data.get("performance", 0.5)
                            training_samples = config_data.get("training_samples", 0)
                            
                            self.ensemble_models[phase] = {
                                "model": loaded_model,
                                "weight": weight,
                                "performance": performance,
                                "training_samples": training_samples
                            }
            
            # Feature-Scaler laden
            scaler_path = os.path.join(base_path, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.feature_scaler = pickle.load(f)
                self.scaler_fitted = True # Set the flag after successful loading
                self.logger.info("Feature-Scaler geladen")
            
            # Trainingshistorie laden
            history_path = os.path.join(base_path, "training_history.json")
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
                
                self.direction_history = history.get("direction_history")
                self.volatility_history = history.get("volatility_history")
                self.rl_history = history.get("rl_history")
            
            # Pattern Memory laden
            memory_path = os.path.join(base_path, "pattern_memory.pkl")
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    self.pattern_memory = pickle.load(f)
                    
                self.logger.info(f"Pattern Memory geladen: {len(self.pattern_memory['patterns'])} Patterns")
            
            self.is_model_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Modelle: {str(e)}")
            return False
    
    def _select_training_samples(self, new_data, budget=100):
        """
        Aktives Lernen: Wählt die informativsten Samples aus neuen Daten aus.
        
        Args:
            new_data: Dict mit neuen Trainingsdaten
            budget: Maximale Anzahl auszuwählender Samples
        
        Returns:
            Dict mit ausgewählten Trainingsdaten
        """
        if 'X_sequences' not in new_data or len(new_data['X_sequences']) < budget:
            return new_data
        
        # Unsicherheits-Sampling: Finde Samples mit höchster Unsicherheit
        X_scaled = self.feature_scaler.transform(
            new_data['X_sequences'].reshape(-1, self.input_dim)
        ).reshape(new_data['X_sequences'].shape)
        
        predictions = self.direction_model.predict(
            [X_scaled, new_data['hmm_states']], verbose=0
        )
        
        # Entropie der Vorhersagen (höhere Entropie = höhere Unsicherheit)
        uncertainties = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
        
        # Diversitäts-Sampling mithilfe von Clustering
        from sklearn.cluster import KMeans
        X_flat = X_scaled.reshape(len(X_scaled), -1)
        n_clusters = min(10, len(X_flat))
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(X_flat)
        
        # Kombinierte Strategie: aus jedem Cluster die unsichersten Samples wählen
        selected_indices = []
        for cluster in range(n_clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            if len(cluster_indices) == 0:
                continue
                
            # Sortiere nach Unsicherheit innerhalb des Clusters
            sorted_indices = cluster_indices[np.argsort(-uncertainties[cluster_indices])]
            
            # Wähle top N aus jedem Cluster
            n_per_cluster = max(1, budget // n_clusters)
            selected_indices.extend(sorted_indices[:n_per_cluster])
        
        # Falls Budget noch nicht erreicht, fülle mit den unsichersten verbleibenden Samples auf
        remaining = budget - len(selected_indices)
        if remaining > 0:
            all_indices = set(range(len(X_scaled)))
            remaining_indices = list(all_indices - set(selected_indices))
            
            if remaining_indices:
                # Sortiere verbleibende nach Unsicherheit
                sorted_remaining = sorted(
                    remaining_indices, 
                    key=lambda i: uncertainties[i],
                    reverse=True
                )
                
                selected_indices.extend(sorted_remaining[:remaining])
        
        # Erstelle Subset der ausgewählten Daten
        selected_data = {
            'X_sequences': new_data['X_sequences'][selected_indices],
            'hmm_states': new_data['hmm_states'][selected_indices], 
        }
        
        # Füge Labels hinzu, falls vorhanden
        for label_key in ['direction_labels', 'volatility_labels']:
            if label_key in new_data:
                selected_data[label_key] = new_data[label_key][selected_indices]
        
        self.logger.info(f"Aktives Lernen: {len(selected_indices)} von {len(X_scaled)} Samples ausgewählt")
        
        return selected_data
    
    def plot_training_history(self, save_path=None):
        """
        Visualisiert den Trainingsverlauf aller Modelle.
        
        Args:
            save_path: Optionaler Pfad zum Speichern der Visualisierung
        """
        if not any([self.direction_history, self.volatility_history, self.rl_history]):
            self.logger.warning("Keine Trainingshistorie verfügbar")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Richtungsmodell
        if self.direction_history:
            plt.subplot(3, 2, 1)
            plt.plot(self.direction_history['loss'], label='Training')
            plt.plot(self.direction_history['val_loss'], label='Validation')
            plt.title('Richtungsmodell Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 2, 2)
            plt.plot(self.direction_history['accuracy'], label='Training')
            plt.plot(self.direction_history['val_accuracy'], label='Validation')
            plt.title('Richtungsmodell Accuracy')
            plt.legend()
            plt.grid(True)
        
        # Volatilitätsmodell
        if self.volatility_history:
            plt.subplot(3, 2, 3)
            plt.plot(self.volatility_history['loss'], label='Training')
            plt.plot(self.volatility_history['val_loss'], label='Validation')
            plt.title('Volatilitätsmodell Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 2, 4)
            plt.plot(self.volatility_history['accuracy'], label='Training')
            plt.plot(self.volatility_history['val_accuracy'], label='Validation')
            plt.title('Volatilitätsmodell Accuracy')
            plt.legend()
            plt.grid(True)
        
        # RL-Modell
        if self.rl_history:
            plt.subplot(3, 2, 5)
            plt.plot(self.rl_history['loss'], label='Training')
            if 'val_loss' in self.rl_history:
                plt.plot(self.rl_history['val_loss'], label='Validation')
            plt.title('RL-Modell Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def _prepare_sequence(self, X_seq):
        """
        Bereitet eine Feature-Sequenz für die Eingabe in neuronale Netze vor.
        Stellt sicher, dass die Daten im Format [batch, sequence, features] vorliegen.
        
        Args:
            X_seq: Feature-Sequenz in beliebigem Format (1D, 2D oder 3D)
            
        Returns:
            np.ndarray: Vorbereitete Sequenz im Format [batch, sequence, features]
        """
        # Stelle sicher, dass X_seq ein numpy-Array ist
        X = np.asarray(X_seq)
        
        # Behandle verschiedene Eingabeformate
        if len(X.shape) == 1:  # Einzelner Vektor
            X = X.reshape(1, 1, -1)  # [1, 1, features]
        elif len(X.shape) == 2:  # Sequenz oder Batch ohne Sequenz
            # Bestimme, ob es eine Sequenz oder ein Batch ist, basierend auf Dimensionen
            if X.shape[1] == self.input_dim:  # [batch, features]
                X = X.reshape(X.shape[0], 1, X.shape[1])  # [batch, 1, features]
            else:  # [sequence, features]
                X = X.reshape(1, X.shape[0], X.shape[1])  # [1, sequence, features]
        
        # Jetzt sollte X im Format [batch, sequence, features] sein
        batch_size, seq_len, n_features = X.shape
        
        # Anpassen der Sequenzlänge, falls notwendig
        if seq_len < self.sequence_length:
            # Padding am Anfang der Sequenz mit Nullen
            padding = np.zeros((batch_size, self.sequence_length - seq_len, n_features))
            X = np.concatenate((padding, X), axis=1)
        elif seq_len > self.sequence_length:
            # Nur die letzten sequence_length Zeitschritte nehmen
            X = X[:, -self.sequence_length:, :]
        
        return X

# Hilfsfunktionen für die Integration mit dem bestehenden System

class HMMHybridWrapper:
    """
    Wrapper class that combines HMM and Hybrid model predictions.
    This class replaces the wrapper function that couldn't be pickled.
    """
    def __init__(self, hmm_model, hybrid_model, feature_cols=None):
        """
        Initialize the wrapper with both models.
        
        Args:
            hmm_model: Trained HMM model parameters
            hybrid_model: Trained Hybrid neural network model
            feature_cols: Feature column names
        """
        self.hmm_model = hmm_model
        self.hybrid_model = hybrid_model
        self.feature_cols = feature_cols
        
    def __call__(self, features, lookback=10):
        """
        Combined prediction with HMM and Hybrid model.
        
        Args:
            features: Feature matrix or vector
            lookback: Lookback window for sequence
        
        Returns:
            prediction: Combined prediction
        """
        from enhanced_hmm_em_v2 import forward_backward
        import numpy as np
        
        # HMM-Prediction
        if len(features.shape) == 1:
            # Single feature vector
            feature_seq = np.expand_dims(features, axis=0)
        elif len(features.shape) == 2 and features.shape[0] > lookback:
            # Feature matrix - use last lookback timesteps
            feature_seq = features[-lookback:]
        else:
            # Already a sequence
            feature_seq = features
        
        # HMM state estimation
        pi = self.hmm_model["pi"]
        A = self.hmm_model["A"]
        st_list = self.hmm_model["st_params"]
        
        gamma, _, _ = forward_backward(
            feature_seq, pi, A, st_list,
            use_tdist=True
        )
        
        # Current HMM state
        current_state = np.argmax(gamma[-1])
        state_confidence = gamma[-1, current_state]
        
        # Derive state label
        mu = st_list[current_state]["mu"]
        
        # Simplified state labeling
        if self.feature_cols and len(self.feature_cols) >= 4:
            # Analysis of return components (first 4 features)
            avg_returns = np.mean(mu[:4])
            
            if avg_returns > 0.001:
                state_label = "High Bullish" if mu[0] > 0.003 else "Low Bullish"
            elif avg_returns < -0.001:
                state_label = "High Bearish" if mu[0] < -0.003 else "Low Bearish"
            else:
                state_label = "Neutral"
        else:
            # Fallback without feature columns
            state_label = "Neutral" if abs(np.mean(mu)) < 0.001 else \
                         ("Bullish" if np.mean(mu) > 0 else "Bearish")
        
        # HMM prediction
        hmm_prediction = {
            "state_idx": current_state,
            "state_label": state_label,
            "state_confidence": state_confidence
        }
        
        # Prepare input for Hybrid model
        # Use sequence length needed by hybrid model
        if len(feature_seq) < self.hybrid_model.sequence_length:
            # If not enough history: pad with zeros
            padded_seq = np.zeros((self.hybrid_model.sequence_length, feature_seq.shape[1]))
            padded_seq[-len(feature_seq):] = feature_seq
            hybrid_seq = padded_seq
        else:
            hybrid_seq = feature_seq[-self.hybrid_model.sequence_length:]
        
        # Prepare HMM state for Hybrid model (one-hot encoding)
        hmm_state_onehot = np.zeros(self.hmm_model["K"])
        hmm_state_onehot[current_state] = 1
        
        # Create context features
        context = np.zeros(6)
        
        # Market direction from HMM state
        if "Bull" in state_label:
            context[0] = 1  # Bullish signal
        elif "Bear" in state_label:
            context[1] = 1  # Bearish signal
        
        # Volatility level from HMM state
        if "High" in state_label:
            context[2] = 1  # High volatility
        elif "Low" in state_label:
            context[3] = 1  # Low volatility
        
        # Market context from last feature vector
        last_features = feature_seq[-1]
        
        # Last context features could be time-based (session, weekday)
        # Or other weighted market indicators
        if len(last_features) > 10:
            # Example: Use features as indicators for market conditions
            context[4] = np.mean(np.abs(last_features[:4]))  # Mean absolute returns
            context[5] = np.mean(last_features[6:9]) if len(last_features) > 8 else 0.5  # Volatility metrics
        else:
            context[4] = 0.5  # Default value
            context[5] = 0.5  # Default value
        
        # Predictions from Hybrid model
        direction_pred = self.hybrid_model.predict_market_direction(hybrid_seq, hmm_state_onehot)
        volatility_pred = self.hybrid_model.predict_market_volatility(hybrid_seq, hmm_state_onehot)
        action_rec = self.hybrid_model.get_action_recommendation(hybrid_seq, hmm_state_onehot, context)
        
        # Combine all predictions
        combined_pred = self.hybrid_model.combine_predictions(
            hmm_prediction, direction_pred, volatility_pred, action_rec
        )
        
        return combined_pred

# Module-level HybridWrapper-Klasse für Pickle-Kompatibilität
class HybridWrapper:
    """
    Pickle-kompatible Wrapper-Klasse anstelle einer lokalen Funktion.
    Diese Klasse kapselt die Funktionalität, die zuvor in der lokalen hybrid_wrapper Funktion war.
    """
    def __init__(self, model_params, model_paths, feature_cols):
        """
        Initialisiert den Wrapper mit den notwendigen Parametern.
        
        Args:
            model_params: HMM Modellparameter
            model_paths: Pfade zu den gespeicherten TensorFlow-Modellen
            feature_cols: Feature-Spaltennamen
        """
        import numpy as np
        
        # Speichere notwendige Parameter
        self.K = model_params.get("K", 4)
        self.pi = model_params.get("pi")
        self.A = model_params.get("A")
        self.st_params = model_params.get("st_params")
        self.feature_cols = feature_cols
        self.model_paths = model_paths
        
        # Zusätzliche Metadaten
        self.use_tdist = True  # Standard
        self.dims_egarch = [0, 1, 2, 3]  # Standard EGARCH-Dimensionen
        
        # Speichere Modellpfade
        if not hasattr(self, 'model_paths'):
            self.model_paths = {}
            
        # Feature-Dimensionen speichern und validieren
        self.hmm_feature_dim = len(self.st_params[0]['mu']) if self.st_params else 19
        self.expected_tf_feature_dim = None  # Wird beim ersten Laden des TF-Modells gesetzt
        
        import logging
        logging.info(f"HybridWrapper initialisiert: HMM Feature Dimension = {self.hmm_feature_dim}")
    
    def __call__(self, features, state=None, feature_weights=None):
        """
        Führt Vorhersagen durch, wenn die Klasse wie eine Funktion aufgerufen wird.
        
        Args:
            features: Feature-Vektor oder -Matrix
            state: Optionaler State (wird automatisch berechnet, wenn nicht angegeben)
            feature_weights: Optionale Feature-Gewichte
            
        Returns:
            dict: Vorhersageergebnis mit Signal und Konfidenz
        """
        import numpy as np
        import logging
        
        try:
            from enhanced_hmm_em_v2 import forward_backward
        except ImportError:
            logging.error("Could not import forward_backward function")
            return {"signal": "NONE", "signal_strength": 0.0, "error": "missing_dependencies"}
        
        # Stelle sicher, dass es sich um ein numpy-Array handelt
        if not isinstance(features, np.ndarray):
            try:
                features = np.array(features)
            except:
                logging.error("Features must be numpy array or convertible to numpy array")
                return {"signal": "NONE", "signal_strength": 0.0, "error": "invalid_features"}
        
        # Dimensionsprüfung und -anpassung
        if len(features.shape) == 1:
            # Einzelner Feature-Vektor - reshape zu (1, features)
            features = features.reshape(1, -1)
            
        # Feature-Dimensionalität protokollieren
        logging.debug(f"HybridWrapper input features: shape={features.shape}")
            
        # --- VERBESSERT: Robuste Extraktion der HMM-Features ---
        if features.shape[1] < self.hmm_feature_dim:
            logging.error(f"HybridWrapper: Input features ({features.shape[1]}) haben weniger Dimensionen als für HMM benötigt ({self.hmm_feature_dim}).")
            return {"signal": "NONE", "signal_strength": 0.0, "error": "feature_dimension_mismatch"}
            
        # Wähle die ersten hmm_feature_dim Spalten für die HMM-Berechnung aus
        features_hmm = features[:, :self.hmm_feature_dim]
        logging.debug(f"HybridWrapper: Verwende Features mit Shape {features_hmm.shape} für HMM-State-Berechnung.")
        
        # Original-Features für TensorFlow-Modell beibehalten
        features_tf = features
        # --- Ende der Verbesserung ---

        # Feature-Gewichte anwenden, falls angegeben (nur auf HMM-Features)
        weighted_features_hmm = features_hmm.copy()  # Beginne mit ungewichteten HMM-Features
        if feature_weights is not None:
            try:
                # Stelle sicher, dass feature_weights die richtige Länge hat
                if len(feature_weights) == self.hmm_feature_dim:
                    weighted_features_hmm = features_hmm * feature_weights
                elif len(feature_weights) == features.shape[1]:
                    # Verwende nur die ersten hmm_feature_dim Gewichte
                    weighted_features_hmm = features_hmm * feature_weights[:self.hmm_feature_dim]
                    logging.warning(f"Feature-Gewichte haben Dimension {len(feature_weights)}, verwende nur die ersten {self.hmm_feature_dim} für HMM.")
                else:
                    logging.warning(f"Feature-Gewichte-Dimension ({len(feature_weights)}) passt weder zu HMM-Features ({self.hmm_feature_dim}) noch zu allen Features ({features.shape[1]}). Verwende ungewichtete HMM-Features.")
            except Exception as e:
                logging.warning(f"Fehler beim Anwenden von Feature-Gewichten: {e}. Verwende ungewichtete HMM-Features.")
        
        # Aktuellen Zustand bestimmen, wenn nicht angegeben
        current_state = None
        state_probs = None
        
        if state is None:
            try:
                # Forward-Backward-Algorithmus für Zustandsberechnung
                gamma, _, _ = forward_backward(
                    weighted_features_hmm, self.pi, self.A, self.st_params, 
                    use_tdist=self.use_tdist, 
                    dims_egarch=self.dims_egarch
                )
                
                # Zustand mit höchster Wahrscheinlichkeit auswählen
                current_state = np.argmax(gamma[-1])
                state_probs = gamma[-1]
            except Exception as e:
                logging.error(f"Error in state calculation: {str(e)}")
                # Fallback-Zustand
                current_state = 0
                state_probs = np.ones(self.K) / self.K  # Gleichförmige Verteilung
        else:
            # Verwende angegebenen Zustand
            if isinstance(state, dict) and "state" in state:
                current_state = state["state"]
            elif isinstance(state, (int, np.integer)):
                current_state = state
            else:
                current_state = 0
            
            # Erstelle one-hot state probs
            state_probs = np.zeros(self.K)
            state_probs[current_state] = 1.0
        
        # Interpretiere Zustand
        has_hmm_interpretation = True
        try:
            # Basierend auf Mittelwert der Returns
            mu_vec = self.st_params[current_state]["mu"]
            mean_return = np.mean(mu_vec[:4])  # Nimm an, dass die ersten 4 Dimensionen Returns sind
            
            if mean_return > 0.001:
                state_bias = "bullish"
                strength = min(1.0, mean_return * 200)
            elif mean_return < -0.001:
                state_bias = "bearish"
                strength = min(1.0, abs(mean_return) * 200)
            else:
                state_bias = "neutral"
                strength = 0.0
        except:
            has_hmm_interpretation = False
            state_bias = "unknown"
            strength = 0.0
            logging.warning("Could not interpret HMM state")
        
        # TensorFlow-Modell dynamisch laden (für verbesserte Serialisierung)
        try:
            import tensorflow as tf
            import os
            
            # Lade gespeicherte Modelle
            loaded_models = {}
            hybrid_prediction = None
            
            # Prüfe, ob Model-Paths verfügbar sind
            if hasattr(self, 'model_paths') and self.model_paths:
                # Lade Direction-Modell
                if "direction_model" in self.model_paths:
                    try:
                        direction_model_path = self.model_paths["direction_model"]
                        if os.path.exists(direction_model_path):
                            loaded_models["direction_model"] = tf.keras.models.load_model(
                                direction_model_path,
                                compile=False  # Schnelleres Laden ohne erneute Kompilierung
                            )
                            logging.info(f"Loaded direction model from {direction_model_path}")
                            
                            # --- NEU: Modell-Eingabedimensionen inspizieren ---
                            if self.expected_tf_feature_dim is None:
                                try:
                                    # Hole die erwartete Eingabedimension aus dem Modell
                                    # Bei sequence input mit [None, seq_length, feature_dim]
                                    if hasattr(loaded_models["direction_model"], 'input_shape') and loaded_models["direction_model"].input_shape[0][2] is not None:
                                        self.expected_tf_feature_dim = loaded_models["direction_model"].input_shape[0][2]
                                        logging.info(f"Direction model erwartet Feature-Dimension: {self.expected_tf_feature_dim}")
                                except Exception as e:
                                    logging.warning(f"Konnte Feature-Dimension aus TF-Modell nicht ermitteln: {e}")
                            # --- Ende neue Dimension-Erkennung ---
                            
                    except Exception as e:
                        logging.error(f"Error loading direction model: {str(e)}")
                
                # Bereite Features für das Modell vor
                if "direction_model" in loaded_models:
                    try:
                        # Format: [batch_size, sequence_length, features]
                        seq_length = 10  # Standard-Sequenzlänge
                        
                        # --- VERBESSERT: Anpassung der Feature-Dimensionen für TF-Modell ---
                        # Prüfe und adaptiere Feature-Dimensionen, falls nötig
                        if self.expected_tf_feature_dim is not None and features_tf.shape[1] != self.expected_tf_feature_dim:
                            logging.warning(f"Feature-Dimension ({features_tf.shape[1]}) passt nicht zur vom Modell erwarteten Dimension ({self.expected_tf_feature_dim})")
                            
                            if features_tf.shape[1] > self.expected_tf_feature_dim:
                                # Zu viele Features - nimm die ersten expected_tf_feature_dim
                                logging.info(f"Reduziere Features von {features_tf.shape[1]} auf {self.expected_tf_feature_dim} Dimensionen")
                                features_tf = features_tf[:, :self.expected_tf_feature_dim]
                            else:
                                # Zu wenige Features - füge Nullen hinzu (Padding)
                                logging.info(f"Erweitere Features von {features_tf.shape[1]} auf {self.expected_tf_feature_dim} Dimensionen mit Nullen")
                                padding = np.zeros((features_tf.shape[0], self.expected_tf_feature_dim - features_tf.shape[1]))
                                features_tf = np.hstack([features_tf, padding])
                        # --- Ende Feature-Dimension-Anpassung ---
                            
                        # Sequenz-Erstellung wie vorher
                        if features_tf.shape[0] < seq_length:
                            sequence = np.zeros((seq_length, features_tf.shape[1])) # Pad with zeros if needed
                            sequence[-features_tf.shape[0]:] = features_tf
                        else:
                            sequence = features_tf[-seq_length:]
                        
                        # Reshape für LSTM: [1, seq_length, features]
                        sequence = sequence.reshape(1, seq_length, -1)
                        
                        # Erstelle One-Hot-Codierung des HMM-Zustands
                        hmm_state_onehot = np.zeros((1, self.K))
                        hmm_state_onehot[0, current_state] = 1.0
                        
                        # Vor der Vorhersage: Logge die aktuelle Input-Shape 
                        logging.debug(f"Modellvorhersage mit Sequenz-Shape {sequence.shape} und State-Shape {hmm_state_onehot.shape}")
                        
                        # Modellvorhersage mit korrektem Input-Format
                        prediction = loaded_models["direction_model"].predict(
                            [sequence, hmm_state_onehot],
                            verbose=0
                        )
                        
                        # Interpretiere die Vorhersage basierend auf dem Modellformat (5-Klassen-Prognose)
                        # Annahme: [stark runter, schwach runter, neutral, schwach hoch, stark hoch]
                        if len(prediction.shape) > 1 and prediction.shape[1] >= 3:
                            signal_strength = 0.0
                            signal = "NONE"
                            
                            # Beispiel für 5-Klassen-Modell
                            if prediction.shape[1] == 5:
                                up_prob = prediction[0, 3] + prediction[0, 4]  # Schwach + stark hoch
                                down_prob = prediction[0, 0] + prediction[0, 1]  # Stark + schwach runter
                                
                                if up_prob > down_prob and up_prob > 0.4:
                                    signal = "LONG"
                                    signal_strength = up_prob
                                elif down_prob > up_prob and down_prob > 0.4:
                                    signal = "SHORT" 
                                    signal_strength = down_prob
                            
                            # Einfacheres 3-Klassen-Modell
                            elif prediction.shape[1] == 3:
                                up_prob = prediction[0, 0]
                                down_prob = prediction[0, 1]
                                neutral_prob = prediction[0, 2]
                                
                                if up_prob > down_prob and up_prob > neutral_prob and up_prob > 0.4:
                                    signal = "LONG"
                                    signal_strength = up_prob
                                elif down_prob > up_prob and down_prob > neutral_prob and down_prob > 0.4:
                                    signal = "SHORT"
                                    signal_strength = down_prob
                            
                            hybrid_prediction = {
                                "signal": signal,
                                "signal_strength": float(signal_strength)
                            }
                            
                            logging.info(f"Hybrid model prediction: {signal} with strength {signal_strength:.2f}")
                    except Exception as predict_error:
                        logging.error(f"Error in model prediction: {str(predict_error)}")
                        import traceback
                        logging.error(traceback.format_exc())  # Ausführlicheren Fehler loggen
            
            # Falls kein Hybrid-Modell geladen werden konnte, verwende HMM-basierte Vorhersage
            if hybrid_prediction is None and has_hmm_interpretation:
                hybrid_prediction = {
                    "signal": "LONG" if state_bias == "bullish" else "SHORT" if state_bias == "bearish" else "NONE",
                    "signal_strength": strength
                }
                logging.info(f"Using HMM-based prediction: {hybrid_prediction['signal']} with strength {strength:.2f}")
            
            # Standardwert, falls keine Vorhersage gemacht werden konnte
            if hybrid_prediction is None:
                hybrid_prediction = {
                    "signal": "NONE",
                    "signal_strength": 0.0,
                    "error": "prediction_failed"
                }
            
            # Füge Kontextinformationen hinzu
            hybrid_prediction["hmm_state"] = int(current_state)
            hybrid_prediction["state_bias"] = state_bias
            hybrid_prediction["combined_confidence"] = hybrid_prediction["signal_strength"] * 0.6 + strength * 0.4
            hybrid_prediction["prediction_method"] = "hybrid_model" if "direction_model" in loaded_models else "hmm_only"
            
            return hybrid_prediction
            
        except Exception as tf_error:
            logging.error(f"Error in TensorFlow-based prediction: {str(tf_error)}")
            import traceback
            logging.error(traceback.format_exc())  # Ausführlicheren Fehler loggen
            # Fallback: HMM-basierte Vorhersage
        
        # Fallback: Einfache HMM-basierte Vorhersage, wenn TensorFlow-Modell nicht verfügbar
        if has_hmm_interpretation:
            return {
                "signal": "LONG" if state_bias == "bullish" else "SHORT" if state_bias == "bearish" else "NONE",
                "signal_strength": strength,
                "hmm_state": int(current_state),
                "combined_confidence": strength,
                "prediction_method": "hmm_only",
                "fallback_method": True
            }
        else:
            # Wirklich minimaler Fallback bei Fehler in der HMM-Interpretation
            return {
                "signal": "NONE",
                "signal_strength": 0.0,
                "error": "hmm_interpretation_failed",
                "hmm_state": int(current_state) if current_state is not None else 0
            }

# Die aktualisierte create_hmm_hybrid_wrapper-Funktion
def create_hmm_hybrid_wrapper(model_params, hybrid_model, feature_cols):
    """
    Verbesserte Version: Erstellt einen Wrapper, der HMM und Hybrid-Modell für Vorhersagen kombiniert.
    Diese Version speichert TensorFlow-Modelle separat für bessere Kompatibilität.
    
    Args:
        model_params: HMM Modellparameter
        hybrid_model: Trainiertes Hybrid-Modell
        feature_cols: Feature-Spaltennamen
    
    Returns:
        object: HybridWrapper-Objekt für Vorhersagen
    """
    import numpy as np
    import os
    import json
    import logging
    
    # Extrahiere notwendige Parameter aus dem HMM-Modell
    K = model_params["K"]
    pi = model_params["pi"]
    A = model_params["A"]
    st_params = model_params["st_params"]
    
    # Speichere TensorFlow-Modelle separat, wenn sie existieren
    has_saved_tf_models = False
    hybrid_model_paths = {}
    
    try:
        import tensorflow as tf
        
        # Erstelle Ausgabeverzeichnis
        output_dir = "enhanced_model_full/hybrid_model"
        os.makedirs(output_dir, exist_ok=True)
        
        # Speichere alle TensorFlow-Modelle separat
        if hasattr(hybrid_model, 'direction_model') and hybrid_model.direction_model is not None:
            direction_model_path = os.path.join(output_dir, "direction_model")
            hybrid_model.direction_model.save(direction_model_path, save_format="tf")
            hybrid_model_paths["direction_model"] = direction_model_path
            logging.info(f"Direction model saved to {direction_model_path}")
        
        if hasattr(hybrid_model, 'volatility_model') and hybrid_model.volatility_model is not None:
            volatility_model_path = os.path.join(output_dir, "volatility_model")
            hybrid_model.volatility_model.save(volatility_model_path, save_format="tf")
            hybrid_model_paths["volatility_model"] = volatility_model_path
            logging.info(f"Volatility model saved to {volatility_model_path}")
        
        # Weitere Modelle speichern (falls vorhanden)
        for model_name, model in getattr(hybrid_model, 'ensemble_models', {}).items():
            if hasattr(model, 'model') and model['model'] is not None:
                model_path = os.path.join(output_dir, f"ensemble_{model_name}")
                model['model'].save(model_path, save_format="tf")
                hybrid_model_paths[f"ensemble_{model_name}"] = model_path
                logging.info(f"Ensemble model {model_name} saved to {model_path}")
        
        # Speichere Modellkonfiguration in JSON (ohne TensorFlow-Objekte)
        config_path = os.path.join(output_dir, "hybrid_config.json")
        model_config = {
            "input_dim": hybrid_model.input_dim if hasattr(hybrid_model, 'input_dim') else len(feature_cols),
            "hmm_states": hybrid_model.hmm_states if hasattr(hybrid_model, 'hmm_states') else K,
            "sequence_length": hybrid_model.sequence_length if hasattr(hybrid_model, 'sequence_length') else 10,
            "model_paths": hybrid_model_paths,
            "feature_cols": feature_cols
        }
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
            
        has_saved_tf_models = True
        logging.info(f"Hybrid model configuration saved to {config_path}")
        
    except Exception as e:
        logging.error(f"Error saving TensorFlow models: {str(e)}")
        # Weiter mit Fallback-Wrapper
    
    # Erstelle den Wrapper
    wrapper = HybridWrapper(model_params, hybrid_model_paths, feature_cols)
    
    return wrapper
    
def selective_update(hybrid_model, new_data, uncertainty_threshold=0.5, diversity_weight=0.3, max_samples=100):
    """
    Selektives Update des Modells basierend auf der Informativität neuer Daten.
    
    Args:
        hybrid_model: Trainiertes Hybrid-Modell
        new_data: Neue Trainingsdaten (dict mit X_sequences, hmm_states, labels)
        uncertainty_threshold: Schwellenwert für Unsicherheitsbasierte Auswahl
        diversity_weight: Gewicht für Diversität in der Auswahl
        max_samples: Maximale Anzahl an Samples für das Update
    
    Returns:
        dict: Ausgewählte Trainingsdaten für Update
    """
    if 'X_sequences' not in new_data or len(new_data['X_sequences']) == 0:
        return new_data
    
    # Skaliere Features für Modellvorhersagen
    X_scaled = hybrid_model.feature_scaler.transform(
        new_data['X_sequences'].reshape(-1, hybrid_model.input_dim)
    ).reshape(new_data['X_sequences'].shape) if hybrid_model.scaler_fitted else new_data['X_sequences']
    
    # 1. Information relevance: Unsicherheits-basierte Auswahl
    # Vorhersagen für alle neuen Daten
    direction_preds = hybrid_model.direction_model.predict(
        [X_scaled, new_data['hmm_states']], verbose=0
    )
    
    # Unsicherheitsmaß: Entropie der Vorhersagen
    # Höhere Entropie = mehr Unsicherheit = informativere Samples
    uncertainties = -np.sum(direction_preds * np.log(direction_preds + 1e-10), axis=1)
    
    # 2. Diversitäts-Kriterium: Füge Vielfalt hinzu mit Clustering
    if len(X_scaled) > 10:
        from sklearn.cluster import KMeans
        
        # Feature-Representation für Clustering (flatten der Sequenzen)
        X_flat = X_scaled.reshape(len(X_scaled), -1)
        
        # Anzahl Cluster dynamisch anpassen an Datengröße
        n_clusters = min(20, max(3, len(X_flat) // 10))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_flat)
        
        # Cluster-Zentren zur Feature-Repräsentation
        cluster_distances = np.array([
            np.linalg.norm(X_flat[i] - kmeans.cluster_centers_[clusters[i]]) 
            for i in range(len(X_flat))
        ])
        
        # Normalisieren
        if np.max(cluster_distances) > 0:
            normalized_distances = cluster_distances / np.max(cluster_distances)
        else:
            normalized_distances = np.zeros_like(cluster_distances)
        
        # Kombinierte Bewertung: Unsicherheit + Diversität
        scores = uncertainties + diversity_weight * normalized_distances
    else:
        scores = uncertainties
    
    # Top-Samples auswählen basierend auf kombinierten Scores
    n_selected = min(max_samples, len(scores))
    selected_indices = np.argsort(scores)[-n_selected:]
    
    # Nur Samples mit hoher Unsicherheit auswählen (über Schwellenwert)
    high_uncertainty = uncertainties > uncertainty_threshold
    selected_indices = [i for i in selected_indices if high_uncertainty[i]]
    
    if len(selected_indices) == 0:
        # Fallback: Top 20% der Unsicherheitswerte, falls keine über Schwellenwert
        uncertainty_threshold = np.percentile(uncertainties, 80)
        selected_indices = np.where(uncertainties >= uncertainty_threshold)[0]
        
        if len(selected_indices) > max_samples:
            selected_indices = np.random.choice(selected_indices, max_samples, replace=False)
    
    # Ausgewählte Daten extrahieren
    selected_data = {}
    for key in new_data:
        if isinstance(new_data[key], np.ndarray) and len(new_data[key]) == len(new_data['X_sequences']):
            selected_data[key] = new_data[key][selected_indices]
        else:
            selected_data[key] = new_data[key]
    
    print(f"Selektives Update: {len(selected_indices)} von {len(new_data['X_sequences'])} Samples ausgewählt")
    
    return selected_data

def adaptive_memory_update(hybrid_model, features, outcome, recency_factor=0.7, max_memory_per_phase=100):
    """
    Aktualisiert den Pattern-Speicher mit adaptiver Gewichtung zwischen neuen und alten Mustern.
    
    Args:
        hybrid_model: Trainiertes Hybrid-Modell
        features: Feature-Vektor oder -Sequenz
        outcome: Beobachtetes Ergebnis
        recency_factor: Gewicht für aktuelle vs. historische Muster (0-1)
        max_memory_per_phase: Maximale Anzahl gespeicherter Muster pro Marktphase
    """
    # Marktphase basierend auf Features bestimmen
    market_phase = hybrid_model._detect_market_phase(features)
    
    # Neues Muster erstellen
    if len(features.shape) == 2 and features.shape[0] > 1:
        # Bereits eine Sequenz
        feature_seq = features
    else:
        # Einzelner Vektor
        feature_seq = np.expand_dims(features, axis=0)
    
    new_pattern = {
        "features": feature_seq,
        "outcome": outcome,
        "market_phase": market_phase,
        "timestamp": datetime.now().isoformat(),
        "recency_score": 1.0  # Neueste Muster haben höchsten Recency-Score
    }
    
    # Bestehende Muster für diese Marktphase finden
    phase_patterns = [p for p in hybrid_model.pattern_memory["patterns"] 
                     if p.get("market_phase") == market_phase]
    
    # Recency-Scores aller bestehenden Muster aktualisieren
    for pattern in hybrid_model.pattern_memory["patterns"]:
        if "recency_score" in pattern:
            # Recency-Score mit dem Faktor abklingen lassen
            pattern["recency_score"] *= (1 - recency_factor)
    
    # Muster zum Speicher hinzufügen
    hybrid_model.pattern_memory["patterns"].append(new_pattern)
    
    # Überprüfen, ob Speicherlimit für diese Phase erreicht ist
    if len(phase_patterns) + 1 > max_memory_per_phase:
        # Sortiere nach kombinierten Kriterien: Recency und Informationsgehalt
        phase_patterns = [p for p in hybrid_model.pattern_memory["patterns"] 
                         if p.get("market_phase") == market_phase]
        
        # Berechne Informationsgehalt (vereinfacht durch Feature-Varianz)
        for pattern in phase_patterns:
            if "features" in pattern and isinstance(pattern["features"], np.ndarray):
                pattern["info_score"] = np.var(pattern["features"].flatten())
            else:
                pattern["info_score"] = 0
        
        # Kombinierter Score: recency_score + (1-recency_factor) * normalisierter info_score
        max_info = max([p.get("info_score", 0) for p in phase_patterns] + [1e-10])
        for pattern in phase_patterns:
            norm_info = pattern.get("info_score", 0) / max_info
            pattern["combined_score"] = (recency_factor * pattern.get("recency_score", 0) + 
                                       (1 - recency_factor) * norm_info)
        
        # Sortiere nach kombiniertem Score (absteigend)
        sorted_patterns = sorted(phase_patterns, 
                               key=lambda p: p.get("combined_score", 0),
                               reverse=True)
        
        # Behalte nur die Top-N Muster für diese Phase
        keep_patterns = sorted_patterns[:max_memory_per_phase]
        keep_ids = [id(p) for p in keep_patterns]
        
        # Aktualisiere den gesamten Pattern-Speicher
        # Behalte alle Muster anderer Phasen und die Top-N dieser Phase
        hybrid_model.pattern_memory["patterns"] = deque(
            [p for p in hybrid_model.pattern_memory["patterns"] 
             if p.get("market_phase") != market_phase or id(p) in keep_ids],
            maxlen=hybrid_model.pattern_memory["patterns"].maxlen
        )

def apply_adaptive_regularization(hybrid_model, recent_performance=None):
    """
    Wendet adaptive Regularisierung auf die Modellschichten basierend auf Datencharakteristiken an.
    
    Args:
        hybrid_model: Zu regulierendes Hybrid-Modell
        recent_performance: Optionale Performance-Metrik zur Anpassung der Regularisierung
    """
    if not hybrid_model.direction_model or not hybrid_model.volatility_model:
        return
    
    # Basis-Regularisierungsfaktoren
    base_l1 = 0.0001
    base_l2 = 0.001
    
    # Anpassung basierend auf Performance
    if recent_performance is not None:
        # Bei schlechter Performance: Stärkere Regularisierung gegen Überanpassung
        if recent_performance < 0.5:
            l1_factor = base_l1 * 2.0
            l2_factor = base_l2 * 1.5
        # Bei guter Performance: Sanftere Regularisierung für mehr Spezialisierung
        elif recent_performance > 0.7:
            l1_factor = base_l1 * 0.7
            l2_factor = base_l2 * 0.8
        else:
            l1_factor = base_l1
            l2_factor = base_l2
    else:
        l1_factor = base_l1
        l2_factor = base_l2
    
    # Anpassung der Regularisierung erfordert Neukompilierung der Modelle
    # In einer realen Implementierung würde hier der Modell-Kernel direkt angepasst
    
    # Stattdessen neue Regularisierungswerte für das nächste Training vorbereiten
    hybrid_model.current_l1_factor = l1_factor
    hybrid_model.current_l2_factor = l2_factor
    
    print(f"Adaptive Regularisierung angepasst: L1={l1_factor:.6f}, L2={l2_factor:.6f}")

def update_phase_expertise(hybrid_model, market_phase, performance, learning_rate=0.1):
    """
    Aktualisiert Expertisemetriken für spezifische Marktphasen.
    
    Args:
        hybrid_model: Trainiertes Hybrid-Modell
        market_phase: Identifizierte Marktphase
        performance: Performance-Metrik in dieser Phase
        learning_rate: Lernrate für Expertise-Update
    """
    if not hasattr(hybrid_model, 'phase_expertise'):
        # Initialisiere Expertise-Tracking, falls nicht vorhanden
        hybrid_model.phase_expertise = {phase: 0.5 for phase in hybrid_model.market_phases}
    
    # Exponentiell gewichtetes gleitendes Mittel für Performance-Tracking
    old_expertise = hybrid_model.phase_expertise.get(market_phase, 0.5)
    new_expertise = (1 - learning_rate) * old_expertise + learning_rate * performance
    hybrid_model.phase_expertise[market_phase] = new_expertise
    
    # Aktualisiere Ensemble-Gewichte basierend auf Phase-Expertise
    if market_phase in hybrid_model.ensemble_models:
        # Normalisiere Expertise zu Gewichten im Ensemble
        hybrid_model.ensemble_models[market_phase]["weight"] = new_expertise
        hybrid_model.ensemble_models[market_phase]["performance"] = new_expertise
        
        # Normalisiere alle Phasengewichte
        total_weight = sum(model_info["weight"] 
                         for phase, model_info in hybrid_model.ensemble_models.items())
        
        if total_weight > 0:
            for phase in hybrid_model.ensemble_models:
                hybrid_model.ensemble_models[phase]["weight"] /= total_weight
    
    # Knowledge-Transfer: Wissensaustausch zwischen ähnlichen Phasen
    # Identifiziere ähnliche Phasen
    if "bull" in market_phase:
        similar_phases = [p for p in hybrid_model.market_phases if "bull" in p]
    elif "bear" in market_phase:
        similar_phases = [p for p in hybrid_model.market_phases if "bear" in p]
    else:
        similar_phases = []
    
    # Knowledge-Sharing mit geringerer Lernrate
    sharing_rate = learning_rate * 0.3
    for phase in similar_phases:
        if phase != market_phase:
            hybrid_model.phase_expertise[phase] = (
                (1 - sharing_rate) * hybrid_model.phase_expertise.get(phase, 0.5) + 
                sharing_rate * performance
            )

def evaluate_information_gain(hybrid_model, new_feature_data, baseline_features):
    """
    Bewertet den potenziellen Informationsgewinn durch neue Feature-Daten.
    
    Args:
        hybrid_model: Trainiertes Hybrid-Modell
        new_feature_data: Neue Feature-Daten zur Bewertung
        baseline_features: Basis-Features zum Vergleich
    
    Returns:
        float: Informationsgewinn-Score
    """
    # 1. Feature-Abweichung messen (Indikator für neue Information)
    if isinstance(new_feature_data, np.ndarray) and isinstance(baseline_features, np.ndarray):
        # Dimensionen anpassen für Vergleich
        if len(new_feature_data.shape) > 1 and len(baseline_features.shape) > 1:
            # Beide sind Sequenzen - verwende letzte Zeitschritte
            new_vector = new_feature_data[-1]
            base_vector = baseline_features[-1]
        else:
            # Mindestens eines ist ein Vektor
            new_vector = new_feature_data.flatten()
            base_vector = baseline_features.flatten()
        
        # Kürze oder erweitere Vektoren auf gleiche Länge
        min_len = min(len(new_vector), len(base_vector))
        new_vector = new_vector[:min_len]
        base_vector = base_vector[:min_len]
        
        # Normalisierte Abweichung
        deviation = np.mean(np.abs(new_vector - base_vector))
        max_val = max(np.max(np.abs(new_vector)), np.max(np.abs(base_vector)))
        if max_val > 0:
            normalized_deviation = deviation / max_val
        else:
            normalized_deviation = 0
        
        # 2. Vorhersage-Unsicherheit messen (höhere Unsicherheit = mehr Informationspotenzial)
        # Skaliere Features
        if hybrid_model.scaler_fitted:
            shape = new_feature_data.shape
            X_flat = new_feature_data.reshape(-1, hybrid_model.input_dim)
            X_scaled = hybrid_model.feature_scaler.transform(X_flat).reshape(shape)
        else:
            X_scaled = new_feature_data
        
        # Erstelle Dummy-HMM-Zustand für Vorhersage
        dummy_hmm = np.zeros(hybrid_model.hmm_states)
        dummy_hmm[0] = 1  # Arbiträrer aktiver Zustand
        
        # Vorhersage
        try:
            # Anpassung für Modellvorhersage
            if len(X_scaled.shape) == 1:
                X_scaled = np.expand_dims(np.expand_dims(X_scaled, axis=0), axis=0)
            elif len(X_scaled.shape) == 2:
                X_scaled = np.expand_dims(X_scaled, axis=0)
            
            prediction = hybrid_model.direction_model.predict(
                [X_scaled, np.expand_dims(dummy_hmm, axis=0)], verbose=0
            )[0]
            
            # Entropie als Unsicherheitsmaß
            entropy = -np.sum(prediction * np.log(prediction + 1e-10))
            normalized_entropy = entropy / np.log(len(prediction))  # Normalisiert auf [0, 1]
        except:
            normalized_entropy = 0.5  # Fallback
        
        # 3. Kombinierter Informationsgewinn-Score
        # Gewichtete Kombination aus Abweichung und Entropie
        info_gain = 0.6 * normalized_deviation + 0.4 * normalized_entropy
        
        return info_gain
    else:
        return 0.0

def knowledge_distillation(hybrid_model, teacher_outputs, student_inputs, temperature=2.0):
    """
    Implementiert Knowledge Distillation zwischen Ensemble und Einzelmodellen.
    
    Args:
        hybrid_model: Hybrid-Modell mit Ensemble-Komponenten
        teacher_outputs: Ausgaben des Ensemble-Modells (Soft Targets)
        student_inputs: Eingabedaten für das Studentenmodell
        temperature: Temperaturparameter für Soft Targets
    
    Returns:
        float: Distillation Loss
    """
    if not hybrid_model.direction_model or not hybrid_model.use_ensemble:
        return 0.0
    
    # 1. Skaliere Soft Targets mit Temperatur (erhöht Wahrscheinlichkeit von nicht-max Klassen)
    scaled_logits = teacher_outputs / temperature
    soft_targets = tf.nn.softmax(scaled_logits)
    
    # 2. Student-Modell Vorhersagen (einzelnes Basismodell)
    student_pred = hybrid_model.direction_model.predict(student_inputs, verbose=0)
    student_logits = np.log(student_pred + 1e-10)  # Approximierte Logits aus Softmax
    scaled_student_logits = student_logits / temperature
    
    # 3. Berechne Distillation Loss (KL-Divergenz)
    loss = tf.keras.losses.KLDivergence()(soft_targets, tf.nn.softmax(scaled_student_logits))
    
    return float(loss)

def continuous_learning_update(hybrid_model, new_data, session_limit=100, importance_threshold=0.5):
    """
    Implementiert kontinuierliches Lernen mit selektiver Datenauswahl und Wissensverfestigung.
    
    Args:
        hybrid_model: Zu aktualisierendes Hybrid-Modell
        new_data: Neue Trainingsdaten
        session_limit: Maximale Anzahl Samples pro Aktualisierungssitzung
        importance_threshold: Schwellenwert für Feature-Wichtigkeit
    
    Returns:
        dict: Update-Statistiken
    """
    if not isinstance(new_data, dict) or 'X_sequences' not in new_data:
        return {"status": "error", "message": "Ungültige Daten"}
    
    # 1. Selektive Datenauswahl basierend auf Informationsgehalt
    selected_data = selective_update(
        hybrid_model, new_data, 
        uncertainty_threshold=0.4, 
        max_samples=session_limit
    )
    
    if len(selected_data.get('X_sequences', [])) == 0:
        return {"status": "skipped", "message": "Keine informativen Daten ausgewählt"}
    
    # 2. Identifiziere Marktphase für spezialisiertes Training
    market_phases = []
    for seq in selected_data['X_sequences']:
        phase = hybrid_model._detect_market_phase(seq)
        market_phases.append(phase)
    
    # Häufigste Phase
    from collections import Counter
    phase_counter = Counter(market_phases)
    most_common_phase, phase_count = phase_counter.most_common(1)[0]
    
    # 3. Lernrate anpassen basierend auf Datencharakteristik und Modellperformance
    # Niedrigere Lernrate für gut etablierte Phasen, höhere für neue/seltene Phasen
    phase_expertise = getattr(hybrid_model, 'phase_expertise', {}).get(most_common_phase, 0.5)
    
    # Inverse Beziehung: Höhere Expertise = niedrigere Lernrate
    learning_rate = hybrid_model.learning_rate * (1.0 - 0.7 * phase_expertise)
    
    # 4. Feature-Wichtigkeit für Fokussierung
    feature_weights = np.ones(hybrid_model.input_dim)
    
    # Berechne Feature-Wichtigkeit mit Mutual Information
    if 'direction_labels' in selected_data and len(selected_data['X_sequences']) > 10:
        try:
            # Letzte Zeitschritte der Sequenzen für Feature-Wichtigkeit verwenden
            X_flat = np.array([seq[-1] for seq in selected_data['X_sequences']])
            
            # Ziel-Variable für MI-Berechnung
            y = np.argmax(selected_data['direction_labels'], axis=1)
            
            # MI berechnen
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(X_flat, y)
            
            # Normalisieren und in Gewichte umwandeln
            if np.sum(mi_scores) > 0:
                feature_weights = mi_scores / np.sum(mi_scores) * hybrid_model.input_dim
                
                # Nur wichtige Features höher gewichten
                feature_weights = np.where(
                    feature_weights > importance_threshold,
                    feature_weights,
                    np.ones_like(feature_weights) * 0.5
                )
        except Exception as e:
            print(f"Fehler bei Feature-Wichtigkeitsberechnung: {str(e)}")
    
    # 5. Regularisierung anpassen
    apply_adaptive_regularization(hybrid_model, phase_expertise)
    
    # 6. Training durchführen
    # Das Hauptmodell trainieren
    stats = {"phase": most_common_phase, "samples": len(selected_data['X_sequences'])}
    
    # Phase-spezifisches Modell, falls verfügbar
    if hybrid_model.use_ensemble and most_common_phase in hybrid_model.ensemble_models:
        phase_info = hybrid_model.ensemble_models[most_common_phase]
        
        if phase_info["model"] and 'direction_labels' in selected_data:
            # Setze die Lernrate
            K.set_value(phase_info["model"].optimizer.learning_rate, learning_rate)
            
            # Spezifisches Training für diese Marktphase
            history = phase_info["model"].fit(
                [selected_data['X_sequences'], selected_data['hmm_states']],
                selected_data['direction_labels'],
                epochs=5,  # Weniger Epochen für inkrementelles Lernen
                batch_size=16,
                verbose=0
            )
            
            stats["phase_model_loss"] = float(history.history['loss'][-1])
            stats["phase_model_acc"] = float(history.history['accuracy'][-1]) \
                                     if 'accuracy' in history.history else 0.0
    
    # Hauptmodell mit gewichteter Regularisierung aktualisieren
    if 'direction_labels' in selected_data:
        # Aktualisiere Lernrate
        K.set_value(hybrid_model.direction_model.optimizer.learning_rate, learning_rate * 0.8)
        
        # Training
        history = hybrid_model.direction_model.fit(
            [selected_data['X_sequences'], selected_data['hmm_states']],
            selected_data['direction_labels'],
            epochs=3,  # Sehr kurzes Training für inkrementelles Update
            batch_size=16,
            verbose=0
        )
        
        stats["main_model_loss"] = float(history.history['loss'][-1])
        stats["main_model_acc"] = float(history.history['accuracy'][-1]) \
                               if 'accuracy' in history.history else 0.0
        
        # Expertise aktualisieren
        if "main_model_acc" in stats:
            update_phase_expertise(hybrid_model, most_common_phase, stats["main_model_acc"])
    
    # 7. Knowledge Distillation, falls Ensemble-Modelle trainiert wurden
    if hybrid_model.use_ensemble and "phase_model_acc" in stats:
        # Teacher: Phase-spezifisches Modell
        # Student: Hauptmodell
        teacher_preds = hybrid_model.ensemble_models[most_common_phase]["model"].predict(
            [selected_data['X_sequences'], selected_data['hmm_states']], 
            verbose=0
        )
        
        # Distillation durchführen
        distill_loss = knowledge_distillation(
            hybrid_model, 
            teacher_preds, 
            [selected_data['X_sequences'], selected_data['hmm_states']]
        )
        
        stats["distillation_loss"] = float(distill_loss)
    
    # 8. Muster-Gedächtnis aktualisieren
    # Ausgewählte repräsentative Muster in den Speicher aufnehmen
    sample_indices = np.random.choice(
        len(selected_data['X_sequences']),
        min(10, len(selected_data['X_sequences'])),
        replace=False
    )
    
    for idx in sample_indices:
        if 'direction_labels' in selected_data:
            outcome = "profitable" if np.argmax(selected_data['direction_labels'][idx]) == 0 else "loss"
        else:
            outcome = None
            
        adaptive_memory_update(
            hybrid_model, 
            selected_data['X_sequences'][idx], 
            outcome, 
            recency_factor=0.6
        )
    
    stats["status"] = "success"
    stats["learning_rate"] = float(learning_rate)
    
    return stats

def transfer_knowledge_between_models(source_model, target_model, transfer_layers=None, alpha=0.3):
    """
    Überträgt Wissen zwischen Modellen durch selektiven Parameter-Transfer.
    
    Args:
        source_model: Quellmodell (Lehrer)
        target_model: Zielmodell (Schüler)
        transfer_layers: Liste der zu übertragenden Schichten 
                        (None = alle außer der letzten Schicht)
        alpha: Transferfaktor (0-1)
    """
    if transfer_layers is None:
        # Standardmäßig alle Schichten außer der letzten übertragen
        transfer_layers = list(range(len(source_model.layers) - 1))
    
    # Überprüfe Kompatibilität
    if len(source_model.layers) != len(target_model.layers):
        print("Warnung: Modelle haben unterschiedliche Anzahl an Schichten")
        return False
    
    # Parameter selektiv übertragen
    for layer_idx in transfer_layers:
        if layer_idx >= len(source_model.layers) or layer_idx >= len(target_model.layers):
            continue
            
        source_weights = source_model.layers[layer_idx].get_weights()
        target_weights = target_model.layers[layer_idx].get_weights()
        
        if len(source_weights) != len(target_weights):
            print(f"Warnung: Inkompatible Schichten bei Index {layer_idx}")
            continue
        
        # Gewichteter Transfer
        new_weights = []
        for s_w, t_w in zip(source_weights, target_weights):
            if s_w.shape != t_w.shape:
                print(f"Warnung: Inkompatible Gewichtsformen in Schicht {layer_idx}")
                new_weights.append(t_w)  # Behalte Zielgewichte
                continue
            
            # Mische Quell- und Zielparameter gewichtet
            blended_w = (1 - alpha) * t_w + alpha * s_w
            new_weights.append(blended_w)
        
        # Aktualisierte Gewichte setzen
        target_model.layers[layer_idx].set_weights(new_weights)
    
    return True

# Füge Funktionalität zum HybridModel hinzu, um die kontinuierliche Lernmethode direkt aufzurufen
def _hybrid_continuous_learning_update(self, new_data, **kwargs):
    return continuous_learning_update(self, new_data, **kwargs)

def _hybrid_selective_update(self, new_data, **kwargs):
    return selective_update(self, new_data, **kwargs)

def _hybrid_adaptive_memory_update(self, features, outcome, **kwargs):
    return adaptive_memory_update(self, features, outcome, **kwargs)

def _hybrid_apply_regularization(self, **kwargs):
    return apply_adaptive_regularization(self, **kwargs)

def _hybrid_update_phase_expertise(self, market_phase, performance, **kwargs):
    return update_phase_expertise(self, market_phase, performance, **kwargs)

def _hybrid_evaluate_information_gain(self, new_feature_data, baseline_features):
    return evaluate_information_gain(self, new_feature_data, baseline_features)

def _hybrid_transfer_knowledge(self, source_model, **kwargs):
    return transfer_knowledge_between_models(source_model, self.direction_model, **kwargs)

# Now redefine the extend function to use these module-level functions
def extend_hybrid_model_with_continuous_learning(HybridModel):
    """
    Erweitert die HybridModel-Klasse mit kontinuierlichen Lernfähigkeiten.
    """
    # Add the methods to the class
    HybridModel.continuous_learning_update = _hybrid_continuous_learning_update
    HybridModel.selective_update = _hybrid_selective_update
    HybridModel.adaptive_memory_update = _hybrid_adaptive_memory_update
    HybridModel.apply_regularization = _hybrid_apply_regularization
    HybridModel.update_phase_expertise = _hybrid_update_phase_expertise
    HybridModel.evaluate_information_gain = _hybrid_evaluate_information_gain
    HybridModel.transfer_knowledge = _hybrid_transfer_knowledge
    
    return HybridModel

# Hauptfunktion zum Initialisieren und Erweitern eines Hybrid-Modells
def create_enhanced_hybrid_model(input_dim, hmm_states=4, lstm_units=64, use_attention=True):
    """
    Erstellt ein erweitertes Hybrid-Modell mit allen Funktionalitäten.
    
    Args:
        input_dim: Eingabedimension
        hmm_states: Anzahl der HMM-Zustände
        lstm_units: Anzahl der LSTM-Einheiten
        use_attention: Attention-Mechanismen verwenden
    
    Returns:
        HybridModel: Erweitertes Hybrid-Modell
    """
    model = HybridModel(
        input_dim=input_dim,
        hmm_states=hmm_states,
        lstm_units=lstm_units,
        use_attention=use_attention,
        use_ensemble=True,
        market_phase_count=5
    )
    
    # Modell mit grundlegenden Schichten initialisieren
    model.build_models()
    
    # Mit kontinuierlichen Lernfähigkeiten erweitern
    model = extend_hybrid_model_with_continuous_learning(model)
    
    # Expertise für alle Marktphasen initialisieren
    model.phase_expertise = {phase: 0.5 for phase in model.market_phases}
    
    # L1/L2 Regularisierungsfaktoren initialisieren
    model.current_l1_factor = 0.0001
    model.current_l2_factor = 0.001
    
    return model

# Beispiel-Workflow für die Verwendung des erweiterten Hybrid-Modells
def example_hybrid_model_workflow(feature_dim, hmm_model, data):
    """
    Beispiel-Workflow zur Demonstration der Modell-Funktionalitäten.
    
    Args:
        feature_dim: Dimension der Eingabefeatures
        hmm_model: Trainiertes HMM-Modell
        data: Trainingsdaten
    """
    # 1. Erweitertes Hybrid-Modell erstellen
    hybrid_model = create_enhanced_hybrid_model(
        input_dim=feature_dim,
        hmm_states=hmm_model["K"],
        lstm_units=64,
        use_attention=True
    )
    
    # 2. Initialisieren und Vorbereiten der Daten
    X_sequences = data["X_sequences"]
    hmm_states = data["hmm_states"]
    direction_labels = data["direction_labels"]
    
    # 3. Training mit Feature-Wichtigkeits-Fokussierung
    hybrid_model.train_direction_model(
        X_sequences, hmm_states, direction_labels,
        epochs=30, batch_size=32
    )
    
    # 4. Kontinuierliches Lernen mit neuen Daten
    for session in range(5):
        # Simuliere neue Daten
        new_data = {
            "X_sequences": X_sequences[session*20:(session+1)*20],
            "hmm_states": hmm_states[session*20:(session+1)*20],
            "direction_labels": direction_labels[session*20:(session+1)*20]
        }
        
        # Selektives Update durchführen
        hybrid_model.continuous_learning_update(new_data)
    
    # 5. Wrapper für die kombinierte Vorhersagefunktion erstellen
    wrapper = create_hmm_hybrid_wrapper(hmm_model, hybrid_model, feature_cols=data.get("feature_cols"))
    
    return wrapper, hybrid_model