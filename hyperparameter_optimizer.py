#!/usr/bin/env python3
"""
hyperparameter_optimizer.py

Advanced hyperparameter optimization module for HMM trading models.
Supports multiple optimization strategies including:
- Bayesian Optimization
- Tree-structured Parzen Estimator (TPE)
- Evolutionary Strategies
- BOHB (Bayesian Optimization Hyperband)

Key features:
- Parallel evaluation for efficiency
- Cross-validation support
- Comprehensive result logging and visualization
- Flexible parameter space definition
- Multiple objective functions (likelihood, trading performance, etc.)
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple, Callable, Union, Any, Optional
import warnings

# Try to import optimization libraries, with fallbacks
try:
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
    from skopt.plots import plot_objective, plot_evaluations
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("scikit-optimize not available, some optimization methods will be disabled")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("optuna not available, some optimization methods will be disabled")

try:
    import ray
    from ray import tune
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune.suggest.optuna import OptunaSearch
    from ray.tune.suggest.bayesopt import BayesOptSearch
    from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("ray not available, distributed optimization will be disabled")

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    warnings.warn("hyperopt not available, some optimization methods will be disabled")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(f"hyperparameter_opt_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('hyperparameter_optimizer')

# File paths and directories
DEFAULT_RESULTS_DIR = "hyperparameter_results"
DEFAULT_CHECKPOINT_DIR = "hyperparameter_checkpoints"

# Ensure directories exist
for directory in [DEFAULT_RESULTS_DIR, DEFAULT_CHECKPOINT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

###############################################################################
# Parameter Space Definitions
###############################################################################

def get_hmm_parameter_space(strategy='optuna'):
    """
    Get parameter space for HMM hyperparameters based on the optimization strategy.
    
    Args:
        strategy: Optimization strategy ('optuna', 'skopt', 'hyperopt', 'ray')
        
    Returns:
        Dict or list of parameter space definitions
    """
    if strategy == 'optuna':
        # Optuna parameter space
        def define_optuna_space(trial):
            param_space = {
                # Model structure
                'K': trial.suggest_int('K', 2, 8),  # Number of states
                'use_tdist': trial.suggest_categorical('use_tdist', [True, False]),
                
                # EGARCH parameters range for initialization
                'omega_init_min': trial.suggest_float('omega_init_min', -0.1, 0.05),
                'omega_init_max': trial.suggest_float('omega_init_max', 0.0, 0.15),
                'alpha_init_min': trial.suggest_float('alpha_init_min', 0.0, 0.1),
                'alpha_init_max': trial.suggest_float('alpha_init_max', 0.05, 0.2),
                'gamma_init_min': trial.suggest_float('gamma_init_min', -0.2, 0.0),
                'gamma_init_max': trial.suggest_float('gamma_init_max', 0.0, 0.2),
                'beta_init_min': trial.suggest_float('beta_init_min', 0.5, 0.7),
                'beta_init_max': trial.suggest_float('beta_init_max', 0.7, 0.99),
                
                # T-distribution parameters
                'df_t_min': trial.suggest_int('df_t_min', 3, 10),
                'df_t_max': trial.suggest_int('df_t_max', 10, 30),
                
                # Advanced features
                'use_skewed_t': trial.suggest_categorical('use_skewed_t', [True, False]),
                'use_regime_specific_egarch': trial.suggest_categorical('use_regime_specific_egarch', [True, False]),
                'use_time_varying_transition': trial.suggest_categorical('use_time_varying_transition', [True, False]),
                
                # Training parameters
                'n_starts': trial.suggest_int('n_starts', 1, 10),
                'max_iter': trial.suggest_int('max_iter', 10, 50),
                
                # Feature scaling and selection
                'use_feature_scaling': trial.suggest_categorical('use_feature_scaling', [True, False]),
                'use_feature_selection': trial.suggest_categorical('use_feature_selection', [True, False]),
                
                # Feature groups to include
                'use_returns': trial.suggest_categorical('use_returns', [True]),  # Always true
                'use_oscillators': trial.suggest_categorical('use_oscillators', [True, False]),
                'use_volatility': trial.suggest_categorical('use_volatility', [True, False]),
                'use_session': trial.suggest_categorical('use_session', [True, False]),
                'use_weekday': trial.suggest_categorical('use_weekday', [True, False]),
                'use_volume': trial.suggest_categorical('use_volume', [True, False]),
                
                # Regularization for stability
                'stability_regularization': trial.suggest_float('stability_regularization', 0.0, 0.2),
                
                # Cross-validation parameters
                'cv_fold_count': trial.suggest_int('cv_fold_count', 3, 5)
            }
            return param_space
            
        return define_optuna_space
        
    elif strategy == 'skopt':
        # scikit-optimize parameter space
        param_space = [
            Integer(2, 8, name='K'),
            Categorical([True, False], name='use_tdist'),
            
            Real(-0.1, 0.05, name='omega_init_min'),
            Real(0.0, 0.15, name='omega_init_max'),
            Real(0.0, 0.1, name='alpha_init_min'),
            Real(0.05, 0.2, name='alpha_init_max'),
            Real(-0.2, 0.0, name='gamma_init_min'),
            Real(0.0, 0.2, name='gamma_init_max'),
            Real(0.5, 0.7, name='beta_init_min'),
            Real(0.7, 0.99, name='beta_init_max'),
            
            Integer(3, 10, name='df_t_min'),
            Integer(10, 30, name='df_t_max'),
            
            Categorical([True, False], name='use_skewed_t'),
            Categorical([True, False], name='use_regime_specific_egarch'),
            Categorical([True, False], name='use_time_varying_transition'),
            
            Integer(1, 10, name='n_starts'),
            Integer(10, 50, name='max_iter'),
            
            Categorical([True, False], name='use_feature_scaling'),
            Categorical([True, False], name='use_feature_selection'),
            
            Categorical([True], name='use_returns'),  # Always True
            Categorical([True, False], name='use_oscillators'),
            Categorical([True, False], name='use_volatility'),
            Categorical([True, False], name='use_session'),
            Categorical([True, False], name='use_weekday'),
            Categorical([True, False], name='use_volume'),
            
            Real(0.0, 0.2, name='stability_regularization'),
            Integer(3, 5, name='cv_fold_count')
        ]
        return param_space
        
    elif strategy == 'hyperopt':
        # hyperopt parameter space
        param_space = {
            'K': hp.quniform('K', 2, 8, 1),
            'use_tdist': hp.choice('use_tdist', [True, False]),
            
            'omega_init_min': hp.uniform('omega_init_min', -0.1, 0.05),
            'omega_init_max': hp.uniform('omega_init_max', 0.0, 0.15),
            'alpha_init_min': hp.uniform('alpha_init_min', 0.0, 0.1),
            'alpha_init_max': hp.uniform('alpha_init_max', 0.05, 0.2),
            'gamma_init_min': hp.uniform('gamma_init_min', -0.2, 0.0),
            'gamma_init_max': hp.uniform('gamma_init_max', 0.0, 0.2),
            'beta_init_min': hp.uniform('beta_init_min', 0.5, 0.7),
            'beta_init_max': hp.uniform('beta_init_max', 0.7, 0.99),
            
            'df_t_min': hp.quniform('df_t_min', 3, 10, 1),
            'df_t_max': hp.quniform('df_t_max', 10, 30, 1),
            
            'use_skewed_t': hp.choice('use_skewed_t', [True, False]),
            'use_regime_specific_egarch': hp.choice('use_regime_specific_egarch', [True, False]),
            'use_time_varying_transition': hp.choice('use_time_varying_transition', [True, False]),
            
            'n_starts': hp.quniform('n_starts', 1, 10, 1),
            'max_iter': hp.quniform('max_iter', 10, 50, 1),
            
            'use_feature_scaling': hp.choice('use_feature_scaling', [True, False]),
            'use_feature_selection': hp.choice('use_feature_selection', [True, False]),
            
            'use_returns': True,  # Always True
            'use_oscillators': hp.choice('use_oscillators', [True, False]),
            'use_volatility': hp.choice('use_volatility', [True, False]),
            'use_session': hp.choice('use_session', [True, False]),
            'use_weekday': hp.choice('use_weekday', [True, False]),
            'use_volume': hp.choice('use_volume', [True, False]),
            
            'stability_regularization': hp.uniform('stability_regularization', 0.0, 0.2),
            'cv_fold_count': hp.quniform('cv_fold_count', 3, 5, 1)
        }
        return param_space
        
    elif strategy == 'ray':
        # Ray Tune parameter space
        param_space = {
            'K': tune.randint(2, 9),  # 2 to 8
            'use_tdist': tune.choice([True, False]),
            
            'omega_init_min': tune.uniform(-0.1, 0.05),
            'omega_init_max': tune.uniform(0.0, 0.15),
            'alpha_init_min': tune.uniform(0.0, 0.1),
            'alpha_init_max': tune.uniform(0.05, 0.2),
            'gamma_init_min': tune.uniform(-0.2, 0.0),
            'gamma_init_max': tune.uniform(0.0, 0.2),
            'beta_init_min': tune.uniform(0.5, 0.7),
            'beta_init_max': tune.uniform(0.7, 0.99),
            
            'df_t_min': tune.randint(3, 11),  # 3 to 10
            'df_t_max': tune.randint(10, 31),  # 10 to 30
            
            'use_skewed_t': tune.choice([True, False]),
            'use_regime_specific_egarch': tune.choice([True, False]),
            'use_time_varying_transition': tune.choice([True, False]),
            
            'n_starts': tune.randint(1, 11),  # 1 to 10
            'max_iter': tune.randint(10, 51),  # 10 to 50
            
            'use_feature_scaling': tune.choice([True, False]),
            'use_feature_selection': tune.choice([True, False]),
            
            'use_returns': True,  # Always True
            'use_oscillators': tune.choice([True, False]),
            'use_volatility': tune.choice([True, False]),
            'use_session': tune.choice([True, False]),
            'use_weekday': tune.choice([True, False]),
            'use_volume': tune.choice([True, False]),
            
            'stability_regularization': tune.uniform(0.0, 0.2),
            'cv_fold_count': tune.randint(3, 6)  # 3 to 5
        }
        return param_space
    
    else:
        raise ValueError(f"Unknown optimization strategy: {strategy}")

def get_hybrid_parameter_space(strategy='optuna'):
    """
    Get parameter space for hybrid HMM+NN model hyperparameters.
    
    Args:
        strategy: Optimization strategy ('optuna', 'skopt', 'hyperopt', 'ray')
        
    Returns:
        Dict or list of parameter space definitions
    """
    if strategy == 'optuna':
        # Optuna parameter space
        def define_optuna_space(trial):
            param_space = {
                # HMM parameters
                'K': trial.suggest_int('K', 2, 8),
                'use_tdist': trial.suggest_categorical('use_tdist', [True, False]),
                
                # Hybrid model structure
                'lstm_units': trial.suggest_categorical('lstm_units', [32, 64, 128, 256]),
                'dense_units': trial.suggest_categorical('dense_units', [16, 32, 64, 128]),
                'sequence_length': trial.suggest_int('sequence_length', 5, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                
                # Dropout for regularization
                'lstm_dropout': trial.suggest_float('lstm_dropout', 0.0, 0.5),
                'dense_dropout': trial.suggest_float('dense_dropout', 0.0, 0.5),
                
                # Training parameters
                'epochs': trial.suggest_int('epochs', 10, 100),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
                'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 20),
                
                # Hybrid contribution
                'hmm_weight': trial.suggest_float('hmm_weight', 0.2, 0.8),
                'nn_weight': trial.suggest_float('nn_weight', 0.2, 0.8),
                
                # Feature selection and preprocessing
                'use_feature_selection': trial.suggest_categorical('use_feature_selection', [True, False]),
                'use_pca': trial.suggest_categorical('use_pca', [True, False]),
                'pca_components': trial.suggest_int('pca_components', 5, 20) if 
                                 trial.suggest_categorical('use_pca', [True, False]) else None
            }
            return param_space
            
        return define_optuna_space
        
    elif strategy == 'skopt':
        # scikit-optimize parameter space
        param_space = [
            Integer(2, 8, name='K'),
            Categorical([True, False], name='use_tdist'),
            
            Categorical([32, 64, 128, 256], name='lstm_units'),
            Categorical([16, 32, 64, 128], name='dense_units'),
            Integer(5, 20, name='sequence_length'),
            Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
            
            Real(0.0, 0.5, name='lstm_dropout'),
            Real(0.0, 0.5, name='dense_dropout'),
            
            Integer(10, 100, name='epochs'),
            Categorical([16, 32, 64, 128, 256], name='batch_size'),
            Integer(5, 20, name='early_stopping_patience'),
            
            Real(0.2, 0.8, name='hmm_weight'),
            Real(0.2, 0.8, name='nn_weight'),
            
            Categorical([True, False], name='use_feature_selection'),
            Categorical([True, False], name='use_pca'),
            Integer(5, 20, name='pca_components')
        ]
        return param_space
        
    elif strategy == 'hyperopt':
        # hyperopt parameter space
        param_space = {
            'K': hp.quniform('K', 2, 8, 1),
            'use_tdist': hp.choice('use_tdist', [True, False]),
            
            'lstm_units': hp.choice('lstm_units', [32, 64, 128, 256]),
            'dense_units': hp.choice('dense_units', [16, 32, 64, 128]),
            'sequence_length': hp.quniform('sequence_length', 5, 20, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
            
            'lstm_dropout': hp.uniform('lstm_dropout', 0.0, 0.5),
            'dense_dropout': hp.uniform('dense_dropout', 0.0, 0.5),
            
            'epochs': hp.quniform('epochs', 10, 100, 1),
            'batch_size': hp.choice('batch_size', [16, 32, 64, 128, 256]),
            'early_stopping_patience': hp.quniform('early_stopping_patience', 5, 20, 1),
            
            'hmm_weight': hp.uniform('hmm_weight', 0.2, 0.8),
            'nn_weight': hp.uniform('nn_weight', 0.2, 0.8),
            
            'use_feature_selection': hp.choice('use_feature_selection', [True, False]),
            'use_pca': hp.choice('use_pca', [True, False]),
            'pca_components': hp.quniform('pca_components', 5, 20, 1)
        }
        return param_space
        
    elif strategy == 'ray':
        # Ray Tune parameter space
        param_space = {
            'K': tune.randint(2, 9),  # 2 to 8
            'use_tdist': tune.choice([True, False]),
            
            'lstm_units': tune.choice([32, 64, 128, 256]),
            'dense_units': tune.choice([16, 32, 64, 128]),
            'sequence_length': tune.randint(5, 21),  # 5 to 20
            'learning_rate': tune.loguniform(1e-4, 1e-2),
            
            'lstm_dropout': tune.uniform(0.0, 0.5),
            'dense_dropout': tune.uniform(0.0, 0.5),
            
            'epochs': tune.randint(10, 101),
            'batch_size': tune.choice([16, 32, 64, 128, 256]),
            'early_stopping_patience': tune.randint(5, 21),
            
            'hmm_weight': tune.uniform(0.2, 0.8),
            'nn_weight': tune.uniform(0.2, 0.8),
            
            'use_feature_selection': tune.choice([True, False]),
            'use_pca': tune.choice([True, False]),
            'pca_components': tune.randint(5, 21)  # 5 to 20
        }
        return param_space
    
    else:
        raise ValueError(f"Unknown optimization strategy: {strategy}")

def get_ensemble_parameter_space(strategy='optuna'):
    """
    Get parameter space for ensemble model hyperparameters.
    
    Args:
        strategy: Optimization strategy ('optuna', 'skopt', 'hyperopt', 'ray')
        
    Returns:
        Dict or list of parameter space definitions
    """
    if strategy == 'optuna':
        # Optuna parameter space
        def define_optuna_space(trial):
            param_space = {
                # Ensemble structure
                'n_models': trial.suggest_int('n_models', 3, 10),
                'ensemble_type': trial.suggest_categorical('ensemble_type', 
                                                         ['voting', 'bayes', 'adaptive']),
                
                # Base model parameters
                'base_K': trial.suggest_int('base_K', 2, 8),  # Base number of states
                'K_variation': trial.suggest_int('K_variation', 0, 3),  # Allow K to vary by this much
                
                # Model diversity parameters
                'feature_subset_ratio': trial.suggest_float('feature_subset_ratio', 0.6, 1.0),
                'sample_subset_ratio': trial.suggest_float('sample_subset_ratio', 0.6, 1.0),
                'model_type_diversity': trial.suggest_categorical('model_type_diversity', 
                                                               [True, False]),
                
                # Training parameters
                'n_starts_per_model': trial.suggest_int('n_starts_per_model', 1, 5),
                'max_iter_per_model': trial.suggest_int('max_iter_per_model', 10, 30),
                
                # Weight assignment strategy
                'weights_method': trial.suggest_categorical('weights_method', 
                                                         ['equal', 'likelihood', 'cv_performance']),
                
                # Dynamic weights for adaptive ensemble
                'use_dynamic_weights': trial.suggest_categorical('use_dynamic_weights', 
                                                              [True, False]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                
                # Feature selection
                'use_feature_selection': trial.suggest_categorical('use_feature_selection', 
                                                                 [True, False])
            }
            return param_space
            
        return define_optuna_space
        
    elif strategy == 'skopt':
        # scikit-optimize parameter space
        param_space = [
            Integer(3, 10, name='n_models'),
            Categorical(['voting', 'bayes', 'adaptive'], name='ensemble_type'),
            
            Integer(2, 8, name='base_K'),
            Integer(0, 3, name='K_variation'),
            
            Real(0.6, 1.0, name='feature_subset_ratio'),
            Real(0.6, 1.0, name='sample_subset_ratio'),
            Categorical([True, False], name='model_type_diversity'),
            
            Integer(1, 5, name='n_starts_per_model'),
            Integer(10, 30, name='max_iter_per_model'),
            
            Categorical(['equal', 'likelihood', 'cv_performance'], name='weights_method'),
            
            Categorical([True, False], name='use_dynamic_weights'),
            Real(0.01, 0.2, name='learning_rate'),
            
            Categorical([True, False], name='use_feature_selection')
        ]
        return param_space
        
    elif strategy == 'hyperopt':
        # hyperopt parameter space
        param_space = {
            'n_models': hp.quniform('n_models', 3, 10, 1),
            'ensemble_type': hp.choice('ensemble_type', ['voting', 'bayes', 'adaptive']),
            
            'base_K': hp.quniform('base_K', 2, 8, 1),
            'K_variation': hp.quniform('K_variation', 0, 3, 1),
            
            'feature_subset_ratio': hp.uniform('feature_subset_ratio', 0.6, 1.0),
            'sample_subset_ratio': hp.uniform('sample_subset_ratio', 0.6, 1.0),
            'model_type_diversity': hp.choice('model_type_diversity', [True, False]),
            
            'n_starts_per_model': hp.quniform('n_starts_per_model', 1, 5, 1),
            'max_iter_per_model': hp.quniform('max_iter_per_model', 10, 30, 1),
            
            'weights_method': hp.choice('weights_method', ['equal', 'likelihood', 'cv_performance']),
            
            'use_dynamic_weights': hp.choice('use_dynamic_weights', [True, False]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            
            'use_feature_selection': hp.choice('use_feature_selection', [True, False])
        }
        return param_space
        
    elif strategy == 'ray':
        # Ray Tune parameter space
        param_space = {
            'n_models': tune.randint(3, 11),  # 3 to 10
            'ensemble_type': tune.choice(['voting', 'bayes', 'adaptive']),
            
            'base_K': tune.randint(2, 9),  # 2 to 8
            'K_variation': tune.randint(0, 4),  # 0 to 3
            
            'feature_subset_ratio': tune.uniform(0.6, 1.0),
            'sample_subset_ratio': tune.uniform(0.6, 1.0),
            'model_type_diversity': tune.choice([True, False]),
            
            'n_starts_per_model': tune.randint(1, 6),  # 1 to 5
            'max_iter_per_model': tune.randint(10, 31),  # 10 to 30
            
            'weights_method': tune.choice(['equal', 'likelihood', 'cv_performance']),
            
            'use_dynamic_weights': tune.choice([True, False]),
            'learning_rate': tune.uniform(0.01, 0.2),
            
            'use_feature_selection': tune.choice([True, False])
        }
        return param_space
    
    else:
        raise ValueError(f"Unknown optimization strategy: {strategy}")

###############################################################################
# Objective Functions
###############################################################################

class HMMObjectiveBase:
    """Base class for HMM optimization objectives"""
    def __init__(self, features, feature_cols=None, dims_egarch=None, times=None, 
                cross_validation=True, n_splits=5):
        """
        Initialize the objective function.
        
        Args:
            features: Feature matrix [T, D]
            feature_cols: Feature column names
            dims_egarch: EGARCH dimensions (default: [0, 1, 2, 3])
            times: Optional time series for time-varying transitions
            cross_validation: Whether to use cross-validation
            n_splits: Number of cross-validation splits
        """
        self.features = features
        self.feature_cols = feature_cols
        self.dims_egarch = dims_egarch if dims_egarch is not None else [0, 1, 2, 3]
        self.times = times
        self.cross_validation = cross_validation
        self.n_splits = n_splits
        
        # Import necessary functions - must be dynamically imported here
        try:
            from enhanced_hmm_em_v2 import train_hmm_once, forward_backward
            self.train_hmm_once = train_hmm_once
            self.forward_backward = forward_backward
        except ImportError:
            logger.error("Could not import necessary HMM functions. Make sure enhanced_hmm_em_v2.py is in your path.")
            raise
        
        # Try to import ensemble functions if available
        try:
            from enhanced_hmm_em_v2 import HMMEnsemble
            self.HMMEnsemble = HMMEnsemble
            self.has_ensemble = True
        except ImportError:
            logger.warning("HMMEnsemble not available. Ensemble optimization will be disabled.")
            self.has_ensemble = False
    
    def select_features(self, params):
        """
        Select features based on hyperparameters.
        
        Args:
            params: Parameter dictionary with feature selection flags
            
        Returns:
            Selected features and feature columns
        """
        if not self.feature_cols:
            return self.features, None
            
        # Create feature groups
        feature_groups = {
            'returns': [col for col in self.feature_cols if 'return' in col.lower()],
            'oscillators': [col for col in self.feature_cols if 'rsi' in col.lower() or 'macd' in col.lower()],
            'volatility': [col for col in self.feature_cols if 'atr' in col.lower() or 'volatility' in col.lower()],
            'session': [col for col in self.feature_cols if 'session' in col.lower()],
            'weekday': [col for col in self.feature_cols if 'day_' in col.lower()],
            'volume': [col for col in self.feature_cols if 'volume' in col.lower()]
        }
        
        # Select feature columns based on params
        selected_cols = []
        
        if params.get('use_returns', True):  # Returns always included
            selected_cols.extend(feature_groups['returns'])
            
        if params.get('use_oscillators', True):
            selected_cols.extend(feature_groups['oscillators'])
            
        if params.get('use_volatility', True):
            selected_cols.extend(feature_groups['volatility'])
            
        if params.get('use_session', True):
            selected_cols.extend(feature_groups['session'])
            
        if params.get('use_weekday', True):
            selected_cols.extend(feature_groups['weekday'])
            
        if params.get('use_volume', True):
            selected_cols.extend(feature_groups['volume'])
        
        # Get indices of selected columns
        selected_indices = [i for i, col in enumerate(self.feature_cols) if col in selected_cols]
        
        # Select features
        selected_features = self.features[:, selected_indices]
        selected_feature_cols = [self.feature_cols[i] for i in selected_indices]
        
        return selected_features, selected_feature_cols
    
    def create_cv_folds(self, features, n_splits=5):
        """
        Create cross-validation folds.
        
        Args:
            features: Feature matrix
            n_splits: Number of splits
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        T = len(features)
        fold_size = T // n_splits
        folds = []
        
        for i in range(n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_splits - 1 else T
            
            test_idx = list(range(test_start, test_end))
            train_idx = list(range(0, test_start)) + list(range(test_end, T))
            
            folds.append((train_idx, test_idx))
        
        return folds
    
    def select_egarch_dims(self, selected_features_cols):
        """
        Map EGARCH dimensions to the new feature indices after selection.
    
        Args:
            selected_features_cols: Selected feature column names
        
        Returns:
            List of EGARCH dimension indices in the new feature space
        """
        if not self.feature_cols:
            return self.dims_egarch
            
        # Sicherstellen, dass wir keine ungültigen Indizes verwenden
        safe_dims_egarch = [i for i in self.dims_egarch if i < len(self.feature_cols)]
        if not safe_dims_egarch:
            # Fallback: Verwende die erste Dimension, falls verfügbar
            safe_dims_egarch = [0] if len(self.feature_cols) > 0 else []
            
        original_egarch_cols = [self.feature_cols[i] for i in safe_dims_egarch]
    
        # Map to new indices
        new_dims_egarch = [
            i for i, col in enumerate(selected_features_cols) 
            if col in original_egarch_cols
        ]
    
        # Ensure we have at least one dimension
        if not new_dims_egarch and selected_features_cols:
            new_dims_egarch = [0]  # Default to first feature
    
        return new_dims_egarch
    
    def train_hmm_model(self, features, params, dims_egarch, times=None):
        """
        Train an HMM model with given parameters.
        
        Args:
            features: Feature matrix
            params: Parameter dictionary
            dims_egarch: EGARCH dimensions
            times: Optional time series
            
        Returns:
            Trained model and log-likelihood
        """
        # Set modified params for enhanced_hmm_em_v2.py
        K = int(params.get('K', 4))
        use_tdist = params.get('use_tdist', True)
        n_starts = int(params.get('n_starts', 3))
        max_iter = int(params.get('max_iter', 20))
        
        # Configure global variables in the enhanced_hmm_em_v2 module if possible
        try:
            import enhanced_hmm_em_v2
            
            # Configure global settings
            enhanced_hmm_em_v2.CONFIG['use_skewed_t'] = params.get('use_skewed_t', False)
            enhanced_hmm_em_v2.CONFIG['use_time_varying_transition'] = params.get('use_time_varying_transition', False)
            enhanced_hmm_em_v2.CONFIG['regime_specific_egarch'] = params.get('use_regime_specific_egarch', True)
        except (ImportError, AttributeError):
            logger.warning("Could not configure enhanced_hmm_em_v2 global settings.")
        
        # Train the model
        pi, A, st_list, ll = self.train_hmm_once(
            features, K, n_starts=n_starts, max_iter=max_iter, 
            use_tdist=use_tdist, dims_egarch=dims_egarch, times=times
        )
        
        return (pi, A, st_list), ll
    
    def evaluate_cv_fold(self, features, train_idx, test_idx, params, dims_egarch, times=None):
        """
        Evaluate a single cross-validation fold.
        
        Args:
            features: Feature matrix
            train_idx: Training indices
            test_idx: Testing indices
            params: Parameter dictionary
            dims_egarch: EGARCH dimensions
            times: Optional time series
            
        Returns:
            Test log-likelihood
        """
        X_train = features[train_idx]
        X_test = features[test_idx]
        
        times_train = None if times is None else times[train_idx]
        times_test = None if times is None else times[test_idx]
        
        # Train model on training set
        (pi, A, st_list), train_ll = self.train_hmm_model(
            X_train, params, dims_egarch, times_train
        )
        
        # Evaluate on test set
        gamma, xi, scale = self.forward_backward(
            X_test, pi, A, st_list, 
            use_tdist=params.get('use_tdist', True), 
            dims_egarch=dims_egarch,
            times=times_test
        )
        
        test_ll = np.sum(np.log(scale))
        test_ll_per_sample = test_ll / len(X_test)
        
        return test_ll_per_sample
    
    def __call__(self, params):
        """
        Objective function to minimize.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Negative log-likelihood or other score to minimize
        """
        # Process parameters
        if isinstance(params, list):
            # Convert from skopt format
            param_names = [p.name for p in self.param_space]
            params = dict(zip(param_names, params))
        
        # Cast integer parameters
        for key in ['K', 'n_starts', 'max_iter', 'df_t_min', 'df_t_max', 'cv_fold_count']:
            if key in params:
                params[key] = int(params[key])
        
        # Select features based on parameters
        selected_features, selected_feature_cols = self.select_features(params)

        # Überprüfe, ob die Feature-Matrix gültig ist
        if selected_features is None or selected_features.shape[0] == 0 or selected_features.shape[1] == 0:
            logger.error(f"Leere Feature-Matrix nach Selektion: Shape={getattr(selected_features, 'shape', 'None')}")
            return 1e10  # Hoher Wert für fehlgeschlagenen Trial
        
        # Map EGARCH dimensions to selected features
        dims_egarch = self.select_egarch_dims(selected_feature_cols)
        
        # Create cv folds if using cross-validation
        if self.cross_validation:
            n_splits = params.get('cv_fold_count', self.n_splits)
            folds = self.create_cv_folds(selected_features, n_splits)
            
            # Evaluate each fold
            cv_scores = []
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                try:
                    test_ll = self.evaluate_cv_fold(
                        selected_features, train_idx, test_idx, 
                        params, dims_egarch, self.times
                    )
                    cv_scores.append(test_ll)
                except Exception as e:
                    logger.error(f"Error evaluating fold {fold_idx}: {str(e)}")
                    cv_scores.append(-1e10)  # Very negative score for failed fold
            
            # Use mean CV score
            if cv_scores:
                final_score = np.mean(cv_scores)
            else:
                final_score = -1e10
                
            # Log progress
            param_str = ', '.join([f"{k}={v}" for k, v in params.items() 
                                  if k in ['K', 'use_tdist', 'n_starts']])
            logger.info(f"CV Score: {final_score:.4f} with {param_str}")
            
        else:
            # Train on full dataset
            try:
                (pi, A, st_list), ll = self.train_hmm_model(
                    selected_features, params, dims_egarch, self.times
                )
                final_score = ll / len(selected_features)  # Per-sample log-likelihood
                
                # Log progress
                param_str = ', '.join([f"{k}={v}" for k, v in params.items() 
                                      if k in ['K', 'use_tdist', 'n_starts']])
                logger.info(f"LL: {final_score:.4f} with {param_str}")
                
            except Exception as e:
                logger.error(f"Error training model: {str(e)}")
                final_score = -1e10
        
        # Return negative score since optimizers minimize
        return -final_score

class EnsembleHMMObjective(HMMObjectiveBase):
    """Objective function for HMM ensemble optimization"""
    def __init__(self, features, feature_cols=None, dims_egarch=None, times=None, 
                cross_validation=True, n_splits=5):
        """Initialize the ensemble objective function"""
        super().__init__(features, feature_cols, dims_egarch, times, cross_validation, n_splits)
        
        if not self.has_ensemble:
            raise ImportError("HMMEnsemble not available. Cannot optimize ensemble.")
    
    def create_ensemble(self, features, params, dims_egarch, times=None):
        """
        Create an HMM ensemble with given parameters.
        
        Args:
            features: Feature matrix
            params: Parameter dictionary
            dims_egarch: EGARCH dimensions
            times: Optional time series
            
        Returns:
            HMMEnsemble instance and log-likelihood
        """
        # Get ensemble parameters
        n_models = int(params.get('n_models', 5))
        base_K = int(params.get('base_K', 4))
        K_variation = int(params.get('K_variation', 1))
        ensemble_type = params.get('ensemble_type', 'voting')
        
        # Generate K values for each model
        K_values = []
        for i in range(n_models):
            if K_variation > 0:
                # Random K within variation range
                K_values.append(max(2, base_K + np.random.randint(-K_variation, K_variation+1)))
            else:
                K_values.append(base_K)
        
        # Training parameters
        n_starts = int(params.get('n_starts_per_model', 2))
        max_iter = int(params.get('max_iter_per_model', 15))
        use_tdist = params.get('use_tdist', True)
        
        # Feature subset parameters
        feature_subset_ratio = params.get('feature_subset_ratio', 1.0)
        
        # Sample subset parameters for diversity
        sample_subset_ratio = params.get('sample_subset_ratio', 1.0)
        
        # Train each model
        models = []
        likelihoods = []
        
        for i, K in enumerate(K_values):
            # Optionally select feature subset for diversity
            if feature_subset_ratio < 1.0:
                n_features = int(features.shape[1] * feature_subset_ratio)
                feature_indices = np.random.choice(features.shape[1], n_features, replace=False)
                feature_subset = features[:, feature_indices]
                dims_egarch_subset = [d for d in dims_egarch if d in feature_indices]
            else:
                feature_subset = features
                dims_egarch_subset = dims_egarch
            
            # Optionally select sample subset for diversity
            if sample_subset_ratio < 1.0:
                n_samples = int(features.shape[0] * sample_subset_ratio)
                sample_indices = np.random.choice(features.shape[0], n_samples, replace=False)
                sample_indices.sort()  # Keep time order
                sample_subset = feature_subset[sample_indices]
                times_subset = None if times is None else times[sample_indices]
            else:
                sample_subset = feature_subset
                times_subset = times
            
            # Train the model
            try:
                pi, A, st_list, ll = self.train_hmm_once(
                    sample_subset, K, n_starts=n_starts, max_iter=max_iter, 
                    use_tdist=use_tdist, dims_egarch=dims_egarch_subset, times=times_subset
                )
                
                models.append((pi, A, st_list, ll))
                likelihoods.append(ll)
                
                logger.info(f"Trained ensemble model {i+1}/{n_models} with K={K}, LL={ll:.4f}")
                
            except Exception as e:
                logger.error(f"Error training ensemble model {i+1}: {str(e)}")
        
        # Calculate weights
        weights_method = params.get('weights_method', 'equal')
        
        if weights_method == 'likelihood':
            # Weights based on log-likelihood
            ll_array = np.array(likelihoods)
            ll_min = np.min(ll_array)
            normalized_ll = ll_array - ll_min
            weights = np.exp(normalized_ll)
            weights = weights / np.sum(weights)
        elif weights_method == 'cv_performance':
            # This would require separate CV for each model - use equal weights for now
            weights = np.ones(len(models)) / len(models)
        else:
            # Equal weights
            weights = np.ones(len(models)) / len(models)
        
        # Create ensemble
        ensemble = self.HMMEnsemble(models, weights, ensemble_type=ensemble_type)
        
        # Evaluate ensemble
        try:
            # Forward-backward for each model in ensemble
            ensemble_ll = 0
            for i, (pi, A, st_list, _) in enumerate(models):
                gamma, _, scale = self.forward_backward(
                    features, pi, A, st_list, 
                    use_tdist=use_tdist, 
                    dims_egarch=dims_egarch,
                    times=times
                )
                
                model_ll = np.sum(np.log(scale))
                ensemble_ll += weights[i] * model_ll
            
            # Normalize
            ensemble_ll /= len(features)
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            ensemble_ll = -1e10
        
        return ensemble, ensemble_ll
    
    def __call__(self, params):
        """
        Objective function to minimize.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Negative ensemble score
        """
        # Process parameters
        if isinstance(params, list):
            # Convert from skopt format
            param_names = [p.name for p in self.param_space]
            params = dict(zip(param_names, params))
        
        # Cast integer parameters
        for key in ['n_models', 'base_K', 'K_variation', 'n_starts_per_model', 
                   'max_iter_per_model']:
            if key in params:
                params[key] = int(params[key])
        
        # Select features based on parameters
        selected_features, selected_feature_cols = self.select_features(params)
        
        # Map EGARCH dimensions to selected features
        dims_egarch = self.select_egarch_dims(selected_feature_cols)
        
        # Create cv folds if using cross-validation
        if self.cross_validation:
            n_splits = params.get('cv_fold_count', self.n_splits)
            folds = self.create_cv_folds(selected_features, n_splits)
            
            # Evaluate each fold
            cv_scores = []
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                try:
                    X_train = selected_features[train_idx]
                    X_test = selected_features[test_idx]
                    
                    times_train = None if self.times is None else self.times[train_idx]
                    times_test = None if self.times is None else self.times[test_idx]
                    
                    # Train ensemble on training data
                    ensemble, _ = self.create_ensemble(X_train, params, dims_egarch, times_train)
                    
                    # Use ensemble to predict test data
                    # This is a simplified evaluation - in practice, you'd use the ensemble's
                    # prediction method to get predictions for test data
                    test_score = 0
                    for i, (pi, A, st_list, _) in enumerate(ensemble.models):
                        gamma, _, scale = self.forward_backward(
                            X_test, pi, A, st_list, 
                            use_tdist=params.get('use_tdist', True), 
                            dims_egarch=dims_egarch,
                            times=times_test
                        )
                        
                        model_ll = np.sum(np.log(scale))
                        test_score += ensemble.weights[i] * model_ll
                    
                    # Normalize
                    test_score /= len(X_test)
                    cv_scores.append(test_score)
                    
                except Exception as e:
                    logger.error(f"Error evaluating ensemble fold {fold_idx}: {str(e)}")
                    cv_scores.append(-1e10)  # Very negative score for failed fold
            
            # Use mean CV score
            if cv_scores:
                final_score = np.mean(cv_scores)
            else:
                final_score = -1e10
                
            # Log progress
            param_str = ', '.join([f"{k}={v}" for k, v in params.items() 
                                  if k in ['n_models', 'ensemble_type', 'base_K']])
            logger.info(f"Ensemble CV Score: {final_score:.4f} with {param_str}")
            
        else:
            # Train ensemble on full dataset
            try:
                ensemble, ensemble_ll = self.create_ensemble(
                    selected_features, params, dims_egarch, self.times
                )
                final_score = ensemble_ll
                
                # Log progress
                param_str = ', '.join([f"{k}={v}" for k, v in params.items() 
                                      if k in ['n_models', 'ensemble_type', 'base_K']])
                logger.info(f"Ensemble LL: {final_score:.4f} with {param_str}")
                
            except Exception as e:
                logger.error(f"Error training ensemble: {str(e)}")
                final_score = -1e10
        
        # Return negative score since optimizers minimize
        return -final_score

class TradingPerformanceObjective(HMMObjectiveBase):
    """
    Objective function using simulated trading performance.
    This evaluates the HMM model based on trading metrics like Sharpe ratio,
    win rate, etc. rather than just log-likelihood.
    """
    def __init__(self, features, prices, feature_cols=None, dims_egarch=None, times=None, 
                cross_validation=True, n_splits=5, metric='sharpe', 
                pip_value=0.01, risk_percent=1.0):
        """
        Initialize the trading performance objective.
        
        Args:
            features: Feature matrix [T, D]
            prices: Price series [T]
            feature_cols: Feature column names
            dims_egarch: EGARCH dimensions
            times: Optional time series
            cross_validation: Whether to use cross-validation
            n_splits: Number of cross-validation splits
            metric: Performance metric ('sharpe', 'win_rate', 'profit_factor', 'avg_profit', 'sortino')
            pip_value: Value of one pip
            risk_percent: Risk percentage per trade
        """
        super().__init__(features, feature_cols, dims_egarch, times, cross_validation, n_splits)
        
        self.prices = prices
        self.metric = metric
        self.pip_value = pip_value
        self.risk_percent = risk_percent
    
    def simulate_trading(self, features, prices, model, dims_egarch, times=None):
        """
        Simulate trading with the HMM model.
        
        Args:
            features: Feature matrix
            prices: Price series
            model: HMM model (pi, A, st_list)
            dims_egarch: EGARCH dimensions
            times: Optional time series
            
        Returns:
            dict: Trading performance metrics
        """
        # Extract model components
        pi, A, st_list = model
        
        # Run forward-backward to get state probabilities
        gamma, xi, scale = self.forward_backward(
            features, pi, A, st_list, 
            use_tdist=True,  # Assuming t-distribution here
            dims_egarch=dims_egarch,
            times=times
        )
        
        # Get most likely states
        states = np.argmax(gamma, axis=1)
        
        # Interpret states as trading signals
        # Simplified approach: odd states are bullish, even states are bearish
        signals = np.zeros(len(states))
        
        for i, state in enumerate(states):
            # Analyze state parameters to determine if bullish or bearish
            # This logic would need to be modified based on your state interpretation
            mu = st_list[state]["mu"]
            
            # Simple heuristic: if first return feature is positive, state is bullish
            if mu[0] > 0:
                signals[i] = 1  # Long signal
            elif mu[0] < 0:
                signals[i] = -1  # Short signal
            else:
                signals[i] = 0  # No signal
        
        # Simulate trades
        trades = []
        current_position = 0
        entry_price = 0
        entry_time = 0
        
        for i in range(1, len(signals)):
            # Trading logic
            if current_position == 0:
                # No position -> check for entry
                if signals[i] == 1:
                    # Enter long
                    current_position = 1
                    entry_price = prices[i]
                    entry_time = i
                elif signals[i] == -1:
                    # Enter short
                    current_position = -1
                    entry_price = prices[i]
                    entry_time = i
            elif current_position == 1:
                # Long position -> check for exit
                if signals[i] == -1:
                    # Exit long
                    exit_price = prices[i]
                    profit_pips = (exit_price - entry_price) / self.pip_value
                    
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": "LONG",
                        "profit_pips": profit_pips,
                        "duration": i - entry_time
                    })
                    
                    current_position = 0
            elif current_position == -1:
                # Short position -> check for exit
                if signals[i] == 1:
                    # Exit short
                    exit_price = prices[i]
                    profit_pips = (entry_price - exit_price) / self.pip_value
                    
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": "SHORT",
                        "profit_pips": profit_pips,
                        "duration": i - entry_time
                    })
                    
                    current_position = 0
        
        # Close any open position at the end
        if current_position == 1:
            exit_price = prices[-1]
            profit_pips = (exit_price - entry_price) / self.pip_value
            
            trades.append({
                "entry_time": entry_time,
                "exit_time": len(signals) - 1,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": "LONG",
                "profit_pips": profit_pips,
                "duration": len(signals) - 1 - entry_time
            })
        elif current_position == -1:
            exit_price = prices[-1]
            profit_pips = (entry_price - exit_price) / self.pip_value
            
            trades.append({
                "entry_time": entry_time,
                "exit_time": len(signals) - 1,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": "SHORT",
                "profit_pips": profit_pips,
                "duration": len(signals) - 1 - entry_time
            })
        
        # Calculate performance metrics
        if not trades:
            return {
                "sharpe": -10.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_profit": 0.0,
                "sortino": -10.0,
                "total_profit": 0.0,
                "num_trades": 0
            }
        
        # Extract trade profits
        profits = [trade["profit_pips"] for trade in trades]
        
        # Win rate
        win_count = sum(1 for p in profits if p > 0)
        win_rate = win_count / len(profits) if profits else 0
        
        # Profit factor
        gains = sum(p for p in profits if p > 0)
        losses = abs(sum(p for p in profits if p < 0))
        profit_factor = gains / losses if losses > 0 else 10.0  # Cap at 10 if no losses
        
        # Average profit
        avg_profit = np.mean(profits)
        
        # Total profit
        total_profit = sum(profits)
        
        # Sharpe ratio (annualized)
        returns = np.array(profits) / 100  # Normalize pips
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        sharpe *= np.sqrt(252 * 8)  # Annualized (assuming 8 trades per day, 252 trading days)
        
        # Sortino ratio (similar to Sharpe but only considers downside deviation)
        downside_returns = np.array([r for r in returns if r < 0])
        sortino = np.mean(returns) / np.std(downside_returns) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        sortino *= np.sqrt(252 * 8)  # Annualized
        
        return {
            "sharpe": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_profit": avg_profit,
            "sortino": sortino,
            "total_profit": total_profit,
            "num_trades": len(trades)
        }
    
    def evaluate_trading_cv_fold(self, features, prices, train_idx, test_idx, params, dims_egarch, times=None):
        """
        Evaluate trading performance on a single cross-validation fold.
        
        Args:
            features: Feature matrix
            prices: Price series
            train_idx: Training indices
            test_idx: Testing indices
            params: Parameter dictionary
            dims_egarch: EGARCH dimensions
            times: Optional time series
            
        Returns:
            Trading performance score
        """
        X_train = features[train_idx]
        X_test = features[test_idx]
        
        prices_train = prices[train_idx]
        prices_test = prices[test_idx]
        
        times_train = None if times is None else times[train_idx]
        times_test = None if times is None else times[test_idx]
        
        # Train model on training set
        model, _ = self.train_hmm_model(X_train, params, dims_egarch, times_train)
        
        # Evaluate trading performance on test set
        performance = self.simulate_trading(X_test, prices_test, model, dims_egarch, times_test)
        
        # Return the selected metric
        metric_value = performance.get(self.metric, -10.0)
        
        # Ensure the metric value is reasonable
        if self.metric == 'sharpe' or self.metric == 'sortino':
            # Limit extreme values
            metric_value = max(-10.0, min(10.0, metric_value))
        elif self.metric == 'profit_factor':
            # Limit extreme values
            metric_value = max(0.0, min(10.0, metric_value))
        
        return metric_value
    
    def __call__(self, params):
        """
        Objective function to maximize trading performance.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Negative performance score
        """
        # Process parameters
        if isinstance(params, list):
            # Convert from skopt format
            param_names = [p.name for p in self.param_space]
            params = dict(zip(param_names, params))
        
        # Cast integer parameters
        for key in ['K', 'n_starts', 'max_iter', 'df_t_min', 'df_t_max', 'cv_fold_count']:
            if key in params:
                params[key] = int(params[key])
        
        # Select features based on parameters
        selected_features, selected_feature_cols = self.select_features(params)
        
        # Map EGARCH dimensions to selected features
        dims_egarch = self.select_egarch_dims(selected_feature_cols)
        
        # Create cv folds if using cross-validation
        if self.cross_validation:
            n_splits = params.get('cv_fold_count', self.n_splits)
            folds = self.create_cv_folds(selected_features, n_splits)
            
            # Evaluate each fold
            cv_scores = []
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                try:
                    score = self.evaluate_trading_cv_fold(
                        selected_features, self.prices, train_idx, test_idx, 
                        params, dims_egarch, self.times
                    )
                    cv_scores.append(score)
                except Exception as e:
                    logger.error(f"Error evaluating trading fold {fold_idx}: {str(e)}")
                    cv_scores.append(-10.0)  # Very negative score for failed fold
            
            # Use mean CV score
            if cv_scores:
                final_score = np.mean(cv_scores)
            else:
                final_score = -10.0
                
            # Log progress
            param_str = ', '.join([f"{k}={v}" for k, v in params.items() 
                                  if k in ['K', 'use_tdist', 'n_starts']])
            logger.info(f"Trading {self.metric} CV Score: {final_score:.4f} with {param_str}")
            
        else:
            # Train on full dataset and evaluate in-sample
            try:
                model, _ = self.train_hmm_model(
                    selected_features, params, dims_egarch, self.times
                )
                
                performance = self.simulate_trading(
                    selected_features, self.prices, model, dims_egarch, self.times
                )
                
                final_score = performance.get(self.metric, -10.0)
                
                # Log progress
                param_str = ', '.join([f"{k}={v}" for k, v in params.items() 
                                      if k in ['K', 'use_tdist', 'n_starts']])
                logger.info(f"Trading {self.metric}: {final_score:.4f} with {param_str}")
                
            except Exception as e:
                logger.error(f"Error evaluating trading performance: {str(e)}")
                final_score = -10.0
        
        # Return negative score since optimizers minimize
        # (but only for metrics that should be maximized)
        if self.metric in ['sharpe', 'win_rate', 'profit_factor', 'avg_profit', 'sortino', 'total_profit']:
            return -final_score
        else:
            return final_score

###############################################################################
# Optimization Implementations
###############################################################################

class BayesianOptimizer:
    """
    Bayesian optimization using scikit-optimize.
    """
    def __init__(self, objective_func, param_space, n_calls=50, n_initial_points=10, 
                random_state=None):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            objective_func: Objective function to minimize
            param_space: Parameter space definition
            n_calls: Total number of function evaluations
            n_initial_points: Number of initial random evaluations
            random_state: Random state for reproducibility
        """
        self.objective_func = objective_func
        self.param_space = param_space
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for BayesianOptimizer")
    
    def optimize(self):
        """
        Run the optimization process.
        
        Returns:
            Best parameters and optimization results
        """
        # Create optimizer
        optimizer = Optimizer(
            dimensions=self.param_space,
            random_state=self.random_state,
            base_estimator="GP"  # Gaussian Process
        )
        
        # Run optimization
        result = None
        
        for i in range(self.n_calls):
            # Get next point to evaluate
            if i < self.n_initial_points:
                # Initial random search
                x = optimizer.ask()
                logger.info(f"Iteration {i+1}/{self.n_calls} (Random)")
            else:
                # Bayesian optimization
                x = optimizer.ask()
                logger.info(f"Iteration {i+1}/{self.n_calls} (Bayesian)")
            
            # Evaluate objective function
            try:
                y = self.objective_func(x)
                
                # Log result
                logger.info(f"Iteration {i+1} result: {y:.6f}")
                
                # Tell optimizer about result
                result = optimizer.tell(x, y)
                
                # Save checkpoint
                self._save_checkpoint(result, i)
                
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {str(e)}")
                
                # Tell optimizer about failure (high value)
                result = optimizer.tell(x, 1e10)
        
        # Process and save final results
        best_params = self._process_results(result)
        
        return best_params, result
    
    def _save_checkpoint(self, result, iteration):
        """
        Save a checkpoint of the optimization state.
        
        Args:
            result: Optimization result
            iteration: Current iteration
        """
        checkpoint_file = os.path.join(
            DEFAULT_CHECKPOINT_DIR, 
            f"bayesian_opt_checkpoint_{iteration}.pkl"
        )
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(result, f)
    
    def _process_results(self, result):
        """
        Process optimization results.
        
        Args:
            result: Optimization result
            
        Returns:
            Dict of best parameters
        """
        # Extract best parameters
        x_best = result.x
        
        # Convert to dictionary
        param_names = [p.name for p in self.param_space]
        best_params = dict(zip(param_names, x_best))
        
        # Save results
        results_file = os.path.join(
            DEFAULT_RESULTS_DIR, 
            f"bayesian_opt_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        
        # Convert numpy types to Python types
        best_params_json = {k: float(v) if isinstance(v, (np.float32, np.float64)) 
                           else int(v) if isinstance(v, (np.int32, np.int64)) 
                           else v 
                           for k, v in best_params.items()}
        
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": best_params_json,
                "best_value": float(result.fun),
                "all_values": [float(v) for v in result.func_vals],
                "n_iterations": len(result.func_vals),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        # Visualize results if possible
        self._visualize_results(result)
        
        return best_params
    
    def _visualize_results(self, result):
        """
        Visualize optimization results.
        
        Args:
            result: Optimization result
        """
        try:
            # Create plot directory
            plot_dir = os.path.join(DEFAULT_RESULTS_DIR, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Plot convergence
            plt.figure()
            plt.plot(result.func_vals)
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title('Optimization Convergence')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 
                                     f"convergence_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            
            # Plot evaluations
            if len(result.x_iters) > 0 and len(self.param_space) <= 10:
                _ = plot_evaluations(result)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 
                                         f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            
            # Plot objective for important parameters
            if len(result.x_iters) > 0 and len(self.param_space) <= 10:
                _ = plot_objective(result)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 
                                         f"objective_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            
        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")

class OptunaOptimizer:
    """
    Hyperparameter optimization using Optuna.
    """
    def __init__(self, objective_func, param_space_func, n_trials=50, 
                timeout=None, n_jobs=1, show_progress_bar=True):
        """
        Initialize the Optuna optimizer.
        
        Args:
            objective_func: Original objective function
            param_space_func: Function that defines the parameter space
            n_trials: Number of trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            show_progress_bar: Whether to show progress bar
        """
        self.base_objective_func = objective_func
        self.param_space_func = param_space_func
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.show_progress_bar = show_progress_bar
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for OptunaOptimizer")
    
    def objective(self, trial):
        """
        Optuna objective function wrapper.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Objective value
        """
        # Define parameter space
        params = self.param_space_func(trial)
        
        # Evaluate objective function
        try:
            return self.base_objective_func(params)
        except Exception as e:
            logger.error(f"Error in Optuna trial: {str(e)}")
            return 1e10  # High value for failed trials
    
    def optimize(self):
        """
        Run the optimization process.
        
        Returns:
            Best parameters and optimization results
        """
        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=self.show_progress_bar
        )
        
        # Process and save results
        best_params = self._process_results(study)
        
        return best_params, study
    
    def _process_results(self, study):
        """
        Process optimization results.
        
        Args:
            study: Optuna study
            
        Returns:
            Dict of best parameters
        """
        # Extract best parameters
        best_params = study.best_params
        
        # Save results
        results_file = os.path.join(
            DEFAULT_RESULTS_DIR, 
            f"optuna_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        
        # Create a dictionary with trial information
        trial_data = []
        for trial in study.trials:
            if trial.value is not None:
                trial_data.append({
                    "number": trial.number,
                    "params": {k: float(v) if isinstance(v, (np.float32, np.float64)) 
                              else int(v) if isinstance(v, (np.int32, np.int64)) 
                              else v 
                              for k, v in trial.params.items()},
                    "value": float(trial.value),
                    "datetime": trial.datetime.isoformat() if hasattr(trial, 'datetime') else None
                })
        
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": best_params,
                "best_value": float(study.best_value),
                "n_trials": len(study.trials),
                "trials": trial_data,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        # Visualize results if possible
        self._visualize_results(study)
        
        return best_params
    
    def _visualize_results(self, study):
        """
        Visualize optimization results.
        
        Args:
            study: Optuna study
        """
        try:
            # Create plot directory
            plot_dir = os.path.join(DEFAULT_RESULTS_DIR, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Plot optimization history
            plt.figure()
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 
                                     f"optuna_history_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            
            # Plot parameter importances
            plt.figure()
            try:
                optuna.visualization.matplotlib.plot_param_importances(study)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 
                                         f"optuna_importance_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            except Exception:
                logger.warning("Could not plot parameter importances, skipping")
            
            # Plot parallel coordinate plot
            plt.figure(figsize=(12, 8))
            try:
                optuna.visualization.matplotlib.plot_parallel_coordinate(study)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 
                                         f"optuna_parallel_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            except Exception:
                logger.warning("Could not plot parallel coordinates, skipping")
            
            # Plot slice plot
            plt.figure(figsize=(12, 8))
            try:
                optuna.visualization.matplotlib.plot_slice(study)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 
                                         f"optuna_slice_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            except Exception:
                logger.warning("Could not plot slice plot, skipping")
            
        except Exception as e:
            logger.error(f"Error visualizing Optuna results: {str(e)}")

class HyperoptOptimizer:
    """
    Hyperparameter optimization using Hyperopt.
    """
    def __init__(self, objective_func, param_space, max_evals=50, 
                timeout=None, show_progressbar=True):
        """
        Initialize the Hyperopt optimizer.
        
        Args:
            objective_func: Original objective function
            param_space: Hyperopt parameter space
            max_evals: Maximum number of evaluations
            timeout: Timeout in seconds
            show_progressbar: Whether to show progress bar
        """
        self.base_objective_func = objective_func
        self.param_space = param_space
        self.max_evals = max_evals
        self.timeout = timeout
        self.show_progressbar = show_progressbar
        
        if not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is required for HyperoptOptimizer")
    
    def objective(self, params):
        """
        Hyperopt objective function wrapper.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Dict with objective value and status
        """
        # Fix integer parameters
        for key in ['K', 'n_starts', 'max_iter', 'df_t_min', 'df_t_max', 'cv_fold_count', 
                   'n_models', 'base_K', 'K_variation', 'sequence_length', 'n_starts_per_model', 
                   'max_iter_per_model', 'early_stopping_patience']:
            if key in params:
                params[key] = int(params[key])
        
        # Evaluate objective function
        try:
            value = self.base_objective_func(params)
            
            # Log result
            logger.info(f"Hyperopt trial result: {value:.6f}")
            
            return {
                'loss': value,
                'status': STATUS_OK,
                'params': params
            }
        except Exception as e:
            logger.error(f"Error in Hyperopt trial: {str(e)}")
            
            return {
                'loss': 1e10,
                'status': STATUS_OK,
                'params': params
            }
    
    def optimize(self):
        """
        Run the optimization process.
        
        Returns:
            Best parameters and optimization results
        """
        # Create trials object to store results
        trials = Trials()
        
        # Run optimization
        result = fmin(
            fn=self.objective,
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            show_progressbar=self.show_progressbar,
            rstate=np.random.RandomState(42)
        )
        
        # Process and save results
        best_params = self._process_results(result, trials)
        
        return best_params, trials
    
    def _process_results(self, result, trials):
        """
        Process optimization results.
        
        Args:
            result: Hyperopt result
            trials: Hyperopt trials
            
        Returns:
            Dict of best parameters
        """
        # Extract best parameters
        best_params = space_eval(self.param_space, result)
        
        # Fix integer parameters
        for key in ['K', 'n_starts', 'max_iter', 'df_t_min', 'df_t_max', 'cv_fold_count', 
                   'n_models', 'base_K', 'K_variation', 'sequence_length', 'n_starts_per_model', 
                   'max_iter_per_model', 'early_stopping_patience']:
            if key in best_params:
                best_params[key] = int(best_params[key])
        
        # Save results
        results_file = os.path.join(
            DEFAULT_RESULTS_DIR, 
            f"hyperopt_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        
        # Create a dictionary with trial information
        trial_data = []
        for trial in trials.trials:
            if 'result' in trial and 'loss' in trial['result']:
                trial_data.append({
                    "tid": trial['tid'],
                    "params": {k: float(v) if isinstance(v, (np.float32, np.float64)) 
                              else int(v) if isinstance(v, (np.int32, np.int64)) 
                              else v 
                              for k, v in trial['misc']['vals'].items()},
                    "value": float(trial['result']['loss']),
                    "status": trial['result']['status']
                })
        
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": {k: float(v) if isinstance(v, (np.float32, np.float64)) 
                               else int(v) if isinstance(v, (np.int32, np.int64)) 
                               else v 
                               for k, v in best_params.items()},
                "best_value": float(trials.best_trial['result']['loss']),
                "n_trials": len(trials.trials),
                "trials": trial_data,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        # Visualize results if possible
        self._visualize_results(trials)
        
        return best_params
    
    def _visualize_results(self, trials):
        """
        Visualize optimization results.
        
        Args:
            trials: Hyperopt trials
        """
        try:
            # Create plot directory
            plot_dir = os.path.join(DEFAULT_RESULTS_DIR, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Plot convergence
            plt.figure()
            plt.plot([t['result']['loss'] for t in trials.trials])
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.title('Hyperopt Optimization Convergence')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 
                                     f"hyperopt_convergence_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            
            # Plot parameter distributions for a few important parameters
            important_params = ['K', 'use_tdist', 'n_starts', 'use_skewed_t']
            
            param_values = {}
            for param in important_params:
                try:
                    param_values[param] = [
                        space_eval(self.param_space, t['misc']['vals'])[param]
                        for t in trials.trials
                        if 'result' in t and 'loss' in t['result']
                    ]
                except (KeyError, ValueError):
                    continue
            
            # Plot distributions
            if param_values:
                plt.figure(figsize=(12, 8))
                n_params = len(param_values)
                rows = (n_params + 1) // 2
                cols = min(2, n_params)
                
                for i, (param, values) in enumerate(param_values.items()):
                    plt.subplot(rows, cols, i+1)
                    
                    if isinstance(values[0], (int, float, np.int32, np.int64, np.float32, np.float64)):
                        plt.hist(values, bins=10)
                    else:
                        # Categorical
                        from collections import Counter
                        counts = Counter(values)
                        plt.bar(range(len(counts)), list(counts.values()), align='center')
                        plt.xticks(range(len(counts)), list(counts.keys()))
                    
                    plt.title(param)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 
                                         f"hyperopt_params_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            
        except Exception as e:
            logger.error(f"Error visualizing Hyperopt results: {str(e)}")

class RayTuneOptimizer:
    """
    Hyperparameter optimization using Ray Tune.
    """
    def __init__(self, objective_func, param_space, num_samples=50, 
                max_concurrent=4, search_alg='bayesopt', scheduler='asha'):
        """
        Initialize the Ray Tune optimizer.
        
        Args:
            objective_func: Original objective function
            param_space: Ray Tune parameter space
            num_samples: Number of samples to evaluate
            max_concurrent: Maximum number of concurrent trials
            search_alg: Search algorithm ('bayesopt', 'optuna', 'random')
            scheduler: Scheduler for trials ('asha', 'hyperband', 'pbt', None)
        """
        self.base_objective_func = objective_func
        self.param_space = param_space
        self.num_samples = num_samples
        self.max_concurrent = max_concurrent
        self.search_alg_name = search_alg
        self.scheduler_name = scheduler
        
        if not RAY_AVAILABLE:
            raise ImportError("ray is required for RayTuneOptimizer")
    
    def objective(self, config):
        """
        Ray Tune objective function wrapper.
        
        Args:
            config: Parameter configuration
            
        Returns:
            Reports objective value to Ray Tune
        """
        # Fix integer parameters
        for key in ['K', 'n_starts', 'max_iter', 'df_t_min', 'df_t_max', 'cv_fold_count', 
                   'n_models', 'base_K', 'K_variation', 'sequence_length', 'n_starts_per_model', 
                   'max_iter_per_model', 'early_stopping_patience']:
            if key in config:
                config[key] = int(config[key])
        
        # Evaluate objective function
        try:
            value = self.base_objective_func(config)
            
            # Report to Ray Tune
            tune.report(objective=value)
            
        except Exception as e:
            logger.error(f"Error in Ray Tune trial: {str(e)}")
            
            # Report failure (high value)
            tune.report(objective=1e10)
    
    def optimize(self):
        """
        Run the optimization process.
        
        Returns:
            Best parameters and optimization results
        """
        # Create search algorithm
        search_alg = None
        
        if self.search_alg_name == 'bayesopt':
            search_alg = BayesOptSearch()
        elif self.search_alg_name == 'optuna':
            search_alg = OptunaSearch()
        
        # Add concurrency limiter
        if search_alg is not None:
            search_alg = ConcurrencyLimiter(
                search_alg, max_concurrent=self.max_concurrent
            )
        
        # Create scheduler
        scheduler = None
        
        if self.scheduler_name == 'asha':
            scheduler = ASHAScheduler(metric='objective', mode='min')
        elif self.scheduler_name == 'hyperband':
            scheduler = HyperBandScheduler(metric='objective', mode='min')
        
        # Run optimization
        analysis = tune.run(
            self.objective,
            config=self.param_space,
            num_samples=self.num_samples,
            search_alg=search_alg,
            scheduler=scheduler,
            resources_per_trial={"cpu": 1},
            verbose=1,
            local_dir=DEFAULT_CHECKPOINT_DIR
        )
        
        # Process and save results
        best_params = self._process_results(analysis)
        
        return best_params, analysis
    
    def _process_results(self, analysis):
        """
        Process optimization results.
        
        Args:
            analysis: Ray Tune analysis
            
        Returns:
            Dict of best parameters
        """
        # Extract best parameters
        best_config = analysis.best_config
        
        # Fix integer parameters
        for key in ['K', 'n_starts', 'max_iter', 'df_t_min', 'df_t_max', 'cv_fold_count', 
                   'n_models', 'base_K', 'K_variation', 'sequence_length', 'n_starts_per_model', 
                   'max_iter_per_model', 'early_stopping_patience']:
            if key in best_config:
                best_config[key] = int(best_config[key])
        
        # Save results
        results_file = os.path.join(
            DEFAULT_RESULTS_DIR, 
            f"ray_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        
        # Get all trials data
        trials_data = []
        for trial in analysis.trials:
            if trial.last_result and 'objective' in trial.last_result:
                trials_data.append({
                    "trial_id": trial.trial_id,
                    "params": {k: float(v) if isinstance(v, (np.float32, np.float64)) 
                              else int(v) if isinstance(v, (np.int32, np.int64)) 
                              else v 
                              for k, v in trial.config.items()},
                    "value": float(trial.last_result['objective']),
                    "iterations": trial.last_result.get('training_iteration', 1)
                })
        
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": {k: float(v) if isinstance(v, (np.float32, np.float64)) 
                               else int(v) if isinstance(v, (np.int32, np.int64)) 
                               else v 
                               for k, v in best_config.items()},
                "best_value": float(analysis.best_result['objective']),
                "n_trials": len(analysis.trials),
                "trials": trials_data,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        # Visualize results if possible
        self._visualize_results(analysis)
        
        return best_config
    
    def _visualize_results(self, analysis):
        """
        Visualize optimization results.
        
        Args:
            analysis: Ray Tune analysis
        """
        try:
            # Create plot directory
            plot_dir = os.path.join(DEFAULT_RESULTS_DIR, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Plot convergence
            plt.figure(figsize=(10, 6))
            
            # Get all results
            df = analysis.results_df
            
            # Sort by time
            if 'training_iteration' in df.columns:
                df = df.sort_values('training_iteration')
            
            # Plot convergence
            plt.plot(df['objective'].values)
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.title('Ray Tune Optimization Convergence')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 
                                    f"ray_convergence_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            
            # Plot parameter importance if possible
            try:
                # Create a dataframe with parameters and results
                param_df = pd.DataFrame()
                
                # Get important parameters
                important_params = ['K', 'n_starts', 'use_tdist', 'dims_egarch_count']
                
                for param in important_params:
                    if param in df.columns:
                        param_df[param] = df[param]
                
                if len(param_df.columns) > 0:
                    param_df['objective'] = df['objective']
                    
                    # Plot pair grid
                    plt.figure(figsize=(12, 10))
                    sns.pairplot(param_df, height=2.5, diag_kind='kde', 
                                plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                                diag_kws={'alpha': 0.6})
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, 
                                            f"ray_pairgrid_{datetime.now().strftime('%Y%m%d_%H%M')}.png"))
            except Exception as e:
                logger.warning(f"Could not plot parameter importance: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error visualizing Ray Tune results: {str(e)}")

###############################################################################
# Parameter Space Definitions
###############################################################################

def create_skopt_param_space():
    """
    Create parameter space for scikit-optimize.
    
    Returns:
        List of parameter space dimensions
    """
    return [
        # Number of states
        Integer(2, 8, name='K'),
        
        # Distribution selection
        Categorical([True, False], name='use_tdist'),
        Categorical([True, False], name='use_skewed_t'),
        Categorical([True, False], name='use_hybrid_distribution'),
        
        # EGARCH parameters
        Integer(1, 4, name='dims_egarch_count'),
        
        # Training parameters
        Integer(3, 10, name='n_starts'),
        Integer(10, 50, name='max_iter'),
        Categorical([True, False], name='early_stopping'),
        
        # T-distribution parameters
        Integer(3, 15, name='df_t_min'),
        Integer(5, 30, name='df_t_max'),
        
        # Cross-validation parameters
        Integer(3, 5, name='cv_fold_count'),
        
        # Ensemble parameters
        Integer(2, 5, name='n_models'),
        Categorical(['voting', 'bayes', 'adaptive'], name='ensemble_type'),
        Integer(2, 6, name='base_K'),
        Integer(0, 2, name='K_variation'),
        
        # Hybrid model parameters
        Integer(5, 20, name='sequence_length'),
        Integer(32, 128, name='lstm_units'),
        Integer(16, 64, name='dense_units'),
        Integer(1, 5, name='n_starts_per_model'),
        Integer(5, 20, name='max_iter_per_model'),
        Integer(2, 10, name='early_stopping_patience'),
        Real(0.0001, 0.01, name='learning_rate'),
        
        # Feature selection parameters
        Categorical([True, False], name='use_feature_selection'),
        Categorical([True, False], name='use_pca'),
        Integer(3, 10, name='pca_components')
    ]

def create_optuna_param_space(trial):
    """
    Create parameter space for Optuna.
    
    Args:
        trial: Optuna trial
        
    Returns:
        Dict of parameters
    """
    params = {
        # Number of states
        'K': trial.suggest_int('K', 2, 8),
        
        # Distribution selection
        'use_tdist': trial.suggest_categorical('use_tdist', [True, False]),
        'use_skewed_t': trial.suggest_categorical('use_skewed_t', [True, False]),
        'use_hybrid_distribution': trial.suggest_categorical('use_hybrid_distribution', [True, False]),
        
        # EGARCH parameters
        'dims_egarch_count': trial.suggest_int('dims_egarch_count', 1, 4),
        
        # Training parameters
        'n_starts': trial.suggest_int('n_starts', 3, 10),
        'max_iter': trial.suggest_int('max_iter', 10, 50),
        'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
        
        # T-distribution parameters
        'df_t_min': trial.suggest_int('df_t_min', 3, 15),
        'df_t_max': trial.suggest_int('df_t_max', 5, 30),
        
        # Cross-validation parameters
        'cv_fold_count': trial.suggest_int('cv_fold_count', 3, 5),
        
        # Ensemble parameters
        'n_models': trial.suggest_int('n_models', 2, 5),
        'ensemble_type': trial.suggest_categorical('ensemble_type', ['voting', 'bayes', 'adaptive']),
        'base_K': trial.suggest_int('base_K', 2, 6),
        'K_variation': trial.suggest_int('K_variation', 0, 2),
        
        # Hybrid model parameters
        'sequence_length': trial.suggest_int('sequence_length', 5, 20),
        'lstm_units': trial.suggest_int('lstm_units', 32, 128),
        'dense_units': trial.suggest_int('dense_units', 16, 64),
        'n_starts_per_model': trial.suggest_int('n_starts_per_model', 1, 5),
        'max_iter_per_model': trial.suggest_int('max_iter_per_model', 5, 20),
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        
        # Feature selection parameters
        'use_feature_selection': trial.suggest_categorical('use_feature_selection', [True, False]),
        'use_pca': trial.suggest_categorical('use_pca', [True, False]),
        'pca_components': trial.suggest_int('pca_components', 3, 10)
    }
    
    return params

def create_hyperopt_param_space():
    """
    Create parameter space for Hyperopt.
    
    Returns:
        Dict of parameter space
    """
    return {
        # Number of states
        'K': hp.quniform('K', 2, 8, 1),
        
        # Distribution selection
        'use_tdist': hp.choice('use_tdist', [True, False]),
        'use_skewed_t': hp.choice('use_skewed_t', [True, False]),
        'use_hybrid_distribution': hp.choice('use_hybrid_distribution', [True, False]),
        
        # EGARCH parameters
        'dims_egarch_count': hp.quniform('dims_egarch_count', 1, 4, 1),
        
        # Training parameters
        'n_starts': hp.quniform('n_starts', 3, 10, 1),
        'max_iter': hp.quniform('max_iter', 10, 50, 1),
        'early_stopping': hp.choice('early_stopping', [True, False]),
        
        # T-distribution parameters
        'df_t_min': hp.quniform('df_t_min', 3, 15, 1),
        'df_t_max': hp.quniform('df_t_max', 5, 30, 1),
        
        # Cross-validation parameters
        'cv_fold_count': hp.quniform('cv_fold_count', 3, 5, 1),
        
        # Ensemble parameters
        'n_models': hp.quniform('n_models', 2, 5, 1),
        'ensemble_type': hp.choice('ensemble_type', ['voting', 'bayes', 'adaptive']),
        'base_K': hp.quniform('base_K', 2, 6, 1),
        'K_variation': hp.quniform('K_variation', 0, 2, 1),
        
        # Hybrid model parameters
        'sequence_length': hp.quniform('sequence_length', 5, 20, 1),
        'lstm_units': hp.quniform('lstm_units', 32, 128, 8),
        'dense_units': hp.quniform('dense_units', 16, 64, 8),
        'n_starts_per_model': hp.quniform('n_starts_per_model', 1, 5, 1),
        'max_iter_per_model': hp.quniform('max_iter_per_model', 5, 20, 1),
        'early_stopping_patience': hp.quniform('early_stopping_patience', 2, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
        
        # Feature selection parameters
        'use_feature_selection': hp.choice('use_feature_selection', [True, False]),
        'use_pca': hp.choice('use_pca', [True, False]),
        'pca_components': hp.quniform('pca_components', 3, 10, 1)
    }

def create_ray_param_space():
    """
    Create parameter space for Ray Tune.
    
    Returns:
        Dict of parameter space
    """
    return {
        # Number of states
        'K': tune.randint(2, 9),
        
        # Distribution selection
        'use_tdist': tune.choice([True, False]),
        'use_skewed_t': tune.choice([True, False]),
        'use_hybrid_distribution': tune.choice([True, False]),
        
        # EGARCH parameters
        'dims_egarch_count': tune.randint(1, 5),
        
        # Training parameters
        'n_starts': tune.randint(3, 11),
        'max_iter': tune.randint(10, 51),
        'early_stopping': tune.choice([True, False]),
        
        # T-distribution parameters
        'df_t_min': tune.randint(3, 16),
        'df_t_max': tune.randint(5, 31),
        
        # Cross-validation parameters
        'cv_fold_count': tune.randint(3, 6),
        
        # Ensemble parameters
        'n_models': tune.randint(2, 6),
        'ensemble_type': tune.choice(['voting', 'bayes', 'adaptive']),
        'base_K': tune.randint(2, 7),
        'K_variation': tune.randint(0, 3),
        
        # Hybrid model parameters
        'sequence_length': tune.randint(5, 21),
        'lstm_units': tune.randint(32, 129),
        'dense_units': tune.randint(16, 65),
        'n_starts_per_model': tune.randint(1, 6),
        'max_iter_per_model': tune.randint(5, 21),
        'early_stopping_patience': tune.randint(2, 11),
        'learning_rate': tune.loguniform(0.0001, 0.01),
        
        # Feature selection parameters
        'use_feature_selection': tune.choice([True, False]),
        'use_pca': tune.choice([True, False]),
        'pca_components': tune.randint(3, 11)
    }

###############################################################################
# Objective Functions
###############################################################################

def create_likelihood_objective(features, K_range=[2, 8], use_cross_validation=True):
    """
    Create objective function based on log-likelihood.
    
    Args:
        features: Feature matrix
        K_range: Range of states to consider
        use_cross_validation: Whether to use cross-validation
        
    Returns:
        Objective function
    """
    from enhanced_hmm_em_v2 import train_hmm_once, forward_backward
    
    def objective(params):
        # Extract parameters
        K = int(params.get('K', 4))
        n_starts = int(params.get('n_starts', 5))
        max_iter = int(params.get('max_iter', 20))
        use_tdist = params.get('use_tdist', True)
        use_skewed_t = params.get('use_skewed_t', False)
        use_hybrid_distribution = params.get('use_hybrid_distribution', False)
        dims_egarch_count = int(params.get('dims_egarch_count', 4))
        early_stopping = params.get('early_stopping', True)
        
        # Validate K against range
        if K < K_range[0] or K > K_range[1]:
            K = max(min(K, K_range[1]), K_range[0])
        
        # Set up EGARCH dimensions
        dims_egarch = list(range(dims_egarch_count))
        
        # Set up training parameters
        train_kwargs = {
            'n_starts': n_starts,
            'max_iter': max_iter,
            'use_tdist': use_tdist,
            'dims_egarch': dims_egarch,
            'early_stopping': early_stopping
        }
        
        # Configure distribution type
        if use_tdist:
            CONFIG = {
                'use_skewed_t': use_skewed_t,
                'use_hybrid_distribution': use_hybrid_distribution
            }
            
            # Temporarily modify global CONFIG
            from enhanced_hmm_em_v2 import CONFIG as GLOBAL_CONFIG
            original_config = GLOBAL_CONFIG.copy()
            GLOBAL_CONFIG.update(CONFIG)
        
        # Perform training with cross-validation if enabled
        if use_cross_validation:
            cv_fold_count = int(params.get('cv_fold_count', 5))
            
            # Cross-validation
            T, D = features.shape
            fold_size = T // cv_fold_count
            
            cv_ll = []
            
            for fold in range(cv_fold_count):
                # Define train/test split
                test_start = fold * fold_size
                test_end = (fold + 1) * fold_size if fold < cv_fold_count - 1 else T
                
                train_indices = list(range(0, test_start)) + list(range(test_end, T))
                test_indices = list(range(test_start, test_end))
                
                X_train = features[train_indices]
                X_test = features[test_indices]
                
                # Train model
                pi, A, st_list, _ = train_hmm_once(X_train, K, **train_kwargs)
                
                # Evaluate on test set
                _, _, scale = forward_backward(X_test, pi, A, st_list, use_tdist=use_tdist, 
                                             dims_egarch=dims_egarch)
                
                # Log-likelihood on test set
                test_ll = np.sum(np.log(scale))
                cv_ll.append(test_ll / len(X_test))  # Normalize by sequence length
            
            # Average log-likelihood across folds
            avg_ll = np.mean(cv_ll)
            
            # Convert to a minimization problem (negative log-likelihood)
            result = -avg_ll
            
        else:
            # Train on all data
            pi, A, st_list, ll = train_hmm_once(features, K, **train_kwargs)
            
            # Convert to a minimization problem (negative log-likelihood)
            result = -ll
        
        # Restore original CONFIG if modified
        if use_tdist:
            from enhanced_hmm_em_v2 import CONFIG as GLOBAL_CONFIG
            GLOBAL_CONFIG.update(original_config)
        
        return result
    
    return objective

def create_trading_performance_objective(features, prices, trade_evaluator, 
                                        lookback_window=50, forward_window=20):
    """
    Create objective function based on trading performance.
    
    Args:
        features: Feature matrix
        prices: Price series
        trade_evaluator: Function to evaluate trades
        lookback_window: Window for feature lookback
        forward_window: Window for forward evaluation
        
    Returns:
        Objective function
    """
    from enhanced_hmm_em_v2 import train_hmm_once, forward_backward
    
    def objective(params):
        # Extract parameters
        K = int(params.get('K', 4))
        n_starts = int(params.get('n_starts', 5))
        max_iter = int(params.get('max_iter', 20))
        use_tdist = params.get('use_tdist', True)
        use_skewed_t = params.get('use_skewed_t', False)
        use_hybrid_distribution = params.get('use_hybrid_distribution', False)
        dims_egarch_count = int(params.get('dims_egarch_count', 4))
        
        # Set up EGARCH dimensions
        dims_egarch = list(range(dims_egarch_count))
        
        # Set up ensemble parameters
        use_ensemble = params.get('n_models', 1) > 1
        
        if use_ensemble:
            n_models = int(params.get('n_models', 3))
            ensemble_type = params.get('ensemble_type', 'voting')
            base_K = int(params.get('base_K', 4))
            K_variation = int(params.get('K_variation', 1))
            
            # Create ensemble
            models = []
            
            for i in range(n_models):
                # Vary K if specified
                model_K = max(2, base_K + np.random.randint(-K_variation, K_variation + 1))
                
                # Train model
                pi, A, st_list, ll = train_hmm_once(
                    features, model_K,
                    n_starts=n_starts,
                    max_iter=max_iter,
                    use_tdist=use_tdist,
                    dims_egarch=dims_egarch
                )
                
                models.append((pi, A, st_list, ll))
            
            # Use ensemble for trading
            from ensemble_hmm_impl import HMMEnsemble
            ensemble = HMMEnsemble(models, ensemble_type=ensemble_type)
            
            # Evaluate trading performance with ensemble
            profits = []
            
            for t in range(lookback_window, len(features) - forward_window):
                # Get sequence for prediction
                feature_seq = features[t-lookback_window:t]
                
                # Get ensemble prediction
                state, label, confidence = ensemble.predict(feature_seq)
                
                # Generate trading signal
                if "Bullish" in label:
                    signal = "LONG"
                elif "Bearish" in label:
                    signal = "SHORT"
                else:
                    signal = "NONE"
                
                # Evaluate trade
                if signal != "NONE":
                    entry_price = prices[t]
                    exit_price = prices[t + forward_window]
                    
                    # Calculate profit
                    if signal == "LONG":
                        profit = (exit_price - entry_price) / entry_price
                    else:  # SHORT
                        profit = (entry_price - exit_price) / entry_price
                    
                    profits.append(profit)
            
            # Calculate performance metrics
            if profits:
                # Sharpe ratio (simplified, assumes zero risk-free rate)
                mean_return = np.mean(profits)
                std_return = np.std(profits) if len(profits) > 1 else 1e-6
                sharpe = mean_return / std_return if std_return > 0 else 0
                
                # Win rate
                win_rate = np.mean([p > 0 for p in profits])
                
                # Profit factor
                gains = sum([p for p in profits if p > 0])
                losses = sum([-p for p in profits if p < 0])
                profit_factor = gains / losses if losses > 0 else gains
                
                # Combined score (to be minimized)
                score = -sharpe * win_rate * profit_factor
            else:
                score = 0
            
        else:
            # Train single model
            pi, A, st_list, _ = train_hmm_once(
                features, K,
                n_starts=n_starts,
                max_iter=max_iter,
                use_tdist=use_tdist,
                dims_egarch=dims_egarch
            )
            
            # Evaluate trading performance
            profits = []
            
            for t in range(lookback_window, len(features) - forward_window):
                # Get sequences for prediction
                feature_tminus1 = features[t-1]
                feature_t = features[t]
                
                # Initialize HMM state if first iteration
                if t == lookback_window:
                    # Initialize with feature sequence
                    from enhanced_live_inference_mt5_v3 import EnhancedLiveHMMMt5
                    live_hmm = EnhancedLiveHMMMt5({
                        "K": K,
                        "pi": pi,
                        "A": A,
                        "st_params": st_list
                    }, dims_egarch=dims_egarch)
                    
                    live_hmm.partial_init(feature_tminus1)
                
                # Update HMM state
                state_info = live_hmm.partial_step(
                    feature_tminus1, feature_t,
                    time_info=None,
                    current_price=prices[t]
                )
                
                # Generate trading signal
                if "Bullish" in state_info["state_label"]:
                    signal = "LONG"
                elif "Bearish" in state_info["state_label"]:
                    signal = "SHORT"
                else:
                    signal = "NONE"
                
                # Evaluate trade
                if signal != "NONE":
                    entry_price = prices[t]
                    exit_price = prices[t + forward_window]
                    
                    # Calculate profit
                    if signal == "LONG":
                        profit = (exit_price - entry_price) / entry_price
                    else:  # SHORT
                        profit = (entry_price - exit_price) / entry_price
                    
                    profits.append(profit)
            
            # Calculate performance metrics
            if profits:
                # Sharpe ratio (simplified, assumes zero risk-free rate)
                mean_return = np.mean(profits)
                std_return = np.std(profits) if len(profits) > 1 else 1e-6
                sharpe = mean_return / std_return if std_return > 0 else 0
                
                # Win rate
                win_rate = np.mean([p > 0 for p in profits])
                
                # Profit factor
                gains = sum([p for p in profits if p > 0])
                losses = sum([-p for p in profits if p < 0])
                profit_factor = gains / losses if losses > 0 else gains
                
                # Combined score (to be minimized)
                score = -sharpe * win_rate * profit_factor
            else:
                score = 0
        
        # Return score (negative for minimization)
        return score
    
    return objective

def create_hybrid_objective(features, prices, hybrid_evaluator, 
                           train_ratio=0.8, sequence_length=10):
    """
    Create objective function for hybrid HMM + neural network models.
    
    Args:
        features: Feature matrix
        prices: Price series
        hybrid_evaluator: Function to evaluate hybrid model
        train_ratio: Ratio of data to use for training
        sequence_length: Sequence length for LSTM
        
    Returns:
        Objective function
    """
    from enhanced_hmm_em_v2 import train_hmm_once
    if HybridModel is None:
        raise ImportError("hybrid_model module is required for hybrid optimization")
    
    def objective(params):
        # Extract parameters
        K = int(params.get('K', 4))
        n_starts = int(params.get('n_starts', 5))
        max_iter = int(params.get('max_iter', 20))
        use_tdist = params.get('use_tdist', True)
        dims_egarch_count = int(params.get('dims_egarch_count', 4))
        
        # Hybrid model parameters
        sequence_length = int(params.get('sequence_length', 10))
        lstm_units = int(params.get('lstm_units', 64))
        dense_units = int(params.get('dense_units', 32))
        learning_rate = float(params.get('learning_rate', 0.001))
        
        # Set up EGARCH dimensions
        dims_egarch = list(range(dims_egarch_count))
        
        # Split data into train/test
        T = len(features)
        train_size = int(T * train_ratio)
        
        X_train = features[:train_size]
        X_test = features[train_size:]
        
        prices_train = prices[:train_size]
        prices_test = prices[train_size:]
        
        # Train HMM model
        pi, A, st_list, _ = train_hmm_once(
            X_train, K,
            n_starts=n_starts,
            max_iter=max_iter,
            use_tdist=use_tdist,
            dims_egarch=dims_egarch
        )
        
        # Prepare sequences for hybrid model
        # Calculate HMM states for training data
        from enhanced_hmm_em_v2 import forward_backward
        gamma, _, _ = forward_backward(
            X_train, pi, A, st_list,
            use_tdist=use_tdist,
            dims_egarch=dims_egarch
        )
        
        # Get the most probable state for each time point
        hmm_states = np.argmax(gamma, axis=1)
        
        # Convert to one-hot encoding
        hmm_states_onehot = np.zeros((len(hmm_states), K))
        for i, state in enumerate(hmm_states):
            hmm_states_onehot[i, state] = 1
        
        # Prepare sequential data for LSTM
        X_sequences = []
        hmm_sequence = []
        
        for i in range(sequence_length, train_size):
            X_sequences.append(X_train[i-sequence_length:i])
            hmm_sequence.append(hmm_states_onehot[i])
        
        X_sequences = np.array(X_sequences)
        hmm_sequence = np.array(hmm_sequence)
        
        # Generate labels for supervised training
        # Direction: 0 (up), 1 (down), 2 (sideways)
        future_returns = []
        for i in range(sequence_length, train_size):
            if i + 1 < train_size:
                future_return = (prices_train[i+1] - prices_train[i]) / prices_train[i]
                future_returns.append(future_return)
            else:
                future_returns.append(0)
        
        direction_labels = np.zeros((len(future_returns), 3))  # One-hot encoding
        
        for i, ret in enumerate(future_returns):
            if ret > 0.0005:  # Up
                direction_labels[i, 0] = 1
            elif ret < -0.0005:  # Down
                direction_labels[i, 1] = 1
            else:  # Sideways
                direction_labels[i, 2] = 1
        
        # Create and train hybrid model
        hybrid_model = HybridModel(
            input_dim=X_train.shape[1],
            hmm_states=K,
            lstm_units=lstm_units,
            dense_units=dense_units,
            sequence_length=sequence_length,
            learning_rate=learning_rate
        )
        
        hybrid_model.build_models()
        
        # Train direction model
        hybrid_model.train_direction_model(
            X_sequences, hmm_sequence, direction_labels,
            epochs=20, batch_size=32, validation_split=0.2
        )
        
        # Evaluate on test set
        test_score = hybrid_evaluator(
            hybrid_model, X_test, prices_test, pi, A, st_list,
            use_tdist=use_tdist, dims_egarch=dims_egarch,
            sequence_length=sequence_length
        )
        
        # Return negative score for minimization
        return -test_score
    
    return objective

###############################################################################
# Utility Functions
###############################################################################

def load_data(features_file, prices_file=None):
    """
    Load data for optimization.
    
    Args:
        features_file: Path to features file
        prices_file: Optional path to prices file
        
    Returns:
        Tuple of (features, prices)
    """
    # Load features
    if features_file.endswith('.csv'):
        features_df = pd.read_csv(features_file)
        
        # Extract feature columns
        feature_cols = [col for col in features_df.columns 
                       if col.startswith('log_return') or 
                          col.startswith('rsi_') or
                          col.startswith('atr_') or
                          col.startswith('macd_') or
                          col.startswith('session_') or
                          col.startswith('day_') or
                          col.startswith('log_volume')]
        
        features = features_df[feature_cols].values
    elif features_file.endswith('.npy'):
        features = np.load(features_file)
    else:
        raise ValueError(f"Unsupported file format: {features_file}")
    
    # Load prices if provided
    prices = None
    if prices_file:
        if prices_file.endswith('.csv'):
            prices_df = pd.read_csv(prices_file)
            prices = prices_df['close'].values
        elif prices_file.endswith('.npy'):
            prices = np.load(prices_file)
        else:
            raise ValueError(f"Unsupported file format: {prices_file}")
    
    return features, prices

def save_optimized_model(params, model_file):
    """
    Save optimized model parameters.
    
    Args:
        params: Optimized parameters
        model_file: Output file path
    """
    try:
        # Import required modules
        from enhanced_hmm_em_v2 import train_hmm_once
        
        # Load data
        features_file = params.get('features_file', 'features.csv')
        features, _ = load_data(features_file)
        
        # Extract parameters
        K = int(params.get('K', 4))
        n_starts = int(params.get('n_starts', 5))
        max_iter = int(params.get('max_iter', 20))
        use_tdist = params.get('use_tdist', True)
        use_skewed_t = params.get('use_skewed_t', False)
        use_hybrid_distribution = params.get('use_hybrid_distribution', False)
        dims_egarch_count = int(params.get('dims_egarch_count', 4))
        
        # Set up EGARCH dimensions
        dims_egarch = list(range(dims_egarch_count))
        
        # Configure distribution type
        if use_tdist:
            CONFIG = {
                'use_skewed_t': use_skewed_t,
                'use_hybrid_distribution': use_hybrid_distribution
            }
            
            # Temporarily modify global CONFIG
            from enhanced_hmm_em_v2 import CONFIG as GLOBAL_CONFIG
            original_config = GLOBAL_CONFIG.copy()
            GLOBAL_CONFIG.update(CONFIG)
        
        # Train final model with optimal parameters
        pi, A, st_list, ll = train_hmm_once(
            features, K,
            n_starts=n_starts,
            max_iter=max_iter,
            use_tdist=use_tdist,
            dims_egarch=dims_egarch
        )
        
        # Restore original CONFIG if modified
        if use_tdist:
            from enhanced_hmm_em_v2 import CONFIG as GLOBAL_CONFIG
            GLOBAL_CONFIG.update(original_config)
        
        # Create model data structure
        model_data = {
            "model": {
                "K": K,
                "pi": pi,
                "A": A,
                "st_params": st_list,
                "trainLL": ll
            },
            "feature_cols": params.get('feature_cols'),
            "dims_egarch": dims_egarch,
            "use_tdist": use_tdist,
            "use_skewed_t": use_skewed_t,
            "use_hybrid_distribution": use_hybrid_distribution,
            "optimization_params": params,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save model
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Optimized model saved to {model_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving optimized model: {str(e)}")
        return False

###############################################################################
# Main Function
###############################################################################

def optimize_hyperparameters(features_file, method='optuna', objective_type='likelihood',
                           n_trials=50, output_dir=DEFAULT_RESULTS_DIR, prices_file=None,
                           save_model=True, n_jobs=1):
    """
    Main function to run hyperparameter optimization.
    
    Args:
        features_file: Path to features file
        method: Optimization method ('bayesian', 'optuna', 'hyperopt', 'ray')
        objective_type: Type of objective function ('likelihood', 'trading', 'hybrid')
        n_trials: Number of trials for optimization
        output_dir: Output directory for results
        prices_file: Path to prices file (required for 'trading' and 'hybrid' objectives)
        save_model: Whether to save the optimized model
        n_jobs: Number of parallel jobs
        
    Returns:
        Best parameters
    """
    # Load data
    logger.info(f"Loading data from {features_file}")
    features, prices = load_data(features_file, prices_file)
    
    # Check if prices are available for trading objectives
    if objective_type in ['trading', 'hybrid'] and prices is None:
        raise ValueError(f"Prices file is required for '{objective_type}' objective")
    
    # Create objective function
    logger.info(f"Creating {objective_type} objective function")
    
    if objective_type == 'likelihood':
        objective_func = create_likelihood_objective(features)
    elif objective_type == 'trading':
        # Simple trade evaluator
        def trade_evaluator(signals, prices, forward_window=20):
            # Calculate performance based on trading signals
            return 0.0
        
        objective_func = create_trading_performance_objective(
            features, prices, trade_evaluator
        )
    elif objective_type == 'hybrid':
        # Simple hybrid evaluator
        def hybrid_evaluator(hybrid_model, X_test, prices_test, pi, A, st_list,
                           use_tdist=True, dims_egarch=None, sequence_length=10):
            # Evaluate hybrid model performance
            return 0.0
        
        objective_func = create_hybrid_objective(
            features, prices, hybrid_evaluator
        )
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
    
    # Run optimization
    logger.info(f"Starting {method} optimization with {n_trials} trials")
    
    if method == 'bayesian':
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        
        param_space = create_skopt_param_space()
        optimizer = BayesianOptimizer(
            objective_func, param_space,
            n_calls=n_trials,
            n_initial_points=min(10, n_trials // 3),
            random_state=42
        )
        
    elif method == 'optuna':
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for Optuna optimization")
        
        optimizer = OptunaOptimizer(
            objective_func, create_optuna_param_space,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
    elif method == 'hyperopt':
        if not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is required for Hyperopt optimization")
        
        param_space = create_hyperopt_param_space()
        optimizer = HyperoptOptimizer(
            objective_func, param_space,
            max_evals=n_trials,
            show_progressbar=True
        )
        
    elif method == 'ray':
        if not RAY_AVAILABLE:
            raise ImportError("ray is required for Ray Tune optimization")
        
        param_space = create_ray_param_space()
        optimizer = RayTuneOptimizer(
            objective_func, param_space,
            num_samples=n_trials,
            max_concurrent=n_jobs,
            search_alg='optuna',
            scheduler='asha'
        )
        
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Run optimization
    start_time = time.time()
    best_params, opt_result = optimizer.optimize()
    optimize_time = time.time() - start_time
    
    logger.info(f"Optimization completed in {optimize_time:.2f} seconds")
    logger.info(f"Best parameters: {best_params}")
    
    # Save model if requested
    if save_model:
        model_file = os.path.join(output_dir, f"optimized_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
        
        # Add data file paths to parameters
        best_params['features_file'] = features_file
        best_params['prices_file'] = prices_file
        
        save_optimized_model(best_params, model_file)
    
    return best_params

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for HMM trading system")
    parser.add_argument("--features", type=str, required=True, help="Path to features file")
    parser.add_argument("--prices", type=str, default=None, help="Path to prices file")
    parser.add_argument("--method", type=str, default="optuna", 
                      choices=["bayesian", "optuna", "hyperopt", "ray"],
                      help="Optimization method")
    parser.add_argument("--objective", type=str, default="likelihood",
                      choices=["likelihood", "trading", "hybrid"],
                      help="Objective function type")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--output", type=str, default=DEFAULT_RESULTS_DIR, help="Output directory")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--no-save-model", action="store_true", help="Do not save optimized model")
    
    args = parser.parse_args()
    
    # Run optimization
    best_params = optimize_hyperparameters(
        features_file=args.features,
        method=args.method,
        objective_type=args.objective,
        n_trials=args.trials,
        output_dir=args.output,
        prices_file=args.prices,
        save_model=not args.no_save_model,
        n_jobs=args.jobs
    )
    
    # Print best parameters
    print("\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")