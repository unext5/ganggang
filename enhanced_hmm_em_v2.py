# enhanced_hmm_em_v2.py
# Optimierte Version für höherdimensionale Feature-Vektoren
# Mit Unterstützung für zeitvariable Übergangswahrscheinlichkeiten
# Und regime-spezifische EGARCH-Parameter

import os
import numpy as np
import math
from scipy.optimize import minimize
import logging
from multiprocessing import Pool, cpu_count
import time
from functools import partial
import warnings
import concurrent.futures
from dynamic_feature_selection import DynamicFeatureSelector

# Importiere optimierte Volatilitätsmodelle
from enhanced_vol_models_v2 import (
    t_pdf_safe_diag, t_pdf_diag_multidim, 
    update_df_t, update_egarch_regime_specific,
    update_skewed_t_params, skewed_t_pdf_diag, hybrid_pdf_diag,
    egarch_recursion, compute_sigma_array
)

# Logger einrichten
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhanced_hmm_em_v2')

# Globale Konfigurationseinstellungen
# Verbesserte Konfiguration mit automatischer CPU-Erkennung
available_cpus = cpu_count()
CONFIG = {
    'use_hybrid_distribution': False,    # T-Verteilung + Normalverteilung
    'use_skewed_t': False,               # Schiefe T-Verteilung
    'use_time_varying_transition': True, # Zeitvariable Übergangsmatrix
    'regime_specific_egarch': True,      # Regime-spezifische EGARCH-Parameter
    'parallel_processing': True,         # Parallele Verarbeitung wo möglich
    'max_workers': min(available_cpus, 8),   # Dynamische Anzahl parallel Worker
    'memory_efficient': True,            # Speichereffizientere Berechnung für große Datensätze
    'gpu_acceleration': False,           # GPU-Beschleunigung falls verfügbar (muss manuell aktiviert werden)
    'debug_level': 'INFO'                # Logging-Level für Debugging
}

###############################################
# Forward-Backward
###############################################

def forward_backward(features, pi, A, state_params_list, use_tdist=True, dims_egarch=None, 
                     times=None, external_factors=None):
    """
    Forward-Backward-Algorithmus mit verbesserten Performance-Optimierungen.
    
    Parameter:
        features: Feature-Matrix [T, D]
        pi: Anfangswahrscheinlichkeiten [K]
        A: Übergangsmatrix [K, K]
        state_params_list: Liste mit State-Parametern
        use_tdist: T-Verteilung oder Normalverteilung verwenden
        dims_egarch: Liste der Dimensionen, die EGARCH verwenden
        times: Optional, Zeitstempel für zeitvariable Übergangswahrscheinlichkeiten
        external_factors: Optional, externe Faktoren für Übergangswahrscheinlichkeiten
    
    Returns:
        gamma_arr: Posteriori-Wahrscheinlichkeiten [T, K]
        xi: Zustands-Übergangswahrscheinlichkeiten [T-1, K, K]
        scaling: Skalierungsfaktoren [T]
    """
    T, D = features.shape
    K = len(state_params_list)
    
    # Emission-Wahrscheinlichkeiten
    B = np.zeros((T, K))
    
    # Berechne B-Matrix mit optimierten numerischen Verfahren
    for i in range(K):
        stp = state_params_list[i]
        mu_d = stp["mu"]
        df_t = stp.get("df_t", 10)
        vol_model = stp["vol_model"]
        params_vol = stp["params_vol"]
        
        # Optionale Parameter für erweiterte Verteilungen
        skew_params = stp.get("skew_params", None)
        mixture_weight = stp.get("mixture_weight", 0.5)
        
        # Sigma-Array mit selektiven EGARCH-Dimensionen
        sigma_array = compute_sigma_array(features, mu_d, vol_model, params_vol, dims_egarch)
        
        # Standardisierte Fehler
        e_array = (features - mu_d) / sigma_array
        
        # Berechne Emissionswahrscheinlichkeiten
        for t in range(T):
            x_ = e_array[t, :]
            
            # Verschiedene Verteilungen zur Auswahl
            if use_tdist:
                if CONFIG['use_skewed_t'] and skew_params is not None:
                    val_ = skewed_t_pdf_diag(x_, df_t, skew_params)
                elif CONFIG['use_hybrid_distribution']:
                    val_ = hybrid_pdf_diag(x_, df_t, mixture_weight)
                else:
                    val_ = t_pdf_safe_diag(x_, df_t)
            else:
                # Normal fallback => N(0,I)
                D_ = len(x_)
                dot_product = np.dot(x_, x_)
                val_ = (2*math.pi)**(-0.5*D_) * math.exp(-0.5*dot_product)
                val_ = max(val_, 1e-15)
            
            B[t, i] = max(val_, 1e-15)
    
    # Zeitvariable Übergangsmatrizen, falls aktiviert
    A_t = np.zeros((T, K, K))
    if CONFIG['use_time_varying_transition'] and times is not None:
        for t in range(T):
            A_t[t] = _compute_time_varying_transition(A, t, times[t], features[t], external_factors)
    else:
        # Konstante Übergangsmatrix für alle Zeitpunkte
        for t in range(T):
            A_t[t] = A
    
    # Forward-Algorithmus
    alpha_arr = np.zeros((T, K))
    alpha_arr[0, :] = pi * B[0, :]
    scaling = np.zeros(T)
    scaling[0] = np.sum(alpha_arr[0, :])
    
    if scaling[0] < 1e-300:
        scaling[0] = 1e-300
    
    alpha_arr[0, :] /= scaling[0]
    
    for t in range(1, T):
        # Vektorisierte Multiplikation für alle States
        for j in range(K):
            alpha_arr[t, j] = np.sum(alpha_arr[t-1, :] * A_t[t-1, :, j]) * B[t, j]
        
        scaling[t] = np.sum(alpha_arr[t, :])
        if scaling[t] < 1e-300:
            scaling[t] = 1e-300
        
        alpha_arr[t, :] /= scaling[t]
    
    # Backward-Algorithmus
    beta_arr = np.zeros((T, K))
    beta_arr[-1, :] = 1
    
    for t in range(T-2, -1, -1):
        for i in range(K):
            beta_arr[t, i] = np.sum(A_t[t, i, :] * B[t+1, :] * beta_arr[t+1, :])
        
        beta_arr[t, :] /= scaling[t+1]
    
    # Posteriori-Wahrscheinlichkeiten
    gamma_arr = alpha_arr * beta_arr
    
    # State-Übergangswahrscheinlichkeiten
    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        denom = np.sum(alpha_arr[t, :] * np.dot(A_t[t], B[t+1, :] * beta_arr[t+1, :]))
        if denom < 1e-300:
            denom = 1e-300
        
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = alpha_arr[t, i] * A_t[t, i, j] * B[t+1, j] * beta_arr[t+1, j] / denom
    
    return gamma_arr, xi, scaling

def weighted_forward_backward(features, feature_weights, pi, A, state_params_list, use_tdist=True, dims_egarch=None, 
                          times=None, external_factors=None):
    """
    Forward-Backward algorithm with feature importance weighting.
    
    Parameters:
        features: Feature matrix [T, D]
        feature_weights: Importance weights for each feature [D]
        pi: Initial probabilities [K]
        A: Transition matrix [K, K]
        state_params_list: List of state parameters
        use_tdist: Whether to use t-distribution or normal distribution
        dims_egarch: List of dimensions using EGARCH
        times: Optional timestamps for time-varying transitions
        external_factors: Optional external factors for transitions
    
    Returns:
        gamma_arr: Posterior probabilities [T, K]
        xi: State transition probabilities [T-1, K, K]
        scaling: Scaling factors [T]
    """
    T, D = features.shape
    K = len(state_params_list)
    
    # Apply feature weights to standardize features
    weighted_features = features.copy()
    for d in range(D):
        weighted_features[:, d] *= np.sqrt(feature_weights[d])
    
    # Calculate emission probabilities with weighted features
    B = np.zeros((T, K))
    
    for i in range(K):
        stp = state_params_list[i]
        mu_d = stp["mu"]
        df_t = stp.get("df_t", 10)
        vol_model = stp["vol_model"]
        params_vol = stp["params_vol"]
        
        # Optional parameters for extended distributions
        skew_params = stp.get("skew_params", None)
        mixture_weight = stp.get("mixture_weight", 0.5)
        
        # Compute sigma array with selective EGARCH dimensions
        sigma_array = compute_sigma_array(weighted_features, mu_d, vol_model, params_vol, dims_egarch)
        
        # Standardized errors
        e_array = (weighted_features - mu_d) / sigma_array
        
        # Calculate emission probabilities
        for t in range(T):
            x_ = e_array[t, :]
            
            # Different distribution options
            if use_tdist:
                if CONFIG['use_skewed_t'] and skew_params is not None:
                    val_ = skewed_t_pdf_diag(x_, df_t, skew_params)
                elif CONFIG['use_hybrid_distribution']:
                    val_ = hybrid_pdf_diag(x_, df_t, mixture_weight)
                else:
                    val_ = t_pdf_safe_diag(x_, df_t)
            else:
                # Normal distribution fallback
                D_ = len(x_)
                dot_product = np.dot(x_, x_)
                val_ = (2*math.pi)**(-0.5*D_) * math.exp(-0.5*dot_product)
                val_ = max(val_, 1e-15)
            
            B[t, i] = max(val_, 1e-15)
    
    # Time-varying transition matrices if enabled
    A_t = np.zeros((T, K, K))
    if CONFIG['use_time_varying_transition'] and times is not None:
        for t in range(T):
            A_t[t] = _compute_time_varying_transition(A, t, times[t], weighted_features[t], external_factors)
    else:
        # Constant transition matrix for all time points
        for t in range(T):
            A_t[t] = A
    
    # Forward algorithm
    alpha_arr = np.zeros((T, K))
    alpha_arr[0, :] = pi * B[0, :]
    scaling = np.zeros(T)
    scaling[0] = np.sum(alpha_arr[0, :])
    
    if scaling[0] < 1e-300:
        scaling[0] = 1e-300
    
    alpha_arr[0, :] /= scaling[0]
    
    for t in range(1, T):
        # Vectorized multiplication for all states
        for j in range(K):
            alpha_arr[t, j] = np.sum(alpha_arr[t-1, :] * A_t[t-1, :, j]) * B[t, j]
        
        scaling[t] = np.sum(alpha_arr[t, :])
        if scaling[t] < 1e-300:
            scaling[t] = 1e-300
        
        alpha_arr[t, :] /= scaling[t]
    
    # Backward algorithm
    beta_arr = np.zeros((T, K))
    beta_arr[-1, :] = 1
    
    for t in range(T-2, -1, -1):
        for i in range(K):
            beta_arr[t, i] = np.sum(A_t[t, i, :] * B[t+1, :] * beta_arr[t+1, :])
        
        beta_arr[t, :] /= scaling[t+1]
    
    # Posterior probabilities
    gamma_arr = alpha_arr * beta_arr
    
    # State transition probabilities
    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        denom = np.sum(alpha_arr[t, :] * np.dot(A_t[t], B[t+1, :] * beta_arr[t+1, :]))
        if denom < 1e-300:
            denom = 1e-300
        
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = alpha_arr[t, i] * A_t[t, i, j] * B[t+1, j] * beta_arr[t+1, j] / denom
    
    return gamma_arr, xi, scaling

def _compute_time_varying_transition(A_base, t, time_info=None, features=None, external_factors=None):
    """
    Berechnet zeitvariable Übergangswahrscheinlichkeiten.
    
    Args:
        A_base: Basis-Übergangsmatrix [K, K]
        t: Aktueller Zeitschritt
        time_info: Zeitstempel (datetime-Objekt oder numpy.datetime64)
        features: Aktueller Feature-Vektor
        external_factors: Zusätzliche externe Faktoren
    
    Returns:
        A_t: Übergangsmatrix für Zeitpunkt t
    """
    A_t = A_base.copy()
    
    # 1. Zeitbasierte Anpassungen
    if time_info is not None:
        # Convert numpy.datetime64 to Python datetime if needed
        if hasattr(time_info, 'dtype') and np.issubdtype(time_info.dtype, np.datetime64):
            import pandas as pd
            time_info = pd.Timestamp(time_info).to_pydatetime()
            
        hour = time_info.hour
        day_of_week = time_info.weekday()
        
        # Volatilitätsfaktor
        volatility_factor = 1.0
        
        # Erhöhte Übergangswahrscheinlichkeiten während Volatilitätsphasen
        # Sitzungsübergänge
        if (hour == 7 or hour == 8) or (hour == 13 or hour == 14):  # London/Tokio oder London/NY Überlappung
            volatility_factor = 1.5
        
        # Wochentagseffekte
        if day_of_week == 0:  # Montag - höhere Übergangswahrscheinlichkeit
            volatility_factor *= 1.2
        elif day_of_week == 4:  # Freitag - Tendenz zu stabilen Zuständen
            volatility_factor *= 0.8
        
        # 2. Feature-basierte Anpassungen
        if features is not None:
            # Verwende Volatilitäts-Features (ATR, falls indiziert bei 6-7)
            avg_volatility = 0
            vol_indices = [6, 7]  # Annahme: ATR-Indizes
            
            for vol_idx in vol_indices:
                if vol_idx < len(features):
                    avg_volatility += abs(features[vol_idx])
            
            if len(vol_indices) > 0:
                avg_volatility /= len(vol_indices)
                
                # Normalisiere Volatilität auf einen sinnvollen Bereich
                norm_vol = min(max(avg_volatility, 0.5), 2.0)
                volatility_factor *= norm_vol
        
        # 3. Externe Faktoren berücksichtigen
        if external_factors is not None:
            if 'volatility_factor' in external_factors:
                volatility_factor *= external_factors['volatility_factor']
        
        # Anpassung der Off-Diagonal-Elemente der Matrix
        for i in range(len(A_t)):
            for j in range(len(A_t)):
                if i != j:  # Nur Übergangswahrscheinlichkeiten anpassen
                    A_t[i, j] *= volatility_factor
    
    # Renormalisieren der Zeilen
    for i in range(len(A_t)):
        row_sum = A_t[i].sum()
        if row_sum > 0:
            A_t[i] /= row_sum
    
    return A_t


###############################################
# M-Step
###############################################

def m_step(features, gamma_arr, xi, state_params_list, A, pi, cycles=2, use_tdist=True, 
          dims_egarch=None, times=None):
    """
    M-Step mit erweiterter Unterstützung für regimespezifische Parameter und zeitvariable Übergänge.
    
    Parameter:
        features: Feature-Matrix [T, D]
        gamma_arr: Posteriori-Wahrscheinlichkeiten [T, K]
        xi: Übergangswahrscheinlichkeiten [T-1, K, K]
        state_params_list: Liste der Zustandsparameter
        A: Übergangsmatrix [K, K]
        pi: Anfangswahrscheinlichkeiten [K]
        cycles: Anzahl der Optimierungszyklen
        use_tdist: T-Verteilung verwenden
        dims_egarch: Dimensionen für EGARCH
        times: Zeitpunkte für zeitvariable Übergänge
    
    Returns:
        pi_new: Neue Anfangswahrscheinlichkeiten
        A_new: Neue Übergangsmatrix
        st_list_new: Neue Zustandsparameter
    """
    T, D = features.shape
    K = len(state_params_list)
    
    # Aktualisiere Anfangswahrscheinlichkeiten (pi)
    pi_new = gamma_arr[0, :] / np.sum(gamma_arr[0, :])
    
    # Aktualisiere Übergangsmatrix (A)
    if CONFIG['use_time_varying_transition'] and times is not None:
        # Bei zeitvariablen Übergängen: Durchschnittliche Übergangsmatrix schätzen
        A_new = np.zeros((K, K))
        for i in range(K):
            denom = np.sum(gamma_arr[:-1, i])
            if denom > 0:  # Sicherheitsüberprüfung
                # Berechne gewichtete Summe aller Übergänge
                for j in range(K):
                    A_new[i, j] = np.sum(xi[:, i, j]) / denom
    else:
        # Standard-Update für Übergangsmatrix
        A_new = np.zeros((K, K))
        for i in range(K):
            denom = np.sum(gamma_arr[:-1, i])
            if denom > 0:  # Sicherheitsüberprüfung
                for j in range(K):
                    A_new[i, j] = np.sum(xi[:, i, j]) / denom
    
    # Kopiere Zustandsparameter
    new_list = []
    for i in range(K):
        new_list.append(state_params_list[i].copy())
    
    # Mehrere Optimierungszyklen
    for cyc in range(cycles):
        # Parallele Verarbeitung für State-Updates, falls aktiviert
        if CONFIG['parallel_processing'] and K > 1:
            # Bestimme optimale Anzahl von Workern basierend auf Systemlast
            workers = min(K, CONFIG['max_workers'])
            
            # Wähle zwischen ThreadPoolExecutor (weniger Overhead, teilt Memory) 
            # und ProcessPoolExecutor (separate Prozesse, besser für CPU-intensive Tasks)
            use_process_pool = K >= 4 and T > 1000  # Für größere Probleme verwende ProcessPool
            
            if use_process_pool:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    # Erstelle Future-Tasks für jeden State
                    futures = []
                    for i in range(K):
                        future = executor.submit(
                            _update_state_params, i, features, gamma_arr, new_list[i], 
                            use_tdist, dims_egarch, times
                        )
                        futures.append(future)
                    
                    # Sammle Ergebnisse mit Fortschrittsanzeige
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            new_list[i] = future.result()
                            if cyc == 0 and i % max(1, K//5) == 0:
                                logger.debug(f"State {i+1}/{K} aktualisiert (Prozesspool)")
                        except Exception as e:
                            logger.error(f"Error updating state {i}: {str(e)}")
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    # Erstelle Future-Tasks für jeden State
                    futures = []
                    for i in range(K):
                        future = executor.submit(
                            _update_state_params, i, features, gamma_arr, new_list[i], 
                            use_tdist, dims_egarch, times
                        )
                        futures.append(future)
                    
                    # Sammle Ergebnisse
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            result = future.result()
                            # Finde den richtigen State-Index für dieses Ergebnis
                            for j, f in enumerate(futures):
                                if f == future:
                                    new_list[j] = result
                                    break
                        except Exception as e:
                            logger.error(f"Error updating state (threadpool): {str(e)}")
        else:
            # Sequentielle Verarbeitung
            for i in range(K):
                new_list[i] = _update_state_params(
                    i, features, gamma_arr, new_list[i], 
                    use_tdist, dims_egarch, times
                )
    
    return pi_new, A_new, new_list

def _update_state_params(i, features, gamma_arr, old_st, use_tdist, dims_egarch, times):
    """
    Aktualisiert Parameter für einen einzelnen Zustand.
    """
    gamma_i = gamma_arr[:, i]
    w_sum = np.sum(gamma_i)
    
    if w_sum <= 0:
        return old_st  # Keine Änderung, wenn keine Gewichte
    
    # Aktualisiere Mittelwerte
    mu_new = np.zeros(features.shape[1])
    for d_ in range(features.shape[1]):
        mu_new[d_] = np.sum(gamma_i * features[:, d_]) / w_sum
    
    old_st["mu"] = mu_new
    
    # Bestimme Regime-Typ für EGARCH
    state_label = None
    if CONFIG['regime_specific_egarch']:
        # Einfache Bestimmung des Regime-Typs basierend auf den Return-Features
        return_indices = [0, 1, 2, 3]  # log_return30, log_return5, log_return1h, log_return4h
        
        # Gewichtete Summe der Returns
        returns_sum = 0
        for idx in return_indices:
            if idx < len(mu_new):
                returns_sum += mu_new[idx]
        
        # Bestimme Regime-Typ
        if returns_sum > 0.001:
            state_label = "High Bullish"
        elif returns_sum > 0:
            state_label = "Low Bullish"
        elif returns_sum > -0.001:
            state_label = "Low Bearish"
        else:
            state_label = "High Bearish"
    
    # Aktualisiere Volatilitätsmodell-Parameter
    vol_model = old_st["vol_model"]
    if vol_model == "EGARCH":
        (om, al, ga, be) = old_st["params_vol"]
        
        if CONFIG['regime_specific_egarch'] and state_label:
            # Bestimme Regime-Typ für EGARCH-Parameter
            regime_type = "neutral"
            if "High" in state_label and "Bullish" in state_label:
                regime_type = "high_bull"
            elif "High" in state_label and "Bearish" in state_label:
                regime_type = "high_bear"
            elif "Low" in state_label:
                regime_type = "low_vol"
            
            # Regime-spezifische EGARCH-Parameter
            new_params = []
            for d_idx in dims_egarch:
                if d_idx < features.shape[1]:
                    r_d = features[:, d_idx]
                    result = update_egarch_regime_specific(
                        r_d, mu_new[d_idx], gamma_i, (om, al, ga, be), regime_type
                    )
                    new_params.append(result)
            
            # Mittlung der Parameter
            if new_params:
                avg_params = np.mean(np.array(new_params), axis=0)
                old_st["params_vol"] = tuple(avg_params)
        else:
            # Standard EGARCH-Update für die angegebenen Dimensionen
            def obj_eg(p):
                (omega, alpha, ga, beta) = p
                val = 0.0
                
                for d_idx in dims_egarch:
                    if d_idx < features.shape[1]:
                        r_d = features[:, d_idx]
                        from enhanced_vol_models_v2 import egarch_neglog
                        val += egarch_neglog((omega, alpha, ga, beta), r_d, mu_new[d_idx], gamma_i)
                
                return val
            
            bnds = [(-10, None), (0, None), (None, None), (0, 0.999)]
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = minimize(obj_eg, (om, al, ga, be), bounds=bnds, method='L-BFGS-B')
                
                if res.success:
                    old_st["params_vol"] = tuple(res.x)
            except Exception as e:
                logger.warning(f"Exception in EGARCH optimization: {str(e)}")
    
    # Aktualisiere T-Verteilungs-Parameter
    if use_tdist:
        # Berechne standardisierte Residuen
        from enhanced_vol_models_v2 import compute_sigma_array
        sigma_array = compute_sigma_array(
            features, mu_new, vol_model, old_st["params_vol"], dims_egarch
        )
        e_array = (features - mu_new) / sigma_array
        
        if CONFIG['use_skewed_t'] and "skew_params" in old_st:
            # Update für schiefe T-Verteilung
            old_params = [old_st.get("df_t", 10)]
            if "skew_params" in old_st:
                old_params.extend(old_st["skew_params"])
            else:
                old_params.extend([1.0] * features.shape[1])
            
            try:
                new_params = update_skewed_t_params(e_array, gamma_i, old_params)
                old_st["df_t"] = new_params[0]
                old_st["skew_params"] = new_params[1:]
            except Exception as e:
                logger.warning(f"Exception in skewed T optimization: {str(e)}")
        else:
            # Standard-Update für T-Verteilung
            df_old = old_st.get("df_t", 10)
            try:
                df_new = update_df_t(e_array, gamma_i, df_old)
                old_st["df_t"] = df_new
            except Exception as e:
                logger.warning(f"Exception in T-dist df optimization: {str(e)}")
    
    return old_st

###############################################
# Train-Funktionen
###############################################

def train_hmm_once(features, K, n_starts=5, max_iter=20, use_tdist=True, debug=False, 
                  dims_egarch=None, times=None, external_factors=None, early_stopping=True,
                  feature_weights=None):
    """
    Training of HMM with various optimizations and extended functions.
    
    Parameters:
        features: Feature matrix [T, D]
        K: Number of states
        n_starts: Number of initialization attempts
        max_iter: Maximum number of EM iterations
        use_tdist: Use T-distribution
        debug: Enable debug output
        dims_egarch: Dimensions for EGARCH
        times: Time points for time-varying transitions
        external_factors: External factors for transition probabilities
        early_stopping: Enable early stopping at convergence
        feature_weights: Optional weights for features (for weighted_forward_backward)
    
    Returns:
        pi_best: Optimized initial probabilities
        A_best: Optimized transition matrix
        st_list_best: Optimized state parameters
        ll_best: Log-likelihood of best model
    """
    start_time = time.time()
    
    # Standardization for time-varying transitions
    if times is not None and len(times) != len(features):
        logger.warning(f"Number of time points ({len(times)}) does not match number of features ({len(features)}).")
        times = None
    
    # Use multiple initializations for more robust results
    if CONFIG['parallel_processing'] and n_starts > 1:
        # Parallel initializations
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_starts, CONFIG['max_workers'])) as executor:
            futures = [executor.submit(_init_single_model, features, K, use_tdist, dims_egarch) 
                     for _ in range(n_starts)]
            
            # Collect results
            init_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    init_results.append(future.result())
                except Exception as e:
                    logger.error(f"Initialization error: {str(e)}")
            
            # Find best initialization
            best_idx = np.argmax([ll for _, _, _, ll in init_results])
            pi_, A_, st_list_ = init_results[best_idx][:3]
    else:
        # Sequential initialization
        (pi_, A_, st_list_) = _init_states(features, K, n_starts, use_tdist, dims_egarch)
    
    if debug:
        logger.info(f"Initialization completed in {time.time() - start_time:.2f} seconds")
    
    # EM algorithm
    old_ll = -1e15
    best_ll = -1e15
    
    # Store best parameters
    pi_best = pi_.copy()
    A_best = A_.copy()
    st_list_best = [st.copy() for st in st_list_]
    
    # Early stopping criteria
    patience = 3  # Number of iterations without significant improvement
    min_improvement = 1e-2  # Minimum improvement in log-likelihood
    no_improvement_count = 0
    
    for it in range(max_iter):
        iter_start = time.time()
        
        # E-Step
        if feature_weights is not None:
            gamma, xi, scale = weighted_forward_backward(
                features, feature_weights, pi_, A_, st_list_, use_tdist, dims_egarch, 
                times, external_factors
            )
        else:
            gamma, xi, scale = forward_backward(
                features, pi_, A_, st_list_, use_tdist, dims_egarch, 
                times, external_factors
            )
            
        ll_ = np.sum(np.log(scale))
        
        if debug:
            logger.info(f"Iter={it+1}, LL={ll_:.3f}, Time: {time.time() - iter_start:.2f}s")
        
        # Save best model
        if ll_ > best_ll:
            best_ll = ll_
            pi_best = pi_.copy()
            A_best = A_.copy()
            st_list_best = [st.copy() for st in st_list_]
            no_improvement_count = 0
        else:
            # Check for early stopping
            improvement = ll_ - old_ll
            if early_stopping and improvement < min_improvement:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    if debug:
                        logger.info(f"Early stopping at iteration {it+1} (No significant improvement for {patience} iterations)")
                    break
        
        # M-Step
        pi_, A_, st_list_ = m_step(
            features, gamma, xi, st_list_, A_, pi_, 
            cycles=2, use_tdist=use_tdist, dims_egarch=dims_egarch, times=times
        )
        
        # Convergence check
        if abs(ll_ - old_ll) < 1e-3:
            if debug:
                logger.info(f"Converged at iteration {it+1} (DeltaLL < 1e-3)")
            break
        
        old_ll = ll_
    
    if debug:
        logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")
    
    # Return best model
    return (pi_best, A_best, st_list_best, best_ll)

def _init_single_model(features, K, use_tdist, dims_egarch=None):
    """
    Initialisiert ein einzelnes Modell für parallele Verarbeitung.
    """
    # Zufällige Initialisierung
    pi_ = np.ones(K) / K
    
    # Nahezu gleichverteilte Übergangsmatrix mit leichter Diagonaldominanz
    A_ = np.ones((K, K)) / K
    for i in range(K):
        A_[i, i] += 0.1  # Erhöhe Diagonalwerte leicht
        A_[i, :] /= np.sum(A_[i, :])  # Normalisieren
    
    # State-Parameter initialisieren
    st_list_ = []
    T, D = features.shape
    # Globale Mittelwerte für die Initialisierung
    mu_global = np.mean(features, axis=0)
    
    for i in range(K):
        # Zufällige EGARCH-Parameter innerhalb sinnvoller Grenzen
        om = np.random.rand() * 0.05 - 0.025
        al = abs(np.random.rand() * 0.05)
        ga = (np.random.rand() - 0.5) * 0.05
        be = 0.5 + np.random.rand() * 0.4  # Zwischen 0.5 und 0.9
        
        # T-Verteilungs-Parameter
        df_ = 5 + int(np.random.rand() * 10)
        
        # Verzerrung für schiefe T-Verteilung
        skew_params = None
        if CONFIG['use_skewed_t']:
            skew_params = [0.8 + np.random.rand() * 0.4 for _ in range(D)]
        
        # Mischungsgewicht für Hybrid-Verteilung
        mixture_weight = 0.5
        if CONFIG['use_hybrid_distribution']:
            mixture_weight = 0.3 + np.random.rand() * 0.4
        
        # Erstelle Grundzustand
        stp = {
            "mu": mu_global.copy(),
            "vol_model": "EGARCH",
            "params_vol": (om, al, ga, be),
            "df_t": df_
        }
        
        # Füge optionale Parameter hinzu
        if skew_params:
            stp["skew_params"] = skew_params
        
        if CONFIG['use_hybrid_distribution']:
            stp["mixture_weight"] = mixture_weight
        
        st_list_.append(stp)
    
    # Evaluiere Log-Likelihood
    gamma, xi, scale = forward_backward(features, pi_, A_, st_list_, use_tdist, dims_egarch)
    ll_ = np.sum(np.log(scale))
    
    return pi_, A_, st_list_, ll_

def _init_states(features, K, n_starts, use_tdist, dims_egarch=None):
    """
    Verbesserte Zustandsinitialisierung mit Unterstützung für selektive EGARCH.
    """
    best_ll = -1e15
    best_pi = None
    best_A = None
    best_st = None
    T, D = features.shape
    
    # Berechne globale Mittelwerte für die Initialisierung
    mu_global = np.mean(features, axis=0)
    
    # Multiple Startpunkte für Robustheit
    for s_ in range(n_starts):
        # Gleichverteilte Anfangswahrscheinlichkeiten
        pi_ = np.ones(K) / K
        
        # Nahezu gleichverteilte Übergangsmatrix mit leichter Diagonaldominanz
        A_ = np.ones((K, K)) / K
        for i in range(K):
            A_[i, i] += 0.1  # Erhöhe Diagonalwerte leicht
            A_[i, :] /= np.sum(A_[i, :])  # Normalisieren
        
        # State-Parameter initialisieren
        st_list_ = []
        for i in range(K):
            # Zufällige EGARCH-Parameter innerhalb sinnvoller Grenzen
            om = np.random.rand() * 0.05 - 0.025
            al = abs(np.random.rand() * 0.05)
            ga = (np.random.rand() - 0.5) * 0.05
            be = 0.5 + np.random.rand() * 0.4  # Zwischen 0.5 und 0.9
            
            # T-Verteilungs-Parameter
            df_ = 5 + int(np.random.rand() * 10)
            
            # Verzerrung für schiefe T-Verteilung
            skew_params = None
            if CONFIG['use_skewed_t']:
                skew_params = [0.8 + np.random.rand() * 0.4 for _ in range(D)]
            
            # Mischungsgewicht für Hybrid-Verteilung
            mixture_weight = 0.5
            if CONFIG['use_hybrid_distribution']:
                mixture_weight = 0.3 + np.random.rand() * 0.4
            
            # Erstelle Grundzustand
            stp = {
                "mu": mu_global.copy(),
                "vol_model": "EGARCH",
                "params_vol": (om, al, ga, be),
                "df_t": df_
            }
            
            # Füge optionale Parameter hinzu
            if skew_params:
                stp["skew_params"] = skew_params
            
            if CONFIG['use_hybrid_distribution']:
                stp["mixture_weight"] = mixture_weight
            
            st_list_.append(stp)
        
        # Berechne Log-Likelihood für diesen Startpunkt
        gamma, xi, scale = forward_backward(features, pi_, A_, st_list_, use_tdist, dims_egarch)
        ll_ = np.sum(np.log(scale))
        
        # Behalte den besten Startpunkt
        if ll_ > best_ll:
            best_ll = ll_
            best_pi = pi_
            best_A = A_
            best_st = st_list_
    
    return (best_pi, best_A, best_st)

###############################################
# Posterior-Confidence und Zustandsanalyse
###############################################

def compute_confidence(gamma_arr):
    """
    Berechnet die Konfidenz der State-Zuordnungen.
    
    Args:
        gamma_arr: Posteriori-Wahrscheinlichkeiten [T, K]
    
    Returns:
        confidence: Konfidenzwerte [T]
    """
    max_probs = np.max(gamma_arr, axis=1)
    return max_probs

def compute_state_stability(gamma_arr, window_size=10):
    """
    Berechnet die Stabilität der Zustandszuordnungen über die Zeit.
    
    Args:
        gamma_arr: Posteriori-Wahrscheinlichkeiten [T, K]
        window_size: Größe des Fensters für die gleitende Betrachtung
    
    Returns:
        stability: Stabilitätswerte [T]
    """
    T, K = gamma_arr.shape
    stability = np.zeros(T)
    
    # Bestimme zugeordnete Zustände
    states = np.argmax(gamma_arr, axis=1)
    
    # Berechne Stabilität basierend auf der Konsistenz der Zustandszuordnung
    for t in range(T):
        # Definiere das Fenster
        start = max(0, t - window_size)
        end = min(T, t + 1)
        
        # Aktueller Zustand
        current_state = states[t]
        
        # Zähle, wie oft der aktuelle Zustand im Fenster vorkommt
        state_count = np.sum(states[start:end] == current_state)
        
        # Stabilität als Anteil der Zeit im aktuellen Zustand
        stability[t] = state_count / (end - start)
    
    return stability

def validate_current_state(gamma_arr, features, current_state_idx, lookback=10):
    """
    Validiert den aktuellen Zustand durch Analyse der letzten N Features.
    
    Args:
        gamma_arr: Posteriori-Wahrscheinlichkeiten [T, K]
        features: Feature-Matrix [T, D]
        current_state_idx: Index des aktuellen Zustands
        lookback: Anzahl der zurückliegenden Zeitpunkte für die Analyse
    
    Returns:
        validity_score: Validitätswert des aktuellen Zustands
        needs_retraining: Flag, ob Neutraining empfohlen wird
    """
    # Ensure features and gamma_arr have compatible shapes for validation
    T_gamma, K = gamma_arr.shape
    T_feat, D = features.shape

    # Check if dimensions are plausible before proceeding
    if not (isinstance(features, np.ndarray) and features.ndim == 2 and
            isinstance(gamma_arr, np.ndarray) and gamma_arr.ndim == 2 and
            T_feat >= T_gamma): # Feature length must be at least gamma length
        logger.error(f"Invalid input dimensions in validate_current_state! Features: {features.shape}, Gamma: {gamma_arr.shape}. Skipping validation.")
        return {'valid': False, 'reason': 'Invalid input dimensions'}
        
    # Use only the relevant slice of features corresponding to gamma_arr's length
    relevant_features = features[-T_gamma:]

    # Further checks remain the same, using relevant_features if needed
    # (Example: if features were used directly later, use relevant_features instead)
    # ... (rest of the checks like current_state_prob, trend analysis etc.) ...

    # Ensure current_state_idx is valid
    if not (0 <= current_state_idx < K):
        logger.error(f"Invalid current_state_idx: {current_state_idx}. Skipping validation.")
        return {'valid': False, 'reason': 'Invalid state index'}

    # 1. Probability of the current state
    current_state_prob = gamma_arr[-1, current_state_idx]  # Use the last entry of gamma

    # 2. Trend analysis of the state probability
    state_prob_trend = []
    for t in range(max(0, T_gamma - lookback), T_gamma): # Iterate over gamma_arr's time steps
        state_prob_trend.append(gamma_arr[t, current_state_idx])
    
    # Negative Steigung ist ein Warnsignal
    slope = 0
    if len(state_prob_trend) > 1:
        # Einfache lineare Regression
        x = np.arange(len(state_prob_trend))
        slope = np.polyfit(x, state_prob_trend, 1)[0]
    
    # 3. Feature-Änderungsanalyse
    feature_changes = []
    for i in range(1, min(lookback, T_gamma)):
        if T_gamma-i-1 >= 0 and T_gamma-i >= 0:
            prev = relevant_features[T_gamma-i-1]
            curr = relevant_features[T_gamma-i]
            # Prozentuale Änderung der wichtigsten Features (hier: first 4 = returns)
            changes = [(curr[j] - prev[j]) / (abs(prev[j]) + 1e-6) for j in range(min(4, D))]
            feature_changes.append(changes)
    
    # Mittlere Änderung über alle Features
    mean_change = 0
    if feature_changes:
        # Flattening und Mittelwertbildung
        all_changes = [change for sublist in feature_changes for change in sublist]
        mean_change = np.mean(np.abs(all_changes))
    
    # 4. Validitätsscore berechnen (0-1 Skala)
    # Höhere Werte = höhere Gültigkeit
    validity_score = current_state_prob
    
    # Reduziere Score bei stark negativer Steigung
    if slope < -0.05:
        validity_score *= (1 + 2*slope)  # Slope is negative, so this reduces score
    
    # Reduziere Score bei hohen Feature-Änderungen
    if mean_change > 0.05:  # Starke Änderung
        validity_score *= max(0.5, 1 - 2*mean_change)
    
    # Begrenze den Score auf sinnvollen Bereich
    validity_score = max(0.01, min(1.0, validity_score))
    
    # 5. Empfehlung für Retraining
    needs_retraining = validity_score < 0.3
    
    # 3. Check consistency with feature volatility (Example using relevant_features)
    # Assuming volatility is one of the feature dimensions (e.g., index 6, 7)
    vol_indices = [idx for idx in [6, 7] if idx < D] # Check indices are valid
    if vol_indices:
        recent_volatility = np.mean(np.abs(relevant_features[-min(lookback, T_gamma):, vol_indices]))
        # Compare recent_volatility with expected volatility for current_state_idx
        # This logic needs to be defined based on model specifics
        # Example placeholder:
        # expected_vol = state_params_list[current_state_idx].get("expected_volatility_level") 
        # if expected_vol and not check_volatility_consistency(recent_volatility, expected_vol):
        #     return {'valid': False, 'reason': 'Volatility inconsistent'}

    # 4. Stability check (Optional, based on state changes)
    # states = np.argmax(gamma_arr, axis=1) # Use gamma_arr directly
    # recent_states = states[-min(lookback, T_gamma):]
    # if len(set(recent_states)) > stability_threshold:
    #    return {'valid': False, 'reason': 'State unstable'}

    # Combine checks (Example logic)
    # ... existing logic combining checks ...
    
    # Example: Basic validation based on probability threshold
    prob_threshold = 0.6 # Example threshold
    if current_state_prob < prob_threshold:
         return {'valid': False, 'reason': f'Low probability ({current_state_prob:.2f})'}

    # If all checks pass
    logger.debug(f"State {current_state_idx} validated successfully. Probability: {current_state_prob:.2f}")
    return {'valid': True, 'probability': current_state_prob}

###############################################
# Ensemble und Multi-Model Funktionen
###############################################

class HMMEnsemble:
    """
    Ensemble von HMM-Modellen für robustere Vorhersagen.
    """
    def __init__(self, models, weights=None):
        """
        Initialisiert das Ensemble.
        
        Args:
            models: Liste von HMM-Modellen (pi, A, st_list, dims_egarch)
            weights: Optionale Gewichtungen für die Modelle
        """
        self.models = models
        self.n_models = len(models)
        
        if weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            self.weights = np.array(weights) / sum(weights)
    
    def predict(self, features, times=None):
        """
        Vorhersage des Ensembles durch gewichtetes Voting.
        
        Args:
            features: Feature-Matrix [T, D]
            times: Optionale Zeitpunkte für zeitvariable Übergänge
        
        Returns:
            state: Vorhergesagter Zustand
            label: Zustandslabel
            confidence: Konfidenz der Vorhersage
        """
        state_votes = {}
        
        for i, (pi, A, st_list, dims_egarch) in enumerate(self.models):
            # Forward-Berechnung für jedes Modell
            gamma, _, _ = forward_backward(
                features, pi, A, st_list, use_tdist=True, 
                dims_egarch=dims_egarch, times=times
            )
            
            # Bestimme wahrscheinlichsten Zustand
            state = np.argmax(gamma[-1, :])
            prob = gamma[-1, state]
            
            # Interpretiere Zustand (vereinfachte Version)
            label = self._interpret_state(st_list[state])
            
            # Gewichtetes Voting
            vote_weight = self.weights[i] * prob
            
            if state not in state_votes:
                state_votes[state] = {
                    "total_weight": vote_weight,
                    "labels": [label],
                    "probs": [prob]
                }
            else:
                state_votes[state]["total_weight"] += vote_weight
                state_votes[state]["labels"].append(label)
                state_votes[state]["probs"].append(prob)
        
        # Ermittle den Zustand mit der höchsten Stimmgewichtung
        winner_state = max(state_votes, key=lambda s: state_votes[s]["total_weight"])
        
        # Bestimme häufigstes Label und durchschnittliche Konfidenz
        winner_info = state_votes[winner_state]
        most_common_label = max(set(winner_info["labels"]), key=winner_info["labels"].count)
        avg_prob = sum(winner_info["probs"]) / len(winner_info["probs"])
        
        return winner_state, most_common_label, avg_prob
    
    def _interpret_state(self, state_params):
        """
        Vereinfachte Interpretation eines Zustands basierend auf Mittelwerten.
        """
        mu = state_params["mu"]
        
        # Gewichtetes Mittel der Returns (ersten 4 Dimensionen)
        return_mean = 0
        weights = [0.25, 0.15, 0.3, 0.3]  # Gewichtungen für die verschiedenen Returns
        
        for i in range(min(4, len(mu))):
            return_mean += mu[i] * weights[i]
        
        # Einfache Kategorisierung
        if return_mean > 0.001:
            phase = "High Bullish"
        elif return_mean > 0:
            phase = "Low Bullish"
        elif return_mean > -0.001:
            phase = "Low Bearish"
        else:
            phase = "High Bearish"
        
        return phase

def select_best_hmm(models, features, criteria="likelihood"):
    """
    Wählt das beste HMM-Modell basierend auf verschiedenen Kriterien.
    
    Args:
        models: Liste von HMM-Modellen (pi, A, st_list, ll)
        features: Feature-Matrix für die Auswahl
        criteria: Kriterium für die Auswahl ("likelihood", "stability", "bic")
    
    Returns:
        best_model: Bestes Modell (pi, A, st_list, ll)
    """
    scores = []
    
    for i, (pi, A, st_list, ll) in enumerate(models):
        if criteria == "likelihood":
            # Einfach die beste Log-Likelihood
            scores.append(ll)
        
        elif criteria == "stability":
            # Stabilität der Zuordnung
            gamma, _, scale = forward_backward(features, pi, A, st_list, use_tdist=True)
            state_stability = compute_state_stability(gamma)
            stability_score = np.mean(state_stability)
            scores.append(stability_score)
        
        elif criteria == "bic":
            # Bayesian Information Criterion
            T, D = features.shape
            K = len(st_list)
            n_params = K * (D + 4) + K * K  # Anzahl Parameter im Modell
            bic = -2 * ll + n_params * np.log(T)
            scores.append(-bic)  # Negativ, da wir maximieren wollen
    
    # Finde Modell mit bestem Score
    best_idx = np.argmax(scores)
    return models[best_idx]

###############################################
# GPU-Beschleunigung
###############################################

# Prüfe, ob TensorFlow GPU-Unterstützung verfügbar ist
try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        logger.info("TensorFlow GPU-Unterstützung verfügbar.")
        CONFIG['has_gpu'] = True
    else:
        logger.info("Keine GPU für TensorFlow verfügbar.")
        CONFIG['has_gpu'] = False
except ImportError:
    logger.info("TensorFlow nicht installiert, GPU-Beschleunigung nicht verfügbar.")
    CONFIG['has_gpu'] = False
