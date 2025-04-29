# enhanced_vol_models_v2.py
# Modified to handle TensorFlow imports safely

import numpy as np
import math
from scipy.optimize import minimize
import warnings
import logging
import time

# Safe TensorFlow import
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logging.warning("TensorFlow not available, GPU acceleration disabled")

# Logger einrichten
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhanced_vol_models')


import math
from scipy.optimize import minimize
import warnings
import logging
import time

# Logger einrichten
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhanced_vol_models')

###############################################
# EGARCH(1,1) mit selektiven Dimensionen
###############################################

def egarch_recursion(r, mu, omega, alpha, gamma, beta):
    """
    EGARCH(1,1):
      log(sigma[t]^2) = omega + alpha*|z[t-1]| + gamma*z[t-1] + beta*log(sigma[t-1]^2)
      wobei z[t-1] = (r[t-1] - mu) / sigma[t-1]
    => Verbesserte Version mit numerischer Stabilität.
    """
    T = len(r)
    sigma = np.zeros(T)
    if T == 0:
        return sigma
    
    # Start:
    den = (1 - beta)
    if abs(den) < 1e-6:
        den = 1e-6 if den >= 0 else -1e-6
    log_sig2_0 = omega/den
    log_sig2_0 = max(min(log_sig2_0, 50.0), -50.0)
    tmp = math.exp(0.5 * log_sig2_0)
    if tmp < 1e-12:
        tmp = 1e-12
    sigma[0] = tmp
    
    for t in range(1, T):
        prev = max(sigma[t-1], 1e-12)
        prev_log = math.log(prev**2)
        
        z_ = (r[t-1] - mu)/prev
        val = omega + alpha*abs(z_) + gamma*z_ + beta*prev_log
        
        val_clamped = max(min(val, 50.0), -50.0)
        tmp2 = math.exp(0.5 * val_clamped)
        if tmp2 < 1e-12:
            tmp2 = 1e-12
        sigma[t] = tmp2
    
    return sigma

def egarch_neglog(params, r, mu, gamma_i):
    """
    Negative Log-Likelihood für EGARCH-Parameter.
    Optimiert für schnellere Berechnung und frühen Abbruch bei ungültigen Parametern.
    """
    (omega, alpha, gma, beta) = params
    if beta > 0.999 or beta < 0 or alpha < 0 or omega < -10:
        return np.inf
    
    try:
        sigma = egarch_recursion(r, mu, omega, alpha, gma, beta)
        
        # Vektorisierte Berechnung der Likelihood
        T = len(r)
        epsilon_sq = (r - mu)**2
        sig_sq = sigma**2
        
        log_densities = 0.5 * (np.log(2*math.pi*sig_sq) + epsilon_sq/sig_sq)
        ll = np.sum(gamma_i * log_densities)
        
        return ll
    except:
        return np.inf

def update_egarch_params(r, mu, gamma_i, old_params):
    """
    Update der EGARCH-Parameter mit verbesserter Fehlerbehandlung.
    """
    def obj_eg(p):
        return egarch_neglog(p, r, mu, gamma_i)
    
    bnds = [(-10, None), (0, None), (None, None), (0, 0.999)]
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(obj_eg, old_params, bounds=bnds, method='L-BFGS-B')
        
        if res.success:
            return tuple(res.x)
        
        # Bei Misserfolg: Probiere einen einfacheren Algorithmus
        res = minimize(obj_eg, old_params, bounds=bnds, method='SLSQP')
        if res.success:
            return tuple(res.x)
    except:
        logger.warning("Exception in EGARCH param optimization, keeping old values")
    
    return old_params

def update_egarch_regime_specific(r, mu, gamma_i, old_params, regime_type="neutral"):
    """
    Update der EGARCH-Parameter mit regime-spezifischen Bounds.
    
    Args:
        r: Return-Zeitreihe
        mu: Mittelwert
        gamma_i: Zustandsgewichte
        old_params: Bisherige EGARCH-Parameter (omega, alpha, gamma, beta)
        regime_type: Typ des Regimes ("high_bull", "high_bear", "low_vol", "neutral")
    
    Returns:
        Neue EGARCH-Parameter (omega, alpha, gamma, beta)
    """
    # Parameter-Constraints basierend auf Regime
    if regime_type == "high_bull":
        # Bullish Trend: Asymmetrie erlauben (positiver Gamma)
        bnds = [(-10, None), (0, 0.2), (0, 0.2), (0.5, 0.999)]
    elif regime_type == "high_bear":
        # Bearish Trend: Asymmetrie erlauben (negativer Gamma)
        bnds = [(-10, None), (0, 0.2), (-0.2, 0), (0.5, 0.999)]
    elif regime_type == "low_vol":
        # Niedrige Volatilität: Höhere Persistenz (Beta)
        bnds = [(-10, None), (0, 0.1), (-0.1, 0.1), (0.8, 0.999)]
    else:
        # Standard-Bounds
        bnds = [(-10, None), (0, None), (None, None), (0, 0.999)]
    
    def obj_eg(p):
        return egarch_neglog(p, r, mu, gamma_i)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(obj_eg, old_params, bounds=bnds, method='L-BFGS-B')
        
        if res.success:
            return tuple(res.x)
        
        # Bei Misserfolg: Probiere einen einfacheren Algorithmus
        res = minimize(obj_eg, old_params, bounds=bnds, method='SLSQP')
        if res.success:
            return tuple(res.x)
    except:
        logger.warning(f"Exception in EGARCH regime-specific optimization ({regime_type}), keeping old values")
    
    return old_params

def update_egarch_selected_dims(features, mu_d, gamma_i, old_params, dims_egarch=None, state_label=None):
    """
    Aktualisiert EGARCH-Parameter nur für die angegebenen Dimensionen.
    Mit regime-spezifischer Optimierung, wenn state_label bereitgestellt wird.
    
    Parameter:
        features: Feature-Matrix [T, D]
        mu_d: Mittelwertvektor [D]
        gamma_i: State-Gewichte [T]
        old_params: Alte EGARCH-Parameter (omega, alpha, gamma, beta)
        dims_egarch: Liste der Dimensionen (Indizes), die EGARCH verwenden sollen
                     None = alle Dimensionen
        state_label: Optional, Label des Zustands für regime-spezifische Optimierung
    """
    T, D = features.shape
    
    # Standardwert: Alle Dimensionen verwenden EGARCH
    if dims_egarch is None:
        dims_egarch = list(range(D))
    
    # Filterung auf relevante Dimensionen für EGARCH
    if len(dims_egarch) == 0:
        # Keine EGARCH-Dimensionen => behalte alte Parameter
        return old_params
    
    # Regime-Typ aus state_label extrahieren, falls vorhanden
    regime_type = "neutral"
    if state_label:
        if "High" in state_label and "Bullish" in state_label:
            regime_type = "high_bull"
        elif "High" in state_label and "Bearish" in state_label:
            regime_type = "high_bear"
        elif "Low" in state_label:
            regime_type = "low_vol"
    
    # EGARCH-Parameter-Update mit Regime-Berücksichtigung
    if state_label:
        # Für jede relevante Dimension separate Parameter optimieren
        all_params = [old_params] * len(dims_egarch)
        
        for i, d_idx in enumerate(dims_egarch):
            if d_idx < D:
                r_d = features[:, d_idx]
                # Regime-spezifische Optimierung
                all_params[i] = update_egarch_regime_specific(
                    r_d, mu_d[d_idx], gamma_i, old_params, regime_type
                )
        
        # Durchschnittliche Parameter berechnen
        new_params = [sum(p[j] for p in all_params) / len(all_params) for j in range(4)]
        return tuple(new_params)
    else:
        # EGARCH-Parameter-Update ohne Regime-Berücksichtigung
        def obj_eg(p):
            (omega, alpha, ga, beta) = p
            val = 0.0
            
            for d_idx in dims_egarch:
                if d_idx < D:  # Sicherheitsüberprüfung
                    r_d = features[:, d_idx]
                    val += egarch_neglog((omega, alpha, ga, beta), r_d, mu_d[d_idx], gamma_i)
            
            return val
        
        bnds = [(-10, None), (0, None), (None, None), (0, 0.999)]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(obj_eg, old_params, bounds=bnds, method='L-BFGS-B')
            
            if res.success:
                return tuple(res.x)
            
            # Bei Misserfolg: Probiere einen einfacheren Algorithmus
            res = minimize(obj_eg, old_params, bounds=bnds, method='SLSQP')
            if res.success:
                return tuple(res.x)
        except:
            logger.warning("Exception in EGARCH selected dims optimization, keeping old values")
        
        return old_params

###############################################
# EGARCH Sigma-Berechnung für multidimensionale Features
###############################################

def compute_sigma_array(features, mu_d, vol_model, params_vol, dims_egarch=None):
    """
    Berechnet Sigma-Arrays für jede Dimension.
    Optimiert für selektive EGARCH-Anwendung.
    
    Parameter:
        features: Feature-Matrix [T, D]
        mu_d: Mittelwertvektor [D]
        vol_model: Volatilitätsmodell ("EGARCH" oder "GARCH")
        params_vol: Parameter des Volatilitätsmodells
        dims_egarch: Liste der Dimensionen, die EGARCH verwenden sollen
                     None = alle Dimensionen
    
    Returns:
        sigma_array: Matrix [T, D] mit Volatilitäten
    """
    T, D = features.shape
    sigma_array = np.ones((T, D))
    
    # Standardwert: Alle Dimensionen verwenden EGARCH
    if dims_egarch is None:
        dims_egarch = list(range(D))
    
    if vol_model == "EGARCH":
        (om, al, ga, be) = params_vol
        # Nur für ausgewählte Dimensionen EGARCH anwenden
        for d_ in dims_egarch:
            if d_ < D:  # Sicherheitsüberprüfung
                r_d = features[:, d_]
                s_ = egarch_recursion(r_d, mu_d[d_], om, al, ga, be)
                sigma_array[:, d_] = s_
    else:
        # Fallback auf GARCH(1,1)
        (om_, al_, be_) = params_vol
        for d_ in dims_egarch:
            if d_ < D:
                r_d = features[:, d_]
                s_ = garch_recursion_1d(r_d, mu_d[d_], om_, al_, be_)
                sigma_array[:, d_] = s_
    
    return sigma_array

###############################################
# Fallback GARCH(1,1)
###############################################

def garch_recursion_1d(r, mu, om, al, be):
    """
    GARCH(1,1) Rekursion für eine einzelne Dimension.
    """
    T = len(r)
    sigma = np.zeros(T)
    
    # Equilibrium-Varianz
    val = max(om, 1e-12)/(1 - min(al+be, 0.999) + 1e-12)
    if val < 1e-12:
        val = 1e-12
    sigma[0] = math.sqrt(val)
    
    # GARCH-Rekursion
    for t in range(1, T):
        vv = max(om, 1e-12) + al*(r[t-1]-mu)**2 + be*sigma[t-1]**2
        if vv < 1e-12:
            vv = 1e-12
        sigma[t] = math.sqrt(vv)
    
    return sigma

###############################################
# T-Verteilung und Erweiterungen
###############################################

def t_pdf_diag(x, df):
    """
    Optimierte T-Verteilungs-PDF für diagonale Kovarianzmatrizen.
    Annahme: Kovarianzmatrix ist Einheitsmatrix (I).
    
    Mit verbesserter numerischer Stabilität, arbeitet mit logarithmischen Werten
    und stabilem Umgang mit hohen Dimensionen.
    """
    D = len(x)
    # Verwende dot product für val_
    val_ = np.dot(x, x)  # Cov=I => x^T x = Sum(x_i^2)
    
    # Berechne Dichten im Log-Space für bessere numerische Stabilität
    log_numer = math.lgamma((df + D)/2)
    log_denom = math.lgamma(df/2) + (D/2) * math.log(df*math.pi)
    log_factor = log_numer - log_denom
    
    # Exponent berechnen
    power = -0.5*(df + D)
    
    # Berechne log(1 + val_/df)
    log_term = math.log1p(val_/df)
    
    # Kombiniere zur log-Dichte
    log_pdf = log_factor + power * log_term
    
    # Konvertiere zurück (exp(log_pdf))
    pdf = math.exp(log_pdf)
    
    return pdf

def skewed_t_pdf_diag(x, df, skew_params):
    """
    Schiefe t-Verteilung mit unterschiedlichen Verzerrungsparametern pro Dimension.
    Kann für asymmetrische Return-Verteilungen verwendet werden.
    
    Args:
        x: Feature-Vektor
        df: Freiheitsgrade
        skew_params: Parameter für Schiefe pro Dimension
    
    Returns:
        Dichte der schiefen t-Verteilung
    """
    D = len(x)
    
    if len(skew_params) != D:
        # Fallback, wenn die Dimensionen nicht übereinstimmen
        return t_pdf_diag(x, df)
    
    try:
        # Wir berechnen die Dichte für jede Dimension getrennt
        log_dens = 0
        
        for d in range(D):
            skew = skew_params[d]
            xd = x[d]
            
            # Anpassung der Punktdichte basierend auf Vorzeichen
            if xd < 0:
                mod_x = xd * (2 - skew)  # Streckung/Stauchung für negative Werte
            else:
                mod_x = xd * skew  # Streckung/Stauchung für positive Werte
            
            # Univariate t-Verteilung (log-Dichte)
            log_c = math.lgamma((df + 1)/2) - math.lgamma(df/2) - 0.5 * math.log(df * math.pi)
            log_dens_d = log_c - ((df + 1)/2) * math.log1p((mod_x**2)/df)
            
            # Anpassungsfaktor für Schiefe
            log_dens_d += math.log(2 / (skew + 1/skew))
            
            log_dens += log_dens_d
        
        # Zurück zum linearen Bereich
        density = math.exp(log_dens)
        if math.isnan(density) or density < 1e-15:
            density = 1e-15
        
        return density
    except:
        # Fallback bei numerischen Problemen
        return t_pdf_diag(x, df)

def t_pdf_safe_diag(x, df):
    """
    Sichere Version der T-Verteilungs-PDF mit Fehlerbehandlung
    und Begrenzung extremer Werte.
    """
    try:
        val = t_pdf_diag(x, df)
        if math.isnan(val) or val < 1e-15:
            val = 1e-15
        return val
    except:
        return 1e-15

def normal_pdf_diag(x):
    """
    Multivariate Normalverteilung mit Diagonalkovarianz = I.
    """
    D = len(x)
    dot_product = np.dot(x, x)
    val = (2*math.pi)**(-0.5*D) * math.exp(-0.5*dot_product)
    return max(val, 1e-15)

def hybrid_pdf_diag(x, df, mixture_weight=0.5):
    """
    Hybridverteilung: Mischung aus T und Normalverteilung.
    
    Args:
        x: Feature-Vektor
        df: Freiheitsgrade der T-Verteilung
        mixture_weight: Gewicht der T-Verteilung (0-1), Normal = 1-weight
    
    Returns:
        Dichte der Mischverteilung
    """
    t_dens = t_pdf_safe_diag(x, df)
    normal_dens = normal_pdf_diag(x)
    
    # Gewichtete Mischung
    w = max(0, min(1, mixture_weight))  # Sicherstellen, dass w in [0,1]
    
    hybrid_density = w * t_dens + (1-w) * normal_dens
    return max(hybrid_density, 1e-15)

def neglog_tdist_df(df, e_array, gamma_i):
    """
    Negative Log-Likelihood der T-Verteilung für df-Optimierung.
    Optimiert mit frühem Abbruch für ungültige df-Werte.
    """
    if df < 2 or df > 200:
        return 1e15
    
    T, D = e_array.shape
    val = 0.0
    
    # Vektorisierte Berechnung mit Schleifen-Abroller für größere Datensätze
    chunk_size = 50  # Abroller-Größe
    for chunk_start in range(0, T, chunk_size):
        chunk_end = min(chunk_start + chunk_size, T)
        
        for t in range(chunk_start, chunk_end):
            p_ = t_pdf_safe_diag(e_array[t, :], df)
            val_ = -math.log(p_)
            val += gamma_i[t] * val_
    
    return val

def neglog_skewed_tdist(params, e_array, gamma_i):
    """
    Negative Log-Likelihood für schiefe T-Verteilung.
    
    Args:
        params: Parameter [df, skew_1, ..., skew_D]
        e_array: Standardisierte Fehler [T, D]
        gamma_i: Zustandsgewichte [T]
    
    Returns:
        Negative Log-Likelihood
    """
    df = params[0]
    skew_params = params[1:]
    
    if df < 2 or df > 200:
        return 1e15
    
    T, D = e_array.shape
    
    # Überprüfe, ob genug Skew-Parameter vorhanden sind
    if len(skew_params) < D:
        # Ergänze fehlende Parameter mit 1.0 (keine Schiefe)
        skew_params = np.concatenate([skew_params, np.ones(D - len(skew_params))])
    
    # Begrenze Skew-Parameter auf sinnvollen Bereich
    skew_params = np.clip(skew_params, 0.5, 2.0)
    
    val = 0.0
    for t in range(T):
        p_ = skewed_t_pdf_diag(e_array[t, :], df, skew_params)
        val_ = -math.log(p_)
        val += gamma_i[t] * val_
    
    return val

def update_df_t(e_array, gamma_i, old_df):
    """
    Update des Freiheitsgrads der T-Verteilung.
    Mit Stabilisierung für Konvergenzprobleme.
    """
    def obj_f(df_):
        return neglog_tdist_df(df_[0], e_array, gamma_i)
    
    bnds = [(2, 200)]
    
    # Versuche zuerst mit dem Standard-Algorithmus
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(obj_f, [old_df], bounds=bnds, method='L-BFGS-B')
        
        if res.success:
            return res.x[0]
        
        # Bei Misserfolg: Probiere einen einfacheren Algorithmus
        res = minimize(obj_f, [old_df], bounds=bnds, method='SLSQP')
        if res.success:
            return res.x[0]
    except:
        logger.warning("Exception in df optimization, keeping old value")
    
    return old_df

def update_skewed_t_params(e_array, gamma_i, old_params):
    """
    Aktualisiert die Parameter der schiefen t-Verteilung.
    
    Args:
        e_array: Standardisierte Fehler [T, D]
        gamma_i: Zustandsgewichte [T]
        old_params: [df, skew_1, ..., skew_D]
    
    Returns:
        Neue Parameter [df, skew_1, ..., skew_D]
    """
    T, D = e_array.shape
    
    # Stelle sicher, dass genügend Parameter vorhanden sind
    if len(old_params) < D + 1:
        old_params = np.concatenate([old_params[:1], np.ones(D)])
    
    def obj_f(params):
        return neglog_skewed_tdist(params, e_array, gamma_i)
    
    # Bounds: df und alle Skew-Parameter
    bnds = [(2, 200)] + [(0.5, 2.0)] * D
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(obj_f, old_params, bounds=bnds, method='L-BFGS-B')
        
        if res.success:
            return res.x
        
        # Vereinfachte Optimierung nur für df
        def obj_df(df_):
            return neglog_tdist_df(df_[0], e_array, gamma_i)
        
        res = minimize(obj_df, [old_params[0]], bounds=[(2, 200)], method='L-BFGS-B')
        if res.success:
            # Behalte alte Skew-Parameter
            return np.concatenate([res.x, old_params[1:]])
    except:
        logger.warning("Exception in skewed t-distribution optimization, keeping old values")
    
    return old_params

###############################################
# Live-Anwendung Funktionen
###############################################

def egarch_update_1step_multi(r_tminus1_vec, mu_vec, sigma_prev_vec, omega, alpha, gamma, beta, dims_egarch=None):
    """
    EGARCH-Update für einen einzelnen Zeitschritt, mehrere Dimensionen.
    Nur die in dims_egarch angegebenen Dimensionen verwenden EGARCH,
    die anderen bleiben konstant bei 1.0.
    """
    # Ensure we have numpy arrays with proper dimensions
    import numpy as np
    r_tminus1_vec = np.atleast_1d(r_tminus1_vec)
    mu_vec = np.atleast_1d(mu_vec)
    sigma_prev_vec = np.atleast_1d(sigma_prev_vec)
    
    # Get dimensions with safety checks
    D_r = len(r_tminus1_vec)
    D_mu = len(mu_vec)
    D_sigma = len(sigma_prev_vec)
    
    # Use minimum dimension to avoid index errors
    D = min(D_r, D_mu, D_sigma)
    
    # Create a copy of sigma_prev_vec to modify
    new_sigma = np.copy(sigma_prev_vec)
    
    # Standardwert: Alle Dimensionen verwenden EGARCH (aber nur bis D)
    if dims_egarch is None:
        dims_egarch = list(range(D))
    
    # Filter invalid dimensions
    valid_dims_egarch = [d for d in dims_egarch if d < D]
    
    # Log warning if dimensions were filtered
    if len(valid_dims_egarch) < len(dims_egarch):
        import logging
        logging.warning(f"Filtered {len(dims_egarch) - len(valid_dims_egarch)} invalid EGARCH dimensions. "
                      f"Feature vector has {D} dimensions, but dims_egarch contained indices up to {max(dims_egarch) if dims_egarch else 0}")
    
    for d_ in range(D):
        if d_ not in valid_dims_egarch:
            # Nicht-EGARCH-Dimensionen: Konstante Volatilität
            new_sigma[d_] = 1.0
        else:
            # EGARCH-Dimensionen: Update mit EGARCH-Formel
            try:
                prev = max(sigma_prev_vec[d_], 1e-12)
                z_ = (r_tminus1_vec[d_] - mu_vec[d_]) / prev
                prev_log = math.log(prev**2)
                val = omega + alpha*abs(z_) + gamma*z_ + beta*prev_log
                val_clamped = max(min(val, 50.0), -50.0)
                sig_ = math.exp(0.5*val_clamped)
                if sig_ < 1e-12:
                    sig_ = 1e-12
                new_sigma[d_] = sig_
            except Exception as e:
                # If anything goes wrong, use previous sigma value as fallback
                import logging
                logging.error(f"Error in EGARCH update for dimension {d_}: {str(e)}")
                new_sigma[d_] = sigma_prev_vec[d_]
    
    return new_sigma

def t_pdf_diag_multidim(x, df):
    """
    T-Verteilung für multidimensionale Vektoren mit diagonaler Kovarianzmatrix.
    Optimiert für Live-Anwendung mit verbesserter numerischer Stabilität.
    """
    D = len(x)
    if D < 1:
        return 1e-15
    
    try:
        # Optimierte Version für höhere Dimensionen
        if D > 5:
            return t_pdf_safe_diag(x, df)
        
        # Für niedrigere Dimensionen: Explizite Berechnung
        import math
        c1 = math.gamma((df+1)/2)/( math.sqrt(df*math.pi)*math.gamma(df/2) )
        
        # Logarithmische Berechnung für bessere Stabilität
        log_val = 0.0
        for d_ in range(D):
            xd = x[d_]
            log_part = math.log(c1) - ((df+1)/2) * math.log1p(xd*xd/df)
            log_val += log_part
        
        # Konvertiere zurück
        val = math.exp(log_val)
        
        if math.isnan(val) or val < 1e-15:
            val = 1e-15
        
        return val
    except:
        return 1e-15

###############################################
# Mögliche GPU-Beschleunigung mit TensorFlow
###############################################

def setup_gpu():
    """
    Konfiguriert die GPU-Nutzung und gibt zurück, ob GPU verfügbar ist.
    Optimiert die Einstellungen für bessere Performance.
    
    Returns:
        bool: True wenn GPU verfügbar und konfiguriert, False sonst
    """
    try:
        import tensorflow as tf
        from tensorflow.python.client import device_lib
        
        # Überprüfe, ob GPU vorhanden
        gpus = tf.config.list_physical_devices('GPU')
        has_gpu = len(gpus) > 0
        
        if has_gpu:
            # Optimierte Einstellungen für bessere Performance
            # Verhindert, dass TF alle GPU-Ressourcen reserviert
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Konfiguriere für Mixed Precision (schnellere Berechnungen)
            try:
                from tensorflow.keras.mixed_precision import experimental as mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision aktiviert für bessere GPU-Performance")
            except:
                logger.info("Mixed precision nicht unterstützt, verwende standard Präzision")
            
            # Optimiere für kürzere Batch-Verarbeitungen
            tf.config.optimizer.set_jit(False)  # JIT für kurze Operationen oft langsamer
            
            logger.info(f"GPU-Unterstützung aktiviert: {len(gpus)} GPU(s) verfügbar")
            logger.info(f"Verfügbare Geräte: {[d.name for d in device_lib.list_local_devices()]}")
            
            return True
        else:
            logger.info("Keine GPU verfügbar, verwende CPU")
            return False
    except ImportError:
        logger.info("TensorFlow ist nicht installiert. GPU-Beschleunigung nicht verfügbar.")
        return False
    except Exception as e:
        logger.warning(f"Fehler bei GPU-Konfiguration: {str(e)}")
        return False

class EGARCH_GPU:
    """
    GPU-beschleunigte EGARCH-Berechnung mit TensorFlow.
    """
    def __init__(self, use_mixed_precision=True):
        # Check if TensorFlow is available
        if not HAS_TF:
            logging.warning("TensorFlow not available, EGARCH_GPU disabled")
            self.has_gpu = False
            return
            
        # TensorFlow is available
        self.tf = tf
        self.use_mixed_precision = use_mixed_precision
        
        # Standardparameter 
        self.omega = tf.Variable(0.0, dtype=tf.float32)
        self.alpha = tf.Variable(0.1, dtype=tf.float32)
        self.gamma = tf.Variable(0.0, dtype=tf.float32)
        self.beta = tf.Variable(0.8, dtype=tf.float32)
        
        # Status-Flags
        self.is_compiled = False
        self.has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        
        if self.has_gpu and use_mixed_precision:
            try:
                from tensorflow.keras.mixed_precision import experimental as mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                logging.info("EGARCH_GPU: Mixed precision aktiviert")
            except:
                logging.info("EGARCH_GPU: Mixed precision nicht verfügbar")
        
        logging.info(f"EGARCH_GPU initialisiert, GPU verfügbar: {self.has_gpu}")
class OptimizedEGARCH:
    """
    Optimierte EGARCH-Berechnung mit automatischer GPU/CPU-Auswahl.
    Integriert beide Implementierungen und wählt die beste aus.
    """
    def __init__(self):
        """Initialisiert den optimierten EGARCH-Prozessor."""
        self.has_gpu = False
        self.gpu_engine = None
        self.params = {"omega": 0.0, "alpha": 0.1, "gamma": 0.0, "beta": 0.8}
        
        # Versuche GPU-Initialisierung
        try:
            has_gpu = setup_gpu()
            if has_gpu:
                self.gpu_engine = EGARCH_GPU()
                self.has_gpu = self.gpu_engine.has_gpu
                if self.has_gpu:
                    logger.info("OptimizedEGARCH: GPU-Modus aktiviert")
                else:
                    logger.info("OptimizedEGARCH: GPU-Setup fehlgeschlagen, verwende CPU")
        except Exception as e:
            logger.warning(f"OptimizedEGARCH: GPU-Initialisierung fehlgeschlagen: {str(e)}")
            self.has_gpu = False
    
    def set_params(self, omega, alpha, gamma, beta):
        """Setzt die EGARCH-Parameter."""
        self.params["omega"] = omega
        self.params["alpha"] = alpha
        self.params["gamma"] = gamma
        self.params["beta"] = beta
        
        # Aktualisiere auch GPU-Engine, falls verfügbar
        if self.has_gpu and self.gpu_engine:
            self.gpu_engine.set_params(omega, alpha, gamma, beta)
    
    def compute_sigma(self, returns, mu, use_gpu=None):
        """
        Berechnet Sigma mit optimaler Methode.
        
        Args:
            returns: Returns-Array
            mu: Mittelwert
            use_gpu: Überschreibt GPU-Auswahl (None=automatisch)
        
        Returns:
            Sigma-Array
        """
        # Bestimme, ob GPU verwendet werden soll
        should_use_gpu = self.has_gpu
        if use_gpu is not None:
            should_use_gpu = use_gpu and self.has_gpu
        
        # Größenheuristik: GPU ist oft langsamer für kleine Arrays
        if len(returns) < 50:
            should_use_gpu = False
        
        # Verwende entsprechende Implementierung
        if should_use_gpu:
            return self.gpu_engine.compute_sigma_np(returns, mu)
        else:
            # CPU-Implementierung
            return egarch_recursion(returns, mu, 
                                   self.params["omega"], 
                                   self.params["alpha"], 
                                   self.params["gamma"], 
                                   self.params["beta"])
    
    def compute_sigma_multidim(self, features, mu_vec, dims_egarch=None):
        """
        Berechnet Sigma für mehrere Dimensionen.
        
        Args:
            features: Feature-Matrix [T, D]
            mu_vec: Mittelwertvektor [D]
            dims_egarch: EGARCH-Dimensionen
        
        Returns:
            Sigma-Matrix [T, D]
        """
        if self.has_gpu and self.gpu_engine:
            return self.gpu_engine.compute_sigma_multidim(features, mu_vec, dims_egarch)
        else:
            # CPU-Fallback
            T, D = features.shape
            sigma_array = np.ones((T, D))
            
            # Standardwert: Alle Dimensionen
            if dims_egarch is None:
                dims_egarch = list(range(D))
            
            # Für jede EGARCH-Dimension
            for d_ in dims_egarch:
                if d_ < D:
                    r_d = features[:, d_]
                    mu_d = mu_vec[d_]
                    sigma_array[:, d_] = self.compute_sigma(r_d, mu_d, use_gpu=False)
            
            return sigma_array

# Globale Instanzen erstellen für einfachere Verwendung
try:
    GLOBAL_EGARCH = OptimizedEGARCH()
    HAS_GPU = GLOBAL_EGARCH.has_gpu
except Exception as e:
    logger.error(f"Fehler bei globaler EGARCH-Initialisierung: {str(e)}")
    GLOBAL_EGARCH = None
    HAS_GPU = False
    
    # Teste, ob GPU verfügbar ist
    if tf.config.list_physical_devices('GPU'):
        logger.info("TensorFlow GPU-Unterstützung ist verfügbar.")
        HAS_GPU = True
    else:
        logger.info("Keine GPU für TensorFlow verfügbar. Verwende CPU-Implementierung.")
        HAS_GPU = False
except ImportError:
    logger.info("TensorFlow ist nicht installiert. GPU-Beschleunigung nicht verfügbar.")
    HAS_GPU = False
