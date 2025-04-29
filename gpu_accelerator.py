#!/usr/bin/env python3
"""
Enhanced GPU Acceleration Module with Memory Management for HMM Trading System
----------------------------------------------
This module provides GPU-accelerated versions of computation-intensive functions
with robust memory management and graceful fallback mechanisms.

Key features:
- Automatic GPU memory monitoring and management
- Chunked processing for large datasets
- Dynamic batch sizing based on available memory
- Graceful fallback to CPU when needed
- Memory-efficient implementations of critical algorithms
"""

import numpy as np
import math
import logging
import time
import os
import warnings
from functools import wraps
import gc
import threading
import queue
from typing import Dict, List, Tuple, Callable, Union, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Global flags for module state
HAS_TF = False
HAS_GPU = False
MIXED_PRECISION = False
GPU_MEMORY_LIMIT = None  # Will be set during initialization
GPU_MEMORY_FRACTION = 0.6  # Default: use 80% of available GPU memory
MEMORY_GROWTH_ENABLED = True
ENABLE_CHUNKING = True  # Enable processing data in chunks
MAX_TENSOR_SIZE = 1000  # Maximum size of tensor dimensions for safety

# Try to import TensorFlow with proper error handling
try:
    import tensorflow as tf
    HAS_TF = True
    
    # Configure TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF log verbosity
    tf.get_logger().setLevel('ERROR')
    
    # GPU availability check
    gpus = tf.config.list_physical_devices('GPU')
    HAS_GPU = len(gpus) > 0
    
    if HAS_GPU:
        # Save original GPU memory settings
        original_gpu_configs = {}
        
        # Configure memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            try:
                if MEMORY_GROWTH_ENABLED:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Dynamic memory growth enabled for GPU {gpu.name}")
                
                # Save original config for potential restoration
                original_gpu_configs[gpu.name] = {"memory_growth": MEMORY_GROWTH_ENABLED}
                
            except RuntimeError as e:
                logger.warning(f"Error configuring GPU {gpu.name}: {e}")
        
        # Try to estimate available GPU memory
        try:
            gpu_mem_info = tf.config.experimental.get_memory_info('GPU:0')
            total_memory = gpu_mem_info['total'] if 'total' in gpu_mem_info else None
            
            if total_memory:
                # Convert bytes to MB for easier interpretation
                total_memory_mb = total_memory / (1024 * 1024)
                available_memory_mb = total_memory_mb * GPU_MEMORY_FRACTION
                
                GPU_MEMORY_LIMIT = available_memory_mb
                logger.info(f"GPU memory limit set to {GPU_MEMORY_LIMIT:.2f} MB " +
                          f"({GPU_MEMORY_FRACTION*100:.1f}% of {total_memory_mb:.2f} MB)")
            else:
                logger.warning("Could not determine GPU memory size, using default safety limits")
        except Exception as e:
            logger.warning(f"Error estimating GPU memory: {e}")
        
        # Try enabling mixed precision
        if tf.__version__ >= '2.4.0':
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                MIXED_PRECISION = True
                logger.info("Mixed precision (FP16) enabled for GPU computations")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
        
        logger.info(f"GPU acceleration available: {len(gpus)} GPU(s) detected")
    else:
        logger.info("No GPU detected. Using CPU for computations.")
        
except ImportError as e:
    logger.warning(f"TensorFlow import error: {e}")
    logger.warning("GPU acceleration not available. Using CPU implementations.")
except Exception as e:
    logger.warning(f"Error initializing TensorFlow: {e}")
    logger.warning("GPU acceleration not available. Using CPU implementations.")

def gpu_memory_monitor():
    """
    Returns current GPU memory usage information
    
    Returns:
        dict: Memory information with total, available, and used memory
    """
    if not HAS_TF or not HAS_GPU:
        return None
    
    try:
        # Get memory info
        gpu_info = tf.config.experimental.get_memory_info('GPU:0')
        
        # Calculate used memory
        total_bytes = gpu_info['total'] if 'total' in gpu_info else 0
        available_bytes = gpu_info['available'] if 'available' in gpu_info else 0
        used_bytes = total_bytes - available_bytes
        
        # Convert to MB for readability
        total_mb = total_bytes / (1024 * 1024)
        available_mb = available_bytes / (1024 * 1024)
        used_mb = used_bytes / (1024 * 1024)
        
        # Calculate percentage
        usage_percent = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
        
        return {
            "total_mb": total_mb,
            "available_mb": available_mb,
            "used_mb": used_mb,
            "usage_percent": usage_percent,
        }
    except Exception as e:
        logger.debug(f"Error monitoring GPU memory: {e}")
        return None

def clear_gpu_memory():
    """
    Attempts to clear GPU memory by clearing TensorFlow session
    and forcing garbage collection
    """
    if not HAS_TF:
        return False
    
    try:
        # Clear TensorFlow session to release memory
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        gc.collect()
        
        # Try to run a small TensorFlow operation to trigger memory cleanup
        if HAS_GPU:
            with tf.device('/GPU:0'):
                # Small dummy operation
                _ = tf.constant([1.0, 2.0])
                _ = tf.constant([3.0, 4.0])
                
        logger.debug("GPU memory cleared")
        return True
    except Exception as e:
        logger.warning(f"Error clearing GPU memory: {e}")
        return False

def memory_safe_execution(func):
    """
    Decorator for memory-safe execution of GPU operations with fallback to CPU
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function that handles memory safely
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_TF or not HAS_GPU:
            return func(*args, **kwargs)
        
        # Check memory before execution
        mem_info = gpu_memory_monitor()
        if mem_info and mem_info["usage_percent"] > 95:
            logger.warning(f"GPU memory usage critical: {mem_info['usage_percent']:.1f}%, clearing memory")
            clear_gpu_memory()
        
        try:
            # Try GPU execution
            result = func(*args, **kwargs)
            return result
        except tf.errors.ResourceExhaustedError as e:
            # Handle out of memory error
            logger.warning(f"GPU out of memory during {func.__name__}. Clearing memory and retrying on CPU.")
            clear_gpu_memory()
            
            # Force CPU execution by adding CPU device context to kwargs
            kwargs['force_cpu'] = True
            try:
                return func(*args, **kwargs)
            except Exception as e2:
                logger.error(f"CPU fallback also failed: {str(e2)}")
                raise
        except Exception as e:
            # Handle other errors
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
            
    return wrapper

def get_optimal_chunk_size(tensor_size, dtype=np.float32):
    """
    Calculate optimal chunk size based on tensor size and available GPU memory
    
    Args:
        tensor_size: Size of a single element in the tensor
        dtype: Data type of the tensor
        
    Returns:
        int: Optimal chunk size for processing
    """
    if not GPU_MEMORY_LIMIT:
        # Conservative default if we can't determine memory
        return 1000
    
    # Get element size in bytes
    element_size = np.dtype(dtype).itemsize
    
    # Calculate total memory needed for tensor
    total_tensor_bytes = tensor_size * element_size
    
    # Convert memory limit from MB to bytes
    memory_limit_bytes = GPU_MEMORY_LIMIT * 1024 * 1024
    
    # Allocate 50% of available memory for this operation
    # (leaving room for intermediate results and other operations)
    safe_memory_bytes = memory_limit_bytes * 0.5
    
    # Calculate chunk size
    chunk_size = safe_memory_bytes / total_tensor_bytes
    
    # Ensure chunk size is at least 100 but no more than input size
    chunk_size = max(100, min(tensor_size, int(chunk_size)))
    
    return chunk_size

class BatchProcessor:
    """
    Processes large datasets in batches to avoid memory overflow
    """
    def __init__(self, batch_size=None, device=None):
        """
        Initialize the batch processor
        
        Args:
            batch_size: Size of batches (None for auto-determination)
            device: Device to use for processing
        """
        self.batch_size = batch_size
        self.device = device or ('GPU:0' if HAS_GPU else 'CPU:0')
    
    def process(self, data, process_fn, **kwargs):
        """
        Process data in batches
        
        Args:
            data: Input data (numpy array)
            process_fn: Function to process each batch
            **kwargs: Additional arguments for process_fn
            
        Returns:
            Processed results
        """
        if not ENABLE_CHUNKING or not HAS_TF:
            # Process all at once if chunking disabled or TF not available
            return process_fn(data, **kwargs)
        
        # Determine batch size if not specified
        if self.batch_size is None:
            # Auto-determine based on input size
            self.batch_size = get_optimal_chunk_size(data.shape[0])
        
        total_samples = data.shape[0]
        results = []
        
        # Process data in batches
        for start_idx in range(0, total_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_samples)
            batch = data[start_idx:end_idx]
            
            # Process batch
            with tf.device(self.device):
                batch_result = process_fn(batch, **kwargs)
                results.append(batch_result)
        
        # Combine results
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results, axis=0)
        elif isinstance(results[0], tf.Tensor):
            return tf.concat(results, axis=0)
        else:
            return results

class GPUAccelerator:
    """
    Enhanced GPU accelerator for HMM trading system with memory management
    """
    def __init__(self, use_gpu=True, memory_fraction=0.8, enable_mixed_precision=True, 
                 device=None, verbose=False, enable_chunking=True):
        """
        Initialize GPU accelerator with memory management
        
        Args:
            use_gpu: Use GPU if available
            memory_fraction: Fraction of GPU memory to use
            enable_mixed_precision: Enable mixed precision (FP16) for faster computation
            device: Specific device to use
            verbose: Enable verbose logging
            enable_chunking: Enable processing data in chunks
        """
        self.use_gpu = use_gpu and HAS_GPU
        self.memory_fraction = memory_fraction
        self.mixed_precision = enable_mixed_precision and MIXED_PRECISION
        self.verbose = verbose
        self.enable_chunking = enable_chunking
        self.device = device or ('GPU:0' if self.use_gpu else 'CPU:0')
        
        # Update global settings
        global GPU_MEMORY_FRACTION, ENABLE_CHUNKING
        GPU_MEMORY_FRACTION = memory_fraction
        ENABLE_CHUNKING = enable_chunking
        
        # Initialize TensorFlow functions
        self.tf = tf if HAS_TF else None
        self.compiled_functions = {}
        self.batch_processor = BatchProcessor(device=self.device)
        
        # Track initialization status
        self.initialized = False
        
        if self.tf is not None:
            self._create_tf_functions()
            self.initialized = True
        
        # Warm up the GPU with a small operation
        if self.use_gpu:
            self._warmup_gpu()
        
        logger.info(f"GPU Accelerator initialized: GPU={self.use_gpu}, "
                   f"Mixed precision={self.mixed_precision}, "
                   f"Chunking={self.enable_chunking}")
    
    def _warmup_gpu(self):
        """Warm up the GPU with a small operation to initialize resources"""
        if not HAS_TF or not self.use_gpu:
            return
        
        try:
            with tf.device(self.device):
                # Simple matrix multiplication to warm up GPU
                a = tf.random.normal((10, 10))
                b = tf.random.normal((10, 10))
                _ = tf.matmul(a, b)
                tf.keras.backend.clear_session()  # Clear the graph
                
                logger.debug("GPU warmup completed")
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")
    
    def _create_tf_functions(self):
        """Create and compile TensorFlow functions for GPU acceleration"""
        if not HAS_TF:
            return
        
        with tf.device(self.device):
            # EGARCH recursion with memory optimization
            @tf.function
            def tf_egarch_recursion_optimized(r, mu, omega, alpha, gamma, beta, chunk_size=None):
                """
                Memory-optimized EGARCH(1,1) implementation that processes data in chunks
                
                Args:
                    r: Return time series
                    mu: Mean
                    omega, alpha, gamma, beta: EGARCH parameters
                    chunk_size: Size of chunks for processing
                    
                Returns:
                    Sigma values
                """
                # Get tensor length
                T = tf.shape(r)[0]
                
                # Use chunking if enabled and tensor is large
                use_chunking = self.enable_chunking and chunk_size is not None and T > chunk_size
                
                if use_chunking:
                    # Process in chunks
                    chunks = tf.math.ceil(tf.cast(T, tf.float32) / tf.cast(chunk_size, tf.float32))
                    chunks = tf.cast(chunks, tf.int32)
                    
                    # Initialize result tensor
                    sigma = tf.TensorArray(tf.float32, size=T)
                    
                    # Calculate equilibrium variance for initialization
                    den = (1.0 - beta)
                    den = tf.where(tf.abs(den) < 1e-6, tf.sign(den) * 1e-6, den)
                    log_sig2_0 = omega / den
                    log_sig2_0 = tf.clip_by_value(log_sig2_0, -20.0, 20.0)
                    sigma_0 = tf.math.exp(0.5 * log_sig2_0)
                    sigma_0 = tf.maximum(sigma_0, 1e-6)
                    
                    # Initialize first value
                    sigma = sigma.write(0, sigma_0)
                    
                    # Process each chunk
                    for chunk_idx in tf.range(chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = tf.minimum(start_idx + chunk_size, T)
                        
                        # Process indices in current chunk (skip first point which is already set)
                        for t in tf.range(tf.maximum(start_idx, 1), end_idx):
                            prev_sigma = sigma.read(t-1)
                            prev_r = r[t-1]
                            
                            # Calculate standardized residual
                            prev_sigma = tf.maximum(prev_sigma, 1e-6)
                            z_ = (prev_r - mu) / prev_sigma
                            
                            # EGARCH equation
                            val = omega + alpha * tf.abs(z_) + gamma * z_ + beta * tf.math.log(tf.square(prev_sigma))
                            
                            # Numerical stability
                            val_clamped = tf.clip_by_value(val, -20.0, 20.0)
                            sigma_t = tf.math.exp(0.5 * val_clamped)
                            sigma_t = tf.maximum(sigma_t, 1e-6)
                            
                            # Store result
                            sigma = sigma.write(t, sigma_t)
                    
                    # Return stacked result
                    return sigma.stack()
                else:
                    # Process entire sequence at once (original implementation)
                    sigma = tf.TensorArray(tf.float32, size=T)
                    
                    # Start with equilibrium variance
                    den = (1.0 - beta)
                    den = tf.where(tf.abs(den) < 1e-6, tf.sign(den) * 1e-6, den)
                    log_sig2_0 = omega / den
                    log_sig2_0 = tf.clip_by_value(log_sig2_0, -20.0, 20.0)
                    sigma_0 = tf.math.exp(0.5 * log_sig2_0)
                    sigma_0 = tf.maximum(sigma_0, 1e-6)
                    sigma = sigma.write(0, sigma_0)
                    
                    # EGARCH recursion
                    for t in tf.range(1, T):
                        prev = tf.maximum(sigma.read(t-1), 1e-6)
                        z_ = (r[t-1] - mu) / prev
                        val = omega + alpha * tf.abs(z_) + gamma * z_ + beta * tf.math.log(tf.square(prev))
                        val_clamped = tf.clip_by_value(val, -20.0, 20.0)
                        sigma_t = tf.math.exp(0.5 * val_clamped)
                        sigma_t = tf.maximum(sigma_t, 1e-6)
                        sigma = sigma.write(t, sigma_t)
                    
                    return sigma.stack()
            
            # T-distribution PDF with numerical stability
            @tf.function
            def tf_t_pdf_diag_stable(x, df):
                """
                Numerically stable implementation of T-distribution PDF
                
                Args:
                    x: Input data
                    df: Degrees of freedom
                    
                Returns:
                    PDF value
                """
                # Get dimension
                D = tf.shape(x)[0]
                
                # Compute sum of squares (equivalent to x^T x for spherical covariance)
                val_ = tf.reduce_sum(tf.square(x))
                
                # Calculate log-PDF for numerical stability
                log_numer = tf.math.lgamma((df + tf.cast(D, tf.float32)) / 2.0)
                log_denom = tf.math.lgamma(df / 2.0) + (tf.cast(D, tf.float32) / 2.0) * tf.math.log(df * np.pi)
                log_factor = log_numer - log_denom
                
                # Compute exponent
                power = -0.5 * (df + tf.cast(D, tf.float32))
                
                # Compute log(1 + val_/df) with numerical stability
                log_term = tf.math.log1p(val_ / df)
                
                # Combine to log-density
                log_pdf = log_factor + power * log_term
                
                # Convert to linear scale with numerical stability
                pdf = tf.math.exp(log_pdf)
                pdf = tf.maximum(pdf, 1e-15)  # Ensure minimum value for stability
                
                return pdf
            
            # Forward-backward algorithm components optimized for memory
            @tf.function
            def tf_forward_step_stable(alpha_prev, A, B_t):
                """
                Numerically stable forward step for HMM
                
                Args:
                    alpha_prev: Previous alpha values
                    A: Transition matrix
                    B_t: Emission probabilities
                    
                Returns:
                    alpha_t, scale factor
                """
                # Calculate alpha_t
                alpha_unnorm = tf.matmul(tf.expand_dims(alpha_prev, 0), A)[0] * B_t
                
                # Apply scaling for numerical stability
                scale = tf.reduce_sum(alpha_unnorm)
                scale = tf.maximum(scale, 1e-300)  # Minimum scale to prevent division by zero
                alpha_t = alpha_unnorm / scale
                
                return alpha_t, scale
            
            @tf.function
            def tf_backward_step_stable(beta_next, A, B_next, scale_next):
                """
                Numerically stable backward step for HMM
                
                Args:
                    beta_next: Next beta values
                    A: Transition matrix
                    B_next: Next emission probabilities
                    scale_next: Next scale factor
                    
                Returns:
                    beta_t
                """
                # Calculate beta_t
                beta_t = tf.linalg.matvec(A, B_next * beta_next)
                
                # Scale by same factor as forward pass
                beta_t = beta_t / scale_next
                
                return beta_t
        
        # Store compiled functions
        self.compiled_functions['egarch_recursion'] = tf_egarch_recursion_optimized
        self.compiled_functions['t_pdf_diag'] = tf_t_pdf_diag_stable
        self.compiled_functions['forward_step'] = tf_forward_step_stable
        self.compiled_functions['backward_step'] = tf_backward_step_stable
    
    @memory_safe_execution
    def egarch_recursion(self, r, mu, omega, alpha, gamma, beta, force_cpu=False):
        """
        Memory-safe EGARCH(1,1) recursion with automatic chunking
        
        Args:
            r: Return time series
            mu: Mean
            omega, alpha, gamma, beta: EGARCH parameters
            force_cpu: Force CPU execution
            
        Returns:
            numpy array of sigma values
        """
        # Check if TensorFlow is available
        if not self.initialized or self.tf is None or force_cpu:
            # Fall back to CPU implementation
            from enhanced_vol_models_v2 import egarch_recursion
            return egarch_recursion(r, mu, omega, alpha, gamma, beta)
        
        try:
            # Get memory information
            mem_info = gpu_memory_monitor()
            
            # Convert inputs to TensorFlow tensors
            r_tf = tf.convert_to_tensor(r, dtype=tf.float32)
            mu_tf = tf.constant(mu, dtype=tf.float32)
            omega_tf = tf.constant(omega, dtype=tf.float32)
            alpha_tf = tf.constant(alpha, dtype=tf.float32)
            gamma_tf = tf.constant(gamma, dtype=tf.float32)
            beta_tf = tf.constant(beta, dtype=tf.float32)
            
            # Determine chunk size based on sequence length and available memory
            chunk_size = None
            if self.enable_chunking and len(r) > 1000:
                chunk_size = get_optimal_chunk_size(len(r))
                if self.verbose:
                    logger.debug(f"Processing EGARCH in chunks of {chunk_size}")
            
            # Execute TensorFlow function
            with tf.device(self.device if not force_cpu else '/CPU:0'):
                sigma_tf = self.compiled_functions['egarch_recursion'](
                    r_tf, mu_tf, omega_tf, alpha_tf, gamma_tf, beta_tf, chunk_size
                )
                
                # Convert to NumPy array
                return sigma_tf.numpy()
            
        except tf.errors.ResourceExhaustedError as e:
            # Handle out of memory error
            if self.verbose:
                logger.warning(f"GPU EGARCH out of memory: {str(e)}")
            
            # Clear GPU memory
            clear_gpu_memory()
            
            # Fall back to CPU implementation
            from enhanced_vol_models_v2 import egarch_recursion
            return egarch_recursion(r, mu, omega, alpha, gamma, beta)
            
        except Exception as e:
            # Handle other errors
            if self.verbose:
                logger.error(f"GPU EGARCH error: {str(e)}")
            
            # Fall back to CPU implementation
            from enhanced_vol_models_v2 import egarch_recursion
            return egarch_recursion(r, mu, omega, alpha, gamma, beta)
    
    @memory_safe_execution
    def t_pdf_diag(self, x, df, force_cpu=False):
        """
        GPU-accelerated T-distribution PDF calculation
        
        Args:
            x: Input vector
            df: Degrees of freedom
            force_cpu: Force CPU execution
            
        Returns:
            PDF value
        """
        if not self.initialized or self.tf is None or force_cpu:
            # Fall back to CPU implementation
            from enhanced_vol_models_v2 import t_pdf_safe_diag
            return t_pdf_safe_diag(x, df)
        
        try:
            # Convert to TensorFlow tensors
            x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
            df_tf = tf.constant(df, dtype=tf.float32)
            
            # Execute on appropriate device
            with tf.device(self.device if not force_cpu else '/CPU:0'):
                pdf = self.compiled_functions['t_pdf_diag'](x_tf, df_tf)
                
                # Convert to Python float
                return float(pdf.numpy())
            
        except Exception as e:
            if self.verbose:
                logger.error(f"GPU T-PDF error: {str(e)}")
            
            # Fall back to CPU implementation
            from enhanced_vol_models_v2 import t_pdf_safe_diag
            return t_pdf_safe_diag(x, df)
    
    @memory_safe_execution
    def compute_sigma_array(self, features, mu_d, vol_model, params_vol, dims_egarch=None, force_cpu=False):
        """
        Memory-optimized sigma array computation for all dimensions
        
        Args:
            features: Feature matrix [T, D]
            mu_d: Mean vector [D]
            vol_model: Volatility model ("EGARCH" or "GARCH")
            params_vol: Volatility model parameters
            dims_egarch: EGARCH dimensions
            force_cpu: Force CPU execution
            
        Returns:
            sigma_array: Matrix [T, D] with volatilities
        """
        if not self.initialized or self.tf is None or force_cpu:
            # Fall back to CPU implementation
            from enhanced_vol_models_v2 import compute_sigma_array
            return compute_sigma_array(features, mu_d, vol_model, params_vol, dims_egarch)
        
        try:
            # Validate inputs
            if features is None or mu_d is None or params_vol is None:
                raise ValueError("One of the inputs is None")
            
            T, D = features.shape
            sigma_array = np.ones((T, D))
            
            # Process in chunks if the sequence is very long
            chunk_size = None
            if self.enable_chunking and T > 1000:
                chunk_size = get_optimal_chunk_size(T)
            
            # EGARCH with GPU acceleration
            if vol_model == "EGARCH":
                (om, al, ga, be) = params_vol
                
                # Process each EGARCH dimension
                for d_ in dims_egarch if dims_egarch is not None else range(D):
                    if d_ < D:  # Safety check
                        r_d = features[:, d_]
                        # GPU-accelerated computation
                        s_ = self.egarch_recursion(r_d, mu_d[d_], om, al, ga, be)
                        sigma_array[:, d_] = s_
            
            # GARCH(1,1) fallback
            elif vol_model == "GARCH":
                from enhanced_vol_models_v2 import garch_recursion_1d
                (om_, al_, be_) = params_vol
                for d_ in dims_egarch if dims_egarch is not None else range(D):
                    if d_ < D:
                        r_d = features[:, d_]
                        s_ = garch_recursion_1d(r_d, mu_d[d_], om_, al_, be_)
                        sigma_array[:, d_] = s_
            
            return sigma_array
            
        except Exception as e:
            if self.verbose:
                logger.error(f"GPU sigma array computation error: {str(e)}")
            
            # Fall back to CPU implementation
            from enhanced_vol_models_v2 import compute_sigma_array
            return compute_sigma_array(features, mu_d, vol_model, params_vol, dims_egarch)
    
    @memory_safe_execution
    def forward_backward(self, features, pi, A, state_params_list, use_tdist=True, 
                        dims_egarch=None, times=None, external_factors=None, force_cpu=False):
        """
        Enhanced forward-backward algorithm with memory optimization
        
        Args:
            features: Feature matrix [T, D]
            pi: Initial state probabilities [K]
            A: Transition matrix [K, K]
            state_params_list: List of state parameters
            use_tdist: Use T-distribution
            dims_egarch: EGARCH dimensions
            times: Optional time series for time-varying transitions
            external_factors: Optional external factors
            force_cpu: Force CPU execution
            
        Returns:
            gamma, xi, scaling: Posterior probabilities, transition probabilities, scaling factors
        """
        # For stability, always use the CPU implementation
        # The forward-backward algorithm is complex and CPU execution is more reliable
        from enhanced_hmm_em_v2 import forward_backward
        return forward_backward(features, pi, A, state_params_list, use_tdist, 
                             dims_egarch, times, external_factors)
    
    @memory_safe_execution
    def weighted_forward_backward(self, features, feature_weights, pi, A, state_params_list, 
                                use_tdist=True, dims_egarch=None, times=None, 
                                external_factors=None, force_cpu=False):
        """
        Weighted forward-backward algorithm with memory optimization
        
        Args:
            features: Feature matrix [T, D]
            feature_weights: Feature weights [D]
            pi: Initial state probabilities [K]
            A: Transition matrix [K, K]
            state_params_list: List of state parameters
            use_tdist: Use T-distribution
            dims_egarch: EGARCH dimensions
            times: Optional time series
            external_factors: Optional external factors
            force_cpu: Force CPU execution
            
        Returns:
            gamma, xi, scaling: Posterior probabilities, transition probabilities, scaling factors
        """
        # For stability, use CPU implementation
        from enhanced_hmm_em_v2 import weighted_forward_backward
        return weighted_forward_backward(features, feature_weights, pi, A, state_params_list, 
                                     use_tdist, dims_egarch, times, external_factors)
    
    def batch_process_sigma(self, features_batch, mu_batch, vol_model, params_vol_batch, dims_egarch=None):
        """
        Batch processing for sigma arrays with memory optimization
        
        Args:
            features_batch: List of feature matrices
            mu_batch: List of mean vectors
            vol_model: Volatility model
            params_vol_batch: List of volatility parameters
            dims_egarch: EGARCH dimensions
            
        Returns:
            List of sigma arrays
        """
        if not self.initialized or self.tf is None or not self.use_gpu:
            # Sequential processing on CPU
            from enhanced_vol_models_v2 import compute_sigma_array
            return [compute_sigma_array(feat, mu, vol_model, params, dims_egarch) 
                   for feat, mu, params in zip(features_batch, mu_batch, params_vol_batch)]
        
        try:
            # Process in smaller batches for better GPU utilization
            batch_size = len(features_batch)
            results = []
            
            # Determine optimal sub-batch size based on GPU memory
            max_batch = min(16, batch_size)
            
            for i in range(0, batch_size, max_batch):
                sub_batch = features_batch[i:i+max_batch]
                sub_mu = mu_batch[i:i+max_batch]
                sub_params = params_vol_batch[i:i+max_batch]
                
                # Calculate sigma arrays
                sub_results = []
                for feat, mu, params in zip(sub_batch, sub_mu, sub_params):
                    sigma = self.compute_sigma_array(feat, mu, vol_model, params, dims_egarch)
                    sub_results.append(sigma)
                
                results.extend(sub_results)
                
                # Clear GPU memory between batches
                if i + max_batch < batch_size:
                    clear_gpu_memory()
            
            return results
            
        except Exception as e:
            logger.error(f"GPU batch processing error: {str(e)}")
            
            # Fall back to CPU implementation
            from enhanced_vol_models_v2 import compute_sigma_array
            return [compute_sigma_array(feat, mu, vol_model, params, dims_egarch) 
                   for feat, mu, params in zip(features_batch, mu_batch, params_vol_batch)]

def check_gpu_acceleration():
    """
    Check if GPU acceleration is available and run a simple benchmark
    
    Returns:
        dict: Information about GPU acceleration
    """
    result = {
        "gpu_available": HAS_GPU,
        "tensorflow_available": HAS_TF,
        "mixed_precision_available": MIXED_PRECISION,
        "gpu_models": [],
        "memory_info": gpu_memory_monitor(),
        "benchmark_results": {}
    }
    
    # Get GPU details
    if HAS_GPU and HAS_TF:
        gpus = tf.config.list_physical_devices('GPU')
        result["num_gpus"] = len(gpus)
        
        for i, gpu in enumerate(gpus):
            try:
                gpu_details = {
                    "name": gpu.name,
                    "device_type": gpu.device_type
                }
                
                # Try to get more detailed information on Linux
                if os.path.exists('/proc/driver/nvidia/gpus'):
                    try:
                        import subprocess
                        nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader']).decode('utf-8')
                        gpu_lines = nvidia_smi.strip().split('\n')
                        if i < len(gpu_lines):
                            name, total, free = gpu_lines[i].split(',')
                            gpu_details["memory_total"] = total.strip()
                            gpu_details["memory_free"] = free.strip()
                    except:
                        pass
                
                result["gpu_models"].append(gpu_details)
            except:
                result["gpu_models"].append({"name": f"GPU {i}", "device_type": "unknown"})
        
        # Simple benchmark
        try:
            # Create accelerator with conservative memory settings
            accelerator = GPUAccelerator(
                use_gpu=True,
                memory_fraction=0.5,  # Use only 50% of GPU memory for benchmark
                enable_mixed_precision=MIXED_PRECISION,
                verbose=False,
                enable_chunking=True
            )
            
            # Create test data
            np.random.seed(42)  # For reproducibility
            test_data = np.random.randn(1000)
            
            # CPU benchmark (original function)
            start_time = time.time()
            from enhanced_vol_models_v2 import egarch_recursion
            cpu_result = egarch_recursion(test_data, 0, 0.01, 0.1, 0, 0.85)
            cpu_time = time.time() - start_time
            
            # GPU benchmark (with memory management)
            start_time = time.time()
            gpu_result = accelerator.egarch_recursion(test_data, 0, 0.01, 0.1, 0, 0.85)
            gpu_time = time.time() - start_time
            
            # Compare results
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            
            result["benchmark_results"]["egarch"] = {
                "cpu_time": cpu_time,
                "gpu_time": gpu_time,
                "speedup": cpu_time / gpu_time if gpu_time > 0 else float('inf'),
                "max_diff": float(max_diff),
                "accuracy": "High" if max_diff < 1e-5 else "Medium" if max_diff < 1e-3 else "Low"
            }
            
        except Exception as e:
            result["benchmark_error"] = str(e)
    
    return result

def get_accelerator(use_gpu=True, memory_fraction=0.6, enable_mixed_precision=True, 
                  device=None, verbose=False, enable_chunking=True):
    """
    Create or return a global GPU accelerator instance
    
    Args:
        use_gpu: Use GPU if available
        memory_fraction: Fraction of GPU memory to use
        enable_mixed_precision: Enable mixed precision for faster computation
        device: Specific device to use
        verbose: Enable verbose logging
        enable_chunking: Enable processing data in chunks
        
    Returns:
        GPUAccelerator instance
    """
    global _global_accelerator
    
    # Create new accelerator if not exists or parameters changed
    if '_global_accelerator' not in globals() or _global_accelerator is None:
        _global_accelerator = GPUAccelerator(
            use_gpu=use_gpu,
            memory_fraction=memory_fraction,
            enable_mixed_precision=enable_mixed_precision,
            device=device,
            verbose=verbose,
            enable_chunking=enable_chunking
        )
    
    return _global_accelerator

def accelerate_hmm_functions(memory_fraction=0.6, enable_chunking=True):
    """
    Replace CPU implementations of computation-intensive functions with
    memory-optimized GPU accelerated versions
    
    Args:
        memory_fraction: Fraction of GPU memory to use
        enable_chunking: Enable processing data in chunks
        
    Returns:
        bool: Success status
    """
    try:
        # Create global accelerator with safe memory settings
        accelerator = get_accelerator(
            use_gpu=True,
            memory_fraction=memory_fraction,
            enable_chunking=enable_chunking,
            verbose=True
        )
        
        if not accelerator.use_gpu or not accelerator.initialized:
            logger.info("GPU acceleration not available. Original functions remain unchanged.")
            return False
        
        # Import modules for patching
        import enhanced_vol_models_v2
        import enhanced_hmm_em_v2
        
        # Store original functions
        original_functions = {
            'egarch_recursion': enhanced_vol_models_v2.egarch_recursion,
            't_pdf_diag': enhanced_vol_models_v2.t_pdf_diag_multidim,
            'compute_sigma_array': enhanced_vol_models_v2.compute_sigma_array
        }
        
        # Replace volatility model functions with GPU-accelerated versions
        enhanced_vol_models_v2.egarch_recursion = accelerator.egarch_recursion
        enhanced_vol_models_v2.t_pdf_diag_multidim = accelerator.t_pdf_diag
        enhanced_vol_models_v2.compute_sigma_array = accelerator.compute_sigma_array
        
        # Keep forward-backward algorithm on CPU for stability
        
        logger.info("EGARCH functions successfully replaced with GPU-accelerated versions.")
        logger.info("Forward-backward algorithm uses CPU for stability.")
        
        # Store original functions for potential restoration
        accelerator.original_functions = original_functions
        
        return True
        
    except Exception as e:
        logger.error(f"Error replacing functions: {str(e)}")
        return False

def restore_original_functions():
    """
    Restore original CPU implementations
    
    Returns:
        bool: Success status
    """
    try:
        accelerator = get_accelerator()
        
        if not hasattr(accelerator, 'original_functions'):
            logger.warning("No original functions found to restore")
            return False
        
        # Import modules for patching
        import enhanced_vol_models_v2
        import enhanced_hmm_em_v2
        
        # Restore original functions
        enhanced_vol_models_v2.egarch_recursion = accelerator.original_functions['egarch_recursion']
        enhanced_vol_models_v2.t_pdf_diag_multidim = accelerator.original_functions['t_pdf_diag']
        enhanced_vol_models_v2.compute_sigma_array = accelerator.original_functions['compute_sigma_array']
        
        logger.info("Original functions restored")
        return True
        
    except Exception as e:
        logger.error(f"Error restoring original functions: {str(e)}")
        return False

# Global accelerator instance
_global_accelerator = None

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check GPU acceleration
    gpu_info = check_gpu_acceleration()
    print("GPU acceleration:", "Available" if gpu_info["gpu_available"] else "Not available")
    
    if gpu_info["gpu_available"]:
        print(f"GPUs found: {gpu_info['num_gpus']}")
        for i, gpu in enumerate(gpu_info["gpu_models"]):
            print(f"  GPU {i}: {gpu.get('name', 'Unknown')}")
        
        if "memory_info" in gpu_info and gpu_info["memory_info"]:
            mem = gpu_info["memory_info"]
            print(f"Memory usage: {mem['used_mb']:.1f}MB / {mem['total_mb']:.1f}MB ({mem['usage_percent']:.1f}%)")
        
        if "benchmark_results" in gpu_info and "egarch" in gpu_info["benchmark_results"]:
            benchmark = gpu_info["benchmark_results"]["egarch"]
            print(f"\nEGARCH benchmark:")
            print(f"  CPU time: {benchmark['cpu_time']*1000:.2f} ms")
            print(f"  GPU time: {benchmark['gpu_time']*1000:.2f} ms")
            print(f"  Speedup: {benchmark['speedup']:.2f}x")
            print(f"  Accuracy: {benchmark['accuracy']}")
        
        # Replace CPU functions with GPU-accelerated versions
        success = accelerate_hmm_functions(memory_fraction=0.6, enable_chunking=True)
        
        if success:
            print("\nGPU acceleration enabled. HMM functions have been replaced.")
            
            # Run a simple performance test
            print("\nPerformance test with memory management:")
            accelerator = get_accelerator()
            
            # Test data
            test_size = 5000
            print(f"Testing with {test_size} data points...")
            np.random.seed(42)
            test_data = np.random.randn(test_size)
            
            # Test EGARCH with memory management
            start_time = time.time()
            result = accelerator.egarch_recursion(test_data, 0, 0.01, 0.1, 0, 0.85)
            test_time = time.time() - start_time
            
            print(f"EGARCH calculation time: {test_time*1000:.2f} ms")
            print(f"Memory usage after test: {gpu_memory_monitor()['usage_percent']:.1f}%")
            
            # Clean up
            clear_gpu_memory()
            print(f"Memory usage after cleanup: {gpu_memory_monitor()['usage_percent']:.1f}%")
    else:
        print("No GPU acceleration available. CPU will be used.")
