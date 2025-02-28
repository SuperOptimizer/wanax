import numpy as np
from numba import njit, prange

@njit(parallel=False,fastmath=True)
def min_max_normalize(data, new_min=0, new_max=1):
    old_min = np.min(data)
    old_max = np.max(data)
    if old_max == old_min:
        return np.full_like(data, new_min, dtype=np.float32)
    return ((data - old_min) * (new_max - new_min) / (old_max - old_min) + new_min).astype(np.float32)

@njit(parallel=False,fastmath=True)
def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - mean) / std).astype(np.float32)

@njit(parallel=True,fastmath=True)
def robust_normalize(data):
    # Extract non-zero values
    non_zero_values = np.empty(data.size, dtype=np.float32)
    non_zero_count = 0

    for i in prange(data.size):
        if data.flat[i] != 0:
            non_zero_values[non_zero_count] = data.flat[i]
            non_zero_count += 1

    # If no non-zero values, return all zeros
    if non_zero_count == 0:
        return np.zeros_like(data, dtype=np.float32)

    # Calculate statistics on non-zero values only
    q25 = np.percentile(non_zero_values[:non_zero_count], 25)
    q75 = np.percentile(non_zero_values[:non_zero_count], 75)
    iqr = q75 - q25

    if iqr == 0:
        # If IQR is zero, set non-zero values to 1.0
        result = np.zeros_like(data, dtype=np.float32)
        for i in prange(data.size):
            if data.flat[i] != 0:
                result.flat[i] = 1.0
        return result

    median = np.median(non_zero_values[:non_zero_count])

    # Normalize only non-zero values
    result = np.zeros_like(data, dtype=np.float32)
    for i in prange(data.size):
        if data.flat[i] != 0:
            result.flat[i] = (data.flat[i] - median) / iqr

    return result

@njit(parallel=False,fastmath=True)
def softmax_normalize(data):
    shifted = data - np.max(data)  # For numerical stability
    exp_data = np.exp(shifted)
    return (exp_data / np.sum(exp_data)).astype(np.float32)

@njit(parallel=False,fastmath=True)
def l1_normalize(data):
    norm = np.sum(np.abs(data))
    if norm == 0:
        return np.zeros_like(data, dtype=np.float32)
    return (data / norm).astype(np.float32)

@njit(parallel=False,fastmath=True)
def l2_normalize(data):
    norm = np.sqrt(np.sum(data * data))
    if norm == 0:
        return np.zeros_like(data, dtype=np.float32)
    return (data / norm).astype(np.float32)

@njit(parallel=False,fastmath=True)
def quantile_normalize(data, n_quantiles=1000):
    sorted_vals = np.sort(data.ravel())
    ranks = np.searchsorted(sorted_vals, data)
    quantiles = np.linspace(0, 1, n_quantiles)
    normalized = np.interp(ranks.astype(np.float32) / len(sorted_vals), quantiles, quantiles)
    return normalized.reshape(data.shape).astype(np.float32)

import numpy as np
from numba import njit, prange

@njit(parallel=False,fastmath=True)
def z_score_normalize_3d_u8(data, z_min=-3.0, z_max=3.0):
    """
    Perform z-score normalization on 3D uint8 data and return uint8 result.
    Maps z-scores between z_min and z_max to the 0-255 prange.

    Parameters:
    -----------
    data : ndarray (uint8)
        Input 3D array of uint8 values
    z_min : float, optional (default=-3.0)
        Minimum z-score to map to 0
    z_max : float, optional (default=3.0)
        Maximum z-score to map to 255

    Returns:
    --------
    result : ndarray (uint8)
        Z-score normalized array scaled to uint8 prange
    """
    # Handle empty input
    if data.size == 0:
        return np.zeros_like(data)

    # Calculate statistics using higher precision
    mean = np.mean(data, dtype=np.float32)
    std = np.std(data, dtype=np.float32)

    # Handle constant input
    if std < 1e-10:
        return np.zeros_like(data)

    # Pre-compute normalization parameters for efficiency
    scale = 255.0 / (z_max - z_min)
    inv_std = 1.0 / std
    scaling_factor = scale * inv_std
    bias = -mean * scaling_factor + scale * (-z_min)

    # Create output array
    result = np.zeros_like(data)

    # Apply normalization with pre-computed parameters
    for z in prange(data.shape[0]):
        for y in prange(data.shape[1]):
            for x in prange(data.shape[2]):
                # Optimized single-step calculation
                val = float(data[z, y, x]) * scaling_factor + bias

                # Clip and convert to uint8
                if val < 0:
                    result[z, y, x] = 0
                elif val > 255:
                    result[z, y, x] = 255
                else:
                    result[z, y, x] = np.uint8(round(val))

    return result

def adaptive_normalize(data, method='z_score'):
    normalizers = {
        'min_max': min_max_normalize,
        'z_score': z_score_normalize,
        'robust': robust_normalize,
        'softmax': softmax_normalize,
        'l1': l1_normalize,
        'l2': l2_normalize,
        'quantile': quantile_normalize
    }
    return normalizers.get(method, z_score_normalize)(data)