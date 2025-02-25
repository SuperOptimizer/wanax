import numpy as np
from numba import njit, prange

@njit(parallel=True)
def min_max_normalize(data, new_min=0, new_max=1):
    old_min = np.min(data)
    old_max = np.max(data)
    if old_max == old_min:
        return np.full_like(data, new_min, dtype=np.float32)
    return ((data - old_min) * (new_max - new_min) / (old_max - old_min) + new_min).astype(np.float32)

@njit(parallel=True)
def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - mean) / std).astype(np.float32)

@njit(parallel=True)
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

@njit(parallel=True)
def softmax_normalize(data):
    shifted = data - np.max(data)  # For numerical stability
    exp_data = np.exp(shifted)
    return (exp_data / np.sum(exp_data)).astype(np.float32)

@njit(parallel=True)
def l1_normalize(data):
    norm = np.sum(np.abs(data))
    if norm == 0:
        return np.zeros_like(data, dtype=np.float32)
    return (data / norm).astype(np.float32)

@njit(parallel=True)
def l2_normalize(data):
    norm = np.sqrt(np.sum(data * data))
    if norm == 0:
        return np.zeros_like(data, dtype=np.float32)
    return (data / norm).astype(np.float32)

@njit(parallel=True)
def quantile_normalize(data, n_quantiles=1000):
    sorted_vals = np.sort(data.ravel())
    ranks = np.searchsorted(sorted_vals, data)
    quantiles = np.linspace(0, 1, n_quantiles)
    normalized = np.interp(ranks.astype(np.float32) / len(sorted_vals), quantiles, quantiles)
    return normalized.reshape(data.shape).astype(np.float32)

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