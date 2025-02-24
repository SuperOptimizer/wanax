import numpy as np
from numba import njit, prange
import cv2

@njit
def linear_stretching(data, l=256):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return data.astype(np.uint8)
    return ((l - 1) * (data - data_min) / (data_max - data_min)).astype(np.uint8)

@njit(parallel=True)
def compute_histogram(data, l=256):
    hist = np.zeros(l, dtype=np.float32)
    total_elements = 0

    if data.ndim == 1:
        for i in prange(data.shape[0]):
            if data[i] == 0: continue
            hist[data[i]] += 1
            total_elements += 1
    elif data.ndim == 2:
        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                if data[y, x] == 0: continue
                hist[data[y, x]] += 1
                total_elements += 1
    else:  # 3D
        for z in prange(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    if data[z, y, x] == 0: continue
                    hist[data[z, y, x]] += 1
                    total_elements += 1

    return hist / total_elements if total_elements > 0 else hist

@njit
def compute_mapping(hist, l=256):
    cum_sum = 0
    t = np.zeros(l, dtype=np.int32)
    for i in range(l):
        cum_sum += hist[i]
        t[i] = int((l - 1) * cum_sum + 0.5)
    return t

@njit(parallel=True)
def apply_mapping(data, mapping):
    result = np.zeros_like(data, dtype=np.uint8)

    if data.ndim == 1:
        for i in prange(data.shape[0]):
            if data[i] == 0: continue
            result[i] = mapping[data[i]]
    elif data.ndim == 2:
        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                if data[y, x] == 0: continue
                result[y, x] = mapping[data[y, x]]
    else:  # 3D
        for z in prange(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    if data[z, y, x] == 0: continue
                    result[z, y, x] = mapping[data[z, y, x]]
    return result

@njit
def compute_tone_distortion(mapping, hist, l=256):
    max_distortion = 0
    for i in range(l):
        if hist[i] == 0: continue
        for j in range(i):
            if hist[j] == 0: continue
            if mapping[i] == mapping[j]:
                max_distortion = max(max_distortion, i - j)
    return max_distortion

def optimize_lambda(data_u8, lambda_range=np.linspace(0.1, 10, 20)):
    hist = compute_histogram(data_u8)
    uniform_hist = np.ones(256) / 256
    min_distortion = float('inf')
    optimal_lambda = 1.0

    for lam in lambda_range:
        modified_hist = (1.0 / (1.0 + lam)) * hist + (lam / (1.0 + lam)) * uniform_hist
        mapping = compute_mapping(modified_hist)
        distortion = compute_tone_distortion(mapping, hist)
        if distortion < min_distortion:
            min_distortion = distortion
            optimal_lambda = lam
    return optimal_lambda

def global_contrast_adaptive_enhancement(data, lambda_param=None):
    data_u8 = linear_stretching(data) if data.dtype != np.uint8 else data.copy()
    lambda_param = optimize_lambda(data_u8) if lambda_param is None else lambda_param
    hist = compute_histogram(data_u8)
    uniform_hist = np.ones(256) / 256
    modified_hist = (1.0 / (1.0 + lambda_param)) * hist + (lambda_param / (1.0 + lambda_param)) * uniform_hist
    mapping = compute_mapping(modified_hist)
    return apply_mapping(data_u8, mapping)

def local_contrast_adaptive_enhancement(data, clip_limit=2.0):
    data_u8 = linear_stretching(data) if data.dtype != np.uint8 else data.copy()
    result = np.zeros_like(data_u8, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    if data.ndim == 2:
        return clahe.apply(data_u8)
    else:  # 3D
        for z in range(data_u8.shape[0]):
            result[z] = clahe.apply(data_u8[z])
        return result

@njit(parallel=True)
def compute_laplacian(data):
    laplacian = np.zeros_like(data, dtype=np.float32)

    if data.ndim == 2:
        height, width = data.shape
        for y in prange(1, height - 1):
            for x in range(1, width - 1):
                if data[y, x] == 0: continue
                center_val = data[y, x]
                neighbors_sum = (
                        data[y - 1, x] + data[y + 1, x] +
                        data[y, x - 1] + data[y, x + 1]
                )
                laplacian[y, x] = abs(-4.0 * center_val + neighbors_sum)
    else:  # 3D
        depth, height, width = data.shape
        for z in prange(1, depth - 1):
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if data[z, y, x] == 0: continue
                    center_val = data[z, y, x]
                    neighbors_sum = (
                            data[z - 1, y, x] + data[z + 1, y, x] +
                            data[z, y - 1, x] + data[z, y + 1, x] +
                            data[z, y, x - 1] + data[z, y, x + 1]
                    )
                    laplacian[z, y, x] = abs(-6.0 * center_val + neighbors_sum)
    return laplacian

def compute_weights(data, sigma=0.2):
    contrast = compute_laplacian(data)
    max_contrast = np.max(contrast)
    if max_contrast > 0:
        contrast = contrast / max_contrast
    data_norm = data / 255.0
    brightness = np.exp(-((data_norm - 0.5) ** 2) / (2 * sigma ** 2))
    return np.minimum(contrast, brightness)

def fusion_blend(global_enhanced, local_enhanced, w_global_norm):
    w_local_norm = 1.0 - w_global_norm
    result = global_enhanced * w_global_norm + local_enhanced * w_local_norm
    return np.clip(result, 0, 255).astype(np.uint8)

def global_and_local_contrast_enhancement(data, lambda_param=None, clip_limit=2.0):
    if data.ndim == 1:
        return global_contrast_adaptive_enhancement(data, lambda_param)

    data_u8 = linear_stretching(data) if data.dtype != np.uint8 else data.copy()
    global_enhanced = global_contrast_adaptive_enhancement(data_u8, lambda_param)
    local_enhanced = local_contrast_adaptive_enhancement(data_u8, clip_limit)
    w_global = compute_weights(global_enhanced)
    w_local = compute_weights(local_enhanced)
    sum_weights = w_global + w_local
    sum_weights[sum_weights == 0] = 1
    w_global_norm = w_global / sum_weights
    return fusion_blend(global_enhanced, local_enhanced, w_global_norm)