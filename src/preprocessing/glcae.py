import numpy as np
from numba import njit, prange, vectorize
import cv2

import numpy as np
from numba import njit, prange, vectorize
import cv2

@vectorize(['uint8(float32)'], nopython=True, fastmath=True)
def float_to_uint8(x):
    """Clip and convert float to uint8, branchless."""
    return np.uint8(max(0, min(255, round(x))))

@njit(fastmath=True)
def linear_stretching_3d(data):
    """Vectorized linear stretching for 3D uint8 data."""
    zero_mask = (data > 0).astype(np.uint8)

    non_zero = data[data > 0]
    if len(non_zero) == 0:
        return data.copy()

    data_min = np.min(non_zero)
    data_max = np.max(non_zero)

    if data_max == data_min:
        return data.copy()

    scale = np.float32(255.0) / np.float32(data_max - data_min)
    temp = (data.astype(np.float32) - np.float32(data_min)) * scale

    result = float_to_uint8(temp)
    result = result * zero_mask

    return result

@njit(parallel=True, fastmath=True)
def compute_histogram_3d(data):
    """Compute histogram for 3D uint8 data, ignoring zeros."""
    hist = np.zeros(256, dtype=np.float32)
    flat_data = data.ravel()
    non_zero = flat_data[flat_data > 0]
    total_elements = len(non_zero)

    for i in prange(total_elements):
        hist[non_zero[i]] += np.float32(1.0)

    if total_elements > 0:
        hist = hist / np.float32(total_elements)

    return hist

@njit(fastmath=True)
def compute_mapping(hist):
    """Compute mapping function from histogram."""
    cum_hist = np.cumsum(hist)
    mapping = np.round(np.float32(255.0) * cum_hist + np.float32(0.5)).astype(np.uint8)
    return mapping

@njit(parallel=False, fastmath=True)
def apply_mapping_3d(data, mapping):
    """Apply mapping to 3D uint8 data with explicit iteration."""
    result = np.zeros_like(data)
    depth, height, width = data.shape

    for z in prange(depth):
        for y in prange(height):
            for x in prange(width):
                if data[z, y, x] > 0:
                    result[z, y, x] = mapping[data[z, y, x]]

    return result

@njit(fastmath=True)
def compute_tone_distortion(mapping, hist):
    """Compute tone distortion from mapping and histogram."""
    max_distortion = np.int32(0)
    non_zero_indices = np.where(hist > np.float32(0.0))[0]

    for idx_i in prange(len(non_zero_indices)):
        i = non_zero_indices[idx_i]
        mapping_i = mapping[i]

        for idx_j in prange(idx_i):
            j = non_zero_indices[idx_j]
            if j >= i:
                continue

            is_equal = np.int32(mapping_i == mapping[j])
            distortion = np.int32(i - j) * is_equal
            max_distortion = max(max_distortion, distortion)

    return max_distortion

@njit(parallel=False, fastmath=True)
def compute_laplacian_3d(data):
    """Compute Laplacian for 3D uint8 data."""
    depth, height, width = data.shape
    laplacian = np.zeros_like(data, dtype=np.float32)
    zero_mask = (data > 0)

    for z in prange(1, depth - 1):
        slice_current = data[z].astype(np.float32)
        slice_before = data[z-1].astype(np.float32)
        slice_after = data[z+1].astype(np.float32)

        for y in prange(1, height - 1):
            for x in prange(1, width - 1):
                if not zero_mask[z, y, x]:
                    continue

                center_val = slice_current[y, x]
                neighbors_sum = (
                        slice_before[y, x] +
                        slice_after[y, x] +
                        slice_current[y-1, x] +
                        slice_current[y+1, x] +
                        slice_current[y, x-1] +
                        slice_current[y, x+1]
                )

                laplacian[z, y, x] = abs(np.float32(-6.0) * center_val + neighbors_sum)

    return laplacian

@njit(fastmath=True)
def compute_weights_3d(data, sigma=0.2):
    """Compute weights for fusion blending."""
    contrast = compute_laplacian_3d(data)
    max_contrast = np.max(contrast)

    if max_contrast > 0:
        contrast = contrast / max_contrast

    inv_sigma_sq_2 = np.float32(1.0) / (np.float32(2.0) * np.float32(sigma) * np.float32(sigma))

    norm_data = data.astype(np.float32) / np.float32(255.0)
    diff = norm_data - np.float32(0.5)
    brightness = np.exp(-(diff * diff) * inv_sigma_sq_2)

    zero_mask = (data > 0)
    brightness = brightness * zero_mask

    weights = np.minimum(contrast, brightness)

    return weights

@njit(fastmath=True)
def fusion_blend_3d(global_enhanced, local_enhanced, w_global_norm):
    """Blend global and local enhanced images based on weights."""
    data_mask = ((global_enhanced > 0) | (local_enhanced > 0)).astype(np.float32)

    w_local_norm = np.float32(1.0) - w_global_norm

    global_contrib = global_enhanced.astype(np.float32) * w_global_norm
    local_contrib = local_enhanced.astype(np.float32) * w_local_norm

    blended = global_contrib + local_contrib

    result = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    result = result * data_mask.astype(np.uint8)

    return result

@njit(parallel=False, fastmath=True)
def optimize_lambda(data, lambda_range=None):
    """Find optimal lambda for 3D uint8 data."""
    if lambda_range is None:
        lambda_range = np.linspace(0.1, 10.0, 20).astype(np.float32)

    hist = compute_histogram_3d(data)
    uniform_hist = np.ones(256, dtype=np.float32) / np.float32(256.0)

    min_distortion = np.int32(2147483647)  # Max int32 value
    optimal_lambda = np.float32(1.0)

    modified_hist = np.zeros(256, dtype=np.float32)

    for i in prange(len(lambda_range)):
        lam = lambda_range[i]
        factor1 = np.float32(1.0) / (np.float32(1.0) + lam)
        factor2 = lam / (np.float32(1.0) + lam)

        for j in prange(256):
            modified_hist[j] = factor1 * hist[j] + factor2 * uniform_hist[j]

        mapping = compute_mapping(modified_hist)
        distortion = compute_tone_distortion(mapping, hist)

        if distortion < min_distortion:
            min_distortion = distortion
            optimal_lambda = lam

    return optimal_lambda

@njit(fastmath=True)
def global_contrast_enhancement(data, lambda_param):
    """Global contrast enhancement with vectorized operations."""
    hist = compute_histogram_3d(data)
    uniform_hist = np.ones(256, dtype=np.float32) / np.float32(256.0)

    factor1 = np.float32(1.0) / (np.float32(1.0) + lambda_param)
    factor2 = lambda_param / (np.float32(1.0) + lambda_param)

    modified_hist = factor1 * hist + factor2 * uniform_hist

    mapping = compute_mapping(modified_hist)
    return apply_mapping_3d(data, mapping)

@njit(fastmath=True)
def clip_histogram(hist, clip_limit):
    """Clip histogram and redistribute excess."""
    # Calculate clipping threshold
    non_zero_count = np.sum(hist)
    if non_zero_count == 0:
        return hist

    avg_pixels_per_bin = np.float32(non_zero_count) / np.float32(256.0)
    clip_threshold = np.int32(clip_limit * avg_pixels_per_bin)

    # Count clipped pixels
    clipped = np.int32(0)
    for i in prange(256):
        if hist[i] > clip_threshold:
            clipped += hist[i] - clip_threshold
            hist[i] = clip_threshold

    # Redistribute clipped pixels
    redistrib_per_bin = np.float32(clipped) / np.float32(256.0)
    for i in prange(256):
        hist[i] += np.int32(redistrib_per_bin)

    return hist

@njit(fastmath=True)
def create_mapping_from_hist(hist):
    """Create mapping function from histogram."""
    mapping = np.zeros(256, dtype=np.uint8)
    cum_sum = np.int32(0)
    non_zero_count = np.sum(hist)

    if non_zero_count > 0:
        for i in prange(256):
            cum_sum += hist[i]
            # Normalized CDF * 255
            value = np.float32(cum_sum) * np.float32(255.0) / np.float32(non_zero_count)
            mapping[i] = np.uint8(min(255, max(0, round(value))))
    else:
        # If no non-zero pixels, use identity mapping
        for i in prange(256):
            mapping[i] = np.uint8(i)

    return mapping

@njit(parallel=True, fastmath=True)
def local_contrast_enhancement(data, clip_limit=2.0):
    """Simplified tile-based CLAHE implementation."""
    depth, height, width = data.shape
    result = np.zeros_like(data)

    # Use fewer, larger tiles to reduce boundary artifacts
    grid_size = 4  # Number of tiles in each dimension

    # Calculate tile size
    tile_height = height // grid_size
    if tile_height == 0:
        tile_height = 1
    tile_width = width // grid_size
    if tile_width == 0:
        tile_width = 1

    for z in prange(depth):
        # Skip empty slices
        if not np.any(data[z] > 0):
            continue

        slice_data = data[z]
        slice_result = np.zeros_like(slice_data)

        # Process each tile
        for ty in prange(grid_size):
            y_start = ty * tile_height
            y_end = min((ty + 1) * tile_height, height)

            for tx in prange(grid_size):
                x_start = tx * tile_width
                x_end = min((tx + 1) * tile_width, width)

                # Skip empty tiles
                if y_end <= y_start or x_end <= x_start:
                    continue

                # Extract tile
                tile = slice_data[y_start:y_end, x_start:x_end]

                # Skip if tile is all zeros
                if np.max(tile) == 0:
                    continue

                # Compute histogram of non-zero values
                hist = np.zeros(256, dtype=np.int32)
                for y in prange(tile.shape[0]):
                    for x in prange(tile.shape[1]):
                        if tile[y, x] > 0:
                            hist[tile[y, x]] += 1

                # Clip histogram and create mapping
                hist = clip_histogram(hist, clip_limit)
                mapping = create_mapping_from_hist(hist)

                # Apply mapping to the tile
                for y in prange(y_start, y_end):
                    for x in prange(x_start, x_end):
                        if slice_data[y, x] > 0:
                            slice_result[y, x] = mapping[slice_data[y, x]]

        result[z] = slice_result

    return result

def enhance_contrast_3d(data, lambda_param=None, clip_limit=2.0):
    """
    Global and local contrast enhancement for 3D uint8 data.

    Parameters:
    -----------
    data : ndarray (uint8)
        Input 3D array of uint8 values (0-255)
    lambda_param : float, optional
        Lambda parameter for global enhancement. If None, will be optimized.
    clip_limit : float, optional (default=2.0)
        Clip limit for local contrast enhancement

    Returns:
    --------
    result : ndarray (uint8)
        Enhanced 3D volume with same shape as input
    """
    if data.ndim != 3:
        raise ValueError("Input must be 3D data")

    if data.dtype != np.uint8:
        data_u8 = linear_stretching_3d(data)
    else:
        data_u8 = data.copy()

    if lambda_param is None:
        lambda_param = np.float32(optimize_lambda(data_u8))
    else:
        lambda_param = np.float32(lambda_param)

    global_enhanced = global_contrast_enhancement(data_u8, lambda_param)
    local_enhanced = local_contrast_enhancement(data_u8, clip_limit)

    w_global = compute_weights_3d(global_enhanced)
    w_local = compute_weights_3d(local_enhanced)

    sum_weights = w_global + w_local

    w_global_norm = np.zeros_like(w_global)
    mask = (sum_weights > 0)
    w_global_norm[mask] = w_global[mask] / sum_weights[mask]
    w_global_norm[~mask] = np.float32(0.5)

    return fusion_blend_3d(global_enhanced, local_enhanced, w_global_norm)
