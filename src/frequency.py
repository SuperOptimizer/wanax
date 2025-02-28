import numpy as np
from numba import njit, prange, vectorize

@vectorize(['float32(uint8)'], nopython=True, fastmath=True)
def uint8_to_float32(x):
    """Convert uint8 to float32 efficiently."""
    return np.float32(x)

@vectorize(['uint8(float32)'], nopython=True, fastmath=True)
def float32_to_uint8(x):
    """Convert float32 to uint8 with proper clipping."""
    return np.uint8(max(0, min(255, round(x))))

@njit(parallel=False, fastmath=True)
def haar_wavelet_transform_3d_optimized(data):
    """
    Optimized 3D Haar wavelet transform with vectorization and parallelization.

    Parameters:
    -----------
    data : ndarray (float32)
        Input 3D array of float32 values

    Returns:
    --------
    result : ndarray (float32)
        Wavelet transformed 3D volume with same shape as input
    """
    if data.size <= 1:
        return data.copy()

    shape = data.shape
    result = np.zeros_like(data, dtype=np.float32)

    # Use sqrt(2) as a constant for better performance
    sqrt2 = np.float32(np.sqrt(2.0))
    inv_sqrt2 = np.float32(1.0 / sqrt2)

    # Transform along z-axis (axis 0)
    for i in prange(0, shape[0] // 2):
        # Compute averages and differences for even-odd pairs
        result[i] = (data[2*i] + data[2*i+1]) * inv_sqrt2
        result[shape[0]//2 + i] = (data[2*i] - data[2*i+1]) * inv_sqrt2

    # Handle extra element if odd length
    if shape[0] % 2 == 1:
        result[shape[0]//2] = data[-1]

    # Transform along y-axis (axis 1)
    temp = result.copy()
    for i in prange(shape[0]):
        for j in range(0, shape[1] // 2):
            # Compute averages and differences for even-odd pairs
            result[i, j] = (temp[i, 2*j] + temp[i, 2*j+1]) * inv_sqrt2
            result[i, shape[1]//2 + j] = (temp[i, 2*j] - temp[i, 2*j+1]) * inv_sqrt2

        # Handle extra element if odd length
        if shape[1] % 2 == 1:
            result[i, shape[1]//2] = temp[i, -1]

    # Transform along x-axis (axis 2)
    temp = result.copy()
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(0, shape[2] // 2):
                # Compute averages and differences for even-odd pairs
                result[i, j, k] = (temp[i, j, 2*k] + temp[i, j, 2*k+1]) * inv_sqrt2
                result[i, j, shape[2]//2 + k] = (temp[i, j, 2*k] - temp[i, j, 2*k+1]) * inv_sqrt2

            # Handle extra element if odd length
            if shape[2] % 2 == 1:
                result[i, j, shape[2]//2] = temp[i, j, -1]

    return result

@njit(parallel=False, fastmath=True)
def inverse_haar_wavelet_3d_optimized(data):
    """
    Optimized 3D inverse Haar wavelet transform with vectorization and parallelization.

    Parameters:
    -----------
    data : ndarray (float32)
        Input 3D array of wavelet coefficients

    Returns:
    --------
    result : ndarray (float32)
        Reconstructed 3D volume with same shape as input
    """
    if data.size <= 1:
        return data.copy()

    shape = data.shape
    result = np.zeros_like(data, dtype=np.float32)

    # Use sqrt(2) as a constant for better performance
    sqrt2 = np.float32(np.sqrt(2.0))
    inv_sqrt2 = np.float32(1.0 / sqrt2)

    # Inverse transform along x-axis (axis 2)
    half_x = shape[2] // 2
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(half_x):
                avg = data[i, j, k]
                diff = data[i, j, k + half_x]

                result[i, j, 2*k] = (avg + diff) * inv_sqrt2
                result[i, j, 2*k+1] = (avg - diff) * inv_sqrt2

            # Handle extra element if odd length
            if shape[2] % 2 == 1:
                result[i, j, -1] = data[i, j, half_x]

    # Inverse transform along y-axis (axis 1)
    temp = result.copy()
    half_y = shape[1] // 2
    for i in prange(shape[0]):
        for j in range(half_y):
            for k in range(shape[2]):
                avg = temp[i, j, k]
                diff = temp[i, j + half_y, k]

                result[i, 2*j, k] = (avg + diff) * inv_sqrt2
                result[i, 2*j+1, k] = (avg - diff) * inv_sqrt2

        # Handle extra element if odd length
        if shape[1] % 2 == 1:
            for k in range(shape[2]):
                result[i, -1, k] = temp[i, half_y, k]

    # Inverse transform along z-axis (axis 0)
    temp = result.copy()
    half_z = shape[0] // 2
    for i in prange(half_z):
        for j in range(shape[1]):
            for k in range(shape[2]):
                avg = temp[i, j, k]
                diff = temp[i + half_z, j, k]

                result[2*i, j, k] = (avg + diff) * inv_sqrt2
                result[2*i+1, j, k] = (avg - diff) * inv_sqrt2

    # Handle extra element if odd length
    if shape[0] % 2 == 1:
        for j in range(shape[1]):
            for k in range(shape[2]):
                result[-1, j, k] = temp[half_z, j, k]

    return result

@njit(fastmath=True)
def find_min_max_non_zero(data):
    """
    Find min and max values of non-zero elements without using boolean indexing.

    Parameters:
    -----------
    data : ndarray
        Input array

    Returns:
    --------
    min_val, max_val : float, float
        Minimum and maximum values of non-zero elements
    """
    min_val = np.float32(np.inf)
    max_val = np.float32(0.0)

    # Flatten for faster iteration
    flat_data = data.ravel()

    # Find min/max of non-zero elements
    for i in range(len(flat_data)):
        val = flat_data[i]
        if val > 0:
            min_val = min(min_val, val)
            max_val = max(max_val, val)

    # Handle case where no non-zero elements found
    if min_val == np.inf:
        min_val = np.float32(0.0)
        max_val = np.float32(0.0)

    return min_val, max_val

@njit(fastmath=True)
def wavelet_denoise_3d_u8(data, threshold=0.1):
    """
    Optimized 3D wavelet denoising for uint8 data with optimized processing.

    Parameters:
    -----------
    data : ndarray (uint8)
        Input 3D array of uint8 values
    threshold : float, optional
        Threshold value for coefficient thresholding (default=0.1)

    Returns:
    --------
    denoised : ndarray (uint8)
        Denoised 3D volume with same shape as input
    """
    # Skip processing if input is empty or all zeros
    if data.size == 0 or np.max(data) == 0:
        return data.copy()

    # Create a mask of zeros for later restoration
    zero_mask = (data == 0)
    non_zero_mask = ~zero_mask

    # Convert to float32 for processing
    data_f32 = uint8_to_float32(data)

    # Find min/max of non-zero elements without boolean indexing
    min_val, max_val = find_min_max_non_zero(data_f32)

    # Get shape dimensions for manual indexing
    depth, height, width = data.shape
    hw = height * width

    # Normalize data for better wavelet performance
    data_norm = np.zeros_like(data_f32)
    if max_val > min_val:
        norm_factor = max_val - min_val
        # Normalize non-zero values manually
        for i in range(data.size):
            # Manual 3D index calculation
            z = i // hw
            y = (i % hw) // width
            x = i % width

            if data[z, y, x] > 0:  # Check original uint8 data
                data_norm[z, y, x] = (data_f32[z, y, x] - min_val) / norm_factor
    else:
        # Handle edge case where all non-zero values are identical
        for i in range(data.size):
            # Manual 3D index calculation
            z = i // hw
            y = (i % hw) // width
            x = i % width

            if data[z, y, x] > 0:  # Check original uint8 data
                data_norm[z, y, x] = np.float32(1.0)
        norm_factor = np.float32(1.0)

    # Forward wavelet transform
    coeffs = haar_wavelet_transform_3d_optimized(data_norm)

    # Apply thresholding to coefficients using explicit loop approach
    max_coeff = np.float32(0.0)
    non_zero_count = 0

    # Find maximum coefficient (absolute value) among non-zeros
    flat_coeffs = coeffs.ravel()
    for i in range(len(flat_coeffs)):
        coeff_val = abs(flat_coeffs[i])
        if coeff_val > 1e-10:
            max_coeff = max(max_coeff, coeff_val)
            non_zero_count += 1

    # Apply thresholding if we have non-zero coefficients
    if non_zero_count > 0:
        threshold_value = threshold * max_coeff
        thresholded_coeffs = np.zeros_like(coeffs)

        flat_thresholded = thresholded_coeffs.ravel()
        for i in range(len(flat_coeffs)):
            if abs(flat_coeffs[i]) >= threshold_value:
                flat_thresholded[i] = flat_coeffs[i]
    else:
        thresholded_coeffs = coeffs.copy()

    # Inverse wavelet transform
    denoised_norm = inverse_haar_wavelet_3d_optimized(thresholded_coeffs)

    # Restore original range
    denoised = np.zeros_like(data_f32)
    if max_val > min_val:
        for i in range(data.size):
            # Manual 3D index calculation
            z = i // hw
            y = (i % hw) // width
            x = i % width

            if non_zero_mask[z, y, x]:  # We can use non_zero_mask for checking, just not for indexing
                denoised[z, y, x] = denoised_norm[z, y, x] * norm_factor + min_val
    else:
        for i in range(data.size):
            # Manual 3D index calculation
            z = i // hw
            y = (i % hw) // width
            x = i % width

            if non_zero_mask[z, y, x]:
                denoised[z, y, x] = min_val

    # Convert back to uint8
    return float32_to_uint8(denoised)

@njit(parallel=True, fastmath=True)
def process_volume_blocks(data, block_size=(256,256,256), threshold=0.1):
    """
    Process large volumes in blocks to improve cache efficiency and memory usage.

    Parameters:
    -----------
    data : ndarray (uint8)
        Input 3D array of uint8 values
    block_size : tuple of int
        Size of blocks to process (z, y, x)
    threshold : float, optional
        Threshold value for coefficient thresholding (default=0.1)

    Returns:
    --------
    result : ndarray (uint8)
        Processed 3D volume with same shape as input
    """
    depth, height, width = data.shape
    result = np.zeros_like(data)

    # Calculate number of blocks in each dimension
    z_blocks = (depth + block_size[0] - 1) // block_size[0]
    y_blocks = (height + block_size[1] - 1) // block_size[1]
    x_blocks = (width + block_size[2] - 1) // block_size[2]

    # Process each block in parallel
    for iz in prange(z_blocks):
        z_start = iz * block_size[0]
        z_end = min((iz + 1) * block_size[0], depth)

        for iy in prange(y_blocks):
            y_start = iy * block_size[1]
            y_end = min((iy + 1) * block_size[1], height)

            for ix in prange(x_blocks):
                x_start = ix * block_size[2]
                x_end = min((ix + 1) * block_size[2], width)

                # Extract block - create a copy to ensure we have a contiguous array
                block = np.empty((z_end - z_start, y_end - y_start, x_end - x_start), dtype=np.uint8)
                for z in prange(z_end - z_start):
                    for y in prange(y_end - y_start):
                        for x in prange(x_end - x_start):
                            block[z, y, x] = data[z_start + z, y_start + y, x_start + x]

                # Skip empty blocks
                if np.max(block) == 0:
                    continue

                # Process block
                processed_block = wavelet_denoise_3d_u8(block, threshold)

                # Write back to result
                for z in prange(z_end - z_start):
                    for y in prange(y_end - y_start):
                        for x in prange(x_end - x_start):
                            result[z_start + z, y_start + y, x_start + x] = processed_block[z, y, x]

    return result

def wavelet_denoise_3d_optimized(data):
    return process_volume_blocks(data)