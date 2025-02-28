import numpy as np
from numba import njit, prange

@njit(fastmath=True, parallel=False)
def compute_histogram_u8(data):
    """Compute histogram for uint8 data efficiently with parallelization."""
    hist = np.zeros(256, dtype=np.uint32)
    flat_data = data.ravel()

    for i in prange(len(flat_data)):
        hist[flat_data[i]] += 1

    return hist

@njit(fastmath=True, parallel=False)
def create_equalization_lut(histogram, preserve_zeros=True, output_min=0, output_max=255):
    """Create a lookup table for histogram equalization."""
    # Calculate total pixels (excluding zeros if preserve_zeros is True)
    start_idx = 1 if preserve_zeros else 0
    total_pixels = np.uint64(0)
    for i in prange(start_idx, 256):
        total_pixels += np.uint64(histogram[i])

    # Create lookup table
    lut = np.zeros(256, dtype=np.uint8)

    # Handle special case: preserve zeros
    lut[0] = np.uint8(0) if preserve_zeros else np.uint8(output_min)

    # If no valid pixels, return simple mapping
    if total_pixels == 0:
        for i in prange(1, 256):
            lut[i] = np.uint8(max(min(i, output_max), output_min))
        return lut

    # Compute CDF and find first non-zero bin
    cdf = np.zeros(256, dtype=np.float32)
    min_bin = 256

    for i in prange(start_idx, 256):
        if histogram[i] > 0 and i < min_bin:
            min_bin = i

    # If no non-zero bins found, return LUT filled with output_min
    if min_bin >= 256:
        for i in prange(1, 256):
            lut[i] = np.uint8(output_min)
        return lut

    # Compute CDF
    cdf[min_bin] = np.float32(histogram[min_bin])
    for i in prange(min_bin + 1, 256):
        cdf[i] = cdf[i-1] + np.float32(histogram[i])

    # Constants for normalization
    cdf_min = cdf[min_bin]
    scale = np.float32(output_max - output_min)

    # Set values below min_bin to output_min
    for i in prange(1, min_bin):
        lut[i] = np.uint8(output_min)

    # Apply equalization formula for values >= min_bin
    for i in prange(min_bin, 256):
        normalized_cdf = (cdf[i] - cdf_min) / np.float32(total_pixels)
        value = int(round(normalized_cdf * scale + output_min))
        lut[i] = np.uint8(max(output_min, min(output_max, value)))

    return lut

@njit(parallel=False, fastmath=True)
def apply_lut_3d(data, lut):
    """Apply lookup table to 3D data with parallelization."""
    depth, height, width = data.shape
    result = np.zeros((depth, height, width), dtype=np.uint8)

    for z in prange(depth):
        for y in prange(height):
            for x in prange(width):
                result[z, y, x] = lut[data[z, y, x]]

    return result

@njit(fastmath=True)
def histogram_equalization_3d_u8(data, preserve_zeros=True, output_min=0, output_max=255):
    """Histogram equalization for 3D uint8 data."""
    # Compute histogram
    histogram = compute_histogram_u8(data)

    # Create lookup table
    lut = create_equalization_lut(histogram, preserve_zeros, output_min, output_max)

    # Apply lookup table to 3D data
    return apply_lut_3d(data, lut)

@njit(parallel=False, fastmath=True)
def process_volume_blocks_with_global_lut(data, lut, block_size):
    """Apply a global LUT to blocks of a 3D volume for better cache efficiency."""
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

                # Process each voxel in the block
                for z in prange(z_start, z_end):
                    for y in prange(y_start, y_end):
                        for x in prange(x_start, x_end):
                            result[z, y, x] = lut[data[z, y, x]]

    return result


@njit(fastmath=True)
def histogram_equalize_3d_u8(data):
    block_size = (256, 256, 256)
    histogram = compute_histogram_u8(data)
    lut = create_equalization_lut(histogram, True, 0, 255)
    return process_volume_blocks_with_global_lut(data, lut, block_size)