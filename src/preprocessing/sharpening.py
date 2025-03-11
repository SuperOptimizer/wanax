import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def pad_array_edge_3d(data, pad_width):
    """Manual edge padding for 3D arrays that's compatible with Numba."""
    depth, height, width = data.shape
    padded_shape = (depth + 2 * pad_width, height + 2 * pad_width, width + 2 * pad_width)
    padded = np.zeros(padded_shape, dtype=data.dtype)

    # Copy the original data to the center of the padded array
    padded[pad_width:pad_width + depth, pad_width:pad_width + height, pad_width:pad_width + width] = data

    # Pad along z-axis (depth)
    for z in range(pad_width):
        # Top padding - replicate first slice
        padded[z, pad_width:pad_width + height, pad_width:pad_width + width] = data[0]
        # Bottom padding - replicate last slice
        padded[pad_width + depth + z, pad_width:pad_width + height, pad_width:pad_width + width] = data[-1]

    # Pad along y-axis (height)
    for y in range(pad_width):
        # Top rows
        padded[:, y, pad_width:pad_width + width] = padded[:, pad_width, pad_width:pad_width + width]
        # Bottom rows
        padded[:, pad_width + height + y, pad_width:pad_width + width] = padded[:, pad_width + height - 1,
                                                                         pad_width:pad_width + width]

    # Pad along x-axis (width)
    for x in range(pad_width):
        # Left columns
        padded[:, :, x] = padded[:, :, pad_width]
        # Right columns
        padded[:, :, pad_width + width + x] = padded[:, :, pad_width + width - 1]

    return padded


@njit(parallel=True, fastmath=True)
def unsharp_mask(data, radius=1, amount=1.0, threshold=0):
    """
    Apply unsharp mask to 3D data with proper edge handling using
    a custom padding function compatible with Numba.
    """
    # Pad the input array to handle edge values correctly
    padded_data = pad_array_edge_3d(data, radius)

    # Initialize result and blurred arrays with the same shape as padded_data
    p_depth, p_height, p_width = padded_data.shape
    result = np.zeros_like(padded_data, dtype=np.float32)
    blurred = np.zeros_like(padded_data, dtype=np.float32)

    kernel_size = 2 * radius + 1
    kernel_sum = kernel_size * kernel_size * kernel_size

    # Compute blur for the entire padded volume
    for z in prange(radius, p_depth - radius):
        for y in prange(radius, p_height - radius):
            for x in prange(radius, p_width - radius):
                sum_val = 0.0
                for kz in range(-radius, radius + 1):
                    for ky in range(-radius, radius + 1):
                        for kx in range(-radius, radius + 1):
                            sum_val += padded_data[z + kz, y + ky, x + kx]
                blurred[z, y, x] = sum_val / kernel_sum

    # Apply unsharp mask
    highpass = padded_data - blurred
    mask = np.abs(highpass) > threshold
    result = padded_data + amount * highpass * mask
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Crop back to original dimensions
    depth, height, width = data.shape
    return result[radius:radius + depth, radius:radius + height, radius:radius + width]


@njit(parallel=True, fastmath=True)
def laplacian_sharpen(data, strength=1.0):
    """
    Apply Laplacian sharpening to data. Works with 1D, 2D or 3D arrays.
    This function doesn't need padding as it only references immediate neighbors.
    """
    result = np.zeros_like(data, dtype=np.float32)

    if data.ndim == 1:
        size = data.shape[0]
        for i in prange(1, size - 1):
            lap = -2 * data[i] + data[i - 1] + data[i + 1]
            result[i] = data[i] + strength * lap
        # Handle edges separately
        result[0] = data[0] + strength * (-data[0] + data[1])
        result[size - 1] = data[size - 1] + strength * (-data[size - 1] + data[size - 2])

    elif data.ndim == 2:
        height, width = data.shape
        for y in prange(1, height - 1):
            for x in prange(1, width - 1):
                lap = -4 * data[y, x] + (
                        data[y - 1, x] + data[y + 1, x] +
                        data[y, x - 1] + data[y, x + 1]
                )
                result[y, x] = data[y, x] + strength * lap
        # Handle edges
        for y in range(height):
            result[y, 0] = data[y, 0] + strength * (-2 * data[y, 0] + 2 * data[y, 1])
            result[y, width - 1] = data[y, width - 1] + strength * (-2 * data[y, width - 1] + 2 * data[y, width - 2])
        for x in range(width):
            result[0, x] = data[0, x] + strength * (-2 * data[0, x] + 2 * data[1, x])
            result[height - 1, x] = data[height - 1, x] + strength * (
                        -2 * data[height - 1, x] + 2 * data[height - 2, x])

    else:  # 3D
        depth, height, width = data.shape
        for z in prange(1, depth - 1):
            for y in prange(1, height - 1):
                for x in prange(1, width - 1):
                    lap = -6 * data[z, y, x] + (
                            data[z - 1, y, x] + data[z + 1, y, x] +
                            data[z, y - 1, x] + data[z, y + 1, x] +
                            data[z, y, x - 1] + data[z, y, x + 1]
                    )
                    result[z, y, x] = data[z, y, x] + strength * lap

        # Handle edges - simplified approach for 3D
        # This could be expanded to handle all edge cases more precisely
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if z == 0 or z == depth - 1 or y == 0 or y == height - 1 or x == 0 or x == width - 1:
                        # For edges, use available neighbors only
                        neighbors = 0
                        neighbor_sum = 0

                        if z > 0:
                            neighbor_sum += data[z - 1, y, x]
                            neighbors += 1
                        if z < depth - 1:
                            neighbor_sum += data[z + 1, y, x]
                            neighbors += 1
                        if y > 0:
                            neighbor_sum += data[z, y - 1, x]
                            neighbors += 1
                        if y < height - 1:
                            neighbor_sum += data[z, y + 1, x]
                            neighbors += 1
                        if x > 0:
                            neighbor_sum += data[z, y, x - 1]
                            neighbors += 1
                        if x < width - 1:
                            neighbor_sum += data[z, y, x + 1]
                            neighbors += 1

                        if neighbors > 0:
                            lap = -neighbors * data[z, y, x] + neighbor_sum
                            result[z, y, x] = data[z, y, x] + strength * lap

    return np.clip(result, 0, 255).astype(np.uint8)