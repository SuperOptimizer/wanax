import numpy as np
from numba import njit, prange

@njit
def pad_array(data, pad_width):
    if data.ndim == 1:
        return np.pad(data, pad_width, mode='edge')
    elif data.ndim == 2:
        return np.pad(data, ((pad_width, pad_width), (pad_width, pad_width)), mode='edge')
    else:  # 3D
        return np.pad(data, ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)), mode='edge')

@njit(parallel=True)
def unsharp_mask(data, radius=1, amount=1.0, threshold=0):
    result = np.zeros_like(data, dtype=np.float32)
    blurred = np.zeros_like(data, dtype=np.float32)
    depth, height, width = data.shape
    kernel_size = 2 * radius + 1
    kernel_sum = kernel_size * kernel_size * kernel_size

    # Compute blur ignoring edges
    for z in prange(radius, depth-radius):
        for y in range(radius, height-radius):
            for x in range(radius, width-radius):
                sum_val = 0.0
                for kz in range(-radius, radius+1):
                    for ky in range(-radius, radius+1):
                        for kx in range(-radius, radius+1):
                            sum_val += data[z+kz, y+ky, x+kx]
                blurred[z, y, x] = sum_val / kernel_sum

    # Apply unsharp mask
    highpass = data - blurred
    mask = np.abs(highpass) > threshold
    result = data + amount * highpass * mask
    return np.clip(result, 0, 255).astype(np.uint8)

@njit(parallel=True)
def laplacian_sharpen(data, strength=1.0):
    result = np.zeros_like(data, dtype=np.float32)

    if data.ndim == 1:
        size = data.shape[0]
        for i in prange(1, size - 1):
            lap = -2 * data[i] + data[i-1] + data[i+1]
            result[i] = data[i] + strength * lap

    elif data.ndim == 2:
        height, width = data.shape
        for y in prange(1, height - 1):
            for x in range(1, width - 1):
                lap = -4 * data[y, x] + (
                        data[y-1, x] + data[y+1, x] +
                        data[y, x-1] + data[y, x+1]
                )
                result[y, x] = data[y, x] + strength * lap

    else:  # 3D
        depth, height, width = data.shape
        for z in prange(1, depth - 1):
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    lap = -6 * data[z, y, x] + (
                            data[z-1, y, x] + data[z+1, y, x] +
                            data[z, y-1, x] + data[z, y+1, x] +
                            data[z, y, x-1] + data[z, y, x+1]
                    )
                    result[z, y, x] = data[z, y, x] + strength * lap

    return np.clip(result, 0, 255).astype(np.uint8)