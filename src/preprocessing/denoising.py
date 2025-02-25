import numpy as np
from numba import njit, prange
import numpy as np
from numba import jit
import numpy.typing as npt
from typing import Tuple

import numpy as np
from numba import jit
import numpy.typing as npt

@njit
def pad_array(data, pad_width):
    if data.ndim == 1:
        return np.pad(data, pad_width, mode='reflect')
    elif data.ndim == 2:
        return np.pad(data, ((pad_width, pad_width), (pad_width, pad_width)), mode='reflect')
    else:  # 3D
        return np.pad(data, ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)), mode='reflect')

@njit(parallel=True)
def median_filter(data, radius=1):
    padded = pad_array(data, radius)
    result = np.zeros_like(data)
    window_size = 2 * radius + 1

    if data.ndim == 1:
        for i in prange(data.shape[0]):
            window = padded[i:i + window_size]
            result[i] = np.median(window)

    elif data.ndim == 2:
        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                window = padded[y:y + window_size, x:x + window_size]
                result[y, x] = np.median(window.ravel())

    else:  # 3D
        for z in prange(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    window = padded[z:z + window_size, y:y + window_size, x:x + window_size]
                    result[z, y, x] = np.median(window.ravel())

    return result

@njit(parallel=True)
def bilateral_filter(data, spatial_sigma=1.0, intensity_sigma=30.0, radius=2):
    padded = pad_array(data, radius)
    result = np.zeros_like(data, dtype=np.float32)

    spatial_factor = -0.5 / (spatial_sigma ** 2)
    intensity_factor = -0.5 / (intensity_sigma ** 2)

    if data.ndim == 1:
        coords = np.arange(-radius, radius + 1)
        spatial_weights = np.exp(spatial_factor * coords ** 2)

        for i in prange(data.shape[0]):
            center = data[i]
            window = padded[i:i + 2*radius + 1]
            intensity_weights = np.exp(intensity_factor * (window - center) ** 2)
            weights = spatial_weights * intensity_weights
            result[i] = np.sum(window * weights) / np.sum(weights)

    elif data.ndim == 2:
        y_coords, x_coords = np.mgrid[-radius:radius+1, -radius:radius+1]
        spatial_weights = np.exp(spatial_factor * (x_coords**2 + y_coords**2))

        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                center = data[y, x]
                window = padded[y:y + 2*radius + 1, x:x + 2*radius + 1]
                intensity_weights = np.exp(intensity_factor * (window - center) ** 2)
                weights = spatial_weights * intensity_weights
                result[y, x] = np.sum(window * weights) / np.sum(weights)

    else:  # 3D
        z_coords, y_coords, x_coords = np.mgrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
        spatial_weights = np.exp(spatial_factor * (x_coords**2 + y_coords**2 + z_coords**2))

        for z in prange(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    center = data[z, y, x]
                    window = padded[z:z + 2*radius + 1, y:y + 2*radius + 1, x:x + 2*radius + 1]
                    intensity_weights = np.exp(intensity_factor * (window - center) ** 2)
                    weights = spatial_weights * intensity_weights
                    result[z, y, x] = np.sum(window * weights) / np.sum(weights)

    return np.clip(result, 0, 255).astype(np.uint8)

@njit(parallel=True)
def nlm_denoising(data, search_radius=5, patch_radius=1, h=10):
    padded = pad_array(data, search_radius + patch_radius)
    result = np.zeros_like(data, dtype=np.float32)
    patch_size = 2 * patch_radius + 1
    factor = -1 / (h ** 2)

    if data.ndim == 1:
        for i in prange(data.shape[0]):
            weights_sum = 0
            pixel_sum = 0
            center_patch = padded[i + search_radius - patch_radius:i + search_radius + patch_radius + 1]

            for j in range(i, min(i + 2*search_radius + 1, data.shape[0])):
                cmp_patch = padded[j + search_radius - patch_radius:j + search_radius + patch_radius + 1]
                dist = np.sum((center_patch - cmp_patch) ** 2)
                weight = np.exp(factor * dist)
                weights_sum += weight
                pixel_sum += weight * padded[j + search_radius]

            result[i] = pixel_sum / weights_sum

    elif data.ndim == 2:
        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                weights_sum = 0
                pixel_sum = 0
                y_start = y + search_radius
                x_start = x + search_radius
                center_patch = padded[y_start - patch_radius:y_start + patch_radius + 1,
                               x_start - patch_radius:x_start + patch_radius + 1]

                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        if y + dy < 0 or y + dy >= data.shape[0] or x + dx < 0 or x + dx >= data.shape[1]:
                            continue

                        cmp_patch = padded[y_start + dy - patch_radius:y_start + dy + patch_radius + 1,
                                    x_start + dx - patch_radius:x_start + dx + patch_radius + 1]
                        dist = np.sum((center_patch - cmp_patch) ** 2)
                        weight = np.exp(factor * dist)
                        weights_sum += weight
                        pixel_sum += weight * padded[y_start + dy, x_start + dx]

                result[y, x] = pixel_sum / weights_sum

    else:  # 3D
        for z in prange(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    weights_sum = 0
                    pixel_sum = 0
                    z_start = z + search_radius
                    y_start = y + search_radius
                    x_start = x + search_radius
                    center_patch = padded[z_start - patch_radius:z_start + patch_radius + 1,
                                   y_start - patch_radius:y_start + patch_radius + 1,
                                   x_start - patch_radius:x_start + patch_radius + 1]

                    for dz in range(-search_radius, search_radius + 1):
                        for dy in range(-search_radius, search_radius + 1):
                            for dx in range(-search_radius, search_radius + 1):
                                if (z + dz < 0 or z + dz >= data.shape[0] or
                                        y + dy < 0 or y + dy >= data.shape[1] or
                                        x + dx < 0 or x + dx >= data.shape[2]):
                                    continue

                                cmp_patch = padded[z_start + dz - patch_radius:z_start + dz + patch_radius + 1,
                                            y_start + dy - patch_radius:y_start + dy + patch_radius + 1,
                                            x_start + dx - patch_radius:x_start + dx + patch_radius + 1]
                                dist = np.sum((center_patch - cmp_patch) ** 2)
                                weight = np.exp(factor * dist)
                                weights_sum += weight
                                pixel_sum += weight * padded[z_start + dz, y_start + dy, x_start + dx]

                    result[z, y, x] = pixel_sum / weights_sum

    return np.clip(result, 0, 255).astype(np.uint8)




@njit
def get_neighbors_3d(z, y, x, depth, height, width):
    directions = np.array([
        [-1, 0, 0], [1, 0, 0],
        [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1]
    ])

    neighbors = []

    for i in range(6):
        nz = z + directions[i, 0]
        ny = y + directions[i, 1]
        nx = x + directions[i, 2]

        if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
            neighbors.append((nz, ny, nx))

    return neighbors


@njit
def flood_fill_f32(volume, iso_threshold, start_threshold):
    depth, height, width = volume.shape
    mask = np.zeros_like(volume, dtype=np.uint8)
    visited = np.zeros_like(volume, dtype=np.uint8)

    iso_threshold_u8 = min(255, max(0, int(iso_threshold)))
    start_threshold_u8 = min(255, max(0, int(start_threshold)))

    max_queue_size = depth * height * width
    queue_z = np.zeros(max_queue_size, dtype=np.int32)
    queue_y = np.zeros(max_queue_size, dtype=np.int32)
    queue_x = np.zeros(max_queue_size, dtype=np.int32)
    queue_start = 0
    queue_end = 0

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if volume[z, y, x] >= start_threshold_u8:
                    queue_z[queue_end] = z
                    queue_y[queue_end] = y
                    queue_x[queue_end] = x
                    queue_end += 1

                    mask[z, y, x] = 1
                    visited[z, y, x] = 1

    directions = np.array([
        [-1, 0, 0], [1, 0, 0],
        [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1]
    ])

    while queue_start < queue_end:
        current_z = queue_z[queue_start]
        current_y = queue_y[queue_start]
        current_x = queue_x[queue_start]
        queue_start += 1

        for i in range(6):
            nz = current_z + directions[i, 0]
            ny = current_y + directions[i, 1]
            nx = current_x + directions[i, 2]

            if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                if visited[nz, ny, nx] == 0 and volume[nz, ny, nx] >= iso_threshold_u8:
                    mask[nz, ny, nx] = 1
                    visited[nz, ny, nx] = 1

                    queue_z[queue_end] = nz
                    queue_y[queue_end] = ny
                    queue_x[queue_end] = nx
                    queue_end += 1

    return mask


@njit(parallel=True)
def segment_and_clean_u8(volume_u8, iso_threshold=127, start_threshold=200):
    mask = flood_fill_f32(volume_u8, iso_threshold, start_threshold)

    result = np.zeros_like(volume_u8)
    for z in prange(volume_u8.shape[0]):
        for y in range(volume_u8.shape[1]):
            for x in range(volume_u8.shape[2]):
                if mask[z, y, x]:
                    result[z, y, x] = volume_u8[z, y, x]
                else:
                    result[z, y, x] = 0

    return result


@njit(parallel=True)
def segment_and_clean_f32(volume_u8, iso_threshold=127, start_threshold=200):
    mask = flood_fill_f32(volume_u8, iso_threshold, start_threshold)

    result = np.zeros_like(volume_u8)
    for z in prange(volume_u8.shape[0]):
        for y in range(volume_u8.shape[1]):
            for x in range(volume_u8.shape[2]):
                if mask[z, y, x]:
                    result[z, y, x] = volume_u8[z, y, x]
                else:
                    result[z, y, x] = 0

    return result


@njit(parallel=True)
def avgpool_denoise_3d(volume_u8, kernel=3):
    depth, height, width = volume_u8.shape
    result = np.zeros_like(volume_u8)
    half = kernel // 2
    kernel_size = 2 * half + 1
    kernel_volume = kernel_size ** 3

    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                value_sum = 0
                count = 0

                for zi in range(-half, half + 1):
                    for yi in range(-half, half + 1):
                        for xi in range(-half, half + 1):
                            nz, ny, nx = z + zi, y + yi, x + xi

                            if not (0 <= nz < depth and 0 <= ny < height and 0 <= nx < width):
                                continue

                            value_sum += volume_u8[nz, ny, nx]
                            count += 1

                if count > 0:
                    result[z, y, x] = value_sum // count

    return result


@njit
def avgpool_denoise_3d_fast(volume_u8, kernel=3):
    depth, height, width = volume_u8.shape
    result = np.zeros_like(volume_u8)
    half = kernel // 2

    padded = np.zeros((depth + 2 * half, height + 2 * half, width + 2 * half), dtype=np.uint8)
    padded[half:half + depth, half:half + height, half:half + width] = volume_u8

    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                neighborhood = padded[z:z + kernel, y:y + kernel, x:x + kernel]
                result[z, y, x] = int(np.mean(neighborhood))

    return result


@jit(nopython=True)
def compute_local_stats(volume: npt.NDArray[np.uint8], kernel_size: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute local standard deviation and mean with given kernel size."""
    pad = kernel_size // 2
    shape = volume.shape
    std = np.zeros_like(volume, dtype=np.float32)
    mean = np.zeros_like(volume, dtype=np.float32)

    for x in range(pad, shape[0] - pad):
        for y in range(pad, shape[1] - pad):
            for z in range(pad, shape[2] - pad):
                neighborhood = volume[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1]
                std[x,y,z] = np.std(neighborhood)
                mean[x,y,z] = np.mean(neighborhood)

    return std, mean


@jit(nopython=True)
def compute_local_stats(volume: npt.NDArray[np.uint8], kernel_size: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    pad = kernel_size // 2
    shape = volume.shape
    std = np.zeros_like(volume, dtype=np.float32)
    mean = np.zeros_like(volume, dtype=np.float32)

    for x in range(pad, shape[0] - pad):
        for y in range(pad, shape[1] - pad):
            for z in range(pad, shape[2] - pad):
                neighborhood = volume[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1]
                std[x,y,z] = np.std(neighborhood)
                mean[x,y,z] = np.mean(neighborhood)

    return std, mean

def determine_thresholds(volume: npt.NDArray[np.uint8], kernel_size: int) -> Tuple[float, float]:
    """
    Automatically determine std and mean thresholds using data statistics.
    Returns (std_threshold, mean_threshold)
    """
    # Compute stats for smallest kernel to get initial segmentation
    std, mean = compute_local_stats(volume, kernel_size)

    # Remove edge effects
    valid_mask = (std > 0) & (mean > 0)
    std_valid = std[valid_mask]
    mean_valid = mean[valid_mask]

    # Compute percentiles for both distributions
    percentiles = np.linspace(0, 100, 100)
    std_percentiles = np.percentile(std_valid, percentiles)
    mean_percentiles = np.percentile(mean_valid, percentiles)

    # Find crossover point where std starts increasing faster than mean
    derivatives = np.diff(std_percentiles) - np.diff(mean_percentiles)
    crossover_idx = np.argmax(derivatives > 0)

    std_threshold = std_percentiles[crossover_idx]
    mean_threshold = mean_percentiles[crossover_idx]

    return std_threshold, mean_threshold

@jit(nopython=True)
def identify_noise(volume: npt.NDArray[np.uint8],
                   kernel_sizes: list[int],
                   std_threshold: float,
                   mean_threshold: float) -> npt.NDArray[np.bool_]:
    noise_mask = np.zeros_like(volume, dtype=np.bool_)

    for kernel_size in kernel_sizes:
        std, mean = compute_local_stats(volume, kernel_size)
        noise_mask |= (std > std_threshold) & (mean < mean_threshold)

    return noise_mask

def clean_volume(volume: npt.NDArray[np.uint8],
                 kernel_sizes: list[int] = [3, 5]) -> npt.NDArray[np.uint8]:
    """
    Clean noise from volume using automatically determined thresholds.
    Returns cleaned volume.
    """
    # Use smallest kernel for threshold determination
    std_threshold, mean_threshold = determine_thresholds(volume, kernel_sizes[0])

    noise_mask = identify_noise(volume, kernel_sizes, std_threshold, mean_threshold)
    cleaned = volume.copy()
    cleaned[noise_mask] = 0
    return cleaned