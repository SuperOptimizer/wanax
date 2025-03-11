import numpy as np
from numba import njit, prange
import numpy as np
from numba import jit
import numpy.typing as npt
from typing import Tuple

import numpy as np
from numba import jit
import numpy.typing as npt

import numpy as np
from numba import njit, prange
import numpy.typing as npt
from typing import Tuple, List

import numpy as np
from numba import njit, prange


@njit
def reflect_indices(idx, size):
    """
    Calculate the reflected index for boundary handling.
    Implements reflection padding similar to np.pad with mode='reflect'.
    """
    if idx < 0:
        # Reflect back from left boundary
        return -idx - 1 if -idx <= size else (2 * size - idx - 1) % (2 * size)
    elif idx >= size:
        # Reflect back from right boundary
        return 2 * size - idx - 1 if idx < 2 * size else idx % (2 * size)
    else:
        # Regular index
        return idx


@njit
def pad_array_reflect_1d(arr, pad_width):
    """Numba-compatible 1D array padding with reflect mode."""
    result = np.zeros(arr.shape[0] + 2 * pad_width, dtype=arr.dtype)
    size = arr.shape[0]

    # Copy the original array to the center
    result[pad_width:pad_width + size] = arr

    # Pad the left side
    for i in range(pad_width):
        idx = reflect_indices(-i - 1, size)
        result[pad_width - i - 1] = arr[idx]

    # Pad the right side
    for i in range(pad_width):
        idx = reflect_indices(size + i, size)
        result[pad_width + size + i] = arr[idx]

    return result


@njit
def pad_array_reflect_2d(arr, pad_width):
    """Numba-compatible 2D array padding with reflect mode."""
    h, w = arr.shape
    result = np.zeros((h + 2 * pad_width, w + 2 * pad_width), dtype=arr.dtype)

    # Copy the original array to the center
    result[pad_width:pad_width + h, pad_width:pad_width + w] = arr

    # Pad the top and bottom
    for i in range(pad_width):
        # Top pad (reflect from top rows)
        top_idx = reflect_indices(-i - 1, h)
        result[pad_width - i - 1, pad_width:pad_width + w] = arr[top_idx, :]

        # Bottom pad (reflect from bottom rows)
        bottom_idx = reflect_indices(h + i, h)
        result[pad_width + h + i, pad_width:pad_width + w] = arr[bottom_idx, :]

    # Pad the left and right
    for j in range(pad_width):
        # Left pad (reflect from left columns)
        left_idx = reflect_indices(-j - 1, w)
        result[:, pad_width - j - 1] = result[:, pad_width + left_idx]

        # Right pad (reflect from right columns)
        right_idx = reflect_indices(w + j, w)
        result[:, pad_width + w + j] = result[:, pad_width + right_idx]

    return result


@njit
def pad_array_reflect_3d(arr, pad_width):
    """Numba-compatible 3D array padding with reflect mode."""
    d, h, w = arr.shape
    result = np.zeros((d + 2 * pad_width, h + 2 * pad_width, w + 2 * pad_width), dtype=arr.dtype)

    # Copy the original array to the center
    result[pad_width:pad_width + d, pad_width:pad_width + h, pad_width:pad_width + w] = arr

    # Pad along z axis (depth)
    for i in range(pad_width):
        # Front pad (reflect from front slices)
        front_idx = reflect_indices(-i - 1, d)
        result[pad_width - i - 1, pad_width:pad_width + h, pad_width:pad_width + w] = arr[front_idx, :, :]

        # Back pad (reflect from back slices)
        back_idx = reflect_indices(d + i, d)
        result[pad_width + d + i, pad_width:pad_width + h, pad_width:pad_width + w] = arr[back_idx, :, :]

    # Pad along y axis (height)
    for j in range(pad_width):
        # Top pad (reflect from top rows)
        top_idx = reflect_indices(-j - 1, h)
        result[:, pad_width - j - 1, pad_width:pad_width + w] = result[:, pad_width + top_idx, pad_width:pad_width + w]

        # Bottom pad (reflect from bottom rows)
        bottom_idx = reflect_indices(h + j, h)
        result[:, pad_width + h + j, pad_width:pad_width + w] = result[:, pad_width + bottom_idx,
                                                                pad_width:pad_width + w]

    # Pad along x axis (width)
    for k in range(pad_width):
        # Left pad (reflect from left columns)
        left_idx = reflect_indices(-k - 1, w)
        result[:, :, pad_width - k - 1] = result[:, :, pad_width + left_idx]

        # Right pad (reflect from right columns)
        right_idx = reflect_indices(w + k, w)
        result[:, :, pad_width + w + k] = result[:, :, pad_width + right_idx]

    return result


@njit
def pad_array(data, pad_width):
    """
    Numba-compatible version of np.pad with reflect mode.
    Handles 1D, 2D, and 3D arrays.
    """
    if data.ndim == 1:
        return pad_array_reflect_1d(data, pad_width)
    elif data.ndim == 2:
        return pad_array_reflect_2d(data, pad_width)
    else:  # 3D
        return pad_array_reflect_3d(data, pad_width)


@njit(parallel=True, fastmath=True)
def median_filter(data, radius=1):
    """
    Apply median filter to an array.

    Parameters:
    -----------
    data : ndarray
        Input array (1D, 2D, or 3D)
    radius : int
        Filter radius (default: 1)

    Returns:
    --------
    ndarray
        Filtered array
    """
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

@njit(parallel=False)
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

@njit(parallel=False)
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


@njit(parallel=False)
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


@njit(parallel=False)
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


@njit(parallel=False)
def avgpool_denoise_3d(volume_u8, kernel=3):
    depth, height, width = volume_u8.shape
    result = np.zeros_like(volume_u8)
    half = kernel // 2

    for z in prange(depth):
        for y in prange(height):
            for x in prange(width):
                value_sum = np.uint16(0)
                count = np.uint16(0)

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


@njit(parallel=False)
def avgpool_denoise_3d_fast(volume_u8, kernel=3):
    """
    Alternative implementation using flattened array.
    """
    depth, height, width = volume_u8.shape
    result = np.zeros_like(volume_u8)
    half = kernel // 2

    # Create padded array
    padded = np.zeros((depth + 2 * half, height + 2 * half, width + 2 * half), dtype=np.uint8)
    padded[half:half + depth, half:half + height, half:half + width] = volume_u8

    # Calculate total elements in kernel
    kernel_size = kernel * kernel * kernel

    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                # Extract neighborhood and calculate mean without dtype parameter
                neighborhood = padded[z:z + kernel, y:y + kernel, x:x + kernel]
                flat_neighborhood = neighborhood.ravel()  # Flatten array

                # Sum as uint32 to prevent overflow
                sum_val = np.uint32(0)
                for i in range(len(flat_neighborhood)):
                    sum_val += flat_neighborhood[i]

                # Convert to uint8 with rounding
                result[z, y, x] = np.uint8(sum_val // kernel_size)

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

import numpy as np
from numba import njit, prange
import numpy.typing as npt
from typing import Tuple

@njit(parallel=False)
def compute_local_stats_3d_u8(volume, kernel_size):
    """Optimized local statistics computation for 3D uint8 volumes."""
    pad = kernel_size // 2
    depth, height, width = volume.shape
    std = np.zeros_like(volume, dtype=np.float32)
    mean = np.zeros_like(volume, dtype=np.float32)

    for z in prange(pad, depth - pad):
        for y in range(pad, height - pad):
            for x in range(pad, width - pad):
                # Skip computation for zero values if they represent background
                if volume[z, y, x] == 0:
                    continue

                # Extract neighborhood
                neighborhood = volume[z-pad:z+pad+1, y-pad:y+pad+1, x-pad:x+pad+1]

                # Compute statistics efficiently
                sum_vals = np.float32(0)
                sum_sq_vals = np.float32(0)
                count = 0

                # Manual calculation is faster than np.std/mean with Numba
                for kz in range(kernel_size):
                    for ky in range(kernel_size):
                        for kx in range(kernel_size):
                            val = neighborhood[kz, ky, kx]
                            sum_vals += val
                            sum_sq_vals += val * val
                            count += 1

                if count > 0:
                    mean_val = sum_vals / count
                    mean[z, y, x] = mean_val

                    if count > 1:
                        variance = (sum_sq_vals - (sum_vals * sum_vals) / count) / (count - 1)
                        std[z, y, x] = np.sqrt(max(0, variance))  # Avoid negative values due to floating point errors

    return std, mean

@njit
def determine_thresholds_3d_u8(std, mean):
    """Efficiently determine std and mean thresholds without boolean indexing."""
    # Create arrays to count and collect valid values
    depth, height, width = std.shape

    # Count valid values first
    valid_count = 0
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if std[z, y, x] > 0 and mean[z, y, x] > 0:
                    valid_count += 1

    if valid_count == 0:
        return 0.0, 0.0

    # Create arrays to hold valid values
    std_valid = np.zeros(valid_count, dtype=np.float32)
    mean_valid = np.zeros(valid_count, dtype=np.float32)

    # Fill arrays with valid values
    idx = 0
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if std[z, y, x] > 0 and mean[z, y, x] > 0:
                    std_valid[idx] = std[z, y, x]
                    mean_valid[idx] = mean[z, y, x]
                    idx += 1

    # Use approximate percentiles with histogram for better performance
    num_bins = 100

    # Compute min/max for std and mean
    std_min = std_valid[0]
    std_max = std_valid[0]
    mean_min = mean_valid[0]
    mean_max = mean_valid[0]

    for i in range(1, valid_count):
        if std_valid[i] < std_min:
            std_min = std_valid[i]
        if std_valid[i] > std_max:
            std_max = std_valid[i]
        if mean_valid[i] < mean_min:
            mean_min = mean_valid[i]
        if mean_valid[i] > mean_max:
            mean_max = mean_valid[i]

    # Handle edge cases
    if std_min == std_max or mean_min == mean_max:
        return std_max * 0.5, mean_max * 0.5

    # Compute histograms
    std_hist = np.zeros(num_bins, dtype=np.int32)
    mean_hist = np.zeros(num_bins, dtype=np.int32)

    std_range = std_max - std_min
    mean_range = mean_max - mean_min

    for i in range(valid_count):
        std_val = std_valid[i]
        mean_val = mean_valid[i]

        # Find bin index for std
        std_idx = min(int((std_val - std_min) / std_range * (num_bins-1)), num_bins-1)
        std_hist[std_idx] += 1

        # Find bin index for mean
        mean_idx = min(int((mean_val - mean_min) / mean_range * (num_bins-1)), num_bins-1)
        mean_hist[mean_idx] += 1

    # Compute cumulative histograms
    std_cum = np.zeros(num_bins, dtype=np.float32)
    mean_cum = np.zeros(num_bins, dtype=np.float32)

    std_cum[0] = float(std_hist[0])
    mean_cum[0] = float(mean_hist[0])

    for i in range(1, num_bins):
        std_cum[i] = std_cum[i-1] + float(std_hist[i])
        mean_cum[i] = mean_cum[i-1] + float(mean_hist[i])

    # Normalize cumulative histograms
    for i in range(num_bins):
        std_cum[i] = std_cum[i] / float(valid_count)
        mean_cum[i] = mean_cum[i] / float(valid_count)

    # Compute derivatives
    derivatives = np.zeros(num_bins-1, dtype=np.float32)
    for i in range(num_bins-1):
        derivatives[i] = (std_cum[i+1] - std_cum[i]) - (mean_cum[i+1] - mean_cum[i])

    # Find crossover point
    crossover_idx = 0
    max_derivative = derivatives[0]

    for i in range(1, num_bins-1):
        if derivatives[i] > max_derivative:
            max_derivative = derivatives[i]
            crossover_idx = i

    # Get threshold values
    std_threshold = std_min + std_range * (float(crossover_idx) / float(num_bins))
    mean_threshold = mean_min + mean_range * (float(crossover_idx) / float(num_bins))

    return std_threshold, mean_threshold

@njit(parallel=False)
def identify_noise_3d_u8(volume, kernel_sizes, std_threshold, mean_threshold):
    """Identify noise in 3D volume using multiple kernel sizes."""
    noise_mask = np.zeros_like(volume, dtype=np.uint8)

    for kernel_size in kernel_sizes:
        std, mean = compute_local_stats_3d_u8(volume, kernel_size)

        # Update noise mask
        for z in prange(volume.shape[0]):
            for y in range(volume.shape[1]):
                for x in range(volume.shape[2]):
                    if std[z, y, x] > std_threshold and mean[z, y, x] < mean_threshold:
                        noise_mask[z, y, x] = 1

    return noise_mask

@njit(parallel=False)
def clean_volume_3d_u8(volume, kernel_sizes=(3, 5)):
    """
    Optimized noise cleaning for 3D uint8 volumes.
    Avoids boolean indexing for Numba compatibility.

    Parameters:
    -----------
    volume : ndarray (uint8)
        Input 3D array of uint8 values
    kernel_sizes : tuple of int
        Kernel sizes to use for noise detection (default: (3, 5))

    Returns:
    --------
    cleaned : ndarray (uint8)
        Cleaned volume with noise removed
    """
    # Handle empty volume
    if volume.size == 0:
        return volume.copy()

    # Compute statistics for smallest kernel
    std, mean = compute_local_stats_3d_u8(volume, kernel_sizes[0])

    # Determine thresholds
    std_threshold, mean_threshold = determine_thresholds_3d_u8(std, mean)

    # If thresholds couldn't be determined, return the original volume
    if std_threshold == 0 and mean_threshold == 0:
        return volume.copy()

    # Identify noise
    noise_mask = identify_noise_3d_u8(volume, kernel_sizes, std_threshold, mean_threshold)

    # Apply mask to create cleaned volume
    cleaned = np.zeros_like(volume)

    for z in prange(volume.shape[0]):
        for y in range(volume.shape[1]):
            for x in range(volume.shape[2]):
                if noise_mask[z, y, x] == 0:
                    cleaned[z, y, x] = volume[z, y, x]

    return cleaned



import numpy as np
from numba import njit, prange


@njit(parallel=False, fastmath=True)
def avgpool_denoise_3d_3x3x3(volume_u8):
    """
    Highly optimized 3D average pooling denoising for exactly 3x3x3 kernel.
    Specialized for uint8 data with branchless vectorized operations.

    Parameters:
    -----------
    volume_u8 : ndarray (uint8)
        Input 3D volume of uint8 values

    Returns:
    --------
    result : ndarray (uint8)
        Denoised 3D volume with same shape as input
    """
    depth, height, width = volume_u8.shape
    result = np.zeros_like(volume_u8)

    # Create padded array with efficient slice assignment
    padded = np.zeros((depth + 2, height + 2, width + 2), dtype=np.uint8)
    padded[1:1 + depth, 1:1 + height, 1:1 + width] = volume_u8

    # Fast division by 27 using multiplicative inverse with proper rounding
    # (2^16 / 27) ≈ 2427.26, floor to 2427 to prevent overflow
    M = np.uint32(2427)
    SHIFT = 16
    ROUND = np.uint32(1 << (SHIFT - 1))  # 2^15 for rounding

    # Process each voxel in parallel with optimized memory access
    for z in prange(depth):
        zp = z + 1
        # Extract current z planes for better cache locality
        zm1_plane = padded[zp-1]
        z_plane = padded[zp]
        zp1_plane = padded[zp+1]

        for y in range(height):
            yp = y + 1
            for x in range(width):
                xp = x + 1

                # Sum all 27 voxels using cached z-planes for better cache locality
                # Using direct addition instead of loops for better vectorization
                sum_val = np.uint32(
                    # Layer 0 (z-1)
                    zm1_plane[yp-1, xp-1] +
                    zm1_plane[yp-1, xp  ] +
                    zm1_plane[yp-1, xp+1] +
                    zm1_plane[yp  , xp-1] +
                    zm1_plane[yp  , xp  ] +
                    zm1_plane[yp  , xp+1] +
                    zm1_plane[yp+1, xp-1] +
                    zm1_plane[yp+1, xp  ] +
                    zm1_plane[yp+1, xp+1] +

                    # Layer 1 (z)
                    z_plane[yp-1, xp-1] +
                    z_plane[yp-1, xp  ] +
                    z_plane[yp-1, xp+1] +
                    z_plane[yp  , xp-1] +
                    z_plane[yp  , xp  ] +
                    z_plane[yp  , xp+1] +
                    z_plane[yp+1, xp-1] +
                    z_plane[yp+1, xp  ] +
                    z_plane[yp+1, xp+1] +

                    # Layer 2 (z+1)
                    zp1_plane[yp-1, xp-1] +
                    zp1_plane[yp-1, xp  ] +
                    zp1_plane[yp-1, xp+1] +
                    zp1_plane[yp  , xp-1] +
                    zp1_plane[yp  , xp  ] +
                    zp1_plane[yp  , xp+1] +
                    zp1_plane[yp+1, xp-1] +
                    zp1_plane[yp+1, xp  ] +
                    zp1_plane[yp+1, xp+1]
                )

                # Fast branchless division by 27 with proper rounding
                # Much faster than standard division for fixed divisor
                result[z, y, x] = np.uint8((sum_val * M + ROUND) >> SHIFT)

    return result


@njit(fastmath=True)
def find_median_u8(values, count):
    """
    Find the median of an array of uint8 values efficiently.
    Uses counting sort which is very efficient for uint8 data.

    Parameters:
    -----------
    values : ndarray (uint8)
        Array of uint8 values
    count : int
        Number of values to consider

    Returns:
    --------
    median : uint8
        Median value
    """
    # Histogram-based approach for finding median (faster than sorting for uint8)
    hist = np.zeros(256, dtype=np.int32)

    # Count occurrences of each value
    for i in range(count):
        hist[values[i]] += 1

    # Find the middle position
    middle = count // 2

    # Cumulative sum to find the median
    cum_sum = 0
    for i in range(256):
        cum_sum += hist[i]
        if cum_sum > middle:
            return np.uint8(i)

    # If we didn't find the median (shouldn't happen unless array is empty)
    return np.uint8(0)

@njit(parallel=True, fastmath=True)
def median_denoise_3d_3x3x3(volume_u8):
    """
    Highly optimized 3D median filtering for exactly 3x3x3 kernel.
    Specialized for uint8 data with efficient median calculation.

    Parameters:
    -----------
    volume_u8 : ndarray (uint8)
        Input 3D volume of uint8 values

    Returns:
    --------
    result : ndarray (uint8)
        Denoised 3D volume with same shape as input
    """
    depth, height, width = volume_u8.shape
    result = np.zeros_like(volume_u8)

    # Create padded array with efficient slice assignment
    padded = np.zeros((depth + 2, height + 2, width + 2), dtype=np.uint8)
    padded[1:1 + depth, 1:1 + height, 1:1 + width] = volume_u8

    # Pre-allocate buffer for neighborhood values
    neighborhood = np.zeros(27, dtype=np.uint8)

    # Process each voxel in parallel
    for z in prange(depth):
        zp = z + 1  # Padded z coordinate

        # Extract the three z-planes for better cache locality
        zm1_plane = padded[zp-1]
        z_plane = padded[zp]
        zp1_plane = padded[zp+1]

        for y in range(height):
            yp = y + 1  # Padded y coordinate

            for x in range(width):
                xp = x + 1  # Padded x coordinate

                # Check if center voxel is zero (optimization for sparse volumes)
                center_value = z_plane[yp, xp]
                if center_value == 0:
                    # Optional: If all neighbors are also zero, skip processing
                    # This check can be removed if your volumes aren't sparse
                    if (zm1_plane[yp, xp] == 0 and zp1_plane[yp, xp] == 0 and
                            z_plane[yp-1, xp] == 0 and z_plane[yp+1, xp] == 0 and
                            z_plane[yp, xp-1] == 0 and z_plane[yp, xp+1] == 0):
                        continue

                # Collect neighborhood values (flatten 3x3x3 cube)
                idx = 0

                # Layer 0 (z-1)
                neighborhood[idx] = zm1_plane[yp-1, xp-1]; idx += 1
                neighborhood[idx] = zm1_plane[yp-1, xp  ]; idx += 1
                neighborhood[idx] = zm1_plane[yp-1, xp+1]; idx += 1
                neighborhood[idx] = zm1_plane[yp  , xp-1]; idx += 1
                neighborhood[idx] = zm1_plane[yp  , xp  ]; idx += 1
                neighborhood[idx] = zm1_plane[yp  , xp+1]; idx += 1
                neighborhood[idx] = zm1_plane[yp+1, xp-1]; idx += 1
                neighborhood[idx] = zm1_plane[yp+1, xp  ]; idx += 1
                neighborhood[idx] = zm1_plane[yp+1, xp+1]; idx += 1

                # Layer 1 (z)
                neighborhood[idx] = z_plane[yp-1, xp-1]; idx += 1
                neighborhood[idx] = z_plane[yp-1, xp  ]; idx += 1
                neighborhood[idx] = z_plane[yp-1, xp+1]; idx += 1
                neighborhood[idx] = z_plane[yp  , xp-1]; idx += 1
                neighborhood[idx] = z_plane[yp  , xp  ]; idx += 1
                neighborhood[idx] = z_plane[yp  , xp+1]; idx += 1
                neighborhood[idx] = z_plane[yp+1, xp-1]; idx += 1
                neighborhood[idx] = z_plane[yp+1, xp  ]; idx += 1
                neighborhood[idx] = z_plane[yp+1, xp+1]; idx += 1

                # Layer 2 (z+1)
                neighborhood[idx] = zp1_plane[yp-1, xp-1]; idx += 1
                neighborhood[idx] = zp1_plane[yp-1, xp  ]; idx += 1
                neighborhood[idx] = zp1_plane[yp-1, xp+1]; idx += 1
                neighborhood[idx] = zp1_plane[yp  , xp-1]; idx += 1
                neighborhood[idx] = zp1_plane[yp  , xp  ]; idx += 1
                neighborhood[idx] = zp1_plane[yp  , xp+1]; idx += 1
                neighborhood[idx] = zp1_plane[yp+1, xp-1]; idx += 1
                neighborhood[idx] = zp1_plane[yp+1, xp  ]; idx += 1
                neighborhood[idx] = zp1_plane[yp+1, xp+1]; idx += 1

                # Find median value
                result[z, y, x] = find_median_u8(neighborhood, 27)

    return result


@njit(fastmath=True)
def pad_array_edge_3d(data, pad_width):
    """
    Manual edge padding for 3D arrays that's compatible with Numba.

    Parameters:
    -----------
    data : ndarray
        Input 3D array to be padded
    pad_width : int
        Width of padding to add on all sides

    Returns:
    --------
    padded : ndarray
        Padded 3D array with edge values repeated
    """
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
def anisotropic_diffusion_3d(data, iterations=5, kappa=50.0, gamma=0.1):
    """
    Apply 3D anisotropic diffusion to smooth noise while preserving edges.

    Parameters:
    -----------
    data : ndarray (uint8)
        Input 3D volume
    iterations : int
        Number of diffusion iterations
    kappa : float
        Conductance parameter (controls edge sensitivity)
    gamma : float
        Rate of diffusion (step size)

    Returns:
    --------
    filtered : ndarray (uint8)
        Filtered 3D volume
    """
    # Convert to float for processing
    img = data.astype(np.float32)

    for _ in range(iterations):
        # Create padded volume for gradient calculation
        padded = pad_array_edge_3d(img, 1)

        depth, height, width = img.shape
        for z in prange(0, depth):
            for y in range(0, height):
                for x in range(0, width):
                    # Calculate gradients in all 6 directions
                    north = padded[z + 1, y + 1 + 1, x + 1] - padded[z + 1, y + 1, x + 1]
                    south = padded[z + 1, y + 1 - 1, x + 1] - padded[z + 1, y + 1, x + 1]
                    east = padded[z + 1, y + 1, x + 1 + 1] - padded[z + 1, y + 1, x + 1]
                    west = padded[z + 1, y + 1, x + 1 - 1] - padded[z + 1, y + 1, x + 1]
                    top = padded[z + 1 + 1, y + 1, x + 1] - padded[z + 1, y + 1, x + 1]
                    bottom = padded[z + 1 - 1, y + 1, x + 1] - padded[z + 1, y + 1, x + 1]

                    # Calculate diffusion coefficients
                    cn = np.exp(-(north * north) / (kappa * kappa))
                    cs = np.exp(-(south * south) / (kappa * kappa))
                    ce = np.exp(-(east * east) / (kappa * kappa))
                    cw = np.exp(-(west * west) / (kappa * kappa))
                    ct = np.exp(-(top * top) / (kappa * kappa))
                    cb = np.exp(-(bottom * bottom) / (kappa * kappa))

                    # Update using weighted diffusion
                    update = gamma * (
                            cn * north + cs * south +
                            ce * east + cw * west +
                            ct * top + cb * bottom
                    )

                    img[z, y, x] += update

    return np.clip(img, 0, 255).astype(np.uint8)