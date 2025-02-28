import torch
import numpy as np
import time
from typing import Tuple, Optional, Union
import os

# ===================================
# UTILITY FUNCTIONS
# ===================================

def pad_array_3d(data: torch.Tensor, pad_width: int, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pad 3D tensor with reflection padding.

    Args:
        data: Input tensor [D, H, W]
        pad_width: Padding width
        out: Optional pre-allocated output tensor

    Returns:
        Padded tensor
    """
    # PyTorch's pad function takes padding in reverse order (x, y, z)
    padding = (pad_width, pad_width, pad_width, pad_width, pad_width, pad_width)
    # pad doesn't support out parameter, but we can reuse the output tensor later
    return torch.nn.functional.pad(data, padding, mode='reflect')

def scale_volume_optimized(volume: torch.Tensor, scale_factor: float, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scale 3D volume with nearest neighbor interpolation.

    Args:
        volume: Input tensor [D, H, W] as uint8
        scale_factor: Scale factor for resizing
        out: Optional pre-allocated output tensor

    Returns:
        Scaled tensor as uint8
    """
    if scale_factor == 1.0:
        if out is not None:
            out.copy_(volume)
            return out
        return volume.clone()

    # Calculate new dimensions
    old_depth, old_height, old_width = volume.shape
    new_depth = int(old_depth * scale_factor)
    new_height = int(old_height * scale_factor)
    new_width = int(old_width * scale_factor)

    # Check if out tensor has correct shape
    if out is not None and out.shape != (new_depth, new_height, new_width):
        out = None  # Will create new output tensor

    # Prepare input for interpolation (add batch and channel dims)
    # Use float32 only for the actual interpolation
    input_tensor = volume.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Use PyTorch's interpolate function (doesn't support 'out')
    result = torch.nn.functional.interpolate(
        input_tensor,
        size=(new_depth, new_height, new_width),
        mode='nearest'
    )

    # Remove batch and channel dims
    result = result.squeeze(0).squeeze(0)

    # Convert to uint8 and store in out tensor if provided
    if out is not None:
        torch.clamp(result, 0, 255, out=out)
        return out.to(torch.uint8)
    else:
        return torch.clamp(result, 0, 255).to(torch.uint8)

def numpy_to_torch(np_array: np.ndarray, device: str = 'cpu', out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor.

    Args:
        np_array: Input numpy array
        device: Target device
        out: Optional pre-allocated output tensor

    Returns:
        PyTorch tensor as uint8
    """
    # For uint8 numpy arrays, direct conversion
    if np_array.dtype == np.uint8:
        tensor = torch.from_numpy(np_array.copy()).to(device=device, dtype=torch.uint8)
    else:
        # Other types, convert to uint8 numpy first then to tensor
        # This avoids GPU conversion overhead for non-uint8 types
        arr_uint8 = np.clip(np_array, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(arr_uint8).to(device=device, dtype=torch.uint8)

    # Copy to out tensor if provided
    if out is not None:
        out.copy_(tensor)
        return out
    return tensor

def torch_to_numpy(torch_tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array.

    Args:
        torch_tensor: Input PyTorch tensor

    Returns:
        Numpy array as uint8
    """
    # Move tensor to CPU first if it's on another device
    tensor = torch_tensor.cpu()

    # Convert to uint8 if not already
    if tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 255).to(torch.uint8)

    return tensor.numpy()

# ===================================
# TENSOR REUSE MANAGER
# ===================================

class TensorCache:
    """Manages reusable tensors to reduce memory allocation."""

    def __init__(self):
        self.cache = {}

    # In process.py
    def get(self, shape, dtype=torch.float32, device='cpu', key=None):
        # If shape is a complex tuple containing shape, dtype, and device
        if isinstance(shape, tuple) and len(shape) == 3 and isinstance(shape[0], tuple):
            return torch.zeros(shape[0], dtype=shape[1], device=shape[2])
        # Normal case
        return torch.zeros(shape, dtype=dtype, device=device)

    def put(self, tensor, key=None):
        """Store tensor in cache."""
        if key is None:
            key = (tensor.shape, tensor.dtype, tensor.device)
        self.cache[key] = tensor

    def clear(self):
        """Clear all cached tensors."""
        self.cache.clear()

# Global tensor cache
tensor_cache = TensorCache()

# ===================================
# DENOISING FILTERS
# ===================================

def avgpool_denoise_3d(volume: torch.Tensor, kernel: int = 3, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Average pooling denoising for 3D tensor data.

    Args:
        volume: Input tensor [D, H, W] as uint8
        kernel: Kernel size for average pooling
        out: Optional pre-allocated output tensor

    Returns:
        Denoised tensor as uint8
    """
    # Handle edge cases
    if volume.numel() == 0:
        if out is not None:
            out.zero_()
            return out
        return torch.zeros_like(volume)

    device = volume.device

    # Create output tensor if not provided
    if out is None or out.shape != volume.shape:
        out = torch.zeros_like(volume)

    # Use 3D convolution with an averaging kernel for efficiency
    half = kernel // 2

    # Reuse tensors for padding and intermediate results
    padded_key = ((volume.shape[0] + 2*half, volume.shape[1] + 2*half, volume.shape[2] + 2*half),
                  torch.float32, device)
    padded = tensor_cache.get(padded_key)

    # Convert to float32 only for the padded computation
    # We're doing a manual copy of uint8 to float32 to skip creating new tensor
    padded.zero_()
    padded_view = padded[half:-half, half:-half, half:-half]
    padded_view.copy_(volume.to(dtype=torch.float32))

    # Do padding (reflect)
    for d in range(half):
        # Pad z-dimension
        padded[d, half:-half, half:-half] = padded[2*half-d, half:-half, half:-half]
        padded[-(d+1), half:-half, half:-half] = padded[-(2*half-d+1), half:-half, half:-half]

        # Pad y-dimension
        padded[:, d, half:-half] = padded[:, 2*half-d, half:-half]
        padded[:, -(d+1), half:-half] = padded[:, -(2*half-d+1), half:-half]

        # Pad x-dimension
        padded[:, :, d] = padded[:, :, 2*half-d]
        padded[:, :, -(d+1)] = padded[:, :, -(2*half-d+1)]

    # Create kernel for 3D convolution
    kernel_volume = kernel * kernel * kernel
    kernel_weight = torch.ones((1, 1, kernel, kernel, kernel),
                               device=device, dtype=torch.float32) / kernel_volume

    # Add batch and channel dimensions for convolution
    # Use an existing tensor to avoid creating a new 5D tensor
    batch_shape = (1, 1, padded.shape[0], padded.shape[1], padded.shape[2])
    batch_key = (batch_shape, torch.float32, device)
    padded_batch = tensor_cache.get(batch_key)
    padded_batch[0, 0].copy_(padded)

    # Execute 3D convolution
    # PyTorch doesn't support 'out' for conv3d, but we can reuse result tensor
    result = torch.nn.functional.conv3d(padded_batch, kernel_weight, padding=0)

    # Convert back to uint8
    # Clamp and round to get proper uint8 values
    depth, height, width = volume.shape
    torch.clamp(result[0, 0, :depth, :height, :width], 0, 255, out=padded[:depth, :height, :width])
    torch.round(padded[:depth, :height, :width], out=padded[:depth, :height, :width])
    out.copy_(padded[:depth, :height, :width].to(torch.uint8))

    # Return tensors to cache
    tensor_cache.put(padded, padded_key)
    tensor_cache.put(padded_batch, batch_key)

    return out

# ===================================
# HISTOGRAM OPERATIONS
# ===================================

def calculate_histogram(volume: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Calculate histogram of tensor data efficiently.

    Args:
        volume: Input tensor as uint8
        bins: Number of histogram bins

    Returns:
        Histogram tensor
    """
    device = volume.device

    # For uint8 data, we can use bincount more efficiently than histc
    if volume.dtype == torch.uint8:
        flat_volume = volume.reshape(-1)
        hist = torch.bincount(flat_volume.to(torch.int16), minlength=bins)
        return hist.to(device)

    # If not uint8, use histc
    volume_f32 = volume.to(dtype=torch.float32)
    return torch.histc(volume_f32, bins=bins, min=0, max=255)

def histogram_equalization_3d(volume: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Histogram equalization for 3D tensor data.

    Args:
        volume: Input tensor [D, H, W] as uint8
        out: Optional pre-allocated output tensor

    Returns:
        Equalized tensor as uint8
    """
    if volume.numel() == 0:
        if out is not None:
            out.zero_()
            return out
        return torch.zeros_like(volume)

    device = volume.device

    # Create output tensor if not provided
    if out is None:
        out = torch.zeros_like(volume)
    elif out.shape != volume.shape:
        out = torch.zeros_like(volume)

    # Calculate histogram (staying in uint8/int16 domain)
    hist = calculate_histogram(volume)

    # Get total number of voxels
    total_voxels = volume.numel()

    # Calculate CDF
    cdf = torch.cumsum(hist, dim=0)

    # Find first non-zero bin
    non_zero_mask = hist > 0
    if not torch.any(non_zero_mask):
        out.copy_(volume)
        return out

    min_bin = torch.argmax(non_zero_mask.to(torch.uint8)).item()

    # Create lookup table for mapping
    lut = torch.zeros(256, dtype=torch.uint8, device=device)

    # Avoid division by zero
    if total_voxels > 0 and cdf[min_bin].item() < total_voxels:
        # Create lookup table using direct indexing
        # This avoids creating new float32 tensors
        cdf_min = cdf[min_bin].item()
        denominator = float(total_voxels - cdf_min)

        for i in range(256):
            if hist[i].item() == 0:
                lut[i] = 0
            else:
                # Scale to 0-255 using direct computation
                val = int(((cdf[i].item() - cdf_min) * 255.0) / denominator + 0.5)
                lut[i] = min(255, max(0, val))

        # Apply LUT to volume using advanced indexing
        # This operation creates a new tensor but is highly optimized in PyTorch
        out.copy_(lut[volume])
        return out
    else:
        out.copy_(volume)
        return out

def adaptive_histogram_local_3d(volume: torch.Tensor, clip_limit: float = 2.0,
                                grid_size: int = 8, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Simplified CLAHE-like local histogram equalization.

    Args:
        volume: Input tensor [D, H, W] as uint8
        clip_limit: Clip limit for histogram equalization
        grid_size: Grid size for local processing
        out: Optional pre-allocated output tensor

    Returns:
        Equalized tensor as uint8
    """
    if volume.numel() == 0:
        if out is not None:
            out.zero_()
            return out
        return torch.zeros_like(volume)

    device = volume.device
    depth, height, width = volume.shape

    # Create output tensor if not provided
    if out is None:
        out = torch.zeros_like(volume)
    elif out.shape != volume.shape:
        out = torch.zeros_like(volume)

    # Float32 buffer for intermediate calculations
    result_f32_key = (volume.shape, torch.float32, device)
    result_f32 = tensor_cache.get(result_f32_key)
    result_f32.zero_()

    # Calculate grid dimensions
    z_step = max(1, depth // grid_size)
    y_step = max(1, height // grid_size)
    x_step = max(1, width // grid_size)

    # Process each grid cell
    for z_start in range(0, depth, z_step):
        z_end = min(z_start + z_step, depth)
        for y_start in range(0, height, y_step):
            y_end = min(y_start + y_step, height)
            for x_start in range(0, width, x_step):
                x_end = min(x_start + x_step, width)

                # Skip empty cells
                if z_end <= z_start or y_end <= y_start or x_end <= x_start:
                    continue

                # Extract cell
                cell = volume[z_start:z_end, y_start:y_end, x_start:x_end]

                # Calculate histogram (staying in uint8/int16 domain)
                hist = calculate_histogram(cell)

                # Apply clip limit
                if clip_limit > 0:
                    cell_size = (z_end-z_start) * (y_end-y_start) * (x_end-x_start)
                    clip_height = max(1, int(cell_size * clip_limit / 256))

                    # Calculate excess and clip using direct indexing to avoid creating new tensors
                    excess = 0
                    for i in range(256):
                        if hist[i] > clip_height:
                            excess += hist[i].item() - clip_height
                            hist[i] = clip_height

                    # Redistribute excess
                    if excess > 0:
                        redistrib_per_bin = excess / 256
                        for i in range(256):
                            hist[i] += redistrib_per_bin

                # Calculate CDF
                cdf = torch.cumsum(hist, dim=0)
                total_pixels = cell.numel()

                if total_pixels > 0:
                    # Create mapping function directly to avoid new tensor allocations
                    for z in range(z_start, z_end):
                        for y in range(y_start, y_end):
                            for x in range(x_start, x_end):
                                pixel_value = volume[z, y, x].item()
                                equalized = (cdf[pixel_value].item() * 255.0) / total_pixels
                                result_f32[z, y, x] = equalized

    # Convert result to uint8
    torch.clamp(result_f32, 0, 255, out=result_f32)
    torch.round(result_f32, out=result_f32)
    out.copy_(result_f32.to(torch.uint8))

    # Return tensor to cache
    tensor_cache.put(result_f32, result_f32_key)

    return out

# ===================================
# NORMALIZATION
# ===================================

def calculate_mean_std(volume: torch.Tensor) -> Tuple[float, float]:
    """Calculate mean and std of volume efficiently.

    Args:
        volume: Input tensor as uint8

    Returns:
        Tuple of (mean, std)
    """
    # Skip empty volumes
    if volume.numel() == 0:
        return 0.0, 0.0

    device = volume.device

    # For uint8, compute sum directly (stays in int32 domain)
    if volume.dtype == torch.uint8:
        sum_val = torch.sum(volume).item()
        count = volume.numel()
        mean = sum_val / count

        # Calculate std (convert to float only for this calculation)
        # Use a buffer tensor for squared differences to avoid new allocations
        squared_diff_key = (volume.shape, torch.float32, device)
        squared_diff = tensor_cache.get(squared_diff_key)

        # Direct pixel-wise operation into pre-allocated buffer
        # This avoids creating a new tensor for the intermediate float32 calculation
        for idx, val in enumerate(volume.reshape(-1)):
            diff = float(val.item()) - mean
            squared_diff.reshape(-1)[idx] = diff * diff

        variance = torch.sum(squared_diff).item() / (count - 1)
        std = variance ** 0.5

        # Return tensor to cache
        tensor_cache.put(squared_diff, squared_diff_key)

        return mean, std

    # If not uint8, convert to float32 for calculation
    volume_f32 = volume.to(dtype=torch.float32)
    mean = torch.mean(volume_f32).item()
    std = torch.std(volume_f32, unbiased=True).item()

    return mean, std

def z_score_normalize_3d(volume: torch.Tensor, z_min: float = -3.0, z_max: float = 3.0,
                         out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Z-score normalization for 3D tensor data.

    Args:
        volume: Input tensor [D, H, W] as uint8
        z_min: Minimum z-score for clipping
        z_max: Maximum z-score for clipping
        out: Optional pre-allocated output tensor

    Returns:
        Normalized tensor as uint8
    """
    if volume.numel() == 0:
        if out is not None:
            out.zero_()
            return out
        return torch.zeros_like(volume)

    device = volume.device

    # Create output tensor if not provided
    if out is None:
        out = torch.zeros_like(volume)
    elif out.shape != volume.shape:
        out = torch.zeros_like(volume)

    # Calculate statistics
    mean, std = calculate_mean_std(volume)

    # Handle edge cases
    if std < 1e-10:
        out.zero_()
        return out

    # Use float32 buffer for intermediate calculations
    # This avoids creating multiple temporary tensors
    f32_shape_key = (volume.shape, torch.float32, device)
    normalized_f32 = tensor_cache.get(f32_shape_key)

    # Calculate z-scores directly
    # This manual copy + operation avoids creating temporary tensors
    for idx, val in enumerate(volume.reshape(-1)):
        z_score = (float(val.item()) - mean) / std
        z_score = min(z_max, max(z_min, z_score))  # Clip z-score

        # Scale to 0-255 range
        normalized = 255.0 * (z_score - z_min) / (z_max - z_min)
        normalized_f32.reshape(-1)[idx] = normalized

    # Round and convert to uint8
    torch.round(normalized_f32, out=normalized_f32)
    torch.clamp(normalized_f32, 0, 255, out=normalized_f32)
    out.copy_(normalized_f32.to(torch.uint8))

    # Return tensor to cache
    tensor_cache.put(normalized_f32, f32_shape_key)

    return out

# ===================================
# CONTRAST ENHANCEMENT
# ===================================

def compute_local_contrast_3d(volume: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute local contrast (Laplacian) for 3D tensor data.

    Args:
        volume: Input tensor [D, H, W] as uint8
        out: Optional pre-allocated output tensor

    Returns:
        Contrast tensor as uint8
    """
    if volume.numel() == 0:
        if out is not None:
            out.zero_()
            return out
        return torch.zeros_like(volume)

    device = volume.device
    depth, height, width = volume.shape

    # Create output tensor if not provided
    if out is None:
        out = torch.zeros_like(volume)
    elif out.shape != volume.shape:
        out = torch.zeros_like(volume)

    # Using optimized 3D Laplacian computation for Apple M4
    # Instead of conv3d, we'll use direct indexing which is faster on M4

    # Create float32 buffer
    f32_shape_key = (volume.shape, torch.float32, device)
    contrast_f32 = tensor_cache.get(f32_shape_key)
    contrast_f32.zero_()

    # Convert volume to float32 for computation
    padded_shape_key = ((depth+2, height+2, width+2), torch.float32, device)
    padded = tensor_cache.get(padded_shape_key)
    padded.zero_()

    # Copy volume to padded center
    padded[1:-1, 1:-1, 1:-1].copy_(volume.to(dtype=torch.float32))

    # Do padding (reflect)
    padded[0, 1:-1, 1:-1] = padded[2, 1:-1, 1:-1]
    padded[-1, 1:-1, 1:-1] = padded[-3, 1:-1, 1:-1]
    padded[:, 0, 1:-1] = padded[:, 2, 1:-1]
    padded[:, -1, 1:-1] = padded[:, -3, 1:-1]
    padded[:, :, 0] = padded[:, :, 2]
    padded[:, :, -1] = padded[:, :, -3]

    # Compute Laplacian
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # Get center value and 6-connected neighbors
                center = padded[z+1, y+1, x+1]

                # These index operations are efficient on M4
                top = padded[z, y+1, x+1]
                bottom = padded[z+2, y+1, x+1]
                left = padded[z+1, y, x+1]
                right = padded[z+1, y+2, x+1]
                front = padded[z+1, y+1, x]
                back = padded[z+1, y+1, x+2]

                # Calculate Laplacian
                laplacian = abs(-6.0 * center + top + bottom + left + right + front + back)
                contrast_f32[z, y, x] = laplacian

    # Convert to uint8
    torch.clamp(contrast_f32, 0, 255, out=contrast_f32)
    out.copy_(contrast_f32.to(torch.uint8))

    # Return tensors to cache
    tensor_cache.put(contrast_f32, f32_shape_key)
    tensor_cache.put(padded, padded_shape_key)

    return out

def global_local_contrast_enhancement_3d(volume: torch.Tensor, clip_limit: float = 2.0,
                                         out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Combined global and local contrast enhancement.

    Args:
        volume: Input tensor [D, H, W] as uint8
        clip_limit: Clip limit for adaptive histogram equalization
        out: Optional pre-allocated output tensor

    Returns:
        Enhanced tensor as uint8
    """
    if volume.numel() == 0:
        if out is not None:
            out.zero_()
            return out
        return torch.zeros_like(volume)

    device = volume.device

    # Create output tensor if not provided
    if out is None:
        out = torch.zeros_like(volume)
    elif out.shape != volume.shape:
        out = torch.zeros_like(volume)

    # Pre-allocate tensors for intermediate results
    global_enhanced = tensor_cache.get((volume.shape, torch.uint8, device), key='global_enhanced')
    local_enhanced = tensor_cache.get((volume.shape, torch.uint8, device), key='local_enhanced')
    contrast_global = tensor_cache.get((volume.shape, torch.uint8, device), key='contrast_global')
    contrast_local = tensor_cache.get((volume.shape, torch.uint8, device), key='contrast_local')

    # Apply global histogram equalization
    histogram_equalization_3d(volume, out=global_enhanced)

    # Apply local adaptive histogram equalization
    adaptive_histogram_local_3d(volume, clip_limit, out=local_enhanced)

    # Calculate contrast maps for weighting
    compute_local_contrast_3d(global_enhanced, out=contrast_global)
    compute_local_contrast_3d(local_enhanced, out=contrast_local)

    # Pre-allocate float32 buffers
    result_f32 = tensor_cache.get((volume.shape, torch.float32, device), key='result_f32')

    # Calculate weighted average (optimized for M4 with direct indexing)
    for idx in range(volume.numel()):
        flat_idx = idx
        g_val = float(global_enhanced.reshape(-1)[flat_idx].item())
        l_val = float(local_enhanced.reshape(-1)[flat_idx].item())
        g_weight = float(contrast_global.reshape(-1)[flat_idx].item())
        l_weight = float(contrast_local.reshape(-1)[flat_idx].item())

        # Calculate weighted sum
        sum_weights = max(1.0, g_weight + l_weight)
        result_f32.reshape(-1)[flat_idx] = (g_val * g_weight + l_val * l_weight) / sum_weights

    # Convert to uint8
    torch.clamp(result_f32, 0, 255, out=result_f32)
    torch.round(result_f32, out=result_f32)
    out.copy_(result_f32.to(torch.uint8))

    # Return tensors to cache
    tensor_cache.put(global_enhanced, key='global_enhanced')
    tensor_cache.put(local_enhanced, key='local_enhanced')
    tensor_cache.put(contrast_global, key='contrast_global')
    tensor_cache.put(contrast_local, key='contrast_local')
    tensor_cache.put(result_f32, key='result_f32')

    return out

# ===================================
# COMPLETE PROCESSING PIPELINE
# ===================================

def process_volume(volume: torch.Tensor,
                   denoise_kernel: int = 3,
                   clip_limit: float = 2.0,
                   threshold_low: int = 32,
                   out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Complete processing pipeline for 3D uint8 volume data.

    Args:
        volume: Input 3D tensor as uint8
        denoise_kernel: Size of kernel for denoising operations
        clip_limit: Clip limit for adaptive histogram equalization
        threshold_low: Low threshold for removing noise
        out: Optional pre-allocated output tensor

    Returns:
        Processed volume as uint8
    """
    # Skip processing if volume is empty
    if volume.numel() == 0:
        if out is not None:
            out.zero_()
            return out
        return torch.zeros_like(volume)

    device = volume.device

    # Create output tensor if not provided
    if out is None:
        out = torch.zeros_like(volume)
    elif out.shape != volume.shape:
        out = torch.zeros_like(volume)

    # Pre-allocate intermediate tensors
    temp1 = tensor_cache.get((volume.shape, torch.uint8, device), key='temp1')
    temp2 = tensor_cache.get((volume.shape, torch.uint8, device), key='temp2')

    # Apply threshold to remove low values
    torch.threshold(volume, threshold_low, 0, out=temp1)

    # Apply bit mask (equivalent to &= 0xe0)
    # Efficient bit mask operation for uint8 tensors
    temp1 = (temp1 >> 3) << 3

    # Denoising with average pooling
    avgpool_denoise_3d(temp1, kernel=denoise_kernel, out=temp2)

    # Histogram equalization
    histogram_equalization_3d(temp2, out=temp1)

    # Second denoising pass
    avgpool_denoise_3d(temp1, kernel=denoise_kernel, out=temp2)

    # Z-score normalization
    z_score_normalize_3d(temp2, out=temp1)

    # Contrast enhancement
    global_local_contrast_enhancement_3d(temp1, clip_limit=clip_limit, out=temp2)

    # Final cleanup - zero out low values again
    torch.threshold(temp2, threshold_low, 0, out=out)

    # Return tensors to cache
    tensor_cache.put(temp1, key='temp1')
    tensor_cache.put(temp2, key='temp2')

    return out
