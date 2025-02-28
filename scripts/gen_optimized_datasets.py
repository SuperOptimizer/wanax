import zarr
import time
from tqdm import tqdm
import datetime
import concurrent.futures

import numpy as np
import numba as nb
import src.viewer
import src.preprocessing.normalization as normalization
import src.preprocessing.glcae as glcae
import src.preprocessing.denoising as denoising
import src.frequency
import src.preprocessing.equalization


#nb.config.THREADING_LAYER_PRIORITY = ["tbb"]
def alignup(number, alignment):
    number = int(number)
    alignment = int(alignment)
    if alignment & (alignment - 1) != 0 or alignment <= 0:
        raise ValueError("Alignment must be a positive power of 2")
    if number % alignment == 0:
        return number
    return (number + alignment - 1) & ~(alignment - 1)


@nb.njit(parallel=True, fastmath=True)
def binary_dilation_3d_numba(input_array, radius, output_array=None):
    depth, height, width = input_array.shape
    if output_array is None:
        output_array = np.zeros_like(input_array)
    radius_squared = radius * radius
    neighborhood = []
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist_squared = dz*dz + dy*dy + dx*dx
                if dist_squared <= radius_squared:
                    neighborhood.append((dz, dy, dx))

    for z in nb.prange(depth):
        for y in nb.prange(height):
            for x in nb.prange(width):
                if input_array[z, y, x] > 0:
                    output_array[z, y, x] = 1

                    for offset_idx in range(len(neighborhood)):
                        dz, dy, dx = neighborhood[offset_idx]
                        nz = z + dz
                        ny = y + dy
                        nx = x + dx

                        is_valid = (0 <= nz < depth) and (0 <= ny < height) and (0 <= nx < width)

                        if is_valid:
                            output_array[nz, ny, nx] = 1

    return output_array


def process_shard(z_start, y_start, x_start, shard_size, raw_path, masked_path):
    """Process a full shard with direct data processing"""
    shard_start_time = time.time()

    # Open data locally in this process
    raw = zarr.open(raw_path)
    masked = zarr.open(masked_path)

    # Calculate actual data boundaries (limited by raw data dimensions)
    z_end = min(z_start + shard_size[0], raw.shape[0])
    y_end = min(y_start + shard_size[1], raw.shape[1])
    x_end = min(x_start + shard_size[2], raw.shape[2])

    # Calculate dimensions of data we'll actually process
    actual_z_size = z_end - z_start
    actual_y_size = y_end - y_start
    actual_x_size = x_end - x_start

    # Extract full shard data (load once)
    shard_data = raw[z_start:z_end, y_start:y_end, x_start:x_end]

    # Skip empty shards
    if np.max(shard_data) == 0:
        # Create zero array with expected shard size for writing
        result = np.zeros(shard_size, dtype=np.uint8)
        return result, 0

    # Calculate corresponding mask coordinates (32x downsampling)
    mask_z_start = max(0, (z_start // 32) - 1)
    mask_y_start = max(0, (y_start // 32) - 1)
    mask_x_start = max(0, (x_start // 32) - 1)
    mask_z_end = min(masked.shape[0], ((z_end + 31) // 32) + 1)
    mask_y_end = min(masked.shape[1], ((y_end + 31) // 32) + 1)
    mask_x_end = min(masked.shape[2], ((x_end + 31) // 32) + 1)

    # Get the mask data for this shard
    mask_data = masked[mask_z_start:mask_z_end, mask_y_start:mask_y_end, mask_x_start:mask_x_end]
    mask_data = (normalization.z_score_normalize(mask_data)*255).astype(np.uint8)
    mask = mask_data > 64

    # Create a copy of the shard data to process
    processed_data = shard_data.copy()

    # Set up a mask for the processed data (all True initially)
    data_mask = np.ones(processed_data.shape, dtype=bool)

    # Apply the downsampled mask to the data mask
    # Each mask voxel corresponds to a 32³ block in the processed data
    mask_z_offset = mask_z_start * 32 - z_start
    mask_y_offset = mask_y_start * 32 - y_start
    mask_x_offset = mask_x_start * 32 - x_start

    # Efficiently apply mask across the entire shard at once
    for mz in range(mask.shape[0]):
        for my in range(mask.shape[1]):
            for mx in range(mask.shape[2]):
                if mask[mz, my, mx] == 0:
                    # Calculate corresponding block in shard data
                    block_z_start = max(0, mz * 32 + mask_z_offset)
                    block_y_start = max(0, my * 32 + mask_y_offset)
                    block_x_start = max(0, mx * 32 + mask_x_offset)

                    block_z_end = min(processed_data.shape[0], block_z_start + 32)
                    block_y_end = min(processed_data.shape[1], block_y_start + 32)
                    block_x_end = min(processed_data.shape[2], block_x_start + 32)

                    # Check if block coordinates are valid
                    if (block_z_start < processed_data.shape[0] and
                            block_y_start < processed_data.shape[1] and
                            block_x_start < processed_data.shape[2]):
                        data_mask[block_z_start:block_z_end,
                        block_y_start:block_y_end,
                        block_x_start:block_x_end] = False

    # Apply the mask from downsampled data
    processed_data[~data_mask] = 0

    # Create binary mask for active regions
    intensity_mask = processed_data > 128
    if not np.any(intensity_mask):
        # Return zero array with FULL shard size
        return np.zeros(shard_size, dtype=np.uint8), time.time() - shard_start_time

    # Dilate the mask to include nearby regions
    intensity_mask = binary_dilation_3d_numba(intensity_mask, 4)
    processed_data[~intensity_mask] = 0

    # Apply processing pipeline to the entire shard at once
    processed_data = src.preprocessing.equalization.histogram_equalize_3d_u8(processed_data)
    processed_data &= 0xf0
    processed_data = src.frequency.wavelet_denoise_3d_optimized(processed_data)
    processed_data = (normalization.robust_normalize(processed_data)*255).astype(np.uint8)
    processed_data[~intensity_mask] = 0
    processed_data = glcae.enhance_contrast_3d(processed_data)
    processed_data &= 0xf0

    # Create result with the proper size - ALWAYS use the full shard_size
    result = np.zeros(shard_size, dtype=np.uint8)
    # Only copy the actual processed data into the properly sized result
    result[:actual_z_size, :actual_y_size, :actual_x_size] = processed_data

    processing_time = time.time() - shard_start_time
    return result, processing_time


def process_scroll1a(shard_parallel=False, max_shard_workers=None):
    start_time = time.time()
    store_path = "/Volumes/vesuvius/optimized2"
    store = zarr.storage.LocalStore(store_path)
    root = zarr.create_group(store=store, zarr_format=3, overwrite=True)
    compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3)

    shard_size = (256, 256, 256)
    chunk_size = (256, 256, 256)  # Keep for output array chunking

    masked_path = "/Volumes/vesuvius/scroll1a_5_masked/5"
    raw_path = "/Volumes/vesuvius/scroll1a_0"
    scroll_name = "scroll1a"
    energy = "54kev"

    # Get dimensions (we still need to open the files briefly to get shapes)
    masked = zarr.open(masked_path)
    masked_shape = masked.shape
    raw = zarr.open(raw_path)
    raw_shape = raw.shape

    # Close them immediately (we'll reopen in each process)
    del masked
    del raw

    shape = (
        alignup(14376, 1024),
        alignup(7888, 1024),
        alignup(8096, 1024)
    )

    print(f"Output shape: {shape}")
    print(f"Masked shape: {masked_shape}")
    print(f"Raw shape: {raw_shape}")

    # Create output array
    output_array = root.create_array(
        f"{scroll_name}/{energy}/8um",
        compressors=[compressor],
        chunks=chunk_size,
        dtype=np.uint8,
        shape=shape,
        shards=shard_size,
        dimension_names=('z','y','x')
    )

    # Generate shard coordinates
    shard_coords = []
    for z_start in range(0, shape[0], shard_size[0]):
        for y_start in range(0, shape[1], shard_size[1]):
            for x_start in range(0, shape[2], shard_size[2]):
                shard_coords.append((z_start, y_start, x_start))

    total_shards = len(shard_coords)
    print(f"Total shards to process: {total_shards}")

    processed_shards = 0
    total_processing_time = 0

    if not shard_parallel or max_shard_workers == 1:
        # Single-threaded shard processing
        for z_start, y_start, x_start in tqdm(shard_coords, total=total_shards, desc="Processing shards"):
            processed_shard, processing_time = process_shard(
                z_start, y_start, x_start, shard_size, raw_path, masked_path
            )

            # Write the shard to output
            z_end = min(z_start + shard_size[0], shape[0])
            y_end = min(y_start + shard_size[1], shape[1])
            x_end = min(x_start + shard_size[2], shape[2])

            # Use the processed shard directly without additional slicing
            output_array[z_start:z_end, y_start:y_end, x_start:x_end] = processed_shard[:z_end-z_start, :y_end-y_start, :x_end-x_start]

            processed_shards += 1
            total_processing_time += processing_time

            # Calculate progress metrics
            elapsed_time = time.time() - start_time
            shards_per_second = processed_shards / elapsed_time if elapsed_time > 0 else 0
            remaining_shards = total_shards - processed_shards
            eta_seconds = remaining_shards / shards_per_second if shards_per_second > 0 else 0
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))

            print(f"Completed shard ({z_start},{y_start},{x_start}) in {processing_time:.2f}s")
            print(f"Progress: {processed_shards}/{total_shards} shards ({processed_shards/total_shards*100:.2f}%)")
            print(f"Speed: {shards_per_second:.3f} shards/sec, ETA: {eta}")
    else:
        # Multi-threaded shard processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_shard_workers) as executor:
            future_to_shard = {}
            for z_start, y_start, x_start in shard_coords:
                future = executor.submit(
                    process_shard,
                    z_start, y_start, x_start, shard_size, raw_path, masked_path
                )
                future_to_shard[future] = (z_start, y_start, x_start)

            # Create a progress bar
            pbar = tqdm(total=total_shards, desc="Processing shards")

            for future in concurrent.futures.as_completed(future_to_shard):
                z_start, y_start, x_start = future_to_shard[future]
                processed_shard, processing_time = future.result()

                # Write the shard to output
                z_end = min(z_start + shard_size[0], shape[0])
                y_end = min(y_start + shard_size[1], shape[1])
                x_end = min(x_start + shard_size[2], shape[2])

                # Use the processed shard directly without additional slicing
                output_array[z_start:z_end, y_start:y_end, x_start:x_end] = processed_shard[:z_end-z_start, :y_end-y_start, :x_end-x_start]

                processed_shards += 1
                total_processing_time += processing_time

                # Calculate progress metrics
                elapsed_time = time.time() - start_time
                shards_per_second = processed_shards / elapsed_time if elapsed_time > 0 else 0
                remaining_shards = total_shards - processed_shards
                eta_seconds = remaining_shards / shards_per_second if shards_per_second > 0 else 0
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                # Update progress bar
                pbar.set_postfix({
                    'shard': f"({z_start},{y_start},{x_start})",
                    'time': f"{processing_time:.2f}s",
                    'speed': f"{shards_per_second:.3f}/s",
                    'ETA': eta
                })
                pbar.update(1)

            pbar.close()

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    # Configure parallelism:
    # - shard_parallel: process multiple shards in parallel
    # - max_shard_workers: number of concurrent shard processes (None = auto)
    process_scroll1a(
        shard_parallel=True,
        max_shard_workers=None,  # Limit workers to reduce memory pressure
    )