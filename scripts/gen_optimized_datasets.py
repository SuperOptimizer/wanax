import zarr
import time
import os
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


@nb.njit(parallel=False, fastmath=True)
def binary_dilation_3d_numba(input_array, radius, output_array=None):
    depth, height, width = input_array.shape
    if output_array is None:
        output_array = np.zeros_like(input_array)
    radius_squared = radius * radius
    neighborhood = []
    for dz in nb.prange(-radius, radius + 1):
        for dy in nb.prange(-radius, radius + 1):
            for dx in nb.prange(-radius, radius + 1):
                dist_squared = dz*dz + dy*dy + dx*dx
                if dist_squared <= radius_squared:
                    neighborhood.append((dz, dy, dx))

    for z in nb.prange(depth):
        for y in nb.prange(height):
            for x in nb.prange(width):
                if input_array[z, y, x] > 0:
                    output_array[z, y, x] = 1

                    for offset_idx in nb.prange(len(neighborhood)):
                        dz, dy, dx = neighborhood[offset_idx]
                        nz = z + dz
                        ny = y + dy
                        nx = x + dx

                        is_valid = (0 <= nz < depth) and (0 <= ny < height) and (0 <= nx < width)

                        if is_valid:
                            output_array[nz, ny, nx] = 1

    return output_array


def process_shard_worker(z_start, y_start, x_start, shard_size, raw_path, masked_path, root_path):
    """Process a shard and write directly to the output array"""
    shard_start_time = time.time()

    raw = zarr.open(raw_path)
    masked = zarr.open(masked_path)
    root = zarr.open(root_path)
    output_array = root['scroll1a/54kev/8um']

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
        # Create zero array for output
        processed_data = np.zeros((actual_z_size, actual_y_size, actual_x_size), dtype=np.uint8)
        # Write directly to output and return processing time
        output_array[z_start:z_end, y_start:y_end, x_start:x_end] = processed_data
        return time.time() - shard_start_time, (z_start, y_start, x_start)

    # Calculate corresponding mask coordinates (32x downsampling)
    mask_z_start = z_start // 32
    mask_y_start = y_start // 32
    mask_x_start = x_start // 32
    mask_z_end = z_end // 32
    mask_y_end = y_end // 32
    mask_x_end = x_end // 32

    # Get the mask data for this shard
    mask_data = masked[mask_z_start:mask_z_end, mask_y_start:mask_y_end, mask_x_start:mask_x_end]
    mask = mask_data > 0

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
        # Create zero array and write directly to output
        processed_data = np.zeros((actual_z_size, actual_y_size, actual_x_size), dtype=np.uint8)
        output_array[z_start:z_end, y_start:y_end, x_start:x_end] = processed_data
        return time.time() - shard_start_time, (z_start, y_start, x_start)

    # Dilate the mask to include nearby regions
    intensity_mask = binary_dilation_3d_numba(intensity_mask, 6)
    processed_data[~intensity_mask] = 0

    # Apply processing pipeline to the entire shard at once
    processed_data = src.preprocessing.equalization.histogram_equalize_3d_u8(processed_data)
    processed_data &= 0xf0
    processed_data = src.frequency.wavelet_denoise_3d_optimized(processed_data)
    processed_data = (normalization.robust_normalize(processed_data)*255).astype(np.uint8)
    processed_data[~intensity_mask] = 0
    processed_data = glcae.enhance_contrast_3d(processed_data)
    processed_data &= 0xf0
    #src.viewer.VolumeViewer(processed_data).run()
    # Write directly to output array
    output_array[z_start:z_end, y_start:y_end, x_start:x_end] = processed_data


    processing_time = time.time() - shard_start_time
    return processing_time, (z_start, y_start, x_start)


def process_scroll1a(max_shard_workers=None, shard_timeout=60):  # 60 second timeout
    start_time = time.time()
    store_path = "/Volumes/vesuvius/optimized2"
    store = zarr.storage.LocalStore(store_path)
    root = zarr.create_group(store=store, zarr_format=3, overwrite=False)
    compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3)

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


    shape = (
        alignup(14376, 1024),
        alignup(7888, 1024),
        alignup(8096, 1024)
    )

    print(f"Output shape: {shape}")
    print(f"Masked shape: {masked_shape}")
    print(f"Raw shape: {raw_shape}")
    print(f"Shard timeout set to: {shard_timeout} seconds")

    # Create output array
    root.create_array(
        f"{scroll_name}/{energy}/8um",
        compressors=[compressor],
        chunks=chunk_size,
        dtype=np.uint8,
        shape=shape,
        dimension_names=('z','y','x')
    )

    # Generate shard coordinates
    shard_coords = []
    for z_start in range(0, shape[0], chunk_size[0]):
        for y_start in range(0, shape[1], chunk_size[1]):
            for x_start in range(0, shape[2], chunk_size[2]):
                shard_coords.append((z_start, y_start, x_start))

    total_shards = len(shard_coords)
    print(f"Total shards to process: {total_shards}")

    processed_shards = 0
    successful_shards = 0
    total_processing_time = 0
    timed_out_shards = []

    # Use ProcessPoolExecutor for parallel processing
    if max_shard_workers is None:
        max_shard_workers = os.cpu_count()

    print(f"Using {max_shard_workers} workers for processing")
    print(f"Starting processing at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_shard_workers) as executor:
        # Submit all tasks at once and track them with a dictionary
        future_to_coord = {}
        for z_start, y_start, x_start in shard_coords:
            future = executor.submit(
                process_shard_worker,
                z_start, y_start, x_start,
                chunk_size,
                raw_path,
                masked_path,
                store_path
            )
            future_to_coord[future] = (z_start, y_start, x_start)

        # Process results as they complete with timeout
        for future in concurrent.futures.as_completed(future_to_coord.keys()):
            z_start, y_start, x_start = future_to_coord[future]
            processed_shards += 1

            try:
                # Apply timeout to the future
                processing_time, coords = future.result(timeout=shard_timeout)

                successful_shards += 1
                total_processing_time += processing_time

                # Calculate progress metrics
                elapsed_time = time.time() - start_time
                shards_per_second = successful_shards / elapsed_time if elapsed_time > 0 else 0
                remaining_shards = total_shards - processed_shards
                eta_seconds = remaining_shards / shards_per_second if shards_per_second > 0 else 0
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                # Print progress
                print(f"Processed shard ({z_start},{y_start},{x_start}) in {processing_time:.2f}s - Progress: {processed_shards}/{total_shards} - ETA: {eta}")

            except concurrent.futures.TimeoutError:
                # Handle timeout case
                timed_out_shards.append((z_start, y_start, x_start))
                print(f"TIMEOUT: Shard ({z_start},{y_start},{x_start}) exceeded {shard_timeout} seconds")

                # Create empty data for timed out shard
                # This ensures we don't have missing data in the output array
                z_end = min(z_start + chunk_size[0], raw_shape[0])
                y_end = min(y_start + chunk_size[1], raw_shape[1])
                x_end = min(x_start + chunk_size[2], raw_shape[2])

                actual_z_size = z_end - z_start
                actual_y_size = y_end - y_start
                actual_x_size = x_end - x_start

                # Create a temporary zero array to fill the space
                empty_data = np.zeros((actual_z_size, actual_y_size, actual_x_size), dtype=np.uint8)

                # Open the root storage directly
                root = zarr.open(store_path)
                output_array = root['scroll1a/54kev/8um']

                # Write the empty data to the output array
                output_array[z_start:z_end, y_start:y_end, x_start:x_end] = empty_data

            except Exception as e:
                # Handle other exceptions
                print(f"ERROR: Processing shard ({z_start},{y_start},{x_start}) failed: {str(e)}")
                timed_out_shards.append((z_start, y_start, x_start))

    end_time = time.time()
    total_time = end_time - start_time

    print("-" * 80)
    print(f"Processing completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Successfully processed: {successful_shards}/{total_shards} shards")

    if timed_out_shards:
        print(f"Timed out shards: {len(timed_out_shards)}")
        print("First 10 timed out shard coordinates:")
        for i, coords in enumerate(timed_out_shards[:10]):
            print(f"  {coords}")

        if len(timed_out_shards) > 10:
            print(f"  ... and {len(timed_out_shards) - 10} more")

        # Save timed out shard information
        timeout_log_file = f"timed_out_shards_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(timeout_log_file, 'w') as f:
            f.write(f"# Timed out shards from run at {datetime.datetime.now()}\n")
            f.write(f"# Total run time: {total_time:.2f} seconds\n")
            f.write(f"# Timeout threshold: {shard_timeout} seconds\n")
            for coords in timed_out_shards:
                f.write(f"{coords[0]},{coords[1]},{coords[2]}\n")

        print(f"Timed out shard coordinates saved to {timeout_log_file}")


if __name__ == '__main__':
    # Configure parallelism and timeout:
    # - max_shard_workers: number of concurrent shard processes (None = auto)
    # - shard_timeout: maximum time in seconds for a shard to process before being killed
    process_scroll1a(
        max_shard_workers=None,  # Limit workers to reduce memory pressure
        shard_timeout=60,     # 60 seconds timeout per shard
    )