import skimage.exposure
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
import src.preprocessing.sharpening

import numpy as np
from numba import njit, prange

CHUNKS = (512, 512, 512)
SHARDS = (512, 512, 512)
COMPRESSOR = zarr.codecs.BloscCodec(cname='zstd', clevel=9)
DIMNAMES = ('z', 'y', 'x')
DTYPE = np.uint8


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
def median_filter_3d(data, radius=1):
    """Apply a 3D median filter to the data."""
    # Pad the input array to handle edge values
    padded_data = pad_array_edge_3d(data, radius)

    # Initialize result array
    depth, height, width = data.shape
    result = np.zeros_like(data, dtype=np.uint8)

    kernel_size = 2 * radius + 1
    kernel_elements = kernel_size ** 3

    # For each voxel in the original volume
    for z in prange(depth):
        for y in prange(height):
            for x in prange(width):
                # Extract neighborhood
                neighbors = np.zeros(kernel_elements, dtype=np.uint8)
                idx = 0
                for kz in range(-radius, radius + 1):
                    for ky in range(-radius, radius + 1):
                        for kx in range(-radius, radius + 1):
                            pz, py, px = z + radius + kz, y + radius + ky, x + radius + kx
                            neighbors[idx] = padded_data[pz, py, px]
                            idx += 1

                # Find median value
                result[z, y, x] = np.median(neighbors)

    return result


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
                dist_squared = dz * dz + dy * dy + dx * dx
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


def check_shard_exists(root_path, array_name, z_start, y_start, x_start):
    """
    Check if a shard already exists on disk.
    This function examines the directory structure to see if the chunk has already been processed.
    """
    # Calculate chunk indices from coordinate positions
    chunk_z = z_start // CHUNKS[0]
    chunk_y = y_start // CHUNKS[1]
    chunk_x = x_start // CHUNKS[2]

    # Construct the expected path for the chunk
    # Based on your directory structure, chunks appear to be stored in c/z/y/x format
    chunk_path = os.path.join(root_path, array_name, 'c', str(chunk_z), str(chunk_y), str(chunk_x))

    # Check if the chunk directory exists
    return os.path.exists(chunk_path)


def process_shard_worker(z_start, y_start, x_start, shard_size, raw_path, masked_path, root_path, array_name, iso):

    shard_start_time = time.time()

    raw = zarr.open(raw_path)
    root = zarr.open(root_path)
    output_array = root[array_name]

    z_end = min(z_start + shard_size[0], raw.shape[0])
    y_end = min(y_start + shard_size[1], raw.shape[1])
    x_end = min(x_start + shard_size[2], raw.shape[2])

    actual_z_size = z_end - z_start
    actual_y_size = y_end - y_start
    actual_x_size = x_end - x_start
    data_mask = np.ones((actual_z_size,actual_y_size,actual_x_size), dtype=bool)
    if masked_path:
        try:
            masked = zarr.open(masked_path)
            mask_z_start = z_start // 32
            mask_y_start = y_start // 32
            mask_x_start = x_start // 32
            mask_z_end = z_end // 32
            mask_y_end = y_end // 32
            mask_x_end = x_end // 32
            mask_data = masked[mask_z_start:mask_z_end, mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            if np.max(mask_data) == 0:
                return time.time() - shard_start_time, (z_start, y_start, x_start)
            mask = mask_data > 0
            mask_z_offset = mask_z_start * 32 - z_start
            mask_y_offset = mask_y_start * 32 - y_start
            mask_x_offset = mask_x_start * 32 - x_start
            for mz in range(mask.shape[0]):
                for my in range(mask.shape[1]):
                    for mx in range(mask.shape[2]):
                        if mask[mz, my, mx] == 0:
                            block_z_start = max(0, mz * 32 + mask_z_offset)
                            block_y_start = max(0, my * 32 + mask_y_offset)
                            block_x_start = max(0, mx * 32 + mask_x_offset)

                            block_z_end = min(z_end - z_start+1, block_z_start + 32)
                            block_y_end = min(y_end - y_start+1, block_y_start + 32)
                            block_x_end = min(x_end - x_start+1, block_x_start + 32)

                            if (block_z_start < z_end - z_start+1 and
                                    block_y_start < y_end - y_start+1 and
                                    block_x_start < x_end - x_start+1):
                                data_mask[block_z_start:block_z_end,
                                block_y_start:block_y_end,
                                block_x_start:block_x_end] = False
        except:
            pass

    shard_data = raw[z_start:z_end, y_start:y_end, x_start:x_end]
    if shard_data.dtype == np.uint16:
        shard_data = src.preprocessing.normalization.min_max_normalize(shard_data, 0, 255).astype(np.uint8)
    processed_data = shard_data.copy()
    data_mask = np.ones(processed_data.shape, dtype=bool)
    if np.max(shard_data) == 0:
        processed_data = np.zeros((actual_z_size, actual_y_size, actual_x_size), dtype=np.uint8)
        output_array[z_start:z_end, y_start:y_end, x_start:x_end] = processed_data
        return time.time() - shard_start_time, (z_start, y_start, x_start)


    processed_data[~data_mask] = 0
    if iso > 0:
        intensity_mask = processed_data > iso
        if not np.any(intensity_mask):
            processed_data = np.zeros((actual_z_size, actual_y_size, actual_x_size), dtype=np.uint8)
            output_array[z_start:z_end, y_start:y_end, x_start:x_end] = processed_data
            return time.time() - shard_start_time, (z_start, y_start, x_start)
        intensity_mask = binary_dilation_3d_numba(intensity_mask, 2)
        processed_data[~intensity_mask] = 0
    if np.max(processed_data) == 0:
        return time.time() - shard_start_time, (z_start, y_start, x_start)
    processed_data[processed_data < iso] = 0
    if np.max(processed_data) == 0:
        return time.time() - shard_start_time, (z_start, y_start, x_start)
    processed_data = median_filter_3d(processed_data, radius=2)
    processed_data = src.preprocessing.equalization.histogram_equalize_3d_u8(processed_data)
    processed_data = src.preprocessing.glcae.enhance_contrast_3d(processed_data)
    # only really needed for visual
    # processed_data = src.preprocessing.normalization.min_max_normalize(processed_data,127,255).astype(np.uint8)
    # processed_data[processed_data < 128] = 0
    processed_data &= 0xf0

    src.viewer.VolumeViewer(processed_data).run()
    output_array[z_start:z_end, y_start:y_end, x_start:x_end] = processed_data
    processing_time = time.time() - shard_start_time
    return processing_time, (z_start, y_start, x_start)


def process_scroll(root, name, raw_path, masked5_path, energy, resolution, indims, outdims, iso, max_shard_workers=None,
                   shard_timeout=60):  # 60 second timeout
    start_time = time.time()
    array_name = f"{name}/{energy}/{resolution}"

    # Check if array exists, create if not
    try:
        _ = root[array_name]
        print(f"Array {array_name} already exists. Continuing with processing...")
    except KeyError:
        print(f"Creating array {array_name}...")
        root.create_array(array_name, compressors=[COMPRESSOR], chunks=CHUNKS, dtype=DTYPE, shape=outdims,
                          dimension_names=DIMNAMES, shards=SHARDS)

    chunks = []
    skipped_chunks = 0
    for z_start in range(0, indims[0], CHUNKS[0]):
        for y_start in range(0, indims[1], CHUNKS[1]):
            for x_start in range(0, indims[2], CHUNKS[2]):
                # Check if the chunk already exists
                if check_shard_exists(store_path, array_name, z_start, y_start, x_start):
                    print(f"skipping {z_start,y_start,x_start}")
                    skipped_chunks += 1
                    continue
                chunks.append((z_start, y_start, x_start))

    total_shards = len(chunks) + skipped_chunks
    print(f"Total shards: {total_shards}")
    print(f"Shards already processed: {skipped_chunks}")
    print(f"Shards to process: {len(chunks)}")

    if len(chunks) == 0:
        print(f"All shards for {array_name} already processed. Skipping.")
        return

    processed_shards = 0
    successful_shards = 0
    total_processing_time = 0
    timed_out_shards = []

    if max_shard_workers is None:
        max_shard_workers = os.cpu_count()

    print(f"Using {max_shard_workers} workers for processing")
    print(f"Starting processing at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_shard_workers) as executor:
        future_to_coord = {}
        for z_start, y_start, x_start in chunks:
            future = executor.submit(process_shard_worker, z_start, y_start, x_start, CHUNKS, raw_path, masked5_path,
                                     store_path, array_name, iso)
            future_to_coord[future] = (z_start, y_start, x_start)

        for future in concurrent.futures.as_completed(future_to_coord.keys()):
            z_start, y_start, x_start = future_to_coord[future]
            processed_shards += 1

            try:
                processing_time, coords = future.result(timeout=shard_timeout)

                # If processing_time is 0, it means the shard already existed
                if processing_time == 0:
                    print(f"Shard ({z_start},{y_start},{x_start}) already exists. Skipping.")
                    continue

                successful_shards += 1
                total_processing_time += processing_time

                elapsed_time = time.time() - start_time
                shards_per_second = successful_shards / elapsed_time if elapsed_time > 0 else 0
                remaining_shards = len(chunks) - processed_shards
                eta_seconds = remaining_shards / shards_per_second if shards_per_second > 0 else 0
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(
                    f"Processed shard ({z_start},{y_start},{x_start}) in {processing_time:.2f}s - Progress: {processed_shards}/{len(chunks)} - ETA: {eta}")

            except concurrent.futures.TimeoutError:
                print(f"TIMEOUT: Shard ({z_start},{y_start},{x_start}) exceeded {shard_timeout} seconds")


            except Exception as e:
                print(f"ERROR: Processing shard ({z_start},{y_start},{x_start}) failed: {e}")
                timed_out_shards.append((z_start, y_start, x_start))

    end_time = time.time()
    total_time = end_time - start_time

    print("-" * 80)
    print(f"Processing completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Successfully processed: {successful_shards}/{len(chunks)} shards")
    print(f"Total skipped shards (already existed): {skipped_chunks}")


if __name__ == '__main__':
    store_path = "/Volumes/vesuvius/optimized"
    store = zarr.storage.LocalStore(store_path)

    # Open the existing group instead of creating a new one with overwrite=True
    try:
        root = zarr.open_group(store=store, zarr_format=3)
        print("Opened existing Zarr group")
    except zarr.errors.GroupNotFoundError:
        # Create new group if it doesn't exist
        root = zarr.create_group(store=store, zarr_format=3)
        print("Created new Zarr group")

    configs = [
        ("scroll1a", "/Volumes/vesuvius/scroll1a_0", "/Volumes/vesuvius/scroll1a_5_masked/5", "54kev", "7.91um",
         (10532, 7812, 8316), (alignup(10532, 1024), alignup(7812, 1024), alignup(8316, 1024)), 64),
        ("scroll1b", "/Volumes/vesuvius/scroll1b_0", None, "54kev", "7.91um",
         (10532, 7812, 8316), (alignup(10532, 1024), alignup(7812, 1024), alignup(8316, 1024)), 64),
        ("scroll2", "/Volumes/vesuvius/scroll2_0", "/Volumes/vesuvius/scroll2_5_masked", "54kev", "7.91um",
         (14428, 10112, 11984), (alignup(14428, 1024), alignup(10112, 1024), alignup(11984, 1024)), 0),
        ("scroll3", "/Volumes/vesuvius/scroll3_0", None, "54kev", "7.91um",
         (9778, 3550, 3400), (alignup(9778, 1024), alignup(3550, 1024), alignup(3400, 1024)), 64),
        ("scroll4", "/Volumes/vesuvius/scroll4_0", None, "54kev", "7.91um",
         (11174, 3340, 3400), (alignup(11174, 1024), alignup(3340, 1024), alignup(3400, 1024)), 128),
        ("scroll5", "/Volumes/vesuvius/scroll5_0", None, "54kev", "7.91um",
         (21000, 6700, 9100), (alignup(21000, 1024), alignup(6700, 1024), alignup(9100, 1024)), 128),
        # ("fragment1", "/Volumes/vesuvius/frag1_0", None, "54kev", "7.91um",
        # (7219, 1399, 7198), (alignup(7219, 1024), alignup(1399, 1024), alignup(7198, 1024)), 128),
        # ("fragment2", "/Volumes/vesuvius/frag2_0", None, "54kev", "7.91um",
        # (14111, 2288, 9984), (alignup(14111, 1024), alignup(2288, 1024), alignup(9984, 1024)), 32),
        # ("fragment3", "/Volumes/vesuvius/frag3_0", None, "54kev", "7.91um",
        # (6656, 2288, 6312), (alignup(6656, 1024), alignup(1440, 1024), alignup(6312, 1024)), 1),
    ]
    for name, rawpath, maskedpath, energy, resolution, indims, outdims, iso in configs:
        print(f"\nProcessing {name}...")
        process_scroll(root, name, rawpath, maskedpath, energy, resolution, indims, outdims, iso,
                       max_shard_workers=1,
                       shard_timeout=600,
                       )