import numpy as np
import zarr
import time
from tqdm import tqdm
import concurrent.futures
import numba as nb

def create_mask_arrays(root_path, output_path, scroll_name="scroll1a", energy="54kev"):
    """
    Create empty mask arrays with the same structure as the original arrays.
    """
    input_store = zarr.open(root_path)
    output_store = zarr.open(output_path)

    # Get all available resolutions
    resolutions = []
    for item in input_store[f"{scroll_name}/{energy}"].array_keys():
        resolutions.append(item)

    print(f"Found resolutions: {resolutions}")

    # Create mask arrays for each resolution
    for resolution in resolutions:
        input_array = input_store[f"{scroll_name}/{energy}/{resolution}"]
        input_shape = input_array.shape
        input_chunks = input_array.chunks

        path = f"{scroll_name}/{energy}/{resolution}"

        if path in output_store:
            print(f"Mask array {path} already exists. Skipping.")
        else:
            print(f"Creating mask array {path} with shape {input_shape}")

            creation_args = {
                "shape": input_shape,
                "dtype": np.uint8,
                "compressors": [zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)],
                "dimension_names": ('z','y','x'),
                "shards": input_array.attrs.get('shards', (256, 256, 256)),
                "chunks": input_chunks,
            }

            output_store.create_array(path, **creation_args)


@nb.njit(parallel=True, fastmath=True)
def create_mask_chunk(chunk):
    """
    Create a mask where non-zero values are set to 255 and zero values remain 0.
    """
    mask = np.zeros(chunk.shape, dtype=np.uint8)
    depth, height, width = chunk.shape

    for z in nb.prange(depth):
        for y in range(height):
            for x in range(width):
                if chunk[z, y, x] > 0:
                    mask[z, y, x] = 255

    return mask


def process_shard_mask(args):
    """
    Process a shard of the input data to create the corresponding mask.
    """
    (z_start, y_start, x_start,
     shard_size,
     root_path, output_path, scroll_name, energy,
     resolution) = args

    z_end = min(z_start + shard_size, z_start + shard_size)
    y_end = min(y_start + shard_size, y_start + shard_size)
    x_end = min(x_start + shard_size, x_start + shard_size)

    input_store = zarr.open(root_path)
    output_store = zarr.open(output_path)

    input_array = input_store[f"{scroll_name}/{energy}/{resolution}"]
    output_array = output_store[f"{scroll_name}/{energy}/{resolution}"]

    # Read the input chunk
    input_chunk = input_array[z_start:z_end, y_start:y_end, x_start:x_end]

    # Skip if the chunk is all zeros
    if not np.any(input_chunk > 0):
        return f"Skipped empty chunk at {z_start},{y_start},{x_start}"

    # Create the mask
    mask_chunk = create_mask_chunk(input_chunk)

    # Write the mask to the output array
    output_array[z_start:z_end, y_start:y_end, x_start:x_end] = mask_chunk

    return f"Processed chunk at {z_start},{y_start},{x_start}"


def create_resolution_mask(root_path, output_path, scroll_name, energy, resolution, shard_size=256, max_workers=None):
    """
    Create a mask for a specific resolution.
    """
    start_time = time.time()
    input_store = zarr.open(root_path)
    input_array = input_store[f"{scroll_name}/{energy}/{resolution}"]
    shape = input_array.shape

    print(f"Creating mask for {resolution} with shape {shape}")

    chunk_coords = []
    for z_start in range(0, shape[0], shard_size):
        for y_start in range(0, shape[1], shard_size):
            for x_start in range(0, shape[2], shard_size):
                chunk_coords.append((
                    z_start, y_start, x_start,
                    shard_size,
                    root_path, output_path, scroll_name, energy,
                    resolution
                ))

    total_chunks = len(chunk_coords)
    print(f"Processing {total_chunks} chunks...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_shard_mask, chunk_coords),
            total=total_chunks,
            desc=f"Creating mask for {resolution}"
        ))

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f}s, avg: {elapsed/total_chunks:.2f}s per chunk")

    successes = sum(1 for r in results if not r.startswith("Error") and not r.startswith("Skipped"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    errors = sum(1 for r in results if r.startswith("Error"))

    print(f"Processed: {successes}, Skipped: {skipped}, Errors: {errors}")

    for result in results:
        if result.startswith("Error"):
            print(result)


def create_mask_pyramid(input_path="/Volumes/vesuvius/optimized",
                        output_path="/Volumes/vesuvius/masks",
                        max_workers=None):
    """
    Create masks for all resolutions in the pyramid.
    """
    scroll_name = "scroll1a"
    energy = "54kev"

    # Create the mask arrays structure
    create_mask_arrays(input_path, output_path, scroll_name, energy)

    # Get all available resolutions
    input_store = zarr.open(input_path)
    resolutions = []
    for item in input_store[f"{scroll_name}/{energy}"].array_keys():
        resolutions.append(item)

    # Process each resolution
    for resolution in resolutions:
        create_resolution_mask(
            root_path=input_path,
            output_path=output_path,
            scroll_name=scroll_name,
            energy=energy,
            resolution=resolution,
            max_workers=max_workers
        )
        print(f"Completed creating mask for {resolution}")
        print("=" * 80)


if __name__ == "__main__":
    create_mask_pyramid(
        input_path="/Volumes/vesuvius/optimized",
        output_path="/Volumes/vesuvius/masks",
        max_workers=4
    )