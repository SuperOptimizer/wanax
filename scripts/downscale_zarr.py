import numpy as np
import zarr
import time
from tqdm import tqdm
import concurrent.futures
import numba as nb

def create_downsampled_arrays(root_path, scroll_name="scroll1a", energy="54kev", base_resolution="8um"):
    store = zarr.open(root_path)
    base_array = store[f"{scroll_name}/{energy}/{base_resolution}"]
    base_shape = base_array.shape
    base_chunks = base_array.chunks

    resolutions = ["16um", "32um", "64um", "128um"]

    print(f"Base shape: {base_shape}, chunks: {base_chunks}")

    current_shape = base_shape
    for resolution in resolutions:
        new_shape = tuple(s // 2 for s in current_shape)
        path = f"{scroll_name}/{energy}/{resolution}"

        if path in store:
            print(f"Array {path} already exists. Skipping.")
        else:
            print(f"Creating array {path} with shape {new_shape}")

            creation_args = {
                "shape": new_shape,
                "dtype": np.uint8,
                "compressors": [zarr.codecs.BloscCodec(cname='zstd', clevel=9, shuffle=zarr.codecs.BloscShuffle.shuffle)],
                "dimension_names": ('z','y','x'),
                "shards": (512,512,512),
                "chunks": (256,256,256),
            }

            store.create_array(path, **creation_args)

        current_shape = new_shape

@nb.njit(parallel=False, fastmath=True)
def downsample_chunk_3d(chunk):
    depth, height, width = chunk.shape
    new_depth = depth // 2
    new_height = height // 2
    new_width = width // 2

    result = np.zeros((new_depth, new_height, new_width), dtype=np.uint8)
    for z in nb.prange(new_depth):
        z_offset = z * 2
        for y in range(new_height):
            y_offset = y * 2
            for x in range(new_width):
                x_offset = x * 2

                sum_val =(np.uint16(chunk[z_offset, y_offset, x_offset]) +
                    np.uint16(chunk[z_offset, y_offset, x_offset + 1]) +
                    np.uint16(chunk[z_offset, y_offset + 1, x_offset]) +
                    np.uint16(chunk[z_offset, y_offset + 1, x_offset + 1]) +
                    np.uint16(chunk[z_offset + 1, y_offset, x_offset]) +
                    np.uint16(chunk[z_offset + 1, y_offset, x_offset + 1]) +
                    np.uint16(chunk[z_offset + 1, y_offset + 1, x_offset]) +
                    np.uint16(chunk[z_offset + 1, y_offset + 1, x_offset + 1]))


                # Faster division using bit shift
                result[z, y, x] = sum_val >> 3
    return result


def process_shard(args):
    (z_start, y_start, x_start,
     shard_size, scale_factor,
     root_path, scroll_name, energy,
     base_resolution, target_resolution) = args

    z_end = min(z_start + shard_size, z_start + (shard_size // scale_factor) * scale_factor)
    y_end = min(y_start + shard_size, y_start + (shard_size // scale_factor) * scale_factor)
    x_end = min(x_start + shard_size, x_start + (shard_size // scale_factor) * scale_factor)

    store = zarr.open(root_path)
    base_array = store[f"{scroll_name}/{energy}/{base_resolution}"]
    target_array = store[f"{scroll_name}/{energy}/{target_resolution}"]
    base_chunk = base_array[z_start:z_end, y_start:y_end, x_start:x_end]

    if not np.any(base_chunk > 0):
        return f"Skipped empty chunk at {z_start},{y_start},{x_start}"

    downsampled = downsample_chunk_3d(base_chunk, scale_factor)

    target_z_start = z_start // scale_factor
    target_y_start = y_start // scale_factor
    target_x_start = x_start // scale_factor

    target_z_end = target_z_start + downsampled.shape[0]
    target_y_end = target_y_start + downsampled.shape[1]
    target_x_end = target_x_start + downsampled.shape[2]

    target_array[target_z_start:target_z_end,
    target_y_start:target_y_end,
    target_x_start:target_x_end] = downsampled

    return f"Processed chunk at {z_start},{y_start},{x_start}"

def downsample_resolution(root_path, scroll_name, energy, base_resolution, target_resolution, scale_factor, shard_size=512, max_workers=None):
    start_time = time.time()
    store = zarr.open(root_path)
    base_array = store[f"{scroll_name}/{energy}/{base_resolution}"]
    shape = base_array.shape

    print(f"Downsampling {base_resolution} to {target_resolution} (scale factor: {scale_factor})")

    chunk_coords = []
    for z_start in range(0, shape[0], shard_size):
        for y_start in range(0, shape[1], shard_size):
            for x_start in range(0, shape[2], shard_size):
                chunk_coords.append((
                    z_start, y_start, x_start,
                    shard_size, scale_factor,
                    root_path, scroll_name, energy,
                    base_resolution, target_resolution
                ))

    total_chunks = len(chunk_coords)
    print(f"Processing {total_chunks} chunks...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_shard, chunk_coords),
            total=total_chunks,
            desc=f"Downsampling to {target_resolution}"
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

def create_multiscale_pyramid(root_path="/Volumes/vesuvius/optimized", max_workers=None):
    scroll_name = "scroll1a"
    energy = "54kev"
    base_resolution = "8um"

    create_downsampled_arrays(root_path, scroll_name, energy, base_resolution)

    resolutions = [
        ("8um", "16um", 2),
        ("16um", "32um", 2),
        ("32um", "64um", 2),
        ("64um", "128um", 2)
    ]

    for base_res, target_res, scale in resolutions:
        downsample_resolution(
            root_path=root_path,
            scroll_name=scroll_name,
            energy=energy,
            base_resolution=base_res,
            target_resolution=target_res,
            scale_factor=scale,
            max_workers=max_workers
        )
        print(f"Completed downsampling {base_res} to {target_res}")
        print("=" * 80)

if __name__ == "__main__":
    create_multiscale_pyramid(max_workers=None)