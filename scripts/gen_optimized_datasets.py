import numpy as np
from skimage import exposure, filters, restoration, transform
from scipy import ndimage
import zarr
import requests
import blosc2
from typing import Tuple, Optional
import napari
import skimage
import numpy as np
from skimage import exposure, filters, restoration, transform
from scipy import ndimage
import zarr
import requests
import blosc2
from typing import Tuple, Optional
import napari
import skimage
import concurrent.futures
import logging


import src.preprocessing.glcae as glcae
import src.preprocessing.normalization as normalization
import src.preprocessing.denoising as denoising
import src.viewer as viewer
import src.frequency as frequency
import src.feature_detection as feature_detection
import src.preprocessing.sharpening as sharpening
import src.morphological as morphological
import src.preprocessing.resizing as resizing


RESOLUTION_SPECS = {
    4: {'dtype': np.uint8, 'max_val': 31},
    8: {'dtype': np.uint8, 'max_val': 255},
    16: {'dtype': np.uint16, 'max_val': 2048-1},
    32: {'dtype': np.uint16, 'max_val': 16384-1},
    64: {'dtype': np.uint32, 'max_val': 65535*2-1},
    128: {'dtype': np.uint32, 'max_val': 524288*2-1}
}




# 256 x 256 x 256 chunks instead of 128 x 128 x 128 to save a bit on the amount of files we are creating
# scrolls 1 and 2 are masked in the zarr
# scrolls 3 and 4 have per slice png masks
# scroll 5 needs a mask
# fragments 1 2 and 4 have zeroed out backgrounds
# fragments 3 5 and 6 need masks
# dimensions are padded up to the next multiple of 256 but the volume isnt translated at all

# /source/energy/resolution
# /scroll1a/54kev/8um 16um 32um 64um 128um
# /scroll1b/54kev/8um 16um 32um 64um 128um
# /scroll2a/54kev/8um 16um 32um 64um 128um
# /scroll2b/54kev/8um 16um 32um 64um 128um
# /scroll3/53kev/4um 8um 16um 32um 64um 128um
# /scroll3/70kev/4um 8um 16um 32um 64um 128um
# /scroll4/53kev/8um 16um 32um 64um 128um
# /scroll4/88kev/4um 8um 16um 32um 64um 128um
# /scroll5/53kev/8um 16um 32um 64um 128um
# /frag1/54kev/4um 8um 16um 32um 64um 128um
# /frag1/88kev/4um 8um 16um 32um 64um 128um
# /frag2/54kev/4um 8um 16um 32um 64um 128um
# /frag2/88kev/4um 8um 16um 32um 64um 128um
# /frag3/54kev/4um 8um 16um 32um 64um 128um
# /frag3/88kev/4um 8um 16um 32um 64um 128um
# /frag4/54kev/4um 8um 16um 32um 64um 128um
# /frag4/88kev/4um 8um 16um 32um 64um 128um
# /frag5/70kev/4um 8um 16um 32um 64um 128um
# /frag6/53kev/4um 8um 16um 32um 64um 128um
# /frag6/70kev/4um 8um 16um 32um 64um 128um
# /frag6/88kev/4um 8um 16um 32um 64um 128um



def alignup(number, alignment):
    number = int(number)
    alignment = int(alignment)
    if alignment & (alignment - 1) != 0 or alignment <= 0:
        raise ValueError("Alignment must be a positive power of 2")
    if number % alignment == 0:
        return number
    return (number + alignment - 1) & ~(alignment - 1)

def process_volume(volume):
    vol = volume.copy()
    vol &= 0xf0
    vol[vol < 32] = 0
    vol = denoising.clean_volume(vol)
    vol = denoising.avgpool_denoise_3d(vol,kernel=5)
    vol = normalization.robust_normalize(vol)
    vol = skimage.exposure.equalize_hist(vol,mask=vol>0)
    vol[vol < .1] = 0.05
    vol[vol > .9] = 0.95
    vol = skimage.filters.unsharp_mask(vol)
    vol[vol < .1] = 0.05
    vol[vol > .9] = 0.95
    vol = normalization.min_max_normalize(vol)
    vol = glcae.global_and_local_contrast_enhancement((vol*255).astype(np.uint8))
    return vol

# New function to process a single chunk and write directly to output
def process_chunk(raw, masked, output_zarr, z, y, x, scale=7.91/8):
    try:
        # Calculate chunk boundaries with padding
        zstart = z-32
        zend = z + 128 + 32
        ystart = y-32
        yend = y + 128 + 32
        xstart = x-32
        xend = x + 128 + 32

        # Get mask and check if empty
        mask = masked[zstart//32: zend//32, ystart//32: yend//32, xstart//32: xend//32] > 0
        if not np.any(mask):
            logging.info(f"Skipping empty chunk at {z},{y},{x}")
            return z, y, x

        # Get chunk with padding
        chunk = raw[zstart: zend, ystart: yend, xstart: xend]

        # Apply mask
        for z_ in range(mask.shape[0]):
            for y_ in range(mask.shape[1]):
                for x_ in range(mask.shape[2]):
                    if not mask[z_,y_,x_]:
                        chunk[z_*32:(z_+1)*32, y_*32:(y_+1)*32, x_*32:(x_+1)*32] = 0

        # Remove some padding and process
        chunk = chunk[24:24+128+8, 24:24+128+8, 24:24+128+8]
        processed = process_volume(chunk)

        # Remove remaining padding
        chunk = chunk[8:8+128, 8:8+128, 8:8+128]
        processed = processed[8:8+128, 8:8+128, 8:8+128]

        # Scale the processed volume
        processed = resizing.scale_volume_optimized(processed, scale)
        processed[processed < 32] = 0

        # Determine output shape based on scale
        if scale == 7.91/8:
            new_shape = (127, 127, 127)
        elif scale == 3.24/4:
            new_shape = (104, 104, 104)
        else:
            raise ValueError(f"Unsupported scale: {scale}")

        # Create output chunk
        out_chunk = np.zeros(shape=new_shape, dtype=np.uint8)
        for z_ in range(new_shape[0]):
            for y_ in range(new_shape[1]):
                for x_ in range(new_shape[2]):
                    out_chunk[z_, y_, x_] = processed[z_, y_, x_]

        # Calculate output coordinates
        zoutstart = int(scale*z)
        zoutend = zoutstart + new_shape[0]
        youtstart = int(scale*y)
        youtend = youtstart + new_shape[1]
        xoutstart = int(scale*x)
        xoutend = xoutstart + new_shape[2]

        # Write directly to output zarr
        output_zarr[zoutstart:zoutend, youtstart:youtend, xoutstart:xoutend] = out_chunk

        logging.info(f"Processed chunk at {z},{y},{x}")
        return z, y, x
    except Exception as e:
        logging.error(f"Error processing chunk at {z},{y},{x}: {str(e)}")
        raise

def scroll1a(root):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    scale = 7.91/8
    masked = zarr.open("/Volumes/vesuvius/scroll1a_5_masked/5")
    raw = zarr.open("/Volumes/vesuvius/scroll1a_0")
    output_zarr = root["scroll1a/54kev/8um"]

    # Create a list to store coordinates for all valid chunks
    chunk_coords = []

    # First scan to collect coordinates of non-empty chunks
    logging.info("Scanning for non-empty chunks using masked volume...")
    # The masked volume is 32x smaller in each dimension
    masked_vol = masked[:,:,:]
    for mz in range(4, masked.shape[0]-4):
        for my in range(4, masked.shape[1]-4):
            for mx in range(4, masked.shape[2]-4):
                # Check if this voxel in the masked volume is non-zero
                if masked_vol[mz, my, mx] > 0:
                    # Convert masked coordinates to raw coordinates (32x scaling)
                    z = mz * 32
                    y = my * 32
                    x = mx * 32
                    # Round to nearest 128 block for processing
                    z = (z // 128) * 128
                    y = (y // 128) * 128
                    x = (x // 128) * 128
                    # Only add if within valid range for raw
                    if (128 <= z < raw.shape[0]-128 and
                            128 <= y < raw.shape[1]-128 and
                            128 <= x < raw.shape[2]-128):
                        chunk_coords.append((z, y, x))

    # Remove duplicates (multiple masked voxels might map to same raw chunk)
    chunk_coords = list(set(chunk_coords))
    logging.info(f"Found {len(chunk_coords)} non-empty chunks to process")

    logging.info(f"Found {len(chunk_coords)} non-empty chunks to process")

    # Process chunks in parallel using futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks
        future_to_coords = {
            executor.submit(process_chunk, raw, masked_vol, output_zarr, z, y, x, scale): (z, y, x)
            for z, y, x in chunk_coords
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_coords):
            z, y, x = future_to_coords[future]
            try:
                result_z, result_y, result_x = future.result()
                logging.info(f"Completed chunk at {result_z},{result_y},{result_x}")
            except Exception as exc:
                logging.error(f"Chunk at {z},{y},{x} generated an exception: {exc}")

    logging.info("All chunks processed")
    logging.info(f"Masked shape: {masked.shape}")
    logging.info(f"Raw shape: {raw.shape}")



if __name__ == "__main__":
    store = zarr.storage.LocalStore("/Volumes/vesuvius/optimized")
    root = zarr.create_group(store=store, zarr_format=3, overwrite=True)
    compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=9, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    chunks = (128,128,128)
    shards = (1024,1024,1024)
    dimension_names = ('z','y','x')

    #scroll1a
    shape = (alignup(14376*(7.91/8),1024),  alignup(7888*(7.91/8),1024), alignup(8096*(7.91/8),1024))
    root.create_array("scroll1a/54kev/8um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint8,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll1a/54kev/16um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll1a/54kev/32um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll1a/54kev/64um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll1a/54kev/128um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)
    scroll1a(root)

    #scroll1b
    shape = (alignup(10532*(7.91/8),1024),  alignup(7812*(7.91/8),1024), alignup(8316*(7.91/8),1024))
    root.create_array("scroll1b/54kev/8um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint8,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll1b/54kev/16um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll1b/54kev/32um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll1b/54kev/64um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll1b/54kev/128um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)

    #scroll2a
    shape = (alignup(14428*(7.91/8),1024),  alignup(10112*(7.91/8),1024), alignup(11984*(7.91/8),1024))
    root.create_array("scroll2a/54kev/8um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint8,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll2a/54kev/16um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll2a/54kev/32um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll2a/54kev/64um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll2a/54kev/128um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)

    #scroll3
    shape = (alignup(9778*(7.91/8),1024),  alignup(3550*(7.91/8),1024), alignup(3400*(7.91/8),1024))
    root.create_array("scroll3/53kev/8um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint8,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll3/53kev/16um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll3/53kev/32um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll3/53kev/64um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll3/53kev/128um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)

    #scroll4
    shape = (alignup(11174*(7.91/8),1024),  alignup(3340*(7.91/8),1024), alignup(3440*(7.91/8),1024))
    root.create_array("scroll4/53kev/8um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint8,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll4/53kev/16um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll4/53kev/32um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll4/53kev/64um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll4/53kev/128um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)

    #scroll5
    shape = (alignup(21000*(7.91/8),1024),  alignup(6700*(7.91/8),1024), alignup(9100*(7.91/8),1024))
    root.create_array("scroll5/53kev/8um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint8,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll5/53kev/16um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll5/53kev/32um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint16,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll5/53kev/64um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)
    shape = shape[0]//2,shape[1]//2,shape[2]//2
    root.create_array("scroll5/53kev/128um",compressor=compressor,chunks=chunks,shards=shards,dimension_names=dimension_names,dtype=np.uint32,shape=shape)


    #base_url = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/"
    #chunk_idx = (50, 41, 41)
    #chunk = fetch_volume_chunk(base_url, *chunk_idx)
    #processed = process_volume(chunk)
    #viewer.MultiVolumeViewer([processed,chunk]).run()

