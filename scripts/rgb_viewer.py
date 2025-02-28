import numpy as np
import zarr
import src.viewer



if __name__ == "__main__":

    root = zarr.open("/Volumes/vesuvius/optimized")
    chunk = root["scroll1a/54kev/8um"][2048:3072,2048:3072,2048:3072]
    src.viewer.VolumeViewer(chunk).run()
