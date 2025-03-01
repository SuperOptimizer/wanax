import numpy as np
import zarr
import src.viewer



if __name__ == "__main__":

    root = zarr.open("/Volumes/vesuvius/optimized")
    chunk = root["scroll1a/54kev/8um"][8192-1024*3:8192-1024*2,8192-1024*3:8192-1024*2,8192-1024*3:8192-1024*2]
    src.viewer.VolumeViewer(chunk).run()
