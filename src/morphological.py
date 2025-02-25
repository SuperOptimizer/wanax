import numpy as np
from numba import njit, prange

import numpy as np
from numba import njit, prange



@njit
def pad_3d(data, pad_width):
    padded_shape = (data.shape[0] + 2*pad_width,
                    data.shape[1] + 2*pad_width,
                    data.shape[2] + 2*pad_width)
    padded = np.zeros(padded_shape, dtype=data.dtype)

    # Copy internal data
    padded[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width] = data

    # Edge padding
    # Z faces
    padded[:pad_width, pad_width:-pad_width, pad_width:-pad_width] = data[0:1, :, :]
    padded[-pad_width:, pad_width:-pad_width, pad_width:-pad_width] = data[-1:, :, :]

    # Y faces
    padded[:, :pad_width, pad_width:-pad_width] = padded[:, pad_width:pad_width+1, pad_width:-pad_width]
    padded[:, -pad_width:, pad_width:-pad_width] = padded[:, -pad_width-1:-pad_width, pad_width:-pad_width]

    # X faces
    padded[:, :, :pad_width] = padded[:, :, pad_width:pad_width+1]
    padded[:, :, -pad_width:] = padded[:, :, -pad_width-1:-pad_width]

    return padded

@njit(parallel=True)
def erosion(data, radius=1):
    padded = pad_3d(data, radius)
    result = np.zeros_like(data)

    for z in prange(data.shape[0]):
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                window = padded[z:z + 2*radius + 1,
                         y:y + 2*radius + 1,
                         x:x + 2*radius + 1]
                result[z, y, x] = np.min(window)
    return result

@njit(parallel=True)
def dilation(data, radius=1):
    padded = pad_3d(data, radius)
    result = np.zeros_like(data)

    for z in prange(data.shape[0]):
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                window = padded[z:z + 2*radius + 1,
                         y:y + 2*radius + 1,
                         x:x + 2*radius + 1]
                result[z, y, x] = np.max(window)
    return result

@njit
def opening_njit(data, radius=1):
    return dilation(erosion(data, radius), radius)

@njit
def skeletonize(binary_data):
    skeleton = np.copy(binary_data)
    changed = True

    while changed:
        changed = False
        eroded = erosion(skeleton, 1)
        opened = opening_njit(skeleton, 1)
        endpoints = skeleton - eroded
        skeleton_new = np.where((skeleton - opened) > 0, endpoints, skeleton)

        if not np.array_equal(skeleton, skeleton_new):
            changed = True
            skeleton = skeleton_new

    return skeleton

def opening(data, radius=1):
    return dilation(erosion(data, radius), radius)

def closing(data, radius=1):
    return erosion(dilation(data, radius), radius)

def top_hat(data, radius=1):
    return data - opening(data, radius)

def black_hat(data, radius=1):
    return closing(data, radius) - data

@njit(parallel=True)
def distance_transform(binary_data):
    result = np.zeros_like(binary_data, dtype=np.float32)
    max_dist = np.sum(binary_data.shape)

    # Forward pass
    if binary_data.ndim == 1:
        dist = 0
        for i in range(binary_data.shape[0]):
            if binary_data[i] == 0:
                dist = 0
            else:
                dist += 1
            result[i] = dist
    elif binary_data.ndim == 2:
        for y in prange(binary_data.shape[0]):
            dist = 0
            for x in range(binary_data.shape[1]):
                if binary_data[y, x] == 0:
                    dist = 0
                else:
                    dist += 1
                result[y, x] = min(result[y, x-1] + 1 if x > 0 else max_dist,
                                   result[y-1, x] + 1 if y > 0 else max_dist,
                                   dist)
    else:  # 3D
        for z in prange(binary_data.shape[0]):
            for y in range(binary_data.shape[1]):
                dist = 0
                for x in range(binary_data.shape[2]):
                    if binary_data[z, y, x] == 0:
                        dist = 0
                    else:
                        dist += 1
                    result[z, y, x] = min(
                        result[z, y, x-1] + 1 if x > 0 else max_dist,
                        result[z, y-1, x] + 1 if y > 0 else max_dist,
                        result[z-1, y, x] + 1 if z > 0 else max_dist,
                        dist)

    # Backward pass
    if binary_data.ndim == 1:
        dist = 0
        for i in range(binary_data.shape[0]-1, -1, -1):
            if binary_data[i] == 0:
                dist = 0
            else:
                dist += 1
            result[i] = min(result[i], dist)
    elif binary_data.ndim == 2:
        for y in prange(binary_data.shape[0]-1, -1, -1):
            dist = 0
            for x in range(binary_data.shape[1]-1, -1, -1):
                if binary_data[y, x] == 0:
                    dist = 0
                else:
                    dist += 1
                result[y, x] = min(result[y, x],
                                   result[y, x+1] + 1 if x < binary_data.shape[1]-1 else max_dist,
                                   result[y+1, x] + 1 if y < binary_data.shape[0]-1 else max_dist,
                                   dist)
    else:  # 3D
        for z in prange(binary_data.shape[0]-1, -1, -1):
            for y in range(binary_data.shape[1]-1, -1, -1):
                dist = 0
                for x in range(binary_data.shape[2]-1, -1, -1):
                    if binary_data[z, y, x] == 0:
                        dist = 0
                    else:
                        dist += 1
                    result[z, y, x] = min(result[z, y, x],
                                          result[z, y, x+1] + 1 if x < binary_data.shape[2]-1 else max_dist,
                                          result[z, y+1, x] + 1 if y < binary_data.shape[1]-1 else max_dist,
                                          result[z+1, y, x] + 1 if z < binary_data.shape[0]-1 else max_dist,
                                          dist)

    return result
