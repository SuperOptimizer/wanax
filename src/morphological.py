import numpy as np
from numba import njit, prange

@njit
def pad_array(data, pad_width):
    if data.ndim == 1:
        return np.pad(data, pad_width, mode='edge')
    elif data.ndim == 2:
        return np.pad(data, ((pad_width, pad_width), (pad_width, pad_width)), mode='edge')
    else:  # 3D
        return np.pad(data, ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)), mode='edge')

@njit(parallel=True)
def erosion(data, radius=1):
    padded = pad_array(data, radius)
    result = np.zeros_like(data)

    if data.ndim == 1:
        for i in prange(data.shape[0]):
            window = padded[i:i + 2*radius + 1]
            result[i] = np.min(window)
    elif data.ndim == 2:
        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                window = padded[y:y + 2*radius + 1, x:x + 2*radius + 1]
                result[y, x] = np.min(window)
    else:  # 3D
        for z in prange(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    window = padded[z:z + 2*radius + 1, y:y + 2*radius + 1, x:x + 2*radius + 1]
                    result[z, y, x] = np.min(window)
    return result

@njit(parallel=True)
def dilation(data, radius=1):
    padded = pad_array(data, radius)
    result = np.zeros_like(data)

    if data.ndim == 1:
        for i in prange(data.shape[0]):
            window = padded[i:i + 2*radius + 1]
            result[i] = np.max(window)
    elif data.ndim == 2:
        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                window = padded[y:y + 2*radius + 1, x:x + 2*radius + 1]
                result[y, x] = np.max(window)
    else:  # 3D
        for z in prange(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    window = padded[z:z + 2*radius + 1, y:y + 2*radius + 1, x:x + 2*radius + 1]
                    result[z, y, x] = np.max(window)
    return result

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

@njit(parallel=True)
def skeletonize(binary_data):
    skeleton = np.copy(binary_data)
    changed = True

    while changed:
        changed = False
        eroded = erosion(skeleton, 1)
        opened = opening(skeleton, 1)
        endpoints = skeleton - eroded
        skeleton_new = np.where((skeleton - opened) > 0, endpoints, skeleton)

        if not np.array_equal(skeleton, skeleton_new):
            changed = True
            skeleton = skeleton_new

    return skeleton