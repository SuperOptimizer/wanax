import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter

@njit
def pad_array(data, pad_width):
    if data.ndim == 1:
        return np.pad(data, pad_width, mode='reflect')
    elif data.ndim == 2:
        return np.pad(data, ((pad_width, pad_width), (pad_width, pad_width)), mode='reflect')
    else:  # 3D
        return np.pad(data, ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)), mode='reflect')

@njit(parallel=False)
def sobel_filter(data):
    if data.ndim == 1:
        kernel = np.array([-1, 0, 1])
        padded = pad_array(data, 1)
        gradient = np.zeros_like(data, dtype=np.float32)
        for i in prange(data.shape[0]):
            gradient[i] = np.sum(padded[i:i+3] * kernel)
        return gradient

    elif data.ndim == 2:
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        padded = pad_array(data, 1)
        gx = np.zeros_like(data, dtype=np.float32)
        gy = np.zeros_like(data, dtype=np.float32)

        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                window = padded[y:y+3, x:x+3]
                gx[y, x] = np.sum(window * kx)
                gy[y, x] = np.sum(window * ky)

        return np.sqrt(gx**2 + gy**2)

    else:  # 3D
        kx = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                       [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])

        ky = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                       [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])

        kz = np.array([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                       [[1, 2, 1], [2, 4, 2], [1, 2, 1]]])

        padded = pad_array(data, 1)
        gx = np.zeros_like(data, dtype=np.float32)
        gy = np.zeros_like(data, dtype=np.float32)
        gz = np.zeros_like(data, dtype=np.float32)

        for z in prange(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    window = padded[z:z+3, y:y+3, x:x+3]
                    gx[z, y, x] = np.sum(window * kx)
                    gy[z, y, x] = np.sum(window * ky)
                    gz[z, y, x] = np.sum(window * kz)

        return np.sqrt(gx**2 + gy**2 + gz**2)

@njit(parallel=False)
def prewitt_filter(data):
    if data.ndim == 1:
        kernel = np.array([-1, 0, 1])
        padded = pad_array(data, 1)
        gradient = np.zeros_like(data, dtype=np.float32)
        for i in prange(data.shape[0]):
            gradient[i] = np.sum(padded[i:i+3] * kernel)
        return gradient

    elif data.ndim == 2:
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        padded = pad_array(data, 1)
        gx = np.zeros_like(data, dtype=np.float32)
        gy = np.zeros_like(data, dtype=np.float32)

        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                window = padded[y:y+3, x:x+3]
                gx[y, x] = np.sum(window * kx)
                gy[y, x] = np.sum(window * ky)

        return np.sqrt(gx**2 + gy**2)

    else:  # 3D
        kx = np.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                       [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                       [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])

        ky = np.array([[[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
                       [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
                       [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]])

        kz = np.array([[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                       [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        padded = pad_array(data, 1)
        gx = np.zeros_like(data, dtype=np.float32)
        gy = np.zeros_like(data, dtype=np.float32)
        gz = np.zeros_like(data, dtype=np.float32)

        for z in prange(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    window = padded[z:z+3, y:y+3, x:x+3]
                    gx[z, y, x] = np.sum(window * kx)
                    gy[z, y, x] = np.sum(window * ky)
                    gz[z, y, x] = np.sum(window * kz)

        return np.sqrt(gx**2 + gy**2 + gz**2)

@njit(parallel=False)
def scharr_filter(data):
    if data.ndim == 1:
        kernel = np.array([-1, 0, 1])
        padded = pad_array(data, 1)
        gradient = np.zeros_like(data, dtype=np.float32)
        for i in prange(data.shape[0]):
            gradient[i] = np.sum(padded[i:i+3] * kernel)
        return gradient

    elif data.ndim == 2:
        kx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        ky = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

        padded = pad_array(data, 1)
        gx = np.zeros_like(data, dtype=np.float32)
        gy = np.zeros_like(data, dtype=np.float32)

        for y in prange(data.shape[0]):
            for x in range(data.shape[1]):
                window = padded[y:y+3, x:x+3]
                gx[y, x] = np.sum(window * kx)
                gy[y, x] = np.sum(window * ky)

        return np.sqrt(gx**2 + gy**2)

def canny_edge_detection(data, low_threshold=10, high_threshold=30, sigma=1.0):
    smoothed = gaussian_filter(data, sigma)
    gradient = sobel_filter(smoothed)

    # Non-maximum suppression
    theta = np.arctan2(sobel_filter(smoothed), sobel_filter(smoothed))
    suppressed = np.zeros_like(gradient)

    for y in range(1, gradient.shape[0] - 1):
        for x in range(1, gradient.shape[1] - 1):
            angle = theta[y, x] * 180 / np.pi
            angle = angle % 180

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [gradient[y, x-1], gradient[y, x+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [gradient[y-1, x+1], gradient[y+1, x-1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [gradient[y-1, x], gradient[y+1, x]]
            else:
                neighbors = [gradient[y-1, x-1], gradient[y+1, x+1]]

            if gradient[y, x] >= max(neighbors):
                suppressed[y, x] = gradient[y, x]

    # Double thresholding
    strong_edges = suppressed > high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)

    # Edge tracking by hysteresis
    final_edges = np.copy(strong_edges)

    for y in range(1, suppressed.shape[0] - 1):
        for x in range(1, suppressed.shape[1] - 1):
            if weak_edges[y, x]:
                if np.any(strong_edges[y-1:y+2, x-1:x+2]):
                    final_edges[y, x] = True

    return final_edges

@njit(parallel=False)
def hessian_filter(data):
    result = np.zeros_like(data, dtype=np.float32)

    # Skip edge voxels
    for z in prange(1, data.shape[0]-1):
        for y in range(1, data.shape[1]-1):
            for x in range(1, data.shape[2]-1):
                # Second derivatives
                hxx = data[z, y, x-1] - 2*data[z, y, x] + data[z, y, x+1]
                hyy = data[z, y-1, x] - 2*data[z, y, x] + data[z, y+1, x]
                hzz = data[z-1, y, x] - 2*data[z, y, x] + data[z+1, y, x]

                # Mixed derivatives
                hxy = (data[z, y+1, x+1] + data[z, y-1, x-1] - data[z, y-1, x+1] - data[z, y+1, x-1]) / 4
                hxz = (data[z+1, y, x+1] + data[z-1, y, x-1] - data[z-1, y, x+1] - data[z+1, y, x-1]) / 4
                hyz = (data[z+1, y+1, x] + data[z-1, y-1, x] - data[z-1, y+1, x] - data[z+1, y-1, x]) / 4

                result[z, y, x] = np.sqrt(hxx**2 + hyy**2 + hzz**2 + 2*(hxy**2 + hxz**2 + hyz**2))

    return result

@njit(parallel=False)
def local_binary_pattern(data, radius=1, points=8):
    padded = pad_array(data, radius)
    result = np.zeros_like(data, dtype=np.uint8)

    for y in prange(data.shape[0]):
        for x in range(data.shape[1]):
            center = data[y, x]
            code = 0

            for p in range(points):
                theta = 2 * np.pi * p / points
                yp = y + int(round(-radius * np.sin(theta))) + radius
                xp = x + int(round(radius * np.cos(theta))) + radius
                code |= (padded[yp, xp] > center) << p

            result[y, x] = code

    return result