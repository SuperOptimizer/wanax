import numpy as np
from numba import njit, prange
from scipy.signal import butter, filtfilt

@njit
def fft_transform(data):
    centered = data - np.mean(data)
    f_transform = np.fft.fftn(centered)
    return np.fft.fftshift(f_transform)

@njit
def inverse_fft(f_data):
    ishifted = np.fft.ifftshift(f_data)
    return np.real(np.fft.ifftn(ishifted))

@njit
def make_frequency_filter(shape, cutoff, filter_type='low'):
    center = np.array(shape) // 2
    coords = np.ogrid[tuple(slice(-c, c) for c in center)]
    dist = np.sqrt(sum(coord**2 for coord in coords))

    if isinstance(cutoff, (tuple, list)):
        low, high = cutoff[0] / center[0], cutoff[1] / center[0]
        if filter_type == 'band':
            return (dist >= low * np.max(dist)) & (dist <= high * np.max(dist))
        else:  # notch
            return (dist <= low * np.max(dist)) | (dist >= high * np.max(dist))
    else:
        normalized_dist = dist / (cutoff * np.max(dist))
        if filter_type == 'low':
            return normalized_dist <= 1.0
        else:  # high
            return normalized_dist > 1.0

def butterworth_filter(data, cutoff, order=4, filter_type='low'):
    b, a = butter(order, cutoff, btype=filter_type, fs=2)
    if data.ndim == 1:
        return filtfilt(b, a, data)
    elif data.ndim == 2:
        return np.array([filtfilt(b, a, row) for row in data])
    else:  # 3D
        return np.array([np.array([filtfilt(b, a, row) for row in slice_2d])
                         for slice_2d in data])

def frequency_filter(data, cutoff, filter_type='low'):
    f_data = fft_transform(data)
    f_filter = make_frequency_filter(data.shape, cutoff, filter_type)
    filtered_f = f_data * f_filter
    return inverse_fft(filtered_f)

@njit
def dct_transform(data):
    N = data.shape[0]
    result = np.zeros_like(data, dtype=np.float32)

    for k in prange(N):
        for n in range(N):
            result[k] += data[n] * np.cos(np.pi * (2 * n + 1) * k / (2 * N))

        if k == 0:
            result[k] *= np.sqrt(1.0 / N)
        else:
            result[k] *= np.sqrt(2.0 / N)

    return result

@njit
def inverse_dct(dct_data):
    N = dct_data.shape[0]
    result = np.zeros_like(dct_data, dtype=np.float32)

    for n in prange(N):
        result[n] = dct_data[0] / np.sqrt(N)
        for k in range(1, N):
            result[n] += np.sqrt(2.0/N) * dct_data[k] * np.cos(np.pi * (2 * n + 1) * k / (2 * N))

    return result

def haar_wavelet_transform(data):
    if len(data) <= 1:
        return data

    running_mean = (data[::2] + data[1::2]) / np.sqrt(2)
    running_diff = (data[::2] - data[1::2]) / np.sqrt(2)

    return np.concatenate([
        haar_wavelet_transform(running_mean),
        running_diff
    ])

def inverse_haar_wavelet(data):
    if len(data) <= 1:
        return data

    half = len(data) // 2
    mean = inverse_haar_wavelet(data[:half])
    diff = data[half:]

    return np.ravel(np.column_stack([
        (mean + diff) / np.sqrt(2),
        (mean - diff) / np.sqrt(2)
    ]))

def gabor_filter(data, freq, theta=0, bandwidth=1):
    sigma = bandwidth / freq

    if data.ndim == 1:
        x = np.arange(data.shape[0])
        kernel = np.exp(-x**2/(2*sigma**2)) * np.cos(2*np.pi*freq*x)
        return np.convolve(data, kernel, mode='same')

    elif data.ndim == 2:
        y, x = np.mgrid[-data.shape[0]//2:data.shape[0]//2,
               -data.shape[1]//2:data.shape[1]//2]

        x_theta = x*np.cos(theta) + y*np.sin(theta)
        y_theta = -x*np.sin(theta) + y*np.cos(theta)

        kernel = np.exp(-(x_theta**2 + y_theta**2)/(2*sigma**2)) * \
                 np.cos(2*np.pi*freq*x_theta)

        return np.real(np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(kernel, data.shape)))

    else:  # 3D
        z, y, x = np.mgrid[-data.shape[0]//2:data.shape[0]//2,
                  -data.shape[1]//2:data.shape[1]//2,
                  -data.shape[2]//2:data.shape[2]//2]

        kernel = np.exp(-(x**2 + y**2 + z**2)/(2*sigma**2)) * \
                 np.cos(2*np.pi*freq*(x*np.cos(theta) + y*np.sin(theta)))

        return np.real(np.fft.ifftn(np.fft.fftn(data) * np.fft.fftn(kernel, data.shape)))

def wavelet_denoise(data, threshold=0.1):
    # Forward transform
    coeffs = haar_wavelet_transform(data.ravel())

    # Threshold coefficients
    coeffs[np.abs(coeffs) < threshold * np.max(np.abs(coeffs))] = 0

    # Inverse transform
    denoised = inverse_haar_wavelet(coeffs)
    return denoised.reshape(data.shape)

def haar_wavelet_transform_3d(data):
    if data.size <= 1:
        return data

    shape = data.shape
    result = np.zeros_like(data, dtype=np.float32)

    # Transform along z
    for i in range(0, shape[0], 2):
        if i + 1 < shape[0]:
            result[i//2] = (data[i] + data[i+1]) / np.sqrt(2)
            result[shape[0]//2 + i//2] = (data[i] - data[i+1]) / np.sqrt(2)
        else:
            result[i//2] = data[i]

    # Transform along y
    temp = result.copy()
    for i in range(shape[0]):
        for j in range(0, shape[1], 2):
            if j + 1 < shape[1]:
                result[i,j//2] = (temp[i,j] + temp[i,j+1]) / np.sqrt(2)
                result[i,shape[1]//2 + j//2] = (temp[i,j] - temp[i,j+1]) / np.sqrt(2)
            else:
                result[i,j//2] = temp[i,j]

    # Transform along x
    temp = result.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(0, shape[2], 2):
                if k + 1 < shape[2]:
                    result[i,j,k//2] = (temp[i,j,k] + temp[i,j,k+1]) / np.sqrt(2)
                    result[i,j,shape[2]//2 + k//2] = (temp[i,j,k] - temp[i,j,k+1]) / np.sqrt(2)
                else:
                    result[i,j,k//2] = temp[i,j,k]

    return result

def inverse_haar_wavelet_3d(data):
    if data.size <= 1:
        return data

    shape = data.shape
    result = np.zeros_like(data, dtype=np.float32)

    # Inverse transform along x
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(0, shape[2]//2):
                result[i,j,2*k] = (data[i,j,k] + data[i,j,k + shape[2]//2]) / np.sqrt(2)
                result[i,j,2*k+1] = (data[i,j,k] - data[i,j,k + shape[2]//2]) / np.sqrt(2)
            if shape[2] % 2:
                result[i,j,-1] = data[i,j,shape[2]//2]

    # Inverse transform along y
    temp = result.copy()
    for i in range(shape[0]):
        for j in range(0, shape[1]//2):
            for k in range(shape[2]):
                result[i,2*j,k] = (temp[i,j,k] + temp[i,j + shape[1]//2,k]) / np.sqrt(2)
                result[i,2*j+1,k] = (temp[i,j,k] - temp[i,j + shape[1]//2,k]) / np.sqrt(2)
        if shape[1] % 2:
            result[i,-1] = temp[i,shape[1]//2]

    # Inverse transform along z
    temp = result.copy()
    for i in range(0, shape[0]//2):
        for j in range(shape[1]):
            for k in range(shape[2]):
                result[2*i,j,k] = (temp[i,j,k] + temp[i + shape[0]//2,j,k]) / np.sqrt(2)
                result[2*i+1,j,k] = (temp[i,j,k] - temp[i + shape[0]//2,j,k]) / np.sqrt(2)
    if shape[0] % 2:
        result[-1] = temp[shape[0]//2]

    return result

def wavelet_denoise_3d(data, threshold=0.1):
    # Normalize input
    data_norm = (data - data.min()) / (data.max() - data.min())

    # Forward transform
    coeffs = haar_wavelet_transform_3d(data_norm)

    # Threshold coefficients
    coeffs[np.abs(coeffs) < threshold * np.max(np.abs(coeffs))] = 0

    # Inverse transform
    denoised = inverse_haar_wavelet_3d(coeffs)

    # Restore original range
    denoised = denoised * (data.max() - data.min()) + data.min()

    return denoised.astype(data.dtype)