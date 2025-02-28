import numpy as np
from numba import jit, prange


@jit(nopython=True)
def trilinear_interpolation(volume, x, y, z):
    """
    Perform trilinear interpolation at point (x, y, z) in the volume.

    Parameters:
    volume (numpy.ndarray): 3D array of uint8 values
    x, y, z (float): Coordinates to sample from (can be fractional)

    Returns:
    float: Interpolated value
    """
    shape = volume.shape

    # Get integer and fractional parts
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    z0 = int(np.floor(z))
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Ensure we don't go out of bounds
    x0 = max(0, min(x0, shape[0] - 1))
    y0 = max(0, min(y0, shape[1] - 1))
    z0 = max(0, min(z0, shape[2] - 1))
    x1 = max(0, min(x1, shape[0] - 1))
    y1 = max(0, min(y1, shape[1] - 1))
    z1 = max(0, min(z1, shape[2] - 1))

    # Get fractional part
    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Get values at the corners of the cube
    c000 = volume[x0, y0, z0]
    c001 = volume[x0, y0, z1]
    c010 = volume[x0, y1, z0]
    c011 = volume[x0, y1, z1]
    c100 = volume[x1, y0, z0]
    c101 = volume[x1, y0, z1]
    c110 = volume[x1, y1, z0]
    c111 = volume[x1, y1, z1]

    # Interpolate along x axis
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # Interpolate along y axis
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate along z axis
    c = c0 * (1 - zd) + c1 * zd

    return c

@jit(nopython=True, parallel=False)
def scale_volume(volume, scale_factor):
    """
    Scale the contents of a 3D volume using trilinear interpolation,
    keeping the same output dimensions.

    Parameters:
    volume (numpy.ndarray): 3D array of uint8 values (0-255)
    scale_factor (float): Scaling factor between 0-1

    Returns:
    numpy.ndarray: Volume with contents scaled down but same dimensions
    """
    if scale_factor <= 0 or scale_factor > 1:
        raise ValueError("Scale factor must be between 0 and 1 (exclusive of 0)")

    # Get dimensions
    dim_x, dim_y, dim_z = volume.shape

    # Create output array
    result = np.empty_like(volume)

    # Calculate center offset for scaling around the center of the volume
    center_x, center_y, center_z = dim_x / 2.0, dim_y / 2.0, dim_z / 2.0

    # Coordinate scaling factors
    inverse_scale = 1.0 / scale_factor

    for x in prange(dim_x):
        for y in prange(dim_y):
            for z in range(dim_z):
                # Scale coordinates relative to center
                src_x = center_x + (x - center_x) * inverse_scale
                src_y = center_y + (y - center_y) * inverse_scale
                src_z = center_z + (z - center_z) * inverse_scale

                # Skip if outside bounds
                if (src_x < 0 or src_x > dim_x - 1 or
                        src_y < 0 or src_y > dim_y - 1 or
                        src_z < 0 or src_z > dim_z - 1):
                    result[x, y, z] = 0
                else:
                    # Perform trilinear interpolation
                    value = trilinear_interpolation(volume, src_x, src_y, src_z)
                    result[x, y, z] = np.uint8(value)

    return result

@jit(nopython=True, parallel=False)
def scale_volume_optimized(volume, scale_factor):
    """
    Optimized version of scale_volume_trilinear that uses pre-computed
    coordinate arrays for better memory access patterns.

    Parameters:
    volume (numpy.ndarray): 3D array of uint8 values (0-255)
    scale_factor (float): Scaling factor between 0-1

    Returns:
    numpy.ndarray: Volume with contents scaled down but same dimensions
    """
    if scale_factor <= 0 or scale_factor > 1:
        raise ValueError("Scale factor must be between 0 and 1 (exclusive of 0)")

    # Get dimensions
    dim_x, dim_y, dim_z = volume.shape

    # Create output array
    result = np.empty_like(volume)

    # Calculate center offset for scaling around the center of the volume
    center_x, center_y, center_z = dim_x / 2.0, dim_y / 2.0, dim_z / 2.0

    # Coordinate scaling factor
    inverse_scale = 1.0 / scale_factor

    # Pre-compute source coordinates (this helps with vectorization and cache efficiency)
    for x in prange(dim_x):
        src_x = center_x + (x - center_x) * inverse_scale

        # Skip entire x slice if out of bounds
        if src_x < 0 or src_x > dim_x - 1:
            for y in range(dim_y):
                for z in range(dim_z):
                    result[x, y, z] = 0
            continue

        x0 = int(np.floor(src_x))
        x1 = min(x0 + 1, dim_x - 1)
        xd = src_x - x0

        for y in range(dim_y):
            src_y = center_y + (y - center_y) * inverse_scale

            # Skip entire y row if out of bounds
            if src_y < 0 or src_y > dim_y - 1:
                for z in range(dim_z):
                    result[x, y, z] = 0
                continue

            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, dim_y - 1)
            yd = src_y - y0

            for z in range(dim_z):
                src_z = center_z + (z - center_z) * inverse_scale

                # Skip this voxel if out of bounds
                if src_z < 0 or src_z > dim_z - 1:
                    result[x, y, z] = 0
                    continue

                z0 = int(np.floor(src_z))
                z1 = min(z0 + 1, dim_z - 1)
                zd = src_z - z0

                # Get corner values
                c000 = volume[x0, y0, z0]
                c001 = volume[x0, y0, z1]
                c010 = volume[x0, y1, z0]
                c011 = volume[x0, y1, z1]
                c100 = volume[x1, y0, z0]
                c101 = volume[x1, y0, z1]
                c110 = volume[x1, y1, z0]
                c111 = volume[x1, y1, z1]

                # Trilinear interpolation
                # Interpolate along x
                c00 = c000 * (1 - xd) + c100 * xd
                c01 = c001 * (1 - xd) + c101 * xd
                c10 = c010 * (1 - xd) + c110 * xd
                c11 = c011 * (1 - xd) + c111 * xd

                # Interpolate along y
                c0 = c00 * (1 - yd) + c10 * yd
                c1 = c01 * (1 - yd) + c11 * yd

                # Interpolate along z
                c = c0 * (1 - zd) + c1 * zd

                result[x, y, z] = np.uint8(c)

    return result