"""
https://stackoverflow.com/questions/21836067/interpolate-3d-volume-with-numpy-and-or-scipy

"""
import numpy as np
from numba import njit

@njit(fastmath=True)
def trilinear_interpolation_jit(
    x_volume,
    y_volume,
    z_volume,
    volume,
    x_needed,
    y_needed,
    z_needed
):
    """
    Trilinear interpolation (from Wikipedia)

    :param x_volume: x points of the volume grid 
    :type crack_type: list or numpy.ndarray
    :param y_volume: y points of the volume grid 
    :type crack_type: list or numpy.ndarray
    :param x_volume: z points of the volume grid 
    :type crack_type: list or numpy.ndarray
    :param volume:   volume
    :type crack_type: list or numpy.ndarray
    :param x_needed: desired x coordinate of volume
    :type crack_type: float
    :param y_needed: desired y coordinate of volume
    :type crack_type: float
    :param z_needed: desired z coordinate of volume
    :type crack_type: float

    :return volume_needed: desired value of the volume, i.e. volume(x_needed, y_needed, z_needed)
    :type volume_needed: float
    """

    # dimensinoal check
    assert np.shape(volume) == (
        len(x_volume), len(y_volume), len(z_volume)
    ), "Incompatible lengths"
    # check of the indices needed for the correct control volume definition
    i = np.searchsorted(x_volume, x_needed)
    j = np.searchsorted(y_volume, y_needed)
    k = np.searchsorted(z_volume, z_needed)
    # control volume definition
    control_volume_coordinates = np.array(
        [
            [
                x_volume[i - 1],
                y_volume[j - 1],
                z_volume[k - 1]
            ],
            [
                x_volume[i],
                y_volume[j],
                z_volume[k]
            ]
        ]
    )
    xd = (
        np.array([x_needed, y_needed, z_needed]) - control_volume_coordinates[0]
    ) / (
        control_volume_coordinates[1] - control_volume_coordinates[0]
    )
    # interpolation along x
    c2 = [[0., 0.], [0., 0.]]
    for m, n in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        c2[m][n] = volume[i - 1][j - 1 + m][k - 1 + n] \
        * (1. - xd[0]) + volume[i][j - 1 + m][k - 1 + n] * xd[0]
    # interpolation along y
    c1 = [0., 0.]
    c1[0] = c2[0][0] * (1. - xd[1]) + c2[1][0] * xd[1]
    c1[1] = c2[0][1] * (1. - xd[1]) + c2[1][1] * xd[1]
    # interpolation along z
    volume_needed = c1[0] * (1. - xd[2]) + c1[1] * xd[2]
    return volume_needed

@njit(fastmath=True)
def trilint_jit(
    x_volume,
    y_volume,
    z_volume,
    volume,
    x_needed,
    y_needed,
    z_needed
):
    trilint_size = x_needed.size * y_needed.size * z_needed.size
    jitted_trilint = np.zeros(trilint_size)
    m = 0
    for x in range(0, len(x_needed)):
        for y in range(0, len(y_needed)):
            for z in range(0, len(z_needed)):
                jitted_trilint[m]=trilinear_interpolation_jit(
                    x_volume,
                    y_volume,
                    z_volume,
                    volume,
                    x_needed[x], 
                    y_needed[y],
                    z_needed[z]
                )
                m = m + 1
    return jitted_trilint
############################################
import numpy as np
from itertools import product

def trilinear_interpolation(x_volume, y_volume, z_volume, volume, x_needed, y_needed, z_needed):
    """
    Trilinear interpolation (from Wikipedia)

    :param x_volume: x points of the volume grid 
    :type crack_type: list or numpy.ndarray
    :param y_volume: y points of the volume grid 
    :type crack_type: list or numpy.ndarray
    :param x_volume: z points of the volume grid 
    :type crack_type: list or numpy.ndarray
    :param volume:   volume
    :type crack_type: list or numpy.ndarray
    :param x_needed: desired x coordinate of volume
    :type crack_type: float
    :param y_needed: desired y coordinate of volume
    :type crack_type: float
    :param z_needed: desired z coordinate of volume
    :type crack_type: float

    :return volume_needed: desired value of the volume, i.e. volume(x_needed, y_needed, z_needed)
    :type volume_needed: float
    """
    # dimensinoal check
    if np.shape(volume) != (len(x_volume), len(y_volume), len(z_volume)):
        raise ValueError(f'dimension mismatch, volume must be a ({len(x_volume)}, {len(y_volume)}, {len(z_volume)}) list or numpy.ndarray')
    # check of the indices needed for the correct control volume definition
    i = searchsorted(x_volume, x_needed)
    j = searchsorted(y_volume, y_needed)
    k = searchsorted(z_volume, z_needed)
    # control volume definition
    control_volume_coordinates = np.array(
        [[x_volume[i - 1], y_volume[j - 1], z_volume[k - 1]], [x_volume[i], y_volume[j], z_volume[k]]])
    xd = (np.array([x_needed, y_needed, z_needed]) - control_volume_coordinates[0]) / (control_volume_coordinates[1] - control_volume_coordinates[0])
    # interpolation along x
    c2 = [[0, 0], [0, 0]]
    for m, n in product([0, 1], [0, 1]):
        c2[m][n] = volume[i - 1][j - 1 + m][k - 1 + n] * (1 - xd[0]) + volume[i][j - 1 + m][k - 1 + n] * xd[0]
    # interpolation along y
    c1 = [0, 0]
    c1[0] = c2[0][0] * (1 - xd[1]) + c2[1][0] * xd[1]
    c1[1] = c2[0][1] * (1 - xd[1]) + c2[1][1] * xd[1]
    # interpolation along z
    volume_needed = c1[0] * (1 - xd[2]) + c1[1] * xd[2]
    return volume_needed

def searchsorted(l, x):
    for i in l:
        if i >= x: break
    #return l.index(i)
    return np.where(l == i)


from scipy.interpolate import RegularGridInterpolator
def trilin_interp_regular_grid(x_volume, y_volume, z_volume, volume, x_needed, y_needed, z_needed):
    # dimensinoal check
    if np.shape(volume) != (len(x_volume), len(y_volume), len(z_volume)):
        raise ValueError(f'dimension mismatch, volume must be a ({len(x_volume)}, {len(y_volume)}, {len(z_volume)}) list or numpy.ndarray')
    # trilinear interpolation on a regular grid
    fn = RegularGridInterpolator((x_volume,y_volume,z_volume), volume)
    volume_needed = fn(np.array([x_needed, y_needed, z_needed]))
    return volume_needed
##############################################
import numpy as np
import time

x_volume = np.array([100., 1000.])
y_volume = np.array([0.2, 0.4, 0.6, 0.8, 1])
z_volume = np.array([0, 0.2, 0.5, 0.8, 1.])

the_volume = np.array(
[[[0.902, 0.985, 1.12, 1.267, 1.366],
[0.822, 0.871, 0.959, 1.064, 1.141],
[0.744, 0.77, 0.824, 0.897, 0.954],
[0.669, 0.682, 0.715, 0.765, 0.806],
[0.597, 0.607, 0.631, 0.667, 0.695]],
[[1.059, 1.09, 1.384, 1.682, 1.881],
[0.948, 0.951, 1.079, 1.188, 1.251],
[0.792, 0.832, 0.888, 0.940, 0.971],
[0.726, 0.733, 0.754, 0.777, 0.792],
[0.642, 0.656, 0.675, 0.691, 0.700]]])

x_needed = np.linspace(100, 1000, 10)
y_needed = np.linspace(0.3, 1, 60)
z_needed = np.linspace(0, 1, 7)

start = time.time()
jitted_trilint = trilint_jit(
    x_volume, y_volume, z_volume, the_volume, x_needed, y_needed, z_needed
)
end = time.time()
print('---')
print(f"NUMBA: {end - start}")
print('---')
"""
start = time.time()
manual_trilint = []
for x in x_needed:
    for y in y_needed:
        for z in z_needed:
            manual_trilint.append(
                trilinear_interpolation(
                    x_volume, y_volume, z_volume, the_volume, x, y, z
                )
            )
end = time.time()
print('---')

print(f"Manual: {end - start}")
print('---')
"""
start = time.time()
auto_trilint = []
for x in x_needed:
    for y in y_needed:
        for z in z_needed:
            auto_trilint.append(
                trilin_interp_regular_grid(
                    x_volume, y_volume, z_volume, the_volume, x, y, z
                )
            )
end = time.time()
print('---')
print(f"Auto: {end - start}")
print('---')