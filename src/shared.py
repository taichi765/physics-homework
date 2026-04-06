import numpy as np
from numba import njit

K = 6.33 * (10**4)
R_OTHERS = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
M1_MAG = -18 * (10**-5)
M_OTHERS = np.array([18 * (10**-5), 18 * (10**-5), 18 * (10**-5), 18 * (10**-5)])
MASS = 0.006
DT = 0.005
STEPS = 7000


@njit(cache=True)
def get_acceleration(r1):
    fx = 0.0
    fy = 0.0
    for i in range(4):
        dx = r1[0] - R_OTHERS[i, 0]
        dy = r1[1] - R_OTHERS[i, 1]
        dist_sq = dx * dx + dy * dy
        # クーロンの法則: F = K * m1 * m2 * r / r^3
        f_mag = K * M1_MAG * M_OTHERS[i] / (dist_sq * np.sqrt(dist_sq))
        fx += f_mag * dx
        fy += f_mag * dy
    return np.array((fx / MASS, fy / MASS))
