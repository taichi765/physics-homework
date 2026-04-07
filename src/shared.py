import numpy as np
from numba import njit

# 真空の透磁率
MU0 = 4 * np.pi * (10**-7)
K = 1 / (4 * np.pi * MU0)
R_OTHERS_4 = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
R_OTHERS_2 = np.array([[0.04, 0.0], [-0.04, 0.0]])
# 200ミリテスラ ✕ 底面積
M1_MAG = 0.2 * np.pi * (0.02 * 0.02)
M_OTHERS_4 = np.full((4,), -M1_MAG)
M_OTHERS_2 = np.full((2,), -M1_MAG)
MASS = 0.006
DT = 0.005
STEPS = 7000


@njit(cache=True)
def get_acceleration_4(r1):
    fx = 0.0
    fy = 0.0
    for i in range(4):
        dx = r1[0] - R_OTHERS_4[i, 0]
        dy = r1[1] - R_OTHERS_4[i, 1]
        dist_sq = dx * dx + dy * dy
        # クーロンの法則: F = K * m1 * m2 * r / r^3
        f_mag = K * M1_MAG * M_OTHERS_4[i] / (dist_sq * np.sqrt(dist_sq))
        fx += f_mag * dx
        fy += f_mag * dy
    return np.array((fx / MASS, fy / MASS))


@njit(cache=True)
def get_acceleration_2(r1):
    fx = 0.0
    fy = 0.0
    for i in range(2):
        dx = r1[0] - R_OTHERS_2[i, 0]
        dy = r1[1] - R_OTHERS_2[i, 1]
        dist_sq = dx * dx + dy * dy
        # クーロンの法則: F = K * m1 * m2 * r / r^3
        f_mag = K * M1_MAG * M_OTHERS_2[i] / (dist_sq * np.sqrt(dist_sq))
        fx += f_mag * dx
        fy += f_mag * dy
    return np.array((fx / MASS, fy / MASS))


@njit(cache=True, inline="always")
def check_collision_2(r1):
    """固定されている磁石のいずれかとr1が衝突している場合True"""
    for i in range(2):
        dist = np.linalg.norm(r1 - R_OTHERS_2[i])
        # 磁石の直径がだいたい2cm
        if dist < 0.02:
            return True
    return False
