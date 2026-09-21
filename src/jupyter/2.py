# @title Gemini修正版
import numpy as np
import matplotlib.pyplot as plt

# 定数
K = 6.33 * (10**4)
m1_mag = -24 * (10**-5)  # 磁気量1
m2_mag = 18 * (10**-5)  # 磁気量2
m3_mag = 18 * (10**-5)
m4_mag = 8 * (10**-5)
m5_mag = 2 * (10**-5)
mass = 0.006  # 磁石1の質量 [kg]

dt = 0.005
steps = 7000

# 初期値
r2 = np.array([1.0, 1.0])
r3 = np.array([-1.0, 1.0])
r4 = np.array([-1.0, -1.0])
r5 = np.array([1.0, -1.0])
r1 = np.zeros((steps, 2))
v = np.zeros((steps, 2))
a = np.zeros((steps, 2))
r1[0] = np.array([2.0, 0.0])  # 初期位置
v[0] = np.array([0.3, 0.8])  # 初期速度
a[0] = 0

for i in range(steps - 1):
    # 距離
    r1r2 = np.linalg.norm(r1[i] - r2)
    # 磁気力: F = K * m1 * m2 / r^2
    fmag_r1r2 = K * m1_mag * m2_mag / (r1r2**2)
    fmag_r1r2 = fmag_r1r2 * ((r1[i] - r2) / r1r2)

    r1r3 = np.linalg.norm(r1[i] - r3)
    fmag_r1r3 = K * m1_mag * m3_mag / (r1r3**2)
    fmag_r1r3 = fmag_r1r3 * (r1[i] - r3) / r1r3

    r1r4 = np.linalg.norm(r1[i] - r4)
    fmag_r1r4 = K * m1_mag * m4_mag / (r1r4**2)
    fmag_r1r4 = fmag_r1r4 * (r1[i] - r4) / r1r4

    r1r5 = np.linalg.norm(r1[i] - r5)
    fmag_r1r5 = K * m1_mag * m5_mag / (r1r5**2)
    fmag_r1r5 = fmag_r1r5 * (r1[i] - r5) / r1r5

    # 加速度: a = F / mass
    a[i + 1] = (fmag_r1r2 + fmag_r1r3 + fmag_r1r4 + fmag_r1r5) / mass

    # 速度と位置の更新 (velocity verlet法)
    r1[i + 1] = r1[i] + v[i] * dt + a[i] * (dt**2) / 2
    v[i + 1] = v[i] + (a[i] + a[i + 1]) * dt / 2


def plot(
    r: np.ndarray[np.float64], v: np.ndarray[np.float64], a: np.ndarray[np.float64]
):
    fig = plt.figure(figsize=(6, 6))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(v[:, 0] * v[:, 1], "b", label="Velocity")
    ax1.set_xlabel("t [0.001s]")
    ax1.set_ylabel("v [m/0.001s]")
    ax1.grid()
    ax1.set_title("Velocity")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(r1[:, 0], r1[:, 1], "b", label="Magnet 1")
    ax2.plot(r2[0], r2[1], "ro", label="Magnet 2")
    ax2.plot(r3[0], r3[1], "ro", label="Magnet 3")
    ax2.plot(r4[0], r4[1], "ro", label="Magnet 4")
    ax2.plot(r5[0], r5[1], "ro", label="Magnet 5")
    ax2.grid()
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Magnetic Interaction Trajectory")

    fig.show()


plot(r1, v, a)