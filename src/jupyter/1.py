# @title 磁気力で軌道を作れるか？グラフ

# m2は固定されているものとする(0,0)

import numpy as np
import matplotlib.pyplot as plt

K = 6.33 * (10**4)  # クーロンの法則の定数
# 磁気量
m1 = 4 * (10**-5)
m2 = 4 * (10**-5)

t0 = 0
# m1の初期位置
r0 = np.array([1, 0])
# 初期速度
v0 = np.array([-0.07, 0.05])

t = np.zeros(10000)
t[0] = 0
r1 = np.zeros((10000, 2))
r1[0, :] = r0
v = np.zeros((10000, 2))
v[0, :] = v0

for i in range(9999):
    # 距離
    ri_norm = np.sqrt(r1[i, 0] ** 2 + r1[i, 1] ** 2)
    t[i + 1] = t[i] + 1
    v[i + 1, :] = v[i, :] - K * m1 * m2 / ri_norm
    r1[i + 1, :] = r1[i + 1, :] + v[i, :] * (t[i + 1] - t[i])

plt.plot(r1[:, 0], r1[:, 1], "b")
plt.grid()
plt.gca().set_aspect("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.show()