# @title 周期解を求める
import numpy as np
import matplotlib.pyplot as plt
import itertools
from concurrent.futures import ProcessPoolExecutor
from numba import njit

# 定数
K = 6.33 * (10**4)
r_others = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])

dt = 0.005
steps = 4000

m1_mag = -24 * (10**-5)  # 磁気量1
m_others = np.full(4, 24 * (10**-5))  # 固定された磁石の磁気量
mass = 0.006  # 磁石1の質量 [kg]


# 引数は[x, y, vx, vy]。
# 見つかった場合返り値をそのまま返す。
# 見つからなかった場合len==0の配列を返す。
@njit
def calc(r1_0_and_v_0: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    r1_0 = np.array(r1_0_and_v_0[0:2])
    v_0 = np.array(r1_0_and_v_0[2:4])

    r1_curr = r1_0
    v_curr = v_0
    a_curr = np.zeros(2)

    for i in range(steps - 1):
        if i > 10:
            # r1[i]がr1[0]と原点を結んだ直線上に存在するか(ポアンカレ断面)
            cross_val = r1_0[0] * r1_curr[1] - r1_0[1] * r1_curr[0]
            if abs(cross_val) < 1e-6:
                # 各状態量が十分に近いか
                diff_r = np.linalg.norm(r1_curr - r1_0)
                diff_v = np.linalg.norm(v_curr - v_0)
                if diff_r < 0.001 and diff_v < 0.001:
                    return (r1_0, v_0)

        diff = r1_curr - r_others
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        # クーロンの法則
        force_vecs = K * m1_mag * m_others[:, np.newaxis] * diff / dist**3
        total_force = np.sum(force_vecs, axis=0)

        a_next = total_force / mass

        # 速度と位置の更新 (velocity verlet法)
        r1_curr = r1_curr + v_curr * dt + a_curr * (dt**2) / 2
        v_curr = v_curr + (a_curr + a_next) * dt / 2
        a_curr = a_next

    return np.empty(0)


x = np.arange(0.0, 1.5, 0.1)
y = np.arange(0.0, 1.5, 0.1)
vx = np.arange(0.0, 2.0, 0.1)
vy = np.arange(0.0, 2.0, 0.1)

# パラメータの組み合わせを作る
params = itertools.product(x, y, vx, vy)

with ProcessPoolExecutor() as executor:
    results = list(filter(lambda args: args.len() == 4, executor.map(calc, params)))

print(results)