# @title 周期解を求める(Gemini修正版)
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor
from numba import njit

K = 6.33 * (10**4)
R_OTHERS = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
M1_MAG = -24 * (10**-5)
M_OTHERS = np.array([18 * (10**-5), 18 * (10**-5), 8 * (10**-5), 2 * (10**-5)])
MASS = 0.006
DT = 0.005
STEPS = 7000


@njit(cache=True, parallel=True)
def get_acceleration(r1):
    force_total = np.zeros(2)
    for i in range(4):
        diff = r1 - R_OTHERS[i]
        dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
        # クーロンの法則: F = K * m1 * m2 * r / r^3
        f_mag = K * M1_MAG * M_OTHERS[i] / (dist**3 + 1e-9)  # ゼロ除算防止
        force_total += f_mag * diff
    return force_total / MASS


@njit(cache=True, parallel=True)
def calc(params):
    r1_0 = np.array([params[0], params[1]])
    v_0 = np.array([params[2], params[3]])

    r_curr = r1_0.copy()
    v_curr = v_0.copy()
    a_curr = get_acceleration(r_curr)

    for i in range(STEPS):
        # Velocity Verlet法
        r_next = r_curr + v_curr * DT + 0.5 * a_curr * (DT**2)
        a_next = get_acceleration(r_next)
        v_next = v_curr + 0.5 * (a_curr + a_next) * DT

        # 周期性の判定 (ある程度時間が経過してから)
        if i > 50:
            # ポアンカレ断面
            cross_val = r1_0[0] * r_next[1] - r1_0[1] * r_next[0]
            if abs(cross_val) < 1e-5:
                diff_r = np.sqrt(np.sum((r_next - r1_0) ** 2))
                diff_v = np.sqrt(np.sum((v_next - v_0) ** 2))
                if diff_r < 0.5 and diff_v < 5.0:
                    return np.array(params)
                else:
                    return np.zeros(4)

        r_curr, v_curr, a_curr = r_next, v_next, a_next

    # 見つからなかった場合ゼロ埋め
    return np.zeros(4)


def main():
    x = np.arange(0.1, 3.1, 0.2)
    y = np.array([0.0])
    vx = np.arange(-0.1, 1.1, 0.2)
    vy = np.arange(0.1, 1.1, 0.2)

    params_list = list(itertools.product(x, y, vx, vy))
    print(f"Total combinations: {len(params_list)}")

    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(calc, params_list, chunksize=100):
            if np.any(res):  # 全て0でなければ（周期解が見つかれば）追加
                results.append(res)

    print("Found periodic solutions (initial states):")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()