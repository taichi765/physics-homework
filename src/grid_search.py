# @title 周期解を求める(Gemini修正版)
import csv
from numba.misc.special import prange
import time
import numpy as np
import itertools
from numba import njit
from shared import DT, STEPS, get_acceleration_2, check_collision_2

DUMMY_RET = (-1, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0, 0.0)


@njit(cache=False, parallel=False)
def calc(rx_0, ry_0, vx_0, vy_0):
    r1_0 = np.array([rx_0, ry_0])
    v_0 = np.array([vx_0, vy_0])

    r_curr = r1_0.copy()
    v_curr = v_0.copy()
    a_curr = get_acceleration_2(r_curr)

    for i in range(STEPS):
        # Velocity Verlet法
        r_next = r_curr + v_curr * DT + 0.5 * a_curr * (DT**2)
        a_next = get_acceleration_2(r_next)
        v_next = v_curr + 0.5 * (a_curr + a_next) * DT

        if check_collision_2(r_next):
            return DUMMY_RET

        # 周期性の判定 (ある程度時間が経過してから)
        if i > 50:
            # ポアンカレ断面
            cross_val = r1_0[0] * r_next[1] - r1_0[1] * r_next[0]
            if abs(cross_val) < 1e-5:
                diff_r = np.sqrt(np.sum((r_next - r1_0) ** 2))
                diff_v = np.sqrt(np.sum((v_next - v_0) ** 2))
                if diff_r < 0.01 and diff_v < 0.1:
                    return (
                        i,
                        r_next,
                        v_next,
                        diff_r,
                        diff_v,
                    )
                else:
                    return DUMMY_RET

        r_curr, v_curr, a_curr = r_next, v_next, a_next

    raise Exception("ポアンカレ断面に一度も到達していません", rx_0, ry_0, vx_0, vy_0)


def main():
    start = time.time()

    x = np.arange(0.1, 1.5, 0.1)
    y = np.arange(0.1, 1.5, 0.1)
    vx = np.arange(-1, 1, 0.1)
    vy = np.arange(0.1, 1, 0.1)

    params_list = list(itertools.product(x, y, vx, vy))
    n = len(params_list)
    print(f"Total combinations: {n}")

    results = []
    for i in prange(n):  # ty:ignore[not-iterable]
        p = params_list[i]
        try:
            res = calc(p[0], p[1], p[2], p[3])
            if res[0] != -1:
                results.append((p, res))
        except Exception:
            1 + 1  # 無視する

    with open("res.csv", mode="w") as f:
        w = csv.writer(f)
        w.writerow(
            ["rx_0", "ry_0", "vx_0", "vy_0", "t", "r_i", "v_i", "diff_r", "diff_v"]
        )
        for res in results:
            (rx_0, ry_0, vx_0, vy_0), (t, r_i, v_i, diff_r, diff_v) = res
            w.writerow([rx_0, ry_0, vx_0, vy_0, t, r_i, v_i, diff_r, diff_v])

    print(f"elapsed: {time.time() - start}s")


if __name__ == "__main__":
    main()
