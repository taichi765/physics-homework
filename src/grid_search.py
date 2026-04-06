# @title 周期解を求める(Gemini修正版)
from numba.misc.special import prange
import time
import numpy as np
import itertools
from numba import njit
from shared import get_acceleration, DT, STEPS


@njit(cache=True, parallel=False)
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
                if diff_r < 0.5 and diff_v < 1:
                    return (i, r_next, v_next, diff_r, diff_v)
                else:
                    return None

        r_curr, v_curr, a_curr = r_next, v_next, a_next


def main():
    start = time.time()

    x = np.arange(1.0, 4.0, 0.2)
    y = np.array([0.0])
    vx = np.arange(-1.1, 2.1, 0.1)
    vy = np.arange(0.1, 2.0, 0.1)

    params_list = list(itertools.product(x, y, vx, vy))
    n = len(params_list)
    print(f"Total combinations: {n}")

    results = []
    for i in prange(n):
        p = params_list[i]
        res = calc(p)
        if res is not None:
            results.append((p, res))

    print("Found periodic solutions (initial states):")
    for res in results:
        (rx_0, ry_0, vx_0, vy_0), (t, r_i, v_i, diff_r, diff_v) = res
        print(
            f"r_0:[{rx_0, ry_0}]\nv_0:[{vx_0, vy_0}]\nt:{t}\nr_i:{r_i}\nv_i:{v_i}\ndiff_r:{diff_r}\ndiff_v:{diff_v}\n"
        )

    print(f"elapsed: {time.time() - start}s")


if __name__ == "__main__":
    main()
