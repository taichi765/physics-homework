from datetime import datetime
from argparse import ArgumentParser
import logging
from matplotlib.animation import ArtistAnimation
import numpy as np
import matplotlib.pyplot as plt
from shared import DT, STEPS, get_acceleration_2
import matplotlib

matplotlib.use("QtAgg")


def calc_and_plot(params, args):
    rs = []
    vs = []

    r1_0 = np.array([params[0], params[1]])
    v_0 = np.array([params[2], params[3]])

    r_curr = r1_0.copy()
    v_curr = v_0.copy()
    a_curr = get_acceleration_2(r_curr)

    for i in range(STEPS):
        # Velocity Verlet法
        r_next = r_curr + v_curr * DT + 0.5 * a_curr * (DT**2)
        a_next = get_acceleration_2(r_next)
        v_next = v_curr + 0.5 * (a_curr + a_next) * DT

        r_curr, v_curr, a_curr = r_next, v_next, a_next
        rs.append(r_curr)
        vs.append(v_curr)

    rs = np.array(rs)
    vs = np.array(vs)
    fig = plt.figure(figsize=(10, 20))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(0.04, 0, "ro")
    ax1.plot(-0.04, 0, "ro")
    ax1.grid()
    ax1.set_aspect("equal")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("x-y")
    # ax1.set_xlim(-0.2, 0.2)
    # ax1.set_ylim(-0.2, 0.2)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.grid()
    ax2.set_aspect("equal")
    ax2.set_xlabel("vx")
    ax2.set_ylabel("vy")
    ax2.set_title("vx-vy")
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)

    if args.animation:
        xy_artists = []
        # vxvy_artists = []

        for i in range(STEPS):
            xy_art = ax1.plot(rs[:i, 0], rs[:i, 1], "b")
            xy_artists.append(xy_art)
            # vxvy_art = ax2.plot(vs[:i, 0], vs[:i, 1], "b")
            # vxvy_artists.append(vxvy_art)

        xy_anim = ArtistAnimation(fig, xy_artists, interval=1, blit=True)
        # vxvy_anim = ArtistAnimation(fig, vxvy_artists)

        if args.file:
            print("saving...")
            xy_anim.save(
                f"videos/{datetime.now().strftime('%Y-%m-%d__%H_%M_%S')}.mp4",
                writer="ffmpeg",
                fps=30,
            )
        else:
            plt.show()
    else:
        ax1.plot(rs[:, 0], rs[:, 1], "b")
        ax2.plot(vs[:, 0], vs[:, 1], "b")
        if args.file:
            plt.savefig(f"videos/{datetime.now().strftime('%Y-%m-%d__%H_%M_%S')}.mp4")
        else:
            plt.show()


def main():
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "-a", "--animation", action="store_true", help="アニメーションか一枚絵か"
    )
    parser.add_argument(
        "-f", "--file", action="store_true", help="結果をファイルに出力するか否か"
    )
    parser.add_argument(
        "-i",
        "--init-val",
        help="space-separated 初期値",
        type=lambda x: list(map(int, x.split(" "))),
    )

    args = parser.parse_args()
    init_val = None
    if args.init_val is None:
        init_val = np.array([1.4, 0.9, -0.1, 0.7])
    else:
        init_val = np.array(args.init_val)

    calc_and_plot(init_val, args)


if __name__ == "__main__":
    main()
