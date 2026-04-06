from matplotlib.animation import ArtistAnimation
import numpy as np
import matplotlib.pyplot as plt
from shared import get_acceleration, DT, STEPS
import matplotlib

matplotlib.use("QtAgg")


def calc_and_plot(params):
    rs = []
    vs = []

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

        r_curr, v_curr, a_curr = r_next, v_next, a_next
        rs.append(r_curr)
        vs.append(v_curr)

    rs = np.array(rs)
    vs = np.array(vs)
    fig = plt.figure(figsize=(2, 1))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(1, 1, "ro")
    ax1.plot(-1, 1, "ro")
    ax1.plot(-1, -1, "ro")
    ax1.plot(1, -1, "ro")
    ax1.grid()
    ax1.set_aspect("equal")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("x-y")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.grid()
    ax2.set_aspect("equal")
    ax2.set_xlabel("vx")
    ax2.set_ylabel("vy")
    ax2.set_title("vx-vy")

    xy_artists = []
    vxvy_artists = []

    for i in range(STEPS):
        xy_art = ax1.plot(rs[:i, 0], rs[:i, 1], "b")
        xy_artists.append(xy_art)
        vxvy_art = ax2.plot(vs[:i, 0], vs[:i, 1], "b")
        vxvy_artists.append(vxvy_art)

    xy_anim = ArtistAnimation(fig, xy_artists, interval=50, blit=True)
    # vxvy_anim = ArtistAnimation(fig, vxvy_artists)

    xy_anim.save("anim.mp4", writer="ffmpeg", fps=30)


def main():
    calc_and_plot(np.array([1.2, 0.0, -0.6, 0.2]))


if __name__ == "__main__":
    main()
