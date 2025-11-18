"""
Solving the two-dimensional diffusion equation

Example acquired from
https://scipython.com/book/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
"""

import numpy as np
from .output import create_plot, output_plots
import matplotlib.pyplot as plt

def do_timestep(u_nm1, u, D, dt, dx2, dy2):
    """单步时间推进，用前向时间、中心空间差分。"""
    u[1:-1, 1:-1] = u_nm1[1:-1, 1:-1] + D * dt * (
        (u_nm1[2:, 1:-1] - 2 * u_nm1[1:-1, 1:-1] + u_nm1[:-2, 1:-1]) / dx2
        + (u_nm1[1:-1, 2:] - 2 * u_nm1[1:-1, 1:-1] + u_nm1[1:-1, :-2]) / dy2
    )

    u_nm1 = u.copy()
    return u_nm1, u


def solve(dx=0.1, dy=0.1, D=4.0):
    """
    求解 2D diffusion，并输出 4 个时间点的图像。

    参数
    ----
    dx, dy : float
        空间步长（mm）
    D : float
        热扩散系数（mm^2/s）
    """
    # plate size, mm
    w = h = 10.0

    # Initial cold temperature of square domain
    T_cold = 300

    # Initial hot temperature of circular disc at the center
    T_hot = 700

    # Number of discrete mesh points in X and Y directions
    nx, ny = int(w / dx), int(h / dy)

    # Computing a stable time step
    dx2, dy2 = dx * dx, dy * dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

    print(f"dx = {dx}, dy = {dy}, D = {D}")
    print("dt = {}".format(dt))

    # 初始场
    u0 = T_cold * np.ones((nx, ny))
    u = u0.copy()

    # Initial conditions - circle of radius r centred at (cx,cy) (mm)
    r = min(h, w) / 4.0
    cx = w / 2.0
    cy = h / 2.0
    r2 = r ** 2

    for i in range(nx):
        for j in range(ny):
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                u0[i, j] = T_hot

    # Number of timesteps
    nsteps = 101
    # Output 4 figures at these timesteps
    n_output = [0, 10, 50, 100]

    fig_counter = 0
    fig = plt.figure()
    last_im = None  # 用于在 output_plots 中添加 colorbar

    # Time loop
    for n in range(nsteps):
        u0, u = do_timestep(u0, u, D, dt, dx2, dy2)

        # Create figure
        if n in n_output:
            fig_counter += 1
            last_im = create_plot(
                fig=fig,
                u_slice=u.copy(),
                n=n,
                dt=dt,
                T_cold=T_cold,
                T_hot=T_hot,
                fig_counter=fig_counter,
            )

    # Plot output figures
    if last_im is not None:
        output_plots(fig, last_im)

    # 可选：返回最后的温度场，方便后续测试
    return u


if __name__ == "__main__":
    # 作为脚本运行时，使用默认参数
    solve()
