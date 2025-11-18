# output.py

import matplotlib.pyplot as plt


def create_plot(fig, u_slice, n, dt, T_cold, T_hot, fig_counter):
    """
    在 fig 上生成一个子图，对应时间步 n。

    参数
    ----
    fig : matplotlib.figure.Figure
        整体的 Figure
    u_slice : 2D array
        当前时间步的温度场
    n : int
        当前 time step 编号
    dt : float
        时间步长
    T_cold, T_hot : float
        用于设定颜色条范围
    fig_counter : int
        用于决定子图位置（1~4）
    """
    # 220 + k 的写法和原代码保持一致：2 行 2 列第 k 张
    ax = fig.add_subplot(220 + fig_counter)
    im = ax.imshow(
        u_slice,
        cmap=plt.get_cmap('hot'),
        vmin=T_cold,
        vmax=T_hot,
    )
    ax.set_axis_off()
    ax.set_title('{:.1f} ms'.format(n * dt * 1000))

    return im


def output_plots(fig, im):
    """
    调整布局，添加 colorbar，并显示图像。
    """
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
