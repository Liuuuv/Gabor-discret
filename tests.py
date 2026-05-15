import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from config import*
from tools import*
from methode_iterative import approximate_compact_support_iter
# from zak import plot_zak_transform, dual_dir_base_vec
from base_import import*
from zak_tools import*
from zak import*
from decomposition_K_Kperp import*


from matplotlib.widgets import Slider

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.3)

    plot_window(d_window=d_window, ax=axes[0])

    def update(_=None):
        nu   = int(s_nu.val)
        mult = s_mult.val

        zak_new_window = zak_transform_fast(d_window=d_window).copy()
        zak_new_window[:, nu]        *= mult
        zak_new_window[:, beta + nu] *= mult
        new_window = zak_inverse(zak_new_window)

        axes[1].cla()
        plot_window(d_window=new_window, ax=axes[1])

        cn_window     = plot_cn(d_window=d_window)
        cn_new_window = plot_cn(d_window=new_window)
        print(np.abs(cn_window - cn_new_window))

        fig.canvas.draw_idle()

    ax_nu   = plt.axes([0.15, 0.07, 0.72, 0.03])
    ax_mult = plt.axes([0.15, 0.03, 0.72, 0.03])
    s_nu    = Slider(ax_nu,   "nu",   0,   beta - 1, valinit=5,   valstep=1)
    s_mult  = Slider(ax_mult, "mult", -20, 20,       valinit=5.0, valstep=0.1)

    s_nu.on_changed(update)
    s_mult.on_changed(update)

    update()  # affichage initial
    plt.show()