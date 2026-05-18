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
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.3)

    plot_window(d_window=d_window, ax=axes[0])
    canonical_dual_window = compute_dual_window(d_window)
    plot_window(d_window=canonical_dual_window, ax=axes[2])

    def update(_=None):
        nu   = int(s_nu.val)
        nu_0 = nu % beta
        # mult = np.exp(1j * 2 * np.pi * s_mult.val)
        mult = s_mult.val

        zak_new_window = zak_transform_fast(d_window=d_window).copy()
        zak_new_window[:, nu] *= mult
        zak_new_window[:, nu_0] *= mult
        zak_new_window -= zak_transform_fast(d_window=d_window).copy()
        # new_window = zak_inverse(zak_new_window) + canonical_dual_window
        new_window = zak_inverse(zak_new_window)
        
        zak_g = zak_transform_fast(d_window=d_window)
        test_window = np.zeros_like(new_window)
        
        for j in range(L):
            j_0 = j % alpha
            m = j // alpha
            test_window[j] =  ((mult - 1) / alpha_t) * ((np.exp(1j * (2 * np.pi / alpha_t) * m * nu) * zak_g[j_0, nu]) + (np.exp(1j * (2 * np.pi / alpha_t) * m * nu_0) * zak_g[j_0, nu_0]))
        

        axes[1].cla()
        # plot_window(d_window=new_window, ax=axes[1])
        # plot_window(d_window=test_window - new_window, ax=axes[1], color='green')
        # print(np.max(new_window), "et", np.min(new_window))
        # plot_window(d_window=discretize_window(lambda t:0.4 * np.cos(2 * np.pi * 5 *t)) - new_window, ax=axes[1], color='green')
        # plot_window(d_window=discretize_window(lambda t: 0.4 + 0.05 * np.cos(2 * np.pi * 10 *t)) - new_window, ax=axes[1], color='green')
        plot_window(d_window=compute_dual_window(test_window + d_window), ax=axes[1])
        plot_window(d_window=canonical_dual_window, ax=axes[1], color='purple')
        

        # cn_window     = plot_cn(d_window=d_window)
        # cn_new_window = plot_cn(d_window=new_window)
        # print(np.abs(cn_window - cn_new_window))

        fig.canvas.draw_idle()

    ax_nu   = plt.axes([0.15, 0.07, 0.72, 0.03])
    ax_mult = plt.axes([0.15, 0.03, 0.72, 0.03])
    s_nu    = Slider(ax_nu,   "nu",   beta,   alpha_t - 1, valinit=15,   valstep=1)
    s_mult  = Slider(ax_mult, "mult", -20, 20,       valinit=5.0, valstep=0.1)

    s_nu.on_changed(update)
    s_mult.on_changed(update)

    update()  # affichage initial
    plt.show()
