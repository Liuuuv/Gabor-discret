import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from tools import*
from config import*
from base_import import*
from zak import*


def  compute_canonical_tight_window(d_window, alpha=alpha, beta=beta, bound=2):
    """Given a window g, computes and returns S^{-1/2}g

    Args:
        d_window (_type_): _description_
        alpha (_type_, optional): _description_. Defaults to alpha.
        beta (_type_, optional): _description_. Defaults to beta.
    """
    zak_g = zak_transform_fast(d_window=d_window, alpha=alpha, beta=beta)
    piecewise_zak = piecewise_scalar_zak_transform(d_window=d_window, alpha=alpha, beta=beta)
    c = np.sqrt(bound / alpha)
    new_zak = c * zak_g / np.sqrt(piecewise_zak)
    tight_window = zak_inverse(new_zak)
    return tight_window
    



if __name__ == "__main__":
    plot_shape = (3,1)
    fig, axes = create_subplots(plot_shape)
    
    # ax3d = add_3d_subplot(fig, plot_shape, 1)
    # zak = plot_zak_transform(ax=ax3d, zak_precomputed=piecewise_scalar_zak_transform(d_window=d_window))
    # print(np.min(zak))
    
    # ax3d = add_3d_subplot(fig, plot_shape, 2)
    # zak = plot_zak_transform(ax=ax3d, d_window=d_window, squared=True)
    # print(np.min(np.abs(zak)**2))
    
    
    
    plot_window(d_window=d_window, ax=axes[0])
    
    tight_window = compute_canonical_tight_window(d_window=d_window)
    plot_window(tight_window, ax=axes[1])
    
    ax3d = add_3d_subplot(fig, plot_shape, 3)
    zak = plot_zak_transform(ax=ax3d, d_window=tight_window, piecewise=True)
    # print(np.min(np.abs(zak)**2))
    
    print('norm squared', scalar_product(tight_window, tight_window))
    
    plt.show()




