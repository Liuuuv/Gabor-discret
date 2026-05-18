import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from config import*
from tools import*
from dual_frame import compute_dual_window, compute_tight_frame, construct_operator_matrix, plot_window
from methode_iterative import approximate_compact_support_iter
# from zak import plot_zak_transform, dual_dir_base_vec
from zak_tools import*
from decomposition_K_Kperp import*


cmap = plt.cm.bone  # 'plasma', 'inferno', 'jet', 'rainbow'

num_colors: int = 4

# Fonction qui prend un entier et retourne une couleur
def get_color_from_int(value, vmin=0, vmax=num_colors):
    """Convertit un entier en une couleur continue de la colormap"""
    # Normaliser la valeur entre 0 et 1
    normalized = (value - vmin) / (vmax - vmin)
    return cmap(normalized)

def plot_window_empty(window_, ax, is_discrete=True, label="", custom_y_lim=0.0, color_int: int=0):
    # if is_discrete:
    #     window = window_.copy()
    #     window[:L//2] = window_[L//2:]
    #     window[L//2:] = window_[:L//2]
    # else:
    #     window = window_
    window = window_.copy()
    window[:L//2] = window_[L//2:]
    window[L//2:] = window_[:L//2]
    
    
    # ax.plot(np.linspace(-0.5,0.5,L), np.real(window), color='blue', alpha=0.7, linewidth=1.0)
    ax.plot(np.linspace(-0.5,0.5,L), np.real(window), color=get_color_from_int(i, vmin=0, vmax=5), alpha=0.7, linewidth=1.0)
    # ax.plot(np.linspace(-0.5,0.5,L), np.imag(window), color='red', alpha=0.7, linewidth=1.0)
    # # ax.set_xlabel("Progression")
    # # ax.set_ylabel("Amplitude")
    # ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0, x=None, y=None, tight=True)
    
    # axes[ax_index].plot(np.linspace(-0.5,0.5,L), discretize_window(window, True))
    ax.set_title(label)
    if custom_y_lim:
        ax.set_ylim(-custom_y_lim, custom_y_lim)

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(14, 10)) ## changer 1er argument accordement
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.8     # Espace entre les graphiques
    )

    
    canonical_dual_window = compute_dual_window(d_window)
    
    
    test_kperp = np.zeros(L, dtype=np.complex128)
    # for j in range(alpha):
    #     # for nu in range(beta, alpha_t):
    #     test_kperp += xi[(j,beta+5)] * 0.001 + xi[(j,beta+2)] * 0.001
    
    
    
    sigmas = np.linspace(0.05, 0.2, num_colors)
    for i in range(num_colors):
        
        sigma = sigmas[num_colors-1-i]
        
        # 0.01, 0.2
        # exp_part = discretize_window(window=lambda t: (1 - np.exp(-(t/sigma)**2)))
        # test_kperp = approximate_window_from_dual_dir(-canonical_dual_window * exp_part)
        
        ind_part = discretize_window(window=ind_zero(sigma))
        test_kperp = approximate_window_from_dual_dir(-canonical_dual_window * ind_part)
        
        plot_window_empty(canonical_dual_window + test_kperp, axes[0], color_int=i)
    
    
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()
