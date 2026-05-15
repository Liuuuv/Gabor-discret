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
from decomposition_K_Kperp import*


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(14, 10)) ## changer 1er argument accordement
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.2     # Espace entre les graphiques
    )
    
    plot_cn(d_window, axes[0], fig, ax_phase=axes[1])
    plt.show()
