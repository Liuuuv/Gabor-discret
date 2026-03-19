import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from config import*
from tools import*
from dual_frame import compute_dual_window, compute_tight_frame, construct_operator_matrix, plot_window
# from zak import plot_zak_transform, dual_dir_base_vec
from zak_tools import*



def approximate_compact_support_iter(cutoff, num):
    canonical_dual = compute_dual_window(d_window)
    
    # cutoff = 0.076
    
    i_min = int(L * (cutoff))
    i_max = int(L * (1-cutoff))
    print("i_min, i_max", i_min, i_max)
    approximation = -canonical_dual.copy()
    approximation[i_min:L//2] = 0
    approximation[L//2:i_max] = 0
    # plot_window(approximation/np.max(approximation), axes[2], label="Approximation 0")
    
    # num = 400
    for i in range(num):
        
        approximation[i_min:L//2] = -canonical_dual[i_min:L//2]
        approximation[L//2:i_max] = -canonical_dual[L//2:i_max]
        approximation = approximate_window_from_dual_dir(approximation)
    
    return approximation


if __name__ == "__main__":
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 10)) ## changer 1er argument accordement
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.8     # Espace entre les graphiques
    )
    
    canonical_dual = compute_dual_window(d_window)
    
    cutoff = 0.1
    
    i_min = int(L * (cutoff))
    i_max = int(L * (1-cutoff))
    print("i_min, i_max", i_min, i_max)
    approximation = canonical_dual.copy()
    approximation[i_min:L//2] = 0
    approximation[L//2:i_max] = 0
    plot_window(approximation, axes[2], label="Tronquage")
    
    num = 100
    for i in range(num):
        
        approximation[i_min:L//2] = 0
        approximation[L//2:i_max] = 0
        approximation = approximate_window_from_dual_dir(approximation)
        
        # plot_window(approximation, axes[3+i], f"Approximation {i+1}")
        if i+1 == 5:
            plot_window(approximation, axes[3], label=f"Itération {i+1}-ième")
    
    
    plot_window(d_window, axes[0], label="Fenêtre")
    plot_window(canonical_dual, axes[1], label="Duale canonique")
    plot_window(approximation, axes[4], label=f"Itération {num}-ième")
    
    plt.show()


































