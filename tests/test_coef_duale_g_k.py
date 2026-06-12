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



if __name__ == "__main__":
    fig, axes = plt.subplots(6, 1, figsize=(14, 10))
    plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.4)

    # plot_window(d_window=d_window, ax=axes[0])
    canonical_dual_window = compute_dual_window(d_window)
    # plot_window(d_window=canonical_dual_window, ax=axes[1])
    # plot_window(d_window=compute_dual_window(canonical_dual_window), ax=axes[2])
    
    approximate_window_from_k(d_window, ax_to_plot=axes[0], fig=fig, ax_phase=axes[1])
    approximate_window_from_k(canonical_dual_window, ax_to_plot=axes[2], fig=fig, ax_phase=axes[3])
    
    
    zak_g = zak_transform_fast(d_window)
    chi = build_chi(zak_g=zak_g, orthonormal=True)
    K = np.arange(alpha)
    L_ = np.arange(beta)
    d_window_k = np.zeros((alpha, beta), dtype=np.complex128)
    
    for k in K:
        for l in L_:
            d_window_k[k, l] = scalar_product(chi[(k,l)], d_window)
    
    
    
    canonical_dual_window_k = np.zeros((alpha, beta), dtype=np.complex128)
    
    for k in K:
        for l in L_:
            canonical_dual_window_k[k, l] = scalar_product(chi[(k,l)], canonical_dual_window)
    
    ## module
    test_k = d_window_k * np.conjugate(canonical_dual_window_k)
    test_k = np.transpose(test_k)
    test_k_module = np.abs(test_k)
    test_k_module *= L
    print(np.min(test_k_module), np.max(test_k_module))
    K = np.arange(alpha)
    L_ = np.arange(beta)
    mesh = axes[4].pcolormesh(K, L_, test_k_module)
    cbar = fig.colorbar(mesh, ax=axes[4])
    
    # phase
    ax_phase = axes[5]
    test_k_phase = test_k.copy()
    mask = (np.abs(test_k_phase) < 10e-5) ## contrer les présupposées erreurs numériques
    test_k_phase[mask] = np.nan
    phase = np.angle(test_k_phase)
    mesh_phase = ax_phase.pcolormesh(K, L_, phase, shading='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
    # plt.colorbar(mesh_phase, ax=ax_phase, label='Phase [rad]')
    plt.colorbar(mesh_phase, ax=ax_phase)
    ax_phase.set_title("Phase des coefficients dans K")
    

    plt.show()
