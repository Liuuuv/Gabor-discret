import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
from signal_test import signal_test, plot_time_frequencies_reference
from reconstitution import fft, fstdft, plot_fstdft
from dual_frame import compute_dual_window
from zak import compute_alternate_dual_window, zak_transform, zak_transform_fast, zak_inverse, plot_zak_transform
from base_orth import approximate_window_from_dual_dir
from dual_frame import plot_window
from tools import*
from config import*













# def build_orthonormal_xi(Zg, xi):

#     basis = {}

#     for k in range(alpha):
#         for nu0 in range(beta):

#             indices = [nu0 + m*beta for m in range(1, q)]

#             # coefficients c
#             c = np.array([Zg[k, n] / Zg[k, nu0] for n in indices])

#             norm_c2 = np.sum(np.abs(c)**2)

#             # vecteur correcteur
#             v = sum(np.conj(c[i]) * xi[(k, indices[i])] for i in range(len(indices)))
            

#             for i, n in enumerate(indices):

#                 eta = xi[(k, n)] - np.conj(c[i]) / norm_c2 * v
                
#                 if (eta == 0).all():
#                     print(k,nu0, "AAAA")

#                 e = eta / np.linalg.norm(eta)

#                 basis[(k, n)] = e

#     return basis




def approximate_window_from_k(d_test_window, ax_to_plot=None, ax_phase=None, basis=None, fig=None):
    # if basis is None:
    #     zak_g = zak_transform_fast(d_window)
        
        
    #     if q == 2:
    #         xi_ = xi.copy()
    #         for k,l in xi.keys():
    #             xi_[(k,l)] /= scalar_product(xi[(k,l)], xi[(k,l)]) ** 0.5
    #         basis = xi_
    #     else:
    #         basis = build_orthonormal_xi(zak_g, xi)
    assert q==2
    chi = build_chi(zak_g=zak_g, orthonormal=True)
    
    K = np.arange(alpha)
    L_ = np.arange(beta)
    result_raw = np.zeros((alpha, beta), dtype=np.complex64)
    
    for k in K:
        for l in L_:
            result_raw[k, l] = scalar_product(chi[(k,l)], d_test_window)
            # result_raw[k, l] = scalar_product(xi[(k,l + beta)], d_test_window)
    
    if ax_to_plot:
        result = result_raw.copy()
        result = np.abs(result)
        # result = np.log(result)
        result = np.transpose(result)
        mesh = ax_to_plot.pcolormesh(K, L_, result)
        # m = axes[2].heatmap(result, cmap='dusk')
        ax_to_plot.set_title("Module des coefficients dans K")
        # cbar = fig.colorbar(mesh, ax=ax_to_plot, label='Valeur')
        cbar = fig.colorbar(mesh, ax=ax_to_plot)
    
    if ax_phase is not None:
        result = result_raw.copy()
        mask = (np.abs(result) < 10e-5) ## contrer les présupposées erreurs numériques
        result[mask] = np.nan
        phase = np.angle(result)
        phase = np.transpose(phase)
        mesh_phase = ax_phase.pcolormesh(K, L_, phase, shading='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        # plt.colorbar(mesh_phase, ax=ax_phase, label='Phase [rad]')
        plt.colorbar(mesh_phase, ax=ax_phase)
        ax_phase.set_title("Phase des coefficients dans K")
    
    reconstructed = np.zeros(L, dtype=np.complex64)
    for k in K:
        for l in L_:
            reconstructed += result_raw[k, l] * chi[(k,l)]
            # reconstructed += result_raw[k, l] * xi[(k,l + beta)]
    
    return reconstructed


def testing(d_test_window, ax_to_plot=None, ax_phase=None, basis=None, fig=None):
    assert q==2
    chi = build_chi(zak_g=zak_g, orthonormal=True)
    
    K = np.arange(alpha)
    L_ = np.arange(beta)
    result_raw_window = np.zeros((alpha, beta), dtype=np.complex64)
    
    for k in K:
        for l in L_:
            result_raw_window[k, l] = scalar_product(chi[(k,l)], d_window)
            
    result_raw_dual = np.zeros((alpha, beta), dtype=np.complex64)
    
    for k in K:
        for l in L_:
            result_raw_dual[k, l] = scalar_product(chi[(k,l)], d_window)
    
    result_raw_window /= np.max(np.abs(result_raw_window))
    result_raw_dual /= np.max(np.abs(result_raw_dual))
    result_raw = result_raw_window - np.conj(result_raw_dual)
    
    if ax_to_plot:
        result = result_raw.copy()
        result = np.abs(result)
        # result = np.log(result)
        result = np.transpose(result)
        mesh = ax_to_plot.pcolormesh(K, L_, result)
        # m = axes[2].heatmap(result, cmap='dusk')
        ax_to_plot.set_title("Module des coefficients dans K dsdsdsd")
        # cbar = fig.colorbar(mesh, ax=ax_to_plot, label='Valeur')
        cbar = fig.colorbar(mesh, ax=ax_to_plot)
    
    if ax_phase is not None:
        result = result_raw.copy()
        mask = (np.abs(result) < 10e-5) ## contrer les présupposées erreurs numériques
        result[mask] = np.nan
        phase = np.angle(result)
        phase = np.transpose(phase)
        mesh_phase = ax_phase.pcolormesh(K, L_, phase, shading='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        # plt.colorbar(mesh_phase, ax=ax_phase, label='Phase [rad]')
        plt.colorbar(mesh_phase, ax=ax_phase)
        ax_phase.set_title("Phase des coefficients dans K dsds")
    
    reconstructed = np.zeros(L, dtype=np.complex64)
    for k in K:
        for l in L_:
            reconstructed += result_raw[k, l] * chi[(k,l)]
            # reconstructed += result_raw[k, l] * xi[(k,l + beta)]
    
    return reconstructed



if __name__ == "__main__":
    fig, axes = plt.subplots(10, 1, figsize=(14, 10)) ## changer 1er argument accordement
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.8     # Espace entre les graphiques
    )
    
    support_tolerance = 5*10e-7
    
    
    
    zak_g = zak_transform_fast(d_window)

    xi = build_xi(zak_g=zak_g)

    basis = build_orthonormal_xi(zak_g, xi)
    
    # print("basis.keys()", basis.keys())
    
    # plot_window(d_window, axes[0], label="Fenêtre")
    
    canonical_dual_window = compute_dual_window(d_window)
    # plot_window(d_dual_window, axes[1], label=f"duale canonique, support {get_window_support(d_dual_window, support_tolerance)}")
    # plot_window(d_dual_window, axes[1], label=f"Duale canonique")
    
    
    ########## DUALE ORTH TOUT 1 ##########
    # test = np.zeros(L, dtype=np.complex128)
    # for j in range(alpha):
    #     # for nu in range(beta, alpha_t):
    #     test += xi[(j,beta+5)]
    # plot_window(test, axes[10], label="duale à la main")
    #######################################
    
    
    # d_test_window = discretize_window(window=lambda t: np.sin(2 * np.pi * t * 19))
    # d_test_window = discretize_window(window=lambda t: np.sin(2 * np.pi * t * 5) * np.exp(- (t/0.1)**2))
    
    
    
    exp_part = discretize_window(window=lambda t: (1 - np.exp(-(t/0.05)**2)))
    # exp_part = discretize_window(window=lambda t: (1 - (np.cos(2 * np.pi * t * 2) ** 2) * np.exp(- (t/0.15)**2)))
    # exp_part = discretize_window(window=lambda t: 1 + t - t)
    d_test_window = -canonical_dual_window * exp_part
    # d_test_window += discretize_window(window=lambda t: 1 * np.exp(- (t/0.05)**2)) * np.max(d_dual_window)
    # d_test_window = exp_part
    
    
    # d_test_window = discretize_window(window=lambda t: np.exp(2j * np.pi * t * 5))
    d_test_window = discretize_window(window=lambda t: np.sin(2 * np.pi * t * 5))
    # d_test_window = discretize_window(window=lambda t: t-t + 1)
    # d_test_window = d_window
    
    # from temp import generate_zero_moments_function
    # d_test_window_ = generate_zero_moments_function(12, np.linspace(-5, 5, L), sigma=0.05)
    # d_test_window = d_test_window_.copy()
    # d_test_window[:L//2] = d_test_window_[L//2:]
    # d_test_window[L//2:] = d_test_window_[:L//2]
    # d_test_window - d_dual_window
    
    
    # plot_window(exp_part, axes[2], label=f"Modulation")
    plot_window(d_test_window, axes[0], label="Fenêtre à approcher")
    
    # plot_window(d_test_window + d_dual_window, axes[4], label="sommé")
    # reconstructed = approximate_window_from_dual_dir(signal, ax_to_plot=axes[2], fig=fig, ax_phase=axes[1])
    reconstructed = approximate_window_from_k(canonical_dual_window, ax_to_plot=axes[2], fig=fig, ax_phase=axes[1])
    plot_window(reconstructed, axes[3], label=f"Approchée dans K^\perp")
    
    # plot_cn(axes[7])
    # plot_window(reconstructed + d_dual_window, axes[7], label=f"sommé approché, support {get_window_support(reconstructed + d_dual_window, support_tolerance)}")
    
    plot_window((reconstructed - canonical_dual_window), axes[4], label=f"Différence")
    
    # chi = build_chi()
    # plot_window(chi[(5,5)], axes[7], label=f"chi")
    
    # aa = canonical_dual_window + approximate_compact_support_iter(0.1, 100)
    
    # d_window /= scalar_product(d_window, d_window) ** 0.5
    # canonical_dual_window /= scalar_product(canonical_dual_window, canonical_dual_window) ** 0.5
    # canonical_dual_window *= 1j
    reconstructed_k = approximate_window_from_k(d_window, ax_to_plot=axes[6], fig=fig, ax_phase=axes[5])
    
    
    
    plot_window(reconstructed_k, axes[7], label=f"Approchée dans K")
    # plot_window(d_dual_window, axes[7], label=f"zdzzd")
    
    
    testing(d_window, ax_to_plot=axes[8], ax_phase=axes[9], fig=fig)
    # plot_window((reconstructed_k - d_window), axes[8], label=f"Différence")
    
    
    
    
    
    # reconstructed = test_calculs(d_test_window, ax_to_plot=axes[9], fig=fig, ax_phase=axes[10])
    
    
    # plot_fft(reconstructed, axes[9])
    # plot_fft(basis[(5,beta+5)], axes[9])
    # plot_fft(xi[(7,beta+5)], axes[9])
    # plot_fft(reconstructed - d_test_window, axes[9], label="diff", ylog=True)
    
    # plot_fft(reconstructed, axes[10], label="reconstruit", ylog=True)
    
    # plot_fft(test, axes[9])
    # plot_fstdft(reconstructed - d_test_window, ax=axes[9], d_window=d_window, plot_ref=False)
    
    
    # test = reconstructed - d_test_window + discretize_window(window=lambda t: 0.00075 * (np.sin(2 * np.pi * t * 15) + np.sin(2 * np.pi * t * 25)))
    # plot_window(test, axes[10], label=f"+sin")
    # plot_cn(d_window=d_window, ax=axes[9])
    
    # d_window = compute_tight_frame(d_window)[1]
    
    # plot_fft(xi[(7,beta+5)], axes[10])
    
    # plot_fstdft(reconstructed, axes[9], plot_ref=False)
    # print(np.min(np.abs(d_dual_window)))
    # print()
    
    ######### PLOT BASIS ##########
    # for i in range(len(axes)-5):
    #     plot_window(basis[(0,i+beta)], axes[i+5], label=f"basis, nu_0={i}", custom_y_lim=0.1)
    ###############################
    
    
    ########## PLOT DIFF ##########
    # test = dual_dir_base_vec(5, 1)
    # plot_window(test, axes[2], label="test")
    
    # plot_window(np.abs(xi[(5,1+beta)] - test), axes[3], label="diff")
    ###############################
    
    
    
    # plot_window(basis[(5,1+beta)], axes[3], label="orthonormée")
    
    # fig = plt.figure(figsize=(16, 10))
    # ax3d = fig.add_subplot(projection='3d')
    # plot_zak_transform(xi[(5,1+beta)], ax3d, label="xi base")
    
    # fig = plt.figure(figsize=(16, 10))
    # ax3d = fig.add_subplot(projection='3d')
    # plot_zak_transform(test, ax3d, label="xi avant")
    
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


