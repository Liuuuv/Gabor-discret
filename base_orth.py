import librosa
import librosa.display
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
    chi = build_chi(zak_g=zak_g)
    
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

def test_calculs(d_test_window, ax_to_plot=None, ax_phase=None, basis=None, fig=None):
    # if basis is None:
    #     zak_g = zak_transform_fast(d_window)
    #     xi = build_xi(zak_g=zak_g)
        
    #     if q == 2:
    #         xi_ = xi.copy()
    #         for k,l in xi.keys():
    #             xi_[(k,l)] /= scalar_product(xi[(k,l)], xi[(k,l)]) ** 0.5
    #         basis = xi_
    #     else:
    #         basis = build_orthonormal_xi(zak_g, xi)
    zak_g = zak_transform_fast(d_window)
    xi = build_xi(zak_g=zak_g)
    
    
    K = np.arange(alpha)
    L_ = np.arange(alpha_t - beta)
    result_raw = np.zeros((alpha, alpha_t - beta), dtype=np.complex64)
    
    for k in K:
        for l in L_:
            nu_0 = l % beta
            result_raw[k, l] = (1/alpha_t) * (zak_transform(d_test_window, k, l + beta) - (zak_g[k, l + beta]/zak_g[k, nu_0]) * zak_transform(d_test_window, k, nu_0))
            result_raw[k, l] /= scalar_product(xi[(k,l+beta)], xi[(k,l+beta)]) ** 0.5
            # result_raw[k, l] = scalar_product(xi[(k,l + beta)], d_test_window)
    
    if ax_to_plot:
        result = result_raw.copy()
        result = np.abs(result)
        # result = np.log(result)
        result = np.transpose(result)
        mesh = ax_to_plot.pcolormesh(K, L_, result)
        # m = axes[2].heatmap(result, cmap='dusk')
        ax_to_plot.set_title("Module des coefficients")
        cbar = fig.colorbar(mesh, ax=ax_to_plot, label='Valeur')
    
    if ax_phase is not None:
        result = result_raw.copy()
        mask = (np.abs(result) < 10e-5) ## contrer les présupposées erreurs numériques
        result[mask] = 0
        phase = np.angle(result)
        phase = np.transpose(phase)
        mesh_phase = ax_phase.pcolormesh(K, L_, phase, shading='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(mesh_phase, ax=ax_phase, label='Phase [rad]')
        ax_phase.set_title("Phase des coefficients")
    
    # reconstructed = np.zeros(L, dtype=np.complex64)
    # for k in K:
    #     for l in L_:
    #         reconstructed += result_raw[k, l] * basis[(k,l + beta)]
    #         # reconstructed += result_raw[k, l] * xi[(k,l + beta)]
    
    # return reconstructed

def support_window(f, tol=1e-10):
    """
    Retourne les indices min et max où |f| > tol.
    """
    f_ = f.copy()
    f_[:L//2] = f[L//2:]
    f_[L//2:] = f[:L//2]
    non_zero = np.where(np.abs(f_) > tol)[0]
    if len(non_zero) == 0:
        return np.inf  # fonction nulle partout
    
    return non_zero[0], non_zero[-1]

def get_window_support(f, tol):
    return round(max(support_window(f, tol))/L - 0.5, 5)






# import numpy as np
# from scipy.linalg import lstsq

# def approximate_in_K_perp(gamma, I, basis=None, d_window=None, q=2):
#     """
#     Approche gamma (vecteur de taille L) par un élément de K^perp
#     sur les indices I (liste d'entiers).
    
#     Si basis est fourni, c'est un dict {(k, n): vecteur} des xi normalisés.
#     Sinon, on le calcule à partir de d_window.
    
#     Retourne :
#         x : vecteur dans K^perp (taille L)
#         coeffs : dict {(k, n): coefficient} pour la reconstruction
#         residu : norme de l'erreur sur I
#     """
#     L_global = len(gamma)
    
#     if basis is None:
#         if d_window is None:
#             raise ValueError("Il faut fournir soit basis soit d_window")
#         from zak_tools import zak_transform_fast, build_xi
#         from base_orth import build_orthonormal_xi  # ta fonction d'orthonormalisation
#         zak_g = zak_transform_fast(d_window)
#         xi = build_xi(zak_g=zak_g)
#         basis = build_orthonormal_xi(zak_g, xi)  # version orthonormée
    
#     # Récupérer la liste des (k, n) dans la base
#     keys = list(basis.keys())
#     m = len(keys)  # dimension de K^perp
    
#     # Construire la matrice A et le vecteur y
#     # On va indexer les colonnes par j = 0..m-1 correspondant à keys[j]
#     # et les lignes par i = 0..|I|-1 correspondant à I[i]
    
#     nI = len(I)
#     A = np.zeros((nI, m), dtype=np.complex128)
#     y = np.zeros(nI, dtype=np.complex128)
    
#     for i_idx, i in enumerate(I):
#         y[i_idx] = gamma[i]
#         for j_idx, (k, n) in enumerate(keys):
#             A[i_idx, j_idx] = basis[(k, n)][i]
    
#     # Résoudre le problème de moindres carrés A λ ≈ y
#     # Utiliser scipy.linalg.lstsq pour gérer les cas de rang déficient
#     λ, residus, rang, s = lstsq(A, y)
#     # λ est de taille m
    
#     # Reconstruire x
#     x = np.zeros(L_global, dtype=np.complex128)
#     coeffs = {}
#     for j_idx, (k, n) in enumerate(keys):
#         coeffs[(k, n)] = λ[j_idx]
#         x += λ[j_idx] * basis[(k, n)]
    
#     # Calcul de l'erreur sur I
#     erreur = x[I] - gamma[I]
#     residu = np.linalg.norm(erreur)
    
#     return x, coeffs, residu






if __name__ == "__main__":
    fig, axes = plt.subplots(6, 1, figsize=(14, 10)) ## changer 1er argument accordement
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.8     # Espace entre les graphiques
    )
    
    support_tolerance = 5*10e-7
    
    
    
    zak_g = zak_transform_fast(d_window)

    xi = build_xi(zak_g=zak_g)
    basis_xi = build_orthonormal_xi(zak_g, xi)
    chi = build_chi(orthonormal=True)
    
    # print("basis.keys()", basis.keys())
    
    plot_window(d_window, axes[0], label="Fenêtre")
    
    canonical_dual_window = compute_dual_window(d_window)
    # plot_window(d_dual_window, axes[1], label=f"duale canonique, support {get_window_support(d_dual_window, support_tolerance)}")
    plot_window(canonical_dual_window, axes[1], label=f"Duale canonique")
    
    
    ########## DUALE ORTH TOUT 1 ##########
    test_kperp = np.zeros(L, dtype=np.complex128)
    for j in range(alpha):
        # for nu in range(beta, alpha_t):
        test_kperp += xi[(j,beta+5)]
    plot_window(test_kperp, axes[2], label="xi à la main")
    #######################################
    
    plot_fstdft(test_kperp, axes[3], d_window=discretize_window(gaussian(0.05)), plot_ref=False, label="Test K^perp", linear=True)
    
    test_k = np.zeros(L, dtype=np.complex128)
    for j in range(alpha):
        # for nu in range(beta, alpha_t):
        test_k += chi[(j,5)]
    plot_window(test_k, axes[4], label="chi à la main")
    plot_fstdft(test_k, ax=axes[5], d_window=discretize_window(gaussian(0.05)), plot_ref=False, label="Test K", linear=True)
    
    # d_test_window = discretize_window(window=lambda t: np.sin(2 * np.pi * t * 19))
    # d_test_window = discretize_window(window=lambda t: np.sin(2 * np.pi * t * 5) * np.exp(- (t/0.1)**2))
    
    
    
    
    # d_test_window = discretize_window(window=lambda t: np.exp(2j * np.pi * t * 5))
    # d_test_window = discretize_window(window=lambda t: np.sin(2 * np.pi * t * 5))
    # d_test_window = discretize_window(window=lambda t: t-t + 1)
    # d_test_window = d_window
    
    # from temp import generate_zero_moments_function
    # d_test_window_ = generate_zero_moments_function(12, np.linspace(-5, 5, L), sigma=0.05)
    # d_test_window = d_test_window_.copy()
    # d_test_window[:L//2] = d_test_window_[L//2:]
    # d_test_window[L//2:] = d_test_window_[:L//2]
    # d_test_window - d_dual_window
    
    
    # plot_window(exp_part, axes[2], label=f"Modulation")
    # plot_window(d_test_window, axes[0], label="Fenêtre à approcher")
    
    # plot_window(d_test_window + d_dual_window, axes[4], label="sommé")
    # approximate_window_from_dual_dir(d_test_window, )
    # reconstructed = approximate_window_from_dual_dir(signal, ax_to_plot=axes[2], fig=fig, ax_phase=axes[1])
    # plot_window(reconstructed, axes[3], label=f"Approchée dans K^\perp")
    
    # plot_cn(axes[7])
    # plot_window(reconstructed + d_dual_window, axes[7], label=f"sommé approché, support {get_window_support(reconstructed + d_dual_window, support_tolerance)}")
    
    # plot_window((reconstructed - signal), axes[4], label=f"Différence")
    
    # chi = build_chi()
    # plot_window(chi[(5,5)], axes[7], label=f"chi")
    
    # aa = canonical_dual_window + approximate_compact_support_iter(0.1, 100)
    # reconstructed_k = approximate_window_from_k(signal, ax_to_plot=axes[6], fig=fig, ax_phase=axes[5])
    
    
    # plot_window(reconstructed_k, axes[7], label=f"Approchée dans K")
    # plot_window(d_dual_window, axes[7], label=f"zdzzd")
    
    # plot_window((reconstructed_k - signal), axes[8], label=f"Différence")
    
    
    
    
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














