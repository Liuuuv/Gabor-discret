import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from config import*
from dual_frame import compute_dual_window, compute_tight_frame, construct_operator_matrix, plot_window
from zak_tools import*
from base_orth import build_xi







def compute_alternate_dual_window_orth(d_window):
    print("Computing alternate dual window")
    J = np.arange(alpha)
    NU = np.arange(alpha_t)
    
    # zak_g = np.zeros((alpha, alpha_t), dtype=np.complex128)
    # for j in J:
    #     for nu in NU:
    #         zak_g[j, nu] = zak_transform(d_window, j, nu)
    
    zak_g = zak_transform_fast(d_window)
    
    
    zak_gamma = np.zeros((alpha, alpha_t), dtype=np.complex128)
    
    ################ choix ################
    # choice = 5
    # zak_gamma[:,alpha_t//q:] = choice
    
    choices = np.zeros((alpha, alpha_t - alpha_t//q), np.complex64)
    len_nu = alpha_t - alpha_t//q
    # choices[::2,:] = 1
    # choices[1::2,:] = -1
    # for j in range(alpha):
    #     for nu in range(len_nu):
            
    #         choices[j,nu] = 0.0001 * nu + 0.001 * j
    # choices[-1,0] = 2
    
    canonical = compute_dual_window(d_window)
    
    zak_canonical = np.zeros((alpha, alpha_t), dtype=np.complex128)
    for j in J:
        for nu in NU:
            zak_canonical[j, nu] = zak_transform(canonical, j, nu)
    
    
    # print(zak_gamma)
    # choices[:,:] = -0.25*zak_canonical[:,alpha_t//q:]
    # choices[5,1] = 1
    choices[:,5] = 1
    zak_gamma[:,alpha_t//q:] = choices[:,:]
    ################ choix ################
    
    
    ## calculer contraintes
    for j in J:
        for nu in np.arange(alpha_t//q):
            # zak_gamma[j, nu] = - choices[j, nu] * np.conjugate(zak_g[j, nu - alpha_t//q] / zak_g[j,nu]) ## q=2
            # zak_gamma[j, nu] = - choice * np.conjugate(zak_g[j, nu - alpha_t//2] / zak_g[j,nu]) ## q=2, bandes
            l = np.arange(1,q)
            zak_gamma[j, nu] = (-1 / np.conjugate(zak_g[j,nu])) * np.sum(choices[j, nu + (l-1) * alpha_t//q] * np.conjugate(zak_g[j, nu + l * alpha_t//q]))
    
    return zak_inverse(zak_gamma)

def compute_alternate_dual_window(d_window, canonical_dual=None): ## passe par la zak inverse
    if canonical_dual is None:
        canonical_dual = compute_dual_window(d_window)
    
    orth = compute_alternate_dual_window_orth(d_window)
    # return orth
    return canonical_dual + orth
    

def dual_dir_base_vec(k, n):
    test = np.zeros(L, np.complex128)
    
    J = np.arange(alpha)
    NU = np.arange(alpha_t)
    zak = np.zeros((alpha, alpha_t), dtype=np.complex128)
    for j in J:
        for nu in NU:
            zak[j, nu] = zak_transform(d_window, j, nu)
    
    n += alpha_t//q
    l_0 = (n//beta)
    nu_0: int = n - l_0 * beta
    for j in np.arange(L):
        if j%alpha != k:
            continue
        l = j//alpha
        test[j] = - (1/alpha_t) * (np.conjugate(zak[j%alpha,n]) / np.conjugate(zak[j%alpha,nu_0])) * np.exp(2j * np.pi *l * (nu_0) / alpha_t) + (1/alpha_t) * np.exp(2j * np.pi *l * n / alpha_t)
    
    return test


# def plot_zak_transform(d_window, ax, label="") -> np.ndarray:
#     result_raw = np.zeros((alpha, alpha_t), dtype=np.complex128)
#     J = np.arange(alpha)
#     NU = np.arange(alpha_t)
    
#     for j in J:
#         for nu in NU:
#             result_raw[j, nu] = zak_transform(d_window, j, nu)
#     J, NU = np.meshgrid(J, NU)
    
#     # result[:,:] = 0
#     # result[:,0] = 1
    
    
#     print(np.max(np.imag(result)), np.min(np.imag(result)))
    
#     result = result_raw.copy()
#     result = np.abs(result)
#     result = np.transpose(result)
    
#     print("min", np.min(result))
    
#     ax.plot_surface(J, NU, result, cmap="viridis")
    
#     ax.set_xlabel("j")
#     ax.set_ylabel("nu")
#     ax.set_zlabel("Module de Zak")
#     ax.set_title(label)
#     return result_raw

def plot_zak_transform(d_window, ax, label="", bars=False):
    if bars:
        return plot_zak_transform_bars(d_window, ax, label="")
    print("Plotting zak transform", label)
    J = np.arange(alpha)
    NU = np.arange(alpha_t)
    
    result_raw = np.zeros((alpha, alpha_t), dtype=np.complex128)
    for j in J:
        for nu in NU:
            result_raw[j, nu] = zak_transform(d_window, j, nu)
    
    
    result = result_raw.copy()
    
    # Module pour la hauteur
    Z = np.abs(result)
    # Phase pour la couleur
    phase = np.angle(result)
    
    J, NU = np.meshgrid(J, NU)
    Z = np.transpose(Z)
    phase = np.transpose(phase)
    
    # Normalisation de la phase pour la couleur
    norm_phase = (phase + np.pi) / (2 * np.pi)  # entre 0 et 1
    
    surf = ax.plot_surface(J, NU, Z, facecolors=plt.cm.hsv(norm_phase), alpha=0.9)
    
    ax.set_xlabel("j")
    ax.set_ylabel("nu")
    ax.set_zlabel("Module")
    ax.set_title(f"{label}")
    
    # Barre de couleur pour la phase
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.hsv)
    mappable.set_array(phase)
    plt.colorbar(mappable, ax=ax, label="Phase [rad]")
    
    return result_raw


def plot_zak_transform_bars(d_window, ax, label=""):
    print("Plotting zak transform (bars)", label)
    J = np.arange(alpha)
    NU = np.arange(alpha_t)
    
    # Calcul de la transformée de Zak
    result_raw = np.zeros((alpha, alpha_t), dtype=np.complex128)
    for j in J:
        for nu in NU:
            result_raw[j, nu] = zak_transform(d_window, j, nu)
    
    # Module (hauteur des barres)
    Z = np.abs(result_raw)
    # Phase (pour la couleur)
    phase = np.angle(result_raw)
    
    # Préparation des coordonnées pour bar3d
    # On veut une barre pour chaque couple (j, nu)
    xpos, ypos = np.meshgrid(J, NU, indexing='ij')
    xpos = xpos.ravel()  # position x (j)
    ypos = ypos.ravel()  # position y (nu)
    zpos = np.zeros_like(xpos)  # base des barres à z=0
    
    # Largeur et profondeur des barres (ajustables)
    dx = 0.8 * np.ones_like(xpos)
    dy = 0.8 * np.ones_like(ypos)
    dz = Z.ravel()  # hauteur des barres
    
    # Couleur basée sur la phase normalisée entre 0 et 1
    norm_phase = (phase.ravel() + np.pi) / (2 * np.pi)
    colors = plt.cm.hsv(norm_phase)
    
    # Tracé des barres
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9, shade=True)
    
    ax.set_xlabel("j")
    ax.set_ylabel("nu")
    ax.set_zlabel("Module")
    ax.set_title(f"{label}")
    
    # Barre de couleur pour la phase
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.hsv)
    mappable.set_array(phase)
    plt.colorbar(mappable, ax=ax, label="Phase [rad]")
    
    return result_raw


def plot_A(d_window, ax):
    
    result = np.zeros((alpha, alpha_t), dtype=np.complex64)
    J = np.arange(alpha)
    NU = np.arange(alpha_t)
    
    for j in J:
        for nu in NU:
            result[j, nu] = A_p_eq_1(d_window, j, nu)
    
    J, NU = np.meshgrid(J, NU)
    result = np.abs(result)
    result = np.transpose(result)
    
    
    result /= np.max(result)
    print("min", np.min(result))
    
    ax.plot_surface(J, NU, result, cmap="viridis")
    ax.set_xlabel("j")
    ax.set_ylabel("nu")
    ax.set_zlabel("Module de A")

## cas p=1
def A_p_eq_1(d_window, j, nu): ## à constante près, corriger
    if p != 1:
        print("A_p_eq_1 APPELÉE ALORS QUE P != 1 !!")
    # l = 
    return np.sum(np.abs(zak_transform(d_window, j, nu - beta * l))**2 for l in np.arange(q))
    


if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 10))
    windows = (1,3) ## CHANGER ACCORDEMENT
    
    # gs = fig.add_gridspec(4, 1, height_ratios=[2, 2, 2, 2])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0.1)
    # fig.canvas.manager.full_screen_toggle()  # Mode plein écran
    plt.get_current_fig_manager().window.state('zoomed')
    
    
    
    # d_window = compute_tight_frame(d_window)[1]
    
    dual_window = compute_dual_window(d_window)
    ax3d = fig.add_subplot(*windows, 1, projection='3d')
    # zak = plot_zak_transform(d_window, ax3d, label="Fenêtre", bars=False)
    zak = plot_zak_transform(dual_window, ax3d, label="Fenêtre duale", bars=False)
    # plot_A(d_window, ax3d)
    
    
    
    # xi = build_xi()
    # test = np.zeros(L, dtype=np.complex64)
    # for j in range(alpha):
    #     for nu in range(alpha_t - beta):
    #         t = nu/(alpha_t - beta)
    #         t = (1-t) * np.pi/2 + t
    #         t *= (alpha_t - beta)
    #         # test += np.exp(2j * np.pi * t / alpha_t) * dual_dir_base_vec(j, 5)
    #         test += np.exp(2j * np.pi * t / alpha_t) * xi[(j, beta+5)]
    #         # test += np.exp(2j * np.pi * nu / alpha_t) * dual_dir_base_vec(j, 9)
    # test = dual_dir_base_vec(0, 5)
    
    
    
    
    
    ax3d = fig.add_subplot(*windows, 2, projection='3d')
    # dual_window = compute_dual_window(d_window)
    alternate_dual_window_orth = compute_alternate_dual_window_orth(d_window)
    # plot_zak_transform(dual_window, ax3d, label="duale canonique")
    plot_zak_transform(alternate_dual_window_orth, ax3d, label="\delta = 1")
    # plot_zak_transform(test, ax3d, label="test")
    
    
    
    
    ax3d = fig.add_subplot(*windows, 3, projection='3d')
    from methode_iterative import approximate_compact_support_iter
    approximate = approximate_compact_support_iter(0.1, 25)
    plot_zak_transform(approximate, ax3d, label="approximate")
    
    
    
    # ax3d = fig.add_subplot(*windows, 3, projection='3d')
    # dual_window = compute_dual_window(d_window)
    alternate_dual_window_orth = compute_alternate_dual_window_orth(d_window)
    alternate_dual_window = dual_window + alternate_dual_window_orth
    
    # plot_zak_transform(alternate_dual_window_orth, ax3d, label="Duale dans K^\perp")
    
    
    # ax3d = fig.add_subplot(*windows, 4, projection='3d')
    # plot_zak_transform(alternate_dual_window, ax3d, label="duale non canonique")
    
    
    fig, axes = plt.subplots(5, 1, figsize=(7, 5))
    plot_window(d_window, axes[0], label="Fenêtre")
    lim = max(np.min(np.abs(dual_window)), np.max(np.abs(dual_window)))
    plot_window(dual_window, axes[1], label="canonical dual window", custom_y_lim=lim)
    plot_window(alternate_dual_window, axes[2], label="alternate_dual_window",)
    orth = compute_alternate_dual_window_orth(d_window)
    plot_window(orth, axes[3], label="",)
    # test = np.zeros(L, dtype=np.complex64)
    # for j in range(alpha):
    #     test += .01 * dual_dir_base_vec(j, 10)
    # plot_window(test, axes[3], "orth",)
    
    
    # plot_window(test, axes[4], label="test")
    
    
    # test = np.zeros(L, np.complex128)
    # k = 10
    # n = 2
    
    # n += alpha_t//q
    # l_0 = (n//beta)
    # nu_0: int = n - l_0 * beta
    # for j in np.arange(L):
    #     if j%alpha != k:
    #         continue
    #     l = j//alpha
    #     test[j] = - (1/alpha_t) * (np.conjugate(zak[j%alpha,n]) / np.conjugate(zak[j%alpha,nu_0])) * np.exp(2j * np.pi *l * (nu_0) / alpha_t) + (1/alpha_t) * np.exp(2j * np.pi *l * n / alpha_t)
    
    
    # plot_window(test-alternate_dual_window_orth, axes[3], "test")
    
    
    
    
    
    
    
    plt.show()
    
    
    
    
    
    # num=400
    # parameters = np.linspace(0.0001, 0.5, num)
    
    # J = np.arange(alpha)
    # NU = np.arange(alpha_t)
    # mins = np.zeros(num)
    # for i, param in enumerate(parameters):
    #     print('i=',i)
        
    #     d_window_ = discretize_window(ind_zero(param))
    #     result = np.zeros((alpha, alpha_t), dtype=np.complex128)
        
    #     for j in J:
    #         for nu in NU:
    #             result[j, nu] = zak_transform(d_window_, j, nu)
    #     mins[i] = np.min(np.abs(result))
    
    # plt.plot(parameters, mins)
    # plt.grid()
    # plt.show()
    