import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from config import*
from dual_frame import compute_dual_window, compute_tight_frame, construct_operator_matrix, plot_window


def zak_transform(d_window, j, nu):
    l = np.arange(0, alpha_t)
    return np.sum(d_window[j - alpha * l] * np.exp(1j*(2*np.pi/alpha_t) * nu * l), dtype=np.complex128)


def zak_transform_fast(d_window): ## BIEN? OUI MAIS CA MARCHE?
    """Version vectorisée de la transformée de Zak"""
    L = len(d_window)
    alpha_t = L // alpha
    
    # Reshape en matrice (alpha, alpha_t)
    signal_matrix = d_window.reshape(alpha, alpha_t, order='F')
    
    # FFT sur les colonnes
    zak = np.fft.fft(signal_matrix, axis=1)
    
    return zak

def zak_inverse(zak): ## zak sur alpha, alpha_t
    vec = np.zeros(L, np.complex64)
    for n in np.arange(L):
        for nu in np.arange(alpha_t):
            phase = np.exp(2j * np.pi * nu * (n//alpha) / alpha_t)
            vec[n] += (1/alpha_t) * zak[n%alpha,nu] * phase
    return vec

def compute_alternate_dual_window_orth(d_window):
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
    # choices[:,:] = -zak_canonical[:,alpha_t//q:]
    choices[:,7] = .01
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

def compute_alternate_dual_window(d_window, canonical_dual=None):
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

def plot_zak_transform(d_window, ax, label=""):
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
    # gs = fig.add_gridspec(4, 1, height_ratios=[2, 2, 2, 2])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.1, hspace=0.1)
    # fig.canvas.manager.full_screen_toggle()  # Mode plein écran
    plt.get_current_fig_manager().window.state('zoomed')
    windows = (1,4)
    
    ax3d = fig.add_subplot(*windows, 1, projection='3d')
    zak = plot_zak_transform(d_window, ax3d, label="Fenêtre")
    # plot_A(d_window, ax3d)
    
    
    
    
    
    ax3d = fig.add_subplot(*windows, 2, projection='3d')
    dual_window = compute_dual_window(d_window)
    plot_zak_transform(dual_window, ax3d, label="duale canonique")
    
    ax3d = fig.add_subplot(*windows, 3, projection='3d')
    # dual_window = compute_dual_window(d_window)
    alternate_dual_window_orth = compute_alternate_dual_window_orth(d_window)
    alternate_dual_window = dual_window + alternate_dual_window_orth
    
    plot_zak_transform(alternate_dual_window_orth, ax3d, label="duale orth")
    
    
    ax3d = fig.add_subplot(*windows, 4, projection='3d')
    plot_zak_transform(alternate_dual_window, ax3d, label="duale non canonique")
    
    
    fig, axes = plt.subplots(5, 1, figsize=(7, 5))
    plot_window(d_window, axes[0], "Fenetre")
    lim = max(np.min(np.abs(dual_window)), np.max(np.abs(dual_window)))
    plot_window(dual_window, axes[1], "canonical dual window", custom_y_lim=lim)
    plot_window(alternate_dual_window, axes[2], "alternate_dual_window",)
    orth = compute_alternate_dual_window_orth(d_window)
    plot_window(orth, axes[3], "orth",)
    # test = np.zeros(L, dtype=np.complex64)
    # for j in range(alpha):
    #     test += .01 * dual_dir_base_vec(j, 10)
    # plot_window(test, axes[3], "orth",)
    
    test = np.zeros(L, dtype=np.complex64)
    for j in range(alpha):
        test += .01 * dual_dir_base_vec(j, 19)
    plot_window(test, axes[4], "test")
    
    
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
    
    
    
    # ax3d = fig.add_subplot(*windows, 4, projection='3d')
    # plot_zak_transform(test, ax3d, label="test")
    
    
    
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
    