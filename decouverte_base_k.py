from signal_test import signal_test, plot_time_frequencies_reference
from config import*
from tools import*
from dual_frame import compute_dual_window, compute_tight_frame, construct_operator_matrix, plot_window
from methode_iterative import approximate_compact_support_iter
# from zak import plot_zak_transform, dual_dir_base_vec
from zak_tools import*
from decomposition_K_Kperp import*
from zak import*



def plot_chi_zak(d_window, k, n, ax):
    # print("Computing alternate dual window")
    J = np.arange(alpha)
    NU = np.arange(alpha_t)
    
    c_nk = plot_cn(d_window=d_window)
    
    # zak_g = np.zeros((alpha, alpha_t), dtype=np.complex128)
    # for j in J:
    #     for nu in NU:
    #         zak_g[j, nu] = zak_transform(d_window, j, nu)
    
    # zak_g = zak_transform_fast(d_window)
    
    
    zak_gamma = np.zeros((alpha, alpha_t), dtype=np.complex128)
    
    ################ choix ################
    # choice = 5
    # zak_gamma[:,alpha_t//q:] = choice
    
    choices = np.zeros((alpha, alpha_t//q), np.complex64)
    # choices[::2,:] = 1
    # choices[1::2,:] = -1
    # for j in range(alpha):
    #     for nu in range(len_nu):
            
    #         choices[j,nu] = 0.0001 * nu + 0.001 * j
    # choices[:,:] = 1
    
    # canonical = compute_dual_window(d_window)
    
    # zak_canonical = np.zeros((alpha, alpha_t), dtype=np.complex128)
    # for j in J:
    #     for nu in NU:
    #         zak_canonical[j, nu] = zak_transform(canonical, j, nu)
    
    
    # print(zak_gamma)
    # choices[:,:] = -0.25*zak_canonical[:,alpha_t//q:]
    
    choices[:, :] = 1
    
    
    # for j in range(alpha):
    #     for nu in range(beta):
    #         choices[j,nu] = nu**2 - j**2
    zak_gamma[:,:alpha_t//q] = choices[:,:]
    ################ choix ################
    
    
    ## calculer contraintes
    for j in J:
        for nu in np.arange(beta, alpha_t):
            nu_0 = nu % beta
            # zak_gamma[j, nu] = (-1 / np.conjugate(zak_g[j,nu])) * np.sum(choices[j, nu + (l-1) * alpha_t//q] * np.conjugate(zak_g[j, nu + l * alpha_t//q]))
            zak_gamma[j, nu] = c_nk[j, nu - beta] * choices[j, nu_0]

    # print(zak_gamma)
    
    # zak_inv = zak_inverse(zak_gamma)
    # plt.plot(zak_gamma)
    # plt.show()
    
    plot_zak_transform(d_window=None, ax=ax, label=f"Transformée de Zak trouvé (q={q})", zak_precomputed=zak_gamma)

if __name__ == "__main__":
    # fig, axes = plt.subplots(9, 1, figsize=(14, 10)) ## changer 1er argument accordement
    # # plt.subplots_adjust(hspace=1.5)
    # plt.subplots_adjust(
    #     top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
    #     bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
    #     hspace=0.8     # Espace entre les graphiques
    # )
    
    
    
    fig = plt.figure(figsize=(16, 10))
    ax3d = fig.add_subplot(projection='3d')
    
    zak_g = zak_transform_fast(d_window)
    
    plot_chi_zak(d_window=d_window, k=np.arange(alpha), n=np.arange(beta), ax=ax3d)
    
    # plot_window(basis[(5,1+beta)], axes[3], label="orthonormée")
    
    # fig = plt.figure(figsize=(16, 10))
    # ax3d = fig.add_subplot(projection='3d')
    # plot_zak_transform(xi[(5,1+beta)], ax3d, label="xi base")
    
    # fig = plt.figure(figsize=(16, 10))
    # ax3d = fig.add_subplot(projection='3d')
    # plot_zak_transform(test, ax3d, label="xi avant")
    
    # plt.get_current_fig_manager().window.state('zoomed')
    plt.show()
