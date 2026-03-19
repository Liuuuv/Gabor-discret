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
# from reconstitution import reconstruct_signal

from itertools import product

# def reconstruct_signal_from_lattice(coefs, window, dual_window = None, lattice=np.ndarray):
#     N = coefs.shape[0]
#     signal = np.zeros(N, dtype=np.complex128)
#     if dual_window is None:
#         dual_window = window
#     for n in range(N):
#         for k,l in lattice:
#             tm_dual = dual_window[n - k] * np.exp(1j * 2 * np.pi * (n/N) * l)
#             signal[n] += np.sum(tm_dual * coefs[::beta,k], dtype=np.complex128)
#     return signal

def reconstruct_signal_from_lattice(coefs, window, dual_window=None, lattice=None):
    """
    Reconstruit un signal à partir de coefficients Gabor sur un lattice donné.
    """
    N = coefs.shape[0]
    signal = np.zeros(N, dtype=np.complex128)
    
    if dual_window is None:
        dual_window = window
    
    # # Si lattice est une liste, la convertir en tableau numpy
    if isinstance(lattice, list):
        lattice = np.array(lattice)
    
    # # Si lattice est None, créer un lattice dense
    # if lattice is None:
    #     k_vals = np.arange(N)
    #     l_vals = np.arange(N)
    #     K, L_grid = np.meshgrid(k_vals, l_vals, indexing='ij')
    #     lattice = np.stack([K.ravel(), L_grid.ravel()], axis=-1)
    
    for n in range(N):
        k_indices = lattice[:, 0].astype(int)
        l_indices = lattice[:, 1].astype(int)
        
        k_indices = k_indices % N
        l_indices = l_indices % N
        
        
        t_dual = dual_window[(n - k_indices) % N]
        modulations = np.exp(1j * 2 * np.pi * (n / N) * l_indices)
        tm_dual = t_dual * modulations
        
        coef_vals = coefs[l_indices, k_indices]
        
        signal[n] = np.sum(tm_dual * coef_vals)
    
    return signal

def create_lattice(gap: tuple = (alpha, beta), offset: tuple = (0, 0)):
    assert offset[0] < gap[0] and offset[1] < gap[1]
    x = np.arange(offset[0], L, gap[0])
    y = np.arange(offset[1], L, gap[1])
    lattice = list(product(x,y))
    print(f"gap {gap}, offset, {offset}")
    print(lattice)
    print()
    return lattice

def plot_lattice(lattice):
    print(lattice)
    if isinstance(lattice, list):
        lattice = np.array(lattice)
    
    lattice[:, 0] = lattice[:, 0]//alpha
    lattice[:, 1] = lattice[:, 1]//beta
    
    im = np.zeros((L, L))
    print(lattice[:, 0])
    # x, y = np.zeros(alpha_t), np.zeros(L)
    im[lattice[:, 0], lattice[:, 1]] = 1
    im = im[:alpha_t, :beta_t]
    
    im = np.transpose(im)
    # print(x)
    
    plt.figure(figsize=(8, 8))
    # plt.scatter(lattice[:, 0], lattice[:, 1], 
    #         color='black', s=1, zorder=5, marker='s')
    plt.imshow(im, interpolation='none', cmap='Greys', origin='lower')
    # plt.xlim(0, L)
    # plt.ylim(0, L)
    # plt.grid()
    plt.show()
    return
    
    
    # Obtenir les valeurs uniques et triées
    x_unique = np.sort(np.unique(lattice[:, 0]))
    y_unique = np.sort(np.unique(lattice[:, 1]))

    # Créer une matrice pour imshow
    # Ici je crée une matrice avec des valeurs aléatoires pour démonstration
    # Vous pouvez remplacer par vos propres données
    matrix = np.random.rand(len(y_unique), len(x_unique))

    # Créer la figure
    plt.figure(figsize=(12, 8))

    # Afficher l'image avec imshow
    # extent définit les limites: [gauche, droite, bas, haut]
    extent = [x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]]
    plt.imshow(matrix, extent=extent, origin='lower', aspect='auto', cmap='viridis')

    # Tracer les points en noir par-dessus
    plt.scatter(lattice[:, 0], lattice[:, 1], 
            color='black', s=50, zorder=5, marker='s')

    # Ajouter les lignes de la grille
    plt.xticks(x_unique)
    plt.yticks(y_unique)
    plt.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.7)

    plt.xlabel('k')
    plt.ylabel('l')
    plt.title('Grille avec imshow et points du lattice')
    plt.colorbar(label='Valeurs')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # fig, axes = plt.subplots(6, 1, figsize=(14, 10)) ## changer 1er argument accordement
    # fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1,4,1]}) ## changer 1er argument accordement


    y_lim = 2.5
    # plot_signal(signal, axes[0], custom_y_lim=y_lim)
    # result = plot_fstdft(signal, ax_index=1, window=window, plot_ref=True, tolerance=0.02) ## pour indicatrice
    # result = plot_fstdft(signal, ax_index=1, window=window, plot_ref=False, tolerance=0, linear=True)
    result = plot_fstdft(signal, d_window=d_window, plot_ref=True)


    # plot_scipy_fstdft(signal, ax_index=2, window=window) ## je n'arrive pas à le faire fonctionner correctement..
    # plot_dft(signal, ax_index=2, module_only=True)
    # plot_fft(signal, ax_index=2, module_only=True)

    
    
    # plot_window(discretize_window(window, True), ax_index=4, label="Fenêtre")



    # d_dual_window = compute_dual_window(window, alpha=alpha, beta=beta)
    # d_dual_window = compute_alternate_dual_window(d_window)
    
    
    
    # xi = build_xi()
    # d_dual_window = 0.01*xi[(5,1+beta)] + 0.01*xi[(4,1+beta)] + 0.01*xi[(0,1+beta)] + compute_dual_window(window, alpha=alpha, beta=beta)
    
    
    
    canonical_dual_window = compute_dual_window(window, alpha=alpha, beta=beta)
    
    d_dual_window = canonical_dual_window
    
    
    density = []
    errors = []
    
    lattice = create_lattice(gap=(alpha, beta))
    reconstructed_signal = reconstruct_signal_from_lattice(result, d_window, d_dual_window, lattice=lattice)
    errors.append(np.max(np.abs((reconstructed_signal - signal))))
    density.append(1)
    
    lattice = create_lattice(gap=(2*alpha, 2*beta), offset=(0,0))
    lattice += create_lattice(gap=(2*alpha, 2*beta), offset=(alpha,beta))
    
    reconstructed_signal = reconstruct_signal_from_lattice(result, d_window, d_dual_window, lattice=lattice)
    # plot_lattice(create_lattice(gap=(alpha, 2*beta)))
    # plot_lattice(lattice)
    errors.append(np.max(np.abs((reconstructed_signal - signal))))
    density.append(0.5)
    
    for i in range(3, 8):
        lattice = create_lattice(gap=(i*alpha, i*beta), offset=(0,0))
        for k in range(1, i):
            lattice += create_lattice(gap=(i*alpha, i*beta), offset=(k*alpha,k*beta))
        
        reconstructed_signal = reconstruct_signal_from_lattice(result, d_window, d_dual_window, lattice=lattice)
        errors.append(np.max(np.abs((reconstructed_signal - signal))))
        density.append(1/i)
        plot_lattice(lattice)
    
    for i in np.arange(3, 8):
        lattice = create_lattice(gap=(i * alpha, i * beta))
        reconstructed_signal = reconstruct_signal_from_lattice(result, d_window, d_dual_window, lattice=lattice)
        errors.append(np.max(np.abs((reconstructed_signal - signal))))
        density.append(1/2**i)
    
    plt.scatter(density, errors)
    plt.title("Erreurs en fonction du gap")
    plt.grid()
    
    # plot_signal(reconstructed_signal, axes[2], custom_y_lim=y_lim)

    

    

    ## pour la visualisation
    # dual_window_vis = d_dual_window.copy()
    # dual_window_vis[:L//2] = d_dual_window[L//2:]
    # dual_window_vis[L//2:] = d_dual_window[:L//2]
    # plot_window(d_dual_window, ax_index=5, label="Fenêtre duale")


    ## plot grille alphaZ \times betaZ
    # x = np.arange(0, L, alpha)/L
    # y = np.arange(0, L, beta)
    # X, Y = np.meshgrid(x, y)
    # X_flat = X.flatten()
    # Y_flat = Y.flatten()
    # axes[1].scatter(X_flat, Y_flat, marker='+', color='orange', alpha=0.8)



    # plot_fstdft(reconstructed_signal, axes[3], d_window=d_window, plot_ref=False, label="STFT du signal reconstruit")
    # plot_fstdft(reconstructed_signal, ax_index=3, window=window, plot_ref=False, label="STFT du signal reconstruit", tolerance=0.08)




    ## finish the plot and save
    plt.get_current_fig_manager().window.state('zoomed')
    plt.tight_layout()

    # plt.savefig('plot.pdf', bbox_inches='tight')  # inutilisé
    # plt.savefig('signal_temporel.jpg', dpi=300)
    plt.show()
    


















