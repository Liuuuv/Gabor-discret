import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
from signal_test import signal_test, plot_time_frequencies_reference
from reconstitution import fft, fstdft, plot_fstdft
from dual_frame import compute_dual_window
from zak import compute_alternate_dual_window, zak_transform
from base_orth import approximate_window_from_dual_dir
from dual_frame import plot_window
from tools import*
from config import*

n = 4
poly = lambda t: t**n * (1-t)**n * 2 ** (2*n)
signal_test = poly(np.linspace(0, 1, L))

## renverser
signal_test_ = signal_test.copy()
signal_test[:L//2] = signal_test_[L//2:]
signal_test[L//2:] = signal_test_[:L//2]

def square_partial_sum(coefs, d_dual_window, size_alpha, size_beta, ax=None):
    return square_partial_sum_fft(coefs, d_dual_window, size_alpha, size_beta, ax)
    
    # print("range k", -alpha_t//2, alpha_t//2)
    # print("range l", -beta_t//2, beta_t//2)
    signal = np.zeros(L, dtype=np.complex64)
    
    x = []
    y = []
    
    
    j = np.arange(L)
    for k in range(-alpha_t//2, alpha_t//2):
        x.append(pos_mod(alpha * k, L))
        for l in range(-beta_t//2, beta_t//2):
            if abs(k) > size_alpha or abs(l) > size_beta:
                continue
            y.append(pos_mod(beta * l, L))
            
            
            
            signal[j] += coefs[beta * l, alpha * k] * d_dual_window[pos_mod(j - alpha * k, L)] * np.exp(2j * np.pi * beta * l * j / L)
    
    if ax:
        x = np.array(x)/L
        y = np.array(y)
        
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        ax.scatter(X_flat, Y_flat, marker='+', color='orange', alpha=0.8)
    
    return signal

def square_partial_sum_fft(coefs, d_dual_window, size_alpha, size_beta, ax=None):
    L = len(d_dual_window)
    signal = np.zeros(L, dtype=np.complex64)
    
    k_vals = np.arange(-alpha_t//2, alpha_t//2)
    l_vals = np.arange(-beta_t//2, beta_t//2)
    
    k_valid = k_vals[np.abs(k_vals) <= size_alpha]
    l_valid = l_vals[np.abs(l_vals) <= size_beta]
    
    if ax:
        x = pos_mod(alpha * k_valid, L) / L
        y = beta * l_valid[:, np.newaxis]
        X, Y = np.meshgrid(x, y)
        ax.scatter(X.flatten(), Y.flatten(), marker='+', color='orange', alpha=0.8)
    
    j = np.arange(L)
    
    # Pré-calculer toutes les phases possibles
    # Créer une matrice de phases pour tous les l
    phases = np.exp(2j * np.pi * np.outer(beta * l_valid, j) / L)
    
    for k in k_valid:
        # idx_k = pos_mod(j - alpha * k, L)
        idx_k = (j - alpha * k) % L
        window_k = d_dual_window[idx_k]
        
        # Extraire les coefficients pour ce k
        coefs_k = coefs[beta * l_valid, alpha * k]
        
        # Contribution de tous les l pour ce k
        contribution = np.sum(coefs_k[:, np.newaxis] * phases * window_k, axis=0)
        signal += contribution
    
    return signal



if __name__ == "__main__":
    start_time = time.time()
    # fig, axes = plt.subplots(4, 1, figsize=(14, 10)) ## changer 1er argument accordement
    
    
    # plot_signal(signal=signal_test, ax=axes[0])
    
    
    # canonical_d_dual_window = compute_dual_window(window, alpha=alpha, beta=beta)
    
    
    # # d_dual_window = compute_alternate_dual_window(d_window, canonical_dual=canonical_d_dual_window)
    # # d_dual_window = canonical_d_dual_window
    
    # exp_part = discretize_window(window=lambda t: (1 - (np.cos(2 * np.pi * t * 2) ** 2) * np.exp(- (t/0.15)**2)))
    # d_test_window = -canonical_d_dual_window * exp_part
    
    # # d_test_window = discretize_window(window=lambda t: np.sin(2 * np.pi * t * 1))
    
    # reconstructed = approximate_window_from_dual_dir(d_test_window)
    # d_dual_window = canonical_d_dual_window + reconstructed
    
    
    # # d_dual_window = canonical_d_dual_window
    
    
    # coefs = plot_fstdft(signal_test, axes[1], d_window=d_dual_window, show_full=False)
    
    
    # partial_signal = square_partial_sum(coefs, d_dual_window=d_window ,size_alpha=100000, size_beta=40000, ax=axes[1])
    
    # y_lim = np.max(np.abs(signal_test))
    # plot_signal(signal=partial_signal, ax=axes[2], custom_y_lim=y_lim)
    
    # plot_window(d_dual_window, ax=axes[3])
    
    
    
    
    # # plt.show()
    # # plt.get_current_fig_manager().window.state('zoomed')
    # plt.tight_layout()

    
    # plt.savefig('convergence.jpg', dpi=300)
    
    
    plt.close()
    
    canonical_d_dual_window = compute_dual_window(window, alpha=alpha, beta=beta)
    d_dual_window = canonical_d_dual_window
    coefs = plot_fstdft(signal_test, d_window=d_dual_window)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    sizes_alpha = np.arange(1, alpha_t, 1)
    # print(sizes_alpha)
    sizes_beta = np.arange(1, beta_t, 1)
    diffs = np.zeros((alpha_t-1, beta_t-1))
    for size_alpha in sizes_alpha:
        for size_beta in sizes_beta:
            partial_signal = square_partial_sum(coefs, d_dual_window=d_window ,size_alpha=size_alpha, size_beta=size_beta)
            norm: float = np.max(np.abs(signal_test - partial_signal))
            # norm = min(2.5, norm)
            diffs[size_alpha-1, size_beta-1] = norm
    
    
    diffs = diffs.transpose()
    log_diffs = np.log(diffs)
    # diffs = diffs ** 2
    # im = axes[0].pcolormesh(sizes_alpha, sizes_beta, diffs, shading='gouraud')
    im = axes[0].pcolormesh(sizes_alpha, sizes_beta, log_diffs)


    fig.colorbar(im, orientation='vertical', label="Norme inf, échelle log")
    axes[0].set_xlabel("Taille alpha")
    axes[0].set_ylabel("Taille beta")
    axes[0].set_title("Erreur (norme infinie) des sommes partielles rectangulaires")
    
    
    plot_window(window_=signal_test, ax=axes[1])
    
    
    
    ## plot 1D
    fixed_alpha: int = 10
    axes[2].plot(sizes_beta, diffs[:,fixed_alpha])
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title(f"alpha: {fixed_alpha}, duale canonique, n={n}")
    
    end_time = time.time()
    print(f"TEMPS D'EXECUTION: {end_time - start_time}")
    plt.show()




