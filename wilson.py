import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from config import*
from tools import*
from methode_iterative import approximate_compact_support_iter
# from zak import plot_zak_transform, dual_dir_base_vec
from base_import import*
from tight_frame import*
from zak import*
from decomposition_K_Kperp import*

K = np.arange(2)
L_ = np.arange(L//2)

def get_wilson_func(k:int, n:int, d_window=d_window):
    assert q == 2
    alpha = 1
    beta = L//2
    if n == 0:
        if k == 0:
            wilson = d_window
            wilson /= np.sqrt(scalar_product(wilson, wilson))
        else:
            # wilson = time_shift(d_window, 0) * 0.0
            # wilson = (1/np.sqrt(L)) * time_shift(frequency_shift(d_window, 500) + -1 * frequency_shift(d_window, - 500), 0)
            # wilson /= np.sqrt(scalar_product(wilson, wilson))
            wilson = time_shift(d_window, L//2)
            # wilson = discretize_window(window=lambda t: 1j * np.sin(2 * np.pi * t * 10)) * d_window
            wilson /= np.sqrt(scalar_product(wilson, wilson))
    else:
        wilson = time_shift(frequency_shift(d_window, n) + ((-1) ** (n+k)) * frequency_shift(d_window, - n), k * L//2) 
        wilson /= np.sqrt(scalar_product(wilson, wilson))
    # wilson = (1/np.sqrt(L)) * time_shift(frequency_shift(d_window, n) + ((-1) ** (n)) * frequency_shift(d_window, - n), k * L//2)
    # wilson /= np.sqrt(scalar_product(wilson, wilson))
    # if not wilson.any():
    #     print(wilson)
    return wilson

def create_wilson_base(d_window=d_window):
    wilson = {}
    
    
    # K = np.arange(L//2)
    # L_ = np.arange(2)
    
    for k in K:
        for l in L_:
            wilson[(k,l)] = get_wilson_func(k, l, d_window=d_window)
    
    
    # print(len(wilson))
    return wilson
    

if __name__ == "__main__":
    fig, axes = create_subplots((4,1))
    plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.4)
    
    
    d_window = compute_canonical_tight_window(d_window, alpha=L//2, beta=1)
    # from dual_frame import compute_tight_frame_slow
    # d_window = compute_tight_frame_slow(d_window, alpha=L//2, beta=1)[1]
    
    plot_window(d_window=d_window, ax=axes[0], label="window")
    wilson = create_wilson_base(d_window=d_window)
    
    # d_test_window = d_window
    
    # d_test_window = discretize_window(window=lambda t:  np.cos(2 * np.pi * t * 15) * 2 * t)
    # d_test_window[:L//2] = d_test_window[L//2:]
    # d_test_window[L//2:] = d_test_window[L//2-1::-1]
    d_test_window = discretize_window(window=lambda t: np.cos(2 * np.pi * t * 40) * np.exp(-(t/0.8)**2))
    # d_test_window = discretize_window(window=lambda t: np.cos(2 * np.pi * t * 5))
    
    xi = build_xi(orthonormal=True)
    d_test_window = xi[(0,1)]
    for j in range(1,alpha):
        d_test_window += xi[(j,1)]
    
    # chi = build_chi(orthonormal=True)
    # d_test_window = chi[(0,0)]
    # for j in range(1,alpha):
    #     d_test_window += chi[(j,0)]
    plot_window(d_window=d_test_window, ax=axes[0], color='green')
    
    
    
    # K = np.arange(L//2)
    # L_ = np.arange(2)
    
    result_raw = np.zeros((len(K), len(L_)), dtype=np.complex128)
    for k in K:
        for l in L_:
            result_raw[k, l] = scalar_product(wilson[(k, l)], d_test_window)
    
    ax_to_plot = axes[1]
    result = result_raw.copy()
    result = np.abs(result)
    # result = np.log(result)
    result = np.transpose(result)
    mesh = ax_to_plot.pcolormesh(K, L_, result)
    # m = axes[2].heatmap(result, cmap='dusk')
    ax_to_plot.set_title("Module des coefficients dans K")
    # cbar = fig.colorbar(mesh, ax=ax_to_plot, label='Valeur')
    cbar = fig.colorbar(mesh, ax=ax_to_plot)
    
    reconstructed = np.zeros(L, dtype=np.complex128)
    for k in K:
        for l in L_:
            reconstructed += result_raw[k, l] * wilson[(k, l)]
    
    plot_window(reconstructed - d_test_window, ax=axes[2],label="erreur")
    # plot_window(reconstructed, ax=axes[2],label="reconstruit")
    
    axes[3].plot(L_, abs(result_raw[1, :]), linewidth=0.8)
    axes[3].set_yscale('log')
    axes[3].grid()
    
    # plot_window(get_wilson_func(0,5, d_window=d_window), ax=axes[1],label="wilson")
    # plot_window(get_wilson_func(1,2, d_window=d_window), ax=axes[1],label="wilson")
    
    # for k1 in K:
    #     for n1 in L_:
    #         for k2 in K:
    #             for n2 in L_:
    #                 if abs(scalar_product(wilson[(k1,n1)], wilson[(k2,n2)])) > 10e-5 and (k1,n1) != (k2,n2):
    #                     print((k1,n1),(k2,n2), scalar_product(wilson[(k1,n1)], wilson[(k2,n2)]))
    
    
    # M = np.zeros((L, len(wilson)), dtype=np.complex128)
    # for idx, (k, n) in enumerate(wilson.keys()):
    #     M[:, idx] = wilson[(k, n)]

    # # La base devrait être unitaire : M * M^H = I
    # MMH = M @ M.conj().T
    # error_unitary = np.max(np.abs(MMH - np.eye(L)))
    # print(f"Erreur d'unitarité : {error_unitary:.2e}")

    # # Conditionnement de M
    # cond = np.linalg.cond(M)
    # print(f"Conditionnement de M : {cond:.2e}")
    
    # for k1 in range(2):
    #     for n1 in range(L//2):
    #         print(scalar_product(get_wilson_func(k1,n1, d_window=d_window), get_wilson_func(k1,n1, d_window=d_window)))
    
    # print(scalar_product(get_wilson_func(0,0), get_wilson_func(1,2)))
    # for k1 in range(2):
    #     print(scalar_product(get_wilson_func(k1,0, d_window=d_window), get_wilson_func(k1,0, d_window=d_window))) 
    
    plt.show()