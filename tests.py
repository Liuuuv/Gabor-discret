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

def xi_as_sum_of_gabor(g, k, n):
    """
    Exprime xi_{k,n} comme somme de M_nu T_k g.
    """
    L = len(g)
    alpha_tilde = L // alpha
    n0 = n % beta
    
    Zg = zak_transform_fast(g)
    
    # Zxi[k, nu] : deux pics
    Zxi_k = np.zeros(alpha_tilde, dtype=complex)
    c = Zg[k, n] / Zg[k, n0]
    Zxi_k[n0] = -c
    Zxi_k[n] = 1.0
    
    # FFT
    hat_Zxi = np.fft.fft(Zxi_k)
    hat_Zg = np.fft.fft(Zg[k, :])
    
    # Déconvolution
    hat_a = hat_Zxi / hat_Zg
    a = np.fft.ifft(hat_a)
    
    # Reconstruction
    xi = np.zeros(L, dtype=complex)
    t = np.arange(L)
    for nu in range(alpha_tilde):
        atom = np.roll(g, k)
        atom *= np.exp(1j * 2 * np.pi * nu * t / L)
        xi += a[nu] * atom
    
    return xi

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.4)
    
    zak_g = zak_transform_fast(d_window=d_window)
    xi = build_xi(zak_g)
    
    k=2
    n=15
    
    plot_window(xi[(k,n)], ax=axes[0])
     
    plot_window(xi_as_sum_of_gabor(g=d_window, k=k, n=n), ax=axes[1])

    plt.show()
