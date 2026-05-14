import numpy as np
import soundfile as sf
import time




from config import*
from tools import*


# def ambiguity(signal, signal2=None):
#     if signal2 is None:
#         signal2 = signal
    
#     L = len(signal)
#     result = np.zeros((L, L), dtype=np.complex128)
#     for i in range(L):
#         modified_window = np.zeros(L, dtype=np.complex128)
#         for t in range(L):
#             modified_window[t] = d_window[(t + i//2) % L] * np.conjugate(d_window[(t - i//2) % L])
#         result[i] = fft(signal * modified_window)
#     return result.transpose()


# def wigner(signal, signal2=None):
#     if signal2 is None:
#         signal2 = signal
    
#     L = len(signal)
#     result = np.zeros((L, L), dtype=np.complex128)
    
#     t = np.arange(L)
#     for m in range(L):
#         product = signal[(m + t) % L] * np.conjugate(signal2[(m - t) % L])
#         result[:, m] = np.fft.fft(product)
    
#     return result
    
# def wigner(signal, signal2=None):
#     if signal2 is None:
#         signal2 = signal
    
#     L = len(signal)
#     t   = np.arange(L)
#     tau = np.arange(L)
    
#     # shape (L, L) : axe 0 = t, axe 1 = tau
#     plus  = signal[ (t[:, None] + tau[None, :]) % L]
#     minus = signal2[(t[:, None] - tau[None, :]) % L]
    
#     product = plus * np.conjugate(minus)  # (t, tau)
#     result  = np.fft.fft(product, axis=1) # (t, nu)
    
#     return result.T  # (nu, t)

def wigner(signal, signal2=None):
    if signal2 is None:
        signal2 = signal
    
    L = len(signal)
    t   = np.arange(L)
    tau = np.arange(L)
    
    # Zero-padding du signal pour éviter le wrap
    sig_pad  = np.concatenate([signal,  np.zeros(L)])   # (2L,)
    sig2_pad = np.concatenate([signal2, np.zeros(L)])
    
    # t + tau et t - tau sans modulo
    plus  = sig_pad[ t[:, None] + tau[None, :]]          # (L, L)
    minus = sig2_pad[np.clip(t[:, None] - tau[None, :], 0, 2*L-1)]  # (L, L)
    
    product = plus * np.conjugate(minus)
    result  = np.fft.fft(product, axis=1)
    
    return result.T  # (nu, t)


if __name__ == '__main__':
    
    fig, axes = create_subplots((2,1)) ## changer 1er argument accordement
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.2     # Espace entre les graphiques
    )
    
    plot_fstdft
    
    wig = wigner(signal=signal)
    plot_signal(signal, axes[0])
    
    wig = np.abs(wig)
    # wig = np.log1p(np.abs(wig))
    freq = np.linspace(0, L // 2, len(signal))
    # temps = np.arange(len(signal)) / sr
    temps = np.linspace(min_time, max_time, int(duration * sr))
    # freq = freq[:last_nonzero_y + 1]
    # ax.pcolormesh(temps, freq, result, shading='gouraud')
    axes[1].pcolormesh(temps, freq, wig)
    
    # plt.show()
    
    plt.savefig('TERTrace/python/Wigner.jpg', dpi=300)
    # plt.get_current_fig_manager().window.state('zoomed')
    # plt.show()