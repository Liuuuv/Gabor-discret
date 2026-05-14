import numpy as np
import soundfile as sf
import time

 
from config import*
from tools import*

def v_func(w):
    # t: float = linear(w / 75.0)
    # t = np.clip(t, 0.0, 1.0)
    
    # return 0.2 * (1-t) +  0.05 * t
    # return 0.1
    return 10.0 / (w + 1.0)


    
def modified_fstdft(signal, alpha=1, d_window=None):
    """
    Calcule <T_x M_w phi_{v(w)} | f> pour chaque (x, w)
    où phi_u = gaussian(u), v est une fonction w -> u
    
    Parameters
    ----------
    signal : ndarray (L,)
        Signal d'entrée
    alpha : int
        Pas de sous-échantillonnage temporel (défaut: 1 = toutes les positions)
    d_window : non utilisé (conservé pour compatibilité)
    
    Returns
    -------
    result : ndarray (L, n_x)
        Matrice (fréquences w, temps x)
    """
    L = len(signal)
    
    w_indices = np.arange(L)
    v_values = np.array([v_func(w) for w in w_indices])  # (L,)
    
    # Fenêtres pour chaque fréquence w
    windows = np.array([discretize_window(gaussian(v)) for v in v_values])  # (L, L)
    
    # Positions temporelles (sous-échantillonnées si alpha > 1)
    x_indices = np.arange(0, L, alpha)
    n_x = len(x_indices)
    
    k_indices = np.arange(L)
    
    # shift_indices[x, k] = (k - x) % L  pour chaque position x
    shift_indices = (k_indices[None, :] - x_indices[:, None]) % L  # (n_x, L)
    
    # windows_translated[w, x, k] = phi_{v(w)}[k - x]
    windows_translated = windows[:, shift_indices]  # (L, n_x, L)
    
    # Modulation : modulation[w, k] = e^{2π i w k / L}
    modulation = np.exp(2j * np.pi * w_indices[:, None] * k_indices[None, :] / L)  # (L, L)
    modulation = modulation[:, None, :]  # (L, 1, L) pour broadcasting
    
    # Atomes : M_w T_x phi_{v(w)} [k]
    atoms = modulation * windows_translated  # (L, n_x, L)
    
    # Produit scalaire : result[w, x] = sum_k conj(atoms[w, x, k]) * signal[k]
    result = np.tensordot(np.conjugate(atoms), signal, axes=([2], [0]))  # (L, n_x)
    
    return result  # (w, x)

# def modified_fstdft(signal, d_window=None):
#     L = len(signal)
#     k_indices = np.arange(L)
#     x_indices = np.arange(L)
#     shift_indices = (k_indices[None, :] - x_indices[:, None]) % L  # (L, L)
    
#     # Précalculer v(w) pour tous les w
#     v_values = np.array([v_func(w) for w in range(L)])
    
#     # Dédupliquer pour éviter de recalculer les mêmes fenêtres
#     unique_v, inverse = np.unique(np.round(v_values, 8), return_inverse=True)
#     unique_windows = np.array([discretize_window(gaussian(v)) for v in unique_v])  # (n_unique, L)
    
#     result = np.zeros((L, L), dtype=np.complex128)
    
#     # Grouper les w par valeur de v(w)
#     for ui, v in enumerate(unique_v):
#         ws = np.where(inverse == ui)[0]  # tous les w avec ce v
#         phi = unique_windows[ui]                          # (L,)
#         phi_translated = phi[shift_indices]               # (L, L) : (x, k)
        
#         # windowed_signal[x, k] = conj(phi[k-x]) * signal[k]
#         windowed = np.conjugate(phi_translated) * signal[None, :]  # (L, L)
        
#         # FFT sur k pour tous les x en une fois
#         fft_all = np.fft.fft(windowed, axis=1)  # (L, L) : (x, nu)
        
#         # Extraire seulement les fréquences w qui nous intéressent
#         result[ws, :] = fft_all[:, ws].T  # (len(ws), L)
    
#     return result  # (w, x)


def plot_test(signal, ax=None, plot_ref=True, label="", tolerance:float=-1.0, linear=False, show_full=False, d_window=None, stdft:np.ndarray=None):
    print("Calcul du test...")
    
    if stdft is None:
        if d_window is not None:
            result_raw = modified_fstdft(signal=signal, d_window=d_window)
        else:
            result_raw = modified_fstdft(signal=signal)
    else:
        result_raw = stdft[:]
    
    if not show_full:
        result = result_raw[:L//2,:]
    else:
        result = result_raw[:,:]
    
    if linear:
        result = np.abs(result)
    else:
        result = np.abs(result)**2
    result /= np.max(result) if np.max(result) != 0 else 1
    
    if tolerance >= 0:
        result[result < tolerance] = 0
    else:
        result[result < 0.001] = 0
    nonzero_y, nonzero_x = np.nonzero(result)
    
    if len(nonzero_y) == 0:
        return 0  # Tout est nul, seuil à 0
    
    last_nonzero_y = np.max(nonzero_y)
    # result = result[:last_nonzero_y + 1,:]
    
    
    if ax:
        freq = np.linspace(0, L//2, len(signal)//2) if not show_full else np.linspace(0, L, len(signal))
        # temps = np.arange(len(signal)) / sr
        temps = np.linspace(min_time, max_time, int(duration * sr))
        # freq = freq[:last_nonzero_y + 1]
        # ax.pcolormesh(temps, freq, result, shading='gouraud')
        ax.pcolormesh(temps, freq, result)
        # axes[ax_index].set_yscale('log')
        if label != "":
            ax.set_title(label)
        else:
            ax.set_title("Transformée de Fourier en temps court modifiée")
        ax.set_ylabel('Hz')
        ax.set_xlabel('Progression')
        ax.set_xlim(min_time, max_time)
        ax.set_ylim(0, last_nonzero_y + 1)
    
    # plot_time_frequencies_reference(ax=ax)
    
    return result_raw


if __name__ == '__main__':
    
    fig, axes = create_subplots((2,1)) ## changer 1er argument accordement
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.2     # Espace entre les graphiques
    )
    
    
    
    plot_signal(signal, axes[0])
    
    plot_test(signal, axes[1], linear=True, tolerance=0.0)
    
    # plt.show()
    
    # plt.savefig('TERTrace/python/tests.jpg', dpi=300)
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()