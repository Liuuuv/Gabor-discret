import numpy as np
import time
from matplotlib.widgets import Slider

from signal_test import*
from config import*
from ease import*
from zak_tools import*


def create_subplots(shape: tuple, figsize:tuple=(14, 10)) -> tuple:
    """(rows, columns)

    Args:
        shape (tuple): _description_
        figsize (tuple, optional): _description_. Defaults to (14, 10).

    Returns:
        tuple: _description_
    """
    fig, axes = plt.subplots(*shape, figsize=figsize)
    if type(axes) != np.ndarray:
        axes = np.array([axes])
    return fig, axes

def pos_mod(a, mod): ## inutile car les array le font très bien tout seul
    """positive modulo, a: int or a: np.ndarray

    Args:
        a (int or np.array): 
        mod (int): 

    Returns:
        int or np.array: a modulo mod, with a >= 0
    """
    assert mod >= 0
    remain = a % mod
    if type(remain) is np.ndarray:
        for i in range(len(remain)):
            if remain[i] < 0:
                remain[i] + mod
        return remain
    return remain if remain >= 0 else remain + mod

def plot_signal(signal, ax, custom_y_lim=0.0, label="", color='blue', logscale=False):
    # temps = np.arange(len(signal)) / sr
    temps = np.linspace(min_time, max_time, len(signal))
    ax.plot(temps, signal, color=color, alpha=0.7, linewidth=0.8)
    if label == "":
        ax.set_title(f"Signal, Fréquence d'échantillonnage: {sr}")
    else:
        ax.set_title(label)
    ax.set_xlabel("Progression")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.margins(0, x=None, y=None, tight=True)
    if logscale:
        ax.set_yscale('log')
    if custom_y_lim:
        ax.set_ylim(-custom_y_lim, custom_y_lim)


def fft(signal):
    fourier = np.fft.fft(signal)
    return fourier


# def fstdft(signal, d_window=None):
#     if d_window is None:
#         d_window = discretize_window(window)
#     L = len(signal)
#     result = np.zeros((L, L), dtype=np.complex128)
#     for i in range(L):
#         translated_window = np.ones(L, dtype=np.complex128)
#         for k in range(L):
#             t = k
#             translated_window[k] = d_window[t - i]
#         result[i] = fft(signal * np.conjugate(translated_window))
#     return result.transpose()

def fstdft(signal, d_window=None, only_grid:bool=False):
    print("Calcul de la FSTDFT...")
    start_time: float = time.time()
    if d_window is None:
        d_window = discretize_window(window)
    
    
    L = len(signal)
    
    if not only_grid:
        # Toutes les positions temporelles (dense)
        x_indices = np.arange(L)
    else:
        # Sous-échantillonnage temporel
        x_indices = np.arange(0, L, alpha)
    
    # Construire la matrice des fenêtres translatées en une seule opération
    # indices[i, k] = (k - i) % L
    indices = (np.arange(L)[None, :] - x_indices[:, None]) % L
    window_matrix = d_window[indices]  # (L, L)
    
    if len(x_indices) == 0:
        print("ERREUR: x_indices est vide!")
        print(f"Vérification alpha={alpha}, L={L}")
        raise ValueError("x_indices est vide")
    
    # Multiplier signal par chaque fenêtre conjuguée, puis FFT sur chaque ligne
    windowed = signal[None, :] * np.conjugate(window_matrix)  # (L, L)
    result = np.fft.fft(windowed, axis=1)  # FFT sur chaque ligne
    result = result.T
    print(f"End calcul of FSTDFT, time {time.time() - start_time}s")
    return result


def plot_fstdft(signal, ax=None, plot_ref=True, label="", tolerance=-1, linear=False, show_full=False, d_window=None, stdft:np.ndarray=None, plot_only_grid:bool=False):
    print("plot_fstdft called...")
    start_time: float = time.time()
    
    if stdft is None:
        if d_window is not None:
            result_raw = fstdft(signal=signal, d_window=d_window)
        else:
            result_raw = fstdft(signal=signal)
    else:
        result_raw = np.copy(stdft)
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
        result[result < 0.005] = 0
    nonzero_y, nonzero_x = np.nonzero(result)
    
    if len(nonzero_y) == 0:
        return 0  # Tout est nul, seuil à 0
    
    last_nonzero_y = np.max(nonzero_y)
    # result = result[:last_nonzero_y + 1,:]
    
    
    if ax:
        print("Plotting FSTDFT...")
        if plot_only_grid:
            freq = np.linspace(0, L//2, (len(signal)//2) // beta) if not show_full else np.linspace(0, L, len(signal) // beta)
            temps = np.linspace(min_time, max_time, int(duration * sr) // alpha)
        else:
            freq = np.linspace(0, L//2, len(signal)//2) if not show_full else np.linspace(0, L, len(signal))
            # temps = np.arange(len(signal)) / sr
            temps = np.linspace(min_time, max_time, int(duration * sr))
        # freq = freq[:last_nonzero_y + 1]
        # ax.pcolormesh(temps, freq, result, shading='gouraud')
        if plot_only_grid:
            ax.pcolormesh(temps, freq, result[::beta,::alpha])
        else:
            ax.pcolormesh(temps, freq, result)
        # axes[ax_index].set_yscale('log')
        if label != "":
            ax.set_title(label)
        else:
            ax.set_title("Transformée de Fourier en temps court")
        ax.set_ylabel('Hz')
        ax.set_xlabel('Progression')
        ax.set_xlim(min_time, max_time)
        ax.set_ylim(0, last_nonzero_y + 1)
        
        if plot_ref:
            plot_time_frequencies_reference(ax=ax)
            
    print(f"End call of plot_fstdft, time {time.time() - start_time}s")
    return result_raw

def scalar_product(signal1, signal2):
    return np.sum(np.conj(signal1) * signal2, dtype=np.complex128)

def build_xi(zak_g=None, orthonormal=False):
    if zak_g is None:
        zak_g = zak_transform_fast(d_window)
    xi = {}

    for k in range(alpha):
        for n in range(beta, alpha_t):

            nu0 = n % beta

            vec = np.zeros(L, dtype=np.complex128)

            for l in range(alpha_t):
                j = k + l * alpha

                term1 = np.exp(2j * np.pi * l * n / alpha_t)

                c = np.conj(zak_g[k, n]) / np.conj(zak_g[k, nu0])
                term2 = c * np.exp(2j * np.pi * l * nu0 / alpha_t)

                vec[j] = (term1 - term2) / alpha_t

            xi[(k, n)] = vec
    
    if orthonormal:
        for j_0, n_0 in xi.keys():
            xi[(j_0, n_0)] /= scalar_product(xi[(j_0, n_0)], xi[(j_0, n_0)]) ** 0.5

    return xi

def build_chi(zak_g=None, orthonormal=False):
    assert p==1
    if zak_g is None:
        zak_g = zak_transform_fast(d_window)
    chi = {}
    if q == 2:
        for j_0 in range(alpha):
            for n_0 in range(0, beta):


                vec = np.zeros(L, dtype=np.complex128)

                for l in range(alpha_t):
                    j = j_0 + l * alpha

                    c = zak_g[j_0, n_0 + beta] / zak_g[j_0, n_0]

                    vec[j] = (1/alpha_t) * np.exp(2j * np.pi * l * n_0 / alpha_t) * (1+(-1)**l * c)

                chi[(j_0, n_0)] = vec
    else:
        for j_0 in range(alpha):
            for n_0 in range(0, beta):
                vec = np.zeros(L, dtype=np.complex128)

                for l in range(alpha_t):
                    j = j_0 + l * alpha

                    the_sum = 0.0
                    for mu in range(q):
                        c = zak_g[j_0, n_0 + mu * beta] / zak_g[j_0, n_0]
                        the_sum += c * np.exp(2j * np.pi * l * mu / q)
                    

                    vec[j] = (1/alpha_t) * np.exp(2j * np.pi * l * n_0 / alpha_t) * the_sum

                chi[(j_0, n_0)] = vec

    if orthonormal:
        for j_0, n_0 in chi.keys():
            chi[(j_0, n_0)] /= scalar_product(chi[(j_0, n_0)], chi[(j_0, n_0)]) ** 0.5
    
    return chi


def plot_fft(signal, ax, module_only = True, label="", ylog=False, tight_plot=True):
    print("Calcul de la FFT...")
    freq = np.linspace(0, len(signal)//2, len(signal)//2)
    ft_signal = fft(signal)[:len(signal)//2]
    if module_only:
        ax.plot(freq, np.abs(ft_signal), color='blue', alpha=0.7, linewidth=0.7)
    else:
        ax.plot(freq, np.imag(ft_signal), color='red', alpha=0.7, linewidth=0.7)
        ax.plot(freq, np.real(ft_signal), color='blue', alpha=0.7, linewidth=0.7)
    # ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if label == "":
        ax.set_title("FFT")
    else:
        ax.set_title(label)
    ax.set_xlabel("Fréquence (Hz)")
    ax.grid(True, alpha=0.3)
    if module_only:
        ax.set_ylabel("Module")
    else:
        ax.set_ylabel("Partie réelle/imaginaire (bleu, rouge resp.)")
    if tight_plot:
        ax.margins(0, x=None, y=None, tight=True)


def approximate_window_from_dual_dir(d_test_window, ax_to_plot=None, ax_phase=None, basis=None, fig=None):
    if basis is None:
        zak_g = zak_transform_fast(d_window)
        xi = build_xi(zak_g=zak_g)
        
        if q == 2:
            xi_ = xi.copy()
            for k,l in xi.keys():
                xi_[(k,l)] /= scalar_product(xi[(k,l)], xi[(k,l)]) ** 0.5
            basis = xi_
        else:
            basis = build_orthonormal_xi(zak_g, xi)
    
    
    K = np.arange(alpha)
    L_ = np.arange(alpha_t - beta)
    result_raw = np.zeros((alpha, alpha_t - beta), dtype=np.complex128)
    
    for k in K:
        for l in L_:
            result_raw[k, l] = scalar_product(basis[(k,l + beta)], d_test_window)
            # result_raw[k, l] = scalar_product(xi[(k,l + beta)], d_test_window)
    
    if ax_to_plot:
        result = result_raw.copy()
        result = np.abs(result)
        # result = np.log(result)
        result = np.transpose(result)
        mesh = ax_to_plot.pcolormesh(K, L_, result)
        # m = axes[2].heatmap(result, cmap='dusk')
        ax_to_plot.set_title("Module des coefficients dans K^\perp")
        # cbar = fig.colorbar(mesh, ax=ax_to_plot, label='Valeur')
        cbar = fig.colorbar(mesh, ax=ax_to_plot)
    
    if ax_phase is not None:
        result = result_raw.copy()
        mask = (np.abs(result)/np.max(np.abs(result)) < 10e-2) ## contrer les présupposées erreurs numériques
        result[mask] = np.nan
        phase = np.angle(result)
        phase = np.transpose(phase)
        mesh_phase = ax_phase.pcolormesh(K, L_, phase, shading='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        # plt.colorbar(mesh_phase, ax=ax_phase, label='Phase [rad]')
        plt.colorbar(mesh_phase, ax=ax_phase)
        ax_phase.set_title("Phase des coefficients dans K^\perp")
    
    reconstructed = np.zeros(L, dtype=np.complex128)
    for k in K:
        for l in L_:
            reconstructed += result_raw[k, l] * basis[(k,l + beta)]
            # reconstructed += result_raw[k, l] * xi[(k,l + beta)]
    
    return reconstructed


def build_orthonormal_xi(Zg, xi):
    print("Calcul de la base orthonormée de K^perp...")
    
    ## deep copy of xi
    xi_ = {}
    for k,l in xi.keys():
        xi_[(k,l)] = xi[(k,l)].copy()
    
    
    if q == 2:
        for k,l in xi_.keys():
            xi_[(k,l)] /= scalar_product(xi_[(k,l)], xi_[(k,l)]) ** 0.5
        basis = xi_
        return basis
    
    basis = {}

    for k in range(alpha):
        for nu0 in range(beta):

            indices = [nu0 + m*beta for m in range(1, q)]

            ortho = []

            for n in indices:

                v = xi[(k,n)].copy()

                for u in ortho:
                    v -= np.vdot(u, v) * u

                norm = np.linalg.norm(v)

                if norm < 1e-12:
                    raise ValueError(f"Linear dependence detected for k={k}, nu0={nu0}, n={n}")

                v /= norm
                ortho.append(v)

                basis[(k,n)] = v

    return basis

# def create_subplots(size: tuple, figsize=(14, 10)):
#     fig, axes = plt.create_subplots((1,1), figsize) ## changer 1er argument accordement
#     if type(axes) != np.ndarray:
#         axes = np.array([axes])
#     return fig, axes

def add_3d_subplot(fig, plot_shape, index):
    ax3d = fig.add_subplot(*plot_shape, index, projection='3d')
    # ax3d = fig.add_subplot(projection='3d')
    return ax3d

def time_shift(d_window, x:int):
    new_window = np.zeros_like(d_window, dtype=np.complex128)
    for j in range(len(d_window)):
        new_window[j] = d_window[int((j - x) % L)]
    return new_window

def frequency_shift(d_window, xi:int):
    new_window = np.zeros_like(d_window, dtype=np.complex128)
    for j in range(len(d_window)):
        real_j = L_sampling[j]
        new_window[j] = d_window[j] * np.exp((1j * 2 * np.pi / L) * xi * real_j)
    return new_window