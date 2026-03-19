import numpy as np
from signal_test import*
from config import*
from zak_tools import*


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
        ax.set_title(f"Fréquence d'échantillonnage: {sr}")
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


def fstdft(signal, d_window=None):
    if d_window is None:
        d_window = discretize_window(window)
    L = len(signal)
    result = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        translated_window = np.ones(L, dtype=np.complex128)
        for k in range(L):
            t = k
            translated_window[k] = d_window[t - i]
        result[i] = fft(signal * np.conjugate(translated_window))
    return result.transpose()


def plot_fstdft(signal, ax=None, plot_ref=True, label="", tolerance=-1, linear=False, show_full=False, d_window=None):
    print("Calcul de la FSTDFT...")
    
    if d_window is not None:
        result_raw = fstdft(signal=signal, d_window=d_window)
    else:
        result_raw = fstdft(signal=signal)
    
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
            ax.set_title("Transformée de Fourier en temps court")
        ax.set_ylabel('Hz')
        ax.set_xlabel('Progression')
        ax.set_xlim(min_time, max_time)
        ax.set_ylim(0, last_nonzero_y + 1)
        
        if plot_ref:
            plot_time_frequencies_reference(ax=ax)
    
    return result_raw

def scalar_product(signal1, signal2):
    return np.sum(np.conj(signal1) * signal2)

def build_xi(zak_g=None):
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

    return xi

def build_chi(zak_g=None, orthonormal=False):
    assert q==2
    if zak_g is None:
        zak_g = zak_transform_fast(d_window)
    chi = {}

    for j_0 in range(alpha):
        for n_0 in range(0, beta):


            vec = np.zeros(L, dtype=np.complex128)

            for l in range(alpha_t):
                j = j_0 + l * alpha

                c = zak_g[j_0, n_0 + beta] / zak_g[j_0, n_0]

                vec[j] = (1/alpha_t) * np.exp(2j * np.pi * l * n_0 / alpha_t) * (1+(-1)**l * c)

            chi[(j_0, n_0)] = vec

    if orthonormal:
        for j_0, n_0 in chi.keys():
            chi[(j_0, n_0)] /= scalar_product(chi[(j_0, n_0)], chi[(j_0, n_0)]) ** 0.5
    
    return chi


def plot_fft(signal, ax, module_only = True, label="", ylog=False):
    print("Calcul de la FFT...")
    freq = np.linspace(0, sr//2, len(signal)//2)
    ft_signal = fft(signal)[:sr//2]
    if module_only:
        ax.plot(freq, np.abs(ft_signal), color='blue', alpha=0.7, linewidth=0.5)
    else:
        ax.plot(freq, np.imag(ft_signal), color='red', alpha=0.7, linewidth=0.5)
        ax.plot(freq, np.real(ft_signal), color='blue', alpha=0.7, linewidth=0.5)
    # ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if label == "":
        ax.set_title("FFT")
    else:
        ax.set_title(label)
    ax.set_xlabel("Fréquence")
    ax.grid(True, alpha=0.3)
    if module_only:
        ax.set_ylabel("Module")
    else:
        ax.set_ylabel("Partie réelle/imaginaire (bleu, rouge resp.)")



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
    result_raw = np.zeros((alpha, alpha_t - beta), dtype=np.complex64)
    
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
    
    reconstructed = np.zeros(L, dtype=np.complex64)
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