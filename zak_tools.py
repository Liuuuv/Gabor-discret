import numpy as np
from signal_test import*
from config import*






def piecewise_scalar_zak_transform(d_window, alpha=alpha, beta=beta): ## p=1
    """Piecewise Zak transform (only for p=1)"""
    assert p == 1
    zak_g = zak_transform_fast(d_window=d_window, alpha=alpha, beta=beta)
    piecewise_zak = np.zeros_like(zak_g)
    
    ## VERSION NAIVE
    # piecewise_zak = np.zeros_like(zak_g)
    # for j in range(alpha):
    #     for nu in range(alpha_t):
    #         piecewise_zak[j, nu] = sum([np.abs(zak_g[j, nu - l * beta])**2 for l in range(q)])
    # return piecewise_zak
    
    ## VERSION VECTORISEE
    for l in range(q):
        shifted = np.roll(np.abs(zak_g)**2, shift=-l * beta, axis=1)
        piecewise_zak += shifted
    
    return piecewise_zak


def zak_transform_fast(d_window, alpha=alpha, beta=beta):
    """Version vectorisée de transformée de Zak"""
    L = len(d_window)
    alpha_t = L // alpha
    
    # Reshape en matrice (alpha, alpha_t)
    signal_matrix = d_window.reshape(alpha, alpha_t, order='F')
    
    # FFT sur les colonnes
    zak = np.fft.fft(signal_matrix, axis=1)
    
    return zak


# def zak_transform(d_window, j, nu):
#     if nu is np.ndarray:
#         l = np.arange(alpha_t)                          # (alpha_t,)
#         indices = (j - alpha * l) % L                   # (alpha_t,)
#         window_vals = d_window[indices]                  # (alpha_t,)

#         nu = np.atleast_1d(np.asarray(nu, dtype=float)) # (n,)
#         exponents = np.exp(
#             1j * (2 * np.pi / alpha_t) * nu[:, None] * l[None, :]
#         )                                                # (n, alpha_t)

#         result = np.sum(window_vals[None, :] * exponents, axis=1)  # (n,)
        
#         # retourner scalaire si nu était scalaire
#         return result[0] if result.shape == (1,) else result
#     else:
#         l = np.arange(0, alpha_t)
#         return np.sum(d_window[j - alpha * l] * np.exp(1j * (2*np.pi/alpha_t) * nu * l), dtype=np.complex128)
    
def zak_transform(d_window, j, nu):
    l = np.arange(0, alpha_t)                        # (alpha_t,)
    window_vals = d_window[(j - alpha * l) % L]      # (alpha_t,)
    
    nu = np.asarray(nu)
    if nu.ndim == 0:  # scalaire
        return np.sum(window_vals * np.exp(1j * (2*np.pi/alpha_t) * nu * l), dtype=np.complex128)
    else:             # array (n,)
        exponents = np.exp(1j * (2*np.pi/alpha_t) * nu[:, None] * l[None, :])  # (n, alpha_t)
        return np.sum(window_vals[None, :] * exponents, axis=1, dtype=np.complex128)  # (n,)




def plot_cn(d_window, ax=None, fig=None, ax_phase=None):
    zak_g = zak_transform_fast(d_window)
    
    c_nk_raw = np.zeros((alpha, alpha_t - beta), dtype=np.complex128)
    for k in range(alpha):
        for n in range(alpha_t - beta):
            n_ = n + beta
            nu0 = n_ % beta
            c_nk_raw [k, n] = zak_g[k, n_] / zak_g[k, nu0]
            # if n==5:
            #     print("c_nk [k, 5]=",np.abs(c_nk [k, n]), np.angle(c_nk [k, n]))
    
    if ax:
        if fig is None:
            print("PLEASE PROVIDE A FIG")
            return c_nk_raw
        c_nk = c_nk_raw.copy()
        c_nk = np.transpose(c_nk)
        # c_nk = np.log(np.abs(c_nk))
        c_nk = np.abs(c_nk)
        # c_nk[(np.abs(c_nk - 1) > 10e-11)] = 0
        # c_nk[(np.abs(c_nk - 1) <= 10e-11)] = 1
        mesh = ax.pcolormesh(np.arange(alpha), np.arange(alpha_t - beta), c_nk, cmap='inferno')
        ax.set_ylabel("n")
        ax.set_xlabel("k")
        ax.set_title("ck[n] (module)")
        # m = axes[2].heatmap(result, cmap='dusk')
        cbar = fig.colorbar(mesh, ax=ax)
    if ax_phase is not None:
        result = c_nk_raw.copy()
        mask = (np.abs(result) < 10e-10) ## contrer les présupposées erreurs numériques
        result[mask] = np.nan
        phase = np.angle(result)
        phase = np.transpose(phase)
        mesh_phase = ax_phase.pcolormesh(np.arange(alpha), np.arange(alpha_t - beta), phase, shading='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        # plt.colorbar(mesh_phase, ax=ax_phase, label='Phase [rad]')
        plt.colorbar(mesh_phase, ax=ax_phase)
        ax_phase.set_title("Phase des ck[n]")
    return c_nk_raw


def zak_inverse(zak): ## zak sur alpha, alpha_t
    vec = np.zeros(L, np.complex128)
    for n in np.arange(L):
        for nu in np.arange(alpha_t):
            phase = np.exp(2j * np.pi * nu * (n//alpha) / alpha_t)
            vec[n] += (1/alpha_t) * zak[n%alpha,nu] * phase
    return vec



