import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

from config import*
from tools import*

# ─────────────────────────────────────────────
#  Fenêtre gaussienne discrète
# ─────────────────────────────────────────────

def gaussian_window(L: int, sigma: float) -> np.ndarray:
    """Fenêtre gaussienne périodisée (longueur L, variance sigma²).
    
    La gaussienne est centrée en 0 et enroulée modulo L pour
    respecter la périodicité exigée par la FSTDFT discrète.
    
    Args:
        L     : longueur du signal (et de la fenêtre)
        sigma : écart-type (en échantillons)

    Returns:
        np.ndarray de longueur L, normalisée en énergie.
    """
    t = np.arange(L)
    # Périodisation sur [-L/2, L/2]
    t_centered = t - L // 2
    g = np.exp(-t_centered**2 / (2 * sigma**2))
    # Normalisation L² (énergie unitaire)
    g /= np.linalg.norm(g)
    # Remettre dans l'ordre "causal" (fft-shift inverse)
    g = np.roll(g, L // 2)
    return g


# ─────────────────────────────────────────────
#  FSTDFT vectorisée (reprend ta logique)
# ─────────────────────────────────────────────

def fstdft_gaussian(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Calcule la FSTDFT du signal avec une fenêtre gaussienne.

    Args:
        signal : signal 1D de longueur L
        sigma  : écart-type de la fenêtre gaussienne (en échantillons)

    Returns:
        np.ndarray de forme (L, L), complexe.
        Axe 0 → fréquence, axe 1 → temps.
    """
    L = len(signal)
    d_window = gaussian_window(L, sigma)

    x_indices = np.arange(L)
    # Matrice des indices décalés : indices[i, k] = (k - i) % L
    indices = (np.arange(L)[None, :] - x_indices[:, None]) % L
    window_matrix = d_window[indices]            # (L, L)

    windowed = signal[None, :] * np.conjugate(window_matrix)  # (L, L)
    result = np.fft.fft(windowed, axis=1).T      # (L, L), axe 0 = freq

    return result


# ─────────────────────────────────────────────
#  Affichage interactif avec slider sigma
# ─────────────────────────────────────────────

def plot_fstdft_gaussian_slider(
    signal: np.ndarray,
    sr: int = 1,
    sigma_init: float = None,
    sigma_min: float = 1.0,
    sigma_max: float = None,
    n_sigmas: int = 40,
    show_full: bool = False,
    linear: bool = False,
    tolerance: float = 0.005,
    min_time: float = 0.0,
    max_time: float = None,
):
    """Affiche la FSTDFT avec fenêtre gaussienne et un slider pour sigma.

    Le slider est discret : il choisit parmi `n_sigmas` valeurs uniformément
    réparties entre sigma_min et sigma_max. Cela permet de précalculer toutes
    les FSTDFT et de rendre le slider instantané.

    Args:
        signal     : signal 1D numpy
        sr         : fréquence d'échantillonnage (pour l'axe des fréquences)
        sigma_init : valeur initiale de sigma (défaut : L/8)
        sigma_min  : sigma minimal (défaut : 1.0)
        sigma_max  : sigma maximal (défaut : L/4)
        n_sigmas   : nombre de valeurs de sigma précalculées
        show_full  : si True, affiche les fréquences négatives aussi
        linear     : si True, affiche |STFT| au lieu de |STFT|²
        tolerance  : seuil sous lequel on met à 0 (relatif au max)
        min_time   : valeur min de l'axe temporel
        max_time   : valeur max de l'axe temporel (défaut : len/sr)
    """
    L = len(signal)

    if sigma_max is None:
        sigma_max = L / 4
    if sigma_init is None:
        sigma_init = L / 8
    if max_time is None:
        max_time = L / sr

    sigma_values = np.linspace(sigma_min, sigma_max, n_sigmas)

    # ── Précalcul de toutes les FSTDFT ──────────────────────────────────────
    print(f"Précalcul de {n_sigmas} FSTDFT… (L={L})")
    all_results = []
    for i, sigma in enumerate(sigma_values):
        # raw = fstdft_gaussian(signal, sigma)
        raw = fstdft(signal, d_window=discretize_window(gaussian(sigma)), only_grid=True)[::beta,:]
        if show_full:
            spec = raw[:, :]
        else:
            spec = raw[: L // 2, :]
        if linear:
            spec = np.abs(spec)
        else:
            spec = np.abs(spec) ** 2
        m = np.max(spec)
        if m > 0:
            spec /= m
        spec[spec < tolerance] = 0.0
        all_results.append(spec.astype(np.float32))
        if (i + 1) % 10 == 0 or i == n_sigmas - 1:
            print(f"  {i+1}/{n_sigmas} terminées")
    print("Précalcul terminé.")

    # ── Axes ────────────────────────────────────────────────────────────────
    n_freq = (L // 2) // beta if not show_full else L // beta
    freq_axis = np.linspace(0, sr // 2, n_freq) if not show_full else np.linspace(0, sr, L // beta)
    time_axis = np.linspace(min_time, max_time, L//alpha)

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.18, top=0.93)

    idx_init = int(np.argmin(np.abs(sigma_values - sigma_init)))
    img = ax.pcolormesh(
        time_axis, freq_axis, all_results[idx_init],
        shading="auto", cmap="inferno"
    )

    ax.set_title(f"FSTDFT – fenêtre gaussienne  (σ = {sigma_values[idx_init]:.1f})")
    ax.set_xlabel("Temps" if sr == 1 else "Temps (s)")
    ax.set_ylabel("Fréquence (Hz)" if sr != 1 else "Fréquence (bins)")
    ax.set_xlim(min_time, max_time)
    ax.set_ylim(0, n_freq - 1)
    cbar = fig.colorbar(img, ax=ax, label="Énergie normalisée")

    # ── Slider ──────────────────────────────────────────────────────────────
    ax_slider = plt.axes([0.15, 0.05, 0.72, 0.04])
    slider = Slider(
        ax=ax_slider,
        label="σ (échantillons)",
        valmin=0,
        valmax=n_sigmas - 1,
        valinit=idx_init,
        valstep=1,
        color="#e07b39",
    )
    # Afficher la vraie valeur de sigma sous le slider
    sigma_text = fig.text(
        0.5, 0.01,
        f"σ = {sigma_values[idx_init]:.2f} échantillons",
        ha="center", fontsize=10, color="gray"
    )

    def update(val):
        idx = int(slider.val)
        sigma_cur = sigma_values[idx]
        img.set_array(all_results[idx].ravel())
        img.set_clim(0, np.max(all_results[idx]) or 1)
        ax.set_title(f"FSTDFT – fenêtre gaussienne  (σ = {sigma_cur:.1f})")
        sigma_text.set_text(f"σ = {sigma_cur:.2f} échantillons")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()
    return fig, slider  # garder une référence pour éviter le GC


if __name__ == "__main__":

    plot_fstdft_gaussian_slider(
        signal,
        sr=sr,
        sigma_min=0.005,
        sigma_max=0.5,
        n_sigmas=50,
        show_full=True,
        linear=True,
        tolerance=0.0
    )
    