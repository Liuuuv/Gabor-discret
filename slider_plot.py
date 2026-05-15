"""
slider_plot.py
==============
Utilitaire générique pour créer un tracé matplotlib avec un ou plusieurs
sliders discrets et précalcul optionnel.

Usage minimal
-------------
>>> from slider_plot import slider_plot
>>>
>>> slider_plot(
...     render_fn=lambda sigma: compute_spectrogram(signal, sigma),
...     params={"σ": np.linspace(1, 50, 40)},
... )

Usage complet (plusieurs sliders, axes personnalisés, etc.)
-----------------------------------------------------------
>>> slider_plot(
...     render_fn=lambda sigma, order: compute_something(sigma, order),
...     params={
...         "σ":     np.linspace(1.0, 50.0, 40),
...         "order": np.arange(1, 9),
...     },
...     plot_fn=lambda ax, data, params: ax.pcolormesh(
...         time_axis, freq_axis, data, shading="auto", cmap="inferno"
...     ),
...     title_fn=lambda params: f"Spectrogramme  σ={params['σ']:.1f}  ordre={params['order']}",
...     xlabel="Temps (s)",
...     ylabel="Fréquence (Hz)",
...     figsize=(12, 6),
...     precompute=True,
... )
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

RenderFn  = Callable[..., Any]          # (**param_values) → data
PlotFn    = Callable[[Any, Any, dict], Any]  # (ax, data, param_values) → artist
TitleFn   = Callable[[dict], str]       # (param_values) → str
ParamDict = Dict[str, np.ndarray]       # {name: array_of_values}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ──────────────────────────────────────────────────────────────────────────────

def _default_plot_fn(ax: Any, data: np.ndarray, _params: dict) -> Any:
    """Tracé par défaut : pcolormesh si 2D, sinon plot simple."""
    if np.ndim(data) == 2:
        return ax.pcolormesh(data, shading="auto", cmap="inferno")
    else:
        (line,) = ax.plot(np.asarray(data))
        return line


def _default_title_fn(params: dict) -> str:
    parts = [f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
             for k, v in params.items()]
    return "  |  ".join(parts)


def _update_artist(artist: Any, data: np.ndarray) -> None:
    """Met à jour l'artiste matplotlib de façon générique."""
    import matplotlib.collections as mc
    import matplotlib.lines as ml

    if isinstance(artist, mc.QuadMesh):          # pcolormesh
        artist.set_array(np.asarray(data).ravel())
        vmax = np.max(data) or 1
        artist.set_clim(0, vmax)

    elif isinstance(artist, ml.Line2D):           # plot 1D
        artist.set_ydata(np.asarray(data))

    else:
        # Fallback : on essaie set_data puis set_array
        try:
            artist.set_data(data)
        except (AttributeError, TypeError):
            try:
                artist.set_array(np.asarray(data).ravel())
            except AttributeError:
                pass  # artiste inconnu — l'update sera ignoré


# ──────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ──────────────────────────────────────────────────────────────────────────────

def slider_plot(
    render_fn: RenderFn,
    params: ParamDict,
    *,
    plot_fn: Optional[PlotFn] = None,
    title_fn: Optional[TitleFn] = None,
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (12, 6),
    precompute: bool = True,
    slider_color: str = "#e07b39",
    colorbar: bool = False,
    colorbar_label: str = "",
) -> tuple:
    """Crée un tracé matplotlib avec sliders discrets pour chaque paramètre.

    Parameters
    ----------
    render_fn : callable
        Fonction ``(**param_values) -> data`` appelée pour chaque combinaison
        de paramètres. Les noms des arguments doivent correspondre aux clés
        de ``params``.
    params : dict[str, np.ndarray]
        Dictionnaire ``{nom_paramètre: tableau_de_valeurs}``. Un slider est
        créé pour chaque entrée.
    plot_fn : callable, optional
        ``(ax, data, param_values) -> artist`` pour afficher ``data`` sur
        ``ax``. Par défaut : pcolormesh (2D) ou plot (1D).
    title_fn : callable, optional
        ``(param_values) -> str`` pour le titre. Par défaut : liste des
        valeurs courantes.
    xlabel, ylabel : str
        Labels des axes.
    figsize : tuple
        Taille de la figure.
    precompute : bool
        Si ``True``, précalcule toutes les combinaisons avant d'afficher.
        Recommandé quand le nombre total de combinaisons est ≤ quelques
        centaines et que ``render_fn`` est coûteux.
    slider_color : str
        Couleur des sliders.
    colorbar : bool
        Affiche une colorbar (utile pour pcolormesh).
    colorbar_label : str
        Label de la colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
    sliders : list[matplotlib.widgets.Slider]
        Références aux sliders (à conserver pour éviter le garbage collector).
    """
    if plot_fn is None:
        plot_fn = _default_plot_fn
    if title_fn is None:
        title_fn = _default_title_fn

    param_names  = list(params.keys())
    param_arrays = [params[n] for n in param_names]
    n_params     = len(param_names)

    # ── Précalcul ─────────────────────────────────────────────────────────
    cache: dict = {}
    if precompute:
        all_combos = list(itertools.product(*[range(len(a)) for a in param_arrays]))
        total = len(all_combos)
        print(f"Précalcul de {total} combinaison(s)…")
        for combo in all_combos:
            kw = {name: param_arrays[i][combo[i]]
                  for i, name in enumerate(param_names)}
            cache[combo] = render_fn(**kw)
            done = len(cache)
            if done % max(1, total // 10) == 0 or done == total:
                print(f"  {done}/{total}")
        print("Précalcul terminé.")

    def get_data(indices: tuple) -> Any:
        if precompute:
            return cache[indices]
        kw = {name: param_arrays[i][indices[i]]
              for i, name in enumerate(param_names)}
        return render_fn(**kw)

    def get_param_values(indices: tuple) -> dict:
        return {name: param_arrays[i][indices[i]]
                for i, name in enumerate(param_names)}

    # ── Index initiaux (milieu de chaque plage) ───────────────────────────
    init_indices = tuple(len(a) // 2 for a in param_arrays)

    # ── Figure ────────────────────────────────────────────────────────────
    bottom_margin = 0.06 + 0.07 * n_params   # espace pour les sliders
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=bottom_margin, top=0.93)

    init_data   = get_data(init_indices)
    init_params = get_param_values(init_indices)
    artist      = plot_fn(ax, init_data, init_params)

    ax.set_title(title_fn(init_params))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    cbar = None
    if colorbar:
        cbar = fig.colorbar(artist, ax=ax, label=colorbar_label)

    # ── Sliders ───────────────────────────────────────────────────────────
    sliders: list[Slider] = []
    for k, name in enumerate(param_names):
        arr = param_arrays[k]
        y_pos = 0.03 + 0.065 * (n_params - 1 - k)
        ax_s  = plt.axes([0.15, y_pos, 0.72, 0.035])
        s = Slider(
            ax=ax_s,
            label=name,
            valmin=0,
            valmax=len(arr) - 1,
            valinit=init_indices[k],
            valstep=1,
            color=slider_color,
        )
        sliders.append(s)

    # Label de valeur réelle sous les sliders
    value_text = fig.text(
        0.5, 0.005,
        "  |  ".join(f"{n}={param_arrays[i][init_indices[i]]:.3g}"
                     for i, n in enumerate(param_names)),
        ha="center", fontsize=9, color="gray",
    )

    # ── Callback ─────────────────────────────────────────────────────────
    def update(_val):
        indices = tuple(int(s.val) for s in sliders)
        data    = get_data(indices)
        pvals   = get_param_values(indices)

        _update_artist(artist, data)
        ax.set_title(title_fn(pvals))
        value_text.set_text(
            "  |  ".join(f"{n}={param_arrays[i][indices[i]]:.3g}"
                         for i, n in enumerate(param_names))
        )
        if cbar is not None:
            try:
                cbar.update_normal(artist)
            except Exception:
                pass
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    plt.show()
    return fig, sliders