import numpy as np
import soundfile as sf
import time


from config import*
from tools import*
from dual_frame import*
from reconstitution import*



if __name__ == '__main__':
    
    fig, axes = create_subplots((1,1)) ## changer 1er argument accordement
    if type(axes) != np.ndarray:
        axes = np.array([axes])
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.4     # Espace entre les graphiques
    )
    
    
    # plot_signal(signal=signal, ax=axes[0])
    plot_fft(signal, axes[0], module_only=True, label="Transformée de Fourier (module)")
    
    
    # produit_stdft()
    
    
    # stdft = plot_fstdft(signal=signal, ax=axes[1], plot_ref=False, linear=True, d_window=d_window, plot_only_grid=False, tolerance=-1, show_full=False)
    
    # stdft[:500,:] = 0.0
    # stdft[:2000,:] = 0.0
    
    # plot_fstdft(signal=signal, ax=axes[1], plot_ref=False, linear=True, d_window=d_window, plot_only_grid=False, tolerance=0.01, stdft=stdft)
    
    # plt.get_current_fig_manager().window.state('zoomed')
    plt.show()