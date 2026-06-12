import numpy as np
import soundfile as sf
import time




from config import*
from tools import*
from dual_frame import*
from reconstitution import*





if __name__ == '__main__':
    
    fig, axes = create_subplots((4,1)) ## changer 1er argument accordement
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.4     # Espace entre les graphiques
    )
    
    sf.write('fichier_entree.wav', signal, samplerate=sr)
    
    plot_signal(signal=signal, ax=axes[0])
    
    
    
    
    stdft = plot_fstdft(signal=signal, plot_ref=False, linear=True, d_window=d_window, plot_only_grid=False, tolerance=-1, show_full=False)
    
    # stdft[:500,:] = 0.0
    # stdft[:2000,:] = 0.0
    mult: int = 2
    stdft[:,:L//mult] = stdft[:,::mult]
    stdft[:,L//mult:] = 0.0
    
    plot_fstdft(signal=signal, ax=axes[1], plot_ref=False, linear=True, d_window=d_window, plot_only_grid=False, tolerance=0.01, stdft=stdft)
    
    canonical_dual_window = compute_dual_window(d_window, alpha=alpha, beta=beta)
    d_dual_window = canonical_dual_window
    reconstructed_signal = reconstruct_signal(stdft, d_window, d_dual_window, alpha, beta)

    # S'assurer que le signal est dans [-1, 1] pour éviter la distorsion
    reconstructed_signal = np.real(reconstructed_signal)
    if np.max(np.abs(reconstructed_signal)) > 1.0:
        reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))

    
    # Sauvegarder en WAV
    print("Sauvegarde...")
    sf.write('fichier_sortie.wav', reconstructed_signal[:L//mult], samplerate=sr)
    
    plot_signal(signal=reconstructed_signal, ax=axes[2])
    plot_signal(signal=signal-reconstructed_signal, ax=axes[3], label="Erreur")
    
    
    plt.savefig('ON_S_amuse.jpg', dpi=300)
    # plt.get_current_fig_manager().window.state('zoomed')
    # plt.show()