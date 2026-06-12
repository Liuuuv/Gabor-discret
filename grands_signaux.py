import numpy as np
import soundfile as sf
import time




from config import*
from tools import*
from dual_frame import*
from reconstitution import*


def write_large_audio_sequential(output_file, signal_chunks, samplerate=sr):
    """
    Écrit un fichier audio par fragments successifs
    
    Args:
        output_file: chemin du fichier de sortie
        samplerate: fréquence d'échantillonnage
        signal_chunks: liste ou générateur de fragments numpy
    """
    with sf.SoundFile(output_file, mode='w', samplerate=samplerate, 
                      channels=1, format='WAV') as f:
        for chunk in signal_chunks:
            f.write(np.real(chunk))

def break_signal_parts(signal: np.ndarray, max_len: int = 500) -> np.ndarray:
    parts = []
    len_signal = len(signal)
    num_parts = len_signal // max_len
    for i in range(num_parts - 1):
        parts.append(signal[i * max_len: (i + 1) * max_len])
    if not len_signal % max_len == 0:
        parts.append(signal[(len_signal - 1) * max_len:])
    return parts

def change_speed(signal: np.ndarray, speed_multiplier: float) -> np.ndarray: ## ne marche que pour les diviseurs de la longueur des fragments
    len_signal = len(signal)
    fragment_window = discretize_window(gaussian(sigma), len_signal)
    stdft = plot_fstdft(signal=signal, d_window=fragment_window)
    
    stdft[:len_signal//speed_multiplier] = stdft[::speed_multiplier]
    
    
    fragment_window = fragment_window[::speed_multiplier]
    
    
    d_dual_window = compute_dual_window(fragment_window)
    reconstructed_signal = reconstruct_signal(stdft[:len_signal//speed_multiplier], fragment_window, d_dual_window, alpha, beta)
    # if np.max(np.abs(reconstructed_signal)) > 1.0:
    #     reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))
    return reconstructed_signal

if __name__ == '__main__':
    
    fig, axes = create_subplots((4,1)) ## changer 1er argument accordement
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.4     # Espace entre les graphiques
    )
    
    # sf.write('fichier_entree.wav', signal, samplerate=sr)
   
    
    
    signal_fragments = break_signal_parts(signal=signal, max_len=500)
    
    write_large_audio_sequential('fichier_sortie.wav', signal_fragments, sr)
    
    for i in range(len(signal_fragments)):
        print(f"processing {i+1}/{len(signal_fragments)} fragments...")
        fragment = signal_fragments[i]
        new_fragment = change_speed(fragment, 2)
        signal_fragments[i] = new_fragment
    
    # Trouver le maximum global parmi tous les fragments
    max_normalization = max(np.max(fragment) for fragment in signal_fragments)

    # Normaliser chaque fragment
    for i in range(len(signal_fragments)):
        signal_fragments[i] = signal_fragments[i] / max_normalization
    
    write_large_audio_sequential('fichier_sortie.wav', signal_fragments, sr)
        
    
    # plot_signal(signal=signal, ax=axes[0])
    
    
    # produit_stdft()
    
    
    # stdft = plot_fstdft(signal=signal, ax=axes[1], plot_ref=False, linear=True, d_window=d_window, plot_only_grid=False, tolerance=-1, show_full=False)
    
    # stdft[:500,:] = 0.0
    # stdft[:2000,:] = 0.0
    
    # plot_fstdft(signal=signal, ax=axes[1], plot_ref=False, linear=True, d_window=d_window, plot_only_grid=False, tolerance=0.01, stdft=stdft)
    
    # canonical_dual_window = compute_dual_window(d_window, alpha=alpha, beta=beta)
    # d_dual_window = canonical_dual_window
    # reconstructed_signal = reconstruct_signal(stdft, d_window, d_dual_window, alpha, beta)

    # # S'assurer que le signal est dans [-1, 1] pour éviter la distorsion
    # reconstructed_signal = np.real(reconstructed_signal)
    # if np.max(np.abs(reconstructed_signal)) > 1.0:
    #     reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))

    # # Sauvegarder en WAV
    # print("Sauvegarde...")
    # sf.write('fichier_sortie.wav', reconstructed_signal, samplerate=sr)
    # write_large_audio_sequential('fichier_sortie.wav', signal_fragments, sr)
    
    # plot_signal(signal=reconstructed_signal, ax=axes[2])
    # plot_signal(signal=signal-reconstructed_signal, ax=axes[3], label="Erreur")
    
    
    # plt.savefig('ON_S_amuse.jpg', dpi=300)
    # plt.get_current_fig_manager().window.state('zoomed')
    # plt.show()