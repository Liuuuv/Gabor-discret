import numpy as np
import soundfile as sf
import time




from config import*
from tools import*
from dual_frame import*
from reconstitution import*


def produit_stdft():
    stdft1 = plot_fstdft(signal=signal, ax=axes[1], plot_ref=False, linear=True, d_window=discretize_window(gaussian(0.007)))
    stdft2 = plot_fstdft(signal=signal, ax=axes[2], plot_ref=False, linear=True, d_window=discretize_window(gaussian(0.19)))
    
    stdft = stdft1 * stdft2
    
    stdft = np.sqrt(np.abs(stdft))
    
    plot_fstdft(signal=signal, ax=axes[3], plot_ref=False, linear=True, stdft=stdft)


def test_doubles_inversions():
    
    S = construct_operator_matrix(window=d_window)
    axes[0].imshow(np.abs(S), label='S_g')
    
    
    S_reconstructed = construct_matrix_from_circulant(construct_circulant(S))
    print("erreur circularité:", np.max(np.abs(S - S_reconstructed)))
    
    
    print(f"-- time start --")
    start_time = time.time()
    S_circ = construct_circulant(S, mean=False)
    S_inv_circ = circulant_inverse(S_circ)
    S_inv = construct_matrix_from_circulant(S_inv_circ)
    print(f"time taken: {time.time() - start_time}")
    
    print(f"-- time start --")
    start_time = time.time()
    S_circ = construct_circulant(S_inv, mean=False)
    S_inv_circ= circulant_inverse(S_circ)
    S_inv = construct_matrix_from_circulant(S_inv_circ)
    axes[1].imshow(np.abs(S_inv), label='inverse 2 fois')
    print(f"time taken: {time.time() - start_time}")
    
    print("erreur double inversion", np.max(np.abs(S -  S_inv)))
    print("max", np.max(np.abs(S)))
    print("min", np.min(np.abs(S)))
    
    print(f"-- time start --")
    start_time = time.time()
    S_inv = np.linalg.inv(S)
    S_inv = np.linalg.inv(S_inv)
    print(f"time taken: {time.time() - start_time}")
    print("erreur double inversion naive", np.max(np.abs(S -  S_inv)))




if __name__ == '__main__':
    
    fig, axes = create_subplots((4,1)) ## changer 1er argument accordement
    # plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(
        top=0.95,      # Espace au-dessus du premier graphique (1.0 = bord haut)
        bottom=0.05,   # Espace en-dessous du dernier graphique (0.0 = bord bas)
        hspace=0.4     # Espace entre les graphiques
    )
    
    sf.write('TERTrace/python/fichier_entree.wav', signal, samplerate=sr)
    
    plot_signal(signal=signal, ax=axes[0])
    
    
    # produit_stdft()
    
    
    stdft = plot_fstdft(signal=signal, ax=axes[1], plot_ref=False, linear=True, d_window=d_window, plot_only_grid=False, tolerance=-1, show_full=False)
    
    # stdft[:500,:] = 0.0
    # stdft[:2000,:] = 0.0
    
    # plot_fstdft(signal=signal, ax=axes[1], plot_ref=False, linear=True, d_window=d_window, plot_only_grid=False, tolerance=0.01, stdft=stdft)
    
    canonical_dual_window = compute_dual_window(window, alpha=alpha, beta=beta)
    d_dual_window = canonical_dual_window
    reconstructed_signal = reconstruct_signal(stdft, d_window, d_dual_window, alpha, beta)

    # S'assurer que le signal est dans [-1, 1] pour éviter la distorsion
    reconstructed_signal = np.real(reconstructed_signal)
    if np.max(np.abs(reconstructed_signal)) > 1.0:
        reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))

    # Sauvegarder en WAV
    print("Sauvegarde...")
    sf.write('TERTrace/python/fichier_sortie.wav', reconstructed_signal, samplerate=sr)
    
    plot_signal(signal=reconstructed_signal, ax=axes[2])
    plot_signal(signal=signal-reconstructed_signal, ax=axes[3], label="Erreur")
    
    
    plt.savefig('TERTrace/python/ON_S_amuse.jpg', dpi=300)
    # plt.get_current_fig_manager().window.state('zoomed')
    # plt.show()