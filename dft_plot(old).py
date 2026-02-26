import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference

min_time = 0.0
max_time = 1.0
duration = max_time - min_time

# Charger le fichier MP3
chemin_fichier = "TERTrace/python/bad_apple_loop.mp3"
# signal, sr = librosa.load(chemin_fichier, sr=None)
# signal, sr = librosa.load(chemin_fichier, sr=48000)
signal, sr = signal_test, len(signal_test)

time = np.arange(min_time, max_time, 1/sr)
signal = np.sin(2 * np.pi * 10 * time)

# signal = signal[:int(sr * max_time)]




def plot_signal(signal, ax_index):
    temps = np.arange(len(signal)) / sr
    axes[ax_index].plot(temps, signal, color='blue', alpha=0.7, linewidth=0.5)
    axes[ax_index].set_title(f"Fréquence d'échantillonnage: {sr}")
    axes[ax_index].set_xlabel("Progression")
    axes[ax_index].set_ylabel("Amplitude")
    axes[ax_index].grid(True, alpha=0.3)
    axes[ax_index].margins(0, x=None, y=None, tight=True)

def ft(signal):
    L = len(signal) # nombre d'échantillons
    n = np.arange(L) # indices (somme sur n)
    result = np.zeros(L, dtype=np.complex64)
    for k in range(L):
        result[k] = np.sum(signal * np.exp(-2j * np.pi * k * n / L))
    return result

def fft(signal):
    fourier = np.fft.fft(signal)
    return fourier

def fstdft(signal, window=lambda t: 1):
    N = len(signal)
    temps = np.arange(N) / sr
    result = np.zeros((N, N), dtype=np.complex64)
    for i in range(N):
        x = temps[i]
        if x < 0:
            continue
        translated_window = np.ones(N)
        for k in range(N):
            t = k / sr
            translated_window[k] = window(t - x)
        result[i] = fft(signal * np.conjugate(translated_window))
    return result.transpose()


def plot_dft(signal, ax_index, module_only = False):
    print("Calcul de la DFT...")
    freq = np.linspace(0, sr//2, len(signal)//2)
    ft_signal = ft(signal)[:sr//2]
    if module_only:
        axes[ax_index].plot(freq, np.abs(ft_signal), color='blue', alpha=0.7, linewidth=0.5)
    else:
        axes[ax_index].plot(freq, np.imag(ft_signal), color='red', alpha=0.7, linewidth=0.5)
        axes[ax_index].plot(freq, np.real(ft_signal), color='blue', alpha=0.7, linewidth=0.5)
    axes[ax_index].set_xscale('log')
    axes[ax_index].set_title("DFT")
    axes[ax_index].grid()
    axes[ax_index].set_xlabel("Fréquence")
    if module_only:
        axes[ax_index].set_ylabel("Module")
    else:
        axes[ax_index].set_ylabel("Partie réelle/imaginaire (bleu, rouge resp.)")


def plot_fft(signal, ax_index, module_only = False):
    print("Calcul de la FFT...")
    freq = np.linspace(0, sr//2, len(signal)//2)
    ft_signal = fft(signal)[:sr//2]
    if module_only:
        axes[ax_index].plot(freq, np.abs(ft_signal), color='blue', alpha=0.7, linewidth=0.5)
    else:
        axes[ax_index].plot(freq, np.imag(ft_signal), color='red', alpha=0.7, linewidth=0.5)
        axes[ax_index].plot(freq, np.real(ft_signal), color='blue', alpha=0.7, linewidth=0.5)
    axes[ax_index].set_xscale('log')
    axes[ax_index].set_title("FFT")
    axes[ax_index].set_xlabel("Fréquence")
    axes[ax_index].grid(True, alpha=0.3)
    if module_only:
        axes[ax_index].set_ylabel("Module")
    else:
        axes[ax_index].set_ylabel("Partie réelle/imaginaire (bleu, rouge resp.)")


def plot_fstdft(signal, ax_index):
    
    print("Calcul de la FSTDFT...")
    # result = fstdft(signal=signal, window=lambda t: 1)
    # result = fstdft(signal=signal, window=ind_zero(0.08))
    result = fstdft(signal=signal, window=lambda t: np.exp(-t**2 / 0.0009))
    # result = fstdft(signal=signal, window=lambda t: np.exp(-t**2 / 0.1))
    
    result = result[:sr//2,:]
    result = np.abs(result)**2
    result /= np.max(result)
    
    
    result[result < 0.03] = 0
    nonzero_y, nonzero_x = np.nonzero(result)
    
    if len(nonzero_y) == 0:
        return 0  # Tout est nul, seuil à 0
    
    last_nonzero_y = np.max(nonzero_y)
    result = result[:last_nonzero_y + 1,:]
    
    
    
    freq = np.linspace(0, sr//2, len(signal)//2)
    temps = np.arange(len(signal)) / sr
    freq = freq[:last_nonzero_y + 1]
    axes[ax_index].pcolormesh(temps, freq, result, shading='gouraud')
    # axes[ax_index].set_yscale('log')
    axes[ax_index].set_title("Spectrogramme d'une FSTDFT, normalisé (norme sup)")
    axes[ax_index].set_ylabel('Hz')
    axes[ax_index].set_xlabel('Progression')
    
    plot_time_frequencies_reference(ax=axes[ax_index])


def plot_scipy_fstdft(signal, ax_index): ## SCIPY
    
    print("Calcul de la FSTDFT (scipy)...")
    f, t, result = scipy.signal.stft(signal, fs=sr, window='hamming')
    result = result[:sr//2,:]
    result = np.abs(result)**2
    result /= np.max(result)
    
    result[result < 0.3] = 0
    nonzero_y, nonzero_x = np.nonzero(result)
    
    if len(nonzero_y) == 0:
        return 0  # Tout est nul, seuil à 0
    
    last_nonzero_y = np.max(nonzero_y)
    result = result[:last_nonzero_y + 1,:]
    
    
    freq = np.linspace(0, sr//2, len(signal)//2)
    temps = np.arange(len(signal)) / sr
    f = f[:last_nonzero_y + 1]
    axes[ax_index].pcolormesh(t, f, result, shading='gouraud')
    # axes[ax_index].set_yscale('log')
    axes[ax_index].set_title("Spectrogramme d'une FSTDFT, normalisé (norme sup) (scipy)")
    axes[ax_index].set_ylabel('Hz')
    axes[ax_index].set_xlabel('Progression')


def fast_stft(signal, x, window=lambda t: 1):
    f, t, Zxx = scipy.signal.stft(signal, fs=sr, window='hann')
    return f, t, np.abs(Zxx)


## WINDOWS
def ind_zero(length: float): ## indicatrice normalisée centrée en zéro (sur l'ouvert de taille donnée)
    return lambda t: 1/np.sqrt(length) if abs(t) < length/2 else 0



fig, axes = plt.subplots(2, 1, figsize=(14, 10)) ## changer ça accordement

plot_signal(signal, 0)
plot_dft(signal, ax_index=1, module_only=True)
# plot_fstdft(signal, ax_index=1)
# plot_scipy_fstdft(signal, ax_index=1)
# plot_fft(signal, ax_index=1, module_only=True)

plt.get_current_fig_manager().window.state('zoomed')
plt.tight_layout()

# plt.savefig('plot.pdf', bbox_inches='tight')  # Format vectoriel
plt.savefig('signal_temporel.jpg', dpi=300)  # JPG avec qualité
# plt.show()



