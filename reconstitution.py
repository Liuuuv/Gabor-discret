import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from dual_frame import compute_dual_window
from config import*

min_time = 0
max_time = 1.0
duration = max_time - min_time


# Charger le fichier MP3w
# chemin_fichier = "TERTrace/python/bad_apple_loop.mp3"
# chemin_fichier = "TERTrace/python/kawaki_wo_ameku.mp3"
# chemin_fichier = "TERTrace/python/kawaki_wo_ameku_piano.mp3"
# signal, sr = librosa.load(chemin_fichier, sr=None)
# signal, sr = librosa.load(chemin_fichier, sr=1500)


signal, sr = signal_test, len(signal_test) # ref

# time = np.arange(min_time, max_time, 1/sr)
# signal = np.sin(2 * np.pi * 200 * time)

signal = signal[int(sr * min_time):int(sr * max_time)]

L = len(signal)



def plot_signal(signal, ax_index):
    # temps = np.arange(len(signal)) / sr
    temps = np.linspace(min_time, max_time, len(signal))
    axes[ax_index].plot(temps, signal, color='blue', alpha=0.7, linewidth=0.5)
    axes[ax_index].set_title(f"Fréquence d'échantillonnage: {sr}")
    axes[ax_index].set_xlabel("Progression")
    axes[ax_index].set_ylabel("Amplitude")
    axes[ax_index].grid(True, alpha=0.3)
    axes[ax_index].margins(0, x=None, y=None, tight=True)

def ft(signal):
    N = len(signal) # nombre d'échantillons
    n = np.arange(N) # indices (somme sur n)
    result = np.zeros(N, dtype=np.complex64)
    for k in range(N):
        result[k] = np.sum(signal * np.exp(-2j * np.pi * k * n / N))
    return result

def fft(signal):
    fourier = np.fft.fft(signal)
    return fourier

def fstdft(signal, window=lambda t: 1):
    window = discretize_window(window=window)
    N = len(signal)
    result = np.zeros((N, N), dtype=np.complex64)
    for i in range(N):
        translated_window = np.ones(N)
        for k in range(N):
            t = k
            translated_window[k] = window[t - i]
        result[i] = fft(signal * np.conjugate(translated_window))
    return result.transpose()

def reconstruct_signal(coefs, window, dual_window = None, alpha: int=1, beta: int=1):
    N = coefs.shape[0]
    signal = np.zeros(N, dtype=np.complex64)
    if dual_window is None:
        dual_window = window
    for n in range(N):
        for k in np.arange(0, N, alpha):
            l = np.arange(0, N, beta)
            tm_dual = dual_window[n - k] * np.exp(1j * 2 * np.pi * (n/N) * l)
            signal[n] += np.sum(tm_dual * coefs[::beta,k], dtype=np.complex64)
    # print("produit scalaire", np.sum(np.conjugate(window[x])*dual_window[x] for x in np.arange(N)))
    # signal /= N * np.sum(np.conjugate(window[x]) * dual_window[x] for x in np.arange(N)) + 10e-8
    return signal

    
    
    

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


def plot_fstdft(signal, ax_index, window, plot_ref=True, label=""):
    print("Calcul de la FSTDFT...")
    result_raw = fstdft(signal=signal, window=window)
    
    
    result = result_raw[:L//2,:]
    result = np.abs(result)**2
    # result = np.abs(result)
    # result = 10*np.abs(result)
    result /= np.max(result) if np.max(result) != 0 else 1
    
    
    result[result < 0.005] = 0
    nonzero_y, nonzero_x = np.nonzero(result)
    
    if len(nonzero_y) == 0:
        return 0  # Tout est nul, seuil à 0
    
    last_nonzero_y = np.max(nonzero_y)
    # result = result[:last_nonzero_y + 1,:]
    
    
    
    freq = np.linspace(0, L//2, len(signal)//2)
    # temps = np.arange(len(signal)) / sr
    temps = np.linspace(min_time, max_time, int(duration * sr))
    # freq = freq[:last_nonzero_y + 1]
    axes[ax_index].pcolormesh(temps, freq, result, shading='gouraud')
    # axes[ax_index].set_yscale('log')
    if label != "":
        axes[ax_index].set_title(label)
    else:
        axes[ax_index].set_title("Transformée de Fourier en temps court")
    axes[ax_index].set_ylabel('Hz')
    axes[ax_index].set_xlabel('Progression')
    axes[ax_index].set_xlim(min_time, max_time)
    axes[ax_index].set_ylim(0, last_nonzero_y + 1)
    
    if plot_ref:
        plot_time_frequencies_reference(ax=axes[ax_index])
    
    return result_raw


def plot_scipy_fstdft(signal, ax_index, window=None): ## SCIPY
    print("Calcul de la FSTDFT (scipy)...")
    N = len(signal)
    if window:
        window_array = window(np.linspace(0,1,N))
        f, t, result = scipy.signal.stft(signal, fs=sr, window=window_array, nperseg=N, noverlap=128)
    else:
        f, t, result = scipy.signal.stft(signal, fs=sr, window='hann', nperseg=256, noverlap=128)
    # result = result[:sr//2,:]
    result = np.abs(result)
    result /= np.max(result)
    
    result[result < 0.3] = 0
    nonzero_y, nonzero_x = np.nonzero(result)
    
    if len(nonzero_y) == 0:
        return 0  # Tout est nul, seuil à 0
    
    last_nonzero_y = np.max(nonzero_y)
    # result = result[:last_nonzero_y + 1,:]
    
    
    freq = np.linspace(0, sr//2, len(signal)//2)
    temps = np.arange(len(signal)) / sr
    # f = f[:last_nonzero_y + 1]
    axes[ax_index].pcolormesh(t, f, result, shading='gouraud')
    # axes[ax_index].set_yscale('log')
    axes[ax_index].set_title("Spectrogramme d'une FSTDFT, normalisé (norme sup) (scipy)")
    axes[ax_index].set_ylabel('Hz')
    axes[ax_index].set_xlabel('Progression')
    axes[ax_index].set_ylim(0, last_nonzero_y + 1)


def plot_window(window, ax_index, label=""):
    axes[ax_index].plot(np.linspace(-0.5,0.5,L), np.real(window), color='blue', alpha=0.7, linewidth=0.7)
    axes[ax_index].plot(np.linspace(-0.5,0.5,L), np.imag(window), color='red', alpha=0.7, linewidth=0.7)
    # axes[ax_index].set_xlabel("Progression")
    # axes[ax_index].set_ylabel("Amplitude")
    axes[ax_index].grid(True, alpha=0.3)
    axes[ax_index].margins(0, x=None, y=None, tight=True)
    
    # axes[ax_index].plot(np.linspace(-0.5,0.5,L), discretize_window(window, True))
    axes[ax_index].set_title(label)


def fast_stft(signal, x, window=lambda t: 1):
    f, t, Zxx = scipy.signal.stft(signal, fs=sr, window='hann')
    return f, t, np.abs(Zxx)



################ BEGIN TO IGNORE ################
def get_max_indexes(coefs, num_max_indexes):
    # Obtenir à la fois les valeurs et leurs positions
    k = num_max_indexes
    flat_arr = coefs.flatten()

    # Indices des k plus grandes valeurs
    indices = np.argpartition(flat_arr, -k)[-k:]
    # print(np.argsort(flat_arr))
    # rows, cols = np.unravel_index(np.argsort(flat_arr), coefs.shape)

    # return rows, cols
    
    # Valeurs correspondantes
    values = flat_arr[indices]

    # Trier par valeur décroissante
    sorted_order = np.argsort(values)[::-1]
    values_sorted = values[sorted_order]
    indices_sorted = indices[sorted_order]

    # Convertir les indices plats en indices 2D
    rows, cols = np.unravel_index(indices_sorted, coefs.shape)

    # print("meilleures 40 valeurs avec leurs positions:")
    # for i in range(k):
    #     print(f"{i+1}: valeur={values_sorted[i]:.4f} à position ({rows[i]}, {cols[i]})")
    return rows, cols


## GET BEST COEFFICIENTS
# num_max_indexes = 10000
# rows, cols = get_max_indexes(np.abs(result), num_max_indexes)
# rows, cols = rows[:num_max_indexes], cols[:num_max_indexes]

# result_ = np.zeros(result.shape)
# for i in range(num_max_indexes):
#     result_[rows[i],cols[i]] = result[rows[i],cols[i]]


# d_dual_window = discretize_window(window) # temp
# d_dual_window = np.zeros(L) # temp


## END - GET BEST COEFFICIENTS
################ END TO IGNORE ##################






fig, axes = plt.subplots(6, 1, figsize=(14, 10)) ## changer 1er argument accordement


plot_signal(signal, ax_index=0)
result = plot_fstdft(signal, ax_index=1, window=window, plot_ref=False)
# plot_scipy_fstdft(signal, ax_index=2, window=window) ## je n'arrive pas à le faire fonctionner correctement..
# plot_dft(signal, ax_index=2, module_only=True)
# plot_fft(signal, ax_index=2, module_only=True)

plot_window(discretize_window(window, True), ax_index=4, label="Fenêtre")



d_dual_window = compute_dual_window(window, alpha=alpha, beta=beta)
reconstructed_signal = reconstruct_signal(result, discretize_window(window), d_dual_window, alpha, beta)
plot_signal(reconstructed_signal, 2)





## pour la visualisation
dual_window_vis = d_dual_window.copy()
dual_window_vis[:L//2] = d_dual_window[L//2:]
dual_window_vis[L//2:] = d_dual_window[:L//2]
plot_window(dual_window_vis, ax_index=5, label="Fenêtre duale")


## plot grille alphaZ x betaZ
x = np.arange(0, L, alpha)/L
y = np.arange(0, L, beta)
X, Y = np.meshgrid(x, y)
X_flat = X.flatten()
Y_flat = Y.flatten()
axes[1].scatter(X_flat, Y_flat, marker='+', color='orange', alpha=0.8)



plot_fstdft(reconstructed_signal, ax_index=3, window=window, plot_ref=False, label="STFT du signal reconstruit")




## finish the plot and save
plt.get_current_fig_manager().window.state('zoomed')
plt.tight_layout()

# plt.savefig('plot.pdf', bbox_inches='tight')  # Format vectoriel
plt.savefig('signal_temporel.jpg', dpi=300)
# plt.show()







