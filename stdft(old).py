import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy

min_time = 0.0
max_time = 0.2
duration = max_time - min_time

# Charger le fichier MP3
chemin_fichier = "TERTrace/python/bad_apple_loop.mp3"
# signal, sr = librosa.load(chemin_fichier, sr=None)
signal, sr = librosa.load(chemin_fichier, sr=4000)
signal = signal[:int(sr * max_time)]




time = np.arange(min_time, max_time, 1/sr)
signal = np.sin(2 * np.pi * 200 * time)


def ft(signal):
    N = len(signal) # nombre d'échantillons
    n = np.arange(N) # indices (somme sur n)
    result = np.zeros(N)
    for k in range(N):
        result[k] = np.abs(
            np.sum(signal * np.exp(-2j * np.pi * k * n / N))
        )
    return result



def stft(signal, x, window=lambda t: 1):
    N = len(signal)
    result = np.zeros(N)
    n = np.arange(N)
    for k in range(N):
        result[k] = np.abs(
            np.sum(signal * window(n - x) * np.exp(-2j * np.pi * k * n / N))
        )

    return result

def fstft(signal, x, window=lambda t: 1):
    N = len(signal)
    
    translated_window = np.ones(N)
    for k in range(N):
        t = k / sr
        translated_window[k] = window(t - x)
    result = fast_fft(signal * np.conjugate(translated_window))
    result = np.abs(result)
    return result



# def stft(signal, x, window_func=lambda x: 1):
#     result = np.zeros(len(signal))
#     time = np.arange(len(signal)) / sr
#     for xi in range(0, sr, max(1, len(signal)//1000)):
#         result[xi] = np.abs(np.sum(signal * np.conjugate(window_func(x - time)) * np.exp(-1j * xi * time)))

#     # result = np.abs(result)
#     return result




def fast_fft(signal):
    fourier = np.fft.fft(signal)
    fourier_abs = np.abs(fourier)
    return fourier_abs

def fast_stft(signal, x, window=lambda t: 1):
    f, t, Zxx = scipy.signal.stft(signal, fs=sr, window='hann')
    return f, t, np.abs(Zxx)

def omega(p, q):
   return np.exp((2.0 * np.pi * 1j * q) / p)

def fft(signal):
   n = len(signal)
   if n == 1:
      return signal
   else:
      Feven = fft([signal[i] for i in range(0, n, 2)])
      Fodd = fft([signal[i] for i in range(1, n, 2)])

      combined = [0] * n
      for m in range(n//2):
         combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
         combined[m + n//2] = Feven[m] - omega(n, -m) * Fodd[m]

      return combined


# Afficher les informations du signal
print(f"Fréquence d'échantillonnage : {sr} Hz")
print(f"Durée : {len(signal)/sr:.2f} secondes")
print(f"Nombre d'échantillons : {len(signal)}")


freq = np.linspace(0, sr//2, len(signal)//2)

# 1. Signal temporel complet
temps = np.arange(len(signal)) / sr






print("Calcul de la STFT...")

# T = temps
# X = np.linspace(20, 300, temps.shape[0])
X = temps
MODULE = np.zeros(temps.shape[0], dtype=np.ndarray)
COLOR = np.zeros((X.shape[0], temps.shape[0]), dtype=float)
for i in range(len(X)):
    MODULE[i] = fstft(signal, X[i], lambda t: np.exp(-(t*200)**2))
    for j in range(len(MODULE[i])):
        COLOR[i,j] = MODULE[i][j]
    # plt.scatter(freq, stft(signal, x, lambda t: np.exp(-t**2)), color='green', alpha=0.7, linewidth=0.5, marker='+')
# plt.set_xscale('log')
plt.title("STFT")

print("shape ", COLOR.shape)

# COLOR /= np.max(COLOR)
# print(COLOR)
# f, t, M = fstft(signal, lambda t: np.exp(-(t*200)**2))
# plt.pcolormesh(t, f, Zxx, shading='gouraud')
# plt.imshow(COLOR, interpolation='bilinear')
# plt.show()

plt.close()
print('djsdjszd')
f, t, Zxx = fast_stft(signal=signal, x=temps, window=lambda t: np.exp(-(t*200)**2))
plt.pcolormesh(t, f, Zxx, shading='gouraud')
plt.ylabel('Hz')
plt.xlabel('Sec')


# plt.scatter(T, X, edgecolors='none',c=COLOR.flatten())


# temps2 = np.linspace(0, len(signal), 200)
# for t in temps:
#     y = stft_single_freq(signal, t, lambda x: np.exp(-x**2))
#     # print(freq.shape)
#     # print(y.shape)
#     axes[3].scatter(np.ones(len(y)) * t, y, color='red', alpha=0.7, linewidth=0.5, marker='.')
# axes[3].set_xscale('log')


plt.get_current_fig_manager().window.state('zoomed')
plt.tight_layout()
plt.show()


