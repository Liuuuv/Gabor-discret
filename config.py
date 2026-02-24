from signal_test import signal_test
import numpy as np

############# WINDOWS #############
def ind_zero(length: float): ## indicatrice normalisée centrée en zéro (sur l'ouvert de taille donnée)
    assert length > 0
    
    # mask = (t >= length) & (t <= -length)
    # n_mask = n[mask]
    def ind(t_):
        if type(t_) is np.ndarray:
            t = t_.copy()
            for i in range(len(t)):
                t[i] = 1/np.sqrt(length) if abs(t[i]) < length/2 else 0
            return t
        else:
            return 1/np.sqrt(length) if abs(t_) < length/2 else 0
    return ind
    # return lambda t: 1/np.sqrt(length) if abs(t) < length/2 else 0


def gaussian(sigma: float): ## indicatrice normalisée centrée en zéro (sur l'ouvert de taille donnée)
    assert sigma > 0
    return lambda t: np.exp(-np.pi*(t/sigma)**2) / sigma

############# WINDOWS #############



min_time = 0
max_time = 1.0
duration = max_time - min_time


# Charger le fichier MP3
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

# L_sampling = [0, L//2[ U [-L//2, 0[
L_sampling = np.arange(0, L)
L_sampling[L//2:] = np.arange(-L//2, 0)


alpha: int = 20
beta: int = 25
beta_t = L//beta
alpha_t = L//alpha




def discretize_window(window: callable, normalize=False): ## takes a function and discretizes it into a L-array
    if normalize:
        return window(np.linspace(-0.5, 0.5, L))
        # return window(L_sampling/L) # il faut plot [0,1]
    else:
        return window(L_sampling/L)




# window = ind_zero(0.1)
sigma = 0.1
window = gaussian(sigma)
# window = lambda t: window_(t) * np.sin(2 * np.pi * 100 * t)
# window = lambda t: window_(t - L//2)
d_window = discretize_window(window)

######## BEGIN VERIFICATIONS ########
print()
print("--------- BEGIN VERIFICATIONS config.py ---------")
print("L:", L, "; alpha:", alpha, "; beta:", beta, "; alpha*beta:", alpha*beta)
if L % beta != 0:
    print("BETA NE DIVISE PAS L")
elif L % alpha != 0:
    print("ALPHA NE DIVISE PAS L")
else:
    print("OK DIVISIBILITÉ")


if alpha * beta > L:
    print("SOUS-ECHANTILLONAGE :C")
elif alpha * beta == L:
    print("CAS CRITIQUE :D")
else:
    print("SUR-ECHANTILLONAGE :D")
print("--------- END VERIFICATIONS config.py -----------")
print()
######## END VERIFICATIONS ##########

