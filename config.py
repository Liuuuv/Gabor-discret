## Github: https://github.com/Liuuuv/Gabor-discret

############# BEGIN INFOS #############
# config file, change settings here
#
# (file imported by other files)


from signal_test import signal_test
import numpy as np

############# WINDOWS #############
def ind_zero(length: float): ## indicatrice normalisée centrée en zéro (sur l'ouvert de taille donnée)
    assert length > 0
    
    def ind(t_):
        if type(t_) is np.ndarray:
            t = t_.copy()
            for i in range(len(t)):
                t[i] = 1/np.sqrt(length) if abs(t[i]) < length/2 else 0
            return t
        else:
            return 1/np.sqrt(length) if abs(t_) < length/2 else 0
    return ind

def gaussian(sigma: float): ## indicatrice normalisée centrée en zéro (sur l'ouvert de taille donnée)
    assert sigma > 0
    return lambda t: np.exp(-np.pi*(t/sigma)**2) / sigma

def gaussian_comp_supp(sigma: float):
    assert sigma > 0
    
    def fonction(t_):
        if type(t_) is np.ndarray:
            t = t_.copy()
            for i in range(len(t)):
                t[i] = np.exp(-1/(sigma/2-abs(t[i]))) if abs(t[i]) < sigma else 0
            return t
        else:
            return np.exp(-1/(sigma/2-abs(t_))) if abs(t_) < sigma else 0
    return fonction

# def test_window(sigma: float):
#     def function(t_):
#         if type(t_) is np.ndarray:
#             t = t_.copy()
#             for i in range(len(t)):
#                 temp = t[i]+0.5
#                 if temp >= 0.5:
#                     temp -= 1
#                 t[i] = 1 * np.exp(-np.pi*(t[i]/sigma)**2) / sigma + 1j * np.exp(-np.pi*(temp/sigma)**2) / sigma
#             return t
#         else:
#             temp = t_+0.5
#             if temp >= 0.5:
#                 temp -= 1
#             return 1 * np.exp(-np.pi*(t_/sigma)**2) / sigma + 1j * np.exp(-np.pi*(temp/sigma)**2) / sigma
#     return function

def test_window(sigma: float):
    return lambda t: (np.exp(-np.pi*(t/sigma)**2) / sigma) * np.exp(2j * np.pi*(0.5*t/sigma)) ## gaussienne "tournante"

############# WINDOWS #############



min_time = 0
max_time = 1.0
duration = max_time - min_time


############ LOAD MP3 FILE ############
# chemin_fichier = "TERTrace/python/bad_apple_loop.mp3"
# chemin_fichier = "TERTrace/python/kawaki_wo_ameku.mp3"
# chemin_fichier = "TERTrace/python/kawaki_wo_ameku_piano.mp3"
# signal, sr = librosa.load(chemin_fichier, sr=None)
# signal, sr = librosa.load(chemin_fichier, sr=1500)


signal, sr = signal_test, len(signal_test) # ref

# signal = np.sin(2 * np.pi * 200 * time) # signal ref

signal = signal[int(sr * min_time):int(sr * max_time)]

L = len(signal)

## L_sampling = [0, L//2[ U [-L//2, 0[ | to encode C^L vectors
L_sampling = np.arange(0, L, dtype=np.complex64)
L_sampling[L//2:] = np.arange(-L//2, 0, dtype=np.complex64)


alpha: int = 5
beta: int = 25
beta_t = L//beta
alpha_t = L//alpha



################ BEGIN TOOLS ################
def discretize_window(window: callable, normalize=False, length=L): ## takes a function and discretizes it into a L-array
    if normalize:
        return window(np.linspace(-0.5, 0.5, length, dtype=np.complex64))
        # return window(L_sampling/L) # il faut plot [0,1]
    else:
        return window(L_sampling/length)
################ END TOOLS ###################



# window = ind_zero(0.4)
sigma = 0.08
window = gaussian(sigma)
# window = gaussian_comp_supp(sigma)
# window = test_window(sigma)
# window = lambda t: window_(t) * np.sin(2 * np.pi * 100 * t)
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

gcd = np.gcd(alpha * beta, L)
p = alpha * beta // gcd
q = L // gcd
print(f"alpha*beta / L = {p}/{q}")
print("--------- END VERIFICATIONS config.py -----------")
print()
######## END VERIFICATIONS ##########

