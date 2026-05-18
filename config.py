## Github: https://github.com/Liuuuv/Gabor-discret

############# BEGIN INFOS #############
# config file, change settings here
#
# (this file is imported by other files)
############# END INFOS ###############


from signal_test import signal_test
import numpy as np
import matplotlib.pyplot as plt
import librosa
from enum import Enum, auto

class StudiedSignal(Enum):
    THEORETICAL_REF = auto()
    PRACTICAL_SIGNAL = auto()



# studied_signal: StudiedSignal = StudiedSignal.THEORETICAL_REF
studied_signal: StudiedSignal = StudiedSignal.PRACTICAL_SIGNAL

############# WINDOWS #############
from windows import*
############# WINDOWS #############


match studied_signal:
    case StudiedSignal.THEORETICAL_REF:
        min_time = 0.0
        max_time = 1.0
    case StudiedSignal.PRACTICAL_SIGNAL:
        # min_time = 2.0
        # max_time = 5.0
        
        # min_time = 22.5
        # max_time = 25.5
        
        # ter
        # min_time = 22.5
        # max_time = 24.5
        
        min_time = 19.0
        max_time = 21.0









############ LOAD MP3 FILE ############
match studied_signal:
    case StudiedSignal.THEORETICAL_REF:
        signal, sr = signal_test, len(signal_test) # ref

        # signal = np.sin(2 * np.pi * 200 * time) # signal ref 2
        
        ## q=2
        # alpha: int = 50
        # beta: int = 5

        ## q=2
        # alpha: int = 25
        # beta: int = 10
        
        ## q=1
        # alpha: int = 20
        # beta: int = 25

        ## q=4
        alpha: int = 25
        beta: int = 5
        
    case StudiedSignal.PRACTICAL_SIGNAL:
        # chemin_fichier = "TERTrace/python/bad_apple_loop.mp3"
        # chemin_fichier = "TERTrace/python/kawaki_wo_ameku_short_sped_up.mp3"
        # chemin_fichier = "TERTrace/python/kawaki_wo_ameku_short_piano.mp3"
        chemin_fichier = "kawaki_wo_ameku.wav"
        
        # signal, sr = librosa.load(chemin_fichier, sr=None)
        # signal, sr = librosa.load(chemin_fichier, sr=4000)
        signal, sr = librosa.load(chemin_fichier, sr=4000)
        # signal, sr = librosa.load(chemin_fichier)

        
        alpha: int = 100
        beta: int = 10






signal = signal[int(sr * min_time):int(sr * max_time)]

L = len(signal)
duration = max_time - min_time

## L_sampling = [0, L//2[ U [-L//2, 0[ | to encode C^L vectors
L_sampling = np.arange(0, L, dtype=np.complex128)
L_sampling[L//2:] = np.arange(-L//2, 0, dtype=np.complex128)




## autre + pas signal_ref


beta_t = L//beta
alpha_t = L//alpha



################ BEGIN TOOLS #################
def discretize_window(window: callable, normalize=False, length=L): ## takes a function and discretizes it into a L-array
    if normalize:
        return window(np.linspace(-0.5, 0.5, length, dtype=np.complex128))
        # return window(L_sampling/L) # il faut plot [0,1]
    else:
        return window(L_sampling/length)

# def discretize_window(window: callable, normalize=False, length=L): ## takes a function and discretizes it into a L-array
#     discretized_window = window(np.linspace(-0.5, 0.5, length, dtype=np.complex128))
#     d_window_ = discretized_window.copy()
#     d_window_[:L//2] = discretized_window[L//2:]
#     d_window_[L//2:] = discretized_window[:L//2]
#     return d_window_
################ END TOOLS ###################



# window = ind_zero(0.05)
# sigma = 0.1999999955

sigma = 0.05
window = gaussian(sigma)

# l = 0.1
# window = blackman_window(l)

# window = gaussian_comp_supp(sigma)
# window = test_window(sigma)
# window = lambda t: window_(t) * np.sin(2 * np.pi * 100 * t)
d_window = discretize_window(window)

######## BEGIN VERIFICATIONS ########
print()
print("--------- BEGIN VERIFICATIONS config.py ---------")
print("L:", L, "; alpha:", alpha, "; beta:", beta, "; alpha*beta:", alpha*beta)
print("alpha_tilde", alpha_t, "; beta_tilde:", beta_t, "; alpha_tilde*beta_tilde:", alpha_t*beta_t)
print("alpha_tilde - beta =", alpha_t - beta)
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



if __name__ == "__main__":
    pass
    # for r in range(alpha_t):
    #     sums = np.zeros(alpha)
    #     for j in range(alpha):
            
            
    #         for k in range(beta_t):
                
    #             midsum = 0
    #             for l in range(alpha_t):
    #                 midsum = np.exp(-2 * ( (j - alpha * l + k * beta_t)**2)/(L**2 * sigma**2))
                
    #             sums[j] += beta_t * np.exp(-2j * np.pi * r * k / beta) * midsum

    #     # print(f"j = {j}", sums)
        
    #     plt.plot(np.arange(alpha)[40:], sums[40:])
    
    # plt.show()
    
    
            
    
    
    
    
    
    
