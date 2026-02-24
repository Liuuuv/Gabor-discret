import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from config import*

## matrice circulante: [A_i] où A_i: np.ndarray


def pos_mod(a, mod): ## inutile car les array le font très bien tout seul
    """positive modulo, a: int or a: np.ndarray

    Args:
        a (int or np.array): 
        mod (int): 

    Returns:
        int or np.array: a modulo mod, with a >= 0
    """
    assert mod >= 0
    remain = a % mod
    if type(remain) is np.ndarray:
        for i in range(len(remain)):
            if remain[i] < 0:
                remain[i] + mod
        return remain
    return remain if remain >= 0 else remain + mod

def construct_operator_matrix(window, alpha=alpha, beta=beta):
    beta_t = L//beta
    alpha_t = L//alpha
    
    matrix = np.ones((L, L), dtype=np.complex64)
    for j in range(L):
        if j % beta_t == 0:
            for d in range(L):
                n = np.arange(alpha_t)
                matrix[int(j+d) % L, int(d) % L] = float(beta_t) * sum(
                    np.conjugate(window[int(j+d)%L - alpha * n]) * window[int(d)%L - alpha * n]
                )
                
                ####### version pos_mod #######
                # matrix[int(j+d) % L, int(d) % L] = float(beta_t) * sum(
                #     np.conjugate(window[pos_mod(int(j+d)%L - alpha * n, L)]) * window[pos_mod(int(d)%L - alpha * n, L)]
                # )
                
        else: # 0
            for d in range(L):
                matrix[int(j+d) % L, int(d) % L] = 0.0
    
    # plt.figure()
    # plt.imshow(np.abs(matrix))
    # plt.title("runtime S")
    # plt.show()
    
    return matrix
            

############## BEGIN TO IGNORE ##############
def circlulant_ft(C):
    C_fourier = []
    N = len(C)
    for i in range(N):
        C_fourier.append(
            (1/N) * np.sum(
                np.exp((-1j * 2 * np.pi / N) * i * np.arange(N)) * C
            )
        )
    return C_fourier


def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


def reconstruct_circulant(C): ## cas alpha=1
    matrix = [
        shift(C, k)
    for k in len(C)
    ]
    return np.array(matrix)
############## END TO IGNORE ################


def compute_dual_window(window, alpha, beta):
    print("Calcul de la fenêtre duale")
    
    # print("d_window", d_window)
    S = construct_operator_matrix(d_window, alpha, beta)
    S_inv = np.linalg.inv(S)
    # d_window = window(np.arange(L))
    
    dual_window = np.matmul(S_inv, d_window)
    return dual_window



## cas alpha = 1
# sigma = 50
# window = gaussian(sigma)

# window = lambda t: window_(t - L/2)





def plot_window(window_, ax, is_discrete=True, label=""):
    if is_discrete:
        window = window_.copy()
        window[:L//2] = window_[L//2:]
        window[L//2:] = window_[:L//2]
    else:
        window = window_
    
    ax.plot(np.linspace(-0.5,0.5,L), np.real(window), color='blue', alpha=0.7, linewidth=1.0)
    ax.plot(np.linspace(-0.5,0.5,L), np.imag(window), color='red', alpha=0.7, linewidth=1.0)
    ax.set_xlabel("Progression")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.margins(0, x=None, y=None, tight=True)
    
    # axes[ax_index].plot(np.linspace(-0.5,0.5,L), discretize_window(window, True))
    ax.set_title(label)




if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    
    ##################### CUSTOM VARIABLES BEGIN #####################
    ## if not declared here, uses the config.py values
    # alpha: int = 2
    # beta: int = 250
    
    # sigma = 0.1
    # window = gaussian(sigma)
    # # window = lambda t: window_(t) * np.sin(2 * np.pi * 100 * t)
    # # window = lambda t: window_(t - L//2)
    # d_window = discretize_window(window)
    ##################### CUSTOM VARIABLES END #####################
    
    plot_window(d_window, ax=axes[0], label="Fenêtre (partie réelle/imag en bleu/rouge resp.)")


    S = construct_operator_matrix(d_window, alpha=alpha, beta=beta)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(np.abs(S), cmap='viridis', aspect='auto')
    ax2.set_title("Matrice S")
    plt.colorbar(im, ax=ax2)

    S_inv = np.linalg.inv(S)
    dual_window = np.matmul(S_inv, d_window)
    plot_window(dual_window, ax=axes[1], label="Fenêtre duale (partie réelle/imag en bleu/rouge resp.)")
    
    plt.show()




