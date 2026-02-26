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

def compute_tight_frame(d_window, alpha, beta, S=None):
    if S is None:
        S = construct_operator_matrix(d_window, alpha=alpha, beta=beta)
    
    
    eigenvalues, eigenvectors = np.linalg.eig(S)
    
    eigenvalues = np.real(eigenvalues)
    if min(eigenvalues) <= 0.0:
        print("VALEURS PROPRES NEGATIVES OU NULLES !!!!!", min(eigenvalues))
        mask = (eigenvalues < 0)
        eigenvalues[mask] = 10e-8
    # print(np.sort(eigenvalues))
    
    diagonal_matrix = np.diag(1/np.sqrt(eigenvalues))
    Smdemi = eigenvectors @ diagonal_matrix @ np.linalg.inv(eigenvectors)
    
    d_window_mdemi = Smdemi @ d_window
    
    S_tight = construct_operator_matrix(d_window_mdemi, alpha=alpha, beta=beta)
    return S_tight, d_window_mdemi

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
    # ax.set_xlabel("Progression")
    # ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.margins(0, x=None, y=None, tight=True)
    
    # axes[ax_index].plot(np.linspace(-0.5,0.5,L), discretize_window(window, True))
    ax.set_title(label)


def plot_complex_curve(curve_, ax, fig, color):
    # plt.figure().add_subplot(projection='3d')
    
    curve = curve_.copy()
    curve[:L//2] = curve_[L//2:]
    curve[L//2:] = curve_[:L//2]
    
    # pos = ax.get_position()
    # ax.remove()
    
    # # Crée un nouvel axe 3D
    # ax_3d = fig.add_axes(pos, projection='3d')
    
    curve /= np.max(np.abs(curve))
    
    ax.plot(np.real(curve), np.linspace(-0.5,0.5,L), np.imag(curve), color=color, linewidth=0.7)
    ax.set_title("bonjour")
    ax.set_ylabel("Progression")
    ax.set_xlabel("Partie Réelle")
    ax.set_zlabel("Partie Imaginaire")




if __name__ == "__main__":
    
    ########## B/A ###########
    # num=10
    # range_sigma = np.linspace(0.05, 0.5, num)
    # eigs = np.zeros(num)
    # for i, sigma in enumerate(range_sigma):
    #     print("i=",i)
    #     window = gaussian(sigma)
    #     d_window = discretize_window(window)
    #     S = construct_operator_matrix(d_window, alpha=alpha, beta=beta)
        
    #     eigenvalues, eigenvectors = np.linalg.eig(S)
        
    #     eigenvalues = np.real(eigenvalues)
    #     # print(np.sort(eigenvalues))
        
    #     diagonal_matrix = np.diag(1/np.sqrt(eigenvalues))
    #     Smdemi = eigenvectors @ diagonal_matrix @ np.linalg.inv(eigenvectors)
        
    #     d_window_mdemi = Smdemi @ d_window
        
    #     Stight = construct_operator_matrix(d_window_mdemi, alpha=alpha, beta=beta)
    #     eigenvalues, eigenvectors = np.linalg.eig(Stight)
        
    #     # eigenvalues, eigenvectors = np.linalg.eig(S)
        
    #     eigenvalues = np.real(eigenvalues)
    #     eigs[i] = np.max(eigenvalues)/np.min(eigenvalues)
    # plt.plot(range_sigma, eigs)
    # plt.grid()
    # plt.show()
    ########## B/A ###########
    
    
    
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    
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
    ax2.set_title("S_g sous représentation de Walnut")
    plt.colorbar(im, ax=ax2)
    
        

    S_inv = np.linalg.inv(S)
    dual_window = np.matmul(S_inv, d_window)
    plot_window(dual_window, ax=axes[1], label="Fenêtre duale canonique (partie réelle/imag en bleu/rouge resp.)")
    
        
    # eigenvalues, eigenvectors = np.linalg.eig(S)
    # eigenvalues = np.sort(eigenvalues)
    # print(np.real(eigenvalues))
    
    
    ## eigenvectors
    # fig, axes3 = plt.subplots(num, 1, figsize=(14, 10))
    # for i, vec in enumerate(eigenvectors):
    #     # if i>=num:
    #     #     break
    #     plot_window(vec, ax=axes3[i], label="")
    #     # plot_window(vec, ax=axes3[i], label=eigenvalues[i])
    
    
    S_tight, d_window_mdemi = compute_tight_frame(d_window=d_window, alpha=alpha, beta=beta, S=S)
    plot_window(d_window_mdemi, ax=axes[2], label="Fenêtre serrée canonique")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(np.abs(S_tight), cmap='viridis', aspect='auto')
    ax2.set_title("S_g^{-1/2} sous représentation de Walnut")
    plt.colorbar(im, ax=ax2)
    
    
    # ax3d = fig.add_subplot(111, projection='3d')
    # plot_complex_curve(d_window, ax=ax3d, fig=fig, color='blue')
    # plot_complex_curve(d_window_mdemi, ax=ax3d, fig=fig, color='red')
    # plot_complex_curve(dual_window, ax=ax3d, fig=fig, color='green')
    
    
    
    print("||g||*||gamma|| =", np.sqrt(np.sum(abs(dual_window)**2) * np.sum(abs(d_window)**2)), ">=", (alpha*beta/L), "= (alpha*beta)/L")
    
    plt.show()




