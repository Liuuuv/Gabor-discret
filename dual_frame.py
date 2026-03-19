import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from tools import*
from config import*

## matrice circulante: [A_i] où A_i: np.ndarray




def construct_operator_matrix(window, alt_window=None, alpha=alpha, beta=beta):
    beta_t = L//beta
    alpha_t = L//alpha
    
    if alt_window is None:
        alt_window = window
    
    matrix = np.ones((L, L), dtype=np.complex64)
    for j in range(L):
        if j % beta_t == 0:
            for d in range(L):
                n = np.arange(alpha_t)
                matrix[int(j+d) % L, int(d) % L] = float(beta_t) * sum(
                    np.conjugate(window[int(j+d)%L - alpha * n]) * alt_window[int(d)%L - alpha * n]
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
            

def construct_circulant(matrix): ## pour S walnut
    circ = []
    for s in range(alpha_t):
        # circ.append(matrix[s * alpha:(s+1) * alpha, s * alpha:(s+1) * alpha])
        circ.append(matrix[0:alpha, s * alpha:(s+1) * alpha])

    return np.array(circ)

def circlulant_ft(C: np.ndarray):
    C_fourier = []
    N = len(C)
    for r in range(N):
        matrix = np.zeros_like(C[r])
        for s in range(N):
            matrix += np.exp((-1j * 2 * np.pi / N) * s * r) * C[s]
        C_fourier.append(matrix)
    return np.array(C_fourier)


def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e



def compute_dual_window(window, alpha=alpha, beta=beta):
    print("Calcul de la fenêtre duale")
    
    S = construct_operator_matrix(d_window, alpha=alpha,beta=beta)
    # eigenvalues, eigenvectors = np.linalg.eig(S)
    # eigenvalues = np.real(eigenvalues)
    # print(f"valeurs propres de S: max: {np.max(eigenvalues)}, min: {np.min(eigenvalues)}")
    # print(f"conditionnement de S:", np.linalg.cond(S))
    S_inv = np.linalg.inv(S)
    # d_window = window(np.arange(L))
    
    dual_window = np.matmul(S_inv, d_window)
    return dual_window

def compute_tight_frame(d_window, alpha=alpha, beta=beta, S=None):
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
    
    # eigenvalues, eigenvectors = np.linalg.eig(S_tight)
    # eigenvalues = np.real(eigenvalues)
    # print(np.sort(eigenvalues))
    
    return S_tight, d_window_mdemi

## cas alpha = 1
# sigma = 50
# window = gaussian(sigma)

# window = lambda t: window_(t - L/2)





def plot_window(window_, ax, is_discrete=True, label="", custom_y_lim=0.0):
    # if is_discrete:
    #     window = window_.copy()
    #     window[:L//2] = window_[L//2:]
    #     window[L//2:] = window_[:L//2]
    # else:
    #     window = window_
    window = window_.copy()
    window[:L//2] = window_[L//2:]
    window[L//2:] = window_[:L//2]
    
    
    ax.plot(np.linspace(-0.5,0.5,L), np.real(window), color='blue', alpha=0.7, linewidth=1.0)
    ax.plot(np.linspace(-0.5,0.5,L), np.imag(window), color='red', alpha=0.7, linewidth=1.0)
    # ax.set_xlabel("Progression")
    # ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    
    ax.margins(0, x=None, y=None, tight=True)
    
    # axes[ax_index].plot(np.linspace(-0.5,0.5,L), discretize_window(window, True))
    ax.set_title(label)
    if custom_y_lim:
        ax.set_ylim(-custom_y_lim, custom_y_lim)


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
    
    ########## CONDITIONNEMENT ##########
    # num = 40
    # cond = np.zeros(num)
    # sigmas = np.linspace(0.02, 0.4, num)
    # for i, sigma in enumerate(sigmas):
    #     print("i=", i)
        
    #     window = gaussian(sigma)
    #     d_window_ = discretize_window(window)
    #     S = construct_operator_matrix(d_window_, alpha, beta)
    #     # eigenvalues, eigenvectors = np.linalg.eig(S)
    #     # eigenvalues = np.real(eigenvalues)
    #     # print(f"valeurs propres de S: max: {np.max(eigenvalues)}, min: {np.min(eigenvalues)}")
    #     # print(f"conditionnement de S:", np.linalg.cond(S))
    #     cond[i] = np.linalg.cond(S)
    
    # plt.plot(sigmas, cond)
    # plt.yscale("log")
    # plt.grid()
    # plt.show()
    ########## CONDITIONNEMENT ##########
    
    
    
    
    
    
    
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    
    ##################### CUSTOM VARIABLES BEGIN #####################
    ## if not declared here, uses the config.py values
    # alpha: int = 2
    # beta: int = 250
    
    # sigma = 0.1
    # window = gaussian(sigma)
    # # window = lambda t: window_(t) * np.sin(2 * np.pi * 100 * t)2
    # # window = lambda t: window_(t - L//2)
    # d_window = discretize_window(window)
    ##################### CUSTOM VARIABLES END #####################
    
    plot_window(d_window, ax=axes[0], label="Fenêtre (partie réelle/imag en bleu/rouge resp.)")
    
        
    S = construct_operator_matrix(d_window, alpha=alpha, beta=beta)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(np.abs(S), cmap='viridis', aspect='auto')
    ax2.set_title("S_g sous représentation de Walnut")
    plt.colorbar(im, ax=ax2)
    
    
    
    Scirc = construct_circulant(S)
    print("size S", np.shape(Scirc))
    S_ft = circlulant_ft(Scirc)
    print("size", np.shape(S_ft))
    for i in range(len(S_ft)):
        for j in range(len(S_ft[i])):
            if S_ft[i][j,j] == 0.0:
                print(f"PAS INVERSIBLE AAAAAAAAAA {i}-ème matrice, coef {j,j}")
                break
    
    
    fig2, axes2 = plt.subplots(len(Scirc)//2+1, 1, figsize=(14, 8))
    axes2[0].set_title("Fourier")
    for i in range(len(S_ft)//2+1):
        im = axes2[i].imshow(np.abs(S_ft[i]), cmap='viridis', aspect='auto')
        
        plt.colorbar(im, ax=axes2[i])
    
    
        

    S_inv = np.linalg.inv(S)
    dual_window = np.matmul(S_inv, d_window)
    plot_window(dual_window, ax=axes[1], label="Fenêtre duale canonique (partie réelle/imag en bleu/rouge resp.)")
    
        
    # # eigenvalues, eigenvectors = np.linalg.eig(S)
    # # eigenvalues = np.sort(eigenvalues)
    # # print(np.real(eigenvalues))
    
    
    # ## eigenvectors
    # # fig, axes3 = plt.subplots(num, 1, figsize=(14, 10))
    # # for i, vec in enumerate(eigenvectors):
    # #     # if i>=num:
    # #     #     break
    # #     plot_window(vec, ax=axes3[i], label="")
    # #     # plot_window(vec, ax=axes3[i], label=eigenvalues[i])
    
    
    S_tight, d_window_mdemi = compute_tight_frame(d_window=d_window, alpha=alpha, beta=beta, S=S)
    plot_window(d_window_mdemi, ax=axes[2], label="Fenêtre serrée canonique")
    # fig2, ax2 = plt.subplots(figsize=(8, 6))
    # im = ax2.imshow(np.abs(S_tight), cmap='viridis', aspect='auto')
    # ax2.set_title("S_g^{-1/2} sous représentation de Walnut")
    # plt.colorbar(im, ax=ax2)
    
    
    # # ax3d = fig.add_subplot(111, projection='3d')
    # # plot_complex_curve(d_window, ax=ax3d, fig=fig, color='blue')
    # # plot_complex_curve(d_window_mdemi, ax=ax3d, fig=fig, color='red')
    # # plot_complex_curve(dual_window, ax=ax3d, fig=fig, color='green')
    
    
    
    # print("||g||*||gamma|| =", np.sqrt(np.sum(abs(dual_window)**2) * np.sum(abs(d_window)**2)), ">=", (alpha*beta/L), "= (alpha*beta)/L")
    
    plt.show()




