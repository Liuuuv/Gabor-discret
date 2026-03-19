import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
from signal_test import signal_test, plot_time_frequencies_reference
from dual_frame import compute_dual_window, construct_operator_matrix, construct_circulant, circlulant_ft, pos_mod
from config import*






def analyze_eigenvalues(window, sigma):
    d_window = discretize_window(window)
    S = construct_operator_matrix(d_window, alpha=alpha, beta=beta)
    
    # Calcul des valeurs propres
    eigvals = np.linalg.eigvals(S)
    eigvals = np.sort(np.abs(eigvals))
    
    
    # diag = np.zeros(L)
    diag = []
    Scirc = construct_circulant(S)
    print("size S", np.shape(Scirc))
    S_ft = circlulant_ft(Scirc)
    # print(S_ft)
    print("size", np.shape(S_ft))
    for i in range(len(S_ft)):
        for j in range(len(S_ft[i])):
            diag.append(S_ft[i][j,j])
    diag = np.sort(np.abs(diag))
    
    
    
    
    
    
    
    
    # sums = []
    # for r in range(alpha_t):
    #     for j in range(alpha):
    #         midsum = 0
    #         for k in range(beta):
    #             n = np.arange(alpha_t)
    #             midsum += np.sum(np.exp(- (np.pi / (L**2 * sigma**2)) * ( (j - alpha * n + k * q)**2 + (j - alpha * n)**2 ))) * np.exp(-2j * np.pi * r * k * q / alpha_t)
    #             # midsum += np.sum(np.exp(- (np.pi / (L**2 * sigma**2)) * ( ((j - alpha * n + k * q)**2 + (j - alpha * n)**2) % L ))) * np.exp(-2j * np.pi * r * k * q / alpha_t)
            
    #         sums.append(beta_t * midsum)
    # sums = np.sort(np.abs(sums))
    
    sums = []
    for r in range(alpha_t):
        for j in range(alpha):
            midsum = 0
            for k in range(beta):
                n = np.arange(alpha_t)
                
                # Indices pour la fenêtre
                idx1 = (j - alpha * n + k * q) % L  # avec modulo pour conditions périodiques
                idx2 = (j - alpha * n) % L
                
                # Utilisation directe des valeurs de d_window
                window_product = d_window[idx1] * np.conj(d_window[idx2])
                
                # Somme sur n
                midsum += np.sum(window_product) * np.exp(-2j * np.pi * r * k * q / alpha_t)
            
            sums.append(beta_t * midsum)

    sums = np.sort(np.abs(sums))
    
    
    
    
    
    
    
    
    
    
    
    
    
    plt.figure()
    plt.semilogy(eigvals, 'o-', linewidth=0.5, markersize=1.0, label="Numériques")
    plt.semilogy(diag, 'o-', color='red', linewidth=0.5, markersize=0.6, label="Fourier")
    # plt.semilogy(sums, 'o-', color='green', linewidth=0.5, markersize=1.0, label="Gaussienne")
    plt.title(f"Valeurs propres de S (σ={sigma})")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    S = construct_operator_matrix(d_window)
    eigenvalues, eigenvectors = np.linalg.eig(S)
    eigenvalues = np.real(eigenvalues)
    eigenvalues= np.sort(eigenvalues)
    print(eigenvalues[-1]/eigenvalues[0])
    
    
    
    # # Projection de g sur les petites valeurs propres
    # U, s, Vh = np.linalg.svd(S)
    # # Décomposition en valeurs singulières
    # proj_g = np.abs(Vh @ d_window)
    # plt.figure()
    # plt.semilogy(proj_g, 'o-')
    # plt.title("Coefficients de g dans la base des vecteurs singuliers")
    # plt.grid()
    # plt.show()
    
    # return eigvals, proj_g

sigma_ = 0.3
window_ = gaussian(sigma_)
analyze_eigenvalues(window_, sigma_)


