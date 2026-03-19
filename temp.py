import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma=1.0):
    return np.exp(-0.5 * ((x - mu)/sigma)**2)

def generate_zero_moments_function(M, x_grid, sigma=1.0, centers=None):
    """
    Génère une fonction combinaison linéaire de gaussiennes
    dont les M premiers moments sont nuls.
    """
    if centers is None:
        centers = np.linspace(-2, 2, M)  # positions des gaussiennes
    
    # Matrice des moments : A[p, j] = ∫ t^p * gaussienne_j(t) dt (approx)
    A = np.zeros((M, M))
    dt = x_grid[1] - x_grid[0]
    for p in range(M):
        for j, mu in enumerate(centers):
            g = gaussian(x_grid, mu, sigma)
            moment = np.sum(x_grid**p * g) * dt
            A[p, j] = moment
    
    # On veut que la somme des coefficients soit nulle pour chaque moment
    # Donc on cherche c tel que A @ c = 0
    # Une solution simple : prendre c dans le noyau de A
    U, s, Vt = np.linalg.svd(A)
    c = Vt[-1, :]  # dernier vecteur singulier (associé à la plus petite valeur singulière)
    
    # Construire la fonction
    f = np.zeros_like(x_grid)
    for j, mu in enumerate(centers):
        f += c[j] * gaussian(x_grid, mu, sigma)
    
    return f

if __name__ == "__main__":
    # Exemple
    x = np.linspace(-5, 5, 1000)
    f = generate_zero_moments_function(10, x)

    plt.plot(x, f)
    plt.title("Fonction avec 4 premiers moments nuls")
    plt.grid()
    plt.show()

    # Vérification des moments
    dt = x[1] - x[0]
    for p in range(8):
        moment = np.sum(x**p * f) * dt
        print(f"Moment {p} : {moment:.2e}")