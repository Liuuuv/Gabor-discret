from config import*
from tools import*



def compute_variance(f: np.ndarray, offset:float=0.0) -> float:
    return np.sum([(np.abs(L_sampling[i]/L - offset) ** 2) * (np.abs(f[i]) ** 2) for i in range(L)])

    

if __name__ == "__main__":
    # fig, axes = plt.subplots(2, 1, figsize=(14, 10)) ## changer 1er argument accordement
    fig, axes = plt.subplots(1, 1, figsize=(14, 10)) ## changer 1er argument accordement
    
    
    
    # sigmas = np.linspace(0.04472135954999579392818347337463, 0.04472135954999579392818347337463, num_sigmas)
    ## sigma = 1/L**2 atteint le min en norme sur le y = 1/x (cas c=1)
    
    points = []
    
    
    num_sigmas: int = 100
    sigmas = np.linspace(0.08, 0.6, num_sigmas)
    for sigma in sigmas:
        f = discretize_window(gaussian(sigma))
        f_fourier = fft(f)
        
        norm_sq = np.sum(np.abs(f) ** 2)
        # f /= norm_sq
        
        points.append((compute_variance(f) / norm_sq, compute_variance(f_fourier) / norm_sq))
    
    num_l: int = 100
    ls = np.linspace(0.08, 0.6, num_l)
    for l in ls:
        f = discretize_window(blackman_window(l))
        f_fourier = fft(f)
        
        norm_sq = np.sum(np.abs(f) ** 2)
        # f /= norm_sq
        
        points.append((compute_variance(f) / norm_sq, compute_variance(f_fourier) / norm_sq))
    
    
    # spatial_offsets = np.linspace(-0.2, 0.2, 5)
    # fourier_offsets = np.linspace(-0.2, 0.2, 5)
    # sigma = 0.1
    
    # f = discretize_window(gaussian(sigma))
    # f_fourier = fft(f)
    
    
    # for spatial_offset in spatial_offsets:
    #     for fourier_offset in fourier_offsets:
    #         points.append((compute_variance(f, spatial_offset), compute_variance(f_fourier, fourier_offset)))
    
    points = np.array(points)
    
    x = points[:, 0]   # variances spatiales
    y = points[:, 1]   # variances fréquentielles
    
    x *= 4*np.pi
    y *= 4*np.pi

    # norm_sq = np.sum(np.abs(f) ** 2)
    # x /= norm_sq ** 2
    # y /= norm_sq ** 2
    
    # Création de la grille 2D pour pcolormesh
    X, Y = np.meshgrid(x, y)
    
    axes.plot(x, y, marker='.', color='k', linestyle='none', markersize=2)
    
    axes.grid(alpha=0.4)
    axes.set_aspect("equal")
    axes.set_xlabel("Variance spatiale")
    axes.set_ylabel("Variance fréquentielle")

    max_lim = max(np.max(x), np.max(y))
    max_lim = 0.15
    # max_lim = 25.0
    axes.set_xlim(0.0, max_lim)
    axes.set_ylim(0.0, max_lim)
    
    axes.plot(np.linspace(0,max_lim,100), np.linspace(0,max_lim,100), color='red')
    
    
    
    # plt.show()
    # fig, axes = plt.subplots(1, 1, figsize=(14, 10)) ## changer 1er argument accordement
    # f = discretize_window(blackman_window(0.1))
    # f_fourier = fft(f)
    # axes.plot(L_sampling/L, np.abs(f_fourier))
    # axes.grid(alpha=0.4)
    
    # finish the plot and save
    plt.get_current_fig_manager().window.state('zoomed')
    plt.tight_layout()

    plt.show()