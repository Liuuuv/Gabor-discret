from config import*
from tools import*



def compute_variance(f: np.ndarray, offset:float=0.0) -> float:
    return np.sum([(np.abs(L_sampling[i]/L - offset)**2) * (np.abs(f[i]) ** 2) for i in range(L)])

    

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(14, 10)) ## changer 1er argument accordement
    
    num_sigmas: int = 180
    sigmas = np.linspace(0.04472135954999579392818347337463, 0.04472135954999579392818347337463, num_sigmas)
    ## sigma = 1/L**2 atteint le min en norme sur le y = 1/x (cas c=1)
    
    
    points = []
    
    
    for sigma in sigmas:
        f = discretize_window(gaussian(sigma))
        f_fourier = fft(f)
        
        norm_sq = np.sum(np.abs(f) ** 2)
        f /= norm_sq
        
        points.append((compute_variance(f), compute_variance(f_fourier)))
    
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

    # norm_sq = np.sum(np.abs(f) ** 2)
    # x /= norm_sq ** 2
    # y /= norm_sq ** 2
    
    # Création de la grille 2D pour pcolormesh
    X, Y = np.meshgrid(x, y)
    
    axes[1].plot(x, y, marker='.', color='k', linestyle='none')
    axes[1].plot(np.linspace(0,0.1,100), np.linspace(0,0.1,100), color='red')
    axes[1].grid(alpha=0.4)
    # axes[1].set_aspect("equal")
    axes[1].set_xlabel("spatial")
    axes[1].set_ylabel("fréquence")

    max_lim = 0.009
    axes[1].set_xlim(0.0, max_lim)
    axes[1].set_ylim(0.0, max_lim)
    
    f = discretize_window(gaussian(0.05))
    f_fourier = fft(f)
    axes[0].plot(L_sampling/L, np.abs(f_fourier))
    axes[0].grid(alpha=0.4)
    
    ## finish the plot and save
    # plt.get_current_fig_manager().window.state('zoomed')
    # plt.tight_layout()

    plt.show()