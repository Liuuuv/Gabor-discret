from config import*
from tools import*

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

if __name__ == "__main__":
    
    fig, axes = plt.subplots(6, 1, figsize=(14, 10)) ## changer 1er argument accordement
    
    plot_wi