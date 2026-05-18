from config import*
from tools import*


def plot_window(d_window, ax, is_discrete=True, label="", custom_y_lim=0.0, color=None):
    # if is_discrete:
    #     window = window_.copy()
    #     window[:L//2] = window_[L//2:]
    #     window[L//2:] = window_[:L//2]
    # else:
    #     window = window_
    window = d_window.copy()
    window[:L//2] = d_window[L//2:]
    window[L//2:] = d_window[:L//2]
    
    if color:
        ax.plot(np.linspace(-duration/2,duration/2,L), np.real(window), color=color, alpha=0.7, linewidth=1.0)
    else:
        ax.plot(np.linspace(-duration/2,duration/2,L), np.real(window), color='blue', alpha=0.7, linewidth=1.0)
    ax.plot(np.linspace(-duration/2,duration/2,L), np.imag(window), color='red', alpha=0.7, linewidth=1.0)
    # ax.set_xlabel("Progression")
    # ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    
    ax.margins(0, x=None, y=None, tight=True)
    
    # axes[ax_index].plot(np.linspace(-0.5,0.5,L), discretize_window(window, True))
    ax.set_title(label)
    if custom_y_lim:
        ax.set_ylim(-custom_y_lim, custom_y_lim)

if __name__ == "__main__":
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10)) ## changer 1er argument accordement
    
    plot_window((d_window), ax=axes[0])
    # plot_window(discretize_window(gaussian(0.1)), ax=axes[0])
    # plot_window(discretize_window(blackman_window(0.1)), ax=axes[1])
    
    
    
    plt.show()