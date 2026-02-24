import matplotlib.pyplot as plt
import numpy as np


## {[t_start, t_end]: frequency} | "frequency" is a func defined on [0,1] giving the frequency
test_frequencies = {
    (0, 1/4): lambda t: 20,
    (1/4, 3/4): lambda t: 20 + 40 * t,
    (3/10, 4/10): lambda t: 50,
    (4/10, 5/10): lambda t: 60,
    (4/10, 1): lambda t: 20,
    
#     (0, 1/4): lambda t: 20,
#     (1/4, 3/4): lambda t: 20 + 40 * (1 - np.exp(-5*t)),
#     (3/10, 4/10): lambda t: 50,
#     (4/10, 5/10): lambda t: 60,
#     (3/4, 1): lambda t: 60,
}


def define_signal(dict: dict, N=200): ## N: num points
    x = np.zeros(N) # values
    n = np.linspace(0, 1, N) # time
    scale = 1/1 # scale of time
    for times in dict.keys():
        mask = (n >= times[0]) & (n <= times[1])
        n_mask = n[mask]
        freq = dict[times]( (n_mask * (times[1] - times[0])  - times[0]) / (times[1] - times[0]))
        x[mask] += np.sin(2 * np.pi * freq * n_mask)
    return x

signal_test = define_signal(test_frequencies, 500)



def plot_time_frequencies_reference(signal=signal_test, frequency_dict=test_frequencies, ax = None):
    N = len(signal)
    n = np.linspace(0, 1, N)
    
    if not ax:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 3]})
        
        ## plot signal
        ax1.plot(n, signal, 'b-', linewidth=0.8)
        ax1.set_xlabel('Progression')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Signal test')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
    
    if ax:
        ax2 = ax
    
    ## plot frequencies
    for times in frequency_dict.keys():
        mask = (n >= times[0]) & (n <= times[1])
        
        ax2.plot(n[mask], [frequency_dict[times]((t - times[0]) / (times[1] - times[0])) for t in n[mask]], 'r--', linewidth=2, alpha=0.5)
    
    if not ax:
        ax2.set_xlabel('Progression')
        ax2.set_ylabel('Fréquence (Hz)')
        ax2.set_title('Plan temps-fréquence')
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3)
    



if __name__ == "__main__":
    
    plot_time_frequencies_reference(signal_test, test_frequencies)
    plt.get_current_fig_manager().window.state('zoomed')
    plt.tight_layout()
    plt.show()