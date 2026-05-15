import numpy as np



def ind_zero(length: float): ## indicatrice normalisée centrée en zéro (sur l'ouvert de largeur donnée)
    assert length > 0
    
    def ind(t_):
        if type(t_) is np.ndarray:
            t = t_.copy()
            for i in range(len(t)):
                t[i] = 1/np.sqrt(2*length) if abs(t[i]) <= length else 0
            return t
        else:
            return 1/np.sqrt(2*length) if abs(t_) <= length else 0
    return ind

def gaussian(sigma: float):
    assert sigma > 0
    # return lambda t: np.exp(-np.pi*(t/sigma)**2) * (2**0.25 * sigma)
    return lambda t: np.exp(-np.pi*(t/sigma)**2)
    # return lambda t: np.exp(-np.pi*(t/sigma)**2) * (2 ** 0.25 / (sigma ** 0.5))
    # return lambda t: np.exp(-np.pi*(t/sigma)**2)

# def gaussian_comp_supp(sigma: float):
#     assert sigma > 0
    
#     def fonction(t_):
#         if type(t_) is np.ndarray:
#             t = t_.copy()
#             for i in range(len(t)):
#                 t[i] = np.exp(-1/(sigma/2-abs(t[i]))) if abs(t[i]) < sigma else 0
#             return t
#         else:
#             return np.exp(-1/(sigma/2-abs(t_))) if abs(t_) < sigma else 0
#     return fonction

def blackman_window(l:float = 1.0):
    def fonction(t_):
        if type(t_) is np.ndarray:
            t = t_.copy()
            for i in range(len(t)):
                t[i] = 0.42 + 0.5 * np.cos(1 * np.pi * t[i] / l) + 0.08 * np.cos(2 * np.pi * t[i] / l) if abs(t[i]) <= l else 0
            return t
        else:
            return 0.42 + 0.5 * np.cos(1 * np.pi * t_ / l) + 0.08 * np.cos(2 * np.pi * t_ / l) if abs(t_) <= l else 0
    return fonction

# def test_window(sigma: float):
#     def function(t_):
#         if type(t_) is np.ndarray:
#             t = t_.copy()
#             for i in range(len(t)):
#                 temp = t[i]+0.5
#                 if temp >= 0.5:
#                     temp -= 1
#                 t[i] = 1 * np.exp(-np.pi*(t[i]/sigma)**2) / sigma + 1j * np.exp(-np.pi*(temp/sigma)**2) / sigma
#             return t
#         else:
#             temp = t_+0.5
#             if temp >= 0.5:
#                 temp -= 1
#             return 1 * np.exp(-np.pi*(t_/sigma)**2) / sigma + 1j * np.exp(-np.pi*(temp/sigma)**2) / sigma
#     return function

def test_window(sigma: float):
    # return lambda t: (np.exp(-np.pi*(t/sigma)**2) / sigma) * np.exp(2j * np.pi*(0.5*t/sigma)) ## gaussienne "tournante"
    return lambda t: (np.exp(-np.pi*(t/sigma)**2) / sigma) * np.sin(2 * np.pi * t * 2) ** 2
