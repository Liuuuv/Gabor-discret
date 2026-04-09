import numpy as np
import scipy.linalg as la


## construction "cloche" polynomiale par morceaux C^3
degree = 7
m = 1.0


def get_bell_coef(d: int, m: float):
    matrix = np.zeros((8, d + 1))
    
    matrix[0, 0] = 1 ## P(0) = 0
    matrix[1, 1:] = m ## P(1) = m
    for k in range(1, 4):
        matrix[1 + k, k] = 1 ## P^{(k)}(0) = 0
        matrix[4 + k, k:] = np.array([i + k for i in range(d + 1 - k)]) ## P^{(k)}(1) = 0
    
    print("matrix:")
    print(matrix)
    return matrix
    

def construct_bell(d: int, m: float):
    matrix = get_bell_coef(d, m)
    second_mem = np.zeros(8)
    second_mem[1] = m
    print('sol:')
    # print(np.linalg.inv(matrix))
    null_space = la.null_space(matrix)
    print(null_space)
    X = np.linalg.solve(matrix, second_mem)
    print(X)




# get_bell_coef(degree, m)
construct_bell(degree, m)












