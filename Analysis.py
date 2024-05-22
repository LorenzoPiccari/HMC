import numpy as np
import corner
import matplotlib.pyplot as plt


def corner_plot(matrix, true = None):
    if true is None:
        corner.corner(matrix)
        plt.show()
    else:
        value1 = true
        
        # This is the empirical mean of the sample:
        value2 = np.mean(matrix, axis=0)
        
        # Make the base corner plot
        figure = corner.corner(matrix)
        
        corner.overplot_lines(figure, value1, color="C1")
        corner.overplot_points(figure, value1[None], marker="s", color="C1")
        corner.overplot_lines(figure, value2, color="C2")
        corner.overplot_points(figure, value2[None], marker="s", color="C2")
        plt.show()
    

def mean_sample(matrix):
    return np.array([np.mean(array) for array in matrix.T])


def var_sample(matrix, mean):
    
    N = len(matrix[:,0])
    return np.array([C_K(array,0,m,N) for array, m in zip(matrix.T, mean)])


def var_mean_real(matrix, mean):
    
    N = len(matrix[:,0])
    iat = IAT(matrix)
    return np.array([var_real(array, iat, m, N) for array, m in zip(matrix.T, mean)]), iat

def var_mean(matrix, mean):
    
    N = len(matrix[:,0])
    return var_sample(matrix, mean)/(N-1)

    

def var_real(array, iat, mean, N):
    
    return C_K(array, 0, mean, N)*(1+2*iat)/N


def C_K(array, K, mean, N):
    
    c = 0
    for i in range(N-K):
        c += (array[i] - mean)*(array[i+K] - mean)

    return c/(N-K)


def auto_c(array, mean, N, lag):
    M = int(N/lag+1)
    corr = np.zeros(M)
    for K in range(M):
        corr[K] = C_K(array, K, mean, N)
    
    return corr


def tau(array, mean, N, lag):
    s = auto_c(array, mean, N, lag)
    return np.sum(s[1:])/s[0]


def IAT(matrix, lag = 100):
    N = len(matrix[:,0])
    max_t = 0
    mean = mean_sample(matrix)
    for m, array in zip(mean, matrix.T):
        t = tau(array, m, N, lag)
        if t > max_t: max_t = t
    return max_t


