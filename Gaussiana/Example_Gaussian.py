import numpy as np
from Model.Model import Model
import Analysis as sys
from Algorithms.metropolis import metropolis
from Algorithms.MyNUTS import MyNUTS
from Algorithms.NUTS import NUTS
from Algorithms.HMC import HMC
from Algorithms.MALA import MALA
from Hamiltonian.H import H
from Hamiltonian.PB_H import PB_H
from Compare import compare
from scipy.stats import chi2, norm
from scipy import stats

import matplotlib.pyplot as plt

class Multivariate(Model):
    
    def __init__(self, dim, mean = None, variance = None, bounds = None):
        
        if mean is None: self.mean = np.zeros(dim)
        else: self.mean = mean
        
        if variance is None: self.inv_cov = np.eye(dim)
        else: self.inv_cov = np.linalg.inv(variance)
        
        if bounds is None: bounds = [(self.mean[i]-5, self.mean[i]+5) for i in range(dim)]
        else: bounds = [(bounds[i]-5, bounds[i]+5) for i in range(dim)]
        
        super().__init__(dim, bounds)
        
    def distribution(self, q):
        return 0.5* (q - self.mean) @ (self.inv_cov @ (q - self.mean))


    def gradient(self,q):
        return (self.inv_cov @ (q-self.mean))
    

def LR_test(matrix, real_mean):
    p_value = np.zeros(len(real_mean))  
    i = 0
    for array, mean in zip(matrix.T, real_mean):
        N = len(array)
        esteem_mean = np.mean(array)
        lamda = np.sum(mean**2 - esteem_mean**2 - 2*array*(mean-esteem_mean))
        
        p_value[i] = chi2.sf(lamda, df=N-1)
        
        i+=1
    return p_value  



def chi_square_test(matrix, real_mean):
    chi_square_stat = np.zeros(len(real_mean))
    p_value = np.zeros(len(real_mean))
    i = 0
    for array, mean in zip(matrix.T, real_mean):
        N = len(array)
        num_bins = int(np.log2(N)) + 1
        
        counts, bin_edges = np.histogram(array, bins=num_bins)
        
        cdf_values = norm.cdf(bin_edges, loc=mean, scale=1)
        expected_counts = N * np.diff(cdf_values)
        chi_square_stat[i] = np.sum((counts - expected_counts) ** 2 / expected_counts)
        p_value[i] = chi2.sf(chi_square_stat[i], df=num_bins-1)
        
        i+=1
    return chi_square_stat, p_value

def KS_test(matrix, real_mean):
    ks = np.zeros(len(real_mean))
    p_value = np.zeros(len(real_mean))
    i = 0
    for array, m in zip(matrix.T, real_mean):
        
        res = stats.kstest(array-m,stats.norm.cdf, N = len(array))
        p_value[i] = res[1]
        ks[i] = res[0]
        i+=1

    
        
    return ks, p_value

def plot_CI(matrix, i, label):
    l1,l2,l3 = np.percentile(matrix, [5,50,95], axis = 0)
    lower_error =  l2 - l1.copy()
    upper_error =  l3.copy() - l2
    asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
    
    plt.errorbar(np.arange(np.shape(matrix)[1]) + i, l2, yerr=asymmetric_error, fmt='.' , label = label)
    plt.legend()
    plt.grid()

def CI(all_m, algs, real_mean):
    i= 0
    for m, a in zip(all_m, algs):
        plot_CI(m, -.2 +i, a.alg_name())
        i+= .4/np.shape(m)[1]
    i = 0
    for s in real_mean:
        
        plt.plot(np.linspace(-.4 + i, i + .4, 100), np.ones(100)*s, color = 'black')
        i+=1
    plt.legend()
    plt.show()    
    
def KS_test_a_mano(matrix, real_mean):
    
    for array, m in zip(matrix.T, real_mean):
        N = len(array)
        k = np.sort(array)
        max_diff = 0
        for i, e in enumerate(k):
            cum = (i+1)/N
            diff = np.abs(cum-stats.norm.cdf(e-m))
            if diff > max_diff:
                max_diff = diff
        print(max_diff)
        
def plt_KS(array, real_mean, name):
    N = len(array)
    k = np.sort(array)
    
    cum = np.zeros(N)
    for i, e in enumerate(k):
        
        cum[i] = np.sum(k<=e)/N
    
    plt.step(k, cum, label = name, alpha = 0.8)
    plt.legend()

rng = np.random.default_rng(1234)
dim = 5


mean = rng.uniform(0,10,dim)
M = Multivariate(dim, mean)

H1 = H(np.eye(dim))
def mass_matrix(dim, rng):
    return .001*np.ones(dim)
    
H2 = PB_H(mass_matrix,10)

start_q = np.ones(dim)

nuts =  NUTS(M, H1, dt = 0.01 , rng=rng)
mynuts = MyNUTS(M, H1, dt = 0.01, rng=rng)
pb = MyNUTS(M, H2, dt = 0.01, rng=rng)
metropolis = metropolis(M, .1, rng = rng)
mala = MALA(M, dt = 0.01, rng=rng)
hmc = HMC(M, H1, L = 40, dt = 0.01, rng = rng)

algs = [metropolis, mala, hmc, nuts, mynuts, pb]
iteration = 15000

#all_m = compare(algs, iteration, start_q, 5000, 100)

'''
m, rj = pb.run(iteration, start_q)
print(m)
np.save(pb.alg_name()+".npy", m[5000:, :])
'''

all_m = [np.load(a.alg_name()+".npy") for a in algs]
#for m in all_m:
#    sys.corner_plot(m)
    
CI(all_m, algs, mean)
for m, a in zip(all_m, algs):
    print(a.alg_name())
    print(KS_test(m, mean)[1])
    sum_p = 0
    for k in KS_test(m, mean)[1]:
        sum_p -= 2*np.log(k)
    print(chi2.sf(sum_p, df=2*dim))
    #print(chi_square_test(m, mean)[1])
    #print(LR_test(m, mean))
    
for i in range(dim):
    for m, a in zip(all_m, algs):
        plt_KS(m.T[i], mean[i], a.alg_name())
    plt.grid()
    x = np.linspace(mean[i] - 4, mean[i] + 4, 1000)

    # Compute the CDF values for these x values
    cdf = norm.cdf(x, mean[i], 1)
    
    # Plot the CDF
    plt.plot(x, cdf, label='Real cdf', color = 'black')
    plt.legend()
    plt.title("KS plot")
    plt.show()
   
print("\n real_mean: ", mean)

for m, a in zip(all_m, algs):
    mean_esteem = np.mean(m, axis = 0)
    var, iat = sys.var_mean_real(m, mean)
    print("\n ", a.alg_name() ,"\n")
    for e, v in zip(mean_esteem, var):
        print(e, v)
    


inf= np.load("info.npy")

print(inf)