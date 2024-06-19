from LogisticRegression import Logistic_Regression
import matplotlib.pyplot as plt
import numpy as np
import Analysis as sys
from Algorithms.metropolis import metropolis
from Algorithms.MyNUTS import MyNUTS
from Algorithms.NUTS import NUTS
from Algorithms.HMC import HMC
from Algorithms.MALA import MALA
from Hamiltonian.H import H
from Hamiltonian.PB_H import PB_H
from Compare import compare
  
rng = np.random.default_rng(1234)

n1= 70
n2 = 70

dim = 2
mean1 = np.zeros(dim)
X1 = [rng.multivariate_normal(mean1, np.eye(dim)) for i in range(n1)]

mean2 = np.zeros(dim) + 1.5
X2 = [rng.multivariate_normal(mean2, np.eye(dim)) for i in range(n2)]

X_train = np.concatenate((X1, X2))
Y_train = np.zeros(n1+n2)
Y_train[50:] += 1

n1t = 30
n2t = 30
x1_test = [rng.multivariate_normal(mean1 , np.eye(dim)) for i in range(n1t)]
x2_test = [rng.multivariate_normal(mean2, np.eye(dim)) for i in range(n2t)]

X_test = np.concatenate((x1_test, x2_test))
Y_test = np.zeros(n1t+n2t)
Y_test[30:] +=1
M = Logistic_Regression(X_train, Y_train)
for x1 in X1:
    plt.scatter(x1[0], x1[1], color = 'blue')
for x2 in X2:
    plt.scatter(x2[0], x2[1], color = 'green')
plt.show()

start_q =np.zeros(dim+1)
H1 = H(np.eye(dim+1))
def mass_matrix(dim, rng):
    return rng.lognormal(0.1, 1, dim)
H2 = PB_H(mass_matrix)


nuts =  NUTS(M, H1, dt = 0.001 , rng=rng)
mynuts = MyNUTS(M, H1, dt = 0.01, rng=rng)
pb = MyNUTS(M, H2, dt = 0.005, rng=rng)
metropolis = metropolis(M, .01, rng = rng)
mala = MALA(M, dt = 0.0001, rng=rng)
hmc = HMC(M, H1, L = 10, dt = 0.001, rng = rng)
algs = [metropolis, mala, hmc, nuts, mynuts, pb]
iteration = 10000
all_m = [np.load(a.alg_name()+".npy") for a in algs]
#all_m = compare(algs, iteration, start_q, 100)
'''
all_m = []
for a in algs:
    all_m.append(np.load(a.alg_name()+".npy"))

for m in all_m:
    sys.plot_IAT(m, 10)
'''
for a, m in zip(algs, all_m):
    M.plot_logit2(m, X_test, Y_test, a.alg_name())
plt.show()
for a, m in zip(algs, all_m):
    M.confusion_matrix(X_test, Y_test, m)
    plt.show()
for a, m in zip(algs, all_m):
    plt.grid()
    plt.plot(np.linspace(0, 1,100),np.linspace(0, 1,100), linestyle='dashed')
    M.ROC_curve(X_test, Y_test, m, a.alg_name())
plt.show()

for a, m in zip(algs, all_m):
    print(a.alg_name(), M.accuracy(X_test, Y_test, m))

inf= np.load("info.npy")

print(inf)


