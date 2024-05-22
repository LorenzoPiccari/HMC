import numpy as np


def bursts(t, Nsources, theta):
    signal = 0
    for i in range(Nsources):
        signal += theta[0+i*3]*np.sin(t*theta[1+i*3] + theta[2+i*3])

    return signal

def generate_gpParam(Nsources, bounds, rng):
    
    theta = []

    for i in range(Nsources):
        theta.append(rng.uniform(bounds[0][0], bounds[0][1]))
        theta.append(rng.uniform(bounds[1][0], bounds[1][1]))
        theta.append(rng.uniform(bounds[2][0], bounds[2][1]))
    return np.array(theta)
  
    
  
    
def generate_data(bounds, time, Nsources, sampling_frequency=16, sigma_noise=1, rng=np.random):

    theta = generate_gpParam(Nsources, bounds, rng)

    
    print("\n Real parameters: \n",)
    for i in range( Nsources):
        print(theta[i*3:(i+1)*3], "\n")

    npoints = int(sampling_frequency*(time[1] - time[0]))

    t = np.array([i/sampling_frequency for i in range(npoints)]) # questo dovrebbe generarlo con la frequenza di sampling corretta invece che leggermente alterata, prova a controllore

    signal = bursts(t, Nsources, theta)

    noise = rng.normal(0., sigma_noise, npoints)
    
    return t, signal, theta, noise
