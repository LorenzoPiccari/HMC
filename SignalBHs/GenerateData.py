import numpy as np


def bursts(t, Nsources, theta):
    signal = 0
    for i in range(Nsources):
        signal += theta[0+i*5]*np.exp(-((t - theta[1+i*5])/theta[2+i*5])**2)*np.sin(t*theta[3+i*5] + theta[4+i*5])

    return signal

def generate_gpParam(Nsources, bounds, rng):
    
    theta = []

    for i in range(Nsources):
        theta.append(rng.uniform(bounds[0][0], bounds[0][1]))
        theta.append(rng.uniform(bounds[1][0], bounds[1][1]))
        theta.append(rng.uniform(bounds[2][0], bounds[2][1]))
        theta.append(rng.uniform(bounds[3][0], bounds[3][1]))
        theta.append(rng.uniform(bounds[4][0], bounds[4][1]))
    return np.array(theta)
  
    
  
    
def generate_data(bounds, Nsources, sampling_frequency=16, sigma_noise=1, rng=np.random):

    theta = generate_gpParam(Nsources, bounds, rng)
    
    linspace_array = np.linspace(0, bounds[1][1], Nsources + 2)

    theta[1::5] = linspace_array[1:-1]
    
    print("\n Real parameters: \n",)
    for i in range( Nsources):
        print(theta[i*5:(i+1)*5], "\n")

    npoints = int(sampling_frequency*(bounds[1][1] - bounds[1][0]))

    t = np.array([i/sampling_frequency for i in range(npoints)]) # questo dovrebbe generarlo con la frequenza di sampling corretta invece che leggermente alterata, prova a controllore

    signal = bursts(t, Nsources, theta)

    noise = rng.normal(0., sigma_noise, npoints)
    
    return t, signal, theta, noise
