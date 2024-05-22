import numpy as np
from Model.Model import Model
def bursts(t, Nsources, q):
    signal = 0
    for i in range(Nsources):
        signal += q[0+i*3]*np.sin(t*q[1+i*3] + q[2+i*3])

    return signal


def gradient_bursts(q,t, dim, const):
    grad = np.zeros(len(q))
    for i in range(dim):
        
        grad[0+i*3] = np.sin(t*q[1+i*3] + q[2+i*3]) @ const
        grad[1+i*3] = const @ (t*np.cos(t*q[1+i*3] + q[2+i*3]))
        grad[2+i*3] = const @ (np.cos(t*q[1+i*3] + q[2+i*3]))
        
    return grad

class CompactBinieriesSignal(Model):
    
    def __init__(self, sample, time,  Nsources, sampling_frequency=16, sigma_noise=1, bounds = None):
        
        
        self.sample = sample
        
        self.Nsources = Nsources
        self.sigma_noise = sigma_noise
        
        
        if bounds is None: bounds = [(2.0,10.), (.75,4.), (0, 2*np.pi)]*Nsources
        
        npoints = int(sampling_frequency*(time[1] - time[0]))
        self.t =  np.array([i/sampling_frequency for i in range(npoints)])
        
        super().__init__(3*Nsources ,bounds)
        
        
        
    def log_likelihood(self, q):
        
        if self.in_bounds(q) == False:
            return np.inf
        
        signal = bursts(self.t, self.Nsources, q)
        res = self.sample - signal
        return res @ res *.5/(self.sigma_noise**2)
        
        
    def gradient_log_likelihood(self, q):
        signal = bursts(self.t, self.Nsources, q)
        
        res = self.sample - signal
        const = res/(self.sigma_noise**2)
        
        return -gradient_bursts(q, self.t, self.Nsources, const)