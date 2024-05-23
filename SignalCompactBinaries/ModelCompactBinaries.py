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
        
        
        if bounds is None: bounds = [(2.0,10.), (.75,4.), (0, 2*np.pi)]*Nsources
        
        self.npoints = int(sampling_frequency*(time[1] - time[0]))
        self.t =  np.array([i/sampling_frequency for i in range(self.npoints)])
        self.sigma_noise = sigma_noise
        super().__init__(3*Nsources ,bounds)
        
        
        
    def distribution(self, q):
        
        if self.in_bounds(q) == False:
            return np.inf
        
        signal = bursts(self.t, self.Nsources, q)
        res = self.sample - signal
        return res @ res *.5/(self.sigma_noise**2) 
        
        
    def gradient(self, q):
        signal = bursts(self.t, self.Nsources, q)
        
        res = self.sample - signal
        const = res/(self.sigma_noise**2)
        
        return -gradient_bursts(q, self.t, self.Nsources, const)
    
    
    def reflection(self, q, p):
        '''diff = np.diff(q[1::3])
        refl = []
        
        for i in range(len(diff)):
            if diff[i] < .005:
                refl.append((1 + i*3, 1 + (i+1)*3))
                
        for f in refl:
            p[f[0]] = -p[f[0]]
            p[f[1]] = -p[f[1]]
            '''
        for i in range(self.Nsources):
            if q[3 * i + 2] < 0:
                q[3 * i + 2] += 2*np.pi
            if q[3 * i + 2] > 2*np.pi:
                q[3 * i + 2] -= 2*np.pi
                
        return q, p