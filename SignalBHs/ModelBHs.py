import numpy as np
from Model.Model import Model
def bursts(t, Nsources, q):
    signal = 0
    for i in range(Nsources):
        signal += q[0+i*5]*np.exp(-((t - q[1+i*5])/q[2+i*5])**2)*np.sin(t*q[3+i*5] + q[4+i*5])

    return signal


def gradient_bursts(q,t, dim, const):
    grad = np.zeros(len(q))
    for i in range(dim):
        f = np.exp(-((t - q[1+i*5])/q[2+i*5])**2)
        f1 = q[0 + i*5]*np.exp(-((t - q[1+i*5])/q[2+i*5])**2)
        f2 = np.sin(t*q[3+i*5] + q[4+i*5])
        f3 = np.cos(t*q[3+i*5] + q[4+i*5])
        
        grad[0+i*5] = (f*f2) @ const
        grad[1+i*5] = const @ (2*f1*f2*(t - q[1+i*5])/(q[2 + i*5]**2))
        grad[2+i*5] = const @ (2*f1*f2*((t - q[1+i*5])**2)/(q[2 + i*5]**3))
        grad[3+i*5] = const @ (t*f1*f3)
        grad[4+i*5] = const @ (f1*f3)
        
    return grad

class BHsSignal(Model):
    
    def __init__(self, sample, Nsources, sampling_frequency=16, sigma_noise=1, bounds = None):
        
        
        self.sample = sample
        
        self.Nsources = Nsources
        self.sigma_noise = sigma_noise
        
        if bounds is None: bounds = [(2.0,10.), (0.1,14.), (.1,1.), (.75,4.), (0, 2*np.pi)]*Nsources
        
        self.npoints = int(sampling_frequency*(bounds[1][1] - bounds[1][0]))
        self.t =  np.array([i/sampling_frequency for i in range(self.npoints)])
        
        super().__init__(5*Nsources ,bounds)
        
        
        
    def distribution(self, q):
        
        if self.in_bounds(q) == False:
            return np.inf

        signal = bursts(self.t, self.Nsources, q)
        res = self.sample - signal
        
        return (res @ res *.5/(self.sigma_noise**2))
        
        
    def gradient(self, q):
        signal = bursts(self.t, self.Nsources, q)
        
        res = self.sample - signal
        const = res/(self.sigma_noise**2)
        
        return (-gradient_bursts(q, self.t, self.Nsources, const))
    
    def reflection(self, q, p):
        diff = np.diff(q[1::5])
        refl = []
        
        for i in range(len(diff)):
            if diff[i] < .05:
                refl.append((1 + i*5, 1 + (i+1)*5))
                
        for f in refl:
            p[f[0]] = -p[f[0]]
            p[f[1]] = -p[f[1]]
            
        for i in range(self.Nsources):
            if q[5 * i + 4] < 0:
                q[5 * i + 4] += 2*np.pi
            if q[5 * i + 4] > 2*np.pi:
                q[5 * i + 4] -= 2*np.pi
                
            if q[5* i + 2] > self.bounds[5* i + 2][ 1]:
                p[5*i +2] = -p[5*i +2]
        return q, p
    
    
    