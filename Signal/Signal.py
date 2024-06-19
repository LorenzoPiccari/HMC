import numpy as np
from Model.Model import Model
def bursts_bh(t, Nsources, q):
    signal = 0
    for i in range(Nsources):
        signal += q[0+i*5]*np.exp(-((t - q[1+i*5])/q[2+i*5])**2)*np.sin(t*q[3+i*5] + q[4+i*5])

    return signal


def gradient_bursts_bh(q,t, dim, const):
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

def bursts_cb(t, Nsources, q):
    signal = 0
    for i in range(Nsources):
        signal += q[0+i*3]*np.sin(t*q[1+i*3] + q[2+i*3])

    return signal


def gradient_bursts_cb(q,t, dim, const):
    grad = np.zeros(len(q))
    for i in range(dim):
        
        grad[0+i*3] = np.sin(t*q[1+i*3] + q[2+i*3]) @ const
        grad[1+i*3] = const @ (t*np.cos(t*q[1+i*3] + q[2+i*3]))
        grad[2+i*3] = const @ (np.cos(t*q[1+i*3] + q[2+i*3]))
        
    return grad


class Signal(Model):
    
    def __init__(self, sample, time,  Nsources_bh, Nsources_cb, sampling_frequency=16, sigma_noise=0.8,bounds_bh = None, bounds_cb = None):
        
        
        self.sample = sample
        
        self.Nsources_bh = Nsources_bh
        self.Nsources_cb = Nsources_cb
        
        
        if bounds_cb is None: bounds_cb = [(2.0,10.), (.75,4.), (0, 2*np.pi)]*Nsources_cb
        if bounds_bh is None: bounds_bh = [(2. ,7.), (0.,15.), (.1, 1.), (.75, 4.), (0, 2*np.pi)]*Nsources_bh
        
        
        self.npoints = int(sampling_frequency*(time[1] - time[0]))
        self.t =  np.array([i/sampling_frequency for i in range(self.npoints)])
        self.sigma_noise = sigma_noise 
        super().__init__(5*Nsources_bh + 3*Nsources_cb ,bounds_cb + bounds_bh)
        

        
    def distribution(self, q):
        
        if self.in_bounds(q) == False:
            return np.inf
        
        signal_bh = bursts_bh(self.t, self.Nsources_bh, q)
        signal_cb = bursts_cb(self.t, self.Nsources_cb, q)
        
        res = self.sample - signal_bh - signal_cb
        return (res @ res *.5/(self.sigma_noise**2)) / self.npoints
        
    def gradient(self, q):
        grad = np.zeros(len(q))
        signal_bh = bursts_bh(self.t, self.Nsources_bh, q)
        signal_cb = bursts_cb(self.t, self.Nsources_cb, q)
        
        res = self.sample - signal_bh - signal_cb
        const = res/(self.sigma_noise**2)
        
        grad[:self.Nsources_cb*3] = -gradient_bursts_cb(q[:self.Nsources_cb*3], self.t, self.Nsources_cb, const) / self.npoints
        grad[self.Nsources_cb*3:] = -gradient_bursts_bh(q[self.Nsources_cb*3:], self.t, self.Nsources_bh, const) / self.npoints
        return grad
    
    
    def reflection(self, q, p):
        
        diff = np.diff(q[:self.Nsources_cb*3][1::3])
        refl = []
        
        for i in range(len(diff)):
            if diff[i] < .005:
                refl.append((1 + i*3, 1 + (i+1)*3))
                
        for f in refl:
            p[f[0]] = -p[f[0]]
            p[f[1]] = -p[f[1]]
        
        diff = np.diff(q[self.Nsources_cb*3:][1::5])
        refl = []
        
        for i in range(len(diff)):
            if diff[i] < .005:
                refl.append((1 + i*5, 1 + (i+1)*5))
                
        for f in refl:
            p[f[0]] = -p[f[0]]
            p[f[1]] = -p[f[1]]    
        
        
        for i in range(self.Nsources_cb):
            if q[3 * i + 2] < 0:
                q[3 * i + 2] += 2*np.pi
            if q[3 * i + 2] > 2*np.pi:
                q[3 * i + 2] -= 2*np.pi
                
        for i in range(self.Nsources_bh):
            if q[self.Nsources_cb*3 + 5 * i + 2] < 0:
                q[self.Nsources_cb*3 + 5 * i + 2] += 2*np.pi
            if q[self.Nsources_cb*3 + 5 * i + 2] > 2*np.pi:
                q[self.Nsources_cb*3 + 5 * i + 2] -= 2*np.pi
                
        return q, p