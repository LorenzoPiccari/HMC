from Model.Model import Model
import numpy as np
import matplotlib.pyplot as plt

def likelihood_1_point(point, parameters, y):
    
    l_1 = y* (point @ parameters[:-1] + parameters[-1] - np.log(1 + np.exp(point @ parameters[:-1] + parameters[-1])) )
    
    l_0 = -(1 - y) * np.log(1 + np.exp(point @ parameters[:-1] + parameters[-1]))
    
    return (l_1 + l_0)


class Logistic_Regression(Model):
    
    def __init__(self, sample, y, bounds = None, sigma = 3  ):
        self.sigma = sigma
        self.y = y
        self.sample_dim, self.covariates_dim = np.shape(sample)
        self.sample = sample
        bounds = [(-5,5)]*(self.covariates_dim + 1)
        super().__init__(self.covariates_dim + 1 ,bounds)
        
            
    def distribution(self, q):
        l = 0
        prior = 0
        if self.in_bounds(q):
            for i, s in enumerate(self.sample):
                
                l+= likelihood_1_point(s, q, self.y[i])
                
            prior -= np.sum(((q/self.sigma)**2))
            return (-l - prior)
        else: return np.inf
    
    def gradient(self, q):
        
        g = np.zeros(self.dim)
        for i, s in enumerate(self.sample):
            odds = np.exp(s @ (q[:-1] + q[-1])/(1 + np.exp(s @ (q[:-1] + q[-1]))))
            g_1 = self.y[i] * (1 - odds)
            g_0 = -(1 - self.y[i]) * odds
            g[:-1] += s*(g_0 + g_1)
            g[-1] += g_0 + g_1
            
        g -= q/(self.sigma**2)
        
        return -g
      
    def plot_logit(self, matrix):
        beta = np.array([np.mean(s) for s in matrix.T])
        for s, y in zip(self.sample, self.y):
            plt.scatter(s[0], y, color ='red', marker ='x', alpha = 0.4)
            if y == 0:
                plt.scatter(s[0], np.exp(beta[:-1]@s + beta[-1])/(1 + np.exp(beta[:-1] @ s + beta[-1])), color = 'blue', marker ='.', alpha = 0.4)
            else:
                plt.scatter(s[0], np.exp(beta[:-1]@s + beta[-1])/(1 + np.exp(beta[:-1] @ s + beta[-1])), color = 'green', marker ='.', alpha = 0.4)
        plt.show()
      
    def plot_logit2(self, matrix, x_test, y_test, title):
        post = self.posteriors1(x_test, matrix)
        flagfp = 1
        flagtp = 1
        flagfn = 1
        flagtn = 1
        for s, p, y in zip(x_test, post, y_test):
            if p > .5:
                if y == 1:
                    if self.posterior_predictive_mean(1, s, matrix) > 0.5:
                        if flagtp == 1:
                            flagtp-=1
                            plt.scatter(s[0],s[1], color = 'green', alpha = 0.4, label = 'TP')
                        else:
                            plt.scatter(s[0],s[1], color = 'green', alpha = 0.4)
                else:
                    if flagfp ==1:
                        plt.scatter(s[0],s[1], color = 'red', alpha = 0.4, label = 'FP')
                        flagfp-=1
                    else:
                        plt.scatter(s[0],s[1], color = 'red', alpha = 0.4)
            else:
                if y == 0:
                        if flagtn ==1:
                            flagtn-=1
                            plt.scatter(s[0],s[1], color = 'blue', alpha = 0.4, label = 'TN')
                        else:
                            plt.scatter(s[0],s[1], color = 'blue', alpha = 0.4)
                else:
                    if flagfn == 1:
                        flagfn -=1
                        plt.scatter(s[0],s[1], color = 'orange', alpha = 0.4, label = 'FN')
                        
                    else:
                        plt.scatter(s[0],s[1], color = 'orange', alpha = 0.4)
                

                        
            
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
    def accuracy(self, test_set, y_test, matrix, thr = .5, post = None):
        TP, FP, FN, TN = self.prediction(test_set, y_test, matrix, thr , post )
        return (TP+TN)/(TP+ FP+ FN+ TN)
    def prediction(self, test_set, y_test, matrix, thr = .5, post = None):
        if post is None:
            post = self.posteriors1(test_set, matrix)
        TP = 0
        TN = 0 
        FP = 0 
        FN = 0
        
        i = 0
        for test in test_set:
            if post[i] > thr:
                
                if y_test[i] == 1:
                    TP += 1
                else:
                    FP +=1 
            else:
                if y_test[i] == 1:
                    FN += 1
                else:
                    TN +=1 
            
            i+=1
        return TP, FP, FN, TN
        
    def confusion_matrix(self, test_set, y_test, matrix, thr = .5, post = None):
        TP, FP, FN, TN = self.prediction(test_set, y_test, matrix, thr, post)
        matrix = np.array(np.array([TP, FP, FN,TN])).reshape(2, 2)
        
        # Plot the matrix with the 'Greens' colormap
        plt.imshow(matrix, cmap='Greens', interpolation='nearest')
        
        # Add text to each cell
        for i in range(2):
            for j in range(2):
                number = int(matrix[i, j])
                plt.text(j, i, f"{number}", ha='center', va='center', color='black', fontsize=20)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.show()
        
        return TP, FP, FN,TN
    
    def posteriors1(self,test_set, matrix):
        post = np.zeros(np.shape(test_set)[0])
        for i ,test in enumerate(test_set):
            post[i] = self.posterior_predictive_mean(1, test, matrix)
        return post
        
    
    def ROC_curve(self, test_set, y_test, matrix, label):
        thr = np.linspace(0. ,1, 50)
        x = []
        y = []
        i = 0
        area = 0
        post = self.posteriors1(test_set, matrix)
        for t in thr:
            
            TP, FP, FN,TN = self.prediction(test_set, y_test, matrix, t, post)
            x.append(FP/(FP+TN))
            y.append(TP/(TP+FN))
            area += y[i]*(-x[i]+x[i-1])
            i += 1
        
        area = round(area,3)
        plt.grid()
        plt.title("ROC curve")
        plt.step(x,y, label = label + " = " + f"{area}")
        plt.xlabel('FPR')
        
        plt.ylabel('TPR')
        plt.legend()
        
        
    def posterior_predictive_mean(self, y ,new_sample, sample_q):
        l = 0
        for q in sample_q:
            l_1 = y * (new_sample @ q[:-1] + q[-1] - np.log(1 + np.exp(new_sample @ q[:-1] + q[-1])) )
            
            l_0 = -(1 - y) * np.log(1 + np.exp(new_sample @ q[:-1] + q[-1]))
            
            l += l_1 + l_0
            
        return np.exp(l/np.shape(sample_q)[0])








