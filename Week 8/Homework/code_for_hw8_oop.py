# This OPTIONAL problem has you extend your homework 7
# implementation for building neural networks.  
# PLEASE COPY IN YOUR CODE FROM HOMEWORK 7 TO COMPLEMENT THE CLASSES GIVEN HERE

# Recall that your implementation from homework 7 included the following classes:
    # Module, Linear, Tanh, ReLU, SoftMax, NLL and Sequential

######################################################################
# OPTIONAL: Problem 2A) - Mini-batch GD
######################################################################
import numpy as np
import math as m

class Sequential:
    def __init__(self, modules, loss):            
        self.modules = modules
        self.loss = loss

    def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10):
        D, N = X.shape

        np.random.seed(0)
        num_updates = 0
        indices = np.arange(N)
        while num_updates < iters:

            np.random.shuffle(indices)
            X = X[:,indices]  # Your code
            Y = Y[:,indices]  # Your code

            for j in range(m.floor(N/K)):
                if num_updates >= iters: break

                # Implement the main part of mini_gd here
                # Your code
                Ypred = self.forward(X[:,j*K:(j+1)*K][:,np.newaxis])
                self.loss.forward(Ypred,Y[:,j*K:(j+1)*K].reshape(Ypred.shape))
                self.backward(self.loss.backward())
                self.sgd_step(lrate)
                num_updates += 1

    def forward(self, Xt):                        
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                   
        for m in self.modules[::-1]: delta = m.backward(delta)

    def step(self, lrate):    
        for m in self.modules: m.step(lrate)
          
######################################################################
# OPTIONAL: Problem 2B) - BatchNorm
######################################################################

class Module:
    def step(self, lrate): pass  # For modules w/o weights
class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        #print('A=',A)
        #print('W=',self.W)
        #print('W0',self.W0)
        self.A = A   # (m x b)  Hint: make sure you understand what b stands for
        res = self.W.T@A+self.W0
        #res = ((self.W.T@A).T+self.W0.T).T;
        #print('res=',(self.W.T@A).T+self.W0.T)
        return res
        #return None  # Your code (n x b)

    def backward(self, dLdZ):  # dLdZ is (n x b), uses stored self.A
        #print('dldZ=',dLdZ)
        #print('A=',self.A)
        self.dLdW  = self.A@(dLdZ).T  # Your code
        self.dLdW0 = dLdZ  # Your code
        #print(np.sum(self.dLdW0,axis=1))
        dLdA = self.W@dLdZ
        return dLdA
        #return None        # Your code: return dLdA (m x b)

    def sgd_step(self, lrate):  # Gradient descent step
    
        self.W  = self.W-lrate*self.dLdW  # Your code
        #print(lrate)
        self.W0 = self.W0-lrate*np.sum(self.dLdW0,axis=1).reshape((self.W0.shape[0],1))  # Your code


# Activation modules
#
# Each activation module has a forward method that takes in a batch of
# pre-activations Z and returns a batch of activations A.
#
# Each activation module has a backward method that takes in dLdA and
# returns dLdZ, with the exception of SoftMax, where we assume dLdZ is
# passed in.
class Tanh(Module):  # Layer activation
    def forward(self, Z):
        #print('Z=',Z)
        self.A = np.tanh(Z)
        #print('A=',self.A)
        return self.A

    def backward(self, dLdA):  # Uses stored self.A
        #return dLdA@np.diag((1-self.A**2).flatten())
        #print(dLdA)
        #print(self.A)
        #print(1-self.A**2)
        return dLdA*(1-self.A**2)
        #return None  # Your code: return dLdZ (?, b)


class ReLU(Module):  # Layer activation
    def forward(self, Z):
        res = np.zeros((Z.shape[0],Z.shape[1]))
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if Z[i][j] >= 0:
                    res[i][j]=Z[i][j]
        self.A = res  # Your code: (?, b)
        return self.A

    def backward(self, dLdA):  # uses stored self.A
        return dLdA*(self.A>0).astype(int)
        #return None  # Your code: return dLdZ (?, b)


class SoftMax(Module):  # Output activation
    def forward(self, Z):
        #print('Z=',Z)
        #print('expZ=',np.exp(Z))
        summ = np.sum(np.exp(Z),axis=0)**(-1)
        #print('summ=',summ)
        #print('summ=',np.sum(np.exp(Z),axis=0))
        res = np.exp(Z)
        #print('res=',res)
        for i in range(Z.shape[1]):
            res[:,i]=res[:,i]*summ[i]
        return res
        #return None  # Your code: (?, b)

    def backward(self, dLdZ):  # Assume that dLdZ is passed in
        return dLdZ
    #CONTINUE HERE
    def class_fun(self, Ypred):  # Return class indices
        return np.argmax(Ypred,axis=0) # Your code: (1, b)

class NLL(Module):  # Loss
    def forward(self, Ypred, Y):
        #Ypred - predictions from the last layer of the network
        self.Ypred = Ypred
        #Y - actual classification (one-hot)
        self.Y = Y
        #print(self.Ypred)
        #print(self.Y)
        indarr = np.argmax(Y,axis=0)
        lgpred = np.log(Ypred)
        n = lgpred.shape[1]
        return np.sum(-lgpred[indarr,np.arange(n)],axis=0)
        #return None  # Your code: return loss (scalar)

    def backward(self):  # Use stored self.Ypred, self.Y
        return self.Ypred-self.Y
        #return None  # Your code (?, b)

class BatchNorm(Module):    
    def __init__(self, m):
        np.random.seed(0)
        self.eps = 1e-20
        self.m = m  # number of input channels
        
        # Init learned shifts and scaling factors
        self.B = np.zeros([self.m, 1])
        self.G = np.random.normal(0, 1.0 * self.m ** (-.5), [self.m, 1])
        
    # Works on m x b matrices of m input channels and b different inputs
    def forward(self, A):# A is m x K: m input channels and mini-batch size K
        self.A = A
        self.K = A.shape[1]
        #print('A=',self.A)
        self.mus = np.sum(self.A,axis=1).reshape((self.A.shape[0],1))/self.K  # Your Code
        #print('mus=',self.mus)
        self.vars = ((1/self.K*np.sum((self.A-self.mus)**2,axis=1))).reshape((self.A.shape[0],1))  # Your Code
        #print('vars=',self.vars)
        # Normalize inputs using their mean and standard deviation
        self.norm = np.zeros(A.shape)
        for i in range(self.norm.shape[0]):
            self.norm[i] = (A-self.mus)[i]/(self.vars[i][0]**0.5+self.eps)
        # Return scaled and shifted versions of self.norm
        #print('norm=',self.norm)
        #print('G=',self.G)
        #print('B=',self.B)
        
        res = np.copy(self.norm)
        for i in range(self.m):
            res[i] = res[i]*self.G[i][0]
        res = res + self.B
        #print('res=',res)
        #print(self.G)
        #print(self.B)
        #print(res)
        #print(res[i])
        return res

    def backward(self, dLdZ):
        # Re-usable constants
        std_inv = 1/np.sqrt(self.vars+self.eps)
        A_min_mu = self.A-self.mus
        
        dLdnorm = dLdZ * self.G
        dLdVar = np.sum(dLdnorm * A_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True)
        dLdMu = np.sum(dLdnorm*(-std_inv), axis=1, keepdims=True) + dLdVar * (-2/self.K) * np.sum(A_min_mu, axis=1, keepdims=True)
        dLdX = (dLdnorm * std_inv) + (dLdVar * (2/self.K) * A_min_mu) + (dLdMu/self.K)
        
        self.dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        self.dLdG = np.sum(dLdZ * self.norm, axis=1, keepdims=True)
        return dLdX

    def step(self, lrate):
        self.B = self.B-lrate*self.dLdB  # Your Code
        self.G = self.G-lrate*self.dLdG  # Your Code


######################################################################
# Tests
######################################################################
#b = BatchNorm(3)
#b.forward(np.array([[1,2,3],[4,5,6],[7,8,9]]))
def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)
  
def for_softmax(y):
    return np.vstack([1-y, y])
  
# For problem 1.1: builds a simple model and trains it for 3 iters on a simple dataset
# Verifies the final weights of the model
def mini_gd_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 3, lrate=0.005, K=1)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]
  
# For problem 1.2: builds a simple model with a BatchNorm layer
# Trains it for 1 iter on a simple dataset and verifies, for the BatchNorm module (in order): 
# The final shifts and scaling factors (self.B and self.G)
# The final running means and variances (self.mus_r and self.vars_r)
# The final 'self.norm' value
def batch_norm_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), BatchNorm(2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 1, lrate=0.005, K=2)
    return [np.vstack([nn.modules[3].B, nn.modules[3].G]).tolist(), \
    np.vstack([nn.modules[3].mus_r, nn.modules[3].vars_r]).tolist(), nn.modules[3].norm.tolist()]