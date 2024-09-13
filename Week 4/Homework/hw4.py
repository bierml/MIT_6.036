import numpy as np
import math


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

x_1, y_1 = super_simple_separable()
th1, th1_0 = sep_e_separator

def sd(x,y,th,th0):
    th_ = th[:,0].reshape((th.shape[0],1))
    return y*(th[:,0]@x+th0)/((th_.T@th_)**0.5)

data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5
#print(red_th[:,0].T.shape)
#print(red_th[:,0].T)

val = 0
for i in range(8):
    print(sd(data[:,i],labels[0][i],blue_th,blue_th0))
    val += sd(data[:,i],labels[0][i],blue_th,blue_th0)
print(val)

data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4
print('------------')
for i in range(3):
    print(sd(data[:,i],labels[0][i],th,th0))

def gd(f, df, x0, step_size_fn, max_iter):
    fs = [f(x0)]
    xs = [x0]
    x = x0
    for iter in range(max_iter):
        step = step_size_fn(iter)
        x = x - step*df(x)
        xs.append(x)
        fs.append(f(x))
    return (x,fs,xs)


def hinge(v):
    return np.where(v<1,1-v,0)
def hinge_loss(x, y, th, th0):
    sum = 0
    for i in range(x.shape[1]):
        #print("We are here!")
        #print(y[:,i]*(th[:,0].T@x[:,i]+th0))
        sum += hinge(y[:,i]*(th[:,0].T@x[:,i]+th0))
    return (sum/x.shape[1])[0][0]
def d_hinge(v):
    res = v.copy()
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            res[i][j] = np.where(v[i][j]<1,-1,0)
    return res

X1 = np.array([[1, 2, 3, 9, 10]])
y1 = np.array([[1, 1, 1, -1, -1]])
th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
X2 = np.array([[2, 3, 9, 12],
               [5, 2, 6, 5]])
y2 = np.array([[1, -1, 1, -1]])
th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])


# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    resvec = np.zeros((x.shape[0],x.shape[1]))
    for j in range(x.shape[0]):
        for i in range(x.shape[1]):
            #print(d_hinge(y[0][i]*(th[:,0].T@x[:,i]+th0))[0][0].shape)
            resvec[j][i] = y[0][i]*x[j][i]*d_hinge(y[0][i]*(th[:,0].T@x[:,i]+th0))[0][0]
    return resvec

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    resvec = np.zeros((1,x.shape[1]))
    for i in range(x.shape[1]):
        #print(d_hinge(y[0][i]*(th[:,0].T@x[:,i]+th0)))
        resvec[0][i] = y[0][i]*d_hinge(y[0][i]*(th[:,0].T@x[:,i]+th0))[0][0]
    return resvec
# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    return (np.sum(d_hinge_loss_th(x, y, th, th0),axis=1)/x.shape[1]+2*lam*th.T).T

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.array([np.sum(d_hinge_loss_th0(x, y, th, th0),axis=1)/x.shape[1]])

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(X, y, th, th0, lam):
    return np.vstack((d_svm_obj_th(X, y, th, th0, lam),d_svm_obj_th0(X,y,th,th0,lam).T))

'''def batch_svm_min(data, labels, lam):
    th_ = np.zeros((data.shape[0]+1,1))
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    pass
    def svm_obj_(th_):
        return svm_obj(data,labels,th_[:th_.shape[0]-1,0].reshape((th_.shape[0]-1,1)),th_[th_.shape[0]-1].reshape((1,1)),lam)
    def svm_obj_grad_(th_):
        return svm_obj_grad(data,labels,th_[:th_.shape[0]-1,0].reshape((th_.shape[0]-1,1)),th_[th_.shape[0]-1].reshape((1,1)),lam)
    return gd(svm_obj_,svm_obj_grad_,th_,svm_min_step_size_fn,10)'''