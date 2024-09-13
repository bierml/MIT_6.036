import numpy as np
Y=np.array([[4,3,1],[1,3,-2],[5,2,3]])
(r_eval, r_evec) = np.linalg.eig(Y.T @ Y)
(l_eval, l_evec) = np.linalg.eig((Y @ Y.T).T)

col_sums = np.sum(l_evec**2,axis=0, keepdims=True)
U_ = l_evec / col_sums

#print(U_)

col_sums = np.sum(r_evec**2,axis=0, keepdims=True)
V_ = r_evec / col_sums

#print(V_)

#print(np.diag(r_eval))
#E_=np.sqrt(np.diag(r_eval))
def myfunc(x):
    if(x>0):
        return np.sqrt(x)
    else:
        return 0
vectmyfunc=np.vectorize(myfunc)
E_=vectmyfunc(np.diag(r_eval))
print("U_=",U_)
print("V_=",V_)
print("E_=",E_)
#will not change the result
#U_[:,1]=0
#V_[:,1]=0
print("U_*E_*V_^T=",U_@vectmyfunc(np.diag(r_eval))@V_.T)