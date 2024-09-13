# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:17:39 2024

@author: abcde
"""
import numpy as np
def perceptron(data, labels):
    T = 1000
    # Your implementation here

    th = np.zeros((data.shape[0],1))
    th0 = np.zeros((1,1))
    #th = np.array([[0,1,-0.5]]).T
    mistakesnum = 0
    for t in range(T):
        for i in range(data.shape[1]):
            if labels[:,i].reshape(1,1)*(th.T@(data[:,i].reshape((data.shape[0],1)))+th0)<=0:
                th = th+labels[:,i].reshape(1,1)*data[:,i].reshape((data.shape[0],1))
                th0 = th0+labels[:,i].reshape(1,1)
                #print(th,end='\n')
                mistakesnum = mistakesnum + 1
    print(mistakesnum)
    return (th,th0)
    pass