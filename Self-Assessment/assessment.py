# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def array_mult(A, B):
    res = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            val = 0
            for k in range(len(B)):
                val = val + A[i][k]*B[k][j]
            row.append(val)
        res.append(row)
    return res

def add_n(n):
    def myfunc(l):
        res = []
        for i in range(len(l)):
            res.append(l[i]+n)
        return res
    return myfunc

def dot(v1, v2):
    c = 0
    for i in range(len(v1)):
        c = c + v1[i]*v2[i]
    return c

def add_two_lists(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c
