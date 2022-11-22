import numpy as np

def sustreg(mat_aumentada,n):
    x=np.zeros((n))
    x[n-1]=mat_aumentada[n-1,n]/mat_aumentada[n-1,n-1]
    for i in range(n-2,-1,-1):
        suma=0
        for j in range(n):
            suma+=mat_aumentada[i,j]*x[j]
        x[i]=(mat_aumentada[i,n]-suma)/mat_aumentada[i,i]
    return x


def pivpar(mat_aumentada,n,i):
    mayor=abs(mat_aumentada[i,i])
    maxrow=i
    for j in range(i+1,n):
        if abs(mat_aumentada[j,i])>mayor:
            mayor=abs(mat_aumentada[j,i])
            maxrow=j

    if mayor==0:
       print('El sistema no tiene solución única')
    elif maxrow!=i :
        mat_aumentada[[i]],mat_aumentada[[maxrow]]=mat_aumentada[[maxrow]],mat_aumentada[[i]]
    return mat_aumentada
    
def pivtot(mat_aumentada,mark,n,i):
    mayor=0
    maxrow=i
    maxcol=i
    for r in range(i,n):
        for s in range(i,n):
            if abs(mat_aumentada[r,s])>mayor:
                mayor=abs(mat_aumentada[r,s])
                maxrow=r
                maxcol=s

    if mayor==0:
       print('El sistema no tiene solución única')
    else:
        if maxrow!=i:
            mat_aumentada[[i]],mat_aumentada[[maxrow]]=mat_aumentada[[maxrow]],mat_aumentada[[i]]
        if maxcol!=i:
            mat_aumentada[:,[i]],mat_aumentada[:,[maxcol]]=mat_aumentada[:,[maxcol]],mat_aumentada[:,[i]]
            mark[i],mark[maxcol]=mark[maxcol],mark[i]
            
    return mat_aumentada,mark

def organice_mat(x,mark,n):
    new_x=np.zeros((n))
    for pos, i in enumerate(new_x):
        new_x[pos]=x[mark[pos]]
    return new_x

def error_sol(a,b,x):
    return np.dot(a,x)-b.T
