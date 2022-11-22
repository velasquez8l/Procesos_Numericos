import numpy as np
from utils import sustreg, pivpar, pivtot, organice_mat,error_sol


def gaussPiv(A, b, piv=0):
    mat_aumentada = np.concatenate((A, b), axis=1, dtype=float)
    n = len(mat_aumentada[:, 0])
    mark = [i for i in range(n)]
    for i in range(n):
        if piv == 1:
            mat_aumentada = pivpar(mat_aumentada, n, i)
        elif piv == 2:
            mat_aumentada, mark = pivtot(mat_aumentada, mark, n, i)
        for j in range(i + 1, n):
            M = mat_aumentada[j, i] / mat_aumentada[i, i]
            for k in range(n + 1):
                mat_aumentada[j, k] = mat_aumentada[j, k] - (M * mat_aumentada[i, k])
    x = sustreg(mat_aumentada, n)
    x = organice_mat(x, mark, n)
    return x, mark


def main():
    piv = 2
    a = np.matrix([
        [21,3,1], 
        [-14,-18,8], 
        [-2,3,0]]
        )

    b = np.matrix([[19], [32], [-11]])
    x, mark = gaussPiv(a, b, piv)
    error=error_sol(a,b,x)
    print(f'solucion: {x}')
    print()
    print(f'Error: {error}')
    # print(np.linalg.norm(error,np.inf))

if __name__ == "__main__":
    main()
