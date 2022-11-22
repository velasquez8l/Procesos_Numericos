import numpy as np
import scipy


def mathJacobiSeidel(x0, A, b, tol, n_iter_max, method):

    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, +1)

    message = ""
    errores = [tol + 1]
    itera = [x0]

    for _ in range(n_iter_max):
        if errores[-1] <= tol:
            message = f"{x0} es una aproximación de la solución del sistema con una tolerancia= {tol}"
            break

        if method == "jacobi":
            T = np.linalg.inv(D).dot(L + U)
            C = np.linalg.inv(D).dot(b)
            x1 = T.dot(x0) + C
        elif method == "seidel":
            T = np.linalg.inv(D - L).dot(U)
            print(T)
            C = np.linalg.inv(D - L).dot(b)
            x1 = T.dot(x0) + C
           


        # errores.append(np.linalg.norm(x1 - x0))
        itera.append(x0)
        x0 = x1
    else:
        message = f"Fracasó en {n_iter_max} iteraciones"

    return errores, x0, message, itera


def iterativos(x0, A, b, tol, n_iter_max, method):

    message = ""
    errores = [tol + 1]

    for _ in range(n_iter_max):
        if errores[-1] <= tol:
            message = f"{x0} es una aproximación de la solución del sistema con una tolerancia= {tol}"
            break

        x1 = new_jacobi(x0, A, b) if method == "jacobi" else new_seidel(x0, A, b)
        errores.append(np.linalg.norm(x1 - x0))
        x0 = x1.copy()
    else:
        message = f"Fracasó en {n_iter_max} iteraciones"

    return x0.tolist(), message, errores


def new_jacobi(x0, A, b):
    n = A.shape[0]
    x1 = x0.copy()
    for i in range(n):
        suma = sum(A[i, j] * x0[j] for j in range(n) if j != i)
        x1[i] = (b[i] - suma) / A[i, i]

    return x1


def new_seidel(x0, A, b):

    n = A.shape[0]
    x1 = x0.copy()
    for i in range(n):
        suma = sum(A[i, j] * x1[j] for j in range(n) if j != i)
        x1[i] = (b[i] - suma) / A[i, i]

    return x1


def LU(A, b, pivoteo=None):
    n = A.shape[0]
    P = np.eye(n)
    L = np.eye(n)

    for k in range(n - 1):

        if pivoteo == "parcial":
            A, P = pivoteoLU(A, P, k)

        for i in range(k + 1, n):
            M = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= M * A[k, j]

            A[i, k] = M

    U = np.triu(A)
    L = L + np.tril(A, -1)
    B = P.dot(b)
    LB = np.zeros((len(L), len(L) + 1))
    LB[:, :-1] = L
    LB[:, -1] = b

    z = sustitucion_progresiva(LB)

    Uz = np.zeros((len(U), len(U) + 1))
    Uz[:, :-1] = U
    Uz[:, -1] = z

    x = sustitucion_regresiva(Uz)

    return x, L, U


def LUdirecto(A, met):
    n = A.shape[0]
    U = np.eye(n)
    L = np.eye(n)

    for k in range(n):
        sum1 = sum(L[k, p] * U[p, k] for p in range(k))
        if met == 0:
            U[k, k] = (A[k, k] - sum1) / L[k, k]
        elif met == 1:
            L[k, k] = (A[k, k] - sum1) / U[k, k]
        else:
            U[k, k] = np.sqrt(A[k, k] - sum1)
            L[k, k] = U[k, k]

        for i in range(k, n):
            sum2 = sum(L[i, p] * U[p, k] for p in range(k))
            L[i, k] = (A[i, k] - sum2) / U[k, k]

        for j in range(k, n):
            sum3 = sum(L[k, p] * U[p, j] for p in range(k))
            U[k, j] = (A[k, j] - sum3) / L[k, k]

    return L, U


def pivoteoLU(A, P, row):
    n = A.shape[0]

    mayor = abs(A[row, row])
    maxrow = row
    for i in range(row + 1, n):
        if abs(A[i, row]) > mayor:
            mayor = abs(A[i, row])
            maxrow = i

    if mayor == 0:
        return A, P, "El sistema no tiene solución única"

    if maxrow != row:  # Intercambio de filas
        aux = A[row, :]
        auxP = P[row, :]
        A[row, :] = A[maxrow, :]
        P[row, :] = P[maxrow, :]
        A[maxrow, :] = aux
        P[maxrow, :] = auxP

    return A, P, None


def sustitucion_regresiva(Ab):
    n = Ab.shape[0] - 1

    x = np.zeros(n + 1)
    x[n] = Ab[n, n + 1] / Ab[n, n]
    for i in range(n - 1, -1, -1):
        suma = sum(Ab[i, j] * x[j] for j in range(i + 1, n + 1))
        x[i] = (Ab[i, -1] - suma) / Ab[i, i]

    return x


def sustitucion_progresiva(Ab):
    n = Ab.shape[0] - 1
    x = np.zeros(n + 1)
    x[0] = Ab[0, n + 1] / Ab[1, 1]

    for i in range(1, n):
        suma = sum(Ab[i, j] * x[j] for j in range(i - 1))
        x[i] = (Ab[i, n + 1] - suma) / Ab[i, i]

    return x


# def k(T):
#     return np.log()
def main():

    x0 = np.array([[1], [3], [2]])
    A = np.array([[3, -5, -8], [2, 4, 6], [3, 4, -12]])
    b = np.array([[-15], [12], [8]])
    tol = 0.5e-5
    n_iter_max = 100
    method = "seidel"

    errores, x0, message, itera = mathJacobiSeidel(x0, A, b, tol, n_iter_max, method)
    print(x0)
    print()


if __name__ == "__main__":
    main()
    # A = [[0, 1.66666667, 2.66666667], [-0.5, 0, -1.5], [0.25, 0.33333333, 0]]
    # val,vect=np.linalg.eig(A)
    # print(max(abs(val)))