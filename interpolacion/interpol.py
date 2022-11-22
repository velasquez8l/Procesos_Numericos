import numpy as np
import sympy
import pandas as pd
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    function_exponentiation,
    implicit_multiplication_application,
)
import matplotlib.pyplot as plt
def parse_pol(pols):
    polinomios = []
    x_symbol = sympy.Symbol("x")
    for pol in pols:
        polinomio = sum(
            pol[i] * x_symbol ** (len(pol) - 1 - i) for i in range(len(pol))
        )
        polinomios.append(str(sympy.simplify(polinomio)))

    return polinomios
def lagrange(x, y):
    n = len(x)
    tabla = np.zeros((n, n))
    for i in range(n):
        Li = 1
        den = 1
        for j in range(n):
            if j != i:
                paux = [1, -x[j]]
                Li = np.convolve(Li, paux)
                den *= x[i] - x[j]
        tabla[i, :] = y[i] * Li / den

    pols = [sum(tabla).tolist()]
    polinomio = parse_pol(pols)
    return polinomio, pols


def diferencias_newton(xs, ys,graph=False):
    n = len(xs)
    tabla = np.zeros((n, n + 1))
    tabla[:, 0] = xs
    tabla[:, 1] = ys
    for j in range(2, n + 1):
        for i in range(j - 1, n):
            denominador = tabla[i, 0] - tabla[i - j + 1, 0]
            tabla[i, j] = (tabla[i, j - 1] - tabla[i - 1, j - 1]) / denominador

    pol = 0
    coef = np.diag(tabla[:, 1:])
    x_symbol = sympy.Symbol("x")
    for i in range(len(coef)):
        const = coef[i]
        for j in range(i):
            const *= x_symbol - xs[j]
        pol += const

    pol=str(sympy.simplify(pol))
    tabla.tolist()
    tabla=np.array(tabla)
    tabla=tabla.T
    titulos=["x","y"]
    for i in range(len(xs)-1):
        titulos.append(f"iter{i+1}")
    dic={tit:list(val) for tit,val in zip(titulos,tabla)}
    df=pd.DataFrame(dic)
    x=sympy.Symbol('x')
    pol=get_expr(pol)

    x=sympy.Symbol("x")
    cordsx=list(np.linspace(xs[0],xs[-1],100))
    cordsy=[pol.subs({x:cord})for cord in cordsx]
    if graph:
        figure=plt.figure()
        plt.plot(cordsx,cordsy,"-b")
        plt.plot(xs,ys,"*r")
        plt.title(f"polinomio {pol}")
        plt.grid()
        plt.show()

    return pol, df


def spline_lineal(x, y):
    d = 1
    n = len(x)
    A = np.zeros(((d + 1) * (n - 1), (d + 1) * (n - 1)))
    b = np.zeros(((d + 1) * (n - 1), 1))

    c = 0
    h = 0
    for i in range(n - 1):
        A[i, c] = x[i]
        A[i, c + 1] = 1
        b[i] = y[i]
        c += 2
        h += 1

    c = 0
    for i in range(1, n):
        A[h, c] = x[i]
        A[h, c + 1] = 1
        b[h] = y[i]
        c += 2
        h += 1

    val = np.dot(np.linalg.inv(A), b)
    pols = np.reshape(val, (n - 1, d + 1))
    polinomios = parse_pol(pols)
    return polinomios, pols.tolist()


def spline_cuadratico(x, y):
    d = 2
    n = len(x)
    A = np.zeros(((d + 1) * (n - 1), (d + 1) * (n - 1)))
    b = np.zeros(((d + 1) * (n - 1), 1))
    cua = x ** 2

    c = 0
    h = 0
    for i in range(n - 1):
        A[i, c] = cua[i]
        A[i, c + 1] = x[i]
        A[i, c + 2] = 1
        b[i] = y[i]
        c += 3
        h += 1

    c = 0
    for i in range(1, n):
        A[h, c] = cua[i]
        A[h, c + 1] = x[i]
        A[h, c + 2] = 1
        b[h] = y[i]
        c += 3
        h += 1

    c = 0
    for i in range(1, n - 1):
        A[h, c] = 2 * x[i]
        A[h, c + 1] = 1
        A[h, c + 3] = -2 * x[i]
        A[h, c + 4] = -1
        b[h] = 0
        c += 4
        h += 1

    A[h, 0] = 2
    b[h] = 0

    val = np.dot(np.linalg.inv(A), b)
    pols = np.reshape(val, (n - 1, d + 1))
    polinomios = parse_pol(pols)
    return polinomios, pols.tolist()


def spline_cubico(x, y):
    d = 3
    n = len(x)
    A = np.zeros(((d + 1) * (n - 1), (d + 1) * (n - 1)))
    b = np.zeros(((d + 1) * (n - 1), 1))
    cua = x ** 2
    cub = x ** 3

    c = 0
    h = 0
    for i in range(n - 1):
        A[i, c : c + 4] = [cub[i], cua[i], x[i], 1]
        b[i] = y[i]
        c += 4
        h += 1

    c = 0
    for i in range(1, n):
        A[h, c : c + 4] = [cub[i], cua[i], x[i], 1]
        b[h] = y[i]
        c += 4
        h += 1

    c = 0
    for i in range(1, n - 1):
        A[h, c] = 3 * cua[i]
        A[h, c + 1] = 2 * x[i]
        A[h, c + 2] = 1

        A[h, c + 4] = -3 * cua[i]
        A[h, c + 5] = -2 * x[i]
        A[h, c + 6] = -1
        b[h] = 0
        c += 4
        h += 1

    c = 0
    for i in range(1, n - 1):
        A[h, c] = 6 * x[i]
        A[h, c + 1] = 2
        A[h, c + 4] = -6 * x[i]
        A[h, c + 5] = -2
        b[h] = 0
        c += 4
        h += 1

    A[h, 0] = 6 * x[0]
    A[h, 1] = 2
    b[h] = 0
    h += 1
    A[h, c] = 6 * x[-1]
    A[h, c + 1] = 2
    b[h] = 0

    val = np.dot(np.linalg.inv(A), b)
    pols = np.reshape(val, (n - 1, d + 1))
    polinomios = parse_pol(pols)
    return polinomios, pols.tolist()

def get_expr(function):

    return parse_expr(
        function,
        transformations=(
            standard_transformations
            + (
                function_exponentiation,
                implicit_multiplication_application,
            )
        ),
    )


def error_newton(last,xs):
    error=last
    x_f=xs[-1]
    for x in xs[:-1]:
        error*=(x_f-x)
    return error

def newton():
    graph=False
    xs=[3,3.6667,4.3333]
    ys=[6.7472,10.7997,15.8063]

    pol,tabla=diferencias_newton(xs, ys,graph)

    last=list({key:list(val.values()) for key,val in tabla.to_dict().items()}.values())
    last=last[-1][-1]

    error=error_newton(last,xs)

    print(pol)
    print()
    print(tabla)
    print()
    print(f" Error: {error}")


if __name__=="__main__":
    # newton()
    # x=np.array([1,1.1,1.2])
    # y=np.array([3.21,3.64,4.11])
    # pols,list_pols=spline_cubico(x, y)  
    # print(pols)
    
    ## lagrange
    xs=[-1.8265,2.6988,5.5548]
    val=3.85472
    for i,xi in enumerate(xs) :
        num = 1
        den = 1
        for j,xj in enumerate(xs):
            if j != i:
                num*= val-xj
                den *= xi - xj
        print(num / den)