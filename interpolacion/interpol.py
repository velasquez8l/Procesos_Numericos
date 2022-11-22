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

def f(r,w,ma,x):
    return r*x-ma-((w*x**2)/2)


def sol(a,b,c):
    sol1=(-b+((b)**2-(4*a*c))**0.5)/(2*a)
    sol2=(-b-((b)**2-(4*a*c))**0.5)/(2*a)
    res=[sol1,sol2]
    return list(sorted(res))

def get_points():
    w=10
    L=5
    r=w*L/2
    ma=(w*L**2)/12
    x1,x2=sol(w/2,-r,ma)
    xs=np.array([0,x1,L/2,x2,L])
    ys=np.array([f(r,w,ma,x) for x in xs ])

    return xs,ys

def error_newton(last,xs):
    error=last
    x_f=xs[-1]
    for x in xs[:-1]:
        error*=(x_f-x)
    return error

def newton(graph):
    xs,ys=get_points()
    pol,tabla=diferencias_newton(xs, ys,graph)

    last=list({key:list(val.values()) for key,val in tabla.to_dict().items()}.values())
    last=last[-1][-1]
    tabla.to_excel("tabla_newton.xlsx",index=False)
    error=error_newton(last,xs)
    print(f" Error: {error}")




def splines(graficar):
    xs,ys=get_points()
    pols,list_pols=spline_cubico(xs, ys)
    # pols,list_pols=spline_lineal(xs, ys)

    x=sympy.Symbol('x')
    cx=[]
    cy=[]
    print(pols)

    for i,pol in enumerate(pols):
        pol=get_expr(pol)

        x=sympy.Symbol("x")
        cordsx=list(np.linspace(xs[i],xs[i+1],100))
        cordsy=[pol.subs({x:cord})for cord in cordsx]
        
        cx.append(cordsx)
        cy.append(cordsy)

    if graficar:

        plt.figure()
        ax=plt.subplot()
        for i,cord in enumerate(cx):
            plt.plot(cord,cy[i],"-b")

        for i,co in enumerate(xs):
            plt.plot(co,ys[i],"*r")

        plt.title(f"Momento flector")
        ax.set_xlabel("Eje x")
        ax.set_ylabel("Momentos")
        ax.invert_yaxis()
        plt.grid()
        plt.show()

if __name__=="__main__":
    graficar=False
    newton(graficar)
    # splines(graficar)