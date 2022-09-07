import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from pydantic import BaseModel
from scipy.misc import derivative
from datos import info_metodos


class Solver2P(ABC, BaseModel):
    x0: float = None
    x1: float = None
    TOL: float = None
    N_max: int = None
    delta: float = None

    def f(self, x):
        return (np.exp(-x) * x**2) + (4 * np.exp(-x) * x) + (4 / np.exp(x))

    def fd(self, x):
        return derivative(self.f, x, dx=1e-6)

    def g(self, x):
        return np.exp(-x) - np.log(x)

    def fd2(self, x):
        return derivative(self.f, x, dx=1e-6, n=2)

    def graficar(self, nombre_metodo):
        fig = plt.figure()
        xs = list(np.linspace(-10, 10, 100))
        limitex = 15
        limitey = 15
        ys = [self.f(x) for x in xs]
        plt.plot(xs, ys, color="blue", label="f(x)")
        plt.grid()
        plt.ylim(-limitex, limitex)
        plt.xlim(-limitey, limitey)
        plt.plot([-limitex, limitex], [0, 0], color="black", linestyle="--", alpha=0.5)
        plt.plot([0, 0], [-limitey, limitey], color="black", linestyle="--", alpha=0.5)
        plt.xlabel("EJE X")
        plt.ylabel("EJE Y")
        plt.legend()
        plt.title(f"Metodo de {nombre_metodo}")
        plt.savefig("graficos.pdf")

    @abstractmethod
    def solver(self):
        ...


class Biseccion(Solver2P):

    type = "biseccion"

    def solver(self):

        """
        Dada una función f definida en el intervalo [x0,x1] tal que f(x0)*f(x1)<0,
        devuelve un cero de la función contenido en este intervalo usando el método de la bisección.


        Parámetros:
        * x0: Extremo izquierdo
        * x1: Extremo derecho
        * TOL: Diferencia máxima entre dos iteraciones seguidas
        * N_max: Número máximo de iteraciones
        * f: función definida en los reales

        Valor de retorno
        * cero: f(cero) es aproximadamente 0
        """
        x = (self.x0 + self.x1) / 2
        y = x + 1
        iteracion = []
        x0 = []
        x1 = []
        funcx0 = []
        funcx1 = []
        x_mitad = []
        Error = [None]
        for i in range(self.N_max):
            if abs(x - y) < self.TOL:
                iteracion.append(i + 1)
                funcx0.append(self.f(self.x0))
                funcx1.append(self.f(self.x1))
                x0.append(self.x0)
                x1.append(self.x1)
                x_mitad.append(x)
                df = pd.DataFrame(
                    {
                        "n": iteracion,
                        "x0": x0,
                        "x": x_mitad,
                        "x1": x1,
                        "f(x0)": funcx0,
                        "f(x1)": funcx1,
                        "Error": Error,
                    }
                )
                return df
            x = (self.x0 + self.x1) / 2
            x0.append(self.x0)
            x1.append(self.x1)
            funcx0.append(self.f(self.x0))
            funcx1.append(self.f(self.x1))
            if self.f(self.x0) * self.f(x) < 0:
                self.x1 = x
            else:
                self.x0 = x
            y = (self.x0 + self.x1) / 2
            x_mitad.append(x)
            iteracion.append(i + 1)
            Error.append(abs(x - y))
        del Error[-1]
        df = pd.DataFrame(
            {
                "n": iteracion,
                "x0": x0,
                "x": x_mitad,
                "x1": x1,
                "f(x0)": funcx0,
                "f(x1)": funcx1,
                "Error": Error,
            }
        )
        return df


class RegulaFalsi(Solver2P):
    type = "regla_falsa"

    def solver(self):
        """
        Dada una función f definida en los reales tal que f(x0)*f(x1)<0,
        devuelve un cero de la función usando el método de la regula-falsi.


        Parámetros:
        * x0,x1: Primeras iteraciones
        * TOL: Diferencia máxima entre dos iteraciones seguidas
        * N_max: Número máximo de iteraciones
        * f: función definida en los reales

        Valor de retorno
        * x: f(x) es aproximadamente 0
        """
        y0 = self.f(self.x0)
        y1 = self.f(self.x1)
        iteracion = []
        x0 = []
        x1 = []
        funcx0 = []
        funcx1 = []
        x_mitad = []
        Error = [None]
        for i in range(self.N_max):
            x = self.x1 - ((self.x0 - self.x1) / (y0 - y1)) * y1
            if min(abs(x - self.x1), abs(x - self.x0)) < self.TOL:
                iteracion.append(i + 1)
                funcx0.append(self.f(self.x0))
                funcx1.append(self.f(self.x1))
                x0.append(self.x0)
                x1.append(self.x1)
                x_mitad.append(x)
                df = pd.DataFrame(
                    {
                        "n": iteracion,
                        "x0": x0,
                        "x": x_mitad,
                        "x1": x1,
                        "f(x0)": funcx0,
                        "f(x1)": funcx1,
                        "Error": Error,
                    }
                )
                return df
            y = self.f(x)
            Error.append(min(abs(x - self.x1), abs(x - self.x0)))
            x0.append(self.x0)
            x1.append(self.x1)
            funcx0.append(self.f(self.x0))
            funcx1.append(self.f(self.x1))
            if y * y1 < 0:
                self.x0 = self.x1
                y0 = y1
            self.x1 = x
            y1 = y
            x_mitad.append(x)
            iteracion.append(i + 1)
        del Error[-1]
        df = pd.DataFrame(
            {
                "n": iteracion,
                "x0": x0,
                "x": x_mitad,
                "x1": x1,
                "f(x0)": funcx0,
                "f(x1)": funcx1,
                "Error": Error,
            }
        )
        return df


class Secante(Solver2P):
    type = "secante"

    def solver(self):
        """
        Dada una función f definida en los reales con valores iniciales x0 y x1 con f(x0) distinto de f(x1),
        devuelve un cero de la función usando el método de la secante.


        Parámetros:
        * x0,x1: Primeras iteraciones
        * TOL: Diferencia máxima entre dos iteraciones seguidas
        * N_max: Número máximo de iteraciones
        * f: función definida en los reales


        Valor de retorno
        * cero: f(cero) es aproximadamente 0
        """
        y0 = self.f(self.x0)
        y1 = self.f(self.x1)
        iteracion = []
        x0 = []
        x1 = []
        funcx0 = []
        funcx1 = []
        x_mitad = []
        Error = [None]
        for i in range(self.N_max):
            x = self.x1 - ((self.x0 - self.x1) / (y0 - y1)) * y1
            if abs(x - self.x1) < self.TOL:
                iteracion.append(i + 1)
                funcx0.append(self.f(self.x0))
                funcx1.append(self.f(self.x1))
                x0.append(self.x0)
                x1.append(self.x1)
                x_mitad.append(x)
                df = pd.DataFrame(
                    {
                        "n": iteracion,
                        "x0": x0,
                        "x": x_mitad,
                        "x1": x1,
                        "f(x0)": funcx0,
                        "f(x1)": funcx1,
                        "Error": Error,
                    }
                )
                return df
            x0.append(self.x0)
            x1.append(self.x1)
            Error.append(abs(x - self.x1))
            funcx0.append(self.f(self.x0))
            funcx1.append(self.f(self.x1))
            self.x0 = self.x1
            y0 = y1
            self.x1 = x
            y1 = self.f(x)
            x_mitad.append(x)
            iteracion.append(i + 1)
        del Error[-1]
        df = pd.DataFrame(
            {
                "n": iteracion,
                "x0": x0,
                "x": x_mitad,
                "x1": x1,
                "f(x0)": funcx0,
                "f(x1)": funcx1,
                "Error": Error,
            }
        )
        return df


class NewtonRaphson(Solver2P):
    type = "newton"

    def solver(self):
        """
        Dada una función f definida en los reales dónde f'(x0) és distinto de 0,
        devuelve un cero de la función usando el método de Newton-Raphson.


        Parámetros:
        * x0: Primera iteración
        * TOL: Diferencia máxima entre dos iteraciones seguidas
        * N_max: Número máximo de iteraciones
        * f:funcion definida en los reales
        * fd: derivada de f


        Valor de retorno
        * cero: f(cero) es aproximadamente 0
        """
        cero = self.x0
        ant = self.x0 + 1
        iteracion = []
        xs = []
        func = []
        Error = [None]
        for i in range(self.N_max):
            if abs(cero - ant) < self.TOL:
                iteracion.append(i + 1)
                func.append(self.f(cero))
                xs.append(cero)
                df = pd.DataFrame(
                    {"n": iteracion, "x": xs, "f(x)": func, "Error": Error}
                )
                return df
            ant = cero
            cero = cero - self.f(cero) / self.fd(cero)
            xs.append(ant)
            func.append(self.f(ant))
            iteracion.append(i + 1)
            Error.append(abs(cero - ant))
        del Error[-1]
        df = pd.DataFrame({"n": iteracion, "x": xs, "f(x)": func, "Error": Error})
        return df


class PuntoFijo(Solver2P):
    type = "punto_fijo"

    def solver(self):
        """
        Dada una función g definida en los reales dónde g'(x0)<1,
        devuelve un punto fijo de la función g usando el método
        del punto fijo.


        Parámetros:
        * x0: Primera iteración
        * TOL: Diferencia máxima entre dos iteraciones seguidas
        * N_max: Número máximo de iteraciones
        * g: función definida en los reales


        Valor de retorno
        * pf es el punto fijo de la función g(x)
        """
        pf = self.x0
        ant = self.x0 + 1

        iteracion = []
        xs = []
        func = []
        Error = [None]

        for i in range(self.N_max):
            if abs(pf - ant) <= self.TOL:
                iteracion.append(i + 1)
                func.append(self.f(pf))
                xs.append(pf)
                df = pd.DataFrame(
                    {"n": iteracion, "x": xs, "f(x)": func, "Error": Error}
                )
                return df
            ant = pf
            pf = self.g(pf)
            xs.append(ant)
            func.append(self.f(ant))
            iteracion.append(i + 1)
            Error.append(abs(pf - ant))
        del Error[-1]
        df = pd.DataFrame({"n": iteracion, "x": xs, "f(x)": func, "Error": Error})

        return df


class BusquedaIncremental(Solver2P):
    type = "busqueda_incremental"

    def solver(self):

        x1 = self.x0 + self.delta
        y0 = self.f(self.x0)
        y1 = self.f(x1)
        iteracion = []
        x0 = []
        x1 = []
        funcx0 = []
        funcx1 = []

        for i in range(self.N_max):
            if y0 * y1 < 0:
                iteracion.append(i + 1)
                funcx0.append(self.f(self.x0))
                funcx1.append(self.f(self.x1))
                x0.append(self.x0)
                x1.append(self.x1)
                df = pd.DataFrame(
                    {
                        "n": iteracion,
                        "x0": x0,
                        "x1": x1,
                        "f(x0)": funcx0,
                        "f(x1)": funcx1,
                    }
                )
                return self.x0
            iteracion.append(i + 1)
            funcx0.append(self.f(self.x0))
            funcx1.append(self.f(self.x1))
            x0.append(self.x0)
            x1.append(self.x1)
            self.x0 = x1
            y0 = self.f(self.x0)
            x1 = self.x0 + self.delta
            y1 = self.f(self.x1)
        return self.x1


class CerosMultiplesNewton(Solver2P):
    type = "newton_ceros_multiples"

    def solver(self):
        """
        Dada una función f definida en los reales dónde f'(x0) és distinto de 0,
        devuelve la succesión generada por el método de Newton-Raphson modificado.


        Parámetros:
        * x0: Primera iteración
        * N_max: Número máximo de iteraciones
        * f: función definida en los reales
        * fd: función derivada de f
        * fd2: función derivada de fd

        Valor de retorno
        * cero: sucesión generada por el método de Newton-Raphson modificado
        * fcero: sucesión generada por el método de Newton-Raphson modificado evaluada en f
        * diferencias: diferencia entre dos términos seguidos de cada iteración
        """
        cero = self.x0
        ant = self.x0 + 1
        iteracion = []
        xs = []
        func = []
        Error = [None]
        for i in range(self.N_max):
            if abs(cero - ant) < self.TOL:
                iteracion.append(i + 1)
                func.append(self.f(cero))
                xs.append(cero)
                df = pd.DataFrame(
                    {"n": iteracion, "x": xs, "f(x)": func, "Error": Error}
                )
                return df
            ant = cero
            cero = cero - (
                (self.f(cero) * self.fd(cero))
                / ((((self.fd(cero)) ** 2) - (self.f(cero) * self.fd2(cero))))
            )
            xs.append(ant)
            func.append(self.f(ant))
            iteracion.append(i + 1)
            Error.append(abs(cero - ant))
        del Error[-1]
        df = pd.DataFrame({"n": iteracion, "x": xs, "f(x)": func, "Error": Error})
        return df


def calculo_n_puntofijo(datos):
    xo = datos["xo"]
    a = datos["a"]
    b = datos["b"]
    k = datos["k"]
    tol = datos["tol"]

    return np.log(tol / max([xo - a, b - xo])) / np.log(k)


def calcuo_n_biseccion(datos):
    xv = datos["xv"]
    a = datos["a"]
    b = datos["b"]
    xmv = datos["xmv"]
    return np.log2((b - a) / abs(xv - xmv))


def calculo_n_busqueda(datos):
    delta = datos["delta"]
    x = datos["x"]
    x0 = datos["x0"]
    n = datos["n"]
    respuesta = []
    for pos, i in enumerate(x0):
        respuesta.append(i + delta * n[pos] > x)
    return respuesta


calculo_n = dict(
    punto_fijo=calculo_n_puntofijo,
    biseccion=calcuo_n_biseccion,
    busqueda=calculo_n_busqueda,
)

metodos = dict(
    biseccion=Biseccion,
    regla_falsa=RegulaFalsi,
    secante=Secante,
    newton=NewtonRaphson,
    punto_fijo=PuntoFijo,
    busqueda_incremental=BusquedaIncremental,
    newton_ceros_multiples=CerosMultiplesNewton,
)


def clearConsole():
    command = "clear"
    if os.name in ("nt", "dos"):
        command = "cls"
    os.system(command)


def menu():
    print("Nombres Metodos Numericos:")
    print("1. Biseccion")
    print("2. Regla_falsa")
    print("3. Secante")
    print("4. Newton")
    print("5. Punto_fijo")
    print("6. Busqueda_incremental")
    print("7. Newton_ceros_multiples")
    print()
    print("Presione una opcion(1,2,3..) y luego enter")
    print()


def metodo():
    while True:
        clearConsole()
        menu()
        try:
            tipo_metodo = str(input("ingrese el metodo numerico: "))
            if len(tipo_metodo) == 1:
                return info_metodos[tipo_metodo]
            clearConsole()
            print("porfavor ingrese solo el numero que se le muestra en pantalla")
            print("presione enter para continuar")
            input()
        except ValueError:
            clearConsole()
            print("porfavor ingrese solo el numero que se le muestra en pantalla")
            print("presione enter para continuar")
            input()


def data_fill(data):
    data_mod = dict()

    while True:
        clearConsole()
        contador = 0
        for dato, value in data.items():
            try:
                data_mod[dato] = float(input(f"ingrese el valor de {dato}: "))
                print()
                contador += 1
            except ValueError:
                print("porfavor ingrese un valor numerico(valido)".upper())
                clearConsole()
                break
        if contador == len(data):
            break
    return data_mod


def main():
    info = metodo()
    nombre = list(info.keys())[0]
    data = list(info.values())[0]

    data = data_fill(data)

    solver = metodos[nombre](**data)
    solver_results = solver.solver()
    solver_results.to_excel("tabla_de_resultados.xlsx", index=False)
    solver.graficar(nombre)


def calcular_n():
    # CALCULO DE N
    tipo_metodo = "punto_fijo"
    datos = {
        "punto_fijo": {"xo": -0.5, "a": -3, "b": -1, "k": 0.7, "tol": 0.5e-3},
        "biseccion": {"a": -4, "b": 0, "xv": -3.1415, "xmv": -3.1418},
        "busqueda": {
            "delta": 0.1,
            "x": 0.875,
            "x0": [-0.8, -0.5, 0, 0.8],
            "n": [16, 14, 10, 1],
        },
    }
    n = calculo_n[tipo_metodo](datos[tipo_metodo])
    # print(n)


if __name__ == "__main__":
    main()
