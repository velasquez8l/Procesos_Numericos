# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 18:06:34 2022

@author: USUARIO
"""
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import os
from ceros_funciones import metodos
from datos import info_metodos
from scipy.misc import derivative


def proces_data():

    fc =  float(input('ingrese fc: '))
    fy =  float(input('ingrese fy: '))
    h =   float(input('ingrese altura: '))
    b =   float(input('ingrese base: '))
    Ast = float(input('ingrese acero traccion: '))
    Asc = float(input('ingrese acero compresion: '))
    dp =  float(input('ingrese dp: '))
    Pu =  float(input('ingrese Carga ultima: '))
    Mu =  float(input('ingrese Momento ultimo: '))

    return fc, fy, b, Ast, Asc, dp, h, Pu, Mu


def Falla_Bal(fc, fy, b, d, dp, Ast, Asc, h, B1):

    cb = (6000 * d) / (6000 + (fy))
    fps = 6000 * ((cb - dp) / cb)
    a = B1 * cb
    if fps > fy:
        fps = fy
    Pub = (0.65 * ((0.85 * fc * a * b) + (Asc * fps) - (Ast * fy))) / 1000
    Mub = (
        0.65
        * (
            (0.85 * fc * a * b * ((h / 2) - (a / 2)))
            + (Asc * fps * ((d - dp) / 2))
            + (Ast * fy * ((d - dp) / 2))
        )
    ) / 100000
    return Pub, Mub


def Beta1(fc):
    if fc <= 280:
        B = 0.85
    else:
        B = 0.85 - ((0.05 / 70) * (fc - 280))
    if B < 0.65:
        print("No se puede resolver, aumentar resistencia a flexión del concreto (f'c)")
    return B


def graficar(momentos, cargas, Mu, Pu):
    figura1 = plt.figure()
    ax = figura1.add_subplot()
    ax.set_xlabel("Momentos(tonf-m)")
    ax.set_ylabel("Cargas(tonf)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 200)

    plt.plot(momentos, cargas, "-b")
    plt.plot(Mu, Pu, ".k")
    plt.xticks(np.arange(0, 50, 2))
    plt.yticks(np.arange(0, 500, 20))

    plt.show()


class Column:
    def __init__(
        self, fc, fy, b, d, dp, Ast, Asc, h, P, B1, data, nombre_metodo, graficar=False
    ) -> None:
        self.fc = fc
        self.fy = fy
        self.b = b
        self.d = d
        self.dp = dp
        self.Asc = Asc
        self.Ast = Ast
        self.h = h
        self.P = P
        self.B1 = B1
        self.data = data
        self.nombre_metodo = nombre_metodo
        self.graficar = graficar

    # Biseccion
    def C1(self, x0, x1, TOL, f):
        x = (x0 + x1) / 2
        y = x + 1
        while abs(x - y) >= TOL:
            x = (x0 + x1) / 2
            if f(x0) * f(x) < 0:
                x1 = x
            else:
                x0 = x
            y = (x0 + x1) / 2
        c = x
        return c

    def momento_comp(self, c):
        fps = self.fy
        fs = 6000 * ((self.d - c) / c)
        a = self.B1 * c
        return (
            (
                0.65
                * (
                    ((0.85 * self.fc * a * self.b) + (self.Asc * fps) - (self.Ast * fs))
                    / 1000
                )
            )
            - self.P
        ) * np.sin(np.pi / 4)

    def dev_momento_comp(self, c):
        return derivative(self.momento_comp, c, dx=1e-6)

    def dev2_momento_comp(self, c):
        return derivative(self.momento_comp, c, dx=1e-6, n=2)

    def momento_trac(self, c):
        fs = self.fy
        fps = 6000 * ((c - self.dp) / c)
        a = self.B1 * c
        return (
            (
                0.65
                * (
                    ((0.85 * self.fc * a * self.b) + (self.Asc * fps) - (self.Ast * fs))
                    / 1000
                )
            )
            - self.P
        ) * np.sin(np.pi / 4)

    def dev_momento_trac(self, c):
        return derivative(self.momento_trac, c, dx=1e-6)

    def dev2_momento_trac(self, c):
        return derivative(self.momento_trac, c, dx=1e-6, n=2)

    def momento_trac_ult(self, c):
        Pumin = 0.1 * self.fc * self.b * self.h / 1000
        phi = 0.65 + ((Pumin - self.P) / Pumin) * 0.25
        fs = self.fy
        fps = 6000 * ((c - self.dp) / c)
        a = self.B1 * c
        return (
            (
                phi
                * (
                    ((0.85 * self.fc * a * self.b) + (self.Asc * fps) - (self.Ast * fs))
                    / 1000
                )
            )
            - self.P
        ) * np.sin(np.pi / 4)

    def dev_momento_trac_ult(self, c):
        return derivative(self.momento_trac_ult, c, dx=1e-6)

    def dev2_momento_trac_ult(
        self,
        c,
    ):
        return derivative(self.momento_trac_ult, c, dx=1e-6, n=2)

    def funcs(self, nombre, tipo_falla):
        if tipo_falla == "comp":
            if nombre == "newton":
                return [self.momento_comp, self.dev_momento_comp]
            elif nombre == "newton_ceros_multiples":
                return [
                    self.momento_comp,
                    self.dev_momento_comp,
                    self.dev2_momento_comp,
                ]
            elif nombre == "punto_fijo":
                return [self.momento_comp, self.dev_momento_comp]
            else:
                return [self.momento_comp]

        if tipo_falla == "trac":
            if nombre == "newton":
                return [self.momento_trac, self.dev_momento_trac]
            elif nombre == "newton_ceros_multiples":
                return [
                    self.momento_trac,
                    self.dev_momento_trac,
                    self.dev2_momento_trac,
                ]
            elif nombre == "punto_fijo":
                return [self.momento_trac, self.dev_momento_trac]
            else:
                return [self.momento_trac]

        if tipo_falla == "ult":
            if nombre == "newton":
                return [self.momento_trac_ult, self.dev_momento_trac_ult]
            elif nombre == "newton_ceros_multiples":
                return [
                    self.momento_trac_ult,
                    self.dev_momento_trac_ult,
                    self.dev2_momento_trac_ult,
                ]
            elif nombre == "punto_fijo":
                return [self.momento_trac_ult, self.dev_momento_trac_ult]
            else:
                return [self.momento_trac_ult]

    def Falla_Comp(self):
        # CALCULO DE C
        solver = metodos[self.nombre_metodo](**self.data)
        funciones = self.funcs(self.nombre_metodo, "comp")
        solver_results, c = solver.solver(*funciones)
        if self.graficar:
            solver_results.to_excel("tabla_raices.xlsx", index=False)
            graficar_funcion(self.nombre_metodo, funciones[0])
        # c = self.C1(0.01, self.h, 1e-5, self.momento_comp)
        fps = self.fy
        fs = 6000 * ((self.d - c) / c)
        if fs > self.fy:
            fs = self.fy

        Mu = (
            0.65
            * (
                (
                    0.85
                    * self.fc
                    * self.B1
                    * c
                    * self.b
                    * ((self.h / 2) - (self.B1 * c / 2))
                )
                + (self.Asc * fps * ((self.d - self.dp) / 2))
                + (self.Ast * fs * ((self.d - self.dp) / 2))
            )
        ) / 100000
        return Mu

    def Falla_Tracc(self):
        solver = metodos[self.nombre_metodo](**self.data)
        funciones = self.funcs(self.nombre_metodo, "trac")
        solver_results, c = solver.solver(*funciones)
        # c = (0.01, self.h, 1e-5, self.momento_trac)
        fs = self.fy
        fps = 6000 * ((c - self.dp) / c)

        if fps > self.fy:
            fps = self.fy

        Mu = (
            0.65
            * (
                (
                    0.85
                    * self.fc
                    * self.B1
                    * c
                    * self.b
                    * ((self.h / 2) - (self.B1 * c / 2))
                )
                + (self.Asc * fps * ((self.d - self.dp) / 2))
                + (self.Ast * fs * ((self.d - self.dp) / 2))
            )
        ) / 100000
        return Mu

    def Falla_Tracc_Ult(self):

        # CALCULO DE PHI PARA CADA VALOR DE P
        Pumin = 0.1 * self.fc * self.b * self.h / 1000
        phi = 0.65 + ((Pumin - self.P) / Pumin) * 0.25
        solver = metodos[self.nombre_metodo](**self.data)
        funciones = self.funcs(self.nombre_metodo, "ult")
        solver_results, c = solver.solver(*funciones)
        # c = self.C1(0.01, self.h, 1e-5, self.momento_trac_ult)
        fs = self.fy
        fps = 6000 * ((c - self.dp) / c)

        if fps > self.fy:
            fps = self.fy
        Mu = (
            phi
            * (
                (
                    0.85
                    * self.fc
                    * self.B1
                    * c
                    * self.b
                    * ((self.h / 2) - (self.B1 * c / 2))
                )
                + (self.Asc * fps * ((self.d - self.dp) / 2))
                + (self.Ast * fs * ((self.d - self.dp) / 2))
            )
        ) / 100000

        return Mu


def clearConsole():
    command = "clear"
    if os.name in ("nt", "dos"):
        command = "cls"
    os.system(command)


def menu_metodos():
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
        menu_metodos()
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


def graficar_funcion(nombre_metodo, f):
    fig = plt.figure()
    xs = list(np.linspace(-50, 50, 100))
    limitex = 200
    limitey = 200
    ys = [f(x) for x in xs]
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


def get_data():
    info = metodo()
    nombre = list(info.keys())[0]
    data = list(info.values())[0]
    data = data_fill(data)
    return data, nombre


def main():
    info = metodo()
    nombre = list(info.keys())[0]
    data = list(info.values())[0]

    data = data_fill(data)

    solver = metodos[nombre](**data)

    solver_results, x = solver.solver()
    solver_results.to_excel("tabla_de_resultados.xlsx", index=False)
    solver.graficar_funcion(nombre)


def info_grafica():
    fc, fy, b, Ast, Asc, dp, h, Pu, Mu = proces_data()

    B1 = Beta1(fc)
    As_total = Ast + Asc  # calculo Area acero total
    d = h - dp  # cm         # calculo de d
    Ag = b * h  # cm2      # calculo area bruta
    cuantia = As_total / Ag

    Pub, Mub = Falla_Bal(fc, fy, b, d, dp, Ast, Asc, h, B1)
    Pumin = 0.1 * fc * b * h / 1000
    Pnmax = 0.75 * 0.65 * ((0.85 * fc * (Ag - As_total)) + (fy * As_total)) / 1000

    puntos = 100
    cargas_comp = list(np.linspace(Pnmax, Pub, puntos))
    cargas_tracc = list(np.linspace(Pub, Pumin, puntos))
    cargas_tracc_ult = list(np.linspace(Pumin, 0, puntos))

    data, nombre_metodo = get_data()

    Mcomp = []
    for i, P in enumerate(cargas_comp):
        if i == len(cargas_comp) // 2:
            columna = Column(
                fc, fy, b, d, dp, Ast, Asc, h, P, B1, data, nombre_metodo, True
            )
        else:
            columna = Column(fc, fy, b, d, dp, Ast, Asc, h, P, B1, data, nombre_metodo)

        Mcomp.append(columna.Falla_Comp())

    # Mcomp = [
    #     Column(fc, fy, b, d, dp, Ast, Asc, h, P, B1).Falla_Comp() for P in cargas_comp
    # ]
    Mtrac = []
    for P in cargas_tracc:
        columna = Column(fc, fy, b, d, dp, Ast, Asc, h, P, B1, data, nombre_metodo)

        Mtrac.append(columna.Falla_Tracc())
    # Mtrac = [
    #     Column(fc, fy, b, d, dp, Ast, Asc, h, P, B1).Falla_Tracc() for P in cargas_tracc
    # ]

    Mtrac_ult = []
    for P in cargas_tracc_ult:
        columna = Column(fc, fy, b, d, dp, Ast, Asc, h, P, B1, data, nombre_metodo)

        Mtrac_ult.append(columna.Falla_Tracc_Ult())

    # Mtrac_ult = [
    #     Column(fc, fy, b, d, dp, Ast, Asc, h, P, B1).Falla_Tracc_Ult()
    #     for P in cargas_tracc_ult
    # ]

    Momentos = [0]
    Momentos += Mcomp + Mtrac + Mtrac_ult

    cargas = [Pnmax]
    cargas += cargas_comp + cargas_tracc + cargas_tracc_ult
    return Momentos, cargas, Mu, Pu, cuantia
