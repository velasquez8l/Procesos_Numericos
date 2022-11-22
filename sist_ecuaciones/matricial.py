from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from Met_iterativos import mathJacobiSeidel
import pandas as pd
class Vector3:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def dist(self, coor_i):
        return ((self.x - coor_i.x)**2+ (self.y - coor_i.y)**2)**0.5

    @property
    def norm(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def __sub__(self, coor_i):
        return Vector3(self.x - coor_i.x, self.y - coor_i.y)


class Node(Vector3):
    def __init__(self, id, x, y, rx, ry, rz, fx, fy, fz):
        super().__init__(x, y)
        self.id = id
        self.restrains = [rx, ry, rz]
        self.loads = [fx, fy, fz]

    def set_freedom_degrees(self, degree_i, degree_f):
        degree_i, degree_f, degree_x = self._set_freedom_degree(
            self.restrains[0], degree_i, degree_f
        )
        degree_i, degree_f, degree_y = self._set_freedom_degree(
            self.restrains[1], degree_i, degree_f
        )
        degree_i, degree_f, degree_z = self._set_freedom_degree(
            self.restrains[2], degree_i, degree_f
        )

        self.freedom_degrees = [degree_x, degree_y, degree_z]
        return degree_i, degree_f

    def _set_freedom_degree(self, restrain, degree_i, degree_f):
        if restrain:
            degree = degree_f
            degree_f -= 1
        else:
            degree = degree_i
            degree_i += 1

        return degree_i, degree_f, degree


class Element:
    def __init__(self, id, base, altura, n_i, n_f, E):
        self.id = id
        self.node_i = n_i
        self.node_f = n_f
        self.area = base * altura
        self.inercia = base * altura ** 3 / 12
        self.E = E  # Elastic module

    def _K_local(self):

        #    AE/L             0            0     AE/L             0           0
        #       0     12EI/L**3     6EI/L**2        0    -12EI/L**3    6EI/L**2
        #       0      6EI/L**2        4EI/L        0     -6EI/L**2       2EI/L
        #   -AE/L             0            0     AE/L             0           0
        #       0    -12EI/L**3    -6EI/L**2        0     12EI/L**3   -6EI/L**2
        #       0      6EI/L**2        2EI/L        0     -6EI/L**2       4EI/L

        K_local = np.zeros((6, 6))

        K_local[0, 0] = (self.area * self.E) / self.longitud
        K_local[0, 3] = -K_local[0, 0]
        K_local[3, 0] = -K_local[0, 0]
        K_local[3, 3] = K_local[0, 0]
        K_local[1, 1] = (12 * self.E * self.inercia) / (self.longitud ** 3)
        K_local[1, 4] = -K_local[1, 1]
        K_local[4, 1] = -K_local[1, 1]
        K_local[4, 4] = K_local[1, 1]
        K_local[1, 2] = (6 * self.E * self.inercia) / (self.longitud ** 2)
        K_local[2, 1] = K_local[1, 2]
        K_local[1, 5] = K_local[1, 2]
        K_local[2, 4] = -K_local[1, 2]
        K_local[4, 2] = -K_local[1, 2]
        K_local[5, 1] = K_local[1, 2]
        K_local[4, 5] = -K_local[1, 2]
        K_local[5, 4] = -K_local[1, 2]
        K_local[2, 2] = (4 * self.E * self.inercia) / self.longitud
        K_local[5, 5] = K_local[2, 2]
        K_local[5, 2] = (2 * self.E * self.inercia) / self.longitud
        K_local[2, 5] = K_local[5, 2]

        return K_local

    def _M_trans(self, nodes):

        #      CX    -CY      0      0       0      0
        #      CY     CX      0      0       0      0
        #       0      0      1      0       0      0
        #       0      0      0     CX     -CY      0
        #       0      0      0     CY      CX      0
        #       0      0      0      0       0      1

        M_trans = np.zeros((6, 6))

        node_i = nodes[self.node_i]
        node_f = nodes[self.node_f]

        M_trans[0, 0] = (node_f.x - node_i.x) / self.longitud
        M_trans[1, 0] = (node_f.y - node_i.y) / self.longitud
        M_trans[1, 1] = M_trans[0, 0]
        M_trans[0, 1] = -M_trans[1, 0]
        M_trans[2, 2] = 1
        M_trans[3, 3] = M_trans[0, 0]
        M_trans[3, 4] = -M_trans[1, 0]
        M_trans[4, 4] = M_trans[0, 0]
        M_trans[4, 3] = M_trans[1, 0]
        M_trans[5, 5] = 1

        return M_trans

    def K_global(self, nodes):
        K_local = self._K_local()
        M_trans = self._M_trans(nodes)
        return np.dot(np.dot(M_trans, K_local), np.transpose(M_trans))


class Estruct:
    def __init__(self, elements, nodes):
        self.type = "portico"

        self.nodes = [Node(**node) for node in nodes]
        self.elements = [Element(**element) for element in elements]
        self.n_freedom_degrees = len(self.nodes) * 3

        self.free_degrees = 0
        degree_max = self.n_freedom_degrees - 1

        for node in self.nodes:
            self.free_degrees, degree_max = node.set_freedom_degrees(
                self.free_degrees, degree_max
            )

        for element in self.elements:
            node_i = self.nodes[element.node_i]
            node_f = self.nodes[element.node_f]
            element.longitud = node_i.dist(node_f)

    def solve(self,method,tol,n_iter_max):
        KG = self.K_global()

        NGLL = self.free_degrees
        K0 = KG[:NGLL, :NGLL]
        K1 = KG[:NGLL, NGLL:]
        K2 = KG[NGLL:, :NGLL]
        K3 = KG[NGLL:, NGLL:]

        F = np.zeros(self.n_freedom_degrees)
        x0=np.zeros(6)
        for node in self.nodes:
            for i in range(3):
                F[node.freedom_degrees[i]] = node.loads[i]

        F0 = F[:NGLL]
       
        errores, x0, message, itera,num_iter = mathJacobiSeidel(x0, K0, F0, tol, n_iter_max, method)
        df=pd.DataFrame({"n iteraciones":num_iter,"Xi":itera,"errores":errores})
        df.to_excel("tabla_sist_ecuaciones.xlsx",index=False)
        # ivnersa
        U0 = np.dot(np.linalg.inv(K0), F0)



        return K0,U0
    def K_global(self):
        KG = np.zeros((self.n_freedom_degrees, self.n_freedom_degrees))

        for i in range(len(self.elements)):
            element = self.elements[i]
            K_global_elemento = element.K_global(self.nodes)
            node_i = self.nodes[element.node_i]
            node_f = self.nodes[element.node_f]
            freedom_dregrees = [*node_i.freedom_degrees, *node_f.freedom_degrees]

            for j in range(6):
                row = freedom_dregrees[j]
                for k in range(6):
                    col = freedom_dregrees[k]
                    KG[row, col] += K_global_elemento[k, j]

        return KG


nodes = [
    {
        "id": 0,
        "x": 0,
        "y": 0,
        "rx": 1,
        "ry": 1,
        "rz": 1,
        "fx": 0,
        "fy": 0,
        "fz": 0,
    },
    {
        "id": 1,
        "x": 0,
        "y": 10,
        "rx": 0,
        "ry": 0,
        "rz": 0,
        "fx": 1105,
        "fy": 1100,
        "fz": 0,
    },
    {
        "id": 2,
        "x": 10,
        "y": 10,
        "rx": 0,
        "ry": 0,
        "rz": 0,
        "fx": 0,
        "fy": 1100,
        "fz": 0,
    },
    {
        "id": 3,
        "x": 10,
        "y": 0,
        "rx": 1,
        "ry": 1,
        "rz": 1,
        "fx": 0,
        "fy": 0,
        "fz": 0,
    },
]

eles = [
    {"id": 0, "n_i": 0, "n_f": 1, "base": 40, "altura": 40, "E": 200000},
    {"id": 1, "n_i": 1, "n_f": 2, "base": 40, "altura": 40, "E": 200000},
    {"id": 2, "n_i": 2, "n_f": 3, "base": 40, "altura": 40, "E": 200000},
]

estr = Estruct(eles, nodes)


tol = float(input('ingrese la tolerancia: '))
n_iter = float(input('ingrese el numero de iteraciones maximas: '))
method = str(input('ingrese el metodo que desea utilizar: '))
        

matriz_rigidez,desplazamineto=estr.solve(method,tol,n_iter)
print(desplazamineto)
print(matriz_rigidez)