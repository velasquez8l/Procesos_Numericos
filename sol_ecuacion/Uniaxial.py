import funciones as fn


def main():

    #Datos
   Momentos, cargas, Mu, Pu, cuantia = fn.info_grafica()

   # MUESTRA CUANTIA EN PANTALLA
   print(f"La cuantia es de {cuantia*100}%")

   # Graficar
   fn.graficar(Momentos, cargas, Mu, Pu)


if __name__ == "__main__":
    main()
