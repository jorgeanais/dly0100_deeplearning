"""
El perceptron de Rosenblatt
Autor: Jorge Anais
Fecha: 18 de marzo de 2023
"""


import numpy as np
import matplotlib.pyplot as plt

# PARAMETROS
UMBRAL = 0.5  # Umbral de activación
TASA_APRENDIZAJE = 0.1  # Hiperpárametro
PESOS = [0.0, 0.0, 0.0]  # valores iniciales de [w1, w2, b]
CONJUNTO_ENTRENAMIENTO = [
    # ((x1, x2, 1.), salida_deseada)
    ((0.0, 0.0, 1.0), 1.0),
    ((0.0, 1.0, 1.0), 0.0),
    ((1.0, 0.0, 1.0), 0.0),
    ((1.0, 1.0, 1.0), 1.0),
]


def producto(entradas, pesos):
    """
    Realiza la operación de producto punto:
      w1*x1 + w2*x2 + b*1.0
    """
    return sum(entrada * peso for entrada, peso in zip(entradas, pesos))


def grafica(conjunto_de_entrenamiento, pesos):
    """
    Grafica el conjunto de entrenamiento y la linea de separacion
    """

    # Creamos la linea de separacion
    x1 = np.linspace(-0.2, 1.2, 100)
    x2 = UMBRAL / pesos[1] - pesos[0] / pesos[1] * x1 - pesos[2] / pesos[1]

    # Graficamos la linea de separacion
    plt.plot(x1, x2, label=f"w1={pesos[0]:.2f} w2={pesos[1]:.2f} b={pesos[2]:.2f}")

    # Graficamos los casos
    for muestra in conjunto_de_entrenamiento:
        plt.scatter(
            muestra[0][0],
            muestra[0][1],
            c="g" if muestra[1] == 1 else "r",
            marker="o" if muestra[1] == 1 else "x",
        )

    # Ajustamos el rango a mostrar
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.legend(loc="upper left")
    plt.show()


while True:
    print("-" * 60)
    contador_de_errores = 0

    for vector_de_entrada, salida_deseada in CONJUNTO_ENTRENAMIENTO:
        resultado = producto(vector_de_entrada, PESOS) >= UMBRAL
        error = salida_deseada - resultado
        if error != 0:
            contador_de_errores += 1
            for indice, valor in enumerate(vector_de_entrada):
                PESOS[indice] += TASA_APRENDIZAJE * error * valor

    print(PESOS)

    if contador_de_errores == 0:
        break

grafica(CONJUNTO_ENTRENAMIENTO, PESOS)

