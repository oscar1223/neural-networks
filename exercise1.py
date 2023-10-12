# Script for exercises.

'''
Primer ejercicio con Machine Learning para pasar grados Celsius a Fahrenheit.
Sin saber la formula.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# inputs con respectivos outputs
celsius = np.array([-40, 10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Creamos capa y modelo
capa1 = tf.keras.layers.Dense(units=3, input_shape=[1])
capa2 = tf.keras.layers.Dense(units=3)
capa3salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa1, capa2, capa3salida])

# Compilamos el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenamos el modelo
print('Comienza entrenamiento...')
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print('Modelo entrenado!')

# Visualizamos la funcion de perdida.
plt.xlabel('# Epoca')
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history['loss'])
plt.show()

# Intento de predicción fallido 100c = 212f
print('Hagamos una predicción.')
resultado = modelo.predict([100.0])
print('El resultado es: '+str(resultado)+' grados Fahreinheit.')

# Imprimimos los valores intarnas del modelo.
print('Variables internas del modelo.')
print(capa1.get_weights())
print(capa2.get_weights())
print(capa3salida.get_weights())