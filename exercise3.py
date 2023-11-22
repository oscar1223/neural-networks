'''Recursive Neural Network'''

''' La estructura de una red neuronal artificial es relativamente simple y se refiere principalmente a la multiplicación de matrices. 
Durante el primer paso, las entradas se multiplican por pesos inicialmente aleatorios, y sesgo, 
transformados con una función de activación y los valores de salida se utilizan para hacer una predicción. 
Este paso da una idea de lo lejos que está la red de la realidad. '''

import numpy as np
from keras.layers import Input, Dense, SimpleRNN
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as K

nombres = open('nombres_dinosaurios.txt','r').read()
nombres = nombres.lower()

alfabeto = list(set(nombres))
tam_datos, tam_alfabeto = len(nombres), len(alfabeto)

car_a_ind = { car:ind for ind,car in enumerate(sorted(alfabeto))}

ind_a_car = { ind:car for ind,car in enumerate(sorted(alfabeto))}

n_a = 25    # Número de unidades en la capa oculta
entrada  = Input(shape=(None,tam_alfabeto))
a0 = Input(shape=(n_a,))

celda_recurrente = SimpleRNN(n_a, activation='tanh', return_state = True)

capa_salida = Dense(tam_alfabeto, activation='softmax')

hs, _ = celda_recurrente(entrada, initial_state=a0)
salida = []
salida.append(capa_salida(hs))

modelo = Model([entrada,a0],salida)
opt = SGD(lr=0.0005)
modelo.compile(optimizer=opt, loss='categorical_crossentropy')

with open("nombres_dinosaurios.txt") as f:
    ejemplos = f.readlines()
ejemplos = [x.lower().strip() for x in ejemplos]
np.random.shuffle(ejemplos)

def train_generator():
    while True:
        # Tomar un ejemplo aleatorio
        ejemplo = ejemplos[np.random.randint(0,len(ejemplos))]

        # Convertir el ejemplo a representación numérica
        X = [None] + [car_a_ind[c] for c in ejemplo]

        # Crear "Y", resultado de desplazar "X" un caracter a la derecha
        Y = X[1:] + [car_a_ind['\n']]

        # Representar "X" y "Y" en formato one-hot
        x = np.zeros((len(X),1,tam_alfabeto))
        onehot = to_categorical(X[1:],tam_alfabeto).reshape(len(X)-1,1,tam_alfabeto)
        x[1:,:,:] = onehot
        y = to_categorical(Y,tam_alfabeto).reshape(len(X),tam_alfabeto)

        # Activación inicial (matriz de ceros)
        a = np.zeros((len(X), n_a))

        yield [x, a, y]

 a = np.zeros((len(X)), n_a)
BATCH_SIZE = 80			# Número de ejemplos de entrenamiento a usar en cada iteración
NITS = 10000			# Número de iteraciones

for j in range(NITS):
    historia = modelo.fit_generator(train_generator(), steps_per_epoch=BATCH_SIZE, epochs=1, verbose=0)

    # Imprimir evolución del entrenamiento cada 1000 iteraciones
    if j%1000 == 0:
        print('\nIteración: %d, Error: %f' % (j, historia.history['loss'][0]) + '\n')


def generar_nombre(modelo,car_a_num,tam_alfabeto,n_a):
    # Inicializar x y a con ceros
    x = np.zeros((1,1,tam_alfabeto,))
    a = np.zeros((1, n_a))

    # Nombre generado y caracter de fin de linea
    nombre_generado = ''
    fin_linea = '\n'
    car = -1

    # Iterar sobre el modelo y generar predicción hasta tanto no se alcance
    # "fin_linea" o el nombre generado llegue a los 50 caracteres
    contador = 0
    while (car != fin_linea and contador != 50):
          # Generar predicción usando la celda RNN
          a, _ = celda_recurrente(K.constant(x), initial_state=K.constant(a))
          y = capa_salida(a)
          prediccion = K.eval(y)

          # Escoger aleatoriamente un elemento de la predicción (el elemento con
          # con probabilidad más alta tendrá más opciones de ser seleccionado)
          ix = np.random.choice(list(range(tam_alfabeto)),p=prediccion.ravel())

          # Convertir el elemento seleccionado a caracter y añadirlo al nombre generado
          car = ind_a_car[ix]
          nombre_generado += car

          # Crear x_(t+1) = y_t, y a_t = a_(t-1)
          x = to_categorical(ix,tam_alfabeto).reshape(1,1,tam_alfabeto)
          a = K.eval(a)

          # Actualizar contador y continuar
          contador += 1

          # Agregar fin de línea al nombre generado en caso de tener más de 50 caracteres
          if (contador == 50):
            nombre_generado += '\n'

    print(nombre_generado)

x = np.zeros((1,1,tam_alfabeto,))
a = np.zeros((1, n_a))

while (car != fin_linea and contador != 50):
    # Generar predicción usando la celda RNN
    a, _ = celda_recurrente(K.constant(x), initial_state=K.constant(a))
    y = capa_salida(a)
    prediccion = K.eval(y)

ix = np.random.choice(list(range(tam_alfabeto)),p=prediccion.ravel())
car = ind_a_car[ix]
nombre_generado += car

# Crear x_(t+1) = y_t, y a_t = a_(t-1)
x = to_categorical(ix, tam_alfabeto).reshape(1, 1, tam_alfabeto)
a = K.eval(a)

# Actualizar contador y continuar
contador += 1

# Agregar fin de línea al nombre generado en caso de tener más de 50 caracteres
if (contador == 50):
    nombre_generado += '\n'

print(nombre_generado)

for i in range(100):
    generar_nombre(modelo,car_a_ind,tam_alfabeto,n_a)