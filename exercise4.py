''' LONG SHORT TERM MEMORY '''
# Stock market prediction bot.

import numpy as np
from sklearn.preprocessing import MinMaxScaler
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Preparing the data
dataset = pd.read_csv('files/AAPL.csv')

set_entrenamiento = dataset[:'2016'].iloc[:, 1:2]
set_validacion = dataset['2017':].iloc[:, 1:2]

sc = MinMaxScaler(feature_range=(0, 1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

time_step = 60
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step, m):
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Implement the model in Keras.
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 50

modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train, Y_train, epochs=20, batch_size=32)

x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step, len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Prediction
prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)