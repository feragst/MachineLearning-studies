import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential()

#camadas oculta1
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))

#dropout da camada oculta1
classificador.add(Dropout(0.2))

#camadaoculta2
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))

#dropout da camada oculta2
classificador.add(Dropout(0.2))

#camada saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))



classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size= 10, epochs= 100)

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.25, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500,
                  145.72, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5,
                  2018, 0.014, 0.185, 0.84, 158, 0.368]])

previsao= classificador.predict(novo)
previsao= (previsao > 0.5)