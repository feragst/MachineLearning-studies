import pandas as pd

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras 
from keras.models import Sequential
from keras.layers import Dense 



classificador = Sequential()

#camadas ocultas
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))

#camada saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

from keras.optimizers import Adam

otimizador = Adam(learning_rate= 0.001, clipvalue= 0.5)

classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

#classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs=100)


#camada de entrada pra oculta1
pesos0= classificador.layers[0].get_weights()
print(pesos0)

#camada oculta1 pra oculta2
pesos1= classificador.layers[1].get_weights()
print(pesos1)

#camada oculta2 pra saida
pesos2= classificador.layers[2].get_weights()
print(pesos2)


previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes >0.5)

#from sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)

matriz = confusion_matrix(classe_teste, previsoes)

#from keras
resultado = classificador.evaluate(previsores_teste, classe_teste)