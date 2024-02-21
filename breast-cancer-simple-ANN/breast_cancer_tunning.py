import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()

    #camadas oculta1
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer, input_dim = 30))
    
    #dropout da camada oculta1
    classificador.add(Dropout(0.2))
    
    #camadaoculta2
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer))
    
    #dropout da camada oculta2
    classificador.add(Dropout(0.2))

    #camada saida
    classificador.add(Dense(units = 1, activation = 'sigmoid'))



    classificador.compile(optimizer = optimizer, loss = loss,
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size':[10,30],
              'epochs': [50, 100],
              'model__optimizer': ['adam', 'sgd'],
              'model__loss': ['binary_crossentropy', 'hinge',],
              'model__kernel_initializer': ['random_uniform', 'normal'],
              'model__activation': ['relu', 'tanh'],
              'model__neurons': [16, 8]}

grid_search = GridSearchCV(estimator= classificador,
                           param_grid= parametros,
                           scoring= 'accuracy',
                           cv= 5)

grid_search = grid_search.fit(previsores, classe)

melhores_parametros= grid_search.best_params_
melhor_precisao= grid_search.best_score_