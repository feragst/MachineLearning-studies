import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede():
    classificador = Sequential()

    #camadas oculta1
    classificador.add(Dense(units = 16, activation = 'relu', 
                            kernel_initializer = 'random_uniform', input_dim = 30))
    
    #dropout da camada oculta1
    classificador.add(Dropout(0.2))
    
    #camadaoculta2
    classificador.add(Dense(units = 16, activation = 'relu', 
                            kernel_initializer = 'random_uniform'))
    
    #dropout da camada oculta2
    classificador.add(Dropout(0.2))

    #camada saida
    classificador.add(Dense(units = 1, activation = 'sigmoid'))

    from keras.optimizers import Adam

    otimizador = Adam(learning_rate= 0.001, clipvalue= 0.5)

    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size= 10)

resultados = cross_val_score(estimator= classificador,
                             X= previsores,
                             y= classe,
                             cv = 10, scoring = 'accuracy')

media= resultados.mean()
desvio= resultados.std()