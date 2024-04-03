import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


base = pd.read_csv('iriss.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

print("Dimensões dos previsores:", previsores.shape)
print("Dimensões da classe:", classe.shape)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = to_categorical(classe, num_classes=3)

def criarRede():
    classificador = Sequential()
    
    classificador.add(Dense(units = 4, activation= 'relu', input_dim = 4))
    classificador.add(Dense(units= 4, activation= 'relu'))
    classificador.add(Dense(units= 3, activation= 'softmax'))
    

    
    classificador.compile(optimizer = 'adam', loss= 'categorical_crossentropy', 
                         metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criarRede, epochs = 150, batch_size = 10)

resultados = cross_val_score(estimator=classificador, X=previsores, y=classe_dummy, 
                             cv=10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()