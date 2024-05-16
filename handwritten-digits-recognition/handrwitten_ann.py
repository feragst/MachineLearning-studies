import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.layers import BatchNormalization

(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()
plt.imshow(x_treino[2], cmap = 'gray') 
plt.title('Classe ' + str(y_treino[2]))

#covertendo de int8 para float32, melhorando assim a capacidade da rede
x_treino = x_treino.astype('float32')
x_teste = x_teste.astype('float32')

#normalizacao dos dados
x_treino /= 255
x_teste /= 255

classe_treinamento = to_categorical(y_treino, 10)
classe_teste = to_categorical(y_teste, 10)

#etapa 1 operador de convolucao
classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1),
                          activation = 'relu') )

#etapa 2 pooling
classificador.add(MaxPooling2D(pool_size = (2,2)))

#etapa 3 flattening
classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',metrics =['accuracy'])
classificador.fit(x_treino, classe_treinamento,
                  batch_size = 128, epochs = 5,
                  validation_data = (x_teste, classe_teste))

resultado = classificador.evaluate(x_teste, classe_teste)