import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical 

(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()
plt.imshow(x_treino[2], cmap = 'gray') 
plt.title('Classe ' + str(y_treino[2]))

#covertendo de int8 para float32, melhorando assim a capacidade da rede
x_treino = x_treino.astype('float32')
x_teste = x_teste.astype('float32')

x_treino /= 255
x_teste /= 255

classe_treinamento = to_categorical(y_treino, 10)
classe_teste = to_categorical(y_teste, 10)
