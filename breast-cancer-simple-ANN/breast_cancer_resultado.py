import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')

estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_breast.h5')

novo= np.array([[]])
previsao= classificador.perdict(novo)
previsao= (previsao > 0.5)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador.compile(loss='binary_crossentropy', optimizer= 'adam', metrics= ['binary_accuracy '])

resultado = classificador.evaluate(previsores, classe)



