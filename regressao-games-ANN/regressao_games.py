import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input 
from keras.models import Model

base = pd.read_csv('games.csv')

#pre processing

base = base.drop('Other_Sales', axis =1)
base = base.drop('Global_Sales', axis =1)
base = base.drop('Developer', axis =1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name

base = base.drop('Name', axis =1)

previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values

venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le_previsores = LabelEncoder()

#tratamento dos dados categoricos para numericos
previsores[:, 0] = le_previsores.fit_transform(previsores[:,0])
previsores[:, 2] = le_previsores.fit_transform(previsores[:,1])
previsores[:, 3] = le_previsores.fit_transform(previsores[:,3])
previsores[:, 8] = le_previsores.fit_transform(previsores[:,8])

categorical_columns = [0, 2, 3, 8]
column_transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')
previsores = column_transformer.fit_transform(previsores).toarray()

#estrutura da rede neural

camada_entrada = Input(shape =(70, ))
camada_oculta1 = Dense(units = 32, activation = 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 32, activation = 'sigmoid')(camada_oculta1)
camada_saida1 = Dense(units = 1, activation = 'linear')(camada_oculta2)
camada_saida2 = Dense(units = 1, activation = 'linear')(camada_oculta2)
camada_saida3 = Dense(units = 1, activation = 'linear')(camada_oculta2)

regressor = Model(inputs = camada_entrada,
                  outputs = [camada_saida1, camada_saida2, camada_saida3])

regressor.compile(optimizer = 'adam',
                  loss = 'mse')

regressor.fit(previsores, [venda_na, venda_eu, venda_jp],
              epochs = 5000, batch_size = 100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)