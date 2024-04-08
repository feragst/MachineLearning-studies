import pandas as pd

base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')


#pre processing the database

base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)

base['name'].value_counts()
base = base.drop('name', axis=1)
base['seller'].value_counts()
base = base.drop('seller', axis=1)
base['offerType'].value_counts()
base = base.drop('offerType', axis=1)

#inconsistent1
i1= base.loc[base.price < 10]
base = base[base.price >10]

#media de preços
base.price.mean()

#inconsistent2
i2 = base.loc[base.price>350000]
base = base[base.price < 350000]


base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() #golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() #benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() #nein

valores ={'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 
          'fuelType': 'benzin',
          'notRepairedDamage': 'nein'}

base = base.fillna(value = valores)
previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

