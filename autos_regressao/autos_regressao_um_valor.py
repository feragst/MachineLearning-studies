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

#media de preços
base.price.mean()
base = base[base.price >10]

#inconsistent2
i2 = base.loc[base.price>350000]
base = base[base.price < 350000]

