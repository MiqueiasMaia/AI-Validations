# -*- coding: utf-8 -*-

import pandas

base = pandas.read_csv('Validations/Cross-validation/risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
previsores[:,0] = labelEncoder.fit_transform(previsores[:,0])
previsores[:,1] = labelEncoder.fit_transform(previsores[:,1])
previsores[:,2] = labelEncoder.fit_transform(previsores[:,2])
previsores[:,3] = labelEncoder.fit_transform(previsores[:,3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
resultados = cross_val_score(classificador, previsores, classe, cv = 3)
print(resultados.mean())
print(resultados.std())