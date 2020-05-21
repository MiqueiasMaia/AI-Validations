# -*- coding: utf-8 -*-

import pandas

base = pandas.read_csv('Validations/Consufion-matrix/risco_credito.csv')
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

from sklearn.naive_bayes import GaussianNB
import numpy as np 
a = np.zeros(5)

# print(previsores.shape)
# print(previsores.shape[0])

b = np.zeros(shape=(previsores.shape[0],1))
# print(b)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
resultados = []
matrizes = []
for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(previsores.shape[0], 1))):
    # print('Indice treinamento: ', indice_treinamento, 'Indice teste: ', indice_teste)
    classificador = GaussianNB()
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    resultados.append(precisao)
    matrizes.append(confusion_matrix(classe[indice_teste],previsoes))

resultados = np.asarray(resultados)
matriz_final = np.mean(matrizes,axis=0)
# print(resultados)
# print(resultados.mean())
# print(resultados.std())
print(matriz_final)