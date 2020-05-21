# -*- coding: utf-8 -*-

import pandas
import numpy
base = pandas.read_csv('Validations/StratifieldKFold/credit_dt.csv')

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# from sklearn.model_selection import train_test_split
# previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25, random_state = 0)

# from sklearn.tree import DecisionTreeClassifier, export
# classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classificador.fit(previsores_treinamento, classe_treinamento)

# previsoes = classificador.predict(previsores_teste)

# from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=30)
resultados = []

for indice_treinamento, indice_teste in kfold.split(previsores, numpy.zeros(shape=(previsores.shape[0], 1))):
    classificador = GaussianNB()
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    resultados.append(precisao)

resultados = numpy.array(resultados)
print(resultados.mean())

