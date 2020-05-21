# -*- coding: utf-8 -*-

import pandas
import numpy
base = pandas.read_csv('Validations/30-tests/credit_dt.csv')

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

from sklearn.naive_bayes import GaussianNB #ok
from sklearn.tree import DecisionTreeClassifier #ok
from sklearn.ensemble import RandomForestClassifier #ok
from sklearn.neighbors import KNeighborsClassifier #ok
from sklearn.linear_model import LogisticRegression #ok
from sklearn.svm import SVC #ok
from sklearn.neural_network import MLPClassifier #ok

import Orange


resultados30 = []

for i in range(30):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores, numpy.zeros(shape=(previsores.shape[0], 1))):
        
        # classificador = GaussianNB()
        # classificador = DecisionTreeClassifier()
        # classificador = LogisticRegression()

        # classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
        # classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0, gamma='scale')
        # classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        classificador = MLPClassifier(verbose = True, max_iter = 1000,
                              tol = 0.000010, solver='adam',
                              hidden_layer_sizes=(100), activation = 'relu',
                              batch_size=200, learning_rate_init=0.001)

        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)

    resultados1 = numpy.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)

resultados30 = numpy.asarray(resultados30)
for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.',','))

# resultados = numpy.array(resultados)
# print(resultados.mean())

