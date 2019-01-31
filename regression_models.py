from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams.update({'font.size': 12})

diabetes = load_diabetes()

diabetes_data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_data['progression'] = diabetes.target

X = diabetes_data.drop('progression', axis=1)
X = X[['bmi']]
Y = diabetes_data['progression']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

linearRegression = LinearRegression()
linearRegression.fit(X_train, Y_train)

alpha = 0.15

ridgeRegression = Ridge(alpha=alpha)
ridgeRegression.fit(X_train, Y_train)

lassoRegression = Lasso(alpha=alpha)
lassoRegression.fit(X_train, Y_train)

elasticNetRegression = ElasticNet(alpha=alpha, l1_ratio=0.8)
elasticNetRegression.fit(X_train, Y_train)

models = {"linearRegression": linearRegression, "ridgeRegression": ridgeRegression, "lassoRegression": lassoRegression,
          "elasticNetRegression": elasticNetRegression}


print("alpha =", alpha, "l1_ratio =", 0.8)

for key in models:
    print(key, ": ", "Współczynnik determinacji dla zbioru testowego: ", models[key].score(X_test, Y_test),
          ", Współczynnik determinacji dla zbioru uczącego: ", models[key].score(X_train, Y_train),
          ", Współczynnik 'a' prostej regresji: ", models[key].coef_)
try:
    plt.scatter(X, Y, color='r')
    plt.plot(X, linearRegression.predict(X), color='k')
    plt.ylabel('Miara rozwoju cukrzycy')
    plt.xlabel('Zestandaryzowane BMI')
    plt.show()

    plt.scatter(X, Y, color='g')
    plt.plot(X, lassoRegression.predict(X), color='k')
    plt.ylabel('Miara rozwoju cukrzycy')
    plt.xlabel('Zestandaryzowane BMI')
    plt.show()

    plt.scatter(X, Y, color='b')
    plt.plot(X, ridgeRegression.predict(X), color='k')
    plt.ylabel('Miara rozwoju cukrzycy')
    plt.xlabel('Zestandaryzowane BMI')
    plt.show()

    plt.scatter(X, Y, color='y')
    plt.plot(X, elasticNetRegression.predict(X), color='k')
    plt.ylabel('Miara rozwoju cukrzycy')
    plt.xlabel('Zestandaryzowane BMI')
    plt.show()

    alphas = []
    ridgeCoefs = []
    lassoCoefs = []
    elasticNetCoefs = []

    for alpha in range(1, 51):
        newAlpha = alpha / 10
        alphas.append(newAlpha)

        ridgeRegression = Ridge(alpha=newAlpha)
        ridgeRegression.fit(X_train, Y_train)

        ridgeCoefs.append(ridgeRegression.coef_[0])

        lassoRegression = Lasso(alpha=newAlpha)
        lassoRegression.fit(X_train, Y_train)

        lassoCoefs.append(lassoRegression.coef_[0])

        elasticNetRegression = ElasticNet(alpha=newAlpha, l1_ratio=0.8)
        elasticNetRegression.fit(X_train, Y_train)

        elasticNetCoefs.append(elasticNetRegression.coef_[0])

    plt.plot(alphas, ridgeCoefs, color="r", label="ridge")
    plt.plot(alphas, lassoCoefs, color="g", label="lasso")
    plt.plot(alphas, elasticNetCoefs, color="b", label="elastic net")
    plt.ylabel('Współczynnik nachylenia')
    plt.xlabel('Alfa')
    plt.legend()
    plt.show()
except:
    print("Nie udało sie narysować wykresów. (Wykresy są przeznaczone tylko dla regresji prostej)")
