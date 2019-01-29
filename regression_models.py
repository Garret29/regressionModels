from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams.update({'font.size': 12})

boston = load_boston()
boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_data['Price'] = boston.target

X = boston_data['RM'].values[:, np.newaxis]
Y = boston_data['Price'].values

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

for key in models:
    print(key, ": ", "Współczynnik determinacji dla zbioru testowego: ", models[key].score(X_test, Y_test),
          ", Współczynnik determinacji dla zbioru uczącego: ", models[key].score(X_train, Y_train),
          ", Współczynnik 'a' prostej regresji: ", models[key].coef_)

plt.scatter(X_train, Y_train, color='r')
plt.plot(X_train, linearRegression.predict(X_train), color='k')
plt.ylabel('Mediana ceny mieszkania')
plt.xlabel('Liczba pokoi')
plt.show()

plt.scatter(X_train, Y_train, color='g')
plt.plot(X_train, lassoRegression.predict(X_train), color='k')
plt.ylabel('Mediana ceny mieszkania')
plt.xlabel('Liczba pokoi')
plt.show()

plt.scatter(X_train, Y_train, color='b')
plt.plot(X_train, ridgeRegression.predict(X_train), color='k')
plt.ylabel('Mediana ceny mieszkania')
plt.xlabel('Liczba pokoi')
plt.show()

plt.scatter(X_train, Y_train, color='y')
plt.plot(X_train, elasticNetRegression.predict(X_train), color='k')
plt.ylabel('Mediana ceny mieszkania')
plt.xlabel('Liczba pokoi')
plt.show()

alphas = []
ridgeCoefs = []
lassoCoefs = []
elasticNetCoefs = []

for alpha in range(1, 51):

    newAlpha = alpha/10
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

    models = {"linearRegression": linearRegression, "ridgeRegression": ridgeRegression,
              "lassoRegression": lassoRegression,
              "elasticNetRegression": elasticNetRegression}

plt.plot(alphas, ridgeCoefs, color="r", label="ridge")
plt.plot(alphas, lassoCoefs, color="g", label="lasso")
plt.plot(alphas, elasticNetCoefs, color="b", label="elastic net")
plt.ylabel('Współczynnik nachylenia')
plt.xlabel('Alfa')
plt.legend()
plt.show()
