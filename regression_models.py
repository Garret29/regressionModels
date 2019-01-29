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

X = boston_data['NOX'].values[:, np.newaxis]
Y = boston_data['Price'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

linearRegression = LinearRegression()
linearRegression.fit(X_train, Y_train)

alpha = 0.15

ridgeRegression = Ridge(alpha=alpha)
ridgeRegression.fit(X_train, Y_train)

lassoRegression = Lasso(alpha=alpha)
lassoRegression.fit(X_train, Y_train)

elasticNetRegression = ElasticNet(alpha=alpha, l1_ratio=0.8)
elasticNetRegression.fit(X_train, Y_train)

plt.scatter(X_train, Y_train, color='r')
plt.plot(X_train, linearRegression.predict(X_train), color='k')
plt.show()

plt.scatter(X_train, Y_train, color='g')
plt.plot(X_train, lassoRegression.predict(X_train), color='k')
plt.show()

plt.scatter(X_train, Y_train, color='b')
plt.plot(X_train, ridgeRegression.predict(X_train), color='k')
plt.show()

plt.scatter(X_train, Y_train, color='y')
plt.plot(X_train, elasticNetRegression.predict(X_train), color='k')
plt.show()