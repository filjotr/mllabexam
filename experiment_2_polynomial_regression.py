import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)
data = data[['mpg','displacement']]
data.dropna(inplace=True)

X = data[['displacement']]
y = data['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

print("Linear Regression")
print("MSE =", mean_squared_error(y_test, y_pred_linear))
print("R2 =", r2_score(y_test, y_pred_linear))

degrees = [2,3,4]
for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    print("\nDegree =", degree)
    print("MSE =", mean_squared_error(y_test, y_pred))
    print("R2 =", r2_score(y_test, y_pred))

x_range = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(x_range, linear_model.predict(x_range),
         color='red', label='Linear Regression')

colors = ['green','orange','purple']
for degree, color in zip([2,3,4], colors):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    plt.plot(x_range, model.predict(poly.transform(x_range)),
             color=color, label=f'Polynomial Degree {degree}')

plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.legend()
plt.grid()
plt.show()
