import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
print("Linear Regression")
print("MSE =", mean_squared_error(y_test, y_pred_linear))
print("R2 =", r2_score(y_test, y_pred_linear))

ridge = Ridge()
param_grid = {'alpha':[0.01,0.1,1,10,100]}
ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
best_ridge = ridge_cv.best_estimator_
y_pred_ridge = best_ridge.predict(X_test_scaled)
print("Best Alpha for Ridge:", ridge_cv.best_params_['alpha'])
print("MSE =", mean_squared_error(y_test, y_pred_ridge))
print("R2 =", r2_score(y_test, y_pred_ridge))

lasso = Lasso(max_iter=10000)
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
lasso_cv.fit(X_train_scaled, y_train)
best_lasso = lasso_cv.best_estimator_
y_pred_lasso = best_lasso.predict(X_test_scaled)
print("Best Alpha for Lasso:", lasso_cv.best_params_['alpha'])
print("MSE =", mean_squared_error(y_test, y_pred_lasso))
print("R2 =", r2_score(y_test, y_pred_lasso))

results = pd.DataFrame({
    'Model':['Linear Regression','Ridge Regression','Lasso Regression'],
    'MSE':[mean_squared_error(y_test,y_pred_linear),
           mean_squared_error(y_test,y_pred_ridge),
           mean_squared_error(y_test,y_pred_lasso)],
    'R2 Score':[r2_score(y_test,y_pred_linear),
                r2_score(y_test,y_pred_ridge),
                r2_score(y_test,y_pred_lasso)]
})
print(results)

plt.figure(figsize=(8,5))
models = ['Linear','Ridge','Lasso']
r2_scores = [
    r2_score(y_test,y_pred_linear),
    r2_score(y_test,y_pred_ridge),
    r2_score(y_test,y_pred_lasso)
]
plt.bar(models, r2_scores)
plt.ylabel("R² Score")
plt.title("Model Comparison")
plt.show()
