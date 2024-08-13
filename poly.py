import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data=pd.read_csv('position.csv')
x=data.drop(columns=['Position', 'Salary'])
y=data['Salary']

pf=PolynomialFeatures(degree=6)
lr=LinearRegression()

polyx=pf.fit_transform(x)
model=lr.fit(polyx,y)

plt.scatter(x,y,c='r')
plt.plot(x, lr.predict(pf.fit_transform(x)), c='b')
plt.title('polynomial regression')
plt.xlabel('position label')
plt.ylabel('salary')
plt.show()

new_position = np.array([[6]])
new_position_poly = pf.transform(new_position)
new_salary_prediction = lr.predict(new_position_poly)
print(new_salary_prediction)