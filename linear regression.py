import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# Load the California Housing dataset
california = fetch_california_housing()

# Separate the features and target
x = california.data
y = california.target

lreg= LinearRegression()

plt.scatter(x.T[0],y)
plt.show()

xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=0.2)

model=lreg.fit(xtrain, ytrain)
prediction=model.predict(xtest)

print("predictioons", prediction)
print("accuracy score", lreg.score(x,y))
print("coefficieint", lreg.coef_)