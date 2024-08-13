import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data=pd.read_csv('/social.csv')
print(data.info())
x= data.drop(columns=(['User ID',"Purchased"]))
y=data['Purchased']
categorical_data=['Gender']
numericaldata=['EstimatedSalary','Age']

preprocess=ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_data),
                                          ('num', StandardScaler(), numericaldata)], remainder='passthrough')

xp=preprocess.fit_transform(x)
xtrain, xtest, ytrain, ytest=train_test_split(xp,y,test_size=0.2,random_state=0)
reg=KNeighborsClassifier()
reg.fit(xtrain,ytrain)
ypred=reg.predict(xtest)
print(mean_squared_error(ytest,ypred))
print(r2_score(ytest,ypred))
