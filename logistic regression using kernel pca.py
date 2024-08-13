import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


data=pd.read_csv("wine.csv")
x=data.drop(columns=['Wine'])
y=data['Wine']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=42)

ss=StandardScaler()
xtrain=ss.fit_transform(xtrain)
xtest=ss.transform(xtest)
print(xtrain)

kpca=KernelPCA(n_components=2, kernel='rbf')
xtrain=kpca.fit_transform(xtrain)
xtest=kpca.transform(xtest)
print(xtrain)

Lr=LogisticRegression(random_state=42)
Lr.fit(xtrain, ytrain)

pred=Lr.predict(xtest)
accuracy=accuracy_score(ytest, pred)
confusion=confusion_matrix(ytest, pred)
print(accuracy)
print(confusion)