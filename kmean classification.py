import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


bc=load_breast_cancer()

x=scale(bc.data)
y=bc.target

xtrain, xtest,ytrain, ytest= train_test_split(x,y,test_size=0.2)

model=KMeans(n_clusters=2, random_state=0)
model.fit(xtrain)
predictions=model.predict(xtest)

label=model.labels_
print("labels", label)
print("predictions", predictions)
print("accuracy", accuracy_score(ytest, predictions))
print("actual data", ytest)
print(pd.crosstab(ytrain, label))