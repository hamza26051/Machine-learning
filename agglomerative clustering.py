import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_squared_error
import scipy.cluster.hierarchy as sch
from sklearn.svm import SVR
import matplotlib.pyplot as plt


data=pd.read_csv('/Mall_Customers.csv')
x=data.drop(columns=['CustomerID'])
categorical=['Genre']
preprocess=ColumnTransformer(transformers=[('encode', OneHotEncoder(),categorical) ], remainder='drop')
x=preprocess.fit_transform(x)
demoon=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('euclidean distance')
plt.show()
model=AgglomerativeClustering(n_clusters=1, affinity='euclidean', linkage='ward')
hc=model.fit_predict(x)
print(hc)
categoricalcols=['State']
numericols=['R&D Spend','Administration','Marketing Spend']

preprocess=ColumnTransformer(transformers=[
    ('encode', OneHotEncoder(), categoricalcols),
    ('scale', StandardScaler(), numericols),

], remainder='drop')

x=preprocess.fit_transform(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=42)


regressor=SVR(kernel='rbf')

model=regressor.fit(xtrain, ytrain)
predictions=model.predict(xtest)
error=mean_squared_error(ytest, predictions)
print(error)

for actual, predicted in zip(ytest, predictions):
    print(f"Actual value: {actual}, Predicted value: {predicted}")