import pandas as pd
import numpy as np
from sklearn.preprocessing  import LabelEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data= pd.read_csv("car.data")

x= data[['buying',
         'maint',
         'safety']].values
y=data[['class']]
le = LabelEncoder()

for i in range(len(x[0])):
    x[:,i]=le.fit_transform(x[:,i])


dictionary={
    "unacc":0,
    "acc":1,
    "good":2,
    "vgood":3,
}
y['class']=y['class'].map(dictionary)

y=np.array(y)

knn = KNeighborsClassifier(n_neighbors=25, weights='uniform')
xtrain, xtest, ytrain, ytest=train_test_split(x,y, test_size=0.2)
knn.fit(xtrain, ytrain)
prediction= knn.predict(xtest)
accuracy=accuracy_score(ytest, prediction)
print("prediction", prediction)
print("accuracy", accuracy)