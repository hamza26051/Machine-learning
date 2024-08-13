import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


data=pd.read_csv("/content/restaurant.tsv", delimiter='\t', quoting=3)
print(data.head())

import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range(0,1000):
  review=re.sub('[^A-Za-z]', ' ', data['Review'][i])
  review=review.lower()
  review=review.split()
  ps=PorterStemmer()
  review={ps.stem(word)for word in review if not word in set(stopwords.words('english'))}
  review=' '.join(review)
  corpus.append(review)

cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=data['Liked']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=0)

classifier=GaussianNB()
classifier.fit(xtrain,ytrain)
predictions=classifier.predict(xtest)
cm=confusion_matrix(ytest, predictions)
print(cm)
accuracy=accuracy_score(ytest, predictions)
print(accuracy)