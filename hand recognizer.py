import mnist
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Load the dataset
xtrain = mnist.train_images()
ytrain = mnist.train_labels()

xtest = mnist.test_images()
ytest = mnist.test_labels()

