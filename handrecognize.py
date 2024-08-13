import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Load the dataset
mnist = fetch_openml('mnist_784', version=1)

# Split the data into training and test sets
x = mnist.data
y = mnist.target.astype(np.int)

x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

# Normalize the images to [0, 1] range
x_train = x_train / 255.0
x_test = x_test / 255.0
