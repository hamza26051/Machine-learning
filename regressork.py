import numpy as np 
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the California housing dataset
hmm = fetch_california_housing()
X, y = hmm.data, hmm.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the pipeline with StandardScaler and KNeighborsRegressor
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor())
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# Perform grid search cross-validation
mod = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=3,
)
mod.fit(X_train, y_train)

# Convert negative MSE to RMSE
results = pd.DataFrame(mod.cv_results_)
print(results)
