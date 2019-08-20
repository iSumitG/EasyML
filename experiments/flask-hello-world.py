from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


# Created by: Sumit Gupta. http://sumitg.com/about

import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

# Data parameters
DATASET_SIZE = 100
NUMBER_OF_FEATURES = 2
NUMBER_OF_CLASSES = 2
NUMBER_OF_CLUSTER = 1
NUMBER_OF_REDUNDANT = 0
NOISE = 0.1
TEST_SIZE = 0.3

# Training parameters
PENALTY = 'l2'  # loss
C = 1.0  # inverse of regularization
SOLVER = 'lbfgs'  # solver algorithm to use

# verbose mode
VERBOSE = True

def generateData():
    #print("Generating data...\n")
    X, y = sklearn.datasets.make_classification(n_samples=DATASET_SIZE, n_features=NUMBER_OF_FEATURES, n_classes=NUMBER_OF_CLASSES, flip_y=NOISE, n_informative=NUMBER_OF_FEATURES, n_redundant=NUMBER_OF_REDUNDANT, n_clusters_per_class=NUMBER_OF_CLUSTER)  # generate classification data
    return sklearn.model_selection.train_test_split(X, y, test_size=TEST_SIZE)  # split the data into training / testing

@app.route("/logistic")
def start_logistic_regression():
    X_train, X_test, y_train, y_test = generateData()  # generate the data
    return len(X_train)
