# Created by: Sumit Gupta. http://sumitg.com/about

import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

# Data parameters
DATASET_SIZE = 1000
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


def plot(X, y, model):
    if X.shape[1] > 2:
        print "Cannot plot the 2D graph because training data is of higher dimension"
        return
    print "Plotting logistic regression...\n"

    plt.scatter(X[:,0], X[:,1], marker='.', c=y, s=10)
    line = np.linspace(np.amin(X[:, 0]) - 1, np.amax(X[:, 0]) + 1)
    plt.plot(line, (-1 * model.intercept_ - model.coef_[0][0]*line) / model.coef_[0][1], color='red')
    plt.show()


def test(X_test, y_test, model):
    print "Testing model...\n"
    y_predicted = model.predict(X_test)
    score = model.score(X_test, y_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)

    if VERBOSE:
        print "Accuracy on test data: " + str(accuracy) + "\n"
        print "Score on test data: " + str(score) + "\n"
        print "Coefficients: " + str(model.coef_)
        print "Variance score: " + str(sklearn.metrics.r2_score(y_test, y_predicted))

    return accuracy


def train(X_train, y_train):
    print "Training model...\n"
    model = sklearn.linear_model.LogisticRegression(penalty=PENALTY, solver=SOLVER, C=C)
    model.fit(X_train, y_train)
    return model


def generateData():
    print "Generating data...\n"
    X, y = sklearn.datasets.make_classification(n_samples=DATASET_SIZE, n_features=NUMBER_OF_FEATURES, n_classes=NUMBER_OF_CLASSES, flip_y=NOISE, n_informative=NUMBER_OF_FEATURES, n_redundant=NUMBER_OF_REDUNDANT, n_clusters_per_class=NUMBER_OF_CLUSTER)  # generate classification data
    return sklearn.model_selection.train_test_split(X, y, test_size=TEST_SIZE)  # split the data into training / testing


def start_logistic_regression():
    X_train, X_test, y_train, y_test = generateData()  # generate the data

    if VERBOSE:
        print "X_train matrix is: " + str(X_train.shape[0]) + "x" + str(X_train.shape[1])
        print "X_test matrix is: " + str(X_test.shape[0]) + "x" + str(X_test.shape[1])
        print "y_train matrix is: " + str(y_train.shape)
        print "y_test matrix is: " + str(y_test.shape) + "\n"
        print "Some X_train data: "
        print X_train[0:5,:] if X_train.shape[0] > 5 else X_train
        print "\nSome y_train data: "
        print y_train[0:5] if y_train.shape[0] > 5 else y_train
        print "\n"

    model = train(X_train, y_train)  # train the model
    accuracy = test(X_test, y_test, model)  # test the model
    plot(X_train, y_train, model)  # plot the model


if __name__ == "__main__":
    start_logistic_regression()

