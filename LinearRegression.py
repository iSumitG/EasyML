import numpy as np
import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt
from numpy.linalg import inv

# data parameters
DATASET_SIZE = 100  # number of data points
NUMBER_OF_FEATURES = 1
NUMBER_OF_TARGETS = 1
BIAS = 0.1
NOISE = 15.0
TEST_SIZE = 0.3  # fraction of the data to be used for testing

# training parameters
ALPHA = 0.03  # learning rate
ITERATIONS = 100  # number of iterations to run Gradient Descent
NORMAL_EQUATIONS = True

# verbose mode
VERBOSE = True

def plot(X, y, model):
    if X.shape[1] > 2:
        print "Cannot plot the 2D graph because training data is of higher dimension"
        return
    print "Plotting linear regression...\n"

    plt.scatter(X[:,1], y, marker=".")

    line = np.linspace(np.amin(X[:,1])-1, np.amax(X[:,1])+1)
    plt.plot(line, line * model[1] + model[0], color='red')
    plt.show()

def test(X_test, y_test, model):
    print "Testing model...\n"
    mean_squared_error = (1.0 / (2*X_test.shape[0])) * np.sum((np.dot(X_test, model) - y_test)**2)

    if VERBOSE:
        print "Mean squared error on test data: " + str(mean_squared_error) + "\n"

    return mean_squared_error

def train(X_train, y_train):
    print "Training model...\n"

    m = X_train.shape[1]

    parameters = np.zeros((m, 1))  # initialize the parameters to 0

    # run gradient descent for ITERATIONS
    for i in range(ITERATIONS):
        error = np.dot(X_train, parameters) - y_train  # X*theta - y
        partial_derivative = np.dot(X_train.transpose(), error)  # X' * (X*theta - y)
        parameters = parameters - (ALPHA/DATASET_SIZE) * partial_derivative

    return parameters  # this is the learned model


def train_normal_equations(X_train, y_train):
    print "Training model...\n"

    m = X_train.shape[1]

    temp1 = np.linalg.inv(X_train.transpose().dot(X_train))
    parameters = temp1.dot(X_train.transpose().dot(y_train))
    return parameters  # this is the learned model

def generate_data():
    print "Generating data...\n"

    X, y = sklearn.datasets.make_regression(n_samples=DATASET_SIZE, n_features=NUMBER_OF_FEATURES, n_targets=NUMBER_OF_TARGETS, bias=BIAS, noise=NOISE)  # generate regression data

    vector_of_ones = np.ones((X.shape[0], 1))
    X = np.concatenate((vector_of_ones, X), axis=1)  # add a column of 1's in features
    y = y.reshape((y.shape[0], 1))  # reshape y e.g. from (10,) to (10,1)

    return sklearn.model_selection.train_test_split(X, y, test_size=TEST_SIZE)  # split the dataset into training / testing

def start_regression():
    X_train, X_test, y_train, y_test = generate_data()  # generate the data

    if VERBOSE:
        print "X_train matrix is: " + str(X_train.shape[0]) + "x" + str(X_train.shape[1])
        print "X_test matrix is: " + str(X_test.shape[0]) + "x" + str(X_test.shape[1])
        print "y_train matrix is: " + str(y_train.shape[0]) + "x" + str(y_train.shape[1])
        print "y_test matrix is: " + str(y_test.shape[0]) + "x" + str(y_test.shape[1])+ "\n"
        print "Some X_train data: "
        print X_train[0:5,:] if X_train.shape[0] > 5 else X_train
        print "\nSome y_train data: "
        print y_train[0:5,:] if y_train.shape[0] > 5 else y_train
        print "\n"

    if NORMAL_EQUATIONS:
        model = train_normal_equations(X_train, y_train)  # train the model using Normal Equations method
        mean_squared_error = test(X_test, y_test, model)
        plot(X_train, y_train, model)
    else:
        model = train(X_train, y_train)  # train the model using Gradient Descent
        mean_squared_error = test(X_test, y_test, model)
        plot(X_train, y_train, model)


if __name__ == '__main__':
    start_regression()  # starting the regression
