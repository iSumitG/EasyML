import numpy as np
import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt

# data parameters
DATASET_SIZE = 100 # number of data points
NUMBER_OF_FEATURES = 1
NUMBER_OF_TARGETS = 1
BIAS = 0.1
NOISE = 15.0
TEST_SIZE = 0.3 # fraction of the data to be used for testing

# training parameters
ALPHA = 0.03 # learning rate
ITERATIONS = 100 # number of iterations to run Gradient Descent

# verbose mode
VERBOSE = False

def plot(X, y, model):
    if X.shape[1] > 2:
        print "Cannot plot the 2D graph because training data is of higher dimension"
        return
    print "Plotting linear regression...",

    plt.scatter(X[:,1], y, marker=".")

    line = np.linspace(np.amin(X[:,1])-1, np.amax(X[:,1])+1)
    plt.plot(line, line * model[1] + model[0], color='red')
    plt.show()
    print "Complete\n"

def test(X_test, y_test, model):
    print "\nTesting model..."
    mean_squared_error = (1.0 / (2*X_test.shape[0])) * np.sum((np.dot(X_test, model) - y_test)**2)

    if VERBOSE:
        print "Mean squared error on test data: " + str(mean_squared_error)

    print "Complete\n"
    return mean_squared_error

def train(X_train, y_train, ITERATIONS):
    print "\nTraining model...",

    parameters = np.zeros((X_train.shape[1], 1)) # initialize the parameters to 0

    # run gradient descent for ITERATIONS
    while ITERATIONS != 0:
        parameters = parameters - (ALPHA/DATASET_SIZE) * np.dot(X_train.transpose(), (np.dot(X_train, parameters) - y_train))
        ITERATIONS -= 1

    print "Completed \n"
    return parameters

def generate_data():
    print "Generating data...",

    X, y = sklearn.datasets.make_regression(n_samples=DATASET_SIZE, n_features=NUMBER_OF_FEATURES, n_targets=NUMBER_OF_TARGETS, bias=BIAS, noise=NOISE) # generate regression data

    vector_of_ones = np.ones((X.shape[0], 1))
    X = np.concatenate((vector_of_ones, X), axis=1) # add a column of 1's in features
    y = y.reshape((y.shape[0], 1)) # reshape y e.g. from (10,) to (10,1)

    print "Complete \n"
    return sklearn.model_selection.train_test_split(X, y, test_size=TEST_SIZE) # split the dataset into training / testing

def start_regression():
    X_train, X_test, y_train, y_test = generate_data() # generate the data

    if VERBOSE:
        print "X_train matrix is: " + str(X_train.shape[0]) + "x" + str(X_train.shape[1])
        print "X_test matrix is: " + str(X_test.shape[0]) + "x" + str(X_test.shape[1])
        print "y_train matrix is: " + str(y_train.shape[0]) + "x" + str(y_train.shape[1])
        print "y_test matrix is: " + str(y_test.shape[0]) + "x" + str(y_test.shape[1])+ "\n"
        print "Some X_train data: "
        print X_train[0:5,:] if X_train.shape[0] > 5 else X_train
        print "\nSome y_train data: "
        print y_train[0:5,:] if y_train.shape[0] > 5 else y_train

    model = train(X_train, y_train, ITERATIONS) # train the model
    mean_squared_error = test(X_test, y_test, model)
    plot(X_train, y_train, model)

start_regression() # starting the regression

