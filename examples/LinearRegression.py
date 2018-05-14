# Created by: Sumit Gupta. http://sumitg.com/about

import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt

# data parameters
DATASET_SIZE = 100  # number of data points
NUMBER_OF_FEATURES = 1
NUMBER_OF_TARGETS = 1
BIAS = 0.1
NOISE = 15.0
TEST_SIZE = 0.3 # fraction of the data to be used for testing

# verbose mode
VERBOSE = True


def plot(X_test, y_test, model):
    if X_test.shape[1] > 2:
        print "Cannot plot the 2D graph because training data is of higher dimension"
        return
    print "Plotting linear regression...\n"

    plt.scatter(X_test, y_test, marker=".")
    plt.plot(X_test, model.predict(X_test), color='red')
    plt.show()


def test(X_test, y_test, model):
    print "Testing model...\n"
    y_predicted = model.predict(X_test)

    mean_squared_error = sklearn.metrics.mean_squared_error(y_test, y_predicted)

    if VERBOSE:
        print "Mean squared error on test data: " + str(mean_squared_error) + "\n"
        print "Coefficients: " + str(model.coef_)
        print "Variance score: " + str(sklearn.metrics.r2_score(y_test, y_predicted))

    return mean_squared_error


def train(X_train, y_train):
    print "Training model...\n"

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return model


def generate_data():
    print "Generating data...\n"
    X, y = sklearn.datasets.make_regression(n_samples=DATASET_SIZE, n_features=NUMBER_OF_FEATURES, n_targets=NUMBER_OF_TARGETS, bias=BIAS, noise=NOISE) # generate regression data
    return sklearn.model_selection.train_test_split(X, y, test_size=TEST_SIZE) # split the dataset into training / testing


def start_regression():
    X_train, X_test, y_train, y_test = generate_data() # generate the data

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
    mean_squared_error = test(X_test, y_test, model)  # test the model
    plot(X_test, y_test, model)  # plot the model


if __name__ == "__main__":
    start_regression() # starting the regression
