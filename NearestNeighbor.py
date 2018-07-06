import numpy as np

class NearestNeighbor(object):
    """ Implementation of Nearest Neighbor classifier """

    def __init__(self):
        pass


    def train(self, X_train, y_train):
        """
        Training of a NN classifier.

        X_train is: Training size x Number of features i.e m x n
        y_train is: Training size i.e m
        """
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_test):
        """
        Prediction of new data.

        X_test is: Test size x Number of features i.e. m x n

        Returns
        -------
        y_pred: vector of size m

        """
        print "Testing"

        y_pred = np.zeros(X_test.shape[0], dtype=self.y_train.dtype)

        for i in range(X_test.shape[0]):
            distances = np.sum(np.abs(self.X_train - X_test[i]), axis=1)
            min_distance_index = np.argmin(distances)
            y_pred[i] = self.y_train[min_distance_index]

        return y_pred


if __name__ == '__main__':

    nn = NearestNeighbor()
