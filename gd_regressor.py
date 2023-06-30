import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


# create the class for Batch gradient descent

class BGD_Regressor:

    def __init__(self, epochs=100, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = X.values
        y = y.values
        # initialize the weights and intercept
        self.intercept_ = 0
        self.coef_ = np.ones(shape=X.shape[1])

        # for n number of epochs
        for i in range(self.epochs):

            # do predictions
            y_pred = np.dot(X, self.coef_) + self.intercept_
            resid = y - y_pred

            # update the intercept
            intercept_der = -2 * np.mean(resid)
            self.intercept_ = self.intercept_ - (self.learning_rate * intercept_der)

            # update the coef
            coef_der = (-2 * (np.dot(resid, X))) / X.shape[0]
            self.coef_ = self.coef_ - (self.learning_rate * coef_der)

    def predict(self, X):
        X = X.values
        y_pred = np.dot(X, self.coef_) + self.intercept_
        return y_pred

    def score(self, y_true, y_pred):
        score = r2_score(y_true=y_true, y_pred=y_pred)
        return score
    



# create the class for Stochastic gradient descent

class SGD_Regressor:

    def __init__(self, epochs=100, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = X.values
        y = y.values
        # initialize the weights and intercept
        self.intercept_ = 0
        self.coef_ = np.ones(shape=X.shape[1])

        # number of rows in the data
        n = X.shape[0]

        # for n number of epochs
        for i in range(self.epochs):
            for j in range(n):
                # select the random row from the data
                idx = np.random.randint(low=0,high=n-1)
                # do predictions
                y_pred = np.dot(X[idx], self.coef_) + self.intercept_
                resid = y[idx] - y_pred

                # update the intercept
                intercept_der = -2 * resid
                self.intercept_ = self.intercept_ - (self.learning_rate * intercept_der)

                # update the coef
                coef_der = -2 * (np.dot(resid, X[idx]))
                self.coef_ = self.coef_ - (self.learning_rate * coef_der)

    def predict(self, X):
        X = X.values
        y_pred = np.dot(X, self.coef_) + self.intercept_
        return y_pred

    def score(self, y_true, y_pred):
        score = r2_score(y_true=y_true, y_pred=y_pred)
        return score
    

# create the class for mini batch gradient descent

class MBGD_Regressor:

    def __init__(self, epochs=100, learning_rate=0.01,batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = X.values
        y = y.values
        # initialize the weights and intercept
        self.intercept_ = 0
        self.coef_ = np.ones(shape=X.shape[1])

        # number of batches
        batch_num = (X.shape[0]//self.batch_size) + 1

        # for n number of epochs
        for i in range(self.epochs):
            for j in range(batch_num):
                # select in the index for the rows of data
                idx = np.random.randint(low=0,high=X.shape[0]-1,size=self.batch_size)

                # do predictions
                y_pred = np.dot(X[idx], self.coef_) + self.intercept_
                resid = y[idx] - y_pred

                # update the intercept
                intercept_der = -2 * np.mean(resid)
                self.intercept_ = self.intercept_ - (self.learning_rate * intercept_der)

                # update the coef
                coef_der = (-2 * (np.dot(resid, X[idx]))) / self.batch_size
                self.coef_ = self.coef_ - (self.learning_rate * coef_der)

    def predict(self, X):
        X = X.values
        y_pred = np.dot(X, self.coef_) + self.intercept_
        return y_pred

    def score(self, y_true, y_pred):
        score = r2_score(y_true=y_true, y_pred=y_pred)
        return score