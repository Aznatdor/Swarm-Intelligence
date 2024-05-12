"""
Module with Model class and regression functions
"""

import numpy as np

class Model:
    def __init__(self, X, y, regression_function, *, train_split=0.75):
        '''
        Intialize Model class

        X: 1-D array. Features array
        y: 1-D array. Labels array
        train_split: float in (0, 1) 
        regression_funcion: function to be fitted.

        '''
        self.regularizers = {
            'L2': lambda params, **kwargs: kwargs["l2"] * np.sum(params ** 2),
            'L1': lambda params, **kwargs: kwargs["l1"] * np.sum(np.abs(params)),
            'elastic_net': lambda params, **kwargs: kwargs["l1"] * np.sum(np.abs(params)) + kwargs['l2'] * np.sum(params ** 2)
        }
        self.train_test_partition(X, y, train_split)

        self.fit_function = regression_function

    def train_test_partition(self, X, y, train_split):
        '''
        Splitting data into test and train sets
        '''
        indices = np.random.permutation(len(X))
        n = int(len(X) * train_split)

        X, y = X[indices], y[indices]
        self.X_train, self.y_train = X[:n], y[:n]
        self.X_test, self.y_test = X[n:], y[n:]


    def loss(self, params, regularizer=None, **kwargs):
        '''
        Computes loss function
        '''
        y_hat = self.fit_function(x=self.X_train, params=params)
        mse = np.mean((y_hat - self.y_train) ** 2) / 2
        return mse + self.regularizers.get(regularizer, lambda params, **kwargs: 0)(params, **kwargs)
    
    def test_loss(self, params, regularizer=None, **kwargs):
        '''
        Computes loss function on test set
        '''
        y_hat = self.fit_function(x=self.X_test, params=params)
        mse = np.mean((y_hat - self.y_test) ** 2) / 2
        return mse + self.regularizers.get(regularizer, lambda params, **kwargs: 0)(params, **kwargs)