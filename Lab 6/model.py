"""
Module with Model class and regression functions
"""

import numpy as np

def var7_func(x, params):
    b1, b2, b3, b4, b5, b6, b7 = params
    return (b1 + b2 * x + b3 * x ** 2 + b4 * x ** 3) / (1 + b5 * x + b6 * x ** + b7 ** 3)

def var12_func(x, params):
    b1, b2, b3, b4 = params
    return b1 - b2 * x - (1 / np.pi) * np.arctan(b3 / (x - b4))

def var16_func(x, params):
    b1, b2, b3 = params
    return (b1 / b2) * np.exp(-0.5 * ((x - b3) / b2) ** 2)

class Model:
    def __init__(self, X, y, regression_function, *, train_split=0.75):
        '''
        Intialize Model class

        X: 1-D array. Features array
        y: 1-D array. Labels array
        train_split: float in (0, 1) 
        regression_funcion: function to be fitted.
        '''
        self.train_test_partition(X, y, train_split)

        self.fit_function = regression_function

    def train_test_partition(self, X, y, train_split):
        '''
        Splitting data into test and train sets
        '''
        n = int(len(X) * train_split)

        data = np.array(list(zip(X, y)))
        np.random.shuffle(data)
        self.X_train, self.y_train = data[:n, 0], data[:n, 1]
        self.X_test, self.y_test = data[n:, 0], data[n:, 1]


    def loss(self, params):
        '''
        Computes loss function
        '''
        y_hat = self.fit_function(x=self.X_train, params=params)
        mse = np.mean((y_hat - self.y_train) ** 2) / 2
        return mse
    
    def test_loss(self, params):
        '''
        Computes loss function on test set
        '''
        y_hat = self.fit_function(x=self.X_test, params=params)
        mse = np.mean((y_hat - self.y_test) ** 2) / 2
        return mse