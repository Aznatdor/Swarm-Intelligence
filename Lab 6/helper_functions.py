"""
Module with helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(model, params):
    left, right = min(model.X_train), max(model.X_train)
    r = np.linspace(left, right, 100)
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(model.X_train, model.y_train, label="Real data")
    ax[0].scatter(model.X_train, model.fit_function(model.X_train, params), label="Train")
    ax[0].plot(r, model.fit_function(r, params))
    ax[0].set_title("Training set")
    ax[0].legend()

    left, right = min(model.X_test), max(model.X_test)
    r = np.linspace(left, right, 100)
    ax[1].scatter(model.X_test, model.y_test, label="Real data")
    ax[1].scatter(model.X_test, model.fit_function(model.X_test, params), label="Test")
    ax[1].plot(r, model.fit_function(r, params))
    ax[1].set_title("Test set")
    ax[1].legend()
    plt.show()

def get_data(variant):
    if variant < 10:
        variant = '0' + str(variant)
    var = pd.read_excel("DataRegression.xlsx", sheet_name=f"Var{variant}")
    return var['x'].to_numpy(), var['y'].to_numpy()