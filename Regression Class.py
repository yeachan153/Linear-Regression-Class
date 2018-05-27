'''
ISSUES
1) Consider changing self.data to self.features
2) Should we normalise categorical values? If not, how do we implement this?
3) Mean Normalisation speeds up gradient descent, but rounding errors in pandas dataframe make it
yield less accurate predicitons than just letting it run without normalisation.
4) PARTIALLY SOLVED: Gradient descent very fussy about eta/iteration parameter constantly having to adjust -
5) Plotting in gradient descent is a bit limited if the first number is huge, scaling problem
'''

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import warnings


class LinearRegression(object):
    def __init__(self, data, dependent_var):
        self.data = copy.deepcopy(data)
        self.targets = copy.deepcopy(dependent_var)

    def __repr__(self):
        return 'Linear Regression Class'

    def descriptives(self):
        '''
        Descriptive statement of self.data
        :return: None
        '''
        print(self.data.describe())

    def gradient_descent(self, iteration=1000, cost_function=True, eta=.000001, plot=True):
        '''
        :param iteration: Number of iterations to adjust weight
        :param cost_function: Do you want the MSE values? Useful to plot
        :param eta: Eta value - like a K-Factor in ELO
        :param plot: Do you want a plot of the cost function?
        '''
        self.sample_size = self.data.shape[0]
        self.weights = np.ones(self.data.shape[1])
        self.cost_func = []

        for i in range(int(iteration)):
            predictions = np.dot(self.data, self.weights)
            raw_error = self.targets - predictions
            warnings.simplefilter("error")
            try:
                if cost_function == True:
                    cost = 1 / (2 * len(self.targets)) * sum((predictions - raw_error) ** 2)
                    self.cost_func.append(cost)
                self.weights += eta / self.data.shape[0] * np.dot(raw_error, self.data)
            except RuntimeWarning:
                print('Runtime warning - try reducing the eta! Your gradient descient is overshooting')

        if plot == True and cost_function == True:
            figure, axis = plt.subplots(figsize=(15, 10))
            axis.plot(np.arange(iteration), self.cost_func, 'k')
            axis.set_ylabel('Mean Square Error/Cost')
            axis.set_xlabel('Iterations of gradient descent')

    def add_token_intercept(self):
        '''
        Adds 1's in column 0 of the data, in order for matrix multiplication
        with intercept values to make sense. Only run after running mean_normalise
        if mean_normalise is being run
        :return: self.data now has a new column
        '''
        self.data.insert(0, 'Intercept Token', 1)

    def mean_normalise(self, method='range'):
        '''
        Run this to normalise the data if needed.
        :param method: Either divides with the mean or the standard
        deviation of the column. Enter 'std' or 'range'.
        :return: Returns self.data as a normalised dataset - useful
        if using gradient descent to minimise cost function
        '''
        for i in range(len(self.data.columns)):
            new_col = []
            for each in self.data.iloc[:, i]:
                val_mean = each - np.mean(self.data.iloc[:, i])
                if method == 'range':
                    range1 = max(self.data.iloc[:, i]) - min(self.data.iloc[:, i])
                    new_col.append(val_mean / range1)
                elif method == 'std':
                    std = self.data.iloc[:, i].std()
                    new_col.append(val_mean / std)
            self.data.iloc[:, i] = new_col

    def predict(self, coefficients):
        '''
        TEST THIS!
        Using self.data, we use matrix multiplication to multiply
        against the coefficients matrix to yield predictions.
        :param coefficients: Regression coefficients. MAKE SURE
        THAT EACH COLUMN OF temp_data MATCHES THE COEFFICIENTS
        IN THE COEFFICIENTS MATRIX!!
        :return: Returns an n-dimensional vector of predicted values
        '''
        return np.dot(self.data, coefficients)




