'''
TO DO:
1) ALL ASSUMPTIONS?!
2) MSE for normal fit (Yeachan)
3) regression plot (Natalie)
4) Make things a bit easier to access? E.g. put functions like add_token_intercept inside normal_fit and gradient_descent

IMPROVEMENTS:
5) Feature selection (significant predictors)
6) train/split
7) cross validation
8) regularization

'''
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import warnings

pd.set_option('precision', 10)


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

    def normal_fit(self, MSE = True):
        a = np.linalg.inv(np.dot(self.data.transpose(), self.data))
        b = np.dot(a, self.data.transpose())
        self.weights = np.dot(b, self.targets)
        if MSE == True:
            self.predict(self.weights)
            return 1/len(self.targets) * sum((self.targets - self.predictions)**2)

    def gradient_descent(self, iteration=500000, cost_function=True, eta=.000001, plot=False):
        '''
        CHECK IF THIS WORKS!
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
                print('Runtime warning - try reducing the eta! Your gradient descent is overshooting')

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
        Using self.data, we use matrix multiplication to multiply
        against the coefficients matrix to yield predictions.
        :param coefficients: Regression coefficients.
        '''
        self.predictions = np.dot(self.data, coefficients)

    def r_square(self, adjusted=True):
        sum_sq = sum((self.targets - self.predictions) ** 2)
        mean_matrix = np.full(self.targets.shape, np.mean(self.targets))
        sum_mean = sum((self.targets - mean_matrix) ** 2)
        r_squared = 1 - (sum_sq / sum_mean)
        if adjusted == False:
            if r_squared > 0:
                return r_squared
            else:
                warnings.warn('Something probably went wrong with your gradient descent parameters')
        elif adjusted == True:
            top = (1 - r_squared) * (self.data.shape[0] - 1)
            bottom = self.data.shape[0] - (self.data.shape[1] - 1) - 1
            adj_r_squared = 1 - (top / bottom)
            if adj_r_squared > 0:
                return adj_r_squared
            else:
                warnings.warn('Something probably went wrong with your gradient descent parameters')

    def assumptions(self):
        '''
        Checks all your linear regression assumptions!
        1) Independence of observations - Durbin-Watson
        '''
        self.add_token_intercept()
        self.normal_fit(MSE = False)
        self.predict(self.weights)
        squared_errors = (self.targets - self.predictions)**2
        sum_of_squares = sum(squared_errors)
        numerator =[]
        for i in range(len(self.targets) - 1):
            numerator.append(
                ((self.targets[i + 1] - self.predictions[i + 1]) - (self.targets[i] - self.predictions[i])) ** 2)
        numerator = sum(numerator)
        durbin_watson = numerator / sum_of_squares
        return durbin_watson





