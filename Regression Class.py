'''
NOTES
1) Consider changing self.data to self.features
2) Should we normalise categorical values? If not, how do we implement this?
3) Add error messages for predict function - e.g. if matrix multiplication is not possible.
'''

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

class LinearRegression(object):
    def __init__(self, data):
        self.data = copy.deepcopy(data)

    def __repr__(self):
        return 'Linear Regression Class'

    def descriptives(self):
        '''
        Descriptive statement of self.data
        :return: None
        '''
        print(self.data.describe())

    def cost_function(self):
        pass

    def gradient_descent(self):
        pass

    def mean_normalise(self, method = 'range'):
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

    def train(self):
        pass

    def predict(self, coefficients):
        '''
        Using self.data, we use matrix multiplication to multiply
        against the coefficients matrix to yield predictions.
        :param coefficients: Regression coefficients. MAKE SURE
        THAT EACH COLUMN OF temp_data MATCHES THE COEFFICIENTS
        IN THE COEFFICIENTS MATRIX!!
        :return: Returns an n-dimensional vector of predicted values
        '''
        temp_data = copy.deepcopy(self.data)
        temp_data.insert(0, 'Intercept Token', 1)
        return np.dot(temp_data, coefficients)







