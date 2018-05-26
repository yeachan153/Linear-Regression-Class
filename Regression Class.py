'''
Dependencies
'''
import pandas as pd
import numpy as np
import copy

'''
Creating the regression class
'''
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

    def mean_normalise(self, method = 'mean'):
        '''
        CHECK THIS IS CORRECT!
        Run this to normalise the data if needed.
        :param method: Either divides with the mean or the standard deviation of the column
        :return: Returns self.data as a normalised dataset
        '''
        for i in range(len(self.data.columns)):
            new_col = []
            for each in self.data.iloc[:, i]:
                val_mean = each - np.mean(self.data.iloc[:, i])
                if method == 'mean':
                    range1 = max(self.data.iloc[:, i]) - min(self.data.iloc[:, i])
                    new_col.append(val_mean / range1)
                elif method == 'std':
                    std = self.data.iloc[:, i].std()
                    new_col.append(val_mean / std)
            self.data.iloc[:, i] = new_col

    def train(self):
        pass

    def predict(self):
        pass






