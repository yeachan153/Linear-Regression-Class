'''
TO DO:
3) regression plot (Natalie)

IMPROVEMENTS:
4) Leverage/Influence (cook's distance)
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
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kstest

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
            self.MSE =  1/len(self.targets) * sum((self.targets - self.predictions)**2)

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
                print('Your gradient descent is overshooting! Lower the eta and run again.')


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
        self.residuals =  self.targets  - self.predictions

    def predict_new(self, data, coef, targets):
        data.insert(0, 'Intercept Token', 1)
        self.new_predictions = np.dot(data, coef)
        self.new_resid = targets - self.new_predictions
        self.new_std_res = self.new_resid / np.std(self.new_resid)

    def r_square(self, adjusted=True):
        sum_sq = sum((self.targets - self.predictions) ** 2)
        mean_matrix = np.full(self.targets.shape, np.mean(self.targets))
        sum_mean = sum((self.targets - mean_matrix) ** 2)
        r_squared = 1 - (sum_sq / sum_mean)
        if adjusted == False:
            if r_squared > 0:
                return r_squared
            else:
                warnings.warn('If you used gradient descent, try fitting with inverse transpose')
        elif adjusted == True:
            top = (1 - r_squared) * (self.data.shape[0] - 1)
            bottom = self.data.shape[0] - (self.data.shape[1] - 1) - 1
            adj_r_squared = 1 - (top / bottom)
            if adj_r_squared > 0:
                return adj_r_squared
            else:
                warnings.warn('If you used gradient descent, try fitting with inverse transpose')

    def durbin_watson(self):
        squared_errors = (self.targets - self.predictions) ** 2
        sum_of_squares = sum(squared_errors)
        numerator = []
        for i in range(len(self.targets) - 1):
            numerator.append(
                ((self.targets[i + 1] - self.predictions[i + 1]) - (self.targets[i] - self.predictions[i])) ** 2)
        numerator = sum(numerator)
        durbin_watson = numerator / sum_of_squares
        if durbin_watson < 2.5 and durbin_watson > 1.5:
            print(
                'No evidence of first order auto-correlations between residuals - check the critical tables to be sure. Durbin Watson: ' + str(
                    durbin_watson))
        elif durbin_watson > 2.5:
            print('Evidence of negative first order autocorrelations between residuals. Durbin Watson: ' + str(
                durbin_watson))
        elif durbin_watson < 1.5:
            print('Evidence of positive first order autocorrelations between residuals. Durbin Watson:  ' + str(
                durbin_watson))

    def residual_homoscedastity(self):
        self.std_res = self.residuals / np.std(self.residuals)
        print('Check residual plot!')
        plt.figure()
        sns.set()
        sns.regplot(self.predictions, self.std_res, lowess=True, scatter_kws={'s': 2}, color='.10')
        plt.title('Std. Residuals vs Predicted')
        plt.xlabel('Predicted Values')
        plt.ylabel('Standardized Residuals')

    def multicollinearity(self):
        VIF = pd.Series([variance_inflation_factor(self.data.values, i) for i in range(self.data.shape[1])],
                        index=list(self.data))
        for idx, value in enumerate(VIF[1:]):
            if value > 5:
                print(
                    'The feature ' + VIF.index[idx] + ' shows evidence of multicollinearity.' + ' VIF = ' + str(value))

    def outlier_func(self):
        self.outliers = []
        for i in range(self.data.shape[0]):
            if np.abs(self.std_res[i]) > 3:
                self.outliers.append(i)
        if len(self.outliers) == 0:
            print('No prediction outliers present')
        else:
            print(str(len(self.outliers)) + ' outliers. Check class.outliers for row indexes')

    def leverage(self):
        pass

    def influence(self):
        pass

    def filter_rows(self, row_indexes):
        pass

    def train_split(self):
        pass

    def feature_select(self):
        pass

    def cross_validate(self):
        pass

    def residual_normality(self):
        p_val = kstest(self.std_res, cdf = 'norm')[1]
        if p_val > 0.05:
            print('Residuals normally distributed according to Kolmogorov-Smirnov')
        elif p_val < 0.05:
            print('Residuals not normally distributed according toKolmogorov-Smirnov - check residual histogram')
        plt.figure()
        sns.distplot(self.std_res)
        plt.title('House Price Residuals')
        plt.xlabel('Standardized Residuals')
        plt.ylabel('Count')

    def fit(self, assumptions = True, method = 'inverse_transpose'):
        self.add_token_intercept()
        if method == 'inverse_transpose':
            MSE = bool(input('Do you want a MSE value? - Enter True or False. Access it using class.MSE'))
            self.normal_fit(MSE = MSE)
        elif method == 'gradient descent':
            num_iter = int(input('Enter a whole number to iterate on'))
            eta = float(input('What is your learning rate? Enter a decimal number under 0.5'))
            plot = bool(input('Enter True or False - plot of cost function'))
            self.gradient_descent(num_iter,eta=eta, plot = plot)
        self.predict(self.weights)
        if assumptions == True:
            self.durbin_watson()
            self.residual_homoscedastity()
            self.multicollinearity()
            self.outlier_func()
            self.residual_normality()



