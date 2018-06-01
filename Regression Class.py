'''
NOTE!
1) UPDATE scikit-learn
'''

'''
TO DO:
3) regression plot (Natalie)

IMPROVEMENTS:
3) MAKE EVERYTHING WORK AGAIN! - sort out unhidden functions
4) Regularization (using normal)
5) Leverage/Influential points
6) Maybe feature selection?

self.coef
self.mean_sq_error
'''
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kstest
from sklearn.model_selection import KFold

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

    def z_scores(self,data,threshold):
        temp_data = copy.deepcopy(data)
        cols = list(temp_data.columns)
        for col in cols[1:]:
            temp_data[col] = (temp_data[col] - temp_data[col].mean()) / temp_data[col].std(ddof=0)
        outliers = []
        unique_index = []
        for column in temp_data:
            for idx, each_z in enumerate(temp_data[column]):
                if np.abs(each_z) > threshold:
                    outliers.append('Index ' + str(idx) + ' in the column ' + column + ' is an outlier')
                    unique_index.append(idx)
        return set(unique_index), outliers

    def missing_value(self):
        print(self.data.isnull().sum())

    def original_split(self, split):
        self.data, self.targets, self.test_data, self.test_targets = self.train_split(split)
        print('Your training data can be accessed in class.data and class.targets. Your test data is class.test_data and class.test_targets')

    def normal_fit(self, data, target, regrate, regularise = False):
        '''
        :param data: Data to be fitted
        :param target: Target Data
        :return: Returns regression coefficients
        '''
        if regularise == False:
            a = np.linalg.inv(np.dot(data.transpose(), data))
            b = np.dot(a, data.transpose())
            weights = np.dot(b, target)
            return weights
        elif regularise == True:
            identity = np.identity(data.shape[1])
            identity[0][0] = 0
            a = np.linalg.inv(np.dot(data.transpose(), data) + regrate * identity)
            b = np.dot(a, data.transpose())
            weights = np.dot(b, target)
            return weights

    def MSE(self, test_data, test_targets, weights):
        '''
        :param test_data: Enter test data
        :param test_targets: Enter test target
        :param weights: Enter regression coefficients that you extracted from training
        :return: Retunrs MSE
        '''
        prediction = np.dot(test_data, weights)
        MSE = 1 / len(test_targets) * sum((test_targets - prediction) ** 2)
        return MSE

    # def gradient_descent(self, iteration=50000, cost_function=True, eta=.000001, plot=False):
    #     '''
    #     CHECK IF THIS WORKS!
    #     :param iteration: Number of iterations to adjust weight
    #     :param cost_function: Do you want the MSE values? Useful to plot
    #     :param eta: Eta value - like a K-Factor in ELO
    #     :param plot: Do you want a plot of the cost function?
    #     '''
    #     self.sample_size = self.data.shape[0]
    #     self.weights = np.ones(self.data.shape[1])
    #     self.cost_func = []
    #
    #     for i in range(int(iteration)):
    #         predictions = np.dot(self.data, self.weights)
    #         raw_error = self.targets - predictions
    #         warnings.simplefilter("error")
    #         try:
    #             if cost_function == True:
    #                 cost = 1 / (2 * len(self.targets)) * sum((predictions - raw_error) ** 2)
    #                 self.cost_func.append(cost)
    #             self.weights += eta / self.data.shape[0] * np.dot(raw_error, self.data)
    #         except RuntimeWarning:
    #             print('Your gradient descent is overshooting! Lower the eta and run again.')
    #
    #
    #     if plot == True and cost_function == True:
    #         figure, axis = plt.subplots(figsize=(15, 10))
    #         axis.plot(np.arange(iteration), self.cost_func, 'k')
    #         axis.set_ylabel('Mean Square Error/Cost')
    #         axis.set_xlabel('Iterations of gradient descent')

    def mean_normalise(self, data):
        '''
        Run this to normalise the data if needed.
        :param method: Either divides with the mean or the standard
        deviation of the column. Enter 'std' or 'range'.
        :return: Returns self.data as a normalised dataset - useful
        if using gradient descent to minimise cost function
        '''
        for i in range(len(data.columns)):
            new_col = []
            for each in data.iloc[:, i]:
                val_mean = each - np.mean(data.iloc[:, i])
                range1 = max(data.iloc[:, i]) - min(data.iloc[:, i])
                new_col.append(val_mean / range1)
            data.iloc[:, i] = new_col
        return data

    def predict_new(self, data, targets):
        data = copy.deepcopy(data)
        self.data2 = data
        self.targets2 = copy.deepcopy(targets)
        data.insert(0, 'Intercept Token', 1)
        self.predictions = np.dot(data, self.coef)
        self.resid = self.targets2 - self.predictions
        self.std_res = self.resid / np.std(self.resid)
        self.r_square()
        print('Check class.predictions, class.resid & class.std_res')

    def r_square(self):
        sum_sq = sum((self.targets2 - self.predictions) ** 2)
        mean_matrix = np.full(self.targets2.shape, np.mean(self.targets2))
        sum_mean = sum((self.targets2 - mean_matrix) ** 2)
        r_squared = 1 - (sum_sq / sum_mean)
        if r_squared > 0:
            self.r = r_squared
            print('Check class.r for R^2')
        else:
            warnings.warn('If you used gradient descent, try fitting with inverse transpose')
        top = (1 - r_squared) * (self.data2.shape[0] - 1)
        bottom = self.data2.shape[0] - (self.data2.shape[1] - 1) - 1
        adj_r_squared = 1 - (top / bottom)
        if adj_r_squared > 0:
            self.adj_r = adj_r_squared
            print('Check class.adj_r for adjusted R^2')
        else:
            warnings.warn('If you used gradient descent, try fitting with inverse transpose')

    def durbin_watson(self):
        squared_errors = (self.test_targets - self.predictions) ** 2
        sum_of_squares = sum(squared_errors)
        numerator = []
        for i in range(len(self.test_targets) - 1):
            numerator.append(
                ((self.test_targets[i + 1] - self.predictions[i + 1]) - (self.test_targets[i] - self.predictions[i])) ** 2)
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
        print('Check residual plot!')
        plt.figure()
        sns.set()
        sns.regplot(self.predictions, self.std_res, lowess=True, scatter_kws={'s': 2}, color='.10')
        plt.title('Std. Residuals vs Predicted')
        plt.xlabel('Predicted Values')
        plt.ylabel('Standardized Residuals')

    def multicollinearity(self):
        data = copy.deepcopy(self.data)
        data.insert(0, 'Intercept Token', 1)
        VIF = pd.Series([variance_inflation_factor(data.values, i) for i in range(data.shape[1])],
                        index=list(data))
        for idx, value in enumerate(VIF[1:]):
            if value > 5:
                print(
                    'The feature ' + VIF.index[idx] + ' shows evidence of multicollinearity.' + ' VIF = ' + str(value))

    def outlier_func(self):
        self.outliers = []
        for i in range(self.test_data.shape[0]):
            if np.abs(self.std_res[i]) > 3:
                self.outliers.append(i)
        if len(self.outliers) == 0:
            print('No prediction outliers present')
        else:
            print(str(len(self.outliers)) + ' outliers. Check class.outliers for row indexes')

    def residual_normality(self):
        p_val = kstest(self.std_res, cdf = 'norm')[1]
        if p_val > 0.05:
            print('Residuals normally distributed according to Kolmogorov-Smirnov')
        elif p_val < 0.05:
            print('Residuals not normally distributed according to Kolmogorov-Smirnov - check residual histogram')
        plt.figure()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        sns.distplot(self.std_res)
        plt.title('House Price Residuals')
        plt.xlabel('Standardized Residuals')
        plt.ylabel('Count')

    def train_split(self, split=.7):
        '''
        :param split: Enter the split/train split ratio
        :return: Returns train_data, train_targets, test_data, test_targets
        '''
        data = copy.deepcopy(self.data)
        target = copy.deepcopy(self.targets)
        random = np.random.rand(len(data)) < split
        train_data = data[random]
        test_data = data[~random]
        train_targets = target[random]
        test_targets = target[~random]
        return train_data, train_targets, test_data, test_targets

    def train(self, MCC=True, normalise = False, regularise = False):
        '''
        :param MCC: True for Monte Carlo Cross Validation
        :normalise: True/False to normalise data
        :return: Regression coefficients, Mean Squared Error
        '''
        if normalise == True:
            self.data = self.mean_normalise(self.data)
            self.test_data = self.mean_normalise(self.test_data)
        else:
            pass
        target = copy.deepcopy(self.targets)
        temp_data = copy.deepcopy(self.data)
        temp_data2 = copy.deepcopy(temp_data)
        temp_data2.insert(0, 'Intercept Token', 1)
        coef = self.normal_fit(temp_data2, target, regrate = 0)
        Msq = self.MSE(temp_data2, target, coef)
        if regularise == True and MCC == False:
            regrate = float(input('Input regularisation rate'))
            coef = self.normal_fit(temp_data2, target,regrate, regularise = True)
            Msq = self.MSE(temp_data2, target, coef)
        if MCC == False:
            self.coef = coef
            self.mean_sq_error = Msq
            print('Check class.coef & class.mean_sq_error.')
        elif MCC == True:
            n_folds = int(input('How many times should I run? Enter a whole number'))
            fold_split = float(input('Enter the split ratio per round  (0.01 - 0.90)'))
            summed_coef = 0
            summed_MSE = 0
            if regularise == True:
                regrate = float(input('Input regularisation rate'))
                for every_fold in range(n_folds):
                    train_data, train_targets, test_data, test_targets = self.train_split(split=fold_split)
                    train_data.insert(0, 'Intercept Token', 1)
                    test_data.insert(0, 'Intercept Token', 1)
                    summed_coef += self.normal_fit(train_data, train_targets,regrate, regularise = True)
                    current_coef = self.normal_fit(train_data, train_targets,regrate, regularise = True)
                    summed_MSE += self.MSE(test_data, test_targets, current_coef)
                avg_coef = summed_coef / n_folds
                avg_MSE = summed_MSE / n_folds
            elif regularise == False:
                for every_fold in range(n_folds):
                    train_data, train_targets, test_data, test_targets = self.train_split(split=fold_split)
                    train_data.insert(0, 'Intercept Token', 1)
                    test_data.insert(0, 'Intercept Token', 1)
                    summed_coef += self.normal_fit(train_data, train_targets,regrate = 0)
                    current_coef = self.normal_fit(train_data, train_targets, regrate = 0)
                    summed_MSE += self.MSE(test_data, test_targets, current_coef)
                avg_coef = summed_coef / n_folds
                avg_MSE = summed_MSE / n_folds
            if Msq == avg_MSE:
                print('Your mean squared error has remained the same after cross validation. Check class.coef & class.mean_sq_error.')
            elif Msq > avg_MSE:
                print('Your mean squared error has reduced post MCC from ' + str(Msq) + ' to ' + str(avg_MSE) +
                      '. Check class.coef & class.mean_sq_error.')
            elif Msq < avg_MSE:
                print('Your mean squared error has increased post MCC from ' + str(Msq) + ' to ' + str(avg_MSE) +
                      '. Check class.coef & class.mean_sq_error.')
            self.coef = avg_coef
            self.mean_sq_error = avg_MSE

    def pre_process(self):
        self.multicollinearity()
        self.missing_value()

    def post_process(self):
        self.durbin_watson()
        self.residual_homoscedastity()
        self.residual_normality()
        self.outlier_func()

    '''
    def p_val(self):
        # SE_table = copy.deepcopy(self.test_data)
        # # for column in self.test_data:
        # for i in range(len(self.test_targets)):
        #     num = np.sqrt(sum((self.test_targets[i] - self.predictions[i])**2) / (len(self.test_data) - len(list(self.test_data))))
        #     denom = np.sqrt(sum(self.test_data.loc[:,'CRIM'].values[i] - np.mean(self.test_data.loc[:,'CRIM'])))
        # print(num/denom)
        pass
    '''


