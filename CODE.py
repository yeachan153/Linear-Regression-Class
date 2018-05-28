import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

'''
1. Implement baseline regression model

- Ordinary least squared (OLS) method
    - we find line of best fit by reducing errors in every trial
    - D = sum (for total amount of points) of (distance between line and ith point)^2
    - distance^2 because accounts for all points above (+ve) and below (-ve) of the line
    - minimising D = minimising errors, as following:
        - B1 = sum( (x(i) - mean(x)) * (y(i) - mean(y)) ) 
        - B0 = mean(y) - B1 * mean(x)
'''
from sklearn import datasets
boston = datasets.load_boston()
features = boston.data
target = boston.target
columns = boston.feature_names
print(columns)

class LinearRegression(): # look up what goes in the () eg 'self' etc


    def distance_line_to_point():
        x = [1,2,3,4,5] # target
        y = [.2,.4,.6,.8,1]
        #plt.plot(x,y)
        #plt.show()
        # mean of x and y
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        # total no. of values
        m = len(x)
        # calculate B1 and B0
        numer = 0
        denom = 0
        for i in range(m):
            numer += (x[i] - mean_x) * (y[i] - mean_y)
            denom += (x[i] - mean_x ) ** 2
        b1 = numer / denom
        b0 = mean_y - (b1 * mean_x)
        # print coefficients of y= (b1 * x) + b0
        return b0, b1

    def plot():

        max_x = np.max(x) + 100
        min_x = np.min(x) - 100

        # Calculating line values x and y
        x = np.linspace(min_x, max_x, 1000)
        y = b0 + b1 * x

        # Ploting Line
        plt.plot(x, y, color='#58b970', label='Regression Line')
        # Ploting Scatter Points
        plt.scatter(x, y, c='#ef5423', label='Scatter Plot')
        #plt.xlabel()
        #plt.ylabel()
        #plt.legend()
        plt.show()

    '''
    We can now determine the model fit by calculating (1) RMSE and (2) R^2 or 'coefficient of determination'
    '''
    def RMSE():
        rmse = 0
        for i in range(m):
            y_pred = b0 + b1 * x[i]
            rmse += (y[i] - y_pred) ** 2
        rmse = np.sqrt(rmse/m)
        return rmse

    def evaluate_algorithm(dataset, algorithm):
        test_set = list()
        for row in dataset:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(dataset, test_set)
        print(predicted)
        actual = [row[-1] for row in dataset]
        rmse = rmse_metric(actual, predicted)
        return rmse

    def R_squared():
        ss_t = 0
        ss_r = 0
        for i in range(m):
            y_pred = b0 + b1 * x[i]
            ss_t += (y[i] - mean_y) ** 2
            ss_r += (y[i] - y_pred) ** 2
        r2 = 1 - (ss_r/ss_t)
        print(r2)
'''
2. Improvements to get better predictions
'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def improvements():
    # Cannot use Rank 1 matrix in scikit learn
    X = x.reshape((m, 1))
    # Creating Model
    reg = LinearRegression()
    # Fitting training data
    reg = reg.fit(x, y)
    # Y Prediction
    Y_pred = reg.predict(x)
    # Calculating RMSE and R2 Score
    mse = mean_squared_error(y, Y_pred)
    rmse = np.sqrt(mse)
    r2_score = reg.score(x,y)

    print(np.sqrt(mse))
    print(r2_score)

class got_this_online_just_to_check():
    def mean(values):
        return sum(values) / float(len(values))

    # Calculate covariance between x and y
    def covariance(x, mean_x, y, mean_y):
        covar = 0.0
        for i in range(len(x)):
            covar += (x[i] - mean_x) * (y[i] - mean_y)
        return covar

    # Calculate the variance of a list of numbers
    def variance(values, mean):
        return sum([(x-mean)**2 for x in values])

    # Calculate coefficients
    def coefficients(dataset):
        x = [row[0] for row in dataset]
        y = [row[1] for row in dataset]
        x_mean, y_mean = mean(x), mean(y)
        b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
        b0 = y_mean - b1 * x_mean
        return [b0, b1]

    # Simple linear regression algorithm
    def simple_linear_regression(train, test):
        predictions = list()
        b0, b1 = coefficients(train)
        for row in test:
            yhat = b0 + b1 * row[0]
            predictions.append(yhat)
        return predictions

    # Test simple linear regression
    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    rmse = evaluate_algorithm(dataset, simple_linear_regression)
    print('RMSE: %.3f' % (rmse))
'''
1. Implement multiple linear regression (OLS with multple explanatory variables)
- OLS regression can be extended to include multiple variables by adding additional variables to the equation
- y = b0 + b[1]* x[1] + b[2]*x[2] + b[3]*x[3]
- http://cs229.stanford.edu/notes/cs229-notes1.pdf
'''
from sklearn import datasets
boston = datasets.load_boston()
features = boston.data
target = boston.target
columns = boston.feature_names
print(columns)

class multivariate_linear_regression():

    def distance_line_to_points(self):
        #x = [1,2,3,4,5] # target
        #y = [.2,.4,.6,.8,1]

        '''
        change this to Boston housing dataset
        '''

        m = len(x) # total number of values

        # create vector for the slope B
        theta = np.array()
        b0 = np.array()
        for i in range(columns):
            for i in range(m):
                numer += (x[i] - mean_x) * (y[i] - mean_y)
                denom += (x[i] - mean_x ) ** 2
            theta += numer / denom
            b0 += mean_y - (b1 * mean_x)
            return theta, b0

        # create vector for the columns X
        mean_X = np.array()
        for i in range(columns):
            mean_X += (np.mean(columns[i]))
            return mean_X

        # dot product
        transpose = np.dot(mean_X, theta)
        print(transpose)

     def RMSE():
        rmse = 0
        for i in range(m):
            y_pred = b0 + theta * mean_X[i]
            rmse += (y[i] - y_pred) ** 2
        rmse = np.sqrt(rmse/m)
        return rmse

    def R_squared(self):
        ss_t = 0
        ss_r = 0
        for i in range(m):
            y_pred = b0 + theta * mean_X[i]
            ss_t += (y[i] - mean_y) ** 2
            ss_r += (y[i] - y_pred) ** 2
        r2 = 1 - (ss_r/ss_t)
        print(r2)


#########################################

# Option 1: Split data into training/testing sets:
import


Split the feature data into training/testing sets
features_train = features[:-20]
features_test = features[-20:]

Split the targets into training/testing sets
target_train = target[:-20]
target_test = target[-20:]

Create linear regression object
regr = linear_model.LinearRegression()

Train the model using the training sets
regr.fit(features_train, target_train)

Make predictions using the testing set
features_pred = regr.predict(features_test)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
#print("Mean squared error: %.2f"
#      % mean_squared_error(target_test, features_pred))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(target_test, features_pred))

# Plot outputs
#plt.scatter(features_test, target_test,  color='black')
#plt.plot(features_test, feature_pred, color='blue', linewidth=1)

#plt.xticks(())
#plt.yticks(())

#plt.show()

### OPTION 2
#  make a dataframe from all info needed first
# import train_test_split to test the accuracy without manually writing out like in option 1

from sklearn import train_test_split

#df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
#df_y = pd.DataFrame(boston.target)

# https://www.youtube.com/watch?v=JTj-WgWLKFM
