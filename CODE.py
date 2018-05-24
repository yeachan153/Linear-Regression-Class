import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
'''
Load boston dataset
'''
# make a dataframe for the boston housing data:
boston = load_boston()
boston_df = pd.DataFrame(boston)
boston_df.columns = boston.feature_names
boston_df.head()

# boston_df['name of feature'] = boston.target
# features = boston.data
# target = boston.target

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
class LinearRegression(data,...): # LOOK UP how to write class & def (eg 'self)

    def distance_line_to_point():
        x = [1,2,3,4,5]
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
        print(b1,b0)

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
        print(rmse)

    def r_squared():
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


#########################################

# Option 1: Split data into training/testing sets:

# Split the feature data into training/testing sets
#features_train = features[:-20]
#features_test = features[-20:]

# Split the targets into training/testing sets
#target_train = target[:-20]
#target_test = target[-20:]

# Create linear regression object
#regr = linear_model.LinearRegression()

# Train the model using the training sets
#regr.fit(features_train, target_train)

# Make predictions using the testing set
#features_pred = regr.predict(features_test)

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
