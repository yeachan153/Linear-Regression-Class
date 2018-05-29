import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

boston = datasets.load_boston()
features = boston.data
target = boston.target
columns = boston.feature_names
data = pd.DataFrame(features, columns = columns)

print(data)
print(columns)

for each in columns:
        plt.scatter(data[each], target, color='blue', s=3, marker='x')
        plt.ylabel("Price")
        plt.xlabel(each)
        plt.show()


''' Outliers check '''
def z_scores(data):
    threshold = 3
    cols = list(data.columns)
    for col in cols:
        col_zscore = col + '_zscore'
        data[col_zscore] = (data[col] - data[col].mean())/data[col].std(ddof=0)
    z_scores_df = data.iloc[:,14:26]
    print(z_scores_df)
    # return np.where(np.abs(z_scores) > threshold)

''' Linearity assumption '''
def linearity_check():
    for each in columns:
        plt.scatter(data[each], target, color='blue', s=3, marker='x')
        plt.ylabel("Price")
        plt.xlabel(each)
        plt.show()


''' 
- multicollinearity assumption
- Problem = using "y~x", ols and sm; do we need to write this from scratch?
'''
def VIF(data, target, columns):
    for i in range(0, columns.shape[0]):
        y = data[columns[i]]
        x = target
        rsq = sm.ols(formula="y~x", data=data).fit().rsquared
        vif = round(1/(1-rsq),2)
        print(columns[i], " VIF = " , vif)

''' 
PLOTTING
1) Calculate studentized residual
2) Calculate unstandardized predicted values 
3) Partial regression plots for each independent variable and dependent variable (excluding categorical data)
# '''


def residual_predicted_plot(self):
    pass


def partial_regression_plots(self, coefficients, columns=columns):
    '''
    - Partial regression plots for each independent variables vs. target,
      excluding categorical data (e.g. CHAS).
    - Values of partial coefficients are found using OLS.
    - 12 plots (nrow=3, ncol=4) in one figure.
    :param self:
    '''
    residuals = matrix()
    for i in range(columns):
        residuals_indv_var = pass # residual of regressing response variable vs. indp var (omitting target)
        residuals_least_square = pass # residual of regression vs remaining indp var
        return residuals_indv_var, residuals_least_square

    plt.subplot(341)
    for i in range(columns):
        plt.scatter(residuals_indv_var[i], residuals_least_square[i])
    plt.show()

