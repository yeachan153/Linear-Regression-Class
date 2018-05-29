import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from scipy import stats
import numpy as np

boston = datasets.load_boston()
features = boston.data
target = boston.target
columns = boston.feature_names
data = pd.DataFrame(features, columns = columns)
print(columns)

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

''' Linearity assumption for each of the variables '''
def linearity_check():
    for each in columns:
        plt.scatter(data[each], target, color='blue', s=3, marker='x')
        plt.ylabel("Price")
        plt.xlabel(each)
        plt.show()

