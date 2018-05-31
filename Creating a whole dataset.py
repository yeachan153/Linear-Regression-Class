'''
Run this file to output 'data' which contains one dataframe with house prices and 13 other columns
'''



'''
Loading necessary packages and dependencies
1) features = each row contains all the data about 1 house
2) target = the target price for each house/row in features
3) boston.DESCR = codebook for dataset
4) boston.feature_names = column names
'''

from sklearn import datasets
boston = datasets.load_boston()
features = boston.data
target = boston.target
columns = boston.feature_names
# boston.DESCR
import pandas as pd

'''
Putting the data into pandas and creating 1 dataset
'''
data = pd.DataFrame(features, columns = columns)
target

def train_split(data, split = .6):
    train_data = data.iloc[:round(data.shape[0] * split),:]
    test_data = data.iloc[round(data.shape[0] * split):,:]
    return train_data, test_data


