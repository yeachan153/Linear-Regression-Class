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
target = pd.DataFrame(target)
data['House Prices'] = target

house_price = list(data)[-1]
rest_of_columns = list(data)[:-1]
columns = [house_price] + rest_of_columns

data = pd.DataFrame(data, columns = columns)


