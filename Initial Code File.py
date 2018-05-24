'''
Loading necessary packages and dependencies
1) features = each row contains all the data about 1 house
2) target = the target price for each house/row in features
3) boston.DESCR = codebook for dataset
'''
from sklearn import datasets
boston = datasets.load_boston()
features = boston.data
target = boston.target
# boston.DESCR

'''

'''
