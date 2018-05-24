'''
Loading necessary packages and dependencies
'''
from sklearn import datasets
boston = datasets.load_boston()
# Features variable matrix - each row contains all the data about 1 house
features = boston.data
# Feature interpretations
boston.DESCR
# Target values - the target value for each house/row in features.
target = boston.target

