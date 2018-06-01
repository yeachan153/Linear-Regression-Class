'''
1) Load the Data Set
2) Drop rows where MEDV = 50 (16 rows) and certain irrelvant rows
3) Instsantiate the class, target should be logarithmic
    - WARNING! You should now use np.exp() on final predictions to get the actual housing price
4) Remove extreme continuous outliers where z_score exceeds 10 (53 rows)
5) Make changes to the model
6) Run the script
7) Step 7 is optional. You can use the regression coefs to test on test.test_data instead of test.data. If you do,
ClassInstance.post_process() is also available for use.
'''
# 1)
from sklearn import datasets
import pandas as pd

boston = datasets.load_boston()
features = boston.data
target = boston.target
columns = boston.feature_names
data = pd.DataFrame(features, columns = columns)

# 2)
data_index = np.where(target == 50)
for i in data_index:
    data.drop(data.index[[i]], inplace = True)
target = np.delete(target,data_index)

drop_list = ['CHAS',  'ZN']
for column in drop_list:
    data.drop(column, 1, inplace = True)

# 3)
test = LinearRegression(data,np.log(target))

# 4)
outlier_row, outlier_list = test.z_scores(test.data, 10)
test.data.drop(test.data.index[list(outlier_row)], inplace = True)
test.targets = np.delete(test.targets,list(outlier_row))

# 5)
test.data['LSTAT^2'] = test.data.loc[:,'LSTAT']**2
test.data['LSTAT^3'] = test.data.loc[:,'LSTAT']**3
test.data['RM'] = np.log(test.data['RM'])
test.data['LSTATAGE'] = test.data['LSTAT']**2 * test.data['AGE']
test.data.drop('AGE', 1, inplace = True)

# 6)
test.pre_process()
test.original_split(.7)
test.train(MCC=True, normalise=True, regularise=False)
test.predict_new(test.data, test.targets)
test.r

'''
# 7)
test.predict_new(test.test_data, test.test_targets)
test.r
test.post_process()
'''

