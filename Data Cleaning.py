test = LinearRegression(data,np.log(target))
test.pre_process()
test.original_split(.7)
test.train(MCC=True, normalise=True, regularise=False)
test.predict_new(test.data, test.targets)
print(test.adj_r)
test.predict_new(test.test_data, test.test_targets)
print(test.adj_r)
# If using test data
test.post_process()


'''
Improvements
1) np.log(targets) - implemented in line 1
2) Remove rows where medv = 50.0: seems like an error
3) Remove columns CHAS, AGE, ZN
4) add lstat^2 and latat^3
5) add np.log(RM)
6) add CRIM^2 maybe?
7) add sqrt(DIS)
8) add indus^2 and indus^3
'''
# 2)
data_index = np.where(target == 50)
for i in data_index:
    data.drop(data.index[[i]], inplace = True)
target = np.delete(target,data_index)

# 3)
drop_list = ['CHAS', 'AGE', 'ZN']
for column in drop_list:
    data.drop(column, 1, inplace = True)

# 4)
data['LSTAT^2'] = data.loc[:,'LSTAT']**2
data['LSTAT^3'] = data.loc[:,'LSTAT']**3

# 5)
data['RM_log'] = np.log(data['RM'])

# 6)
data['CRIM^2'] = data.loc[:,'CRIM']**2

# 7)
data['DIS.SQRT'] = np.sqrt(data['DIS'])

# 8)
data['INDUS^2'] = data['INDUS']**2
data['INDUS^3'] = data['INDUS']**3








