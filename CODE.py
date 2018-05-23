import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

# download the data again - read tar.gz file
boston = datasets.load_boston()
print(boston)
features = boston.data
target = boston.target

# Option 1: Split data into training/testing sets:

# Split the feature data into training/testing sets
features_train = features[:-20]
features_test = features[-20:]

# Split the targets into training/testing sets
target_train = target[:-20]
target_test = target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(features_train, target_train)

# Make predictions using the testing set
features_pred = regr.predict(features_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(target_test, features_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(target_test, features_pred))

# Plot outputs
plt.scatter(features_test, target_test,  color='black')
plt.plot(features_test, feature_pred, color='blue', linewidth=1)

plt.xticks(())
plt.yticks(())

plt.show()

### OPTION 2
#  make a dataframe from all info needed first
# import train_test_split to test the accuracy without manually writing out like in option 1

from sklearn import train_test_split

df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)

# https://www.youtube.com/watch?v=JTj-WgWLKFM
