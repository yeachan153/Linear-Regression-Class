Deliverables:
1) Report: 'Report.pdf'
2) Code: 
		-'Regression Class.py': Contains the class LinearRegression() 
    -'Data Cleaning and Execution.py': Puts the data into the correct format, cleans it, improves it, and executes after creating
    an instance of LinearRegression()
3) Readme: Here.

How to run the code (the simple version without explaining):
1) Update scikit-learn.
2) Run 'Regression Class.py'
3) Run 'Data Cleaning and Execution.py'
4) Type 'test.adj_r' to see r^2 values. 

How to run the code (the long version):
1) Update scikit-learn, otherwise the code might fail
2) Run 'Regression Class.py'. This contains the class LinearRegression() which will let you do all the predicting.
3) Open 'Data Cleaning and Execution.py'. There are 6 steps and 1 optional step in this code. You can run them one by one (in order) or 
	 simply execute the whole thing. However, I recommend running step 6 line by line and reading the printed messages.
	 Here, we will break down the steps (also labelled within the code for clarity):
	 - In step 1, you create a pandas dataframe called 'data' that contains the features and predictors without the targets (dependent
	 			variable). You also create a np.array of targets. Both have the same number of rows as we expect.
	 - In step 2, you first eliminate rows in the data (containing the features) and the respective indexes in targets when the target
	 			variable has a value of 50.0. Secondly, you also drop columns 'CHAS' and 'ZN' from the data.
	 - In step 3, you instantiate the class. As arguments, supply data and np.log(target). We logarithmise the target to make it
	 			scale better. NOTE! This means that you will have to apply np.exp() later to the prediction values if you want to take house
				prices at face value. I have instantiated the class as 'test'. 
	 - In step 4, we first identify extreme outliers with |z values| of over 10. If you want to change this value, you can edit it in line 
	 	 36 with another integer. These rows are now deleted from the data and target. There were a total of 53 such cases.
	 - In step 5, we add improvements to the model. We add additional columns LSTAT^2, LSTAT^3 and RM_log (log of RM). We also add in 
	 	 LSTATAGE, an interaction term between LSTAT and AGE. The column age is then dropped.
	 - Step 6 is further broken down into multiple steps:
	 			1) test.pre_process(): Prints VIF statistics that check for multicollinear columns and a column to check for missing values
				2) test.original_split(0.7): Enter a value between 0 - 1. Assigns each index/row of the data a value between 0 - 1, from a 
				   uniform distribution. We then pick values (rows) under 0.7 for training, and those over 0.7 for validating. This way we end
					 up with a random 70/30 data split each run. You can use another value if you wish. This explains why everytime you run the 
					 data, the R^2 changes. Creates test.data, test.targets, test.test_data, test.test_targets. Check printed message
				3) test.train(MCC=True, normalise=True, regularise=False): If normalise = True, the data is normalised. Note, test.data
				   remains changed until reinstantiated. If MCC = True, the data is cross validated using monte carlo principles. The code
					 will prompt you asking how many times you wish it to cross validate (I chose 500). It will also prompt you to choose a
					 train/test split ratio for each iteration (I chose 0.70). If regularise = True, you will be prompted to enter a 
					 regularisation parameter. Enter a value above 0 (e.g. 0.05). You can check the regression coefficients generated using
					 test.coef and the MSE using test.mean_sq_error.
				4) test.predict_new(test.data, test.targets): Predicts training data using test.coef and test.data. It uses test.targets to 
					 calculated residual values. Access the predictions using np.exp(test.predictions), the residuals using test.resid and
					 standardized residuals using test.std_res. Access the r^2 values using test.r and test.adj_r.
   - Step 7 is the same as stage 4 in step 6. To use, uncomment first. However in step 7, we supply test.test_data and calculate 
	   residuals using test.test_targets. These were generated in stage 2 of step 6, and represent data that the model has not 
		 encountered before. Thus, expect the R^2 and adjusted R^2 to be lower. Step 7 also supports the use of test.post_process().
		 Running that will print a check of first order autocorrelations between residuals, and check if standardized residuals are
		 normally distributed using the Kolmogorov-Smirnov test. A histogram is also produced as an aid. A scatterplot of 
		 predicted values against standardized residuals is also produced, with a loess line so that you can check for homoscedastity.
		 Finally, test.outliers can be used to check for residual outliers (index positions).
			
					
