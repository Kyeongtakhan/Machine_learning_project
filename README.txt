Kyeongtak Han / Han00127 / Han00127@umn.edu 
-- How to run
unzip all files in the same directory. Open the terminal then types "python3 driver.py" 
driver.py is the wrapper code that deal with the data preprocessing and call the multigaussian class. 

For the programming part, there are two main python programs: multigaussian.py and hw2q3.py. All codes are written by Python 

-- Version of programs used in the code --
Python - 3.6.9
Scikit-learn - 0.22.1
numpy - 1.13.3

-- Version of dataset --
load_Boston - 0.18
load_digits - 0.18

-- Information of dataset -- 
Boston 
Samples total - 506
Dimensionality - 13 
Features - real, positive
Targets - real 5. - 50.

Digits
Classes - 10 
Samples per class - ~180
Samples total - 1797
Dimensionality - 13
Features - integers 0 -16

Boston50 
This is the dataset contains 2 classes. This data is classified by the median (50 percentile) over all r (response) values from Boston dataset. 

Boston75 
This is the dataset contains 2 classes. This data is classified by the 75 percentile over all r (response) values from Boston dataset.

Digits 
This is the dataset contains 10 classes. 

-- Data usage -- 
Boston50, Boston 75, Digits used in the hw2q3.py.




multigaussian.py 
It contains the class "Multigaussian". This class has two attributes that are fit, predict. 
Constructor -  takes two variables, the number of classes in data, and the dimensionality of data. 
Fit - takes three parameters, data, target and bool value that represents the digonality of the covariance metrices.
Predict - takes one parameter for the testing data and it returns the predicted labeled data. 
my_cross_val - takes four parameters that are model, data, target and the diagonality of the covariance.
cov_generator - This is the function that only used to generate the covariance metrices for each class. 

driver.py
This is only used for calling model, fiting, predicting, and shows the error rates of each model and each data I used.


