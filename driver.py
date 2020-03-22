from __future__ import division
import numpy as np
import math
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from multigaussian import MultiGaussClassify
from multigaussian import my_cross_val

if __name__ == "__main__":
    print("1. MultiGaussClassify with full covariance matrix on Boston50,")
    print("2. MultiGaussClassify with full covariance matrix on Boston75,")
    print("3. MultiGaussClassify with full covariance matrix on Digits,")
    print("4. MultiGaussClassify with diagonal covariance matrix on Boston50,")
    print("5. MultiGaussClassify with diagonal covariance matrix on Boston75,")
    print("6. MultiGaussClassify with diagonal covariance matrix on Digits,")
    print("7. LogisticRegression with on Boston50,")
    print("8. LogisticRegression with on Boston75, and")
    print("9. LogisticRegression with on Digits.")

    boston1_dataset50 = load_boston()
    boston1_dataset75 = load_boston()
    
    digits = load_digits()

    boston50 = np.percentile(boston1_dataset50.data,50)
    boston75 = np.percentile(boston1_dataset75.data,75)

    for i in range (len(boston1_dataset50.target)):
        if boston1_dataset50.target[i] >= boston50:
            boston1_dataset50.target[i] = 1
        else:
            boston1_dataset50.target[i] = 0

    for i in range (len(boston1_dataset75.target)):
        if boston1_dataset75.target[i] >= boston75:
            boston1_dataset75.target[i] = 1
        else:
            boston1_dataset75.target[i] = 0

    X50, y50 = boston1_dataset50.data, boston1_dataset50.target
    X75, y75 = boston1_dataset75.data, boston1_dataset75.target
    Xdigit, ydigit = digits.data, digits.target

    m1 = MultiGaussClassify(2,13)
    m2 = MultiGaussClassify(10,64)
    MyLogReg = LogisticRegression(penalty = 'l2',solver='lbfgs', multi_class ='multinomial')
    methods = ['MultiGaussClassify1', 'MultiGaussClassify2', 'MyLogReg']

    for model in methods:
        if model == 'MultiGaussClassify1':
            print("Error rates for MultiGaussClassify with full convariance matrix on Boston50")
            my_cross_val(m1,X50,y50,5,diag=False)
            print("Error rates for MultiGaussClassify with full convariance matrix on Boston75")
            my_cross_val(m1,X75,y75,5,diag=False)
            print("Error rates for MultiGaussClassify with full convariance matrix on Digits")
            my_cross_val(m2,Xdigit,ydigit,5,diag=False)

        elif model == 'MultiGaussClassify2':
            print("Error rates for MultiGaussClassify with diagonal convariance matrix on Boston50")
            my_cross_val(m1,X50,y50,5,diag=True)
            print("Error rates for MultiGaussClassify with diagonal convariance matrix on Boston75")
            my_cross_val(m1,X75,y75,5,diag=True)
            print("Error rates for MultiGaussClassify with diagonal convariance matrix on Digits")
            my_cross_val(m2,Xdigit,ydigit,5,diag=True)
        else:
            print("Error rates for LogisticRegression with Boston50")
            my_cross_val(MyLogReg,X50,y50,5,diag=False)
            print("Error rates for LogisticRegression with Boston75")
            my_cross_val(MyLogReg,X75,y75,5,diag=False)
            print("Error rates for LogisticRegression with Digits")
            my_cross_val(MyLogReg,Xdigit,ydigit,5,diag=False)