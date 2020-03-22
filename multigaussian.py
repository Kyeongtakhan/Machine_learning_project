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

class MultiGaussClassify():
    def __init__ (self,k,d):
        self.k = k
        self.d = d
        self.Mean_vector_T = np.zeros((self.k,self.d))
        self.Covariance_matrix = np.zeros((self.d,self.d))
        self.Inverse_matrix = np.zeros((self.d,self.d))
        cov_list1 = []
        Pc_vector1 = np.zeros((1,self.k))
        for i in range (self.k):
            Covariance_matrix = np.identity(self.d)
            cov_list1.append(Covariance_matrix)
            Pc_vector1[0][i] = 1 / self.k
        self.cov_list = cov_list1
        self.Pc_vector = Pc_vector1
    def fit(self,X,y, diag =False):

        def cov_generator(x,index,bool_val):
            a = np.subtract(x,self.Mean_vector_T[index])
            b = np.matmul(np.matrix(a).T,np.matrix(a))
            if bool_val == True:
                dig = np.diag(b)
                c = np.zeros((self.d,self.d))
                for i in range (self.d):
                    c[i][i] = dig[i]
                self.cov_list[index] = np.add(self.cov_list[index],c)
            else:
                self.cov_list[index] = np.add(self.cov_list[index],b)

        count = np.zeros((1,self.k))
        # Find the mean_vector 
        for i in range (X.shape[0]):
            if self.k == 2:
                if y[i] == 0:
                    count[0][0] += 1
                    self.Mean_vector_T[0] += X[i]
                else:
                    count[0][1] += 1
                    self.Mean_vector_T[1] += X[i]
            else:
                if y[i] == 0:
                    count[0][0] += 1
                    self.Mean_vector_T[0] += X[i]
                elif y[i] == 1:
                    count[0][1] += 1
                    self.Mean_vector_T[1] += X[i]
                elif y[i] == 2:
                    count[0][2] += 1
                    self.Mean_vector_T[2] += X[i]
                elif y[i] == 3:
                    count[0][3] += 1
                    self.Mean_vector_T[3] += X[i]
                elif y[i] == 4:
                    count[0][4] += 1
                    self.Mean_vector_T[4] += X[i]
                elif y[i] == 5:
                    count[0][5] += 1
                    self.Mean_vector_T[5] += X[i]
                elif y[i] == 6:
                    count[0][6] += 1
                    self.Mean_vector_T[6] += X[i]
                elif y[i] == 7:
                    count[0][7] += 1
                    self.Mean_vector_T[7] += X[i]
                elif y[i] == 8:
                    count[0][8] += 1
                    self.Mean_vector_T[8] += X[i]
                else:
                    count[0][9] += 1
                    self.Mean_vector_T[9] += X[i]
        
        for i in range (self.k):
            if count[0][i] != 0:
                self.Mean_vector_T[i] = self.Mean_vector_T[i] / count[0][i]
                self.Pc_vector[0][i] = count[0][i] / y.shape[0]
        
        # Covariance
        for j in range (X.shape[0]):
            if self.k == 2:
                if y[j] == 1:
                    cov_generator(X[j],1,diag)
                else:
                    cov_generator(X[j],0,diag)            
            else:
                if y[j] == 0:
                    cov_generator(X[j],0,diag)
                elif y[j] == 1:
                    cov_generator(X[j],1,diag)
                elif y[j] == 2:
                    cov_generator(X[j],2,diag)
                elif y[j] == 3:
                    cov_generator(X[j],3,diag)
                elif y[j] == 4:
                    cov_generator(X[j],4,diag)
                elif y[j] == 5:
                    cov_generator(X[j],5,diag)
                elif y[j] == 6:
                    cov_generator(X[j],6,diag)
                elif y[j] == 7:
                    cov_generator(X[j],7,diag)
                elif y[j] == 8:
                    cov_generator(X[j],8,diag)
                else:
                    cov_generator(X[j],9,diag)
        for i in range (self.k):
            if count[0][i] != 0:
                self.cov_list[i] = self.cov_list[i] / count[0][i]

    def predict(self,X):
        # inverse
        inv_list = []
        discrim_list = []
        for i in range (self.k):
            if np.linalg.det(self.cov_list[i]) != 0:
                inv = np.linalg.inv(self.cov_list[i])
                inv_list.append(inv)
            else:
                ii = np.identity(self.d)
                self.cov_list[i] = self.cov_list[i] + (ii * np.finfo(float).eps)
                inv = np.linalg.inv(self.cov_list[i])
                inv_list.append(inv)
            
        for i in range (self.k):
            m_1 = np.subtract(X, self.Mean_vector_T[i])
            m_1_T = np.transpose(m_1)
            a = np.matmul(m_1,inv_list[i])
            b = np.matmul(a,m_1_T)
            b = -0.5 * np.diag(b)
            if np.linalg.det(self.cov_list[i]) != 0:
                ll = -0.5 * math.log(np.linalg.det(self.cov_list[i]))
            else:
                ll = 0
            if self.Pc_vector[0][i] != 0:
                lg = math.log(self.Pc_vector[0][i])
            else:
                lg = 0
            g1x = ll + b + lg 
            discrim_list.append(g1x)

        predicted = []
        result = np.zeros((X.shape[0],self.k))
        for i in range(X.shape[0]):
            for j in range (self.k):
                result[i][j] = discrim_list[j][i]
            predicted.append(np.argmax(result[i]))
        return predicted


def my_cross_val(method,X,y,k,diag):
    kf = StratifiedKFold(n_splits=k)
    j = 1
    scores = []
    for train,test in kf.split(X,y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train],y[test]  
        if diag == False:
            method.fit(X_train,y_train)
        else:        
            method.fit(X_train,y_train,diag)    
        predicted = method.predict(X_test)
        print("Fold", j,":   ", 1 - accuracy_score(predicted,y_test))
        scores.append(1 - accuracy_score(predicted,y_test))
        j += 1
    np.std(scores)
    np.mean(scores)
    print("Mean:   ", np.mean(scores))
    print("Standard Deviation:   ", np.std(scores))
