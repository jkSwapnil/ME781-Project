import argparse
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import linear_model
import numpy as np
from math import sqrt

def NLR(data, degree):
    error=0
    for i in range(0,len(data)):
        #getting the test data
        test_x=data[i,1:degree+1]
        test_y=data[i,0]
        #getting the train data
        train_data= np.delete(data, (i), axis=0)
        train_x= train_data[:,1:degree+1]
        train_y= train_data[:,0]
        test_x=test_x.reshape(1,-1)
        #Non linear regression
        regr = linear_model.LinearRegression(fit_intercept =True)
        test_y_=regr.fit(train_x,train_y).predict(test_x)
        error=error+ (test_y_ - test_y)**2
    return error/len(data)

def KNN(data, n_neighbours):
    error=0
    for i in range(0,len(data)):
        #getting the test data
        test_x=data[i,1]
        test_y=data[i,0]
        #getting the train data
        train_data= np.delete(data, (i), axis=0)
        train_x= train_data[:,1]
        train_y= train_data[:,0]
        train_x=train_x.reshape(-1,1)
        test_x=test_x.reshape(-1,1)
        #KNN classifier object
        knn = neighbors.KNeighborsRegressor(n_neighbours, weights='uniform')
        test_y_=knn.fit(train_x, train_y).predict(test_x)
        error= error + (test_y_ - test_y)**2
    return error/len(data)


def main(args):
    data_file=args.data_file
    with open(data_file, 'r') as df:
        data = df.readlines()

    data = data[1:]
    data = np.array([[float(col) for col in row.split(',')] for row in data])
    np.random.shuffle(data)

    #For the KNN Regression
    for i in [1,2,3,5,7]:
        ans=KNN(data,i)
        print("MSE for KNN regression with %d neighbours: %f" %(i, ans))


    #For the Non-linear regression
    data1=np.hstack((data,np.ones((1000,1))))
    data1[:,2]=np.multiply(data1[:,1], data1[:,1])
    data1[:,3]=np.multiply(data1[:,2], data1[:,1])
    data1[:,4]=np.multiply(data1[:,3], data1[:,1])
    data1[:,5]=np.multiply(data1[:,4], data1[:,1])
    for i in [2,3,4,5]:
        ans=NLR(data1,i)
        print("MSE for Non-linear regression of degree %d: %f" %(i, ans))


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-file", action="store", dest="data_file", type=str, help="Data file", default="ME781_dataset_160100022.csv")
    args = parser.parse_args()

    main(args)