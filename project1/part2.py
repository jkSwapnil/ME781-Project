import argparse
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from math import sqrt

def subset_selection_1(train_data, test_data):
    min_error=float('inf')
    min_parameter=None
    for i in range(1,5):
        #spliting the train data in x and y part
        train_x=train_data[:,i]
        train_x=train_x.reshape(-1,1)
        train_y=train_data[:,0]
        #spliting the test data in x and y part
        test_x=test_data[:,i]
        test_x=test_x.reshape(-1,1)
        test_y=test_data[:,0]
        #fitting linear model trough the data
        regr = linear_model.LinearRegression()
        regr.fit(train_x, train_y)
        RSE= sqrt(mean_squared_error(test_y, regr.predict(test_x)))
        if(RSE<min_error):
            min_error=RSE
            min_parameter=i
    print("The best parameter for single predictor subset sampling is: x%d" %min_parameter)
    print("The RSE is: %f" %min_error)
    return min_parameter

def subset_selection_2(train_data, test_data, param1):
    min_error=float('inf')
    min_parameter=None
    for i in range (1,5):
        if(i==param1):
            continue
        #spliting the train data in x and y part
        train_x=np.array([train_data[:,param1], train_data[:,i]])
        train_x=train_x.transpose()
        train_y=train_data[:,0]
        #spliting the test data in x and y part
        test_x=np.array([test_data[:,param1], test_data[:,i]])
        test_x=test_x.transpose()
        test_y=test_data[:,0]
        #fitting linear model trough the data
        regr = linear_model.LinearRegression(fit_intercept =True)
        regr.fit(train_x, train_y)
        RSE= sqrt(mean_squared_error(test_y, regr.predict(test_x)))
        if(RSE<min_error):
            min_error=RSE
            min_parameter=i
    print("The best parameters for two predictors subset sampling are: x%d, x%d" %(param1, min_parameter))
    print("The RSE is: %f" %min_error)
    return min_parameter

def main(args):
    data_file=args.data_file
    with open(data_file, 'r') as df:
        data = df.readlines()

    data = data[1:]
    data = np.array([[float(col) for col in row.split(',')] for row in data])
    #np.random.shuffle(data)
    #spliting data in training and testing part
    train_data= data[0:749,:]
    test_data= data[750:999,:]
    
    #subset sampling of one dimension
    param1=subset_selection_1(train_data, test_data)
    #subset sampling for 2 Dimensional
    param2=subset_selection_2(train_data, test_data, param1)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-file", action="store", dest="data_file", type=str, help="Data file", default="ME781_dataset_160100022.csv")
    args = parser.parse_args()

    main(args)