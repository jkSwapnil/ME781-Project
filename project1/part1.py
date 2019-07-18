import argparse
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from math import sqrt
from scipy import stats

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
    #spliting the train data in x and y part
    train_x=train_data[:,1:5]
    train_y=train_data[:,0]
    #spliting the test data in x and y part
    test_x=test_data[:,1:5]
    test_y=test_data[:,0]

    #fitting a linear model using scikit_learn linear_model module
    regr = linear_model.LinearRegression(fit_intercept =True)
    regr.fit(train_x, train_y)

    #prdiction by training model
    test_y_pred = regr.predict(test_x)
    params = np.append(regr.intercept_,regr.coef_)
    # The coefficients
    print('Coefficients: %s'  % str(params))
    # The mean squared error
    print("RSE for test data: %f"  % sqrt(mean_squared_error(test_y, test_y_pred)))
    print("RSE for training data: %f"  % sqrt(mean_squared_error(train_y, regr.predict(train_x))))


    #--------code for calculating the p-values of the coefficients-------
    newX = np.append(np.ones((len(train_x),1)), train_x, axis=1)
    MSE = mean_squared_error(test_y, test_y_pred)

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
    p_values=np.round(p_values,3)
    print("p-values: %s" % str(p_values))


if __name__=="__main__":
    #print("This code is part one of project, it performs 4-D Linear Regression on data and gaive different statistics")
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-file", action="store", dest="data_file", type=str, help="Data file", default="ME781_dataset_160100022.csv")
    args = parser.parse_args()

    main(args)