import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from math import sqrt

def pca1(x, y):
    pca = PCA(n_components=1)
    principalComponent = pca.fit_transform(x)
    train_x=principalComponent[0:749,:]
    test_x=principalComponent[750:999,:]
    train_y=y[0:749]
    test_y=y[750:999]
    regr1=linear_model.LinearRegression(fit_intercept=True)
    test_y_=regr1.fit(train_x, train_y).predict(test_x)
    error=sqrt(mean_squared_error(test_y_,test_y))
    print("RSE for single compnent PCA: %f" % error)

def pca2(x, y):
    pca = PCA(n_components=2)
    principalComponent = pca.fit_transform(x)
    train_x=principalComponent[0:749,:]
    test_x=principalComponent[750:999,:]
    train_y=y[0:749]
    test_y=y[750:999]
    regr1=linear_model.LinearRegression(fit_intercept=True)
    test_y_=regr1.fit(train_x, train_y).predict(test_x)
    error=sqrt(mean_squared_error(test_y_,test_y))
    print("RSE for two component PCA: %f"  % error)

def main(args):
    data_file=args.data_file
    with open(data_file, 'r') as df:
        data = df.readlines()

    data = data[1:]
    data = np.array([[float(col) for col in row.split(',')] for row in data])
    #np.random.shuffle(data)
    #seperating the attributes and targets
    x=data[:,1:5]
    y=data[:,0]

    #PCA for one principle component
    pca1(x, y)
    #pca for 2 principle component
    pca2(x, y)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-file", action="store", dest="data_file", type=str, help="Data file", default="ME781_dataset_160100022.csv")
    args = parser.parse_args()

    main(args)