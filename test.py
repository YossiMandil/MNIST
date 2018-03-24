from mnist import MNIST
import xgboost as xgb
from os import getcwd
from time import time
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def learn_with_XGBClassifier(train_img, train_lbl, test_img, test_lbl):
    res = []
    train_time = time()
    xg_cl = xgb.XGBClassifier(objective='binary:logistic', learning_rate=1.5,
                                  n_estimators=30, seed=123)
    xg_cl.fit(train_img, train_lbl)
    train_time = time() - train_time
    test_time = time()
    preds = xg_cl.predict(test_img)
    test_time = time() - test_time
    accuracy = float(np.sum(preds == test_lbl)) / test_lbl.shape[0]
    return {"accuracy: ": accuracy*100, "train time": train_time, "test time:": test_time}



def learn_with_dt(train_img, train_lbl, test_img, test_lbl):
    res = []
    learn_time = time()
    xg_dt = DecisionTreeClassifier()
    xg_dt = xg_dt.fit(train_img, train_lbl)
    learn_time = time() - learn_time
    test_time = time()
    preds = xg_dt.predict(test_img)
    test_time = time() - test_time
    accuracy = float(np.sum(preds == test_lbl)) / test_lbl.shape[0]
    return {"accuracy: ": accuracy, "train time": learn_time, "test time:": test_time}


def learn_with_cv(train_img, train_lbl, test_img, test_lbl):
    '''churn_dmatrix = xgb.DMatrix(data = np.concatenate((test_img, train_img)),
                                label= np.concatenate((train_lbl, test_lbl)))
    params = {"objective": "binary:logistic", "max_depth": 4}
    cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=4, num_boost_round=10,
                       metrics="error", as_pandas=True)'''
    dmat = xgb.DMatrix(np.concatenate((train_img, test_img)), label=np.concatenate((train_lbl, test_lbl)))
    params = {"max_depth": 4, "objective": "multi:softmax", "num_class": 10}
    cv_results = xgb.cv(params, dmat, 10, 3, metrics="error", seed=123)
    print("Accuracy: %f" % ((1 - cv_results["test-error-mean"]).iloc[-1]))

def load_data(path):
    mndata = MNIST(path)
    train_img, train_lbl = mndata.load_training()
    test_img, test_lbl = mndata.load_testing()
    train_img, train_lbl = np.array(train_img), np.array(train_lbl)
    test_img, test_lbl = np.array(test_img), np.array(test_lbl)
    return train_img, train_lbl, test_img, test_lbl

def main():
    print("start loading data...")
    train_img, train_lbl, test_img, test_lbl = load_data("/home/yossi/MNIST_files")
    print("learning with xgb classifier")
    print(learn_with_XGBClassifier(train_img, train_lbl, test_img, test_lbl))

    '''print("learning with classifier")
    a = learn_with_XGBClassifier(train_img, train_lbl, test_img, test_lbl)
    for k,v in a.items():
        print(k, ": ", v)
    print(a)'''


if __name__ == "__main__":
    main()



