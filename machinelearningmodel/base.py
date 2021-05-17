# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:47:55 2020

@author: albert.chen

"""
import os
import numpy as np
import datetime
import xgboost as xgb
from sklearn import model_selection
from datapreprocessor import featureprocessor_tree
import joblib
import warnings

warnings.filterwarnings("ignore")


def xgboostmodel(series_data, model_cycle, goal, forward_len):
    """Complete feature engineering for each xgboost model first, and then
        set hyperparameter of the model and start training.
        Please note: if detect the former trained model, call it directly
    """
    # feature_engineering
    train, test, features, test_pre = featureprocessor_tree.featureprocess(
        series_data, model_cycle, forward_len
    )
    # set_hyperparameter
    depth = 10
    eta = 0.007
    ntrees = 8000
    mcw = 3
    params = {
        "objective": "reg:linear",
        "booster": "gbtree",
        "eta": eta,
        "max_depth": depth,
        "min_child_weight": mcw,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "silent": 1,
    }

    train["SALES_" + str(model_cycle)] = train.groupby(["MATERIAL_ID"])[
        "SALES_0"
    ].transform(lambda x: x.shift(-model_cycle))
    train.dropna(axis=0, inplace=True)
    # end_day = train.iloc[-1][0]
    # holdout_day = datetime.timedelta(days=holdout_len)
    train.set_index("CALENDAR_DATE", inplace=True)
    # Train model with local split
    tsize = 0.05
    X_train, X_test = model_selection.train_test_split(train, test_size=tsize, random_state=1)    
    # X_train = train.truncate(after=end_day - holdout_day)
    # X_test = train.truncate(
    #     before=end_day - holdout_day + datetime.timedelta(days=1)
    # )
    goal = "SALES_{0}".format(model_cycle)
    dtrain = xgb.DMatrix(X_train[features], np.log(X_train[goal] + 1))
    dvalid = xgb.DMatrix(X_test[features], np.log(X_test[goal] + 1))
    watchlist = [(dvalid, "eval"), (dtrain, "train")]
    # Path=os.getcwd()
    path = "./machinelearningmodel/saved_model"
    modelPath = ""
    all_file = os.listdir(path)
    file = "gbm_{0}.pkl".format(model_cycle)
    if all_file == []:
        gbm = xgb.train(
            params,
            dtrain,
            ntrees,
            evals=watchlist,
            early_stopping_rounds=100,
            verbose_eval=True,
        )
        joblib.dump(gbm, path + "./gbm_{0}.pkl".format(model_cycle))
    else:
        for eachFile in all_file:
            if eachFile == file:
                modelPath = path + "./gbm_{0}.pkl".format(model_cycle)
                break
        if len(modelPath) != 0:
            gbm = joblib.load(modelPath)
        else:
            gbm = xgb.train(
                params,
                dtrain,
                ntrees,
                evals=watchlist,
                early_stopping_rounds=100,
                verbose_eval=True,
            )
            joblib.dump(gbm, path + "./gbm_{0}.pkl".format(model_cycle))
    return gbm, features, X_test, test, test_pre


if __name__ == "__main__":
    xgboostmodel()
