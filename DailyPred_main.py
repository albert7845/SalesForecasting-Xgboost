# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:51:29 2020

@author: albert.chen

This is an initial program based on the local data, not released version
Testing task off hours 
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import config
from machinelearningmodel import base
import copy
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")
plot = True

# Load data
def load_data():
    """load data."""
    sales_train = pd.read_csv("../data/TEST_DATA_D.csv", encoding="gb18030")
    calendar = pd.read_csv(
        "../data/2013-2050 -Calendar.csv", encoding="gb18030"
    )
    return (sales_train, calendar)


# Data preprocessing
def process_data(sales_train, calendar):
    """Feature engineering and selection."""
    sales_train = sales_train.drop(
        ["material name", "business", "category", "shop name", "SFS name"], axis=1
    )
    sales_train[sales_train["number"] < 0] = 0
    sales_train = sales_train.groupby(["calendar", "material ID"])[["number"]].sum()
    sales_train = sales_train.reset_index()
    sales_train = sales_train.drop(sales_train[sales_train["calendar date"] == 0].index)

    sales_train.rename(
        columns={"material ID": "MATERIAL_ID", "number": "SALES", "calendar date": "CALENDAR_DATE"},
        inplace=True,
    )
    sales_train.CALENDAR_DATE = sales_train.CALENDAR_DATE.apply(str)
    sales_train["CALENDAR_DATE"] = pd.to_datetime(sales_train.CALENDAR_DATE)
    sales_train = sales_train.sort_values(by="CALENDAR_DATE")
    sales_train.reset_index(inplace=True)
    del sales_train["index"]

    material_ID = [
        mat for mat in sales_train["MATERIAL_ID"].value_counts().index
    ]
    data_all_material = []
    for mat_item in range(len(material_ID)):
        data_all_material.append(
            sales_train[sales_train["MATERIAL_ID"] == material_ID[mat_item]]
        )

    all_series_data = pd.DataFrame()
    for each_mat in data_all_material:
        mat_id = each_mat.iloc[0, 1]
        each_mat = each_mat.drop("MATERIAL_ID", axis=1)
        each_mat = pd.Series(
            each_mat["SALES"].values, index=each_mat["CALENDAR_DATE"]
        )
        each_mat = each_mat.resample("D", closed="left", label="left").sum()
        each_mat = pd.DataFrame(each_mat)
        each_mat.columns = ["SALES"]
        each_mat["MATERIAL_ID"] = mat_id
        all_series_data = all_series_data.append(each_mat)

    all_series_data.index = pd.to_datetime(all_series_data.index)
    all_series_data = all_series_data.sort_index()
    # test_data = all_series_data["2018-08-29":"2018-08-31"]
    all_series_data = all_series_data.truncate(before=all_series_data.index[-1]-datetime.timedelta(days=config.lookback_len))
    all_series_data = all_series_data.reset_index()
    return all_series_data


def rmsle(y, y_pred):
    """Define loss function."""
    return np.sqrt(mean_squared_error(y, y_pred))

def accuracy(y, y_pred):
    """Define accuracy function."""
    return np.mean(1-abs(y-y_pred)/y_pred)
    

def XGB_holdout(all_series_data):
    """ Validate Xgboost model."""
    holdout_value = pd.DataFrame()
    for model_cycle in range(config.holdout_len):
        print("=> load data{0}，validate day{0}...".format(model_cycle + 1))
        goal = "SALES_{0}".format(model_cycle)
        series_data = copy.deepcopy(all_series_data)
        gbm, features, X_test, holdout, holdout_pre = base.xgboostmodel(
            series_data,
            model_cycle,
            goal,
            config.forward_len+config.holdout_len
        )
        holdout_probs = gbm.predict(xgb.DMatrix(holdout[features]))
        indices = holdout_probs < 0
        holdout_probs[indices] = 0
        split_day = holdout_pre.index.tolist()[1]
        cycle_day = holdout_pre.index.tolist()[1] + datetime.timedelta(days=model_cycle)
        pre_day = pd.DataFrame(holdout_pre[cycle_day:cycle_day].index.tolist(),columns=["CALENDAR_DATE"])
        pre_mat = pd.DataFrame(holdout_pre[split_day:split_day]["MATERIAL_ID"]).reset_index(drop=True)
        pre_value = pd.DataFrame((np.exp(holdout_probs) - 1), columns=["HoldoutSales"])
        holdout_value = pd.concat([pre_day, pre_mat, pre_value], axis=1)
        holdout_pre = holdout_pre[cycle_day:cycle_day].reset_index()[["CALENDAR_DATE", "MATERIAL_ID", "SALES"]]
        
        holdout_value["key_pre"] = holdout_value["CALENDAR_DATE"].map(str) + holdout_value["MATERIAL_ID"].map(str)
        holdout_pre["key_act"] = holdout_pre["CALENDAR_DATE"].map(str) + holdout_pre["MATERIAL_ID"].map(str)
        data_com = holdout_value.merge(holdout_pre, left_on="key_pre", right_on="key_act")
        
        holdout_accuracy = accuracy(data_com['SALES'],data_com['HoldoutSales'])
        print("Valid_day{0}_accuracy:".format(model_cycle + 1) + str(holdout_accuracy))
    return data_com


def XGB_native(all_series_data):
    """ Use Xgboost model and Direct strategy to predict."""
    submission = pd.DataFrame()
    for model_cycle in range(config.forward_len):
        print("=> modeling{0}，forcast day{0}...".format(model_cycle + 1))
        goal = "SALES_{0}".format(model_cycle)
        series_data = copy.deepcopy(all_series_data)
        gbm, features, X_test, test, test_pre = base.xgboostmodel(
            series_data,
            model_cycle,
            goal,
            config.forward_len
        )
        train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
        indices = train_probs < 0
        train_probs[indices] = 0
        error = rmsle(train_probs, np.log(X_test[goal] + 1).values)
        print("Pre_day{0}_error:".format(model_cycle + 1) + str(error))

        # Predict and Export
        test_probs = gbm.predict(xgb.DMatrix(test[features]))
        indices = test_probs < 0
        test_probs[indices] = 0
        split_day = test_pre.index.tolist()[1]
        cycle_day = test_pre.index.tolist()[1] + datetime.timedelta(
            days=model_cycle
        )
        pre_day = pd.DataFrame(
            test_pre[cycle_day:cycle_day].index.tolist(),
            columns=["CALENDAR_DATE"],
        )
        pre_mat = pd.DataFrame(
            test_pre[split_day:split_day]["MATERIAL_ID"]
        ).reset_index(drop=True)
        pre_value = pd.DataFrame(
            (np.exp(test_probs) - 1), columns=["PreSales"]
        )
        submission = pd.concat([pre_day, pre_mat, pre_value], axis=1)
        test_pre = test_pre.reset_index()[
            ["CALENDAR_DATE", "MATERIAL_ID", "SALES"]
        ]
                
        submission["key_pre"] = submission["CALENDAR_DATE"].map(str) + submission["MATERIAL_ID"].map(str)
        test_pre["key_act"] = test_pre["CALENDAR_DATE"].map(str) + test_pre["MATERIAL_ID"].map(str)
        data_com = submission.merge(test_pre, left_on="key_pre", right_on="key_act")      
        Pre_accuracy = accuracy(data_com['SALES'],data_com['PreSales'])
        print("Pre_day{0}_accuracy:".format(model_cycle + 1) + str(Pre_accuracy))             
    # end_day_test = all_series_data.iloc[-1][0] - datetime.timedelta(days=config.forward_len-1)
    # all_series_data.set_index("CALENDAR_DATE", inplace=True)
    # test_act_forward = all_series_data.truncate(before=end_day_test)
    return data_com


if __name__ == "__main__":
    print("=> read data...")
    sales_train, calendar = load_data()
    print("=> preprocessor...")
    all_series_data = process_data(sales_train, calendar)
    print("=> XGBoost modeling...")
    submission = XGB_native(all_series_data)
    print("=> validate XGBoost model...")
    data_com = XGB_holdout(all_series_data)



# if not os.path.exists('result/'):
#     os.makedirs('result/')
# submission.to_csv("./result/dat-xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.csv" % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize)) , index=False)
# # Feature importance
# if plot:
#   outfile = open('xgb.fmap', 'w')
#   i = 0
#   for feat in features:
#       outfile.write('{0}\t{1}\tq\n'.format(i, feat))
#       i = i + 1
#   outfile.close()
#   importance = gbm.get_fscore(fmap='xgb.fmap')
#   importance = sorted(importance.items(), key=operator.itemgetter(1))
#   df = pd.DataFrame(importance, columns=['feature', 'fscore'])
#   df['fscore'] = df['fscore'] / df['fscore'].sum()
#   # Plotitup
#   plt.figure()
#   df.plot()
#   df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
#   plt.title('XGBoost Feature Importance')
#   plt.xlabel('relative importance')
#   plt.gcf().savefig('Feature_Importance_xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.png' % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize)))


# error=[]

#     error.append(Modeling(each_mat,lookback_LENGTH))


# from WEIC_ML_modeling import Modeling
