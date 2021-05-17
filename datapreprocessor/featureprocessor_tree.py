# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:56:39 2020

@author: albert.chen
"""

import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin
import copy


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
            Columns of dtype object are imputed with the most frequent value
            in column.
            Columns of other types are imputed with mean of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series(
            [
                X[c].value_counts().index[0]  # mode
                if X[c].dtype == np.dtype("O")
                else X[c].mean()
                for c in X
            ],
            index=X.columns,
        )
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def featureprocess(series_data, model_cycle, forward_len):
    for shift_day in range(1, 8):
        # print("Shifting:", shift_day)
        series_data["lag_" + str(shift_day)] = series_data.groupby(
            ["MATERIAL_ID"]
        )["SALES"].transform(lambda x: x.shift(shift_day))

    for roll_day in [14, 30, 60]:
        # print("Rolling period:", roll_day)
        series_data["rolling_mean_" + str(roll_day)] = series_data.groupby(
            ["MATERIAL_ID"]
        )["SALES"].transform(
            lambda x: x.shift(roll_day).rolling(roll_day).mean()
        )
        series_data["rolling_std_" + str(roll_day)] = series_data.groupby(
            ["MATERIAL_ID"]
        )["SALES"].transform(
            lambda x: x.shift(roll_day).rolling(roll_day).std()
        )

    # TODO(Albert Chen): Consider more meafeature combinations to improve model

    series_data.dropna(axis=0, inplace=True)
    series_data.loc[:, "year"] = series_data["CALENDAR_DATE"].apply(
        lambda x: x.year
    )
    series_data.loc[:, "month"] = series_data["CALENDAR_DATE"].apply(
        lambda x: x.month
    )
    series_data.loc[:, "dow"] = series_data["CALENDAR_DATE"].apply(
        lambda x: x.dayofweek
    )
    series_data.loc[:, "dom"] = series_data["CALENDAR_DATE"].apply(
        lambda x: x.day
    )

    series_data.loc[:, "weekend"] = series_data["CALENDAR_DATE"].apply(
        lambda x: x.dayofweek > 4
    )
    series_data.loc[:, "weekend"] = series_data["CALENDAR_DATE"].apply(
        lambda x: x.dayofweek == 5
    )
    series_data.loc[:, "weekend"] = series_data["CALENDAR_DATE"].apply(
        lambda x: x.dayofweek == 6
    )
    end_day_test = series_data.iloc[-1][0] - datetime.timedelta(
        days=forward_len - 1
    )
    # series_data.set_index(['CALENDAR_DATE'])
    series_data.set_index("CALENDAR_DATE", inplace=True)
    test_pre = series_data.truncate(before=end_day_test)
    train = series_data.truncate(
        after=end_day_test - datetime.timedelta(days=1)
    )  #'2018-08-27'
    test = series_data[end_day_test:end_day_test].drop(
        "SALES", axis=1
    )  #'2018-08-28'
    train.rename(columns={"SALES": "SALES_0"}, inplace=True)
    train = train.reset_index()

    features = test.columns.tolist()
    noisy_features = []
    features = [c for c in features if c not in noisy_features]
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    features_numeric = test.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [f for f in features if f not in features_numeric]

    train = DataFrameImputer().fit_transform(train)
    test = DataFrameImputer().fit_transform(test)

    le = LabelEncoder()
    for col in features_non_numeric:
        le.fit(list(train[col]) + list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    return train, test, features, test_pre

    # scaler = StandardScaler()
    # for col in set(features) - set(features_non_numeric):
    #     scaler.fit(np.array(list(train[col])+list(test[col])).reshape(-1, 1))
    #     train[col] = scaler.transform(np.array(list(train[col])).reshape(-1, 1))
    #     test[col] = scaler.transform(np.array(list(test[col])).reshape(-1, 1))

    # # Apply one-hot encoder to each column with categorical data
    # OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train[features_non_numeric]))
    # OH_cols_test = pd.DataFrame(OH_encoder.transform(test[features_non_numeric]))

    # # One-hot encoding removed index; put it back
    # OH_cols_train.index = train.index
    # OH_cols_test.index = test.index

    # # Remove categorical columns (will replace with one-hot encoding)
    # num_X_train = train.drop(features_non_numeric, axis=1)
    # num_X_test = test.drop(features_non_numeric, axis=1)

    # # Add one-hot encoded columns to numerical features
    # train = pd.concat([num_X_train, OH_cols_train], axis=1)
    # test = pd.concat([num_X_test, OH_cols_test], axis=1)


if __name__ == "__main__":
    featureprocess()
