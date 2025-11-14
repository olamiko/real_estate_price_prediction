#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

# Dataset gotten from 
# https://www.kaggle.com/api/v1/datasets/download/wardabilal/real-estate-price-insights

df = pd.read_csv("../datasets/housing_price_data.csv")
df.isna().sum()
df.dtypes

for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()

# Check the target variable

sns.histplot(df.price, bins=50)

# We will need to use a log distribution due to the shape of the distribution

price_logs = np.log1p(df.price)
sns.histplot(price_logs, bins=50)

# Now let's split the data

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=25)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=25)

len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = np.log1p(df_train.price.values)
y_test = np.log1p(df_test.price.values)
y_val = np.log1p(df_val.price.values)

del df_train['price']
del df_test['price']
del df_val['price']

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# Correlate with price

numerical = ['area','bedrooms','bathrooms','stories', 'parking']
df_full_train[numerical].corrwith(df_full_train.price)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)

print("The RMSE using Linear Regression is: ", rmse(y_val, y_pred))

# Implement XGBoost
import xgboost as xgb

def tune_xgbregressor(x_train, x_test, max_depth=2, n_estimators=10, learning_rate=0.2):
    model = xgb.XGBRegressor(objective='reg:squarederror', 
                             max_depth=max_depth, 
                             learning_rate=learning_rate,
                             n_estimators=n_estimators,
                             random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return rmse(y_val, y_pred)

# Tune learning rate
learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print("Learning rate results")
for l in learning_rate:
    print("{} -> {}".format(l,tune_xgbregressor(X_train, X_val, learning_rate=l)))

# Fix learning_rate
learning_rate = 0.5

# Tune max_depth
max_depth = [2,5,6,10,15,20]

print("Max depth results")
for m in max_depth:
    print("{} -> {}".format(m, tune_xgbregressor(X_train, X_val, 
                                                learning_rate=learning_rate,
                                                max_depth=m)))

# Fix max_depth
max_depth = 2


# Tune n_estimators
n_estimators = [6,10,15,20,50,100]

print("Number of estimators results")
for n in n_estimators:
    print("{} -> {}".format(n, tune_xgbregressor(X_train, X_val, 
                                                learning_rate=learning_rate,
                                                max_depth=max_depth,
                                                n_estimators=n)))


# Fix n_estimators
n_estimators = 20


# Train model with full train
# We will use xgb since it is slightly better than linear regression

y_full_train = np.log1p(df_full_train.price.values)

del(df_full_train['price'])


dicts_full_train = df_full_train.to_dict(orient='records')
dicts_test = df_test.to_dict(orient='records')

pipeline = make_pipeline(
    DictVectorizer(),
    LinearRegression()
)

pipeline.fit(dicts_full_train, y_full_train)

y_pred = pipeline.predict(dicts_test)
print("The RMSE for final Linear Regression Model is: ", rmse(y_test, y_pred))

dicts_full_train = df_full_train.to_dict(orient='records')
dicts_test = df_test.to_dict(orient='records')

pipeline = make_pipeline(
    DictVectorizer(),
    xgb.XGBRegressor(objective='reg:squarederror', 
                         max_depth=max_depth, 
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         random_state=42)
)

pipeline.fit(dicts_full_train, y_full_train)

y_pred = pipeline.predict(dicts_test)
print("The RMSE for final XGB Regression Model is: ", rmse(y_test, y_pred))

output_file = 'models/pipeline.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump(pipeline, f_out)


print(f'the model is saved to {output_file}')

# df_full_train.iloc[30].to_dict()

# np.expm1(y_full_train[30])

