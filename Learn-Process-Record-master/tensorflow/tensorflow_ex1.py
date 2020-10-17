# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:12:01 2018

@author: xzc
"""

# read data from csv file and train_test_split
import pandas as pd
from sklearn import train_test_split

housing = pd.read_csv('cal_housing_clean.csv')
housing.head()
y_val = housing['medianHouseValue']
x_data = housing.drop('medianHouseValue', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_val, test_size=0.3,
                                                    random_state = 101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(data = scaler.transform(X_train), columns = X_train.columns,
                       index = X_train.index)
X_test = pd.DataFrame(data = scaler.transform(X_test), columns = X_test.columns,
                      index = X_test.index)

scaler.transform(X_train)

# create Feature columns
housing.columns
import tensorflow as tf
age = tf.feature_columns.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feat_cols = [age, rooms, bedrooms, pop, households, income]

# cosntruct the input function
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train,
                                                 y = y_train,
                                                 batch_size = 10,
                                                 num_epochs = 1000,
                                                 shuffle = True)
model = tf.estimator.DNNRegressor(hidden_units = [6,6,6], feature_columns=feat_cols)
model.train(input_fn = input_func, steps = 20000)

predict_input_func = tf.estimator.inputs_pandas_input_fn(x = X_test,
                                                         batch_size=10,
                                                         num_epochs=1,
                                                         shuffle=False)
pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)

final_preds = []
for pred in predictions:
    final_pred.append(pred['predictions'])

from sklearn.metrics import mean_squred_error
print(mean_squared_error(y_test, final_preds)**0.5)









