import pandas as pd
import numpy as np
import random
import statistics
import itertools
from collections import Counter
import math


class MultiClassLogisticRegression():
    def __init__(self,
                weights=None,
                bias=None,
                fit_intercept=True,
                epochs=50,
                learning_rate=0.05,
                batch_size=50):
        self.weights = weights
        self.learning_rate = learning_rate
        self.bias = bias
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.batch_size = batch_size
    
    def _softmax(self, z):
        e_x = np.exp(z)
        out = e_x / (1+e_x.sum(axis=1, keepdims=True))
        return out
    
    def _get_true_class_labels(self, P):
        labels = P.argmax(axis=1)
        labels = np.array([self.class_range_to_actual_classes[i] for i in labels])
        return labels
    
    def _calculate_cross_entropy(self, y, log_yhat):
        return -np.sum(y*log_yhat, axis=1)
    
    def _convert_to_indicator(self, y):
        y_indicator=np.zeros((y.shape[0], self.num_classes))
        for index, y_value in enumerate(y):
            class_range_mapping = int(self.actual_classes_to_class_range[y_value])
            y_indicator[index, class_range_mapping]=1
        return y_indicator
    
    def _get_batches(self, X, y):
        for i in range(0, X.shape[0], self.batch_size):
            yield (X[i: i+self.batch_size], y[i: i+self.batch_size])
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))
        row_length, column_length = X.shape
    
        # number of unique classes
        self.actual_classes = sorted(np.unique(y))
        self.num_classes = len(self.actual_classes)

        # map class labels to the original class labels in Y
        self.class_range = list(range(self.num_classes))
        self.class_range_to_actual_classes = dict(zip(*(self.class_range, self.actual_classes)))
        self.actual_classes_to_class_range = dict(zip(*(self.actual_classes, self.class_range)))

        # convert y to indicator matrix form
        y=self._convert_to_indicator(y)

        # define the weights, shape=(P+1, K)
        self.weights = np.zeros((column_length, self.num_classes))

        iterations = 0
        while (iterations < self.epochs):
            iterations += 1
            # get batches
            batches = self._get_batches(X, y)

            # update weights using mini batch stochastic gradient descent
            for (x_batch, y_batch) in batches:
                # get the raw output
                z = x_batch @ self.weights
                # calculate class probabilities from raw output, shape=(B, K): B=batch size
                P = self._softmax(z)
                # calculate gradient
                grad = x_batch.T @ (P-y_batch)
                # update weights
                self.weights -= self.learning_rate * grad
    
    def predict_proba(self, X):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))
        
        z = X @ self.weights
        predicted_probs = self._softmax(z)
        return predicted_probs
    
    def predict(self, X):
        predicted_probs = self.predict_proba(X)
        preds = self._get_true_class_labels(predicted_probs)
        return preds
    
    def get_accuracy(self, y, y_hat):
        return np.mean(y == y_hat)

class Pipeline():
    def __init__(self, filename, dependent_feature, independent_feature_list):
        self.data = pd.read_csv(filename)
        self.data = self.data.drop(['Unnamed: 0'], axis=1)
        self.all_columns = self.data.columns
        self.dependent_feature=dependent_feature
        self.independent_feature_list=independent_feature_list

        # model objects
    
    def train_validate_split(self, train_part=0.8):
        # hold-out method
        # use for design trainset and validateset
        # dependent_feature:= item in feature list
        # type(dependent_feature) = string
        # independent_feature_list:=features used for regression
        # type(independent_feature)=list
        # train_part = float, 
        # 0 < train_part < 1
        # return trainset and validateset
        drop_row = []
        independent_feature_list=self.independent_feature_list
        independent_feature_list.append(self.dependent_feature)
        
        for row_index in range(len(self.data[self.dependent_feature])):
            for feature_ in independent_feature_list:
                if np.isnan(self.data[feature_][row_index]):
                    drop_row.append(row_index)
        
        drop_row = set(drop_row)
        data = self.data
        
        data = data.drop(drop_row)
        
        data = data[independent_feature_list]
        # print(data.columns)
        print('total available data is: ', data.shape[0])
        train_list = []
        validate_list = []
        for row_index in range(len(data)):
            if random.random() < train_part:
                train_list.append(row_index)
            else:
                validate_list.append(row_index)
        trainset = data.iloc[train_list]
        validateset = data.iloc[validate_list]
        print('total train data is: ', trainset.shape[0])
        print('total validate data is: ', validateset.shape[0])
        print('train data percent is: ', trainset.shape[0]/(trainset.shape[0]+validateset.shape[0]))

        del independent_feature_list[-1]
        return trainset, validateset
    
    def data_for_test(self, independent_feature_list):
        drop_row = []
        for row_index in range(self.data.shape[0]):
            for feature_ in independent_feature_list:
                if np.isnan(self.data[feature_][row_index]):
                    drop_row.append(row_index)
        drop_row = set(drop_row)
        data = self.data.drop(drop_row)
        data = data[independent_feature_list]
        # data = np.column_stack((np.ones(len(data)), data))
        return data

    def train(self, train_part):
        # params
        # model = ['linear', 'ridge', 'lasso']
        # validation = ['hold-out', 'k-fold']
        dependent_feature=self.dependent_feature
        independent_feature_list=self.independent_feature_list
        
        # if validation_method=='hold-out':
        trainset, validateset=self.train_validate_split(train_part)
        logistic_regression=MultiClassLogisticRegression()
        X_train = trainset[independent_feature_list]
        X_validate = validateset[independent_feature_list]
        y_train = trainset[dependent_feature]
        y_validate = validateset[dependent_feature]

        self.X_train = np.asarray(X_train)
        self.X_validate = np.asarray(X_validate)
        self.y_train = np.asarray(y_train)
        self.y_validate = np.asarray(y_validate)

        logistic_regression.fit(self.X_train, self.y_train)
        self.logistic_regression = logistic_regression
        y_pred = logistic_regression.predict(X_validate)
        loss = logistic_regression.get_accuracy(y_validate, y_pred)
        print('loss for multi-class logistic regression is: ', loss)
        return loss

        # if validation_method=='k-fold':
    def predict(self, independent_feature_list):
        X_predict = self.data_for_test(independent_feature_list)
        y_predict = self.logistic_regression.predict(X_predict)
        return y_predict

def main():
    # sample usage
    feature_list = ['age', 'worstgrade5', 'worstgrade4', 'worstgrade3']
    pipeline = Pipeline(filename='whatever_you_want_to_call_it.csv', dependent_feature='somehow', independent_feature_list=feature_list)
    pipeline.train(train_part=0.7)
    pipeline.predict(feature_list)

if __name__=="__main__":
    main()
