import pandas as pd
import numpy as np
import random
import statistics
import itertools
from collections import Counter
import math



class LogisticRegression():
    def __init__(self,
                weights=None,
                bias=None,
                fit_intercept=True,
                decision_threshold=0.5,
                epochs=50,
                solver='sgd',
                batch_size=30,
                learning_rate=0.05,
                tolerance=1e-13):
        self.weights=weights
        self.bias=bias
        self.fit_intercept=fit_intercept
        self.tolerance=tolerance
        self.decision_threshold=decision_threshold
        self.epochs=epochs
        self.solver=solver
        self.batch_size=batch_size
        self.learning_rate=learning_rate

        self.solver_func={'newton': self._newton_solver,
                        'sgd': self._sgd_solver}
    
    def _sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def _log_likelihood(self, X, y):
        P = self._sigmoid(X @ self.weights)
        P = P.reshape(-1, 1)
        log_P = np.log(P + 1e-16)

        P_ = 1-P
        log_P_ = np.log(P_ + 1e-16)
        log_likelihood = np.sum(y*log_P + (1-y)*log_P_)
        return log_likelihood
    
    def _get_true_class_labels(self, labels):
        true_labels = np.array([self.class_range_to_actual_classes[i] for i in labels])
        return true_labels
    
    def _get_batches(self, X, y):
        for i in range(X.shape[0], self.batch_size):
            yield (X[i: i+self.batch_size], y[i:i+self.batch_size])
    
    def _convert_y(self, y):
        self.actual_classes = sorted(np.unique(y))# find the unique elements of an array
        print(self.actual_classes)
        self.class_range=[0, 1]
        self.class_range_to_actual_classes = dict(zip(*(self.class_range, self.actual_classes)))
        self.actual_classes_to_class_range = dict(zip(*(self.actual_classes, self.class_range)))

        y_ = np.array([self.actual_classes_to_class_range[i] for i in y])
        y_ = y_.reshape(-1, 1)
        return y_
    
    def _newton_solver(self, X, y):
        log_likelihood = self._log_likelihood(X, y)
        iterations = 0
        delta = np.inf
        while (np.abs(delta) > self.tolerance and iterations < self.epochs):
            iterations += 1

            # calculate positive class probabilities: p = sigmoid(W*x + b)
            z = X @ self.weights
            P = self._sigmoid(z)
            P = P.reshape(-1, 1)

            # first derivative of loss w.r.t weights
            grad = X.T @ (P-y)

            # hessian of loss w.r.t weights
            P_ = 1-P
            W = P*P_
            W = W.reshape(1, -1)[0]
            W = np.diag(W)
            hess = X.T @ W @ X

            # weight update using Newton-Rhapson Method
            self.weights -= np.linalg.inv(hess) @ grad
            
            # calculate new log likelihood
            new_log_likelihood = self._log_likelihood(X, y)
            delta = log_likelihood - new_log_likelihood
            log_likelihood = new_log_likelihood
        
    def _sgd_solver(self, X, y):
        iterations = 0
        while iterations < self.epochs:
            iterations += 1

            # get batches
            batches =  self._get_batches(X, y)

            # update weights using Mini batch stochastic gradient descent
            for (x_batch, y_batch) in batches:

                # raw output
                z = x_batch @ self.weights

                # calculate position class probabilities
                P = self._sigmoid(z)

                # first derivative of loss w.r.t weights
                grad = x_batch.T @ (P-y_batch)

                # update weights
                self.weights -= self.learning_rate*grad
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))
        row_length, column_length = X.shape

        # define the weights
        self.weights = np.zeros((column_length, 1))

        # convert y to [0, 1]
        y = self._convert_y(y)

        # use the solver
        self.solver_func[self.solver](X, y)
    
    def predict_proba(self, X):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))

        z = X @ self.weights
        predict_proba = self._sigmoid(z)
        return predict_proba
    
    def predict(self, X):
        predict_probs = self.predict_proba(X)
        preds = np.where(predict_probs<0.5, 0, 1).flatten()
        true_preds = self._get_true_class_labels(preds)
        return true_preds
    
    def get_accuracy(self, y, y_hat):
        return np.mean(y==y_hat)
        
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
        logistic_regression=LogisticRegression()
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
        print('loss for logistic regression is: ', loss)
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
