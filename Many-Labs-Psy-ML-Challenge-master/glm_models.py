# generating synthetic data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import random
from copy import deepcopy
import itertools
from tqdm import tqdm
import copy



# variational autoencoders
class VAE():
    # sigmoid
    def sigmoid(self, x, derivative=False):
        sig=1/(1+np.exp(-x))
        if derivative:
            return sig*(1-sig)
        else:
            return sig
    
    # relu
    def relu(self, x, derivative=False):
        if derivative:
            return 1.0*(x>0)
        else:
            return x*(x>0)
    
    # leaky relu
    def leaky_relu(self, x, alpha=0.01, derivative=False):
        if derivative:
            return 1 if x>0 else alpha
        return x if x>0 else alpha*x


# linear regression
class linear_regression:
    def __init__(self, weights=None, bias=None):
        self.weights=weights
        self.bias=bias
    
    def predict(self, X):
        # print(X.shape)
        return (np.dot(X, self.weights))+self.bias
    
    def fit(self, X, y, alpha=0.001, iteration=100):
        # initialize parameters
        # print(y.shape)
        n_samples, n_features=X.shape
        self.weights = np.zeros(shape=(n_features, 1))
        self.bias = 0
        J=[]
        y=y.reshape(-1, 1)
        for i in range(iteration):
            # calculate y_predicted
            y_hat = self.predict(X)
            # compute the loss
            # print(y_hat.shape)
            # print(y.shape)
            loss = (1/n_samples)*np.sum((y_hat-y)**2)
            J.append(loss)

            # compute the partial derivative
            dJ_dw = (2/n_samples)*np.dot(X.T, (y_hat-y))
            dJ_db = (2/n_samples)*np.sum((y_hat-y))

            # update the parameters
            self.weights = self.weights - alpha*dJ_dw
            self.bias = self.bias - alpha*dJ_db
        
    def get_mse(self, y_true, y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        return np.sum((y_true - y_pred)**2)/y_true.shape[0]


# ridge regression
class ridge_regression:
    def __init__(self, bias=None, weights=None, lambda_param=10, fit_intercept=True):
        self.bias = bias
        self.weights = weights
        self.fit_intercept = fit_intercept
        self.lambda_param = lambda_param
    
    def fit(self, X, y):
        # print(y.shape)
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))
        self.all_weights = np.linalg.inv(np.dot(X.T, X)+self.lambda_param * np.identity(X.shape[1])).dot(X.T).dot(y)
        self.weights = self.all_weights[1:]
        self.bias = self.all_weights[0]
    
    def predict(self, X):
        self.weights = self.weights.reshape(1, -1)
        predictions = self.bias + np.dot(self.weights, X.T)
        return predictions[0]
    
    def get_mse(self, y_true, y_pred):
        return np.sum((y_true-y_pred)**2)/y_true.shape[0]

class lasso_regression:
    def __init__(self, bias=None, weights=None, lambda_param=10, max_iters=100, fit_intercept=True):
        self.bias = 0
        self.lambda_param = lambda_param
        self.max_iters = max_iters
        self.fit_intercept = fit_intercept
    
    def _soft_threshold(self, x, lambda_):
        if x>0.0 and lambda_ < abs(x):
            return x-lambda_
        elif x<0.0 and lambda_ < abs(x):
            return x+lambda_
        else:
            return 0.0

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))
        
        row_length, column_length = X.shape

        # define the weights
        self.weights = np.zeros((1, column_length))[0]
        if self.fit_intercept:
            self.weights[0] = np.sum(y-np.dot(X[:, 1:], self.weights[1:]))/(X.shape[0])
        
        # loop until max number of iterations
        for iteration in range(self.max_iters):
            start=1 if self.fit_intercept else 0
            # loop through each coordinate
            for j in range(start, column_length):
                tmp_weights = self.weights.copy()
                tmp_weights[j] = 0.0
                r_j = y-np.dot(X, tmp_weights)
                arg1 = np.dot(X[:, j], r_j)
                arg2 = self.lambda_param*X.shape[0]

                self.weights[j] = self._soft_threshold(arg1, arg2)/(X[:, j]**2).sum()
                if self.fit_intercept:
                    self.weights[0]=np.sum(y-np.dot(X[:, 1:], self.weights[1:]))/(X.shape[0])
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
    
    def predict(self, X):
        self.weights = self.weights.reshape(1, -1)
        predictions = self.bias+np.dot(self.weights, X.T)
        return predictions[0]
    
    def get_mse(self, y_true, y_pred):
        return np.sum((y_true-y_pred)**2)/y_true.shape[0]


# create the instances of classes
lin_reg = linear_regression()
ridge = ridge_regression()
lasso = lasso_regression()

models=[lin_reg, ridge, lasso]

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

    def train(self, train_part, validation_method='hold-out', k_fold=10):
        # params
        # model = ['linear', 'ridge', 'lasso']
        # validation = ['hold-out', 'k-fold']
        dependent_feature=self.dependent_feature
        independent_feature_list=self.independent_feature_list
        
        if validation_method=='hold-out':
            trainset, validateset = self.train_validate_split(train_part)
            linear_model = linear_regression()
            ridge_model = ridge_regression()
            lasso_model = lasso_regression()
            X_train = trainset[independent_feature_list]
            X_validate = validateset[independent_feature_list]
            # X_validate = self.data_for_validate(independent_feature_list)
            y_train = trainset[dependent_feature]
            y_validate = validateset[dependent_feature]
            
            self.X_train = np.asarray(X_train)
            self.X_validate = np.asarray(X_validate)
            self.y_train = np.asarray(y_train)
            self.y_validate = np.asarray(y_validate)

            loss = {}

            linear_model.fit(self.X_train, self.y_train)
            self.linear_model = linear_model
            loss['linear'] = self.linear_model.get_mse(self.y_validate, self.linear_model.predict(self.X_validate))
            ridge_model.fit(self.X_train, self.y_train)
            self.ridge_model = ridge_model
            loss['ridge'] = self.ridge_model.get_mse(self.y_validate, self.ridge_model.predict(self.X_validate))
            lasso_model.fit(self.X_train, self.y_train)
            self.lasso_model = lasso_model
            loss['lasso'] = self.lasso_model.get_mse(self.y_validate, self.lasso_model.predict(self.X_validate))

            print('the loss for three models is: ', loss)
            return loss
        if validation_method=='k-fold':
            w_dic = {'linear': [], 'ridge': [], 'lasso': []}
            b_dic = {'linear': [], 'ridge': [], 'lasso': []}
            for k in range(k_fold):
                
                trainset, validateset = self.train_validate_split(train_part=0.9)
                # print(trainset.columns)

                X_train = trainset[independent_feature_list]
                # print(trainset.shape)
                
                # print(X_train.shape)
                X_validate = validateset[independent_feature_list]
                y_train = trainset[dependent_feature]
                y_validate = validateset[dependent_feature]

                X_train = np.asarray(X_train)
                X_validate = np.asarray(X_validate)
                y_train = np.asarray(y_train)
                # print(y_train.shape)
                y_validate = np.asarray(y_validate)

                linear_model = linear_regression()
                ridge_model = ridge_regression()
                lasso_model = lasso_regression()
                linear_model.fit(X_train, y_train)
                w_dic['linear'].append(linear_model.weights)
                b_dic['linear'].append(linear_model.bias)
                
                del linear_model
                ridge_model.fit(X_train, y_train)
                w_dic['ridge'].append(ridge_model.weights)
                b_dic['ridge'].append(ridge_model.bias)
                del ridge_model
                lasso_model.fit(X_train, y_train)
                w_dic['lasso'].append(lasso_model.weights)
                b_dic['lasso'].append(lasso_model.bias)
                del lasso_model
                
            # print(type(w_dic['linear']))
            # print(len(w_dic['linear']))

            self.linear_model = linear_regression()
            self.ridge_model = ridge_regression()
            self.lasso_model = lasso_regression()

            linear_cache=[]
            ridge_cache=[]
            lasso_cache=[]
            for index in range(len(self.independent_feature_list)):
                linear_cache.append(0)
                ridge_cache.append(0)
                lasso_cache.append(0)
            for index in range(len(self.independent_feature_list)):
                for j in range(len(w_dic['linear'])):
                    linear_cache[index] += w_dic['linear'][j][index]
                    ridge_cache[index] += w_dic['ridge'][j][index]
                    lasso_cache[index] += w_dic['lasso'][j][index]
            for index in range(len(self.independent_feature_list)):
                linear_cache[index] /= len(w_dic['linear'])
                lasso_cache[index] /= len(w_dic['linear'])
                ridge_cache[index] /= len(w_dic['linear'])
                
            self.linear_model.bias = np.average(b_dic['linear'])
            self.ridge_model.bias = np.average(b_dic['ridge'])
            self.lasso_model.bias = np.average(b_dic['lasso'])
            
            self.linear_model.weights = np.asarray(linear_cache)
            self.lasso_model.weights = np.asarray(lasso_cache)
            self.ridge_model.weights = np.asarray(ridge_cache)
            # print(self.linear_model.weights)


            # print(self.linear_model.bias)

            loss = {}
            self.trainset, self.validateset = self.train_validate_split(train_part=0.7)
            self.X_train = trainset[independent_feature_list]
            self.X_validate = validateset[independent_feature_list]
            self.y_train = trainset[dependent_feature]
            self.y_validate = validateset[dependent_feature]

            self.X_train = np.asarray(X_train)
            self.X_validate = np.asarray(X_validate)
            self.y_train = np.asarray(y_train)
            self.y_validate = np.asarray(y_validate)
            # print(self.X_validate.shape)
            loss['linear'] = self.linear_model.get_mse(self.y_validate, self.linear_model.predict(self.X_validate))
            loss['ridge'] = self.ridge_model.get_mse(self.y_validate, self.ridge_model.predict(self.X_validate))
            loss['lasso'] = self.lasso_model.get_mse(self.y_validate, self.lasso_model.predict(self.X_validate))
            
            return loss


    def predict(self, independent_feature_list, model='linear'):
        X_predict = self.data_for_test(independent_feature_list)
        if model=='linear':
            y_predict = self.linear_model.predict(np.asarray(X_predict))
        elif model=='ridge':
            y_predict = self.ridge_model.predict(np.asarray(X_predict))
        elif model=='lasso':
            y_predict = self.lasso_model.predict(np.asarray(X_predict))
        return y_predict


def main():
    # sample usage
    # please input the features you want to use for fitting
    feature_list = ['age', 'worstgrade5', 'worstgrade4', 'worstgrade3']
    pipeline = Pipeline(filename='preprocessed_data.csv', dependent_feature='year', independent_feature_list=feature_list)
    pipeline.train(train_part=0.7, validation_method='k-fold')
    print('the weights for the linear model are: ', pipeline.linear_model.weights)
    pipeline.predict(feature_list)


if __name__ == '__main__':
    main()



