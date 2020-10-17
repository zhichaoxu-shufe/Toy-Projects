import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from graphviz import Digraph, Source, Graph
import math
from collections import Counter
import itertools
import random

class Node():
    def __init__(self, 
            data=None,
            split_variable=None,
            split_variable_value=None,
            left=None,
            right=None,
            depth=0,
            criterion_value=None):
        self.data = data
        self.split_variable = split_variable
        self.split_variable_value = split_variable_value
        self.left = left
        self.right = right
        self.criterion_value = criterion_value
        self.depth = depth
    

class DecisionTreeRegressor():
    def __init__(self,
                root=None,
                criterion='mse',
                max_depth=2,
                significance=None,
                significance_threshold=3.841,
                min_samples_split=10):
        self.root = root
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.significance = significance
        self.significance_threshold = significance_threshold

        self.split_score_funcs = {'mse': self._calculate_mse_values}
    
    def _get_mse(self, X):
        if X.empty:
            return 0

        # calculate the mean square error with respect to the mean
        y=X['Y']
        y_mean = np.mean(y)
        mse = np.mean((y-y_mean)**2)
        return mse
    
    def _calculate_mse_values(self, X, feature):
        # calculate unique values of X
        # for a feature, there are different values on which that feature can be split
        classes = X[feature].unique()

        # calculate the gini value for a split on each unique value of the feature
        best_mse_score = np.iinfo(np.int32(10)).max
        best_feature_value=""
        for unique_value in classes:
            # split data
            left_split = X[X[feature] <= unique_value]
            right_split = X[X[feature] <= unique_value]

            # get the gini scores of left, right nodes
            mse_value_left_split = self._get_mse(left_split)
            mse_value_right_split = self._get_mse(right_split)

            # combine the 2 scores to get the overall score for the split
            mse_score_of_current_value = (left_split.shape[0]/X.shape[0])*mse_value_left_split+(right_split.shape[0]/X.shape[0])*mse_value_right_split
            if mse_score_of_current_value < best_mse_score:
                best_mse_score=mse_score_of_current_value
                best_feature_value=unique_value
        return best_mse_score, best_feature_value
    
    def _get_best_split_feature(self, X):
        best_split_score = np.iinfo(np.int32(10)).max
        best_feature = ""
        best_value = None
        columns = X.drop('Y', 1).columns
        for feature in columns:
            # calculate the best split score and the best value for the current feature
            split_score, feature_value = self.split_score_funcs[self.criterion](X, feature)

            # compare this feature's split score with the current best score
            if split_score < best_split_score:
                best_split_score = split_score
                best_feature = feature
                best_value = feature_value
        return best_feature, best_value, best_split_score
    
    def _split_data(self, X, X_depth=None):
        # return if dataframe is empty, depth exceeds maximum depth or sample size exceeds minimum sample size required to split
        if X.empty \
            or len(X['Y'].value_counts())==1 \
            or X_depth == self.max_depth \
            or X.shape[0] <= self.min_samples_split:
            return None, None, "", "", 0
        # calculate the best feature to split X
        best_feature, best_value, best_score=self._get_best_split_feature(X)

        if best_feature=="":
            return None, None, "", "", 0
        # create left and right nodes
        X_left = Node(data=X[X[best_feature]<=best_value].drop(best_feature, 1), depth=X_depth+1)
        X_right = Node(data=X[X[best_feature]>best_value].drop(best_feature, 1), depth=X_depth+1)
        
        return X_left, X_right, best_feature, best_value, best_score
    
    def _fit(self, X):
        # handle the initial case
        if not(type(X)==Node):
            X = Node(data=X)
            self.root=X
        
        # get the splits
        X_left, X_right, best_feature, best_value, best_score=self._split_data(X.data, X.depth)

        # assign attributes of node X
        X.left = X_left
        X.right = X_right
        X.split_variable=best_feature
        X.split_variable_value=round(best_value,3) if type(best_value)!=str else best_value
        X.criterion_value=round(best_score, 3)
    
        # return if no best variable found to split on
        # this mean you have reached the leaf node
        if best_feature == "":
            return
        
        # recurse for left and right children
        self._fit(X_left)
        self._fit(X_right)

    def fit(self, X, y):
        # combine the 2 and fit
        X = pd.DataFrame(X)
        X['Y'] = y
        self._fit(X)
    
    def predict(self, X):
        X = np.asarray(X)
        X = pd.DataFrame(X)

        preds = []
        for index, row in X.iterrows():
            curr_node = self.root
            while (curr_node.left != None and curr_node.right != None):
                split_variable = curr_node.split_variable
                split_variable_value = curr_node.split_variable_value
                if X.loc[index, split_variable] <= split_variable_value:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right
            # get prediction
            preds.append(np.mean(curr_node.data['Y'].values))
        # preds.append(np.mean(curr_node.data['Y'].values))
        return preds
    
    def display_tree_structure(self):
        tree = Digraph('DecisionTree', 
                        filename='tree.dot',
                        node_attr={'shape': 'box'})
        tree.attr(size = '10, 20')
        root = self.root
        id = 0

        # queue with nodes to process
        nodes = [(None, root, 'root')]
        while nodes:
            parent, node, x = nodes.pop(0)

            # generate approriate labels for the nodes
            value_counts_length = len(node.data['Y'].value_counts())
            if node.split_variable != "":
                split_variable = node.split_variable
                split_variable_value = node.split_variable_value
            else:
                split_variable='None'
            if value_counts_length > 1:
                label = str(split_variable) + '\n' + str(self.criterion)+" = "+ str(node.criterion_value)
            else:
                label = 'None'
            
            # make edges between the nodes
            tree.node(name=str(id),
                    label = label,
                    color='black',
                    fillcolor='goldenrod2',
                    style='filled')
            if parent is not None:
                if x == 'left':
                    tree.edge(parent, str(id), color='sienna',
                            style = 'filled', label='<='+' '+str(split_variable_value))
                else:
                    tree.edge(parent, str(id), color='sienna',
                            style='filled', label='>'+' '+str(split_variable_value))
            if node.left is not None:
                nodes.append((str(id), node.left, 'left'))
            if node.right is not None:
                nodes.append((str(id), node.right, 'right'))
            id += 1
        return tree
    
    def get_error(self, y, y_hat):
        return np.mean((y-y_hat)**2)
    

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
        decision_tree_regression=DecisionTreeRegressor()
        X_train = trainset[independent_feature_list]
        X_validate = validateset[independent_feature_list]
        y_train = trainset[dependent_feature]
        y_validate = validateset[dependent_feature]

        self.X_train = np.asarray(X_train)
        self.X_validate = np.asarray(X_validate)
        self.y_train = np.asarray(y_train)
        self.y_validate = np.asarray(y_validate)

        decision_tree_regression.fit(self.X_train, self.y_train)
        self.decision_tree_regression = decision_tree_regression
        y_pred = decision_tree_regression.predict(X_validate)
        loss = decision_tree_regression.get_error(y_validate, y_pred)
        print('loss for decision tree regression is: ', loss)
        return loss

        # if validation_method=='k-fold':
    def predict(self, independent_feature_list):
        X_predict = self.data_for_test(independent_feature_list)
        y_predict = self.decision_tree_regression.predict(X_predict)
        return y_predict

def main():
    # sample usage
    feature_list = ['age', 'worstgrade5', 'worstgrade4', 'worstgrade3']
    pipeline = Pipeline(filename='preprocessed_data.csv', dependent_feature='year', independent_feature_list=feature_list)
    pipeline.train(train_part=0.7)
    pipeline.predict(feature_list)

if __name__=="__main__":
    main()
