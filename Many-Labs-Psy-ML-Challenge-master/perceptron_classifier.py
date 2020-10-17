import pandas as pd
import numpy as np
import random

class Perceptron:
    def __init__(self, input_count):
        self.weights = []
        self.k = 0.01
        for i in range(input_count):
            self.weights.append(random.uniform(-1, 1))
    
    def feed_forward(self, inputs):
        sigma=0
        # print(inputs)
        if len(inputs) == 1:
            inputs = inputs[0]
        for index in range(len(inputs)):
            # print(inputs)
            sigma += inputs[index]*self.weights[index]
            # print(sigma)
        return self.activate(sigma)

    def activate(self, sigma):
        # print(sigma)
        if sigma > 0:
            return 1.0
        else:
            return -1.0
    
    def fit(self, inputs, desired):
        for j in range(len(inputs[:][0])):
            guess = self.feed_forward(inputs[j])
            error_delta = desired[j] - guess
            for i, w in enumerate(self.weights):
                self.weights[i] += error_delta*inputs[j][i]*self.k
    
    def predict(self, inputs):
        predicted=[]
        # print(inputs.shape)
        # print(type(inputs))
        # print(inputs.shape)
        # print(inputs)
        # print(inputs[-1])
        for j in range(inputs.shape[0]):
            # print(j)
            # print(type(inputs[j]))
            if self.feed_forward(inputs[j])==1.0:
                predicted.append(1.0)
            else:
                predicted.append(0.0)
        # print(inputs)
        # print(predicted)
        return predicted

    def calculate_accuracy(self, y_pred, y):
        count = 0
        for index in range(len(y)):
            if y_pred[index] == y[index]:
                count+=1
        # print('the classifying accuracy is: ', count/len(y))
        return count/len(y)

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
        input_count = len(independent_feature_list)
        perceptron_classifier=Perceptron(input_count)
        X_train = trainset[independent_feature_list]
        X_validate = validateset[independent_feature_list]
        y_train = trainset[dependent_feature]
        y_validate = validateset[dependent_feature]

        self.X_train = np.asarray(X_train)
        self.X_validate = np.asarray(X_validate)
        self.y_train = np.asarray(y_train)
        self.y_validate = np.asarray(y_validate)

        # print(self.X_train.shape)
        self.perceptron_classifier = Perceptron(input_count)
        perceptron_classifier.fit(self.X_train, self.y_train)

        y_pred = perceptron_classifier.predict(self.X_validate)
        accuracy = perceptron_classifier.calculate_accuracy(self.y_validate, y_pred)
        print('accuracy for perceptron classifier is: ', accuracy)
        return accuracy

        # if validation_method=='k-fold':
    def predict(self, independent_feature_list):
        X_predict = self.data_for_test(independent_feature_list)
        y_predict = self.perceptron_classifier.predict(X_predict)
        return y_predict

def main():
    # sample usage
    feature_list = ['age', 'worstgrade5', 'worstgrade4', 'worstgrade3']
    pipeline = Pipeline(filename='whatever_you_want_to_call_it.csv', dependent_feature='somehow', independent_feature_list=feature_list)
    pipeline.train(train_part=0.7)
    # pipeline.predict(feature_list)

if __name__=="__main__":
    main()
