import numpy as np
import math
import random
import csv

def NaiveBayes():
    def __init__(self):
        pass
    
    def load_dataset(self, filename):
        lines = csv.reader(open(filename, 'rb'))
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
        return dataset
    
    def mean(self, numbers):
        # returns the mean of numbers
        return np.mean(numbers)

    def stdev(self, numbers):
        # return the sigmoid number
        return 1.0/(1.0+math.exp(numbers))
    
    def cross_validation_split(self, dataset, n_folds):
        # split dataset into the k folds, returns the list of k folds
        dataset_split = []
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = random.randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split
    
    def accuracy_metric(self, actual, predicted):
        # calculate accuracy percentage
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
    
    def evaluate_algorithm(self, dataset, algorithm, n_folds):
        # evaluate an algorithm using a cross validation split
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = []
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores
    
    def separate_by_class(self, dataset):
        # split training set by class value
        separated = {}
        for i in range(len(dataset)):
            row = dataset[i]
            if row[-1] not in separated:
                separated[row[-1]] = []
            separated[row[-1]].append(row)
        return separated
    
    def model(self, dataset):
        # find the mean and standard deviation of each feature in dataset
        models = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]
        models.pop()
        # remove last entry because it is class value
        return models
    
    def model_by_class(self, dataset):
        # find the mean and standard deviation of each feature in dataset by their class
        separated = self.separate_by_class(dataset)
        class_models = {}
        for (class_value, instances) in separated.iteritems():
            class_models[class_value] = self.model(instances)
        return class_models
    
    def calculate_pdf(self, x, mean, stdev):
        # calculate probability using gaussian density function
        if stdev == 0.0:
            if x == mean:
                return 1.0
            else:
                return 0.0
        exponent = math.exp(-(math.pow(x-mean, 2) / (2 * math.pow(stdev, 2))))
        return 1/(math.sqrt(2*math.pi) * stdev) * exponent
    
    def calculate_class_probabilities(self, models, input):
        # calculate the class probability for input sample, combine probability of each feature
        probabilities = {}
        for (class_value, class_models) in models.iteritems():
            probabilities[class_value] = 1
            for i in range(len(class_models)):
                (mean, stdev) = class_models[i]
                x = input[i]
                probabilities[class_value] *= self.calculate_pdf(x, mean, stdev)
        return probabilities
    
    def predict(self, models, input_vector):
        # compare probability for each class
        # return the class label which has max probability
        probabilities = self.calculate_class_probabilities(models, input_vector)
        (best_label, best_prob) = (None, -1)
        for (class_value, probability) in probabilities.iteritems():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def get_predictions(self, models, test_set):
        # get class labels for each value in test set
        predictions = []
        for i in range(len(test_set)):
            result = self.predict(models, test_set[i])
            predictions.append(result)
        return predictions
    
    def naive_bayes(self, train, test, ):
        # create a naive bayes model, test the model and returns the testing result
        summaries = self.model_by_class(train)
        predictions = self.get_predictions(summaries, test)
        return predictions
