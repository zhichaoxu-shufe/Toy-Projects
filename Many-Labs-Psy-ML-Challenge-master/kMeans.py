import numpy as np
import time
from scipy import stats
import math
import pandas as pd
import queue

class Kmeans():
    """Kmeans聚类算法.

    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数. 
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon, 
        则说明算法已经收敛
    """

    def euclidean_distance(self, one_sample, X):
        one_sample = one_sample.reshape(1, -1)
        X = X.reshape(X.shape[0], -1)
        distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
        return distances

    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids
    
    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = self.euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            if len(X[cluster]) == 0:
                return [-1]
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)

        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids
            
            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)
            if len(centroids) == 1 and isinstance(centroids[0], int) and centroids[0] == -1:
                return [False]
            
            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break
            
        return self.get_cluster_labels(clusters, X)

def getData():
    # load data
    data = pd.read_csv('preprocessed_data.csv', index_col = 0)
    noNanData = pd.DataFrame.copy(data)

    # set NaN with means
    average = {}
    for name, value in data.iloc[0:1,].iteritems():
        average[name] = 0

    for index, row in data.iterrows():
        for name, value in row.iteritems():
            if not math.isnan(value):
                average[name] += value

    for key, value in average.items():
        average[key] = value / data.shape[0]

    colNames = data.columns
    dataValue = data.values
    for index, row in enumerate(dataValue):
        for index2, value in enumerate(row):
            if math.isnan(value):
                dataValue[index][index2] = average[colNames[index2]]
    return data, pd.DataFrame(data = dataValue, columns=colNames)

# mode: 0 for average, 1 for majority
def getError(labels, predictCol, noNANData, testNum, mode):
    rows = noNANData.shape[0]
    testData = noNANData.iloc[(rows - testNum):rows, ]
    averageDiff = 0

    if mode == 0:
        predictValue = {}
        predictNum = {}
        for index, row in noNANData.iloc[0:rows - testNum, ].iterrows():
            row = row.to_dict()
            if not math.isnan(row[predictCol]):
                predictValue[int(labels[index])] = predictValue.get(int(labels[index]), 0) + row[predictCol]
                predictNum[int(labels[index])] = predictNum.get(int(labels[index]), 0) + 1
        for key, value in predictValue.items():
            predictValue[key] /= predictNum[key]

        for index, row in testData.iterrows():
            row = row.to_dict()
            if math.isnan(row[predictCol]):
                continue
            averageDiff += abs(row[predictCol] - predictValue[labels[index]])
    elif mode == 1:
        labeledGroup = {}
        for index, row in noNANData.iloc[0:rows - testNum, ].iterrows():
            row = row.to_dict()
            if not math.isnan(row[predictCol]):
                if not int(labels[index]) in labeledGroup:
                    labeledGroup[int(labels[index])] = []
                
                labeledGroup[int(labels[index])].append(row[predictCol])

        majority = {}
        dataNum = 0
        for key, value in labeledGroup.items():
            majority[key] = stats.mode(value)[0][0]
        for index, row in testData.iterrows():
            row = row.to_dict()
            if math.isnan(row[predictCol]):
                continue
            averageDiff += abs(row[predictCol] - majority[labels[index]])
            dataNum += 1

    return averageDiff / dataNum

def KmeansPredict(predictCol, k, noNANData):
    estimator = Kmeans(k)
    trainData = noNANData.drop(predictCol, axis=1)
    labels = estimator.predict(trainData.values)
    while len(labels) == 1 and labels[0] == False:
        print('run again!')
        labels = estimator.predict(trainData.values)
    return labels

def findBestModel(featureName, noNANData, k, mode):
    errors = []
    bestError = getError(KmeansPredict(featureName, k, noNANData), featureName, noNANData, 200, mode)
    dropList = []
    # find appropriate features
    for name in noNANData.columns.values:
        print('Check if drop ', name)
        if name == featureName:
            continue
        dropList.append(name)
        trainData = noNANData.drop(dropList, axis=1)
        error = getError(KmeansPredict(featureName, k, trainData), featureName, noNANData, 200, mode)
        if error > bestError:
            dropList.remove(name)
        else:
            bestError = error
            print(dropList)
    return bestError, dropList

class DistancePoint():
    center = []
    point = []
    pointIndex = 0
    distance = 0
    mode = 0
    # mode 0: Euclidean Distance
    def __init__(self, center, point, index, mode):
        self.center = center
        self.point = point
        self.mode = mode
        self.pointIndex = index
        if mode == 0:
            distance = self.euclidean()
    
    def __lt__(self, other):
        if self.distance < other.distance:
            return False
        else:
            return True
    
    def euclidean(self):
        distance = 0
        for index, value in enumerate(self.point):
            distance += (value - self.center[0][index]) * (value - self.center[0][index])
        self.distance = math.sqrt(distance)

def KNN(k, trainData, testData, mode, featureIndex, voteMode):
    noFeatureTestData = np.delete(testData, featureIndex, axis=1)
    noFeatureTrainData = np.delete(trainData, featureIndex, axis=1)
    que = queue.PriorityQueue()
    for index, value in enumerate(noFeatureTrainData):
        que.put(DistancePoint(noFeatureTestData, value, index, mode))
        if que.qsize() > k:
            que.get()
    voteData = []
    while que.qsize() > 0:
        voteData.append(trainData[que.get().pointIndex][featureIndex])

    if voteMode == 0:
        return np.mean(voteData)
    elif voteMode == 1:
        return stats.mode(voteData)[0][0]


def KMeansMain(noNANData, mode):
    bestError, dropList = findBestModel('stress_01', noNANData, 10, mode)
    print('error:', bestError, ' and the droplist is ', dropList)

def KNNMain(noNANData, featureIndex, testDataSize, mode):
    testData = noNANData[0:testDataSize]
    trainData = noNANData[testDataSize:]
    featureIndex = 8
    k = 3
    errors = 0
    for index, value in enumerate(testData):
        result = KNN(k, trainData, [value], 0, featureIndex, mode)
        errors += abs(result - noNANData[index][featureIndex])
    
    return errors/testDataSize

rawData, noNANData = getData()
# KMeans
mode = 1
#KMeansMain(noNANData, mode)

# KNN
# delete index column
noNANData = np.delete(noNANData.values, 0, axis=1)

featureIndex = 8
testDataSize = 200
mode = 1 # 1 for majority, 0 for mean
error = KNNMain(noNANData, featureIndex, testDataSize, mode)
print(error)