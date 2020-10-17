import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax

def sigmoid(x):
	# compute the sigmoid function for the input here
	s = 1/(1+np.exp(-x))
	return s

def naiveSoftmaxLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset):
	# naive softmax loss & gradient function for word2vec models
	'''
	Implement the naive softmax loss and gradients between a center word's embedding
	and an outside word's embedding. This will be the building block for word2vec model
	
	Arguments:
	centerWordVec -- numpy ndarray, center word's embedding
	outsideWordIdx -- integer, the index of the outside word
	outsideVectors -- outside vectors (rows of matrix) for all words in vocab
	dataset -- needed for negative sampling, unused here

	Return:
	loss -- naive softmax loss
	gradCenterVec -- the gradient with respect to the center word vector
	gradOutsideVecs -- the gradient with respect to all the outside word vectors
	'''

	y_hat = softmax(np.dot(centerWordVec, outsideVectors.T))
	delta = y_hat.copy()
	delta[outsideWordIdx] -= 1

	loss = -np.log(y_hat)[outsideWordIdx]
	gradCenterVec = np.dot(delta, outsideVectors)
	gradOutsideVecs = np.dot(delta[:, np.newaxis], centerWordVec[np.newaxis, :])

	return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
	# sample K indexes which are not the outsideWordIdx
	negSampleWordIndices = [None] * K
	for k in range(K):
		newidx = dataset.sampleTokenIdx()
		while newidx == outsideWordIdx:
			newidx = dataset.sampleTokenIdx()
		negSampleWordIndices[k] = newidx
	return negSampleWordIndices

def negSamplingLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset, K=10):
	'''
	Negative sampling loss function for word2vec models

	Implement the negative sampling loss and gradients for a centerWordVec and a outsideWordIdx
	word vector as a building block for word2vec models. K is the number of negative samples
	to take.

	Note: The same word may be negatively sampled multiple times. For example if an outside word
	is sampled twice, you shall have to double count the gradient with respect to this word.

	Arguments/Return Specification: same as naiveSoftmaxLossAndGradient
	'''
	negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
	indices = [outsideWordIdx] + negSampleWordIndices

	gradOutsideVecs = np.zeros(outsideVectors.shape)
	gradCenterVec = np.zeros(centerWordVec.shape)
	loss = 0.0

	z = sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec))
	loss -= np.log(z)

	gradOutsideVecs[outsideWordIdx] += centerWordVec * (z-1.0)
	gradCenterVec += outsideVectors[outsideWordIdx] * (z-1.0)

	# vectorized implementation
	u_k = outsideVectors[negSampleWordIndices]
	z = sigmoid(-np.dot(u_k, centerWordVec))
	loss += np.sum(-np.log(z))
	gradCenterVec += np.dot((z-1), u_k) * (-1)
	gradOutsideVecs[negSampleWordIndices] += np.outer((z-1), centerWordVec)*(-1)

	return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind, centerWordVectors, outsideVectors, dataset, word2vecLossAndGradient = naiveSoftmaxLossAndGradient):
	# skip-gram model in word2vec
	'''
	Arguments:
	currentCenterWord -- a string of the current center word
	windowSize -- integer, context window size
	outsideWords -- list of no more than 2*windowSize strings, the outside words
	word2Ind -- a dictionary that maps words to their indices in the word vector lis
	centerWordVectors -- center word vectors (as rows) for all words in vocab
	outsideVectors -- outside word vectors (as rows) for all words in vocab
	word2vecLossAndGradient -- the loss and gradient function for a prediction vector given
								the outsideWordInx word vectors, could be one of the two
								loss functions you implemented above

	Return:
	loss -- the loss function value for the skip-gram model
	gradCenterVecs -- the gradient with respect to the center word vectors
	gradOutsideVectors -- the gradient with respect to the outside word vectors
	'''
	currentCenterWordIdx = word2Ind[currentCenterWord]
	centerWordVec = centerWordVectors[currentCenterWordIdx]

	for outsideWord in outsideWords:
		outsideWordIdx = word2Ind[outsideWord]
		(l, gradCenter, gradOutside) = word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset)
		loss += 1
		gradCenterVecs[currentCenterWordIdx] += gradCenter
		gradOutsideVectors += gradOutside

	return loss, gradCenterVecs, gradOutsideVectors


def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")
    print ("Skip-Gram with naiveSoftmaxLossAndGradient")

    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
        )
    )

    print ("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print ("Skip-Gram with negSamplingLossAndGradient")
    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
            dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)
        )
    )
    print ("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)

if __name__ == "__main__":
    test_word2vec()