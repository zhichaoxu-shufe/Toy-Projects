address = 'E:/download/glove.6B/'
filename = 'glove.6B.50d.txt'

glove_vocab = []
glove_embed = []
embedding_dict = {}

file = open(address+filename, 'r', encoding='UTF-8')

for line in file.readlines():
	row = line.strip().split(' ')
	vocab_word = row[0]
	glove_vocab.append(vocab_word)
	embed_vector = [float(i) for i in row[1:]]
	# convert to list of float
	embedding_dict[vocab_word] = embed_vector
	glove_embed.append(embed_vector)

print('Loaded GLOVE')
file.close()


from scipy import spatial
import numpy as np

# look up word vectors and store them as numpy arrays
king = np.array(embedding_dict['king'])
man = np.array(embedding_dict['man'])
woman = np.array(embedding_dict['woman'])

# add/substract vectors
new_vector = king - man + woman

# use a scipy function to create a 'tree' of word vectors that we can run queries against
tree = spatial.KDTree(glove_embed)

# run query with new_vector to find the closest word vectors
nearest_dist, nearest_idx = tree.query(new_vector, 10)
nearest_words = [glove_vocab[i] for i in nearest_idx]

print(nearest_words)
