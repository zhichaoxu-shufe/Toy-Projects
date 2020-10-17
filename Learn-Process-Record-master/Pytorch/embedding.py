# embedding understanding

import torch
import torch.nn as nn

# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(10, 3)

# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
print(embedding(input))

# example with padding_idx
embedding = nn.Embedding(10, 3, padding_idx=0)
input = torch.LongTensor([[0, 2, 0, 5]])
print(embedding(input))


# CLASSMETHOD from_pretrained
# creates Embedding instance from given 2-dimensional FloatTensor

# FloatTensor containing pretrained weights
weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)

# get embeddings for index 1
input = torch.LongTensor([1])
print(embedding(input))