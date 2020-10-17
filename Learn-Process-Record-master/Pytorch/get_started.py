from __future__ import print_function
import torch

# construct a 5*3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)

# construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# construct a matrix filled zeros and of dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

# create a tensor based on an existing tensor
# these methods will reuse properties of the input tensor
# unless new values are provided by user
x = x.new_ones(5, 3, dtype = torch.double)
print(x)
print(x.size())

# override dtype
x = torch.randn_like(x, dtype=torch.float)
print(x.size())

# torch.Size is in fact a tuple, so it supports all tuple operations

y = torch.rand(5, 3)
print(x+y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# in-place addition
y.add_(x)
print(y)

# any operation that mutates a tensor in-place is post-fixed with an _
print(x[:, 1])

# resizing
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())


# for a one-element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())

# the Torch Tensor and Numpy array will share their underlying memory locations,
# and changing one will change the other

# converting a torch tensor to a numpy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

