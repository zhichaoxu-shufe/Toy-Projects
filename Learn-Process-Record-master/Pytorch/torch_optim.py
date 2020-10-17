# the implementation of PyTorch Optim package
# 
# A fully-connected ReLU network with one hidden layer, trained to predict y from x
# by minimizing squared Euclidean distance
#
# This implementation uses the nn package from PyTorch to build the network
#
#
# Rather than manually updating the weights of the model as we have been doing, 
# we use the optim package to define an Optimizer that will update the weights
# for us. The optim package defines many optimization algorithms that are commonly
# used for deep learning, including SGD+momentum, RMSProp, Adam, etc

import torch

# batch size, input dimension, 
# hidden dimension, output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# define model and loss function
model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H),
	torch.nn.ReLU(),
	torch.nn.Linear(H, D_out)
	)

loss_fn = torch.nn.MSELoss(reduction='sum')

# use optim package to define an Optimizer that will update the weights of the model

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
for t in range(500):
	# forward pass
	y_pred = model(x)

	# compute and print loss
	loss = loss_fn(y_pred, y)
	print(t, loss.item())

# before the backward pass, use the optimizer object to zero all of the gradients
# for the variables it will update (which are the learnable weights of the model)

optimizer.zero_grad()

# backward pass: compute gradient of the loss with respect to model parameters
loss.backward()

# calling the step function on an Optimizer makes an update to its parameters
optimizer.step()