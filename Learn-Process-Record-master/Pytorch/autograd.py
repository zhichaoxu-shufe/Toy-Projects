import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)


a = torch.randn(1, requires_grad = True, dtype=torch.float).to(device)
b = torch.randn(1, requires_grad = True, dtype=torch.float).to(device)

# print(a, b)

torch.manual_seed(42)

a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

lr = 1e-1
n_epochs = 1000


# torch.autograd.backward
# computes the sum of gradients of given tensors w.r.t. graph leaves
# The graph is differentiated using the chain rule. If any of tensors
# are non-scalar and require gradient, then the Jacobian-vector
# product would be computed, in this case, the function additionally
# requires specifying grad_tensors. It should be a sequence of matching
# length, that contains the "vector" in the Jacobian-vector product
# usually the gradient of the differentiated function w.r.t. corresponding
# tensors.


for epoch in range(n_epochs):
	yhat = a + b * x_train_tensor
	error = y_train_tensor - yhat
	loss = (error**2).mean()

	loss.backward()
	print(a.grad)
	print(b.grad)

	with torch.no_grad():
		a -= lr * a.grad
		b -= lr * b.grad

	a.grad.zero_()
	b.grad.zero_()