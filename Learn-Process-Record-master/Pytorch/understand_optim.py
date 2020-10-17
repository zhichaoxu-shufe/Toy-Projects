# torch.optim is a package implementing various optimization algorithms

# How to use
# To use torch.optim, you have to construct an optimizer object, that will
# hold the current state and will update the parameters based on the 
# computed gradients

# construct
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)

# take an optimization step

optimizer.step()
# this is a simplified version supported by most optimizers.
# the function can be called once the gradients are computed using e.g. backward()

for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

optimizer.step(closure)
# some optimization algorithms such as Conjugate Gradient and LBFGS
# need to re-evaluate the function multiple times, so you have to 
# pass in a closure that allows them to recompute your model.
# The closure should clear the gradients, compute the loss, and return it

for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)


# CLASS torch.optim.SGD()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad()
loss_fn(model(input), target).backward()
optimizer.step()

