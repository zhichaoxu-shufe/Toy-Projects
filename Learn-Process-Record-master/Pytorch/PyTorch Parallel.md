PyTorch Parallel

#### Multi-Processing Best Practices

**torch.multiprocessing** is a drop in replacement for Python's **multiprocessing** module. It supports the exact same operations, but extends it, so that all tensors sent through a **multiprocessing.Queue**, will have their data moved into shared memory and will only send a handle to another process.

When a **Tensor** is sent to another process, the **Tensor** data is shared. If **torch.Tensor.grad** is not **None**, it is also shared. After a **Tensor** without a **torch.Tensor.grad** field is sent to the other process, it creates a standard process-specific **.grad** **Tensor** that is not automatically shared across all processes, unlike how the **Tensor**'s data has been shared.

#### CUDA in multiprocessing

The CUDA runtime does not support the fork start method. However, **multiprocessing** in Python 2 can only create subprocesses using **fork**. So Python 3 and either **spawn** or **forkserver** start method are required to use CUDA in subprocesses.

Unlike CPU tensors, the sending process is required to keep the original tensor as long as the receiving process retains a copy of the tensor. It is implemented under the hood but requires user to follow the best practives for the program to run correctly. For example, the sending process must stay alive as long as the consumer has references to the tensor, and the refcounting can not save you if the consumer process exits abnormally via a fatal signal.

#### Asynchronous multiprocess training

Using **torch.multiprocessing**, it is possible to train a model asynchronously, with parameters either shared all the time, or being periodically synchronized. In the first case, we recommend sending over the whole model object, while in the latterm we advice to only send the **state_dict()**.

We recommend using **multiprocessing.Queue** for passing all kinds of PyTorch objects between processes. It is possible to e.g. inherit the tensors and storages already in shared memory, while using the **fork** start method, however it is very bug prone and should be used with care, and only by advanced users. 

```python
import torch.multiprocessing as mp
from model import MyModel

def train(model):
	for data, labels in data_loader:
		optimizer.zero_grad()
		loss_fn(model(data), labels).backward()
		optimizer.step() # this will update the shared parameters

if __name__ == '__main__':
	num_processes = 4
	model = MyModel()

	model.share_memory()
	for rank in range(num_processes):
		p = mp.Process(target=train, asgs=(model, ))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
```

