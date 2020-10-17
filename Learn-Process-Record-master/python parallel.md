#### Introduction

Parallel processing is a mode of operation where the task is executed simultaneously in a multiple processors in the same computer. It is meant to reduce the overall processing time.

However, there is usually a bit of overhead when communicating between processes which can actually increase the overall time taken for small tasks instead of decreasing it.

In python, the **multiprocessing** module is used to run independent parallel processes by using subprocesses (instead of threads). It allows you to leverage multiple processors on a machine, which means the processes can be run in completely separate memory locations.

#### How many maximum parallel processes can you run?

The maximum number of processes you can run at a time is limited by the number of processors in your computer.

```python
import multiprocessing as mp
print('Number of processors: ', mp.cpu_count())
```

#### Synchronous and Asynchronous Execution

A synchronous execution is one the processes are completed in the same order in which it was started. This is achieved by locking the main program until the respectively processes are finished.

Asynchronous, on the other hand, doesn't involve locking. As a result, the order of results can get mixed up but usually gets done quicker.

There are two main objects in **multiprocessing** to implement parallel execution of a function: The **Pool** Class and the **Process** Class.

- **Pool** Class:
  1. Synchronous execution:
     - **Pool.map()** and **Pool.starmap()**
     - **Pool.apply()**
  2. Asynchronous execution
     - **Pool.map_async()** and **Pool.starmap_async()**
     - **Pool.apply_async()**

#### How to parallelize any function

The general way to parallelize any operation is to take a particular function that should be run multiple times and make it run parallelly in different processors.

To do this, you initialize a **Pool** with n number of processors and pass the function you want to parallelize to one of the **Pool**s parallization methods.

**multiprocessing.Pool()** provides the **apply(), map(), starmap()** methods o make any function run in parallel.

```python
import numpy as np
import time

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[20000, 5])
data = arr.tolist()
print(data[:5])

# Solution Without Paralleization
def how_many_within_range(row, minimum, maximum):
	# returns how many numbers lie within 'maximum' and 'minimum' in a given row
	count = 0
	for n in row:
		if minimum <= n <= maximum:
			count += 1
	return count

results = []
for row in data:
	results.append(how_many_within_range(row, 4, 8))

print(results[:10])

# Parallize the how_many_within_range() function using multiprocessing.Pool()

# Parallelizing using Pool.apply()
import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: 'pool.apply' the 'how_many_within_range()'
results = [pool.apply(how_many_within_range, args=(row, 4, 8)) for row in data]

# Step 3: Don't forget to close
pool.close()


print(results[:10])
```

