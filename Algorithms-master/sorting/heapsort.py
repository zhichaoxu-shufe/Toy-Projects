from sys import stdin

# Max Heapify
# adjust the end son-node of the heap, such that the son-node
# is always less than father-node
def Heapify(A, i, end):
	n = end;
	l = 2*i + 1
	r = 2 * (i+1)
	if (l <= n-1 and A[l] > A[i]):
		largest = l
	else:
		largest = i
	if (r <= n-1 and A[r] > A[largest]):
		largest = r
	if (largest != i):
		A[i], A[largest] = A[largest], A[i]
		Heapify(A, largest, end)

# Build the Max Heap:
# re-sort all the data in the heap
# this could be implemented iteratively using func Heaptify
def BuildMaxHeap(A):
	x = int((len(A) - 2)/2)
	for i in range(x, -1, -1):
		Heaptify(A, i, len(A))

# HeapSort
# remove the root-node at the first data
# and adjust the BuildMaxHeap iteratively
def HeapSort(A):
	solution = []
	n = len(A) - 1
	BuildMaxHeap(A)
	solution.append(A)
	print(*A, sep = '', end = '')
	for i in range(len(A)-1, 0, -1):
		A[0], A[i] = A[i], A[0]
		Heaptify(A, 0, i)
		print('|', end = '')
		print(*A, sep = '', end = '')
	print(' ; ')

for line in stdin:
	H = [int(elem) for elem in line.split()]
	HeapSort(H)
