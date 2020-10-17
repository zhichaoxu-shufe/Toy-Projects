# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:13:21 2018

@author: zcxu
"""

'''
heapify -> convert a list/array into a binary max heap
push_heap -> pushes a value onto the heap, maintaining the heap property
pop_heap -> pops the max value from the heap, maintaining the heap property
replace_key -> replace a value on the heap with a different one
'''

def heaptify(A):
    # turns a list A into a max-ordered binary heap
    n = len(A)-1
    # start at last parent and go left one node at a time
    for node in range(n/2, -1, -1):
        __siftdown(A, node)
    # this is important because the _siftdown function is constructed in the way that 
    # the node can only be sifted backwards
    return

def push_heap(A, val):
    # pushes a value onto the heap A while keeping the heap property
    # intact, The heap size increase by 1
    A.append(val)
    __siftdown(A, len(A)-1)
    # furthest left node
    return

def pop_heap(A):
    # returns the max value from the heap A while keeping the heap
    # property intact, The heap size decrease by 1
    n = len(A)-1
    __swap(A, 0, n)
    max = A.pop(n)
    __siftdown(A, 0)
    return max

def replace_key(A, node, newval):
    # replace the key at node 'node' om the max-heap 'A' by newval
    # the heapsize do not change
    curval = A[node]
    A[node] = newval
    # increase key
    if newval > curval:
        __siftup(A, node)
    # decrease key
    elif newval < curval:
        __siftdown(A, node)
    return

def __swap(A, i, j):
    # the pythonic swap
    A[i], A[j] = A[j], A[i]
    return

def __siftdown(A, node):
    # traverse down a binary tree 'A' starting at node 'node'
    # and turn it into a max-heap
    child = 2*node +1
    # base case, stop recursing when we hit the end of the heap
    if child > len(A)-1:
        return
    # check that second child exists: if so, find max
    if(child+1 <= len(A)-1) and (A[child+1] > A[child]):
        child += 1
    # preserves heap structure
    if A[node] < A[child]:
        __swap(A, node, child)
        __siftdown(A, child)
    else:
        return

def __siftup(A, node):
    # traverse up an otherwise max-heap 'A' starting at node 'node'
    # which is the only node that breaks the heap property abd restore
    # the heap structure
    parent = (node-1)/2
    if A[parent] < A[node]:
        __swap(A, node, parent)
    # base case; we have reached the top of the heap
    if parent <= 0:
        return 
    else:
        __siftup(A, parent)






















