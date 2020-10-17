# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:30:50 2018

@author: zcxu
"""

# introsort & quicksort
# introsort is an improved sort algorithm based on the quicksort algorithm
# this method starts from quicksort
# and when the recursive depth is over than a certain amount, it convert to
# the heapsort
# through this convertion
# introsort could implement good result in regular dataset
# and maintain O(nlogn) time complexity in the worst case

import random

def findPivot(begin, end):
    return random.randint(begin, end)
    # return begin

def partition(array, begin, end, *, reverse = False):
    pivotIdx = findPivot(begin, end)
    pivot = array[pivotIdx]
    
    array[end], array[pivotIdx] = array[pivotIdx], array[end]
    firstLarger = begin
    for idx in range(begin, end):
        if reverse^(array[idx] <= pivot):
            array[idx], array[firstLarger] = array[firstLarger], array[idx]
            firstLarger += 1
    
    array[end], array[firstLarger] = array[firstLarger], array[idx]
    
    return firstLarger

def quickSort(array, begin=0, end=None, *, reverse=False):
    if end == None:
        end = len(array)-1
    if begin < end:
        mid = partition(array, begin, end, reverse = reverse)
        quickSort(array, begin, mid-1, reverse = reverse)
        quickSort(array, mid+1, end, reverse = reverse)

# heapsort
def percolateDown(heap, idx, maxIdx = None, *, reverse=False):
    if maxIdx == None:
        maxIdx = len(heap)-1
    while idx < maxIdx:
        largestIdx = idx
        if 2*idx+1 <= maxIdx and (reverse^(heap[2*idx+1] > heap[largestIdx])):
            largestIdx = 2*idx+1
        if 2*idx+2 <= maxIdx and (reverse^(heap[2*idx+2] > heap[largestIdx])):
            largestIdx = 2*idx+2
        if largestIdx != idx:
            heap[idx], heap[largestIdx] = heap[largestIdx], heap[idx]
            idx = largestIdx
        else:
            break

def heapify(heap, maxIdx = None, *, reverse=False):
    if maxIdx == None:
        maxIdx = len(heap)-1
    for idx in range(maxIdx // 2, -1, -1):
        percolateDown(heap, idx, reverse=reverse)

def heapSort(heap, reverse = False):
    heapify(heap, reverse = reverse)
    for idx in range(len(heap)-1, 0, -1):
        # iterate through all the nodes
        heap[0], heap[idx] = heap[idx], heap[0]
        percolateDown(heap, 0, idx-1, reverse=reverse)
    return heap

# introSort
from math import log2

def introSort(array, begin=0, end=None, depth=0, *, reverse=False):
    if end == None:
        end = len(array)-1
    
    if depth < log2(len(array)):
        if begin < end:
            mid = partition(array, begin, end, reverse=reverse)
            introSort(array, begin, mid-1, depth+1, reverse=reverse)
            introSort(array, mid+1, end, depth+1, reverse=reverse)
    else:
        array[begin:end+1] = heapSort(array[begin:end+1], reverse=reverse)

if __name__ == '__main__':
    a = [2,4, 5, 1, 5, 9, 1, -3]
    b = [2,4, 5, 1, 5, 9, 1, -3]
    c = [2,4, 5, 1, 5, 9, 1, -3]
    quickSort(a, reverse = True)
    heapSort(b, reverse = True)
    introSort(a, reverse=True)
    print(a)
    print(b)
    print(c)
