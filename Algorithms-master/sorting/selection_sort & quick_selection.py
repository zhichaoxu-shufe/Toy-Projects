# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:32:50 2018

@author: zcxu
"""

import random

x = []
for i in range(100):
    x.append(int(round(random.random()*10000)))
    
def selectSort(lst): # sort from two directions
    srted = 0
    while srted < len(lst)/2:
        small = srted
        large = srted
        for i in range(srted+1, len(lst)-srted):
            if lst[i] < lst[small]:
                small = i
            if lst[i] > lst[large]:
                large = i
        lst[srted], lst[small] = lst[small], lst[srted]
        lst[len(lst)-1-srted], lst[large] = lst[large], lst[len(lst)-1-srted]
        srted += 1
    print(lst)

import time
startTime = time.time()
selectSort(x)
endTime = time.time()
print(endTime - startTime)

# quickselection algorithm
# this algorithms is designed to find the kthlargest(smallest) element in an array
# using the divide & conquer idea

# auxilary function for quickselect
def partition(a, l, r):
    x = a[r]
    i = l-1
    for j in range(l, r):
        if a[j] <= x:
            i += 1
            a[i], a[j] = a[j], a[i]
        else:
            pass

    a[i+1], a[r] = a[r], a[i+1]
    return i+1

# kthlargest(a, k)
# this function return the kth largest element in the array
def kthlargest(a, k):
    l = 0
    r = len(a) - 1
    # choosing a pivot and saving its index
    split_point = partition(a, l, r)
    # if the chosen element is the correct element, then return it
    if split_point == r-k+1:
        result = a[split_point]
    elif split_point > r-k+1:
    # if the element is in the left part to the pivot, then call kthlargest on the left half
        result = kthlargest(a[:split_point], k-(r-split_point+1))
    else:
        result = kthlargest(a[split_point+1: r+1], k)
    return result