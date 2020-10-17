# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 23:27:18 2018

@author: zcxu
"""

def merge(seq, start, mid, stop):
    lst = []
    i = start
    j = mid
    
    # merge the two lists while each has more elements
    while i<mid and j<stop:
        if seq[i] < seq[j]:
            lst.append(seq[i])
            i += 1
        else:
            lst.append(seq[j])
            j += 1
    # copy in the rest of the start to mid sequence
    while i < mid:
        lst.append(seq[i])
        i += 1
    # copy the elements back to the original sequence
    for i in range(len(lst)):
        seq[start+i] = lst[i]


def mergeSortRecursively(seq, start, stop):
    # we must use >= here only when the sequence we are sorting is empty
    # otherwise start == stop-1 in the base case
    if start >= stop-1:
        return
    mid = (start+stop)//2
    
    mergeSortRecursively(seq, start, mid)
    mergeSortRecursively(seq, mid, stop)
    merge(seq, start, mid, stop)

def mergeSort(seq):
    mergeSortRecursively(seq, 0, len(seq))

    