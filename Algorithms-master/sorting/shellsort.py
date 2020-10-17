# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 00:57:15 2018

@author: zcxu
"""

# shellsort, also known as Shell Sort or Shell's method
# The method starts by sorting pairs of elements far apart from each other
# Then progresssively reducing the gap between elements to be compared
def shellSort(list):
    gap = len(list)//2
    while gap > 0:
        for i in range(gap, len(list)):
            var = list[i]
            j = i
            while j >= gap and list[j-gap] > var:
                list[j] = list[j-gap]
                j -= gap
            list[j] = var
        gap //= 2
    return