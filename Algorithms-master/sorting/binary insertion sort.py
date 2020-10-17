# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
from math import floor
import time

def binary_insertion_sort(a):
    for j in range(1, len(a)):
        x = a[j]
        start = i = 0
        end = j
        # binary search
        while start < end:
            m = floor((start + end)/2 )
            if x > a[m]:
                start = m+1
            else:
                end = m
        i = start
        # insertion
        for k in range(j-i):
            a[j-k] = a[j-k-1]
        a[i] = x
    return a

def generate_random_sequence(x):
    a = []
    for i in range(x):
        a.append(random.randint(-100, 100))
    return a

def test(n):
    for trial in range(n):
        a = generate_random_sequence()
        assert binary_insertion_sort(a) == sorted(a)
        print('trial {trial}/{n} passed'.format_map(vars()))

if __name__ == '__main__':
    test(1000000)


def insertion_sort(a):
    for i in range(len(a)):
        temp = a[i]
        j = i - 1
        while j >= 0 and temp < a[j]:
            a[j+1] = a[j]
            j -= 1
        a[j+1] = temp
    return a

lst = generate_random_sequence(10000)

time1 = time.time()
insertion_sort(lst)
time2 = time.time()
print(time2 - time1)

time3 = time.time()
binary_insertion_sort(lst)
time4 = time.time()
print(time4 - time3)


