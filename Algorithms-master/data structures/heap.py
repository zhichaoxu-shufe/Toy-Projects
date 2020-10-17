# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 03:31:53 2018

@author: zcxu
"""

class Queue:
    def __init__(self):
        self.items = []
        self.frontIdx = 0
    
    def __compress(self):
        newlst = []
        for i in range(self.frontIdx, len(self.items)):
            newlst.append(self.items[i])
        self.items = newlst
        self.frontIdx = 0
    
    def dequeue(self):
        if self.isEmpty():
            raise RuntimeError("Attempt to dequeue an empty queue")
        
        # when queue is half full, compress it
        if self.frontIdx*2 > len(self.items):
            self.__compress()
        item = self.items[self.frontIdx]
        self.frontIdx += 1
        return item
    
    def enqueue(self, item):
        self.items.append(item)
    
    def front(self):
        if self.isEmpty():
            raise RuntimeError("Attempt to access front of empty queue")
        return self.items[self.frontIdx]
    
    def isEmpty(self):
        return self.frontIdx == len(self.items)
