# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 11:49:43 2018

@author: zcxu
"""

class ArrayList(object):
    
    def __init__(self):
        self.size = 0
        self.capacity = 0
        self.array = []
    
    def getSize(self):
        return self.size
    
    def getCapacity(self):
        return self.capacity
    
    def setCapacity(self, newCapacity):
        temp = []
        for i in range(newCapacity):
            temp.append(None)
            
        for i in range(len(self.array)):
            temp[i] = self.array[i]
        
        self.array = temp
        self.capacity = newCapacity
    
    def replace(self, index, data):
        if index+1 <= self.capacity:
            self.array.insert(index, data)
            self.capacity += 1
            self.size += 1
    
    def delete(self, index):
        if index+1 <= self.getCapacity():
            self.array.pop(index)
        self.capacity -= 1
        self.size -= 1
    
    def get(self, index):
        if index+1 > self.getCapacity():
            self.getCapacity(index+1)
        return self.array[index]
    
    def getContent(self):
        result = []
        for i in range(len(self.array)):
            result.append(self.array[i])
        print(result)