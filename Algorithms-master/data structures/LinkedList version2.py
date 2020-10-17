# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 00:44:10 2018

@author: zcxu
"""

class Node(object):
    def __init__(self, data, next, previous):
        self.data = data
        self.next = next
        self.previous = previous
    
    def getData(self):
        return self.data
    
    def getNext(self):
        return self.next
    
    def getPrevious(self):
        return self.previous
    
    def setData(self, data):
        self.data = data
    
    def setNext(self, aNode):
        self.next = aNode
    
    def setPrevious(self, aNode):
        self.previous = aNode

class LinkedList(object):
    def __init__(self):
        self.head = None
        self.size = 0
    
    def isEmpty(self):
        return self.size == 0
    
    def getSize(self):
        return self.size
    
    def getHead(self, aNode):
        self.head = aNode
    
    def insertLast(self, data):
        newNode = Node(data, None, None)
        if self.isEmpty():
            self.setHead(newNode)
        else:
            temp = self.head
            while temp.getNext() != None:
                temp = temp.getNext()
            temp.setNext(newNode)
        self.size += 1
    
    def insertFirst(self, data):
        newNode = Node(data, None, None)
        newNode.setNext(self.getHead())
        self.setHead(newNode)
        self.size += 1
    
    def deleteLast(self):
        if self.isEmpty() is not True:
            temp = self.getHead()
            while temp.getNext().getNext() is not True:
                temp = temp.getNext()
            
            temp.setNext(None)
            self.size -= 1
        
    
    def deleteFirst(self):
        if self.isEmpty() is not True:
            self.setHead(self.head.getNext())
            self.size -= 1
    
    def getContent(self):
        result = []
        temp = self.getHead()
        
        while temp:
            result.append(temp.getData())
            temp = temp.getNext()
        
        print(result)
    
    # finds the first occurance of the data and returns its index
    def find(self, data):
        index = 0
        temp = self.getHead()
        
        while temp != None:
            if temp.getData() == data:
                return index
            index += 1
            temp = temp.getNext()
        return -1




































