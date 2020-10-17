# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 23:54:41 2018

@author: zcxu
"""

class Node:
    def __init__(self, item, next=None):
        self.item = item
        self.next = next
    
    def getItem(self):
        return self.item
    
    def getNext(self):
        return self.next
    
    def setItem(self, item):
        self.item = item
    
    def setNext(self, next):
        self.next = next
    
    def getPrevious(self):
        return self.previous
    
    def setPrevious(self, item):
        self.previous = item


class LinkedList:
    class __Node:
        def __init__(self, item, next=None):
            self.item = item
            self.next = next
    
        def getItem(self):
            return self.item
    
        def getNext(self):
            return self.next
    
        def setItem(self, item):
            self.item = item
    
        def setNext(self, next):
            self.next = next
    
    def __init__(self, contents=[]):
        # keep a reference to the first node in the linked list
        # and the last item in the linked list
        # they both point to a dummy node to begin with
        self.first = LinkedList.__Node(None, None)
        self.last = self.first
        self.numItems = 0
        
        for e in contents:
            self.append(e)
    
    def __getitem__(self, index):
        if index >= 0 and index < self.numItems:
            cursor = self.first.getNext()
            for i in range(index):
                cursor = cursor.getNext()
            return cursor.getItem()
        raise IndexError("LinkedList index out of range")
        
    def __setitem__(self, index, val):
        if index >= 0 and index < self.getNext():
            cursor = self.first.getNext()
            for i in range(index):
                cursor = cursor.getNext()
            cursor.setItem(val)
            return
        raise IndexError("LinkedList assignment index out of range")

    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError("Concate undefined for " + str(type(self)) + " + " + str(type(other)))
        result = LinkedList()
        
        cursor = self.first.getNext()
        
        while cursor != None:
            result.append(cursor.getItem())
            cursor = cursor.getNext()
        
        cursor = other.first.getNext()
        
        while cursor != None:
            result.append(cursor.getItem())
            cursor = cursor.getNext()
        
        return result

        # notice that the dummy node from both lists is skipped when concatenating the two lists
        # the dummy node in the new list was created when the constructor was called
    
    def append(self, item):
        node = LinkedList.__Node(item)
        self.last.setNext(node)
        self.last = node
        self.numItems += 1

    def insert(self, index, item):
        cursor = self.first
        
        if index < self.numItems:
            for i in range(index):
                cursor = cursor.getNext()
            
            node = LinkedList.__node(item, cursor.getNext())
            cursor.setNext(node)
            self.numItems += 1
        else:
            self.append(item)

    def delete(self, index):
        cursor = self.first
        
        if index < self.numItems:
            for i in range(index):
                cursor = cursor.getNext()
            cursor.next = cursor.next.next
            cursor.item = cursor.next.item
        else:
            raise IndexError("LinkedList assignment index out of range")
            
    def numItems(self):
        cursor = self.first
        count = 0
        while cursor.next != None:
            count += 1
        return count





























