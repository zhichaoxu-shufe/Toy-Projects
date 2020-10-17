# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 01:21:29 2018

@author: zcxu
"""

class Node:
    __slots__ = "value", "link"
    
    def __init__(self, value, link=None):
        # creating new node
        # param value: value to be stored in the new node
        # param link: the node linked
        self.value = value
        self.link = link
    
    def __str__(self):
        # return a string representation of the contents of this node
        return str(self.value)

class PriorityQueue():
    
    __slots__ = "front", "back", "after"
    
    def __init__(self, after):
        # initialize a new empty priority queue
        # param after: an ordering function, See definition of deque method
        # return: None (constructor)
        self.after = after
        self.front = None
        self.back = None
    
    def __str__(self):
        # return a string representation of the contents of this queue
        # front value first
        result = "Queue["
        p = self.front
        while p != None:
            result += " " + str(p.value)
            p = p.link
        result += " ]"
        return result
    
    def isEmpty(self):
        # return: Turn iff there are no elements in the queue
        return self.front == None
    
    def enqueue(self, newValue):
        # enter a new value into the queue
        # param newValue: the value to be entered into the queue
        # return: None
        newNode = Node(newValue)
        if self.front == None:
            self.front = newNode
        else:
            self.back.link = newNode
        self.back = newNode
    
    insert = enqueue

    def dequeue(self):
        # remove one of the values v from the queue such that for all values u in the queue
        # after (v, u) is False
        # If more than one value satisfies the requirement, the value chosen should be the
        # one that has been in the queue the longest
        # pre: not isEmpty()
        # return: None
        if self.isEmpty():
            print("Empty queue")
        else:
            self.front = self.front.link
            if self.front == None:
                self.back = None
    
    remove = dequeue
    
    def peek(self):
        # find in the queue the value that would be removed were the dequeue method
        # to be called at this time
        # pre: not isEmpty()
        # return: the value described above
        if self.isEmpty():
            "Empty Queue"
        else:
            np = self.front
            nv = np
            while np != None:
                nv = self.front
                nvr = nv.link
                while nvr != None:
                    if self.after(nv.value, nvr.value):
                        tmp = np.value
                        np.value = nv.value
                        nv.value = tmp
                    nvr = nvr.link
                np = np.link
            return self.front.value
    
    def afterFunction(v, u):
        # method to determine v object should be removed before u object or not
        # return: true if value v should be removed after u
        return v>u
    










































    