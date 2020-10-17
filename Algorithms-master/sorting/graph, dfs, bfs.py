# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:44:19 2018

@author: zcxu
"""

class Node:
    def __init__(self, value):
        self.value = value
        self.adjacentNodes = []
    
def buildGraph(vertexNames, edges):
    vertices = dict([(vertexNames[i], Node(vertexNames[i])) for i in range(len(vertexNames))])
    
    for name in vertices:
        vertices[name].value = name
    for (v, w) in edges:
        vertices[v].adjacentNodes.append(vertices[w])
    
    return vertices[vertexNames[0]]

from collections import deque

def breadthFirst(startingNode, soughtValue):
    visitedNodes = set()
    queue = deque([startingNode])
    
    while len(queue) > 0:
        node = queue.pop()
        if node in visitedNodes:
            continue
        
        visitedNodes.add(node)
        if node.value == soughtValue:
            return True
        
        for n in node.adjacentNodes:
            if n not in visitedNodes:
                queue.appendleft(n)
    return False

class FoundNodeException(Exception):
    pass

def recursiveDepthFirst(node, soughtValue):
    try:
        recursiveDepthFirstWorker(node, soughtValue, set())
        return False
    except FoundNodeException:
        return True

def recursiveDepthFirstWorker(node, soughtValue, visitedNodes):
    if node.value == soughtValue:
        raise FoundNodeException
    
    visitedNodes.add(node)
    
    for adjNode in node.adjacentNodes:
        if adjNode not in visitedNodes:
            recursiveDepthFirstWorker(adjNode, soughtValue, visitedNodes)

def depthFirst(startingNode, soughtValue):
    # using a stack
    visitedNodes = set()
    stack = [startingNode]
    
    while len(stack) > 0:
        node = stack.pop()
        if node in visitedNodes:
            continue
        
        visitedNodes.add(node)
        if node.value == soughtValue:
            return True
        
        for n in node.adjacentNodes:
            if n not in visitedNodes:
                stack.append(n)
    return False


def search(startingNode, soughtValue, pile):
    visitedNodes = set()
    nodePile = pile()
    nodePile.add(startingNode)
    
    while len(nodePile) > 0:
        node = nodePile.remove()
        if node in visitedNodes:
            continue
        
        visitedNodes.add(node)
        if node.value == soughtValue:
            return True
        
        for n in node.adjacentNodes:
            if n not in visitedNodes:
                nodePile.add(n)
    return False

class MyStack(deque):
    def add(self, item):
        self.append(item)
    
    def remove(self):
        return self.pop()

class MyQueue(deque):
    def add(self, item):
        self.appendleft(item)
    
    def remove(self):
        return self.pop()

