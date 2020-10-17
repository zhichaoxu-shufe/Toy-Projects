# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:57:47 2018

@author: zcxu
"""

class HashMap:
    class __KVPair:
        def __init__(self, key, value):
            self.key = key
            self.value = value
    
        def __eq__(self, other):
            if type(self) != type(other):
                return False
            
            return self.key == other.key
        
        def getKey(self):
            return self.key
        
        def getValue(self):
            return self.value
        
        def __hash__(self):
            return hash(self.key)
    
    def __init__(self):
        self.hSet = hashset.HashSet()
    
    def __len__(self):
        return len(self.hSet)
    
    def __contains__(self, items):
        return HashMap.__KVPair(item, None) in self.hSet
    
    def not__contains__(self, item):
        return item not in self.hSet
    
    def __setitem__(self, key, value):
        self.hSet.add(HashMap.__KVPair(key, value))
        
    def __getitem__(self, key):
        if HashMap.__KVPair(key, None) in self.hSet:
            return self.hSet[HashMap.__KVPair(key, None)].getValue()
        else:
            return default
    
    def __delitem__(self, key):
        if HashMap.__KVPair(key, None) in self.hSet:
            self.hSet.remove(key)
        else:
            raise KeyError("Key "+key+" not in HashMap")
    
    def items(self):
        result = []
        for x in self.hSet:
            result.append((x.getKey(), x.getValue()))
        return result
    
    def keys(self):
        result = []
        for x in self.hSet:
            result.append(x.getKey())
        return result
    
    def values(self):
        result = []
        for x in self.hSet:
            result.append(x.getValue())
        return result
    
    def pop(self, key):
        if HashMap.__KVPair(key, None) in self.hSet:
            item = self.hSet(key)
            return item.getValue()
        else:
            raise KeyError("Key "+key+" not in HashMap")
    
    def popitem(self):
        item = self.hSet.pop()
        return (item.getKey(), item.getValue())
    
    def setdefault(self):
        pass
    
    def update(self, other):
        pass
    
    def clear(self):
        pass
    
    def copy(self):
        pass
    
    def __iter__(self):
        for x in self.hSet:
            yield x.getKey()
