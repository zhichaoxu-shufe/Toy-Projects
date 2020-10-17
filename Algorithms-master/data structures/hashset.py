# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 09:47:48 2018

@author: zcxu
"""

# Python let's the programmer have some control over hash code by implementing
# a __hash__ method on a class
# if you write a __hash__ method for a class you can return whatever hash value
# integer you like of that class
# We can use a hash value to compute an index into a list to obtain O(1) item
# lookup complexity.

class HashSet:
    def __init__(self, contents=[]):
        self.items = [None]*10
        self.numItems = 0
        for item in contents:
            self.add(item)
    # to store an item in a hashset we first compute its idnex using the hash function
    # 1.The list that items are stored in must be finite in length and definitely cannot
    # be as long as the unique hash values we would generate by calling the hash function
    # Since the list must be shorter than the maximum hash values, we pick a size for our
    # list and then divide hash values by the length of this list.
    # The remainder is used as the index into the list.
    # The remainder after dividing by the length of the list will always be between 0 and
    # the length of the list minus one even if the hash value is a negative value.
    # Hash  values are not necessarily unique.
    
    # Collision Resolution
    # When two objects need to be stored at the same index within the hash set list, because
    # their computed indices are identical, we call this a collision.
    # Linear Probing
    # When a collision occurs while using linear probing, we advance to the next location in
    # the list to see if that location might be available. We can tell if a location is available
    # if we find a None value in that spot in the list.
    # It turns out that there is one other value we might find in the list that means that location
    # is available. A special type of object called a __Placeholder object might also be stored in
    # the list.
    # For now, a None or a __Placeholder object indicates an open location within the hashset list.
    def __add(item, items):
        idx = hash(item) % len(items)
        loc = -1
        
        while items(idx) != None:
            if items[idx] == item:
                # item already in set
                return False
            if loc < 0 and type(items[idx] == HashSet.__Placeholder):
                loc = idx
            idx = (idx+1)%len(items)
        if loc < 0:
            loc = idx
        items(loc) = item
        return True
    # The fullness of the hash set list is valled its load factor
    # We can find the load factor of a hashset by dividing the number of items stored in the list
    # by its length.
    def __rehash(oldList, newList):
        for x in oldList:
            if x != None and type(x) != HashSet.__Placeholder:
                HashSet.__add(x, newList)
        return newList
    
    def add(self, item):
        if HashSet.__add(item, self.items):
            self.numItems += 1
            load = self.numItems/len(self.items)
            if load >= 0.75:
                self.items = HashSet.__rehash(self.items, [None]*2*len(self.items))
    
    def remove(self, item):
        if HashSet.__remove(item, self.items):
            self.numItems -= 1
            load = max(self.numItems, 10)/len(self.items)
            if load <= 0.25:
                self.items = HashSet.rehash(self.items, [None]*int(len(self.items)/2))
        else:
            raise KeyError("Item not in HashSet")
    
    def __contains__(self, item):
        idx = hash(item)%len(self.items)
        while self.items[idx] != None:
            if self.items[idx] == item:
                return True
            idx = (idx+1)% len(self.items)
        return False
    
    # to iterate over the items of a set
    # traverse the list of items skipping over placeholder elements and None references
    def __iter__(self):
        for i in range(len(self.items)):
            if self.items[i] != None and type(self.items[i]) != HashSet.__PlaceHolder:
                yield self.items[i]
    
    # help function used in self.difference()
    def difference_update(self, other):
        for item in other:
            self.discard(item)
    
    # returns a new set which consists of the differences of self and other set
    def difference(self, other):
        result = HashSet(self)
        result.difference_update(other)
        return result
    
    def __remove(item, items):
        idx = hash(item)%len(items)
    
        while items[idx] != None:
            if items[idx] == item:
                nextIdx = (idx+1) % len(items)
                if items[nextIdx] == None:
                    items[idx] = None
                else:
                    items[idx] = HashSet.__Placeholder()
        return True
        idx = (idx+1)%len(items)
    return False

    # hashset remove helper function
    class __Placeholder:
        def __init__(self):
            pass
        def __eg__(self, other):
            return False
    
    def discard(self, item):
        pass
    
    def pop(self):
        pass
    
    def clear(self):
        pass
    
    def update(self, other):
        pass
    
    def intersection_update(self, other):
        pass
    
    