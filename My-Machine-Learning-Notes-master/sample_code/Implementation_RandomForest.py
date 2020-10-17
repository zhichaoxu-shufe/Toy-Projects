from collections import Counter, defaultdict
from functools import partial
import math, random

class RandomForest():
    def __init__(self):
        pass
    
    def entropy(self, class_propabilities):
        # given a list of class probabilities, compute the entropy
        return sum(-p * math.log(p, 2) for p in class_propabilities if p)
    
    def class_propabilities(self, labels):
        total_count = len(labels)
        return [count / total_count for count in Counter(labels).values()]

    def data_entropy(self, labeled_data):
        labels = [label for _, label in labeled_data]
        probabilities = self.class_propabilities(labels)
        return self.entropy(probabilities)
    
    def partition_entropy(self, subsets):
        # find the entropy from this partition of data into subsets
        total_count = sum(len(subset) for subset in subsets)
    
        return sum(self.data_entropy(subset) * len(subset) / total_count for subset in subsets)
    
    def group_by(self, items, key_fn):
        # returns a defaultdict(list), where each input item is in the list whose key is key_fn(item)
        groups = defaultdict(list)
        for item in items:
            key = key_fn(item)
            groups[key].append(item)
        return groups

    def partition_by(self, inputs, attribute):
        # returns a dict of inputs partitioned by the attribute each input is a pair (attributed_dict, attribute)
        partitions = self.partition_by(inputs, attribute)
        return self.partition_entropy(partitions.values())
    
    def partition_entropy_by(self, inputs, attribute):
        # compute the entropy correspondning to the given partition
        partitions = self.partition_by(inputs, attribute)
        return self.partition_by(partitions.values())

    def classify(self, tree, input_):
        # classify the input using the given decision_tree

        # if a leaf node, return its value
        if tree in [True, False]:
            return tree
        # otherwise find the correct subtree
        attribute, subtree_dict = tree

        subtree_key = input_.get(attribute) # None if input is missing attributes

        # if leaf node, use the None subtree
        if subtree_key not in subtree_dict:
            subtree_key = None
        
        # choose the appropriate subtree
        subtree = subtree_dict[subtree_key]
        return self.classify(subtree, input_)
    
    def build_tree_id3(self, inputs, split_candidates=None):
        # if first pass
        # all keys of the first input are split candidates
        if split_candidates == None:
            split_candidates = inputs[0][0].keys
        
        # count Trues and Falses in the input
        num_inputs = len(inputs)
        num_trues = len([label for item, label in inputs if label])
        num_falses = num_inputs - num_trues

        if num_trues == 0:
            return False
        elif num_falses == 0:
            return True
        if not split_candidates:
            # do a majority vote
            return num_trues >= num_falses

        # otherwise, split on the best attribute
        best_attribute = min(split_candidates, key=partial(self.partition_entropy_by, inputs))

        partitions = self.partition_by(inputs, best_attribute)
        new_candidates = [a for a in split_candidates if a != best_attribute]

        # recursively build the subtrees
        subtrees = {attribute: self.build_tree_id3(subset, new_candidates) for attribute, subset in partitions.iteritems()}

        subtrees[None] = num_trues > num_falses

        return (best_attribute, subtrees)

    
    def forest_classify(self, trees, input):
        votes = [self.classify(tree, input) for tree in trees]
        vote_counts = Counter(votes)
        return vote_counts.most_common(1)[0][0]