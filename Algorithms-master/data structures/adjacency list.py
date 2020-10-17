# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 02:29:50 2018

@author: zcxu
"""

from collections import deque

def vertices_and_edges(file):
    # helper function to convert text file into a list of vertices and edges
    # listing vertices, and then the edges(with weights if the graph is weighted)
    vertices = [vertex.strip() for vertex in open(file, 'rU') if len(vertex.strip().split()) == 1]
    edges = [pair.strip().split() for pair in open(file, 'rU') if len(pair.strip().split()) in (2, 3)]
    return vertices, edges

class Adjacency_List(object):
    # a dictionary where the key is a vertex and the value a list of all the vertices
    # it is adjacent to.
    # Include algorithms such as BFS, DFS
    def __init__(self, vertices, edges, directed=False, weighted=False):
        if not isinstance(directed, bool):
            raise TypeError("Directed must be a boolean value")
        if not isinstance(weighted, bool):
            raise TypeError("Weighted must be a boolean value")
        
        self._directed = directed
        self._weighted = weighted
        
        self._adjacency_list = { vertex : [] for vertex in vertices}
        self._edge_list = [tuple(edge[:2]) for edge in edges]
        
        # what if we made self._weighted a dictionary of dictionaries
        # instead of d[(a, b)], you'd have d[a][b]
        # so it's more of an adjacency matrix, and it there is no edge then
        # it returns None
        self._weight = {} 
        # weighted[(a, b)] -> int
        
        # could probably make an add_edge(A, B) or something to make this easier
        for edge in edges:
            if self._weighted and len(edge) != 3:
                raise ValueError('Only two vertices and an edge per tuple')
            if not self._weighted and len(edge) != 2:
                raise ValueError('Only two vertices per tuple')
            
            v1, v2 = edge[:2]
            self._adjacency_list[v1].append(v2)
            self._weight[(v1, v2)] = int(edge[2]) if self._weighted else 1
            # an unweighted graph has edges of weight 1
            if not self._directed:
                self._adjacency_list[v2].append(v1)
                self._weight[(v2, v1)] = int(edge[2]) if self._weighted else 1
                
    def __str__(self):
        # string representation of the adjavency list
        if self._weighted:
            return '\n'.join(
                    ['%s:\t%s' % (vertex, ','.join(                                       
                        # each vertex in the adjacency list
                        ['%s: %s' %(joined_vertex, self._weight[(vertex, joined_vertex)])
                        # each joined vertex and its weight
                        for joined_vertex in joined_vertices]))
                    for vertex, joined_vertices in self._adjacency_list.iteritems()])
        else:
            return '\n'.join(['%s: \t%s' % (vertex, ','.join(joined_vertices))
                              for vertex, joined_vertices in self._adjacency_list.iteritems()])
    
    def _repr_(self):
        # official string representation of the adjacency list
        return str(self._adjacency_list) + (('\n' + str(self._weight)) if self._weighted else '') 
    
    # accessor methods
    def is_weighted(self):
        # returns True if the edges of the graph are weighted, False otherwise
        return self._directed
    
    def vertices(self):
        # returns True if the graph as directed, False otherwise
        return self._adjacency_list.keys()
    
    def edges(self):
        # returns a copy of the original list of tuples fed in
        return self._edge_list
    
    def num_vertices(self):
        # returns the number of vertices in the graph
        return len(self._adjacency.keys())
    
    def num_edges(self):
        # returns the number of edges in the graph
        return len(self._edge_list)
    
    def weight(self, v1, v2):
        # returns the weight of an edge between two given vertices
        return self._weight[(v1,v2)] if (v1, v2) in self._weight.keys() else None
    
    
    # traversal algorithms
    def _XFS(self, is_DFS, root=None):
        # X first-search: generalized for both Depth and Breadth-first searches
        if not isinstance(is_DFS, bool):
            raise ValueError("Must specify whether search type is DFS or not (hense BFS)")
        deq = deque()
        marked = { vertex: False for vertex in self._adjacency_list.iterkey() }
        # make all vertices unmarked
        
        root = root or self._adjacency_list.keys()[0]
        deq.append(root)
        marked[root] = True
        XFS = []
        
        while len(deq):
            vertex = deq.pop() if is_DFS else deq.popleft()
            # pop off the stack if DFS, or deque
            XFS.append(vertex)
            for adjacent_vertex in self._adjacency_list[vertex]:
                # grab each unmarked adjacent vertex
                if not marked[adjacent_vertex]:
                    marked[adjacent_vertex] = True
                    # mark it
                    deq.append(adjacent_vertex)
                    # and shove it into the deque
        return XFS

    def BFS(self, root = None):
        # returns a traversal of the graph via the Breadth-first search algorithm
        # if no root is given then the first node in the graph is chosen
        return self._XFS(True, root)
    
    def DFS(self, root = None):
        # returns a traversal of the graph via the breadth-first search algorithm
        # if no root is given, then the first node in the graph is chosen
        return self._XFS(False, root)
    
    # Minimum paths
    def Floyd_Warshall(self):
        # returns a dictionary where each pair (v1, v2) returns the length of the
        # smallest path between them
        if any((weight<0 for weight in self._weight.itervalues())):
            # check if any edges have negative values
            raise ValueError("Graph cannot have negative weights")
        
        vertices = self._adjacency_list.keys()
        path = { (v1, v2): 0 if v1 == v2
                            else self._weight[(v1, v2)] if (v1,v2) in self._weight.keys()
                            else float('inf')
                        for v1 in vertices
                        for v2 in vertices }
        for k in vertices:
            for i in vertices:
                for j in vertices:
                    path[(i, j)] = min(path[(i, j)], path[(i, k)] + path[(k, j)])
        return path
    
    def Dijkstra(self, root=None, target=None):
        # find the shortest path between the root and a target
        vertices = self._adjacency_list.keys()
        distance = { vertex : float('inf') for vertex in vertices}
        # dictionary of all distances from the root to each vertex
        previous = { vertex : None for vertex in vertices}
        # reference to the previous node in the optimal path from the root
        
        root = root or self._adjacency_list.keys()[0]
        distance[root] = 0
        
        while len(vertices):
            vertex = min(vertices, key = lambda v:distance[v])
            # grab the vertex with the smallest distance
            if target == vertex:
            # if a target was specified then we can determine the optimal path
                curr_vertex = target
                # between the source and itself
                min_path = deque()
                while previous[curr_vertex]:
                    # while the previous vertex in the optimal path exists
                    min_path.appendleft(curr_vertex)
                    # smack it to the front of the path
                    curr_vertex = previous[curr_vertex]
                    # grab the previous vertex of the vertex we just appended
                min_path.appendleft(root)
                return list(min_path)
            # we don't really need a deque, just append and then reverse
            
            if distance[vertex] == float('inf'):
                # if all the remaining vertices are detached then we end the algorithm
                break
            
            vertices.remove(vertex)
            for adjacent_vertex in self._adjacency_list[vertex]:
                # for each adjacent vertex of the current vertex
                if adjacent_vertex in vertices:
                    # which hasn't been removed from the vertices
                    alt = distance[vertex] + self._weight[(vertex, adjacent_vertex)]
                    # get the distance of the alternate path from the adjacent 
                    # to the current vertex
                    if alt < distance[adjacent_vertex]:
                        # if this new distance is smaller
                        distance[adjacent_vertex] = alt
                        # then set it as the new distance
                        previous[adjacent_vertex] = vertex
                        # and set previous (optimal) vertex of adjacent vertex to be
                        # current vertex
                        i = vertices.index(adjacent_vertex)
                        # shift adjacent_vertex left in vertices
                        vertices[i], vertices[i-1] = vertices[i-1], vertices[i]
        return distance
                    






































