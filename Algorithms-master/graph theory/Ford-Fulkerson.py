# python program for implementation of Ford Fulkerson algorithm

from collections import defaultdict

# this class represents a directed graph using adjacency matrix representation

class Graph:
    
    def __init__(self, graph):
        self.graph = graph 
        # residual graph
        self.row = len(graph)
        # self.COL = len(gr[0])

    '''
    return true if there is a path from source 's' to sink 't' in residual graph
    also fills parent[] to store the path
    '''
    def BFS(self, s, t, parent):

        # mark all the vertices as not visited
        visited = [False] * (self.row)

        # create a queue for BFS
        queue = []

        # mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # standard BFS loop
        while queue:
            # dequeue a vertex from queue and print it
            u = queue.pop(0)

            '''
            get all adjacent vertices of the dequeued vertex u 
            If an adjavent has not been visited, mark it visited and enqueue it
            '''
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
        
        # if we reached sink in BFS starting from source, then return True, else false
        return True if visited[t] else False

# returns the maximum flow from s to t in the given graph
def FordFulkerson(self, source, sink):

    # this array is filled by BFS and to store path
    parent = [-1]*(self.row)

    max_flow = 0
    # there is no flow initially

    # augment the flow while there is path from source to sink
    while self.BFS(source, sink, parent):
        '''
        find minimum residual capacity of the edges along the path filled by BFS.
        Or we can say find the maximum flow through the path found
        '''
        path_flow = float("Inf")
        s = sink
        while (s != source):
            path_flow = min(path_flow, self.graph[parent[s]][s])
            s = sink
        while (s != source):
            path_flow = min(path_flow, self.graph[parent[s]][s])
            s = parent[s]
        
        # add path flow to overall flow
        max_flow += path_flow

        # update residual capacities of the edges and reverse edges along the path
        v = sink
        while (v != source):
            u = parent[v]
            self.graph[u][v] -= path_flow
            self.graph[v][u] += path_flow
            v = parent[v]
    
    return max_flow


if __name__ == "__main__":
    # create a graph
    graph = [[0, 16, 13, 0, 0, 0], 
            [0, 0, 10, 12, 0, 0], 
            [0, 4, 0, 0, 14, 0], 
            [0, 0, 9, 0, 0, 20], 
            [0, 0, 0, 7, 0, 4], 
            [0, 0, 0, 0, 0, 0]]

    g = Graph(graph)

    source = 0
    sink = 5
    print("the maximum possible flow is %d " % FordFulkerson(g, source, sink))
