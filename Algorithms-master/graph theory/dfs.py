# print DFS traversal for complete graph
from collections import defaultdict

# this class represents a directed graph using adjacency list representation
class Graph:
	# constructor
	def __init__(self):
		# default dictionary to store graph
		self.graph = defaultdict(list)

	# function to add an edge to store graph
	def add_edge(self, u, v):
		self.graph[u].append(v)

	# a function used by DFS
	def DFS_util(self, v, visited):
		# mark the current node as visited and print it
		visited[v] = True
		print(v)

		# recur for all the vertices adjacent to this vertex
		for i in self.graph[v]:
			if visited[i] == False:
				self.DFS_util(i, visited)

	# the function to df DFS traversal.
	def DFS(self):
		# total vertices
		V = len(self.graph)

		# mark all the vertices as not visited
		visited = [False]*(V)

		# call the recursive util function to print
		# DFS traversal starting from all vertices one by one
		for i in range(V):
			if visited[i] == False:
				self.DFS_util(i, visited)

# Driver code 
# Create a graph given in the above diagram 
g = Graph() 
g.add_edge(0, 1) 
g.add_edge(0, 2) 
g.add_edge(1, 2) 
g.add_edge(2, 0) 
g.add_edge(2, 3) 
g.add_edge(3, 3) 
  
# print "Following is Depth First Traversal"
g.DFS() 