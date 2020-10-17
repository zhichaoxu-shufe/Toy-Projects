
# topological sorting for Directed Acyclic Graph (DAG) is a linear ordering
# of vertices such that for every directed edge uv, vertex u comes before v
# in the ordering.
# topological sorting for a graph is not possible if the graph is not a DAG

from collections import defaultdict

# class to represent a graph
class Graph:
	def __init__(self, vertices):
		# dictionary containing adjacency list
		self.graph = defaultdict(list)
		self.V = vertices

	# function to add an edge to graph
	def add_edge(self, u, v):
		self.graph[u].append(v)

	# a recursive function used by topological sort
	def helper(self, v, visited, stack):
		# mark the current node as visited
		visited[v] = True
		# recur for all the vertices adjacent to this vertex
		for i in self.graph[v]:
			if visited[i] == False:
				self.helper(i, visited, stack)
		stack.insert(0, v)

	# the function to do topological sort
	def topological_sort(self):
		# mark all the vertices are not visited
		visited = [False]*self.V
		stack = []

		# call the recursive helper function to store topological 
		# sort starting from all vertices one by one
		for i in range(self.V):
			if visited[i] == False:
				self.helper(i, visited, stack)

		# print contents of the stack
		print(stack)

g= Graph(6) 
g.add_edge(5, 2); 
g.add_edge(5, 0); 
g.add_edge(4, 0); 
g.add_edge(4, 1); 
g.add_edge(2, 3); 
g.add_edge(3, 1); 
  
# print "Following is a Topological Sort of the given graph"
g.topological_sort() 
