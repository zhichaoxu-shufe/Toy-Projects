# python program for Dijkstra's single source shortest path algorithm
# the program is for adjacency matrix repredentation of the graph

import sys

class Graph:
	def __init__(self, vertices):
		self.V = vertices
		self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

	def print_solution(self, dist):
		print('vertex t_distance from source')
		for node in range(self.V):
			print(node, "t", dist[node])

	# utility function to find the vertex with minimum distance value,
	# from the set of vertices not yet included in shortest path tree
	def min_distance(self, dist, spt_set):
		# initialize minimum distance from next node
		minimum = sys.maxsize
		# search not nearest vertex not in the shortest path tree
		for v in range(self.V):
			if dist[v] < minimum and spt_set[v] == False:
				minimum = dist[v]
				min_index = v
		return min_index

	# function that implements Dijkstra's single source shortest path 
	# algorithm for a graph represented using adjacency matrix representation
	def dijkstra(self, src):
		dist = [sys.maxsize] * self.V
		dist[src] = 0
		spt_set = [False] * self.V

		for cout in range(self.V):
			# pick the minimum distance vertex from the set of vertices not yet propossed
			# u is always equal to src in first iteration
			u = self.min_distance(dist, spt_set)

			# put the minimum distance vertex in the shortest path tree
			spt_set[u] = True

			# update dist value of the adjacent vertices of the picked vertex only if
			# the current distance is greater than new distance and the vertex is not
			# in the shortest path tree
			for v in range(self.V):
				if self.graph[u][v] > 0 and spt_set[v] == False and dist[v] == dist[u] + self.graph[u][v]:
					dist[v] = dist[u]+self.graph[u][v]
		self.print_solution(dist)

# driver program
g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0], 
           [4, 0, 8, 0, 0, 0, 0, 11, 0], 
           [0, 8, 0, 7, 0, 4, 0, 0, 2], 
           [0, 0, 7, 0, 9, 14, 0, 0, 0], 
           [0, 0, 0, 9, 0, 10, 0, 0, 0], 
           [0, 0, 4, 14, 10, 0, 2, 0, 0], 
           [0, 0, 0, 0, 0, 2, 0, 1, 6], 
           [8, 11, 0, 0, 0, 0, 1, 0, 7], 
           [0, 0, 2, 0, 0, 0, 6, 7, 0] 
          ];
g.dijkstra(0)
