import heapq

class Node:
    def __init__(self, point, parent = None):
        self.point = point
        self.parent = parent
    
    def nextMove(x, y, pacman_x, pacman_y, food_x, food_y, grid):
        path = ucs((pacman_x, pacman_y), (food_x, food_y), (x, y), grid)
        if path != None:
            print(len(path) - 1)
            print('\n'.join([str(i[0]) + " " + str(i[1] for i in path)]))
        return
    
    def children(point, size, grid):
        x, y = point
        size_x, size_y = size
        children = [(x-1, y), (x, y - 1), (x, y + 1), (x + 1, y)]
        return [child for child in children if grid[child[0]][child[1]] != '%']
    
    def ucs(node, goal, size, grid):
        # initialize the queue with the root node
        q = [(0, node, [])]
        # the list of seen items
        seen = {}
        # while the queue is not empty
        while q != None:
            # pop the cost, point and path from the queue
            cost, point, path = headq.heappop(q)
            # if it has been seen, and has a lower cost, bail
            if seen.has_key(point) and seen[point] < cost:
                continue
            # update the path
            path = path + [point]
            # if we have found the goal, return the point
            if point == goal:
                return path
            # loop through the children
            for child in children(point, size, grid):
                # calculate the basic cost
                child_cost = 1 if i == goal else 0
                # if the child hasn't been seen
                if child not in seen:
                    # add it to the heap
                    heapq.heappush(q, (cost+child_cost, child, path))
            # add the point to the seen items
            seen[point] = cost
        return None
# tails ends here


pacman_x, pacman_y = [ int[i] for i in raw_input().strip().split() ]
food_x, food_y = [ int(i) for i in raw_input().strip().split() ]
x, y = [ int(i) for i in raw_input().strip().split() ]

grid = []
for i in np.range(0, x):
    grid.append(raw_input().strip())

nextMove(x, y, pacman_x, pacman_y, food_x, food_y, grid)