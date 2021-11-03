
from collections import deque
from typing import List
import heapq
# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives 
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): manhat_dst(i, j)
                for i, j in self.cross(objectives)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

class Node:
    def __init__(self, pos, waypoints, parent, cost):
        self.pos = pos
        self.x = pos[0]
        self.y = pos[1]
        self.cost = cost
        # parent is another node
        self.parent = parent
        self.wpremaining = list(waypoints)
    def __lt__(self, other):
        return self.cost < other.cost
    def removeWaypoint(self, waypoint):
        self.wpremaining.remove(waypoint)


def bfs(maze):
    
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    queue = deque()
    retPath = []
    visited = {
        maze.start: 0
    }
    start = Node(maze.start, maze.waypoints, None, 0)

    queue.append(start)
    while queue:
        curr = queue.popleft()
        if curr.pos == maze.waypoints[0]:
            retPath.append(curr.pos)
            parent = curr.parent
            while parent:
                retPath.append(parent.pos)
                parent = parent.parent
            retPath.reverse()
            return retPath

        neighbors = maze.neighbors(curr.x, curr.y)
        for neighbor in neighbors:
            if neighbor not in visited or visited[neighbor] > curr.cost + 1:
                visited[neighbor] = curr.cost + 1
                queue.append(Node(neighbor, maze.waypoints, curr, curr.cost + 1))

def manhat_dst(a, b):
    return abs(b[1] - a[1]) + abs(b[0] - a[0]) 

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    heap = []
    retPath = []
    visited = {
        maze.start: 0
    }
    start = Node(maze.start, maze.waypoints, None, 0)

    heapq.heappush(heap, (manhat_dst(start.pos, maze.waypoints[0]),start))
    while heap:
        curr = heapq.heappop(heap)[1]
        if curr.pos == maze.waypoints[0]:
            retPath.append(curr.pos)
            parent = curr.parent
            while parent:
                retPath.append(parent.pos)
                parent = parent.parent
            retPath.reverse()
            return retPath

        neighbors = maze.neighbors(curr.x, curr.y)
        for neighbor in neighbors:
            if neighbor not in visited or visited[neighbor] > curr.cost + 1:
                visited[neighbor] = curr.cost + 1
                heapq.heappush(heap, ((curr.cost + 1 + manhat_dst(neighbor, maze.waypoints[0])), Node(neighbor, maze.waypoints, curr, curr.cost + 1)))


def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    WP_CACHE = {}
    MST_CACHE = {}
    heap = []
    retPath = []
    visited = {
        maze.start: 0
    }
    start = Node(maze.start, maze.waypoints, None, 0)
    list = []
    WP_CACHE[maze.start] = {maze.waypoints}
    heapq.heappush(heap, (manhat_dst(start.pos, maze.waypoints[0]),start))
    while heap:
        curr = heapq.heappop(heap)[1]
        if curr.pos in curr.wpremaining:
            curr.removeWaypoint(curr.pos)
            if(not curr.wpremaining):
                retPath.append(curr.pos)
                parent = curr.parent
                while parent:
                    retPath.append(parent.pos)
                    parent = parent.parent
                retPath.reverse()
                return retPath

        neighbors = maze.neighbors(curr.x, curr.y)
        for neighbor in neighbors:
            neighborNode = Node(neighbor, tuple(curr.wpremaining), curr, curr.cost + 1)  
            if (neighbor not in visited) or (visited[neighbor] > curr.cost + 1) or (tuple(curr.wpremaining) not in WP_CACHE[neighbor]):
                minVal = None
                visited[neighbor] = curr.cost + 1
                for waypoint in curr.wpremaining:
                    if(minVal is None):
                        minVal = manhat_dst(curr.pos, waypoint)
                        minWaypoint = waypoint
                    elif(manhat_dst(curr.pos, waypoint) < minVal):
                        minWaypoint = waypoint
                if(tuple(neighborNode.wpremaining) not in MST_CACHE) :
                    MST_CACHE[tuple(neighborNode.wpremaining)] = MST(tuple(neighborNode.wpremaining)).compute_mst_weight() 
                heapq.heappush(heap, ((curr.cost + 1 + manhat_dst(neighbor, minWaypoint) + MST_CACHE[tuple(neighborNode.wpremaining)]), neighborNode))  
                if neighbor in WP_CACHE : 
                    WP_CACHE[neighbor].add(tuple(curr.wpremaining))
                else:
                    WP_CACHE[neighbor] = {tuple(curr.wpremaining)}

def astar_single_target(maze, startc, wp_close):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    heap = []
    retPath = []
    visited = {
        startc: 0
    }
    start = Node(startc, maze.waypoints, None, 0)

    heapq.heappush(heap, (manhat_dst(startc, wp_close),start))
    while heap:
        curr = heapq.heappop(heap)[1]
        if curr.pos == wp_close:
            retPath.append(curr.pos)
            parent = curr.parent
            while parent:
                retPath.append(parent.pos)
                parent = parent.parent
            retPath.reverse()
            return retPath

        neighbors = maze.neighbors(curr.x, curr.y)
        for neighbor in neighbors:
            if neighbor not in visited or visited[neighbor] > curr.cost + 1:
                visited[neighbor] = curr.cost + 1
                heapq.heappush(heap, ((curr.cost + 1 + manhat_dst(neighbor, wp_close)), Node(neighbor, maze.waypoints, curr, curr.cost + 1)))

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    WP_CACHE = {}
    MST_CACHE = {}
    heap = []
    retPath = []
    visited = {
        maze.start: 0
    }
    start = Node(maze.start, maze.waypoints, None, 0)
    list = []
    WP_CACHE[maze.start] = {maze.waypoints}
    heapq.heappush(heap, (manhat_dst(start.pos, maze.waypoints[0]),start))
    while heap:
        curr = heapq.heappop(heap)[1]
        if curr.pos in curr.wpremaining:
            curr.removeWaypoint(curr.pos)
            if(not curr.wpremaining):
                retPath.append(curr.pos)
                parent = curr.parent
                while parent:
                    retPath.append(parent.pos)
                    parent = parent.parent
                retPath.reverse()
                return retPath

        neighbors = maze.neighbors(curr.x, curr.y)
        for neighbor in neighbors:
            neighborNode = Node(neighbor, tuple(curr.wpremaining), curr, curr.cost + 1)  
            if (neighbor not in visited) or (visited[neighbor] > curr.cost + 1) or (tuple(curr.wpremaining) not in WP_CACHE[neighbor]):
                minVal = None
                visited[neighbor] = curr.cost + 1
                for waypoint in curr.wpremaining:
                    if(minVal is None):
                        minVal = manhat_dst(curr.pos, waypoint)
                        minWaypoint = waypoint
                    elif(manhat_dst(curr.pos, waypoint) < minVal):
                        minWaypoint = waypoint
                if(tuple(neighborNode.wpremaining) not in MST_CACHE) :
                    MST_CACHE[tuple(neighborNode.wpremaining)] = MST(tuple(neighborNode.wpremaining)).compute_mst_weight() 
                heapq.heappush(heap, ((curr.cost + 1 + manhat_dst(neighbor, minWaypoint) + 2*MST_CACHE[tuple(neighborNode.wpremaining)]), neighborNode))  
                if neighbor in WP_CACHE : 
                    WP_CACHE[neighbor].add(tuple(curr.wpremaining))
                else:
                    WP_CACHE[neighbor] = {tuple(curr.wpremaining)}