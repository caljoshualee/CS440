# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

class Node:
    def __init__(self, pos, parent, cost):
        self.pos = pos
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.cost = cost
        # parent is another node
        self.parent = parent
    def __lt__(self, other):
        return self.cost < other.cost

def bfs(maze, ispart1=False):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 

    Args:
        maze: Maze instance from maze.py
        ispart1: pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    """
    queue = deque()
    retPath = []
    visited = {
        maze.getStart(): 0
    }
    start = Node(maze.getStart(), None, 0)

    queue.append(start)
    while queue:
        curr = queue.popleft()
        if maze.isObjective(curr.x, curr.y, curr.z, ispart1):
            retPath.append(curr.pos)
            parent = curr.parent
            while parent:
                retPath.append(parent.pos)
                parent = parent.parent
            retPath.reverse()
            return retPath

        neighbors = maze.getNeighbors(curr.x, curr.y, curr.z, ispart1)
        for neighbor in neighbors:
            if neighbor not in visited or visited[neighbor] > curr.cost + 1:
                visited[neighbor] = curr.cost + 1
                queue.append(Node(neighbor, curr, curr.cost + 1))
    return None