# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien

#helper function to find distance from point (p1, p2) to line segment
def dist(seg1x, seg1y, seg2x, seg2y, p1, p2):
    dist = (seg2x-seg1x)**2 + (seg2y-seg1y)**2

    if(dist == 0):
        return math.sqrt((p1 - seg1x)**2 + (p2 - seg1y)**2)
    
    pointDist =  ((p1 - seg1x) * (seg2x-seg1x) + (p2 - seg1y) * (seg2y-seg1y)) / float(dist)

    if pointDist > 1:
        pointDist = 1
    elif pointDist < 0:
        pointDist = 0

    x = seg1x + pointDist * (seg2x-seg1x)
    y = seg1y + pointDist * (seg2y-seg1y)

    dx = x - p1
    dy = y - p2

    dist = math.sqrt(dx**2 + dy**2)

    return dist

# used a geeksforgeeks algorithm for intersecting line segments

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False
 
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
     
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.
     
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):
         
        # Clockwise orientation
        return 1
    elif (val < 0):
         
        # Counterclockwise orientation
        return 2
    else:
         
        # Collinear orientation
        return 0
 
# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
     
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
 
    # Special Cases
 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
 
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
 
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
 
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
 
    # If none of the cases
    return False

def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endy), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    tolerance = granularity/math.sqrt(2)

    if alien.is_circle():
        for wall in walls:
            alienX = alien.get_centroid()[0]
            alienY = alien.get_centroid()[1]
            distance = dist(wall[0], wall[1], wall[2], wall[3], alienX, alienY)
            if distance < alien.get_width() + tolerance or np.isclose(distance, alien.get_width() + tolerance):
                return True
    else:
        for wall in walls:
            headX, headY = alien.get_head_and_tail()[0]
            tailX, tailY = alien.get_head_and_tail()[1]
            d1 = dist(wall[0], wall[1], wall[2], wall[3], headX, headY)
            d2 = dist(wall[0], wall[1], wall[2], wall[3], tailX, tailY)
            d3 = dist(headX, headY, tailX, tailY, wall[0], wall[1])
            d4 = dist(headX, headY, tailX, tailY, wall[2], wall[3])
            minDist = min(d1, d2, d3, d4)
            if minDist < alien.get_width() + tolerance or np.isclose(minDist, alien.get_width() + tolerance):
                return True
            if doIntersect(Point(headX, headY), Point(tailX, tailY), Point(wall[0], wall[1]), Point(wall[2], wall[3])):
                return True
    return False


def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    
    if alien.is_circle():
        for goal in goals:
            alienX = alien.get_centroid()[0]
            alienY = alien.get_centroid()[1]
            distance = math.sqrt((goal[0] - alienX)**2 + (goal[1] - alienY)**2)
            if distance < alien.get_width() + goal[2] or np.isclose(distance, alien.get_width() + goal[2]):
                return True
    else:
        for goal in goals:
            headX, headY = alien.get_head_and_tail()[0]
            tailX, tailY = alien.get_head_and_tail()[1]
            distance = dist(headX, headY, tailX, tailY, goal[0], goal[1])
            if distance < alien.get_width() + goal[2] or np.isclose(distance, alien.get_width() + goal[2]):
                return True

    return False

def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    tolerance = granularity/math.sqrt(2)
    wall1 = (0, 0, window[0], 0)
    wall2 = (0, 0, 0, window[1])
    wall3 = (window[0], 0, window[0], window[1])
    wall4 = (0, window[1], window[0], window[1])
    walls = [wall1, wall2, wall3, wall4]

    if alien.is_circle():
        for wall in walls:
            alienX = alien.get_centroid()[0]
            alienY = alien.get_centroid()[1]
            distance = dist(wall[0], wall[1], wall[2], wall[3], alienX, alienY)
            if distance < alien.get_width() + tolerance or np.isclose(distance, alien.get_width() + tolerance):
                return False
    else:
        for wall in walls:
            headX, headY = alien.get_head_and_tail()[0]
            tailX, tailY = alien.get_head_and_tail()[1]
            d1 = dist(wall[0], wall[1], wall[2], wall[3], headX, headY)
            d2 = dist(wall[0], wall[1], wall[2], wall[3], tailX, tailY)
            d3 = dist(headX, headY, tailX, tailY, wall[0], wall[1])
            d4 = dist(headX, headY, tailX, tailY, wall[2], wall[3])
            minDist = min(d1, d2, d3, d4)
            if minDist < alien.get_width() + tolerance or np.isclose(minDist, alien.get_width() + tolerance):
                return False
            if doIntersect(Point(headX, headY), Point(tailX, tailY), Point(wall[0], wall[1]), Point(wall[2], wall[3])):
                return False
    return True

if __name__ == '__main__':
    #Walls, goals, and aliens taken from Test1 map
    walls =   [(0,100,100,100),  
                (0,140,100,140),
                (100,100,140,110),
                (100,140,140,130),
                (140,110,175,70),
                (140,130,200,130),
                (200,130,200,10),
                (200,10,140,10),
                (175,70,140,70),
                (140,70,130,55),
                (140,10,130,25),
                (130,55,90,55),
                (130,25,90,25),
                (90,55,90,25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    def test_helper(alien : Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0) 
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, expected: {truths[0]}'
        assert touch_goal_result == truths[1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, expected: {truths[1]}'
        assert in_window_result == truths[2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, expected: {truths[2]}'

    #Initialize Aliens and perform simple sanity check. 
    alien_ball = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)	
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)	
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)

    alien_positions = [
                        #Sanity Check
                        (0, 100),

                        #Testing window boundary checks
                        (25.6, 25.6),
                        (25.5, 25.5),
                        (194.4, 174.4),
                        (194.5, 174.5),

                        #Testing wall collisions
                        (30, 112),
                        (30, 113),
                        (30, 105.5),
                        (30, 105.6), # Very close edge case
                        (30, 135),
                        (140, 120),
                        (187.5, 70), # Another very close corner case, right on corner
                        
                        #Testing goal collisions
                        (110, 40),
                        (145.5, 40), # Horizontal tangent to goal
                        (110, 62.5), # ball tangent to goal
                        
                        #Test parallel line oblong line segment and wall
                        (50, 100),
                        (200, 100),
                        (205.5, 100) #Out of bounds
                    ]

    #Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]
    alien_horz_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, True, True),
                            (False, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, False),
                            (True, False, False)
                        ]
    alien_vert_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    #Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110,55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))


    print("Geometry tests passed\n")