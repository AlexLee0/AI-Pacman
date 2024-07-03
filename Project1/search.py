# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    print("Is the start a goal?", problem.getSuccessors((2, 2)))

    """
    startPath = []
    visitedStates = []
    # DFS needs LIFO data structure to prioritize unvisited path
    fringe = util.Stack()
    startState = problem.getStartState()
    fringe.push((startState, startPath))
    while not fringe.isEmpty():
        current = fringe.pop()
        currentState = current[0]
        currentPath = current[1]
        # check if we reached goal
        if problem.isGoalState(currentState):
            return currentPath
        # only explore unvisited states
        if not visitedStates.__contains__(currentState):
            visitedStates.append(currentState)
            # add successors to fringe
            for successor in problem.getSuccessors(currentState):
                # push successor into fringe with the extended path
                fringe.push((successor[0], currentPath + [successor[1]]))
    return currentPath
    util.raiseNotDefined()


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    startPath = []
    visitedStates = []
    # BFS needs FIFO data structure to prioritize unvisited children
    fringe = util.Queue()
    startState = problem.getStartState()
    fringe.push((startState, startPath))
    while not fringe.isEmpty():
        current = fringe.pop()
        currentState = current[0]
        currentPath = current[1]
        # check if we reached goal
        if problem.isGoalState(currentState):
            return currentPath
        # only explore unvisited states
        if not visitedStates.__contains__(currentState):
            visitedStates.append(currentState)
            # add successors to fringe
            for successor in problem.getSuccessors(currentState):
                # push successor into fringe with the extended path
                fringe.push((successor[0], currentPath + [successor[1]]))
    return currentPath

    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    startPath = []
    visitedStates = []
    # UCS needs PriorityQueue data structure to prioritize cost
    fringe = util.PriorityQueue()
    startState = problem.getStartState()
    fringe.push((startState, startPath, 0), 0)
    while not fringe.isEmpty():
        current = fringe.pop()
        currentState = current[0]
        currentPath = current[1]
        currentCost = current[2]
        # check if we reached goal
        if problem.isGoalState(currentState):
            return currentPath
        # only explore unvisited states
        if not visitedStates.__contains__(currentState):
            visitedStates.append(currentState)
            # add successors to fringe
            for successor in problem.getSuccessors(currentState):
                fringe.push(
                    # push successor into fringe with the extended path and cost
                    (successor[0], currentPath + [successor[1]], currentCost + successor[2]), currentCost + successor[2])
    return currentPath
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    startPath = []
    visitedStates = []
    # A* needs PriorityQueue data structure to prioritize heuristic+cost
    fringe = util.PriorityQueue()
    startState = problem.getStartState()
    fringe.push((startState, startPath, 0), heuristic(startState, problem))
    while not fringe.isEmpty():
        current = fringe.pop()
        currentState = current[0]
        currentPath = current[1]
        currentCost = current[2]
        # check if we reached goal
        if problem.isGoalState(currentState):
            return currentPath
        # only explore unvisited states
        if not visitedStates.__contains__(currentState):
            visitedStates.append(currentState)
            # add successors to fringe
            for successor in problem.getSuccessors(currentState):
                fringe.push(
                    # push successor into fringe with the extended path and cost
                    (successor[0], currentPath + [successor[1]], currentCost + successor[2]), heuristic(successor[0], problem) + currentCost + successor[2])
    return currentPath
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
