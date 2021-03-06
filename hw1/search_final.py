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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    
    stack = util.Stack()
    explored = []
    parent = {}

    rootNode = (problem.getStartState(),'Stop',0)
    stack.push(rootNode)

    while not stack.isEmpty():
        node = stack.pop()
        explored.append(node[0])
        if problem.isGoalState(node[0]):
            break
        successors = problem.getSuccessors(node[0])
        for i in successors:
            if i[0] not in explored:
                stack.push(i)
                parent[i] = node
    path = [node[1]]
    while node[0] != problem.getStartState():
        path.append(parent[node][1])
        node = parent[node]
    path.pop()
    path.reverse()

    return path
    """
    stack = util.Stack()
    explored, actions = [], {}

    rootNode = problem.getStartState()
    actions[rootNode] = ()
    stack.push(rootNode)

    while not stack.isEmpty():
        node = stack.pop()
        explored.append(node)
        if problem.isGoalState(node):
            return list(actions[node])
        successors = problem.getSuccessors(node)
        for i in successors:
            if i[0] not in explored:
                stack.push(i[0])
                actions[i[0]] = actions[node] + (i[1],)
    """



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    explored = []
    parent = {}

    rootNode = (problem.getStartState(),'Stop',0)
    queue.push(rootNode)

    while not queue.isEmpty():
        node = queue.pop()
        explored.append(node[0])
        if problem.isGoalState(node[0]):
            break
        successors = problem.getSuccessors(node[0])
        for i in successors:
            if i[0] not in explored and (queue.isEmpty() or i[0] not in [j[0] for j in queue.list]):
                queue.push(i)
                parent[i] = node

    path = [node[1]]
    while node[0] != problem.getStartState():
        path.append(parent[node][1])
        node = parent[node]
    path.pop()
    path.reverse()

    return path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    heap = util.PriorityQueue()
    cur_cost, actions = {}, {}
    rootNode = problem.getStartState()
    cur_cost[rootNode] = 0
    actions[rootNode] = ()
    heap.push(rootNode,0)

    while not heap.isEmpty():
        node = heap.pop()
        if problem.isGoalState(node):
            return list(actions[node])

        successors = problem.getSuccessors(node)

        for i in successors:
            if not (i[0] in cur_cost and i[2] + cur_cost[node] > cur_cost[i[0]]):
                cur_cost[i[0]] = i[2] + cur_cost[node]
                heap.update(i[0],cur_cost[i[0]])
                actions[i[0]] = actions[node] + (i[1],)  #storing in tuple saves memory





def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    heap = util.PriorityQueue()
    cur_cost, actions = {}, {}
    rootNode = problem.getStartState()
    cur_cost[rootNode] = 0
    actions[rootNode] = ()
    heap.push(rootNode,0)

    while not heap.isEmpty():
        node = heap.pop()
        if problem.isGoalState(node):
            return list(actions[node])

        successors = problem.getSuccessors(node)

        for i in successors:
            if not (i[0] in cur_cost and i[2] + cur_cost[node] > cur_cost[i[0]]):
                cur_cost[i[0]] = i[2] + cur_cost[node]
                heap.update(i[0],cur_cost[i[0]]+heuristic(i[0],problem))
                actions[i[0]] = actions[node] + (i[1],)  #storing in tuple saves memory



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
