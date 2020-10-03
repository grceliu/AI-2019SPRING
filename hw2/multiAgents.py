# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition() #type: tuple
        newFood = successorGameState.getFood()  #type: instance
        newGhostStates = successorGameState.getGhostStates() #returns a list
        newGhostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #stored in a list

        #Moves to the food(score will increase by 10)
        score = 0
        score += successorGameState.getScore()
        #Move towards the nearest food
        food_dist = []
        for i in range(newFood.width):
            for j in range(newFood.height):
                if successorGameState.hasFood(i,j):
                    food_dist.append(math.sqrt((i-newPos[0])**2 + (j-newPos[1])**2))
        if food_dist:
            nearest = min(food_dist)
            score -= nearest*0.1
        #Keep reasonable distance from ghosts
        for ghost in newGhostPos:
            ghost_dist = math.sqrt((ghost[0] - newPos[0])**2 + (ghost[1] - newPos[1])**2)
            if ghost_dist <= 1:  #never go next to the ghost
                score -= 100
            elif ghost_dist <= 2:   #run away
                score -= ghost_dist*0.1

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        ans = self.minimax(gameState,0)
        return ans[0]
    def minimax(self, state, current):
        if state.isWin() or state.isLose() or current == self.depth * state.getNumAgents():#reach the limit
            return (None, self.evaluationFunction(state))
        else:
            agent_id = current % state.getNumAgents()
            actions = state.getLegalActions(agent_id)
            successors_states = [state.generateSuccessor(agent_id, i) for i in actions]
            score = [self.minimax(state, current + 1)[1] for state in successors_states]

            if agent_id != 0: #ghost's choice
                return (actions[score.index(min(score))], min(score))
            else: #pacman's choice
                return (actions[score.index(max(score))], max(score))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        ans = self.minimax_ab(gameState,0, -float("inf"), float("inf"))
        return ans[0]

    def minimax_ab(self, state, current, alpha, beta):
        if state.isWin() or state.isLose() or current == self.depth * state.getNumAgents():#reach the limit
            return (None, self.evaluationFunction(state))

        agent_id = current % state.getNumAgents()

        if agent_id == 0: #ghost
            v = (None, -float("inf"))
            actions = state.getLegalActions(agent_id)
            for action in actions:
                successor = state.generateSuccessor(agent_id, action)
                result = self.minimax_ab(successor, current + 1, alpha, beta)
                if result[1] > v[1]:
                    v = (action, result[1])
                if v[1] > beta:
                    return v
                alpha = max(alpha, v[1])
            return v

        else: #pacman
            v = (None, float("inf"))
            actions = state.getLegalActions(agent_id)
            for action in actions:
                successor = state.generateSuccessor(agent_id, action)
                result = self.minimax_ab(successor, current + 1, alpha, beta)
                if result[1] < v[1]:
                    v = (action, result[1])
                if alpha > v[1]:
                    return v
                beta = min(beta, v[1])
            return v



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction.
          All ghosts choose uniformly at random from their legal moves.
        """
        ans = self.expectimax(gameState,0)
        return ans[0]

    def expectimax(self, state, current):
        """
        Input: the game state to be expanded, the current depth
        Ouput: (Action, expected score)
        """
        if state.isWin() or state.isLose() or current == self.depth * state.getNumAgents():#reach the limit
            return (None, self.evaluationFunction(state))
        else:
            agent_id = current % state.getNumAgents()
            actions = state.getLegalActions(agent_id)
            successors_states = [state.generateSuccessor(agent_id, i) for i in actions]
            score = [self.expectimax(state, current + 1)[1] for state in successors_states]

            if agent_id != 0: #ghost's choice
                expected_score = float(sum(score)) / float(len(score))
                return (actions[random.randrange(len(score))], expected_score)
            else: #pacman's choice
                return (actions[score.index(max(score))], max(score))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
