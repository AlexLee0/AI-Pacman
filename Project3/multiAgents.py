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
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostDistances = []
        minGhostDistance = 0
        foodDistances = []
        minFoodDistance = 0
        # get manhattan distance of PacMan and ghosts
        for ghost in newGhostStates:
            ghostDistances.append(manhattanDistance(
                newPos, ghost.getPosition()))
        # get manhattan distance of PacMan and food
        for foodPos in newFood.asList():
            foodDistances.append(manhattanDistance(newPos, foodPos))
        if len(ghostDistances) != 0:
            minGhostDistance = min(ghostDistances)
        if len(foodDistances) != 0:
            minFoodDistance = min(foodDistances)
        # if the ghost is not close to PacMan, just eat the food without worrying!
        if minGhostDistance > 3:
            if successorGameState.getNumFood() < currentGameState.getNumFood():
                return float('inf')
            else:
                return -minFoodDistance
        # closer to food the better, further from ghost the better
        return minGhostDistance - 2*minFoodDistance


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxMax(gameState, 1)
        util.raiseNotDefined()

    def minimaxMax(self, gameState, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        maxValue = -float("inf")
        maxAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # initiate minimizer node to begin minimax on each action state
            minimaxMinValue = self.minimaxMin(successor, depth, 1)
            if minimaxMinValue > maxValue:
                maxAction = action
                maxValue = minimaxMinValue
        # root maximizer node
        if depth == 1:
            return maxAction
        return maxValue

    def minimaxMin(self, gameState, depth, ghost):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        minValue = float("inf")
        for action in gameState.getLegalActions(ghost):
            successor = gameState.generateSuccessor(ghost, action)
            # last ghost in depth and must decide to transition to next depth or evaluate
            if ghost == gameState.getNumAgents() - 1:
                # last depth, update for terminal node
                if depth == self.depth:
                    terminalValue = self.evaluationFunction(successor)
                    minValue = min(minValue, terminalValue)
                else:
                    # initiate maximizer node for next depth
                    minimaxMaxValue = self.minimaxMax(successor, depth+1)
                    minValue = min(minValue, minimaxMaxValue)
            else:
                # not last ghost in depth, must initiate minimizer node for next ghost
                nextGhostMinimaxMinValue = self.minimaxMin(
                    successor, depth, ghost+1)
                minValue = min(minValue, nextGhostMinimaxMinValue)
        return minValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alphaValue = -float("inf")
        betaValue = float("inf")
        return self.alphaBetaMax(gameState, 1, alphaValue, betaValue)
        util.raiseNotDefined()

    def alphaBetaMax(self, gameState, depth, alphaValue, betaValue):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        maxValue = -float("inf")
        maxAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # initiate minimizer node to begin minimax on each action state
            alphaBetaMinValue = self.alphaBetaMin(
                successor, depth, 1, alphaValue, betaValue)
            if alphaBetaMinValue > maxValue:
                maxValue = alphaBetaMinValue
                maxAction = action
            if maxValue > betaValue:
                return maxValue
            alphaValue = max(alphaValue, maxValue)
        # root maximizer node
        if depth == 1:
            return maxAction
        return maxValue

    def alphaBetaMin(self, gameState, depth, agent, alphaValue, betaValue):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        minValue = float("inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            # last ghost in depth and must decide to transition to next depth or evaluate
            if agent == gameState.getNumAgents() - 1:
                # last depth, update for terminal node
                if depth == self.depth:
                    terminalValue = self.evaluationFunction(successor)
                    if terminalValue < minValue:
                        minValue = terminalValue
                else:
                    # initiate maximizer node for next depth
                    alphaBetaMaxValue = self.alphaBetaMax(
                        successor, depth+1, alphaValue, betaValue)
                    if alphaBetaMaxValue < minValue:
                        minValue = alphaBetaMaxValue
            else:
                # not last ghost in depth, must initiate minizer node for next ghost
                nextGhostAlphaBetamaxMinValue = self.alphaBetaMin(
                    successor, depth, agent+1, alphaValue, betaValue)
                if nextGhostAlphaBetamaxMinValue < minValue:
                    minValue = nextGhostAlphaBetamaxMinValue
            if minValue < alphaValue:
                return minValue
            betaValue = min(betaValue, minValue)
        return minValue


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimaxMax(gameState, 1)
        util.raiseNotDefined()

    def expectimaxMax(self, gameState, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        maxValue = -float("inf")
        maxAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # initiate minimizer node to begin minimax on each action state
            expectimaxMinValue = self.expectimax(
                successor, depth, 1)
            if expectimaxMinValue > maxValue:
                maxValue = expectimaxMinValue
                maxAction = action
        # root maximizer node
        if depth == 1:
            return maxAction
        return maxValue

    def expectimax(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        expectimaxValues = []
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            # last ghost in depth and must decide to transition to next depth or evaluate
            if agent == gameState.getNumAgents() - 1:
                # last depth, add terminal values
                if depth == self.depth:
                    terminalValue = self.evaluationFunction(successor)
                    expectimaxValues.append(terminalValue)
                else:
                    # initiate maximizer node for next depth
                    expectimaxMaxValue = self.expectimaxMax(
                        successor, depth+1)
                    expectimaxValues.append(expectimaxMaxValue)
            else:
                # not last ghost in depth, must initiate expectimax node for next ghost
                nextGhostExpectimaxValue = self.expectimax(
                    successor, depth, agent+1)
                expectimaxValues.append(nextGhostExpectimaxValue)
        # return average of expectimax values
        return sum(expectimaxValues)/len(expectimaxValues)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Now that we are evaluating game states and not actions, we can utilize the game state's score to our advantage. 
    We can trust that the score increases when PacMan eat a scared ghost and the score decreases significantly when PacMan is 
    eaten by an alive ghost. By incorporating the game state score into our evaluation function, we do not need to worry 
    about getting eaten by a ghost or eating a scared ghost because the score deals with that for us! Then, I subtracted 
    the minimum food distance and capsule distance from the score because PacMan is better off eating all the food with a 
    high score when it is closer to a food or capsule. I adjusted the weights of these three components by trial and error.
    """
    "*** YOUR CODE HERE ***"
    pacPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood()
    capsulePositions = currentGameState.getCapsules()

    foodDistances = []
    minFoodDistance = 0
    capsuleDistances = []
    minCapsuleDistance = 0
    for foodPos in foodPositions.asList():
        foodDistances.append(manhattanDistance(pacPosition, foodPos))
    for capsulePos in capsulePositions:
        capsuleDistances.append(manhattanDistance(pacPosition, capsulePos))
    if len(foodDistances) != 0:
        minFoodDistance = min(foodDistances)
    if len(capsuleDistances) != 0:
        minCapsuleDistance = min(capsuleDistances)
    return currentGameState.getScore() - 2*minCapsuleDistance - minFoodDistance
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
