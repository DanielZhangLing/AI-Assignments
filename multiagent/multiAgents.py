# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

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

        "Add more of your code here if you want to"

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
        #print "currentGameState:",currentGameState
        #print "action:", action

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print "successorGameState:", successorGameState
        newPos = successorGameState.getPacmanPosition()
        #print "newPos:", newPos
        newFood = successorGameState.getFood()
        #print "newFood:", newFood
        newGhostStates = successorGameState.getGhostStates()

        #print "newGhostStates:", newGhostStates
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print "newScaredTimes:", newScaredTimes
        """
        Here, I try to take both the ghost and the food into consideration
        find the distance to closest ghost and food using manhattan.
        They are called minGhostDistance and foodScore, and with these two
        and original score we got the totalscore
        """
        ghostDistanceScore = []
        for newGhostState in newGhostStates:
            newGhostPosition = newGhostState.getPosition()
            ghostDistanceScore += [manhattanDistance(newPos, newGhostPosition)]
            minGhostDistance = min(ghostDistanceScore)

        foodList = newFood.asList()
        foodDistanceScore = []
        for food in foodList:
            foodDistanceScore += [manhattanDistance(newPos, food)]

        if len(foodDistanceScore) == 0:
            foodScore = 0
        else:
            foodScore = min(foodDistanceScore)

        totalScore = successorGameState.getScore()+ minGhostDistance -foodScore
        "*** YOUR CODE HERE ***"
        return totalScore

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
    """
    Here I wrote the function according to the slices,
    so we have three functions. There're two key points.
    The first one is because there're lots more ghosts.
    So if ghostNumber +1 != agentNumber they will still do
    the minValue, maybe it's kind of minimini. Secondly,
    for the depth, I tried to give a depth at the first call,
    it will count down in the maxValue function every time.
    Until it reaches 0.
    """
    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        pacmanActionList = gameState.getLegalActions(0)
        miniMax = -float("inf")
        adversarialAction = pacmanActionList[0]
        for action in pacmanActionList:
            result = gameState.generateSuccessor(0, action)
            temp = miniMax
            miniMax = max(miniMax, self.minValue(result, 1, self.depth))
            if temp < miniMax:
                adversarialAction = action
        return adversarialAction
        util.raiseNotDefined()

    def maxValue(self,gameState, nowDepth):
        nowDepth -= 1
        if nowDepth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        pacmanActionList = gameState.getLegalActions(0)
        minVal = -float("inf")
        for action in pacmanActionList:
            result = gameState.generateSuccessor(0, action)
            minVal = max(minVal, self.minValue(result, nowDepth, 1))
        return minVal

    def minValue(self, gameState, ghostNumber, nowDepth):
        if nowDepth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        ghostActionList = gameState.getLegalActions(ghostNumber)
        maxVal = float("inf")
        for action in ghostActionList:
            agentNumber = gameState.getNumAgents()
            if ghostNumber +1 != agentNumber:
                result = gameState.generateSuccessor(ghostNumber, action)
                maxVal = min(maxVal, self.minValue(result, ghostNumber + 1, nowDepth))
            else:
                result = gameState.generateSuccessor(ghostNumber, action)
                maxVal = min(maxVal, self.maxValue(result, nowDepth))
        return maxVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    """
    There's not a lot to talk about this,
    but since we use the value as the return of each
    sub function. So actually we do one more loop in
    the getAction function, so we need to do one more
    pruning
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacmanActionList = gameState.getLegalActions(0)
        abMax = -float("inf")
        adversarialAction = pacmanActionList[0]
        iniAlpha = -float("inf")
        iniBeta = float("inf")
        for action in pacmanActionList:
            result = gameState.generateSuccessor(0, action)
            nowDepth = self.depth
            temp = abMax
            abMax = max(abMax, self.minValue(result, 1, nowDepth, iniAlpha, iniBeta))
            if temp < abMax:
                adversarialAction = action
            if abMax>= iniBeta:
                return action
            iniAlpha = max(iniAlpha, abMax)
        return adversarialAction
        util.raiseNotDefined()

    def maxValue(self, gameState, nowDepth, alpha, beta,):
        nowDepth = nowDepth - 1
        if nowDepth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        pacmanActionList = gameState.getLegalActions(0)
        miniVal = -float("inf")
        for action in pacmanActionList:
            result = gameState.generateSuccessor(0, action)
            miniVal = max(miniVal, self.minValue(result, 1, nowDepth, alpha, beta))
            if miniVal > beta:
                return miniVal
            alpha = max(alpha, miniVal)
        return miniVal

    def minValue(self, gameState, ghostNumber, nowDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or nowDepth == 0:
            return self.evaluationFunction(gameState)
        ghostActionList = gameState.getLegalActions(ghostNumber)
        maxVal = float("inf")
        for action in ghostActionList:
            agentNumber = gameState.getNumAgents()
            if ghostNumber + 1 != agentNumber:
                result = gameState.generateSuccessor(ghostNumber, action)
                maxVal = min(maxVal, self.minValue(result, ghostNumber + 1, nowDepth, alpha, beta))
            else:
                result = gameState.generateSuccessor(ghostNumber, action)
                maxVal = min(maxVal, self.maxValue(result, nowDepth, alpha, beta))
            if maxVal < alpha:
                return maxVal
            beta = min(beta, maxVal)
        return maxVal

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    """
    almost the same as miniMax, the only different
    is we changed minValue to expValue
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacmanActionList = gameState.getLegalActions(0)
        expectMax = -float("inf")
        adversarialAction = pacmanActionList[0]
        for action in pacmanActionList:
            result = gameState.generateSuccessor(0, action)
            nowDepth = self.depth
            temp = expectMax
            expectMax = max(expectMax, self.expValue(result, 1, nowDepth))
            if temp < expectMax:
                adversarialAction = action
        return adversarialAction
        util.raiseNotDefined()

    def maxValue(self,gameState, nowDepth):
        nowDepth -= 1
        if nowDepth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        pacmanActionList = gameState.getLegalActions(0)
        minVal = -float("inf")
        for action in pacmanActionList:
            result = gameState.generateSuccessor(0, action)
            minVal = max(minVal, self.expValue(result, nowDepth, 1))
        return minVal

    def expValue(self,gameState, ghostNumber, nowDepth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        ghostActionList = gameState.getLegalActions(ghostNumber)
        expVal = 0
        for action in ghostActionList:
            agentNumber = gameState.getNumAgents()
            p = 1 / float(len(ghostActionList))
            if ghostNumber +1 != agentNumber:
                result = gameState.generateSuccessor(ghostNumber, action)
                expVal += p * self.expValue(result, ghostNumber + 1, nowDepth)
            else:
                result = gameState.generateSuccessor(ghostNumber, action)
                expVal += p * self.maxValue(result, nowDepth)
        return expVal

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    """
    This question is a little different with the first one,
    we need to evaluate the current state score. So as you can see
    I changed the names to cur*. In order to get full marks.
    I added two element. First when the ghost is very close to you,
    we just gave a -inf. What's more we want to get the ghost afraid,
    so the pacman is encouraged to eat the power bean.
    """
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]

    ghostDistanceScore = []
    for curGhostState in curGhostStates:
        curGhostPosition = curGhostState.getPosition()
        ghostDistanceScore += [manhattanDistance(curPos, curGhostPosition)]
    if min(ghostDistanceScore)==0:
        return -float("inf")
    minScaredTime = min(curScaredTimes)

    foodList = curFood.asList()
    foodDistanceScore = []
    for food in foodList:
        foodDistanceScore += [manhattanDistance(curPos, food)]
    if len(foodDistanceScore) == 0:
        foodScore = 0
    else:
        foodScore = min(foodDistanceScore)

    totalScore = currentGameState.getScore() + min(ghostDistanceScore) - foodScore +minScaredTime
    "*** YOUR CODE HERE ***"
    return totalScore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

