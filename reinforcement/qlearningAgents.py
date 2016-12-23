# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter() # A Counter is a dict with default 0
        "*** YOUR CODE HERE ***"

    """
    For this function, we just checked if this sate appeared
    before. Or we had to add it into the dictionary first.
    """
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state, action) not in self.qValues:
            self.qValues.update({(state, action):0.0})
        return self.qValues[(state,action)]
        util.raiseNotDefined()

    """
    Just return the value which is the highest Qvalue.
    """
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        qList = []
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        else:
            for legAction in actions:
                qList.append( self.getQValue(state, legAction))
            return max(qList)
        util.raiseNotDefined()

    """
    In this function, we return the action that has the highest
    Qvalue.
    """
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxQValue = -float("inf")
        maxAction = None
        actions = self.getLegalActions(state)
        if not actions:
            return None
        else:
            for legAction in actions:
                actionQValue =  self.getQValue(state, legAction)
                if actionQValue > maxQValue:
                    maxQValue = actionQValue
                    maxAction = legAction
            return maxAction

        util.raiseNotDefined()

    """
    The difference in this function is that we need to
    consider the epsilon, which is actually. So sometimes,
    it acts randomly, and sometimes, follow the policy we
    already had.
    """
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if not legalActions:
            return None
        else:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action
        util.raiseNotDefined()



    """
    For each update, actually we updated the qValue with a
    tuple of state and action. Actually alpha is the update rate.
    Other things just follow the formulation.
    """
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount * self.getValue(nextState)
        updatedQsa = self.alpha * sample + (1-self.alpha) * (self.getQValue(state,action))
        self.qValues[(state,action)] = updatedQsa


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    """
    In this function, we try to get the approximate
    Q value. we using for each feature we got the result
    from both feature function and the weight.
    """
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        qValue = 0
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state,action)
        for feature in features:
            qValue += features[feature] * self.weights[feature]
        return qValue
        util.raiseNotDefined()

    """
    Just wrote this function according to
    wi←wi+α⋅difference⋅fi(s,a)wi←wi+α⋅difference⋅fi(s,a)
    difference=(r+γmaxa′Q(s′,a′))−Q(s,a).
    """
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        difference = (reward + self.discount * self.getValue(nextState))-self.getQValue(state,action)
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]
        "*** YOUR CODE HERE ***"

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            for w in self.weights:
                print w
            pass
