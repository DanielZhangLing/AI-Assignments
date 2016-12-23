# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        """
        Here for each state we store the Value using the counter
        And because the exercise mentioned that Use the "batch" version of value iteration
        where each vector Vk is computed from a fixed vector Vk-1 . So we use a temp
        dictionary to store the data for each iteration. Finally, give the data to the self.value
        when this iteration ends.
        """
        # Write value iteration code here
        while 0<iterations:
            tempValues = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    tempValues[state] = 0
                    continue
                actions = self.mdp.getPossibleActions(state)
                maxQValue = -float("inf")
                for action in actions:
                    listOfSap = self.mdp.getTransitionStatesAndProbs(state, action)
                    qValue = 0
                    for sap in listOfSap:
                        value = self.getValue(sap[0])
                        reward = self.mdp.getReward(state, action, sap[0])
                        qValue += (value * self.discount + reward) * sap[1]
                    if maxQValue < qValue:
                        maxQValue = qValue
                    tempValues[state] = maxQValue
            iterations -= 1
            self.values = tempValues




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    """
    We use the formulate to calculate the Q-value using transition states and probes
    """
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qValue = 0
        listOfSap = self.mdp.getTransitionStatesAndProbs(state,action)
        for sap in listOfSap:
            value = self.getValue( sap[0])
            reward = self.mdp.getReward( state, action, sap[0])
            qValue += (value * self.discount +reward ) *sap[1]
        return qValue
        util.raiseNotDefined()

    """
    Basically, we just choose the action which provided the highest QValue
    """
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        maxAction = None
        maxQValue = -float("inf")
        for action in actions:
            listOfSap = self.mdp.getTransitionStatesAndProbs(state, action)
            qValue = 0
            for sap in listOfSap:
                value = self.getValue(sap[0])
                reward = self.mdp.getReward(state, action, sap[0])
                qValue += (value * self.discount + reward) * sap[1]
            if maxQValue< qValue:
                maxQValue = qValue
                maxAction = action
        return maxAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
