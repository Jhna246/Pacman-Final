# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
from collections import defaultdict
import pickle

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

        "*** YOUR CODE HERE ***"
        self.q_learn_val = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # just return Q(state,action)
        # print(self.q_learn_val[(state, action)], 'q_learn value')
        return self.q_learn_val[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # so I return max_action over available legal moves. First I have to find the legal moves
        leg_moves = self.getLegalActions(state)     # legal moves available

        # initially set val to a very low number
        val = float("-inf")

        # if there are no legal actions, return a value of 0.0
        if len(leg_moves) == 0:
            return 0.0

        # it says to return max_action Q(state,action) where the max is over legal action
        # so just iterate through legal moves and find max value of q_learn_val
        for i in leg_moves:
            val = max(val, self.getQValue(state, i))
        return val


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        leg_moves = self.getLegalActions(state)
        best_action = None  # set initial best action as none because we don't have a best_action yet

        # Note that if there are no legal actions you should return None.
        if not leg_moves:
            return None

        # so what I did was set a value to a very small number
        # and if that is less than the qvalue, it will be the best action
        val = float("-inf")
        for i in leg_moves:
            if val < self.getQValue(state, i):
                val = self.getQValue(state, i)
                best_action = i
                # print(best_action)
        return best_action

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
        "*** YOUR CODE HERE ***"

        # use util.flipCoin(prob) with probablity self.epsilon and random.choice(list)

        if legalActions:
            # With probability self.epsilon, we should take a random action
            if util.flipCoin(self.epsilon):
                # print(random.choice(legalActions), 'random choice')
                return random.choice(legalActions)
            # Take the best policy action otherwise
            else:
                # print(self.getPolicy(state), 'get policy')
                return self.getPolicy(state)


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # from your slide
        # Q(s,a) <- (1-alpha)Q(s,a) + alpha * sample
        sample = reward + self.discount * self.getValue(nextState)
        self.q_learn_val[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

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
        self.cum_weights = defaultdict(list)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # use getFeatures function from featureExtractors.py
        # I'm assuming w is weights

        # Should return Q(state,action) = w * featureVector
        # so just return w * feature
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # difference is the same as the sample but you subtract it with Q(s,a) according to the equation
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)

        # feature is the one you extract from featureExtractors.py
        features = self.featExtractor.getFeatures(state, action)

        # from slide w <- w + alpha[difference]f(s,a)
        for i in features:
            self.weights[i] = self.weights[i] + self.alpha * difference * features[i]


        "***  DO NOT DELETE BELOW ***"
        self.write()

    def write(self):
        """
          DO NOT DELETE
        """
        for i in ["bias", "#-of-ghosts-1-step-away", "eats-food", "closest-food"]:
            self.cum_weights[i].append(self.weights[i])

    def save(self):
        """
          DO NOT DELETE
        """
        with open('./cmu_weights.pkl','wb') as f:
            pickle.dump(self.cum_weights,f)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # print the weight you obtained
            print(self.weights)


            "***  DO NOT DELETE BELOW ***"
            self.save()