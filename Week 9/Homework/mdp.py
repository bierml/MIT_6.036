import pdb
from dist import uniform_dist, delta_dist, mixture_dist
from util import *
import random

class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn, 
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

def value_iteration(mdp, q, eps = 0.01, interactive_fn = None,
                    max_iters = 10000):
    # Your code here

    def func(q,s):
        def fnc(a):
            return q.get(s,a)
        return fnc
    q_ = q
    iter_ = 0
    while(iter_<max_iters):
        new_q = q_.copy()
        for s in q_.states:
            for a in q_.actions:
                #s and a are chosen on this step
                #we have to cycle over all possible s' values
                #init val to zero
                val = 0
                for s_ in mdp.transition_model(s,a).support():
                    #for found value of s' we have to find a' maximizing Qold for
                    #this state
                    #sum of T(s,a,s')*max(a',(Qold(s',a')))
                    #OLDVER
                    #val += mdp.transition_model(s,a).prob(s_)*q_.get(s_,argmax(q_.actions,func(q_,s_)))
                    #NEWVER
                    #print(mdp.transition_model(s,a).prob(s_))
                    val += mdp.transition_model(s,a).prob(s_)*value(q_,s_)
                #multiplying by gamma
                val *= mdp.discount_factor
                #adding R(s,a)
                val += mdp.reward_fn(s,a)
                new_q.set(s,a,val)
        #after computing result for all (s,a) pairs
        #we have to check the exit condition
        #max difference init value is zero
        maxdiff = 0
        for s in q_.states:
            for a in q_.actions:
                if(abs(q_.get(s,a)-new_q.get(s,a))>maxdiff):
                    maxdiff=abs(q_.get(s,a)-new_q.get(s,a))
        q_=new_q
        #print(q_.q)
        if(maxdiff<eps):
            break
        iter_ = iter_ + 1
    return q_
    pass

# Compute the q value of action a in state s with horizon h, using
# expectimax
def q_em(mdp, s, a, h):
    # Your code here
    if(h==0):
        return 0
    else:
        val=0
        for s_ in mdp.transition_model(s,a).support():
            maxm = 0 
            for a_ in mdp.actions:
                if(q_em(mdp,s_,a_,(h-1))>maxm):
                    maxm=q_em(mdp,s_,a_,(h-1))
            val+=mdp.transition_model(s,a).prob(s_)*maxm
        val *= mdp.discount_factor
        val += mdp.reward_fn(s,a)
        return val

# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    """ Return Q*(s,a) based on current Q

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q_star = value(q,0)
    >>> q_star
    10
    """
    # Your code here
    q_star = q.get(s,q.actions[0])
    for a in q.actions:
        if q.get(s,a)>q_star:
            q_star = q.get(s,a)
    return q_star
    pass

# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    """ Return pi*(s) based on a greedy strategy.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> greedy(q, 0)
    'c'
    >>> greedy(q, 1)
    'b'
    """
    # Your code here
    action = q.actions[0]
    for a in q.actions:
        if q.get(s,a)>q.get(s,action):
            action = a
    return action
    pass

def epsilon_greedy(q, s, eps = 0.5):
    """ Return an action.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> eps = 0.
    >>> epsilon_greedy(q, 0, eps) #greedy
    'c'
    >>> epsilon_greedy(q, 1, eps) #greedy
    'b'
    """
    if random.random() < eps:  # True with prob eps, random action
        # Your code here
        helpobj = uniform_dist(q.actions)
        return helpobj.draw()
    else:
        return greedy(q,s)

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]
