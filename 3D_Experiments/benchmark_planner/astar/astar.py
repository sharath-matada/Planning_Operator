from pqdict import pqdict
from collections import deque
import math

class AState(object):
  __slots__ = ['key', 'coord', 'g', 'h', 'parent', 'parent_action_id',
               'iteration_opened', 'iteration_closed', 'v', 'inconsistent']
  def __init__(self, key, coord, hval):
    self.key = key
    self.coord = coord
    self.g = math.inf
    self.h = hval
    self.parent = None
    self.parent_action_id = -1
    self.iteration_opened = 0
    self.iteration_closed = 0
    self.v = math.inf
    self.inconsistent = False
  def __lt__(self, other):
    return self.g < other.g  


def AStateSpace(eps):
  a_state_space = {}
  a_state_space['il'] = []       # inconsistent list
  a_state_space['pq'] = pqdict() # priority queue
  a_state_space['hm'] = {}       # hashmap
  a_state_space['eps'] = eps
  a_state_space['eps_decrease'] = 0.2
  a_state_space['eps_final'] = 1.0
  a_state_space['eps_satisfied'] = math.inf
  a_state_space['expands'] = 0       # number of expands in the current search
  a_state_space['searchexpands'] = 0 # total number of expands over all searches; Not used for aStar
  a_state_space['use_il'] = False
  a_state_space['reopen_nodes'] = False
  return a_state_space


class AStar(object):
  @staticmethod
  def plan(start_coord, env, eps = 1):
    sss = AStateSpace(eps) # Initialize state space
    curr = AState(tuple(start_coord), start_coord, env.getHeuristic(start_coord)) # env.coord_to_idx(start_coord), 
    curr.g = 0
    curr.iteration_opened = sss['expands']
    
    while True:
      # check if done
      if env.isGoal(curr.coord):
        return AStar.__recoverPath(curr, env)
      
      # count the number of expands
      sss['expands'] += 1
      
      # add curr to the Closed list
      curr.v = curr.g
      curr.iteration_closed = sss['expands']
      
      # update heap
      AStar.__spin( curr, sss, env )
      
      if not sss['pq']:
        return math.inf, deque(), deque()
    
      # remove the element with smallest cost
      curr = sss['pq'].popitem()[1][1]


  ## PRIVATE METHODS ##
  @staticmethod
  def __spin(curr, sss, env):
    # get successors
    succ, succ_cost, succ_act_idx = env.getSuccessors(curr.coord) # succ_idx, 
    num_succ = len(succ_cost)
    # process successors
    for s in range(num_succ):
      #s_idx = succ_idx[s]
      s_coord, s_key = succ[:,s], tuple(succ[:,s])
      # create a new node if necessary
      if s_key not in sss['hm']:
        sss['hm'][s_key] = AState(s_key, s_coord, env.getHeuristic(s_coord))
      child = sss['hm'][s_key]
      
      # see if we can improve the value of succstate
      # taking into account the cost of action
      tentative_gval = curr.v + succ_cost[s]
      
      if( tentative_gval < child.g ):
        child.parent = curr
        child.parent_action_id = succ_act_idx[s]
        child.g = tentative_gval
        fval = tentative_gval + sss['eps']*child.h
        
        # if currently in OPEN, update heap element
        if child.iteration_opened > child.iteration_closed:
          sss['pq'][s_key] = (fval, child)
          sss['pq'].heapify(s_key)

        # if currently in CLOSED
        elif( child.iteration_closed > sss['searchexpands'] ):
          if sss['reopen_nodes']: # reopen node
            sss['pq'][s_key] = (fval, child)
            child.iteration_closed = 0
          elif sss['use_il'] and not child.inconsistent: # inconsistent node
            sss['il'].append(child)
            child.inconsistent = True

        # new node, add to heap
        else:
          sss['pq'][s_key] = (fval, child)
          child.iteration_opened = sss['searchexpands'] + sss['expands']

  @staticmethod
  def __recoverPath(curr, env):
    path_cost = curr.g
    path = deque()
    action_idx = deque()
    while curr.parent is not None:
      path.appendleft(curr.coord)
      action_idx.appendleft(curr.parent_action_id)
      curr = curr.parent
      # deque extendleft is weird and reverses the order
      #path.extendleft( env.forward_action( curr.coord, action_idx[0] ).reverse() )
    path.appendleft(curr.coord)
    return path_cost, path, action_idx

















