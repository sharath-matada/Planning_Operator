import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

import random

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from functools import reduce
from functools import partial

cmap = plt.cm.viridis  # Choose the colormap you want to invert
cmap_inverted = LinearSegmentedColormap.from_list("inverted_viridis", cmap(np.linspace(1, 0, 256)))

from timeit import default_timer
import scipy.io
import os
import sys
from itertools import chain
import time
import pykonal


from scipy.io import loadmat
from astar3D.astar import  AStar
from astar3D.environment_simple import Environment3D
from astar3D.utilities import tic, toc

# import ompl.base as ob
# import ompl.geometric as og
# import ompl.util as ou


from models.TrainPlanningOperator3D import PlanningOperator3D, smooth_chi

def generaterandompos(maps):
    "Generates random positions in the free space of given maps"

    numofmaps = maps.shape[0]
    env_size = maps.shape[1]
    pos = np.zeros((numofmaps,3))

    assert maps.shape[1] == maps.shape[2] == maps.shape[3]

    for i,map in enumerate(maps):

        condition1 = map == 1
        row_indices, col_indices, z_indices = np.indices(map.shape)
        condition2 = (row_indices < env_size) & (col_indices < env_size) & ((z_indices < env_size))
        combined_condition = condition1 & condition2
        passable_indices = np.argwhere(combined_condition)
        point = random.choice(passable_indices)
        pos[i,:] = np.array([point[0],point[1],point[2]])

    return pos.astype(int)



# Primary movements in 3D: forward, backward, left, right, up, down
primary_moves = [
    [-1., 0., 0.],  # left
    [1., 0., 0.],   # right
    [0., 1., 0.],   # forward
    [0., -1., 0.],  # backward
    [0., 0., 1.],   # up
    [0., 0., -1.]   # down
]

# Diagonal movements in 3D
diagonal_moves = [
    [-1., 1., 0.],  # left-forward
    [-1., -1., 0.], # left-backward
    [1., 1., 0.],   # right-forward
    [1., -1., 0.],  # right-backward
    [-1., 0., 1.],  # left-up
    [-1., 0., -1.], # left-down
    [1., 0., 1.],   # right-up
    [1., 0., -1.],  # right-down
    [0., 1., 1.],   # forward-up
    [0., 1., -1.],  # forward-down
    [0., -1., 1.],  # backward-up
    [0., -1., -1.], # backward-down
    [-1., 1., 1.],  # left-forward-up
    [-1., 1., -1.], # left-forward-down
    [-1., -1., 1.], # left-backward-up
    [-1., -1., -1.],# left-backward-down
    [1., 1., 1.],   # right-forward-up
    [1., 1., -1.],  # right-forward-down
    [1., -1., 1.],  # right-backward-up
    [1., -1., -1.]  # right-backward-down
]

# Combine primary and diagonal moves
action_vecs = np.asarray(primary_moves + diagonal_moves)

def perform_gradient_descent(value_function, start_point, goal_point, plotsuccess=False, plotfails=False, learning_rate=1, num_steps=1000):
    path_length = 0
    path_points = [start_point.copy().astype(float)]
    visited_points = set()
    current_point = start_point.copy().astype(float)

    for step in range(num_steps):
        best_gradient = np.inf
        best_action = None

        for action in action_vecs:
            new_point = current_point + learning_rate * action
            new_point_indices = np.round(new_point).astype(int)
            x_index, y_index, z_index = new_point_indices

            if (0 <= x_index < value_function.shape[0] and 
                0 <= y_index < value_function.shape[1] and 
                0 <= z_index < value_function.shape[2] and 
                (x_index, y_index, z_index) not in visited_points):
                gradient = value_function[x_index, y_index, z_index]
                if gradient < best_gradient:
                    best_gradient = gradient
                    best_action = action

        if best_gradient > 100:
            if plotfails:
                print("Failed Path:")
                plot_path(value_function, path_points)
            return False, 0  

        if best_action is not None:
            current_point += learning_rate * best_action
            path_length += np.linalg.norm(learning_rate * best_action)
            path_points.append(current_point.copy())
            visited_points.add(tuple(np.round(current_point).astype(int)))
            if np.array_equal(np.round(current_point).astype(int), np.round(goal_point).astype(int)):
                if plotsuccess:
                    print("Successful Path:")
                    plot_path(value_function, path_points)
                return True, path_length  # Success
        else:
            if plotfails:
                print("Failed Path:")
                plot_path(value_function, path_points)
            return False, 0  # No valid action found
    if plotfails:
        print("Failed Path:")
        plot_path(value_function, path_points)
    return False, 0  

def plot_path(value_function, path_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path_points = np.array(path_points)

    # Identify and plot points where value function is infinite
    inf_indices = np.where(np.isinf(value_function))
    ax.scatter(inf_indices[0], inf_indices[1], inf_indices[2], c='k', marker='o', label='Obstacles')

    # Plot the path
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'ro-', label='Path')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def AStarPlanner(start, goal, map):
    map = 1 - map
    env = Environment3D(goal,map)
    t1 = tic()
    path_cost, path, action_idx = AStar.plan(start, env)
    dt = toc(t1)
    if path_cost > 10e10:
        return False, path_cost, dt
    
    return True, path_cost, dt

def FMMPlanner(start, goal, map):
    envsize = map.shape[0]
    assert map.shape[0] == map.shape[1] == map.shape[2]
    Sx = Sy = Sz = envsize
    t1 = tic()
    solver = pykonal.EikonalSolver(coord_sys = "Cartesian")
    velocity_matrix  = map
    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = 1, 1, 1
    solver.velocity.npts = Sx, Sy, Sz
    solver.velocity.values = velocity_matrix.reshape(Sx, Sy, Sz)
    src_idx = goal[0].astype(int), goal[1].astype(int), goal[2].astype(int)
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    solver.solve()
    success, pathlength = perform_gradient_descent(solver.traveltime.values,start,goal)
    dt = toc(t1)
    return success, pathlength, dt
    
# def RRTPlanner(start, goal, map):
#     environment = map
#     space = ob.RealVectorStateSpace(3)
#     bounds = ob.RealVectorBounds(3)
#     bounds.setLow(0)
#     bounds.setHigh(0, environment.shape[0] - 1)
#     bounds.setHigh(1, environment.shape[1] - 1)
#     bounds.setHigh(2, environment.shape[2] - 1)
#     space.setBounds(bounds)

#     startx = ob.State(space)
#     startx[0], startx[1], startx[2] = start[0], start[1], start[2] 

#     goalx = ob.State(space)
#     goalx[0], goalx[1], goalx[2] = goal[0], goal[1], goal[2] 

#     def isStateValid(state):
#         x, y, z = int(state[0]), int(state[1]), int(state[2])
#         if 0 <= x < environment.shape[0] and 0 <= y < environment.shape[1] and 0 <= z < environment.shape[2]:
#             return environment[x, y, z] == 1
#         return False

#     si = ob.SpaceInformation(space)
#     si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

#     pdef = ob.ProblemDefinition(si)
#     pdef.setStartAndGoalStates(start, goal)

#     planner = og.RRT(si)
#     planner.setProblemDefinition(pdef)

#     # Set the maximum edge length (range)
#     max_edge_length = 1.0  
#     planner.setRange(max_edge_length)
#     planner.setup()
#     solve_time = 0.030
#     solved = planner.solve(solve_time)

#     if solved:
#         success_count += 1
#         path = pdef.getSolutionPath()
#         path_cost = path.cost(ob.PathLengthOptimizationObjective(si))
#         return True, path_cost, solve_time
    
#     return False, np.float('inf'), np.float('inf')


# def RRTStarPlanner(start, goal, map):
#     environment = map
#     space = ob.RealVectorStateSpace(3)
#     bounds = ob.RealVectorBounds(3)
#     bounds.setLow(0)
#     bounds.setHigh(0, environment.shape[0] - 1)
#     bounds.setHigh(1, environment.shape[1] - 1)
#     bounds.setHigh(2, environment.shape[2] - 1)
#     space.setBounds(bounds)

#     startx = ob.State(space)
#     startx[0], startx[1], startx[2] = start[0], start[1], start[2] 

#     goalx = ob.State(space)
#     goalx[0], goalx[1], goalx[2] = goal[0], goal[1], goal[2] 

#     def isStateValid(state):
#         x, y, z = int(state[0]), int(state[1]), int(state[2])
#         if 0 <= x < environment.shape[0] and 0 <= y < environment.shape[1] and 0 <= z < environment.shape[2]:
#             return environment[x, y, z] == 1
#         return False

#     si = ob.SpaceInformation(space)
#     si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

#     pdef = ob.ProblemDefinition(si)
#     pdef.setStartAndGoalStates(start, goal)

#     planner = og.RRTstar(si)
#     planner.setProblemDefinition(pdef)

#     # Set the maximum edge length (range)
#     max_edge_length = 1.0  
#     planner.setRange(max_edge_length)
#     planner.setup()
#     solve_time = 0.030
#     solved = planner.solve(solve_time)

#     if solved:
#         success_count += 1
#         path = pdef.getSolutionPath()
#         path_cost = path.cost(ob.PathLengthOptimizationObjective(si))
#         return True, path_cost, solve_time
    
#     return False, np.float('inf'), np.float('inf')


def PlanningOperatorPlanner(start, goal, map, sdf, model):
    mask = 1-map
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_size_x, env_size_y, env_size_z = map.shape

    t0 = tic()
    # Calculate SDF of eroded map
    map = map.reshape(1,env_size_x,env_size_y,1)
    map = torch.tensor(map,dtype=torch.float)

    sdf = sdf.reshape(1,env_size_x,env_size_y,1)
    sdf = torch.tensor(sdf,dtype=torch.float)

    # Calculate Chi for smoothening
    smooth_coef=5. #Depends on what is it trained on
    chi = smooth_chi(map, sdf, smooth_coef).to(device)

    # Load Goal Position
    goal_coord = np.array([goal[0],goal[1],goal[2]])
    gg = goal_coord.reshape(1,3,1)
    gg = torch.tensor(gg, dtype=torch.float).to(device)

    #Infer value function 
    valuefunction = model(chi,gg)
    valuefunction = valuefunction.detach().cpu().numpy().reshape(env_size_x,env_size_y, env_size_z)
    valuefunction = valuefunction/(mask+10e-10)

    success, pathlength = perform_gradient_descent(valuefunction,start,goal)
    dt = toc(t0)

    return success, pathlength, dt



def testplanneronmaps(starts, goals, maps, planner, plotresults = False, printvalues = True, **kwargs):
    avgpathcost, avgplantime, avginfertime, avgnodesexp, avgsuccessrate = 0, 0, 0, 0, 0
    totpathcost, totplantime, totinfertime, totnodesexp, succcount = 0, 0, 0, 0, 0

    numofmaps = maps.shape[0]

    # for start, goal, map in zip(starts, goals, 1-maps):
    #     do nothing

    # Calculate averages
    avgpathcost = totpathcost / numofmaps
    avgplantime = totplantime / numofmaps
    avginfertime = totinfertime / numofmaps
    avgnodesexp = totnodesexp / numofmaps
    avgsuccessrate = succcount / numofmaps

    if printvalues:
        print(  'Average Path Cost:', avgpathcost, 
                '\nAverage Planning Time:', avgplantime,
                '\nAverage Inference Time:', avginfertime, 
                '\nAverage Number of Node Expansions:', avgnodesexp,
                '\nAverage Success Rate:', avgsuccessrate)

    return avgpathcost, avgplantime, avginfertime, avgnodesexp, avgsuccessrate
