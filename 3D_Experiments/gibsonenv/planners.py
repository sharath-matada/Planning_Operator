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

from scipy.ndimage import distance_transform_edt


from scipy.io import loadmat
from astar3D.astar import  AStar
from astar3D.environment_simple import Environment3D
from astar3D.utilities import tic, toc

from tqdm import tqdm

# import ompl.base as ob
# import ompl.geometric as og
# import ompl.util as ou


from models.TrainPlanningOperator3D import PlanningOperator3D, smooth_chi

def calculate_signed_distance(velocity_matrix):
    distance = distance_transform_edt(velocity_matrix != 0)
    return distance

def generaterandompos(maps):
    "Generates random positions in the free space of given maps"

    numofmaps = maps.shape[0]
    size_x = maps.shape[1]
    size_y = maps.shape[2]
    size_z = maps.shape[3]

    pos = np.zeros((numofmaps,3))

    for i,map in enumerate(maps):

        condition1 = map == 1
        row_indices, col_indices, z_indices = np.indices(map.shape)
        condition2 = (row_indices < size_x) & (col_indices < size_y) & ((z_indices < size_z))
        # condition3 = ( 1*size_x/4 <row_indices < 3*size_x/4) & (1*size_y/4<col_indices < 3*size_y/4) & ((1*size_z/4<z_indices < 3*size_z/4))
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

        if best_gradient > 10e5:
            if plotfails:
                print("Failed Path:")
                plot_path(value_function, path_points)
            return False, 0, path_points 

        if best_action is not None:
            current_point += learning_rate * best_action
            path_length += np.linalg.norm(learning_rate * best_action)
            path_points.append(current_point.copy())
            visited_points.add(tuple(np.round(current_point).astype(int)))
            if np.array_equal(np.round(current_point).astype(int), np.round(goal_point).astype(int)):
                if plotsuccess:
                    print("Successful Path:")
                    plot_path(value_function, path_points)
                return True, path_length, path_points  # Success
        else:
            if plotfails:
                print("Failed Path:")
                plot_path(value_function, path_points)
            return False, 0, path_points  # No valid action found
    if plotfails:
        print("Failed Path:")
        plot_path(value_function, path_points)
    return False, 0, path_points  

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
        return False, path_cost, dt, path
    
    return True, path_cost, dt, path

def FMMPlanner(start, goal, map):
    env_size_x, env_size_y, env_size_z = map.shape[0],map.shape[1],map.shape[2]
    t1 = tic()
    solver = pykonal.EikonalSolver(coord_sys = "Cartesian")
    velocity_matrix  = map
    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = 1, 1, 1
    solver.velocity.npts = env_size_x, env_size_y, env_size_z
    solver.velocity.values = velocity_matrix.reshape(env_size_x, env_size_y, env_size_z)
    src_idx = goal[0].astype(int), goal[1].astype(int), goal[2].astype(int)
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    solver.solve()
    success, pathlength,path = perform_gradient_descent(solver.traveltime.values,start,goal)
    dt = toc(t1)
    return success, pathlength, dt, path


def getFMMVal(goal,map):
    env_size_x, env_size_y, env_size_z = map.shape[0],map.shape[1],map.shape[2]
    t1 = tic()
    solver = pykonal.EikonalSolver(coord_sys = "Cartesian")
    velocity_matrix  = map
    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = 1, 1, 1
    solver.velocity.npts = env_size_x, env_size_y, env_size_z
    solver.velocity.values = velocity_matrix.reshape(env_size_x, env_size_y, env_size_z)
    src_idx = goal[0].astype(int), goal[1].astype(int), goal[2].astype(int)
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    solver.solve()
    dt = toc(t1)
    valuefunction = solver.traveltime.values
    valuefunction = np.where(map == 0, 0, valuefunction)
    return valuefunction, dt


def getPNOVal(goal, map, model):
    mask = map
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_size_x, env_size_y, env_size_z = map.shape[0],map.shape[1],map.shape[2]

    # Calculate SDF of eroded map
    map = map.reshape(1,env_size_x,env_size_y,env_size_z,1)
    map = torch.tensor(map,dtype=torch.float)

    t0 = tic()
    sdf = calculate_signed_distance(mask)
    dt_sdf = toc(t0)

    sdf = sdf.reshape(1,env_size_x,env_size_y,env_size_z,1)
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
    valuefunction = valuefunction*(mask)
    dt_val = toc(t0) - dt_sdf
    dt = toc(t0)

    return valuefunction, dt
    

def PlanningOperatorPlanner(start, goal, map, model):
    mask = map
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_size_x, env_size_y, env_size_z = map.shape[0],map.shape[1],map.shape[2]

    # Calculate SDF of eroded map
    map = map.reshape(1,env_size_x,env_size_y,env_size_z,1)
    map = torch.tensor(map,dtype=torch.float)

    t0 = tic()
    sdf = calculate_signed_distance(mask)
    dt_sdf = toc(t0)

    sdf = sdf.reshape(1,env_size_x,env_size_y,env_size_z,1)
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
    dt_val = toc(t0) - dt_sdf

    success, pathlength, path = perform_gradient_descent(valuefunction,start,goal)
    dt_gd = toc(t0) - dt_sdf - dt_val

    dt = toc(t0)

    return success, pathlength, dt, path


def DoEikPlanningOperatorPlanner(start, goal, map, sdf_model, val_model):
    mask = 1-map
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_size_x, env_size_y, env_size_z = map.shape[0],map.shape[1],map.shape[2]

    t0 = tic()
    map = map.reshape(1,env_size_x,env_size_y,1)
    map = torch.tensor(map,dtype=torch.float).to(device)

    #Infer Signed Distance Function
    sdf = sdf_model(mask)
    sdf = sdf.detach().cpu().numpy()

    # Calculate Chi for smoothening
    smooth_coef=5. #Depends on what is it trained on
    chi = smooth_chi(map, sdf, smooth_coef).to(device)

    # Load Goal Position
    goal_coord = np.array([goal[0],goal[1],goal[2]])
    gg = goal_coord.reshape(1,3,1)
    gg = torch.tensor(gg, dtype=torch.float).to(device)

    #Infer value function 
    valuefunction = val_model(chi,gg)
    valuefunction = valuefunction.detach().cpu().numpy().reshape(env_size_x,env_size_y, env_size_z)
    valuefunction = valuefunction/(mask+10e-10)

    success, pathlength = perform_gradient_descent(valuefunction,start,goal)
    dt = toc(t0)

    return success, pathlength, dt

def testplanneronmap(starts, goals, maps, planner, plotresults = False, printvalues = False, **kwargs):
    avgpathcost, avgplantime, avgsuccessrate = 0, 0, 0, 0, 0
    totpathcost, totplantime, succcount = 0, 0, 0, 0, 0

    numofmaps = maps.shape[0]

    for start, goal, map in zip(starts, goals, maps):
        success,pathcost,planningtime,_ = planner(start, goal,map,**kwargs)
        if success:
            succcount += 1
            totpathcost += path_cost
            totplantime += dt_plan
        
        

    # Calculate averages
    avgpathcost = totpathcost / numofmaps
    avgplantime = totplantime / numofmaps
    avgsuccessrate = succcount / numofmaps

    if printvalues:
        print(  'Average Path Cost:', avgpathcost, 
                '\nAverage Planning Time:', avgplantime, 
                '\nAverage Success Rate:', avgsuccessrate)

    return avgpathcost, avgplantime, avgsuccessrate




def testplanneronmaps(starts, goals, maps, planner, plotresults=False, printvalues=True, **kwargs):
    avgpathcost, avgplantime, avgsuccessrate = 0, 0, 0
    totpathcost, totplantime, succcount = 0, 0, 0
    successful_map_indices = []

    numofmaps = maps.shape[0]

    for idx, (start, goal, map) in enumerate(tqdm(zip(starts, goals, maps))):
        success, path_cost, dt_plan, _ = planner(start, goal, map, **kwargs)
        if success:
            succcount += 1
            totpathcost += path_cost
            totplantime += dt_plan
            successful_map_indices.append(idx)
        
    # Calculate averages
    avgpathcost = totpathcost / succcount if succcount > 0 else 0
    avgplantime = totplantime / succcount if succcount > 0 else 0
    avgsuccessrate = succcount / numofmaps

    if printvalues:
        print('Average Path Cost:', avgpathcost, 
              '\nAverage Planning Time:', avgplantime, 
              '\nAverage Success Rate:', avgsuccessrate)

    return avgpathcost, avgplantime, avgsuccessrate, successful_map_indices


def plot_2d_map_and_two_paths(path_po, path_fmm, map_3d, z_index):
    path_po = np.array(list(path_po))
    path_fmm = np.array(list(path_fmm))
    map_3d = np.array(map_3d)
    map_cross_section = map_3d[:, :, z_index]
    plt.figure(figsize=(8, 8))
    plt.imshow(map_cross_section.T, cmap='gray', origin='lower')  # Transpose for correct orientation
    plt.scatter(path_po[:, 0], path_po[:, 1], c='red', label='Path PO', s=50, marker='o')
    plt.scatter(path_fmm[:, 0], path_fmm[:, 1], c='blue', label='Path FMM', s=50, marker='x')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'2D Scatter Plot of Paths and Map (Cross-section at z={z_index})')
    plt.legend()
    plt.show()    

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


