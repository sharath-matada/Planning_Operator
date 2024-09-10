import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

import matplotlib.pyplot as plt
from utilities import *

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
from astar.astar import  AStar
from astar.environment_simple import Environment3D


from TrainPlanningOperator3D import PlanningOperator3D, smooth_chi

Gradient Descent Function

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


def A*Planner():
    t1 = tic()
    path_cost, path, action_idx = AStar.plan(start_coord, env)
    dt = toc(t1)

def FMMPlanner():        
    t1 = tic()
    solver = pykonal.EikonalSolver(coord_sys = "Cartesian")
    velocity_matrix  = 1 - cmap
    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = 1, 1, 1
    solver.velocity.npts = Sx, Sy, Sz
    solver.velocity.values = velocity_matrix.reshape(Sx, Sy, Sz)
    src_idx = goal_point[0].astype(int), goal_point[1].astype(int), goal_point[2].astype(int)
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    solver.solve()
    success, pathlength = perform_gradient_descent(solver.traveltime.values,start_point,goal_point)
    dt = toc(t1)


def testplanneronmaps(starts, goals, maps, heuristic, plotresults = False, printvalues = True, **kwargs):
    avgpathcost, avgplantime, avginfertime, avgnodesexp, avgsuccessrate = 0, 0, 0, 0, 0
    totpathcost, totplantime, totinfertime, totnodesexp, succcount = 0, 0, 0, 0, 0

    numofmaps = maps.shape[0]

    for start, goal, map in zip(starts, goals, 1-maps):

        # Call the heuristic function with additional arguments
        valuefunction, dt_infer = heuristic(map, goal,**kwargs)
        
        env = Environment2D(goal, map, valuefunction)
        
        t0 = tic()
        path_cost, path, action_idx, nodes_count, sss = AStar.plan(start, env)
        dt_plan = toc(t0)

        if path_cost < 10e10:
            succcount += 1

        totpathcost += path_cost
        totplantime += dt_plan
        totinfertime += dt_infer
        totnodesexp += nodes_count

        path_array = np.asarray(path)
        
        if plotresults:
            f, ax = plt.subplots()
            drawMap(ax, map)
            plotClosedNodes(ax,sss)
            plotInconsistentNodes(ax,sss,env)
            drawPath2D(ax, path_array)
    

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

