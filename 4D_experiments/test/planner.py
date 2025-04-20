import numpy as np
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt
import skfmm


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from functools import reduce
from functools import partial

#   # Create a custom colormap that is inverted
cmap = plt.cm.viridis  # Choose the colormap you want to invert
cmap_inverted = LinearSegmentedColormap.from_list("inverted_viridis", cmap(np.linspace(1, 0, 256)))

from timeit import default_timer
import scipy.io
import sys, random
from itertools import chain


from generator.fmm_data_generator_4d import calculate_signed_distance
from models.TrainPlanningOperator4D import PlanningOperator4D, smooth_chi
from models.utilities import *

# Primary movements in 4D: forward, backward, left, right, up, down, and movements in the 4th dimension
primary_moves_4d = [
    [-1., 0., 0., 0.],  # left
    [1., 0., 0., 0.],   # right
    [0., 1., 0., 0.],   # forward
    [0., -1., 0., 0.],  # backward
    [0., 0., 1., 0.],   # up
    [0., 0., -1., 0.],  # down
    [0., 0., 0., 1.],   # positive w
    [0., 0., 0., -1.]   # negative w
]

# Diagonal movements in 4D
diagonal_moves_4d = [
    [-1., 1., 0., 0.],  # left-forward
    [-1., -1., 0., 0.], # left-backward
    [1., 1., 0., 0.],   # right-forward
    [1., -1., 0., 0.],  # right-backward
    [-1., 0., 1., 0.],  # left-up
    [-1., 0., -1., 0.], # left-down
    [1., 0., 1., 0.],   # right-up
    [1., 0., -1., 0.],  # right-down
    [0., 1., 1., 0.],   # forward-up
    [0., 1., -1., 0.],  # forward-down
    [0., -1., 1., 0.],  # backward-up
    [0., -1., -1., 0.], # backward-down
    [-1., 1., 1., 0.],  # left-forward-up
    [-1., 1., -1., 0.], # left-forward-down
    [-1., -1., 1., 0.], # left-backward-up
    [-1., -1., -1., 0.],# left-backward-down
    [1., 1., 1., 0.],   # right-forward-up
    [1., 1., -1., 0.],  # right-forward-down
    [1., -1., 1., 0.],  # right-backward-up
    [1., -1., -1., 0.], # right-backward-down
    # Additional diagonal movements involving the 4th dimension
    [-1., 0., 0., 1.],  # left-positive w
    [-1., 0., 0., -1.], # left-negative w
    [1., 0., 0., 1.],   # right-positive w
    [1., 0., 0., -1.],  # right-negative w
    [0., 1., 0., 1.],   # forward-positive w
    [0., 1., 0., -1.],  # forward-negative w
    [0., -1., 0., 1.],  # backward-positive w
    [0., -1., 0., -1.], # backward-negative w
    [0., 0., 1., 1.],   # up-positive w
    [0., 0., 1., -1.],  # up-negative w
    [0., 0., -1., 1.],  # down-positive w
    [0., 0., -1., -1.], # down-negative w
]

# Combine primary and diagonal moves
action_vecs_4d = np.asarray(primary_moves_4d + diagonal_moves_4d)

def perform_gradient_descent(value_function, start_point, goal_point, learning_rate=1, num_steps=1000):
    path_length = 0
    path_points = [start_point.copy().astype(float)]
    visited_points = set()
    current_point = start_point.copy().astype(float)

    for step in range(num_steps):
        best_gradient = np.inf
        best_action = None

        for action in action_vecs_4d:
            new_point = current_point + learning_rate * action
            new_point_indices = np.round(new_point).astype(int)
            x_index, y_index, z_index, w_index = new_point_indices

            if (0 <= x_index < value_function.shape[0] and 
                0 <= y_index < value_function.shape[1] and 
                0 <= z_index < value_function.shape[2] and
                0 <= w_index < value_function.shape[3] and
                (x_index, y_index, z_index, w_index) not in visited_points):
                gradient = value_function[x_index, y_index, z_index, w_index]
                if gradient < best_gradient:
                    best_gradient = gradient
                    best_action = action

        if best_gradient > 100:
            return False, 0, path_points  

        if best_action is not None:
            current_point += learning_rate * best_action
            path_length += np.linalg.norm(learning_rate * best_action)
            path_points.append(current_point.copy())
            visited_points.add(tuple(np.round(current_point).astype(int)))
            if np.array_equal(np.round(current_point).astype(int), np.round(goal_point).astype(int)):
                return True, path_length, path_points  # Success
        else:
            return False, 0, path_points  # No valid action found
    return False, 0 ,path_points 

def scikitFMM(map,goal):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map'''

    env_size_x, env_size_y, env_size_z, env_size_a = map.shape
    phi = np.ones((env_size_x, env_size_y, env_size_z, env_size_a))
    phi[goal[0].astype(int),goal[1].astype(int),goal[2].astype(int), goal[3].astype(int)] = 0
    velocity_matrix = (map)
    valuefunction = skfmm.travel_time(phi, speed = velocity_matrix).filled()
    return valuefunction

def generaterandompos(maps):
    "Generates random positions in the free space of given maps"

    numofmaps = maps.shape[0]
    env_size = maps.shape[1]
    pos = np.zeros((numofmaps,4))

    assert maps.shape[1] == maps.shape[2] == maps.shape[3] == maps.shape[4]

    for i,map in enumerate(maps):

        condition1 = map == 1
        x_indices, y_indices, z_indices, t_indices = np.indices(map.shape)
        condition2 = (x_indices < env_size) & (y_indices < env_size) & (z_indices < env_size) & (t_indices < env_size) #Assuming the environment size in all dimensions
        combined_condition = condition1 & condition2
        passable_indices = np.argwhere(combined_condition)
        point = random.choice(passable_indices)
        pos[i,:] = np.array([point[0],point[1],point[2],point[3]])

    return pos.astype(int)

def getPNOValueFunction(map, goal, model):
    mask = (map)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_size_x, env_size_y, env_size_z, env_size_t = map.shape
    sdf = calculate_signed_distance(mask)

    sdf = sdf.reshape(1, env_size_x, env_size_y, env_size_z, env_size_t, 1)
    mask = mask.reshape(1, env_size_x, env_size_y, env_size_z, env_size_t, 1)

    sdf = torch.tensor(sdf, dtype=torch.float, device=device)
    mask = torch.tensor(mask, dtype=torch.float, device=device)

    smooth_coef = 5.  # Depends on what it is trained on
    chi = smooth_chi(mask, sdf, smooth_coef)
    goal_coord = torch.tensor(goal, dtype=torch.float, device=device).reshape(1, 4, 1)

    valuefunction = model(chi, goal_coord)
    valuefunction = valuefunction.detach().cpu().numpy().reshape(env_size_x, env_size_y, env_size_z, env_size_t)
    mask = mask.detach().cpu().numpy().reshape(env_size_x,env_size_y,env_size_z,env_size_t)

    valuefunction = valuefunction / (mask + 1e-20)

    return valuefunction

def getFMMValueFunction(map,goal):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map'''

    env_size_x, env_size_y, env_size_z, env_size_a = map.shape
    phi = np.ones((env_size_x, env_size_y, env_size_z, env_size_a))
    phi[goal[0].astype(int),goal[1].astype(int),goal[2].astype(int), goal[3].astype(int)] = 0
    velocity_matrix = (map)
    valuefunction = skfmm.travel_time(phi, speed = velocity_matrix).filled()
    return valuefunction








