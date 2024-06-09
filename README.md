# Planning Operator: Generalizable Planner using Neural Operators
Abstract: Classical tools for motion planning in complex environments have long suffered from performance
degradation as the dimension of the environment scales. As such, many modern approaches to motion
planning augment classical techniques with neural network (NN) based models to enhance computa-
tional efficacy [24 ]. With this development, recent connections have been made between the robotics
and the scientific machine learning (SciML) community for solving PDEs with physics informed
neural networks (PINNs). In particular, the robot motion planning problem can be formulated from
an optimal control perspective leading to the Eikonal equation, a simplified Hamilton-Jacobi (HJ)
PDE whose solution is the optimal value function. Recent works have explored solving
the Eikonal equation via PINNs with the advantage that no "expert" training data is needed for the
environment. However, as with classical PINNs, these approaches suffer from the fact that retraining
is required for each individual environment and therefore is computationally intractable for dynamic
environments governing the real-world.
In this work, we reformulate the solution to the Eikonal equation as an operator learning problem
between function spaces allowing a single neural operator to be trained across an entire Banach space
of cost functions representing multiple environments. With such an approach, the neural operator can
be trained once offline and then used online in dynamic environments without retraining (See Figure
1). Furthermore, the we take advantage of the resolution invariance property of neural operators 
generating our training set using coarse resolution test data taking only X minutes to generate and
train while applying the neural operator on grids add sizex the training data size.
Our method’s main contributions are summarized as follows (1) A novel formulation of the motion
planning problem as an operator learning problem and the theoretical existence of an neural operator
approximation to the value function’s solution with arbitrary accuracy. To the authors knowledge,
this is the first work to consider operator learning approaches applied to any type of robotic problem.
(2) Numerical experiments on both 2D and 3D environments demonstrating strong accuracy across a
set of randomized environments and computational speedups. (3) The introduction
of hard encoding of obstacles in the operator architecture and an ablation study highlighting the
improvements using the DAFNO architecture.


Datasets available at
https://drive.google.com/drive/folders/1qaw0HXASEdKSN94yIYvmCMTpz0GbKUUJ?usp=sharing

Download and place the files in their respective folders. This is a work in progress!


