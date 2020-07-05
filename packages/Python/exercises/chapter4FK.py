import numpy as np
import modern_robotics as mr
import math

print("Ex 1: Determina the end-effector zero configuration M")
M = np.array([[1, 0, 0, 3.73],
              [0, 1, 0,    0],
              [0, 0, 1, 2.73],
              [0, 0, 0,    1]])
print(M)
# [[1,0,0,3.73],[0,1,0,0],[0,0,1,2.73],[0,0,0,1]]

print("Ex 2: Determine the screw axes Si")
Ss = np.array([[ 0, 0,    0,     0, 0,     0],
               [ 0, 1,    1,     1, 0,     0],
               [ 1, 0,    0,     0, 0,     1],
               [ 0, 0,    1, -0.73, 0,     0],
               [-1, 0,    0,     0, 0, -3.73],
               [ 0, 1, 2.73,  3.73, 1,     0]])
print(Ss)
# [[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1],[0,0,1,-0.73,0,0],[-1,0,0,0,0,-3.73],[0,1,2.73,3.73,1,0]]

print("Ex 3: Determine the screw aes Bi")
Bs = np.array([[   0,     0,    0, 0, 0, 0],
               [   0,     1,    1, 1, 0, 0],
               [   1,     0,    0, 0, 0, 1],
               [   0,  2.73, 3.73, 2, 0, 0],
               [2.73,     0,    0, 0, 0, 0],
               [   0, -2.73,   -1, 0, 1, 0]])
print(Bs)
# [[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1],[0,2.73,3.73,2,0,0],[2.73,0,0,0,0,0],[0,-2.73,-1,0,1,0]]

print("Ex 4: Find the end-effector configuration T using FKinSpace")
theta = np.array([-math.pi/2, math.pi/2, math.pi/3, -math.pi/4, 1, math.pi/6])
T_space = np.around(mr.FKinSpace(M, Ss, theta), 2)
print(T_space)
# [[0.5,0.87,0,1],[0.22,-0.13,-0.97,-1.9],[-0.84,0.48,-0.26,-4.5],[0,0,0,1]]

print("Ex 5: Find the end-effector configuration T using FKinBody")
T_body = np.around(mr.FKinBody(M, Bs, theta), 2)
print(T_body)
# [[0.5,0.87,0,1],[0.22,-0.13,-0.97,-1.9],[-0.84,0.48,-0.26,-4.5,],[0,0,0,1]]
