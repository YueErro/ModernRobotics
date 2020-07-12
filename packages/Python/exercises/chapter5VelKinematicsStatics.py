import numpy as np
import modern_robotics as mr
import math

print("Ex 1: What torques must be aplied at each of the joints?")
Fs = np.array([0, 0, 0, 2, 0, 0]) # tip generates a wrench in x direction
theta = np.array([0, math.pi/4, 0])
Ss = np.array([[0,  0,  0],
               [0,  0,  0],
               [1,  1,  1],
               [0,  0,  0],
               [0, -1, -2],
               [0,  0,  0]])
Js = mr.JacobianSpace(Ss, theta)
Ts = np.around(np.dot(Js.T, Fs), decimals=2)
print(Ts)
# [0,0,1.41]

print("Ex 2: What are the torques at each of the joints")
Fb = np.array([0, 0, 10, 10, 10, 0])
theta = np.array([0, 0, math.pi/2, -math.pi/2])
Bs = np.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 1, 1, 1],
               [0, 0, 0, 0],
               [4, 3, 2, 1],
               [0, 0, 0, 0]])
Jb = mr.JacobianBody(Bs, theta)
Tb = np.around(np.dot(Jb.T, Fb), decimals=2)
print(Tb)
# [30,20,10,20]

print("Ex 3: Use JacobianSpace to calculate the 6x3 space Jacobian")
theta = np.array([math.pi/2, math.pi/2, 1])
Ss = np.array([[0, 1, 0],
              [0, 0, 0],
              [1, 0, 0],
              [0, 0, 0],
              [0, 2, 1],
              [0, 0, 0]])
Js = np.around(mr.JacobianSpace(Ss, theta), decimals=2)
print(Js)
# [[0,0,0],[0,1,0],[1,0,0],[0,-2,-0],[0,0,0],[0,0,1]]

print("Ex 4: Use JacobianBody to calculate the 6x3 body Jacobian")
theta = np.array([math.pi/2, math.pi/2, 1])
Bs = np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 0],
              [3,  0, 0],
              [0,  3, 0],
              [0,  0, 1]])
Jb = np.around(mr.JacobianBody(Bs, theta), decimals=2)
print(Jb)
# [[0,-1,0],[0,0,0],[1,0,0],[0,0,0],[0,4,0],[0,0,1]]

print("Ex 5: Calculate the directions and lengths of the principal semi-axes of three-dimensional linear velocity manipulability ellipsoid")

Jb = np.array([[     0,     -1,     0,      0,    -1,     0, 0],
               [     0,      0,     1,      0,     0,     1, 0],
               [     1,      0,     0,      1,     0,     0, 1],
               [-0.105,      0, 0.006, -0.045,     0, 0.006, 0],
               [-0.889,  0.006,     0, -0.844, 0.006,     0, 0],
               [     0, -0.105, 0.889,      0,     0,     0, 0]])
Jv = Jb[3:] # extract linear velocity
A = np.dot(Jv, Jv.T)
w, v = np.linalg.eig(A)
max_id = np.where(w == np.amax(w))[0][0]
vec_prin_axis = np.around(v[max_id], decimals=2)
print(vec_prin_axis)
# [0.09,1,-0]

print("Ex 6: gi the length of the longest principal semi-axis of the previus exercise")
longest = np.around(math.sqrt(w[max_id]), decimals=2)
print(longest)
# 1.23
