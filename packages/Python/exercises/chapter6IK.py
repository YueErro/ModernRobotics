import numpy as np
import modern_robotics as mr
import math

print("Ex 1: Use Newton-Raphso iterative numerical root finding to perform two steps:")

def f_x_y(theta):
    return np.array([theta[0]**2-9, theta[1]**2-4])

def jacobian(theta):
    return np.array([[2*theta[0], 0], [0, 2*theta[1]]])

count = 0
theta = [1, 1] # initial guess, theta_0
while count < 2:
    count += 1
    theta -= np.dot(np.linalg.inv(jacobian(theta)), f_x_y(theta))

theta = np.around(theta, decimals=2)
print(theta)
# [3.4,2.05]

print("Ex 2: Give theta_d:")
Bs = np.array([[0, 0, 0],
               [0, 0, 0],
               [1, 1, 1],
               [0, 0, 0],
               [3, 2, 1],
               [0, 0, 0]])
M = np.array([[1, 0, 0, 3],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
T_sd = np.array([[-0.585, -0.811, 0, 0.076],
                 [ 0.811, -0.585, 0, 2.608],
                 [     0,      0, 1,     0],
                 [     0,      0, 0,     1]])
theta_0 = [math.pi/4, math.pi/4, math.pi/4]
e_w = 0.001 # omega
e_v = 0.0001

(theta, success) = mr.IKinBody(Bs, M, T_sd, theta_0, e_w, e_v)

if success:
    theta = np.around(theta, decimals=2)
    print(theta)
else:
    raise ValueError("Failed to converge")
# [0.93,0.59,0.68]
