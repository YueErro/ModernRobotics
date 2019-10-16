import numpy as np
import modern_robotics as mr

Ras = [[0, 0, 1],
       [1, 0, 0],
       [0, 1, 0]]
Rbs = [[1, 0, 0],
       [0, 0, -1],
       [0, 1, 0]]
R12 = [[0, 0, 1],
       [-1, 0, 0],
       [0, -1, 0]]
so3 = [[0, 0.5, -1],
       [-0.5, 0, 2],
       [1, -2, 0]]

pb = np.array([1, 2, 3]).T
ws = np.array([3, 2, 1]).T
wtheta = np.array([1, 2, 0]).T
wskew = np.array([1, 2, 0.5]).T

print("Ex1, Rsa:")
Rsa = mr.RotInv(Ras)
print(Rsa, "\n")
# [[0,1,0],[0,0,1],[1,0,0]]

print("Ex2, Inverse of Rsb:")
print(Rbs, "\n")
# [[1,0,0],[0,0,-1],[0,1,0]]

print("Ex3, Rab:")
Rab = mr.RotInv(np.dot(Rbs, Rsa))
print(Rab, "\n")
# [[0,-1,0],[1,0,0],[0,0,1]]

print("Ex5, pb:")
Rsb = mr.RotInv(Rbs)
pb = Rsb * pb
print(pb, "\n")
# [1,3,-2]

print("Ex7, wa:")
wa = Rsa * ws
print(wa, "\n")
# [1,3,2]

print("Ex8, theta:")
MatLogRsa = mr.MatrixLog3(Rsa)
vec = mr.so3ToVec(MatLogRsa)
theta = mr.AxisAng3(vec)[-1]
print(theta, "\n")
# 2.094395102393196

print("Ex9, Matrix exponential:")
skew = mr.VecToso3(wtheta)
MatExp = mr.MatrixExp3(skew)
print(MatExp, "\n")
# [[-0.2938183,0.64690915,0.70368982],[0.64690915,0.67654542,-0.35184491],[-0.70368982,0.35184491,-0.61727288]]

print("Ex10, skew.symmetric matrix:")
skewMat = mr.VecToso3(wskew)
print(skewMat, "\n")
# [[0,-0.5,2],[0.5,0,-1],[-2,1,0]]

print("Ex11, Rotation matrix:")
RotMat = mr.MatrixExp3(so3)
print(RotMat, "\n")
# [[0.60482045,0.796274,-0.01182979],[0.46830057,-0.34361048,0.81401868],[0.64411707,-0.49787504,-0.58071821]]

print("Ex12, Matrix logarithm")
MatLogR12 = mr.MatrixLog3(R12)
print(MatLogR12)
# [[0,1.20919958,1.20919958],[-1.20919958,0,1.20919958],[-1.20919958,-1.20919958,0]]
