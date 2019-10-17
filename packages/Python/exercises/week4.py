import numpy as np
import modern_robotics as mr

Ras = [[0, 0, 1],
       [-1, 0, 0],
       [0, -1, 0]]
Rsa = mr.RotInv(Ras)
pVa = [0, 0, 1]

Rbs = [[1, 0, 0],
       [0, 0, -1],
       [0, 1, 0]]
Rsb = mr.RotInv(Rbs)
pVb = [0, 2, 0]

T = [[0, -1, 0, 3],
     [1, 0, 0, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 1]]
sbracketstheta = [[0, -1.5708, 0, 2.3562],
                  [1.5708, 0, 0, -2.3562],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]]


shat = [1, 0, 0]
p = [0, 0, 2]

h = 1

pb = np.array([1, 2, 3]).T
vs = np.array([3, 2, 1, -1, -2, -3]).T
stheta = np.array([0, 1, 2, 3, 0, 0]).T
fb = np.array([1, 0, 0, 2, 1, 0]).T
v = np.array([1, 0, 0, 0, 2, 3]).T

print("Ex1, Tsa:")
Tsa = mr.RpToTrans(Rsa, pVa)
print(Tsa, "\n")
# [[0,-1,0,0],[0,0,-1,0],[1,0,0,1],[0,0,0,1]]

print("Ex2, Inverse of Tsb:")
Tsb = mr.RpToTrans(Rsb, pVb)
Tbs = mr.TransInv(Tsb)
print(Tbs, "\n")
# [[1,0,0,0],[0,0,-1,0],[0,1,0,-2],[0,0,0,1]]

print("Ex3, Tab:")
Tas = mr.TransInv(Tsa)
Tab = np.dot(Tas, Tsb)
print(Tab, "\n")
# [[0,-1,0,-1],[-1,0,0,0],[0,0,-1,-2],[0,0,0,1]]

print("Ex5, pb:")
pb = np.append(pb, 1)
Tsb = mr.TransInv(Tbs)
pb = np.dot(Tsb, pb)
pb = np.delete(pb, 3)
print(pb, "\n")
# [1,5,-2]

print("Ex7, va:")
Tsa = mr.Adjoint(Tas)
va = np.dot(Tsa, vs)
print(va, "\n")
# [1,-3,-2,-3,-1,5]

print("Ex8, theta:")
MatLogTsa = mr.MatrixLog6(Tsa)
vec = mr.so3ToVec(MatLogTsa)
theta = mr.AxisAng6(vec)[-1]
print(theta, "\n")
# 2.094395102393196

print("Ex9, Matrix exponential:")
se3 = mr.VecTose3(stheta)
MatExp = mr.MatrixExp6(se3)
print(MatExp, "\n")
# [[-0.61727288,-0.70368982,0.35184491,1.05553472],[0.70368982,-0.2938183,0.64690915,1.94072745],[-0.35184491,0.64690915,0.67654542,-0.97036373],[0,0,0,1]]

print("Ex10, fb:")
Tbs = mr.Adjoint(Tbs)
Tsb = np.transpose(Tbs)
fb = np.dot(Tsb, fb)
print(fb, "\n")
# [-1,0,-4,2,0,-1]

print("Ex11, TransInv:")
Tinv = mr.TransInv(T)
print(Tinv, "\n")
# [[0,1,0,0],[-1,0,0,3],[0,0,1,-1],[0,0,0,1]]

print("Ex12, VecTose3:")
se3 = mr.VecTose3(v)
print(se3, "\n")
# [[0,0,0,0],[0,0,-1,2],[0,1,0,3],[0,0,0,0]]

print("Ex13, ScrewToAxis:")
screw = mr.ScrewToAxis(p, shat, h)
print(screw, "\n")
# [1,0,0,1,2,0]

print("Ex14, MatrixExp6:")
TransMat = mr.MatrixExp6(sbracketstheta)
print(TransMat, "\n")
# [[0,-1,0,3],[1,0,0,0],[0,0,1,1],[0,0,0,1]]

print("Ex15, MatrixLog6:")
MatLog = mr.MatrixLog6(T)
print(MatLog)
# [[0,-1.57079633,0,2.35619449],[1.57079633,0,0,-2.35619449],[0,0,0,1],[0,0,0,0]]
