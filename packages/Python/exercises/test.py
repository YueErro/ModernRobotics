import modern_robotics as mr

se3mat = [[0, 0, 0, 0],
          [0, 0, -1.5708, 2.3562],
          [0, 1.5708, 0, 2.3562],
          [0, 0, 0, 0]]

T = mr.MatrixExp6(se3mat)

print("Working fine")
