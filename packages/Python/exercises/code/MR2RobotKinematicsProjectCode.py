import math
import numpy as np
import modern_robotics as mr

def saveLog(i, thetalist, Tsb, Vb, omg_b_norm, v_b_norm, isappend="w"):
    """
    Creates a log.txt file with a report for each iteration of the Newton-Raphson process

    :param i: Iteration number
    :param thetalist: Join vector
    :param Tsb: The desired end-effector configuration
    :param Vb: Error twist
    :param omg_b_norm: Angular error magnitude
    :param v_b_norm: Linear error magnitude
    :param isappend: Text file open mode
    """
    np.set_printoptions(precision=3, suppress=True)
    with open("log.txt", isappend) as text_file:
        print(f"Iteration {i+1}:\n", file=text_file)
        print("joint vector: {}".format(np.array2string(thetalist, separator=",")[1:-1]), file=text_file)
        print("\n" "SE(3) end-effector config:", file=text_file)
        print(np.array2string(Tsb).replace('[', '').replace(']', ''), file=text_file)
        print("\n" "error twist V_b: {}".format(np.array2string(Vb, separator=",")[1:-1]), file=text_file)
        print(f"\nangular error magnitude ||omega_b||: {omg_b_norm:.3f}", file=text_file)
        print(f"\nlinear error magnitude ||v_b||: {v_b_norm:.3f}\n\n ", file=text_file)

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """
    Computes inverse kinematics in the body frame for an open chain robot and saves the report in a log.txt file

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.
    """
    thetalist = np.array(thetalist0).copy()
    thetamatrix = thetalist.T
    i = 0
    maxiterations = 20
    Tsb = mr.FKinBody(M, Blist, thetalist)
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T)))
    omg_b_norm = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    v_b_norm = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    err =  omg_b_norm > eomg or v_b_norm > ev
    saveLog(i, thetalist, Tsb, Vb, omg_b_norm, v_b_norm)

    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, \
                                                         thetalist)), Vb)
        thetamatrix = np.vstack((thetamatrix , thetalist.T))
        i = i + 1
        Tsb = mr.FKinBody(M, Blist, thetalist)
        Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T)))
        omg_b_norm = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        v_b_norm = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
        err =  omg_b_norm > eomg or v_b_norm > ev
        # print details to log.txt
        saveLog(i, thetalist, Tsb, Vb, omg_b_norm, v_b_norm, isappend="a")

    np.savetxt("iterates.csv", thetamatrix, delimiter = ",")

    return (thetamatrix, not err)

if __name__ == "__main__":
    # Example 4.5 of Chapter 4.1.2 (Figure 4.6)
    W1 = 0.109
    W2 = 0.082
    L1 = 0.425
    L2 = 0.392
    H1 = 0.089
    H2 = 0.095
    M = np.array([[-1, 0, 0, L1+L2],
                  [ 0, 0, 1, W1+W2],
                  [ 0, 1, 0, H1-H2],
                  [ 0, 0, 0,     1]])
    Blist = np.array([[    0,      0,   0,  0,   0, 0],
                      [    1,      0,   0,  0,  -1, 0],
                      [    0,      1,   1,  1,   0, 1],
                      [W1+W2,     H2,  H2, H2, -W2, 0],
                      [    0, -L1-L2, -L2,  0,   0, 0],
                      [L1+L2,      0,   0,  0,   0, 0]])
    T = np.array([[ 0, 1,  0, -0.5],
                    [ 0, 0, -1,  0.1],
                    [-1, 0,  0,  0.1],
                    [ 0, 0,  0,    1]])
    thetalist0 = np.array([math.pi*5/6, math.pi/6, -math.pi/2,
                           math.pi/3, -math.pi/6, -math.pi/2])
    eomg = 0.001
    ev = 0.0001

    thetamatrix, isconverge = IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)
