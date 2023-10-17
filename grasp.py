import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import os
import sys



# This function is used to load the URDF file of the robot
def loadURDF(robotName, robotStartPos, robotStartOrientation):
    flags = p.URDF_USE_SELF_COLLISION
    robotId = p.loadURDF(robotName, robotStartPos, robotStartOrientation, flags=flags, useFixedBase=1)
    return robotId

if __name__ == "__main__":
    gripper = loadURDF("gripper.urdf", [0, 0, 1], [0, 0, 0, 1])
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(1)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(0.001)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, lateralFriction=0.9)