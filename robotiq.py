import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
from scipy.spatial import ConvexHull, distance
from scipy.spatial.transform import Rotation as R


class robotiq:

  def __init__(self, urdfRootPath="urdf/", timeStep=1. / 240., isDiscrete=False, multiDiscrete=False):

    self.pybulleturdfRootPath = pybullet_data.getDataPath()
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self._isDescrete = isDiscrete
    self._multiDiscrete = multiDiscrete
    self.maxVelocity = .35
    self.maxForce = 200.
    self.firstjointidx = [1, 5, 9]
    self.secondjointidx = [2, 6, 10]
    self.thirdjointidx = [3, 7, 11]

    self.fingerAForce = 2
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 0
    self.useSimulation = 1
    self.useNullSpace = 21
    self.useOrientation = 1
    #lower limits for null space
    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    #upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    #joint ranges for null space
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]

    self.reset()
    self.getjointstate()
    self.getFingersConvexHull()

  def reset(self):

    startpos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0, -math.pi/2, 0]) #p.getQuaternionFromEuler([0,math.pi/2,0])
    self.robotiqUid = p.loadURDF(os.path.join(self.urdfRootPath, "robotiq.urdf"), startpos, startOrientation)
    # p.changeDynamics(self.robotiqUid, -1, mass=100)
    # self.robotiqUid = p.loadURDF(os.path.join(self.urdfRootPath, "robotiq_macos.urdf"), [0.000000, 0.000000, 1.00000], p.getQuaternionFromEuler([0, math.pi / 2 ,0]))
    self.cid = p.createConstraint(self.robotiqUid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], startpos)
    
    # p.changeDynamics(self.robotiqUid, -1, contactStiffness= 10000, contactDamping= 0.5)
    # for i in range(self.numJoints):
    #   p.changeDynamics(self.robotiqUid, i, contactStiffness= 10000, contactDamping= 0.5)
    self.numJoints = p.getNumJoints(self.robotiqUid)
  
  def getjointstate(self):
    self.jointstate = []
    self.jointPositions = self.numJoints * [ 0.0 ]
    # for jointIndex in range(self.numJoints):
    #   p.resetJointState(self.robotiqUid, jointIndex, self.jointPositions[jointIndex])

      # p.setJointMotorControl2(self.robotiqUid,
      #                         jointIndex,
      #                         p.POSITION_CONTROL,
      #                         targetPosition=self.jointPositions[jointIndex],
      #                         force=self.maxForce)
    #   print(f"joint {jointIndex}:  ")

    # self.trayUid = p.loadURDF(os.path.join(self.pybulleturdfRootPath, "tray/tray.urdf"), 0.640000, 0.075000, 0.010000, 0.000000, 0.000000, 1.000000, 0.000000)

    self.motorNames = []
    self.motorIndices = []
    jointangles = []
    self.linkpos = []

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.robotiqUid, i)
      jointstate = p.getJointState(self.robotiqUid, i)
      linkstate = p.getLinkState(self.robotiqUid, i)
      qIndex = jointInfo[3]
      self.motorNames.append(str(jointInfo[1]))
      self.linkpos.append(linkstate[0])
      self.motorIndices.append(i)
      self.jointstate.append(jointstate[0])
      jointangles.append(jointstate[0])
    # print("joint state: ", jointangles)

  def getActionDimension(self):
    return 6  #position x,y,z and roll/pitch/yaw euler angles of gripper

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self): # return pose, euler, linvel, angvel of gripper base

    pos , orn = p.getBasePositionAndOrientation(self.robotiqUid)
    # pos = np.array(pos,dtype=np.float32)
    ornEuler = p.getEulerFromQuaternion(orn)
    linvel, angvel = np.array(p.getBaseVelocity(self.robotiqUid))
    # linvel = np.asarray(linvel,dtype=np.float32)
    # angvel = np.asarray(angvel,dtype=np.float32)
    l1j1_state = np.array(p.getJointState(self.robotiqUid,1)[0])
    l2j1_state = np.array(p.getJointState(self.robotiqUid,5)[0])
    l3j1_state = np.array(p.getJointState(self.robotiqUid,9)[0])

    fingerpose = (np.array(p.getLinkStates(self.robotiqUid, self.thirdjointidx),dtype=object))[:,0]
    observation = np.array([pos[0], pos[1], pos[2],
                            ornEuler[0], ornEuler[1], ornEuler[2],
                            linvel[0], linvel[1], linvel[2],
                            angvel[0], angvel[1], angvel[2]], dtype=np.float32)
                            # l1j1_state, l2j1_state, l3j1_state]
                            #
    
    observation1 = np.array([pos[0], pos[1], pos[2], 
                            orn[0], orn[1], orn[2], orn[3],
                            linvel[0], linvel[1], linvel[2],
                            angvel[0], angvel[1], angvel[2],
                            l1j1_state, l2j1_state, l3j1_state], dtype=np.float32)
    
    observation2 = np.array([pos[0], pos[1], pos[2], 
                            orn[0], orn[1], orn[2], orn[3],
                            linvel[0], linvel[1], linvel[2],
                            angvel[0], angvel[1], angvel[2],
                            fingerpose[0][0],fingerpose[0][1],fingerpose[0][2],
                            fingerpose[1][0],fingerpose[1][1],fingerpose[1][2],
                            fingerpose[2][0],fingerpose[2][1],fingerpose[2][2]], dtype=np.float32)
    

    return observation

  def getFingersConvexHull(self):

    points = np.array(p.getBasePositionAndOrientation(self.robotiqUid)[0])
    for i in range(self.numJoints):
      linkWorldPosition = p.getLinkState(self.robotiqUid , i)[0]
      points = np.vstack([points,np.array(linkWorldPosition)])

    self.fingvol = ConvexHull(points=points)

    return self.fingvol

  def applyAction(self, BaseCommands):
    
    if (self.useInverseKinematics):

      dx = BaseCommands[0]
      dy = BaseCommands[1]
      dz = BaseCommands[2]
      da = BaseCommands[3]
      fingerAngle = BaseCommands[4]

      state = p.getLinkState(self.robotiqUid, self.kukaEndEffectorIndex)
      actualEndEffectorPos = state[0]
      #print("pos[2] (getLinkState(kukaEndEffectorIndex)")
      #print(actualEndEffectorPos[2])

      self.endEffectorPos[0] = self.endEffectorPos[0] + dx
      if (self.endEffectorPos[0] > 0.65):
        self.endEffectorPos[0] = 0.65
      if (self.endEffectorPos[0] < 0.50):
        self.endEffectorPos[0] = 0.50
      self.endEffectorPos[1] = self.endEffectorPos[1] + dy
      if (self.endEffectorPos[1] < -0.17):
        self.endEffectorPos[1] = -0.17
      if (self.endEffectorPos[1] > 0.22):
        self.endEffectorPos[1] = 0.22

      #print ("self.endEffectorPos[2]")
      #print (self.endEffectorPos[2])
      #print("actualEndEffectorPos[2]")
      #print(actualEndEffectorPos[2])
      #if (dz<0 or actualEndEffectorPos[2]<0.5):
      self.endEffectorPos[2] = self.endEffectorPos[2] + dz

      self.endEffectorAngle = self.endEffectorAngle + da
      pos = self.endEffectorPos
      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
      if (self.useNullSpace == 1):
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.robotiqUid, self.kukaEndEffectorIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp)
        else:
          jointPoses = p.calculateInverseKinematics(self.robotiqUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos,
                                                    lowerLimits=self.ll,
                                                    upperLimits=self.ul,
                                                    jointRanges=self.jr,
                                                    restPoses=self.rp)
      else:
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.robotiqUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd)
        else:
          jointPoses = p.calculateInverseKinematics(self.robotiqUid, self.kukaEndEffectorIndex, pos)

      #print("jointPoses")
      #print(jointPoses)
      #print("self.kukaEndEffectorIndex")
      #print(self.kukaEndEffectorIndex)
      if (self.useSimulation):
        for i in range(self.kukaEndEffectorIndex + 1):
          #print(i)
          p.setJointMotorControl2(bodyUniqueId=self.robotiqUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.robotiqUid, i, jointPoses[i])
      #fingers
      p.setJointMotorControl2(self.robotiqUid,
                              7,
                              p.POSITION_CONTROL,
                              targetPosition=self.endEffectorAngle,
                              force=self.maxForce)
      p.setJointMotorControl2(self.robotiqUid,
                              8,
                              p.POSITION_CONTROL,
                              targetPosition=-fingerAngle,
                              force=self.fingerAForce)
      p.setJointMotorControl2(self.robotiqUid,
                              11,
                              p.POSITION_CONTROL,
                              targetPosition=fingerAngle,
                              force=self.fingerBForce)

      p.setJointMotorControl2(self.robotiqUid,
                              10,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)
      p.setJointMotorControl2(self.robotiqUid,
                              13,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)

    else:
      pos , orn = p.getBasePositionAndOrientation(self.robotiqUid)
      pos = np.array(pos)
      orn = np.array(p.getEulerFromQuaternion(orn))
      poscommand = pos + np.array(BaseCommands[0:3]) * 0.01
      orncommand = orn + np.array(BaseCommands[3:6]) * 0.1
      # orncommand = np.remainder(orncommand, np.pi)
      # print(f"orncommand: {orncommand}")
      orncommand = p.getQuaternionFromEuler(orncommand)
      


      p.changeConstraint( self.cid, jointChildPivot =  poscommand, jointChildFrameOrientation = orncommand, maxForce=500)

if __name__ == "__main__":
  robotiq()