from asyncio import PidfdChildWatcher
import os, inspect
from turtle import distance
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import math
import urdfpy
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import time
import pybullet as p
import trimesh
# from . import robotiq
import robotiq
import random
import pybullet_data
from pkg_resources import parse_version
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull, distance
from pybullet_utils import transformations

largeValObservation = 40

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class robotiqGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               records=False,
               isDiscrete=False,
               multiDiscrete=False,
               rewardType='sparse',
               max_episode_steps=500):
    print("robotiqGymEnv __init__")
    self.reward_type = rewardType
    self._isDiscrete = isDiscrete
    self._multiDiscrete = multiDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._robotiqRoot = "urdf/"
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._achieved_goal = []
    self._desired_goal = []
    self._envStepCounter = 0
    self._renders = renders
    self._records = records
    self._maxSteps = max_episode_steps
    self.terminated = 0
    self._cam_dist = 0.4
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._reach = 0
    self._keypoints = 100
    self.distance_threshold = 0.04
    

    self._p = p
    if self._records:
      cid = p.connect(p.GUI, options="--mp4=test.mp4")#p.connect(p.GUI, options="--mp4=tsse.mp4")
    elif self._renders:
      cid = p.connect(p.GUI)
    else:
      p.connect(p.DIRECT)

    self.reset()
    

    ## observation space
    observation_high = np.array([largeValObservation] * len(self.getExtendedObservation()), dtype=np.float32)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)


    ## action space
    if (self._multiDiscrete):
      self.action_space = spaces.MultiDiscrete([5,5,5])
    elif (self._isDiscrete):
      self.action_space = spaces.Discrete(5)
    else:
      action_dim = 6
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim , dtype=np.float32)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    self.viewer = None

  def reset(self): # return _observaton form getExtendedObservation()
    
    # super().reset(seed=seed)
    self.terminated = 0
    p.resetSimulation()
    print("robot base reset")
    p.setPhysicsEngineParameter(numSolverIterations=150, numSubSteps=4, fixedTimeStep=self._timeStep, contactERP=0.9, globalCFM=0.00001) #globalCFM=0.00001
    p.setTimeStep(self._timeStep)

    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, 0])
    self._robotiq = robotiq.robotiq(urdfRootPath=self._robotiqRoot, timeStep=self._timeStep, isDiscrete=self._isDiscrete, multiDiscrete=self._multiDiscrete)
    grippose, _ = p.getBasePositionAndOrientation(self._robotiq.robotiqUid)
    randnumx  = random.uniform(-1,1)
    randnumy  = random.uniform(-1,1)
    randnumz  = random.uniform(-1,1)
    randnumrol = random.uniform(-1,1)
    randnumpitch = random.uniform(-1,1)
    randnumyaw = random.uniform(-1,1)
    randnumf1 = random.uniform(-1,1)
    randnumf2 = random.uniform(-1,1)
    randnumf3 = random.uniform(-1,1)

    #pos = grippose[1] + 0.6 * randnumy + 0.3 * np.sign(randnumy)
    xpos = 0.0 + 0.50 * randnumx
    # xpos = 0.16
    ypos = 0.8 + 0.1 * randnumy
    # ypos = 0.6
    zpos = 1.0 + 0.50 * randnumz 
    # zpos = 1.05
    targetpos = [xpos, ypos, zpos]

    # rol  = 0.00 + math.pi * random.uniform(-1,1)
    rol  = 0.00
    pitch = 0.00
    yaw = 0.00
    targetorn  = p.getQuaternionFromEuler([rol, pitch, yaw])

    # self.cube = p.loadURDF(os.path.join(self._robotiqRoot, "cube.urdf"), basePosition=[0,0.12,1], baseOrientation=targetorn, useMaximalCoordinates=True, useFixedBase=True)

    self.blockUid = p.loadURDF(os.path.join(self._robotiqRoot, "block.urdf"), 
                                basePosition=targetpos, baseOrientation=targetorn, useMaximalCoordinates=True) #, useFixedBase=True
    self.targetmass = 700
    
    p.changeDynamics(self.blockUid, -1, mass=self.targetmass)
    extforce = np.array([randnumf1, randnumf2, randnumf3]) * (100*self.targetmass)
    # extforce = np.array([-1,.5,1]) * (100*self.targetmass)
    p.applyExternalForce(self.blockUid, -1 , extforce , [0,0,0] , p.LINK_FRAME)
    # p.applyExternalTorque(self.blockUid, -1 , [0,100,0] , p.LINK_FRAME)
    
    p.setGravity(0, 0, 0)
    
    self._envStepCounter = 0
    self.success_counter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()

    return self._observation

  def _is_success(self):
    if self._reward() > 2 :
      self.success_counter += 1
    else:
      self.success_counter = 0

    return np.float32(self.success_counter > 100)

  def close(self): # p.disconnect()
    p.disconnect()

  def getExtendedObservation(self): # adding blockInGripper pose and ori into _observation
    self._observation = self._robotiq.getObservation()
    gripperPos , gripperOrn = p.getBasePositionAndOrientation(self._robotiq.robotiqUid)
    griplinvel, gripangvel = p.getBaseVelocity(self._robotiq.robotiqUid)
    blockPos, blockOri = p.getBasePositionAndOrientation(self.blockUid)
    blocklinVel, blockangVel = p.getBaseVelocity(self.blockUid)
    blockEul = p.getEulerFromQuaternion(blockOri)
    gripEul = p.getEulerFromQuaternion(gripperOrn)

    blockPose = np.array([blockPos[0], blockPos[1], blockPos[2],
                          blockEul[0], blockEul[1], blockEul[2]], dtype=np.float32)
    blockPosequat = np.array([blockPos[0], blockPos[1], blockPos[2],
                          blockOri[0], blockOri[1], blockOri[2], blockOri[3]], dtype=np.float32)
    self._observation = np.append(self._observation, blockPose)

    relPose = np.array([blockPos[0]-gripperPos[0], blockPos[1]-gripperPos[1], blockPos[2]-gripperPos[2],
                        blockEul[0]-gripEul[0], blockEul[1]-gripEul[1], blockEul[2]-gripEul[2]], dtype=np.float32)
    relPosequat = np.array([blockPos[0]-gripperPos[0], blockPos[1]-gripperPos[1], blockPos[2]-gripperPos[2],
                        blockOri[0]-gripperOrn[0], blockOri[1]-gripperOrn[1], blockOri[2]-gripperOrn[2], blockOri[3]-gripperOrn[3]], dtype=np.float32)
    self._observation = np.append(self._observation, relPose)

    blockVel = np.array([blocklinVel[0], blocklinVel[1], blocklinVel[2], 
                         blockangVel[0], blockangVel[1], blockangVel[2]], dtype=np.float32)
    self._observation = np.append(self._observation, blockVel)

    relVel = np.array([blocklinVel[0]-griplinvel[0], blocklinVel[1]-griplinvel[1], blocklinVel[2]-griplinvel[2],
                        blockangVel[0]-gripangvel[0], blockangVel[1]-gripangvel[1], blockangVel[2]-gripangvel[2]], dtype=np.float32)
    self._observation = np.append(self._observation, relVel)

    closestpoint = p.getClosestPoints(self._robotiq.robotiqUid, self.blockUid, 100, -1, -1)
    point1 = np.array(closestpoint[0][5], dtype=np.float32)
    point2 = np.array(closestpoint[0][6], dtype=np.float32)
    minpos = np.subtract(point1, point2)
    self._observation = np.append(self._observation, minpos)
    # self._observation = np.append(self._observation, closestpoint[0][8])
    
    totalforce = self._contactinfo()[4]
    self._observation = np.append(self._observation, totalforce)

    contactInfo = self._contactinfo()
    contactInfo = np.array([contactInfo[0], contactInfo[1], contactInfo[2], 
                            contactInfo[3][0], contactInfo[3][1], contactInfo[3][2]], dtype=np.float32)
    # self._observation = np.append(self._observation, contactInfo)

    return self._observation

  def step(self, action): # create realAction and return step2(realAction)
    dx = action[0]
    dy = action[1]
    dz = action[2]
    droll = action[3]
    dpitch = action[4]
    dyaw = action[5]
    realAction = [dx, dy, dz, droll, dpitch, dyaw]
    return self.step2(realAction)

  def step2(self, action):
    self._action = action
    for i in range(self._actionRepeat):
      self._robotiq.applyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()
    done = self._termination()
    # npaction = np.array([action[3]])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
    reward = self._reward()
    infos = {"is_success": self._is_success()}
    ob, reward, terminated, truncated, info = self._observation, reward, done, False, infos
    return ob, reward, terminated, info 

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":      return np.array([])
    base_pos, base_orn = self._p.getBasePositionAndOrientation(self._robotiq.robotiqUid)
    target_pos, target_orn = self._p.getBasePositionAndOrientation(self.blockUid)
    camera_pos = base_pos + np.array([0, 0.2, 0.1])
    width = RENDER_WIDTH
    height = RENDER_HEIGHT
    view_matrix = p.computeViewMatrix(cameraEyePosition=camera_pos, cameraTargetPosition=target_pos, cameraUpVector=[0, 1, 0])
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0)
    #renderer=self._p.ER_TINY_RENDERER)
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(width, height, viewMatrix = view_matrix, projectionMatrix = proj_matrix, 
                                                               shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL) #ER_BULLET_HARDWARE_OPENGL 


    rgb_array = np.array(rgbImg, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (height, width, 4))

    rgb_array = rgb_array[:, :, :3]
    p.resetDebugVisualizerCamera(cameraDistance = 0.6, cameraYaw = 45, cameraPitch = -10, cameraTargetPosition = camera_pos)

    return rgb_array

  def _termination(self):
    
    if (self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True

    return False

  def _contactinfo(self):
    totalNormalForce = 0
    totalLateralFriction1 = 0
    totalLateralFriction2 = 0
    ftipNormalForce = 0
    ftipLateralFriction1 = 0
    ftipLateralFriction2 = 0
    contactpoints = p.getContactPoints(self.blockUid, self._robotiq.robotiqUid)
    for c in contactpoints:
      totalNormalForce += c[9]
      totalLateralFriction1 += c[10]
      totalLateralFriction2 += c[12]
    ftip1 = p.getContactPoints(self.blockUid, self._robotiq.robotiqUid, -1, self._robotiq.thirdjointidx[0])
    ftip2 = p.getContactPoints(self.blockUid, self._robotiq.robotiqUid, -1, self._robotiq.thirdjointidx[1])
    ftip3 = p.getContactPoints(self.blockUid, self._robotiq.robotiqUid, -1, self._robotiq.thirdjointidx[2])
    for contact in [ftip1, ftip2, ftip3]:
      for c in contact:
        ftipNormalForce += c[9]
        ftipLateralFriction1 += c[10]
        ftipLateralFriction2 += c[12]
    
    ftipContactPoints = np.array([len(ftip1), len(ftip2), len(ftip3)])
    return ftipNormalForce, ftipLateralFriction1, ftipLateralFriction2, ftipContactPoints, totalNormalForce, totalLateralFriction1, totalLateralFriction2
  
  def _reward(self):

    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    blockOrnEuler = p.getEulerFromQuaternion(blockOrn)
    gripperPos , gripperOrn = p.getBasePositionAndOrientation(self._robotiq.robotiqUid)
    gripOrnEuler = p.getEulerFromQuaternion(gripperOrn)
    blocklinvel, blockangvel = p.getBaseVelocity(self.blockUid)
    griplinvel, gripangvel = p.getBaseVelocity(self._robotiq.robotiqUid)
    blocklinvel = np.linalg.norm(blocklinvel)
    blockangvel = np.linalg.norm(blockangvel)
    griplinvel = np.linalg.norm(griplinvel)
    gripangvel = np.linalg.norm(gripangvel)

    closestPoints = np.absolute(p.getClosestPoints(self.blockUid, self._robotiq.robotiqUid, 100, -1, -1)[0][8] - self.distance_threshold)

    r = Rotation.from_quat(gripperOrn)
    normalvec = np.matmul(r.as_matrix(), np.array([0,1,0]))
    diffvector = np.subtract(np.array(blockPos), np.array(gripperPos))
    dotvec = np.dot(diffvector/np.linalg.norm(diffvector) , normalvec/np.linalg.norm(normalvec))
    redpoint = np.add(np.array(gripperPos), np.multiply(normalvec, 0.12))
    orifix = distance.euclidean(np.array(blockOrnEuler),np.array(gripOrnEuler))
    ftipNormalForce, ftipLateralFriction1, ftipLateralFriction1, ftipContactPoints, totalNormalForce, totalLateralFriction1, totalLateralFriction2 = self._contactinfo()
    r_top = self._r_topology()
    r_top = 1 if r_top > 0 else 0
    contactpenalize = -1 if totalNormalForce > 0 else 0
    
    distanceReward = 1 - math.tanh(closestPoints)
    oriReward = 1 - math.tanh(orifix)
    normalForceReward = 1 - math.tanh(totalNormalForce)
    gripangvelReward = 1 - math.tanh(gripangvel)
    fingerActionReward = 1 - math.tanh(abs(self._action[-1]))
    positionActionReward = np.linalg.norm(self._action[0:3]) / np.sqrt(3)
    orientationActionReward = np.linalg.norm(self._action[3:6]) / np.sqrt(3)
    
    # reward = -10*closestPoints + 10*dotvec  + 100*min(ftipContactPoints) - blocklinvel - blockangvel + r_top - totalNormalForce/100 - gripangvel/10
    # reward = distanceReward + oriReward + r_top #+ normalForceReward + gripangvelReward + fingerActionReward + r_top + dotvec + min(ftipContactPoints)
    reward = distanceReward + oriReward + r_top + contactpenalize # - (positionActionReward * distanceReward) - (orientationActionReward * oriReward)

    return reward

  def _r_topology(self):
    red_point_dot_radius = 2 # Small sphere radius
    red_point_dot_color = [1, 0, 0] # Red color: [r, g, b]
    red_point_dot_opacity = 1.0  # Fully opaque

    aabb_min, aabb_max = p.getAABB(self.blockUid)
    x = np.linspace(aabb_min[0], aabb_max[0], 5)
    y = np.linspace(aabb_min[1], aabb_max[1], 5)
    z = np.linspace(aabb_min[2], aabb_max[2], 5)
    
    points = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                points.append([x[i], y[j], z[k]])
    points = np.array(points)
    red_point_dot_color = np.array([red_point_dot_color] * len(points))
    
    # p.addUserDebugPoints(pointPositions=points,
    #                       pointColorsRGB=red_point_dot_color,
    #                       pointSize=red_point_dot_radius)



    gripper_link_pose = []
    for i in range(self._robotiq.numJoints):
      state = p.getLinkState(self._robotiq.robotiqUid, i)
      pos = state[0]
      gripper_link_pose.append(pos)

    gripper_link_pose = np.array(gripper_link_pose)
    hull = trimesh.convex.convex_hull(gripper_link_pose)
    n = hull.contains(points)
    n_count = np.count_nonzero(n) / self._keypoints
    # print(f"n_count {n_count}")

    return n_count


if __name__ == "__main__":
  robotiqGymEnv()