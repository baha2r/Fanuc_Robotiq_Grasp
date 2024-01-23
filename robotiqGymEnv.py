from asyncio import PidfdChildWatcher
import os, inspect
from turtle import distance
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pybullet as p
import trimesh
import robotiq
import pybullet_data
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull, distance
from moviepy.editor import ImageSequenceClip
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

largeValObservation = 40

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class robotiqGymEnv(gym.Env):
    """
    This class describes the environment for the Robotiq robot.
    """

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=1,
                 renders=False,
                 records=False,
                 is_discrete=False,
                 multi_discrete=False,
                 max_episode_steps=1000,
                 savedir="test_data/test1",
                 ):
        """
        Initialize the environment.
        """
        self._is_discrete = is_discrete
        self._multi_discrete = multi_discrete
        self._timeStep = 1. / 240.
        self._urdf_root = urdf_root
        self._robotiqRoot = "urdf/"
        self._action_repeat = action_repeat
        self._observation = []
        self._stepcounter = 0
        self._renders = renders
        self._records = records
        self._max_steps = max_episode_steps
        self.terminated = 0
        self._cam_dist = 0.4
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._reach = 0
        self._keypoints = 100
        self.distance_threshold = 0.04
        self._accumulated_contact_force = 0
        self.distanceReward = 0
        self.oriReward = 0
        self.r_top = 0
        self.contactpenalize = 0
        self.target_yaw = 0
        self.savedir = savedir
        self._cam1_images = []
        self._cam2_images = []
        self._cam3_images = []

        # connect to PyBullet
        rec = f"--mp4={self.savedir}/video.mp4"
        if self._records:
            p.connect(p.GUI, options=rec)
        elif self._renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.reset()

        # Define observation space
        observation_high = np.array([largeValObservation] * len(self.getExtendedObservation()), dtype=np.float32)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)

        # Define action space
        if self._multi_discrete:
            self.action_space = spaces.MultiDiscrete([5,5,5])
        elif self._is_discrete:
            self.action_space = spaces.Discrete(5)
        else:
            action_dim = 6
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim , dtype=np.float32)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.viewer = None

    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(
            numSolverIterations=500, 
            # numSubSteps=4, 
            # fixedTimeStep=self._timeStep,
            contactERP=0.9, 
            globalCFM=0.01,
            enableConeFriction=1,
            contactSlop=0.0001,
            maxNumCmdPer1ms=1000,
            contactBreakingThreshold=0.01,
            enableFileCaching=1,
            restitutionVelocityThreshold=0.01,
        )
        # p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, 0])

        self._robotiq = robotiq.robotiq(
            urdf_root_path=self._robotiqRoot, 
            time_step=self._timeStep, 
        )

        grippose, _ = p.getBasePositionAndOrientation(self._robotiq.robotiq_uid)
        # p.changeDynamics(self._robotiq.robotiq_uid, -1, mass=1000)

        # Generate random values
        randx, randy, randz, randf1, randf2, randf3 = np.random.uniform(-1, 1, 6)

        targetpos = [0.0 + 0.50 * randx, 0.8 + 0.2 * randy, 1.0 + 0.40 * randz]
        # targetorn = p.getQuaternionFromEuler([0, 0, 0])
        targetorn = p.getQuaternionFromEuler([0, 0, self.target_yaw])
        # print(self.target_yaw)

        # targetpos = [0.07336462703085808,0.6302821367352937,0.9215777045808058]
        # targetorn = p.getQuaternionFromEuler([0, 0, 0])

        self.blockUid = p.loadURDF(
            os.path.join(self._robotiqRoot, "block.urdf"), 
            basePosition=targetpos, 
            baseOrientation=targetorn, 
            useMaximalCoordinates=True,
            flags=p.URDF_INITIALIZE_SAT_FEATURES
        )

        self.targetmass = 100
        p.changeDynamics(self.blockUid, -1,
                 mass=self.targetmass, # adjust mass
                 lateralFriction=0.35, # adjust lateral friction
                 spinningFriction=0.001, # adjust spinning friction
                 rollingFriction=0.001, # adjust rolling friction               
                 contactStiffness=1000, 
                 contactDamping=10
                ) # adjust contact stiffness and damping
        
        # for i in range(self._robotiq.num_joints):
        #     p.changeDynamics(self._robotiq.robotiq_uid, i, lateralFriction=1, spinningFriction=1, rollingFriction=1,
        #          restitution=0.001, contactStiffness=1000, contactDamping=10)
        extforce = np.array([randf1, randf2, randf3]) * (20 * self.targetmass)
        # extforce = np.array([1,1,1]) * (30 * self.targetmass)
        p.applyExternalForce(self.blockUid, -1 , extforce , [0,0,0] , p.LINK_FRAME)

        p.setGravity(0, 0, 0)
        self._stepcounter = 0
        self.success_counter = 0
        p.stepSimulation()

        self._observation = self.getExtendedObservation()

        return self._observation
    
    def _is_success(self):
        """
        Check if the current state is successful. 
        Success is defined as having a reward greater than 2 for more than 100 consecutive steps.
        """
        if self.success_counter > 300:
            return np.float32(1.0)

        if self._reward() > 2:
            self.success_counter += 1
        else:
            self.success_counter = 0

        return np.float32(self.success_counter > 300)
    
    def getExtendedObservation(self):
        """
        Return an extended observation that includes information about the block in the gripper.
        The extended observation includes relative position, velocity, and contact information.
        """
        self._observation = self._robotiq.get_observation()
        noise_mean = 0
        noise_std = 0.02

        # Fetch base position and orientation of gripper and block
        gripperPos, gripperOrn = p.getBasePositionAndOrientation(self._robotiq.robotiq_uid)
        griplinvel, gripangvel = p.getBaseVelocity(self._robotiq.robotiq_uid)
        blockPos, blockOri = p.getBasePositionAndOrientation(self.blockUid)
        blocklinVel, blockangVel = p.getBaseVelocity(self.blockUid)
        # change the block position and orientation and block velocity and angular velocity from tuple to np.array
        blockPos    = np.array(blockPos)
        blockOri    = np.array(blockOri)
        blocklinVel = np.array(blocklinVel)
        blockangVel = np.array(blockangVel)
        # add noise to block position and orientation and block velocity and angular velocity
        blockPos    += np.random.normal(noise_mean, noise_std, blockPos.shape)
        blockOri    += np.random.normal(noise_mean, noise_std, blockOri.shape)
        blocklinVel += np.random.normal(noise_mean, noise_std, blocklinVel.shape)
        blockangVel += np.random.normal(noise_mean, noise_std, blockangVel.shape)

        # Convert block and gripper orientation from Quaternion to Euler for ease of manipulation
        blockEul = p.getEulerFromQuaternion(blockOri)
        gripEul = p.getEulerFromQuaternion(gripperOrn)

        # Define block pose and append to observation
        blockPose = np.array([
            *blockPos, 
            *blockEul
        ], dtype=np.float32)
        self._observation = np.append(self._observation, blockPose)

        # Define relative pose and append to observation
        relPose = np.array([
            *(np.subtract(blockPos, gripperPos)), 
            *(np.subtract(blockEul, gripEul))
        ], dtype=np.float32)
        self._observation = np.append(self._observation, relPose)

        # Define block velocity and append to observation
        blockVel = np.array([
            *blocklinVel, 
            *blockangVel
        ], dtype=np.float32)
        self._observation = np.append(self._observation, blockVel)

        # Define relative velocity and append to observation
        relVel = np.array([
            *(np.subtract(blocklinVel, griplinvel)), 
            *(np.subtract(blockangVel, gripangvel))
        ], dtype=np.float32)
        self._observation = np.append(self._observation, relVel)

        # Add minimum distance between the robot and the block to observation
        closestpoints = p.getClosestPoints(self._robotiq.robotiq_uid, self.blockUid, 100, -1, -1)
        minpos = np.subtract(closestpoints[0][5], closestpoints[0][6])
        minpos = np.array(minpos)
        # minpos += np.random.normal(0, 0.01, minpos.shape)
        self._observation = np.append(self._observation, minpos)

        # add noise to self._observation
        # self._observation += np.random.normal(0, 0.01, self._observation.shape)

        # Add contact information to observation
        totalforce = self._contactinfo()[5]
        # self._observation = np.append(self._observation, totalforce)

        return self._observation
    
    def step(self, action):
        """
        Apply an action to the robot, simulate the physics for the defined action repeat,
        and return the current state, reward, termination, and success state of the environment.
        """
        # dx, dy, dz, droll, dpitch, dyaw = action
        # realAction = [dx, dy, dz, droll, dpitch, dyaw]

        self._action = action
        for _ in range(self._action_repeat):
            self._robotiq.apply_action(action)
            p.stepSimulation()
            self.render()
            if self._termination():
                break
            self._stepcounter += 1

        if self._renders:
            time.sleep(self._timeStep)

        self._observation = self.getExtendedObservation()
        done = self._termination()
        reward = self._reward()
        infos = {"is_success": self._is_success()}
        ob, reward, terminated, info = self._observation, reward, done, infos
        return ob, reward, terminated, info

    def render(self, mode="rgb_array", close=False):
        """
        Render the environment to an image array.

        Arguments:
        - mode: The mode to render with. The default is "rgb_array".
        - close: If True, closes the rendering window. The default is False.

        Returns: 
        - A RGB array of the rendered environment if mode is "rgb_array", otherwise an empty array.
        """
        # If mode is not "rgb_array", return an empty array
        if mode != "rgb_array":
            return np.array([])

        # Get the position and orientation of the base and target
        grip_pos, _ = p.getBasePositionAndOrientation(self._robotiq.robotiq_uid)
        target_pos, _ = p.getBasePositionAndOrientation(self.blockUid)

        # Calculate the camera position based on the base position
        camera_pos = grip_pos + np.array([0, -0.05, 0.2])

        # Define the size of the rendered image
        width, height = RENDER_WIDTH, RENDER_HEIGHT

        # # Calculate the view and projection matrices for the camera
        view_matrix_1 = p.computeViewMatrix(cameraEyePosition=camera_pos, cameraTargetPosition=target_pos, cameraUpVector=[0, 1, 0])
        view_matrix_2 = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=grip_pos, distance=0.9, yaw=90, pitch=-20, roll=0, upAxisIndex=2)
        view_matrix_3 = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=target_pos, distance=0.9, yaw=180, pitch=-20, roll=0, upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=10.0)

        # First camera
        _, _, rgbImg1, _, _ = p.getCameraImage(width, height, viewMatrix = view_matrix_1, projectionMatrix = proj_matrix)
        rgbImg1 = rgbImg1[:, :, :3]
        self._cam1_images.append(rgbImg1)

        # Second camera
        _, _, rgbImg2, _, _ = p.getCameraImage(width, height, viewMatrix = view_matrix_2, projectionMatrix = proj_matrix)
        rgbImg2 = rgbImg2[:, :, :3]
        self._cam2_images.append(rgbImg2)
        
        # third camera
        _, _, rgbImg3, _, _ = p.getCameraImage(width, height, viewMatrix = view_matrix_3, projectionMatrix = proj_matrix)
        rgbImg3 = rgbImg3[:, :, :3]
        self._cam3_images.append(rgbImg3)
        

        return np.array([])

    def _termination(self):
        """
        Check whether the environment should be terminated.

        Returns:
        - True if the environment has reached its maximum number of steps.
        - False otherwise.
        """
        if self._stepcounter > self._max_steps:
            return True
        
        return False

    def _contactinfo(self):
        """
        Compute various contact forces between the block and the robotiq and 
        the number of contact points for each fingertip.
        
        Returns:
        - ftipNormalForce: total normal force at the fingertips
        - ftipLateralFriction1: total lateral friction in one direction at the fingertips
        - ftipLateralFriction2: total lateral friction in the other direction at the fingertips
        - ftipContactPoints: array containing the number of contact points for each fingertip
        - totalNormalForce: total normal force between block and robotiq
        - totalLateralFriction1: total lateral friction in one direction between block and robotiq
        - totalLateralFriction2: total lateral friction in the other direction between block and robotiq
        """
        totalNormalForce = 0
        totalLateralFriction1 = 0
        totalLateralFriction2 = 0

        ftipNormalForce = 0
        ftipLateralFriction1 = 0
        ftipLateralFriction2 = 0

        # Get contact points between block and robotiq
        contactpoints = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid)

        # get number of contact points
        number_of_contact_points = len(contactpoints)
        totalLateralFrictionForce = [0, 0, 0]

        for c in contactpoints:
            # Sum up total forces
            totalNormalForce += c[9]
            totalLateralFrictionForce[0] += c[11][0] * c[10] + c[13][0] * c[12]
            totalLateralFrictionForce[1] += c[11][1] * c[10] + c[13][1] * c[12]
            totalLateralFrictionForce[2] += c[11][2] * c[10] + c[13][2] * c[12]
        
        totalLateralFrictionForce = np.array(totalLateralFrictionForce)
        
        # Get contact points between block and each fingertip of robotiq
        ftip1 = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid, -1, self._robotiq.third_joint_idx[0])
        ftip2 = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid, -1, self._robotiq.third_joint_idx[1])
        ftip3 = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid, -1, self._robotiq.third_joint_idx[2])

        # Loop over all the fingertips' contact points
        ftipLateralFrictionForce = [0, 0, 0]
        for contact in [ftip1, ftip2, ftip3]:
            for c in contact:
                # Sum up forces at the fingertips
                ftipNormalForce += c[9]
                ftipLateralFrictionForce[0] = c[11][0] * c[10] + c[13][0] * c[12]
                ftipLateralFrictionForce[1] = c[11][1] * c[10] + c[13][1] * c[12]
                ftipLateralFrictionForce[2] = c[11][2] * c[10] + c[13][2] * c[12]

        # Count the number of contact points for each fingertip
        ftipContactPoints = np.array([len(ftip1), len(ftip2), len(ftip3)])
        
        self._accumulated_contact_force += totalNormalForce

        return ftipNormalForce, ftipLateralFrictionForce, totalLateralFrictionForce, number_of_contact_points, ftipContactPoints, totalNormalForce, totalLateralFriction1, totalLateralFriction2

    def _reward(self):
        """
        Compute the reward for the current state of the environment.
        
        Reward is based on distance, orientation, force applied, speed of movement, and whether fingertips are in contact with the object.

        Returns:
        - reward: The calculated reward for the current state.
        """

        # Get position and orientation for block and gripper
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        blockOrnEuler = p.getEulerFromQuaternion(blockOrn)

        gripperPos , gripperOrn = p.getBasePositionAndOrientation(self._robotiq.robotiq_uid)
        gripOrnEuler = p.getEulerFromQuaternion(gripperOrn)

        # Get velocities for block and gripper
        blocklinvel, blockangvel = p.getBaseVelocity(self.blockUid)
        griplinvel, gripangvel = p.getBaseVelocity(self._robotiq.robotiq_uid)

        # Convert velocities to magnitude (speed)
        blocklinvel = np.linalg.norm(blocklinvel)
        blockangvel = np.linalg.norm(blockangvel)
        griplinvel = np.linalg.norm(griplinvel)
        gripangvel = np.linalg.norm(gripangvel)

        # Get closest distance between block and gripper, and subtract distance threshold
        closestPoints = np.absolute(p.getClosestPoints(self.blockUid, self._robotiq.robotiq_uid, 100, -1, -1)[0][8] - self.distance_threshold)

        # Compute orientation related quantities
        r = Rotation.from_quat(gripperOrn)
        normalvec = np.matmul(r.as_matrix(), np.array([0,1,0]))
        diffvector = np.subtract(np.array(blockPos), np.array(gripperPos))
        dotvec = np.dot(diffvector/np.linalg.norm(diffvector), normalvec/np.linalg.norm(normalvec))

        # Compute contact information
        ftipNormalForce, ftipLateralFriction1, ftipLateralFriction2, number_of_contact_points, ftipContactPoints, totalNormalForce, totalLateralFriction1, totalLateralFriction2 = self._contactinfo()

        r_top = self._r_topology()
        self.r_top = 1 if r_top > 0 else 0

        self.contactpenalize = -1 if totalNormalForce > 0 else 0

        # Compute rewards for different aspects
        self.distanceReward = 1 - math.tanh(closestPoints)
        self.oriReward = 1 - math.tanh(distance.euclidean(np.array(blockOrnEuler),np.array(gripOrnEuler)))
        normalForceReward = 1 - math.tanh(totalNormalForce)
        gripangvelReward = 1 - math.tanh(gripangvel)
        fingerActionReward = 1 - math.tanh(abs(self._action[-1]))

        positionActionReward = np.linalg.norm(self._action[0:3]) / np.sqrt(3)
        orientationActionReward = np.linalg.norm(self._action[3:6]) / np.sqrt(3)

        # Combine rewards to get final reward
        reward = self.distanceReward + self.oriReward + self.r_top + self.contactpenalize 

        return reward

    def _r_topology(self):
        """
        Calculates the ratio of points within a generated point cloud that lie inside the convex hull of the gripper.

        Returns:
        - n_count: The ratio of contained points over total keypoints.
        """

        # Small sphere properties
        red_point_dot_radius = 2
        red_point_dot_color = [1, 0, 0]  # Red color: [r, g, b]
        red_point_dot_opacity = 1.0  # Fully opaque

        # Compute axis-aligned bounding box (AABB) of the block
        aabb_min, aabb_max = p.getAABB(self.blockUid)

        # Generate a grid of points (5 points along each axis) within the AABB
        x = np.linspace(aabb_min[0], aabb_max[0], 5)
        y = np.linspace(aabb_min[1], aabb_max[1], 5)
        z = np.linspace(aabb_min[2], aabb_max[2], 5)
        
        # Store all generated points in a list
        points = []
        for i in x:
            for j in y:
                for k in z:
                    points.append([i, j, k])
        points = np.array(points)

        # Color for each point
        red_point_dot_color = np.array([red_point_dot_color] * len(points))
        
        # Add points to debug visualizer
        # p.addUserDebugPoints(pointPositions=points,
        #                     pointColorsRGB=red_point_dot_color,
        #                     pointSize=red_point_dot_radius)

        # Get the pose of each gripper link
        gripper_link_pose = [p.getLinkState(self._robotiq.robotiq_uid, i)[0] for i in range(self._robotiq.num_joints)]
        gripper_link_pose = np.array(gripper_link_pose)

        # Calculate the convex hull of the gripper link poses
        hull = trimesh.convex.convex_hull(gripper_link_pose)

        # Check which points are inside the hull
        n = hull.contains(points)

        # Calculate the ratio of points inside the hull
        n_count = np.count_nonzero(n) / self._keypoints

        return n_count

    def close(self):
        """
        Disconnects the physics client.
        """
        p.disconnect()
