import os
import inspect
import pybullet as p
import numpy as np
import math
import pybullet_data
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

class robotiq:
    '''
    A class to simulate Robotiq gripper using PyBullet
    '''
    def __init__(self, urdf_root_path="urdf/", time_step=1. / 240.):
        '''
        Initialize the Robotiq gripper
        '''
        self.pybullet_urdf_root_path = pybullet_data.getDataPath()
        self.urdf_root_path = urdf_root_path
        self.time_step = time_step
        self.max_velocity = .35
        self.max_force = 200.
        self.first_joint_idx = [1, 5, 9]
        self.second_joint_idx = [2, 6, 10]
        self.third_joint_idx = [3, 7, 11]
        self.use_inverse_kinematics = 0
        self.use_simulation = 1
        self.reset()
        self.get_joint_state()
        self.get_fingers_convex_hull()

    def reset(self):
        '''
        Reset the Robotiq gripper
        '''
        # radomize the position of the gripper
        start_position = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.1), np.random.uniform(0.5, 1.5)]
        start_position = [0, 0, 1]
        # radomize the orientation of the gripper
        start_orientation = p.getQuaternionFromEuler([np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)])
        start_orientation = p.getQuaternionFromEuler([0, -np.pi/2, 0])
        self.robotiq_uid = p.loadURDF(os.path.join(self.urdf_root_path, "robotiq.urdf"), 
                                      basePosition = start_position, 
                                      baseOrientation = start_orientation,
                                      flags=p.URDF_INITIALIZE_SAT_FEATURES
                                      )
        # p.changeDynamics(self.robotiq_uid, -1, mass=100)
        self.constraint_id = p.createConstraint(self.robotiq_uid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], start_position)
        self.num_joints = p.getNumJoints(self.robotiq_uid)

    def get_joint_state(self):
        '''
        Get the state of the joints
        '''
        self.joint_state = []
        self.joint_positions = self.num_joints * [0.0]
        self.motor_names = []
        self.motor_indices = []
        self.link_positions = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robotiq_uid, i)
            joint_state = p.getJointState(self.robotiq_uid, i)
            self.motor_names.append(str(joint_info[1]))
            self.link_positions.append(p.getLinkState(self.robotiq_uid, i)[0])
            self.motor_indices.append(i)
            self.joint_state.append(joint_state[0])

    def get_observation(self):
        '''
        Get the observations of the gripper's state
        '''
        pos , orn = p.getBasePositionAndOrientation(self.robotiq_uid)
        orn_euler = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = np.array(p.getBaseVelocity(self.robotiq_uid))

        observation = np.array([pos[0], pos[1], pos[2],
                                orn_euler[0], orn_euler[1], orn_euler[2],
                                lin_vel[0], lin_vel[1], lin_vel[2],
                                ang_vel[0], ang_vel[1], ang_vel[2]], dtype=np.float32)                
        return observation

    def get_fingers_convex_hull(self):
        '''
        Get the convex hull of the fingers
        '''
        points = np.array(p.getBasePositionAndOrientation(self.robotiq_uid)[0])
        for i in range(self.num_joints):
            link_world_position = p.getLinkState(self.robotiq_uid , i)[0]
            points = np.vstack([points, np.array(link_world_position)])
        self.finger_volume = ConvexHull(points=points)
        return self.finger_volume

    def apply_action(self, base_commands):
        '''
        Apply actions to the base of the gripper
        '''
        pos , orn = p.getBasePositionAndOrientation(self.robotiq_uid)
        pos = np.array(pos)
        orn = np.array(p.getEulerFromQuaternion(orn))
        pos_command = pos + np.array(base_commands[0:3]) * 0.01
        orn_command = orn + np.array(base_commands[3:6]) * 0.1
        orn_command = p.getQuaternionFromEuler(orn_command)
        p.changeConstraint(self.constraint_id, jointChildPivot =  pos_command, jointChildFrameOrientation = orn_command, maxForce=500)
