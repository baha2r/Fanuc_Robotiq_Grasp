import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
"""
collections['all_step_counters'].append(step_counter)
collections['all_position_actions'].append(position_action)
collections['all_orientation_actions'].append(orientation_action)
collections['all_gripper_positions'].append(gripper_position)
collections['all_gripper_orientations'].append(gripper_orientation)
collections['all_gripper_linear_velocities'].append(gripper_linear_velocity)
collections['all_gripper_angular_velocities'].append(gripper_angular_velocity)
collections['all_block_positions'].append(block_position)
collections['all_block_orientations'].append(block_orientation)
collections['all_block_linear_velocities'].append(block_linear_velocity)
collections['all_block_angular_velocities'].append(block_angular_velocity)
collections['all_closest_points'].append(closest_points)
collections['all_positioning_rewards'].append(positioning_rewards)
collections['all_is_reach'].append(is_reach)
"""

def euler_to_quaternion(angles):
    """Convert Euler angles to quaternion."""
    return R.from_euler('xyz', angles).as_quat()

def quaternion_angle_difference(q1, q2):
    q_diff = R.from_quat(q1).inv() * R.from_quat(q2)
    angle = 2 * np.arccos(np.clip(q_diff.as_quat()[0], -1.0, 1.0))
    return angle

def quaternion_difference_norm(q1, q2):
    q_diff = R.from_quat(q1).inv() * R.from_quat(q2)
    return np.linalg.norm(q_diff.as_quat())

def plot_distribution(distribution, zones, direction, ylabel='Data Value'):
    plt.figure(figsize=(12, 6))
    plt.imshow(distribution.T, aspect='auto', origin='lower', cmap='viridis' , extent=[0, distribution.shape[0], zones[0], zones[-1]])# cmap='gray',  
    plt.colorbar()
    plt.xlabel('Timesteps', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(f'{direction} Distribution', fontsize=20)
    plt.savefig(f"{direction}_distribution.png")

def calculate_distribution(data, num_zones=100):
    min_val = np.min(data)
    max_val = np.max(data)
    zones = np.linspace(min_val, max_val, num_zones + 1)
    distribution = np.zeros((data.shape[1], num_zones))
    
    for t in range(data.shape[1]):
        for zone in range(num_zones):
            distribution[t, zone] = np.sum((data[:, t] >= zones[zone]) & (data[:, t] < zones[zone + 1]))
    
    return distribution, zones

# Load the saved data

path = '/home/baha/pybullet/robotiqGymnasiumApproach/'
data = np.load(os.path.join(path, 'approach_data.npz'))

# Extract the grasp_rewards data
step_counters = data['all_step_counters']
position_rewards = data['all_positioning_rewards']
closest_points = data['all_closest_points']
position_actions = data['all_position_actions']
orientation_actions = data['all_orientation_actions']
gripper_linear_velocities = data['all_gripper_linear_velocities']
gripper_angular_velocities = data['all_gripper_angular_velocities']
gripper_orientations = data['all_gripper_orientations']
block_positions = data['all_block_positions']
block_orientations = data['all_block_orientations']
block_linear_velocities = data['all_block_linear_velocities']
block_angular_velocities = data['all_block_angular_velocities']
# is_reach = data['all_is_reach']
print(f"size block orientations: {block_orientations.shape}")

# plot the position rewards
plt.figure(figsize=(12, 6))
# plt.plot(step_counters[0,:,0], block_orientations[0,:,0,0], label='x')
# plt.plot(step_counters[0,:,0], block_orientations[0,:,0,1], label='y')
# plt.plot(step_counters[0,:,0], block_orientations[0,:,0,2], label='z')
plt.plot(step_counters[0,:,0], gripper_orientations[0,:,0], label='x')
plt.plot(step_counters[0,:,0], gripper_orientations[0,:,1], label='y')
plt.plot(step_counters[0,:,0], gripper_orientations[0,:,2], label='z')
plt.legend()
plt.savefig('ep_position_rewards.png')


print("Plot created successfully!")
