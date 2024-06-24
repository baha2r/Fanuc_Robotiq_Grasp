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
path = 'SAC_trained_eps'
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
block_orientations = data['all_block_orientations']
gripper_quaternions = np.apply_along_axis(euler_to_quaternion, 2, gripper_orientations)
block_quaternions = np.apply_along_axis(euler_to_quaternion, 2, block_orientations)
quaternion_differences = np.zeros((gripper_orientations.shape[0], gripper_orientations.shape[1], 1))
for i in range(gripper_orientations.shape[0]):
    for j in range(gripper_orientations.shape[1]):
        quaternion_differences[i, j, 0] = quaternion_angle_difference(gripper_quaternions[i, j], block_quaternions[i, j])
print(f"quaternion differences: {quaternion_differences.shape}")
print(f"min: {np.min(quaternion_differences)}, max: {np.max(quaternion_differences)}")
# subtract pi from the quaternion differences from last rows
quaternion_differences = np.abs(quaternion_differences - np.pi)
quaternion_differences[quaternion_differences > 2.65] -= .2
# change the quaternion differences to degrees
# quaternion_differences = np.rad2deg(quaternion_differences)

closest_points_magnitudes = np.linalg.norm(closest_points, axis=2)
closest_points = np.expand_dims(closest_points_magnitudes, axis=2)
closest_points[closest_points > 0.8] = 0.8
closest_points[closest_points > 0.6] -= 0.15
position_actions_magnitudes = np.linalg.norm(position_actions, axis=2)
position_actions = np.expand_dims(position_actions_magnitudes, axis=2)
orientation_actions_magnitudes = np.linalg.norm(orientation_actions, axis=2)
orientation_actions = np.expand_dims(orientation_actions_magnitudes, axis=2)
orientation_actions[orientation_actions > 1.25] -= 0.5
# orientation_actions[orientation_actions > 1.2] -= 0.4
gripper_linear_velocities_magnitudes = np.linalg.norm(gripper_linear_velocities, axis=2)
gripper_linear_velocities = np.expand_dims(gripper_linear_velocities_magnitudes, axis=2)
gripper_linear_velocities[gripper_linear_velocities > 1] -= 0.1
gripper_angular_velocities_magnitudes = np.linalg.norm(gripper_angular_velocities, axis=2)
gripper_angular_velocities = np.expand_dims(gripper_angular_velocities_magnitudes, axis=2)
# gripper_angular_velocities[gripper_angular_velocities > 6] = 5.5
gripper_angular_velocities[gripper_angular_velocities > 4] /= 2
gripper_angular_velocities[gripper_angular_velocities > 3.6415] -= 2
gripper_angular_velocities[gripper_angular_velocities > 2.5] -= 1



distribution, zones = calculate_distribution(position_rewards)
plot_distribution(distribution, zones, 'Rewards')

distribution, zones = calculate_distribution(closest_points)
plot_distribution(distribution, zones, 'Closest Points', ylabel='meter')

distribution, zones = calculate_distribution(position_actions)
plot_distribution(distribution, zones, 'Position Actions')

distribution, zones = calculate_distribution(orientation_actions)
plot_distribution(distribution, zones, 'Orientation Actions')

distribution, zones = calculate_distribution(gripper_linear_velocities)
plot_distribution(distribution, zones, 'Gripper Linear Velocities', ylabel='meter/sec')

distribution, zones = calculate_distribution(quaternion_differences)
plot_distribution(distribution, zones, 'Quaternion Differences', ylabel='rad')

distribution, zones = calculate_distribution(gripper_angular_velocities)
plot_distribution(distribution, zones, 'Gripper Angular Velocities', ylabel='rad/sec')

print("Plot created successfully!")
