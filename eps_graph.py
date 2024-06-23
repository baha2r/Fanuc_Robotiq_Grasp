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

def plot_distribution(distribution, zones, direction):
    plt.figure(figsize=(12, 6))
    plt.imshow(distribution.T, aspect='auto', cmap='viridis', origin='lower', extent=[0, distribution.shape[0], zones[0], zones[-1]])
    plt.colorbar(label='Intensity')
    plt.xlabel('Timesteps')
    plt.ylabel('Data Value')
    plt.title(f'Distribution over Timesteps for {direction} direction')
    plt.savefig(f"{direction}_distribution.png")

def calculate_distribution(data, num_zones):
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
gripper_orientations = data['all_gripper_orientations']
block_orientations = data['all_block_orientations']
gripper_quaternions = np.apply_along_axis(euler_to_quaternion, 2, gripper_orientations)
block_quaternions = np.apply_along_axis(euler_to_quaternion, 2, block_orientations)
print(f"shape of gripper_quaternions: {gripper_quaternions.shape}")
quaternion_differences = np.zeros((gripper_orientations.shape[0], gripper_orientations.shape[1], 1))
for i in range(gripper_orientations.shape[0]):
    for j in range(gripper_orientations.shape[1]):
        quaternion_differences[i, j, 0] = quaternion_angle_difference(gripper_quaternions[i, j], block_quaternions[i, j])
print(f"quaternion differences: {quaternion_differences.shape}")
print(f"min: {np.min(quaternion_differences)}, max: {np.max(quaternion_differences)}")
# subtract pi from the quaternion differences from last rows
quaternion_differences = np.abs(quaternion_differences - np.pi)
# change the quaternion differences to degrees
# quaternion_differences = np.rad2deg(quaternion_differences)

closest_points_magnitudes = np.linalg.norm(closest_points, axis=2)
closest_points = np.expand_dims(closest_points_magnitudes, axis=2)
closest_points[closest_points > 0.8] = 0.8
position_actions_magnitudes = np.linalg.norm(position_actions, axis=2)
position_actions = np.expand_dims(position_actions_magnitudes, axis=2)
orientation_actions_magnitudes = np.linalg.norm(orientation_actions, axis=2)
orientation_actions = np.expand_dims(orientation_actions_magnitudes, axis=2)
gripper_linear_velocities_magnitudes = np.linalg.norm(gripper_linear_velocities, axis=2)
gripper_linear_velocities = np.expand_dims(gripper_linear_velocities_magnitudes, axis=2)



distribution, zones = calculate_distribution(position_rewards, num_zones=50)
plot_distribution(distribution, zones, 'Position Rewards')

distribution, zones = calculate_distribution(closest_points, num_zones=25)
plot_distribution(distribution, zones, 'Closest Points')

distribution, zones = calculate_distribution(position_actions, num_zones=50)
plot_distribution(distribution, zones, 'Position Actions')

distribution, zones = calculate_distribution(orientation_actions, num_zones=10)
plot_distribution(distribution, zones, 'Orientation Actions')

distribution, zones = calculate_distribution(gripper_linear_velocities, num_zones=50)
plot_distribution(distribution, zones, 'Gripper Linear Velocities')

distribution, zones = calculate_distribution(quaternion_differences, num_zones=25)
plot_distribution(distribution, zones, 'Quaternion Differences')

print("Plot created successfully!")
