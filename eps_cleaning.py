import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm  # For progress tracking

# Ignore all warnings
warnings.filterwarnings("ignore")


def extract_data(data_list, key):
    return [item[key] for item in data_list if key in item]

def plot_distribution(distribution, zones, direction):
    plt.figure(figsize=(12, 6))
    plt.imshow(distribution.T, aspect='auto', cmap='viridis', origin='lower', extent=[0, distribution.shape[0], zones[0], zones[-1]])
    plt.colorbar(label='Intensity')
    plt.xlabel('Timesteps')
    plt.ylabel('Data Value')
    plt.title(f'Distribution over Timesteps for {direction} direction')

def calculate_distribution(data, num_zones):
    min_val = np.min(data)
    max_val = np.max(data)
    zones = np.linspace(min_val, max_val, num_zones + 1)
    distribution = np.zeros((data.shape[1], num_zones))

    for t in range(data.shape[1]):
        for zone in range(num_zones):
            distribution[t, zone] = np.sum((data[:, t] >= zones[zone]) & (data[:, t] < zones[zone + 1]))

    return distribution, zones

def process_contact_data(data):
    """
    Process the contact data from the dataframe and compute the grasp matrix and quality metrics for each timestep.

    Parameters:
    data (list): The list containing the contact data.

    Returns:
    dict: A dictionary with timesteps as keys and grasp matrices as values.
    """
    df = pd.DataFrame(data)
    grasp_matrix = {}
    grasp_quality_metrics = {}

    for index, row in df.iloc[5:].iterrows():
        timestep = row['stepcounter']
        contact_points = row['contact']

        if len(contact_points) > 0:
            grasp_matrix_ = compute_grasp_wrench_matrix(contact_points)
            grasp_matrix[int(timestep)] = grasp_matrix_
            quality_metrics = compute_quality_metrics(grasp_matrix_)
            grasp_quality_metrics[int(timestep)] = quality_metrics
        else:
            grasp_matrix[int(timestep)] = None  # No contact points available
            grasp_quality_metrics[int(timestep)] = {
                'ferrari_canny_metric': 0,
                'isotropy_index': float('inf'),
                'largest_min_eigenvalue': 0,
                'gws_volume': 0,
                'force_closure': False
            }  # No contact points available

    return grasp_matrix, grasp_quality_metrics

def compute_grasp_wrench_matrix(contact_points):
    """
    Compute the grasp wrench matrix for the given contact points.

    Parameters:
    contact_points (list): A list of contact points dictionaries.

    Returns:
    numpy.ndarray: The grasp wrench matrix.
    """
    link_groups = {}

    # Group contact points by link index B
    for contact in contact_points:
        link_index_b = contact['linkIndexB']
        if link_index_b not in link_groups:
            link_groups[link_index_b] = []
        link_groups[link_index_b].append(contact)

    wrench_matrix = []

    for link_index_b, contacts in link_groups.items():
        num_contacts = len(contacts)

        if num_contacts == 1:
            contact = contacts[0]
            normal_force = np.array(contact['normalForce'])
            position_on_a = np.array(contact['positionOnAInWS'])  # Convert to numpy array
            normal_direction = np.array(contact['contactNormalOnBInWS'])  # Convert to numpy array
        else:
            # Compute the mean normal force and the centroid of positions on A
            normal_forces = np.array([contact['normalForce'] for contact in contacts])
            positions_on_a = np.array([contact['positionOnAInWS'] for contact in contacts])
            normal_directions = np.array([contact['contactNormalOnBInWS'] for contact in contacts])

            mean_normal_force = np.mean(normal_forces, axis=0)
            centroid_position_on_a = np.mean(positions_on_a, axis=0)
            mean_normal_direction = np.mean(normal_directions, axis=0)

            normal_force = mean_normal_force
            position_on_a = centroid_position_on_a
            normal_direction = mean_normal_direction

        # Compute the wrench for this contact
        force = normal_force * normal_direction  # Element-wise multiplication with numpy arrays
        forces = np.array([
            [force[0], 0, 0],  # Force in x-direction
            [0, force[1], 0],  # Force in y-direction
            [0, 0, force[2]]   # Force in z-direction
        ])
        for force in forces:
            torque = np.cross(position_on_a, force)
            wrench = np.hstack((force, torque))
            wrench_matrix.append(wrench)

    # Convert wrench matrix list to numpy array
    wrench_matrix = np.array(wrench_matrix).T  # Transpose to get the correct shape

    return wrench_matrix

def compute_quality_metrics(wrench_matrix):
    """
    Compute the grasp quality metrics for the given wrench matrix.

    Parameters:
    wrench_matrix (numpy.ndarray): The grasp wrench matrix.

    Returns:
    dict: A dictionary with quality metrics.
    """
    if wrench_matrix.shape[0] == 0:
        return {
            'ferrari_canny_metric': 0,
            'isotropy_index': float('inf'),
            'largest_min_eigenvalue': 0,
            'gws_volume': 0,
            'force_closure': False
        }

    try:
        # Calculate singular values
        singular_values = np.linalg.svd(wrench_matrix, compute_uv=False)

        # Ferrari-Canny Metric (minimum singular value)
        ferrari_canny_metric = np.min(singular_values)

        # Grasp Isotropy Index (ratio of the largest to the smallest singular value)
        isotropy_index = np.max(singular_values) / np.min(singular_values)

        # Largest Minimum Eigenvalue (smallest singular value)
        largest_min_eigenvalue = np.min(singular_values)

        # Grasp Wrench Space Volume (approximate by product of singular values)
        gws_volume = np.prod(singular_values)

        # Force Closure (check if matrix has full rank, i.e., rank 6 for 3D)
        rank_G = np.linalg.matrix_rank(wrench_matrix)
        force_closure = (rank_G == 6)
    except:
        return {
            'ferrari_canny_metric': 0,
            'isotropy_index': float('inf'),
            'largest_min_eigenvalue': 0,
            'gws_volume': 0,
            'force_closure': False
        }

    return {
        'ferrari_canny_metric': ferrari_canny_metric,
        'isotropy_index': isotropy_index,
        'largest_min_eigenvalue': largest_min_eigenvalue,
        'gws_volume': gws_volume,
        'force_closure': force_closure
    }

def save_data(path, **arrays):
    """Save multiple numpy arrays to a .npz file."""
    np.savez(os.path.join(path, 'approach_data.npz'), **arrays)
    print("Data saved successfully!")

def load_and_process_files(path, num_files=100):
    """
    Load data from pickle files, process the data, and collect it into lists.

    Parameters:
    path (str): The directory containing the pickle files.
    num_files (int): The number of files to process.

    Returns:
    dict: A dictionary containing all collected data arrays.
    """
    # Initialize lists to collect data
    data_collections = {
        'all_step_counters': [],
        'all_position_actions': [],
        'all_orientation_actions': [],
        'all_gripper_positions': [],
        'all_gripper_orientations': [],
        'all_gripper_linear_velocities': [],
        'all_gripper_angular_velocities': [],
        'all_block_positions': [],
        'all_block_orientations': [],
        'all_block_linear_velocities': [],
        'all_block_angular_velocities': [],
        'all_closest_points': [],
        'all_positioning_rewards': [],
        'all_is_reach': []
    }

    for i in tqdm(range(num_files), desc="Processing files"):
        file_path = os.path.join(path, f'test{i}.pkl')
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                process_file_data(data, data_collections)
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Convert lists to numpy arrays
    for key in data_collections:
        data_collections[key] = np.array(data_collections[key])

    return data_collections

def process_file_data(data, collections):
    """
    Process individual file data and update the collections.

    Parameters:
    data (dict): The loaded data from a file.
    collections (dict): The dictionary of collections to update.
    """
    step_counter = [data[timestep]['stepcounter'] for timestep in range(5, len(data))]
    position_action = [data[timestep]['position_action'] for timestep in range(5, len(data))]
    orientation_action = [data[timestep]['orientation_action'] for timestep in range(5, len(data))]
    gripper_position = [data[timestep]['gripper_position'] for timestep in range(5, len(data))]
    gripper_orientation = [data[timestep]['gripper_orientation'] for timestep in range(5, len(data))]
    gripper_linear_velocity = [data[timestep]['gripper_linear_velocity'] for timestep in range(5, len(data))]
    gripper_angular_velocity = [data[timestep]['gripper_angular_velocity'] for timestep in range(5, len(data))]
    block_position = [data[timestep]['block_position'] for timestep in range(5, len(data))]
    block_orientation = [data[timestep]['block_orientation'] for timestep in range(5, len(data))]
    block_linear_velocity = [data[timestep]['block_linear_velocity'] for timestep in range(5, len(data))]
    block_angular_velocity = [data[timestep]['block_angular_velocity'] for timestep in range(5, len(data))]
    closest_points = [data[timestep]['closest_points'] for timestep in range(5, len(data))]
    positioning_rewards = [data[timestep]['positioning_reward'] for timestep in range(5, len(data))]
    is_reach = [data[timestep]['is_reach'] for timestep in range(5, len(data))]

    

    # Append to collections
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


if __name__ == "__main__":
    path = 'SAC_trained_eps'
    data_collections = load_and_process_files(path, num_files=500)
    save_data(path, **data_collections)
