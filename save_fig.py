import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# Define the directory to save plots
saved_dir = "test_data/test10"
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

def plot_columns(data, plot_name, labels):
    # Check if the number of labels matches the number of columns in the data
    print(data.shape)
    if data.shape[1] != len(labels):
        raise ValueError("The number of labels must match the number of columns in the data.")

    # Create a plot for each column
    for i in range(data.shape[1]):
        plt.plot(data[:, i], label=labels[i])
    
    # Add legend to the plot
    plt.legend()
    
    # Add labels and title to the plot
    plt.xlabel('timesteps')
    # plt.ylabel('Value')
    plt.title(plot_name)

    plt.savefig(f"{saved_dir}/{plot_name}.png")
    plt.close()

def plot_combined_data(data_gripper, data_target, plot_name, labels, upper_bound=None, lower_bound=None):
    """
    Plot gripper and target data on the same plot with the same color for each direction.
    Gripper data will be solid lines, target data will be dashed lines.
    """
    colors = ['b','g', 'r']  # blue, green, red for x, y, z respectively
    line_styles = ['-', '--']  # solid for gripper, dashed for target
    data_gripper = data_gripper.T
    data_target = data_target.T

    plt.figure()
    for i, (d_gripper, d_target, label) in enumerate(zip(data_gripper, data_target, labels)):
        color = colors[i % len(colors)]
        plt.plot(d_gripper, line_styles[0], color=color, label=f'Gripper {label}')
        plt.plot(d_target, line_styles[1], color=color, label=f'Target {label}')
        if upper_bound is not None and lower_bound is not None:
            plt.ylim(lower_bound, upper_bound)
    
    plt.legend()
    plt.xlabel("timestep")
    plt.title(plot_name)
    plt.savefig(f"{saved_dir}/{plot_name}.png")
        # plt.close()

def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def extract_contact_info(contact_points):
    extracted_info = []
    for contact in contact_points:
        position_on_b = contact['positionOnBInWS']
        normal_force = contact['normalForce']
        normal_direction = contact['contactNormalOnBInWS']
        extracted_info.append({
            'position_on_b': position_on_b,
            'normal_force': normal_force,
            'normal_direction': normal_direction
        })
    return extracted_info

def extract_data(data_list, key):
    return [item[key] for item in data_list if key in item]

def compute_grasp_matrix(contact_points):
    """
    Compute the grasp matrix G for the given contact points.
    
    Parameters:
    contact_points (list): A list of contact points dictionaries.
    
    Returns:
    numpy.ndarray: The grasp matrix G.
    """
    link_groups = {}
    
    # Group contact points by link index B
    for contact in contact_points:
        link_index_b = contact['linkIndexB']
        if link_index_b not in link_groups:
            link_groups[link_index_b] = []
        link_groups[link_index_b].append(contact)
    
    grasp_matrix = []

    for link_index_b, contacts in link_groups.items():
        num_contacts = len(contacts)
        
        if num_contacts == 1:
            contact = contacts[0]
            normal_force = contact['normalForce']
            position_on_a = contact['positionOnAInWS']
            normal_direction = contact['contactNormalOnBInWS']
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

        # Create the grasp matrix row for this link index
        grasp_row = np.hstack(([link_index_b],position_on_a, normal_direction, normal_force))
        grasp_matrix.append(grasp_row)
    
    # Convert grasp matrix list to numpy array
    grasp_matrix = np.array(grasp_matrix)
    
    return grasp_matrix

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

def process_contact_data(df):
    """
    Process the contact data from the dataframe and compute the grasp matrix and quality metrics for each timestep.
    
    Parameters:
    df (pandas.DataFrame): The dataframe containing the contact data.
    
    Returns:
    dict: A dictionary with timesteps as keys and grasp matrices as values.
    """
    grasp_matrix = {}
    grasp_quality_metrics = {}
    
    for index, row in df.iterrows():
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

def save_quality_graphs(quality_metrics, timesteps, filename):
    """
    Save the graphs for each of the numeric quality values.
    
    Parameters:
    quality_metrics (list): A list of dictionaries containing the quality metrics for each timestep.
    timesteps (list): A list of timesteps.
    filename (str): The base filename for saving the graphs.
    """
    metrics_names = ['ferrari_canny_metric', 'isotropy_index', 'largest_min_eigenvalue', 'gws_volume']
    for metric in metrics_names:
        values = [qm[metric] for qm in quality_metrics]
        plt.figure()
        plt.plot(timesteps, values) #marker='o')
        plt.title(f'{metric} over time')
        plt.xlabel('Timestep')
        plt.ylabel(metric)
        plt.grid(True)
        plt.savefig(f'{saved_dir}/{filename}_{metric}.png')
        plt.close()

def main():
    filepath = "data.pkl"  # Adjust this to your pickle file's location
    data = load_data(filepath)
    df = pd.DataFrame(data)
    # data contains the following keys: columns = ['stepcounter', 'position_action', 'orientation_action', 'gripper_position', 'gripper_orientation', 
        # 'gripper_linear_velocity', 'gripper_angular_velocity', 'block_position', 'block_orientation', 'block_linear_velocity',
        # 'block_angular_velocity', 'closest_points', 'positioning_reward', 'is_reach']
    stepcounter = np.array(extract_data(data, 'stepcounter'))
    grasp_stepcounter = np.array(extract_data(data, 'grasp_stepcounter'))
    position_action = np.array(extract_data(data, 'position_action'))
    orientation_action = np.array(extract_data(data, 'orientation_action'))
    gripper_positions = np.array(extract_data(data, 'gripper_position'))
    gripper_orientations = np.array(extract_data(data, 'gripper_orientation'))
    gripper_linear_velocity = np.array(extract_data(data, 'gripper_linear_velocity'))
    gripper_angular_velocity = np.array(extract_data(data, 'gripper_angular_velocity'))
    block_positions = np.array(extract_data(data, 'block_position'))
    block_orientations = np.array(extract_data(data, 'block_orientation'))
    block_linear_velocity = np.array(extract_data(data, 'block_linear_velocity'))
    block_angular_velocity = np.array(extract_data(data, 'block_angular_velocity'))
    closest_points = np.array(extract_data(data, 'closest_points'))
    positioning_reward = np.array(extract_data(data, 'positioning_reward'))
    is_reach = np.array(extract_data(data, 'is_reach'))

    # print(f"length of stepcounter: {len(stepcounter)}")

    # grasp_matrices, grasp_quality_metrics = process_contact_data(df)

    # # Extract timesteps and quality metrics
    # timesteps = list(grasp_quality_metrics.keys())
    # quality_metrics = list(grasp_quality_metrics.values())

    # # Save the quality graphs
    # save_quality_graphs(quality_metrics, timesteps, 'quality_metrics')
    # for timestep, metrics in grasp_quality_metrics.items():
    #     print(f"Quality metrics at timestep {timestep}:")
    #     print(metrics)
    #     print()  # Add a blank line for readability
    # for timestep, grasp_matrix in grasp_matrices.items():
    #     if grasp_matrix is not None:
    #         print(f"Grasp matrix at timestep {timestep}:")
    #         print(f"size: {grasp_matrix.shape}")
    #         print(grasp_matrix)
    #         print()  # Add a blank line for readability
    #     else:
    #         print(f"No contact points available at timestep {timestep}")


    # for index, row in df.iterrows():
    #     timestep = row['stepcounter']
    #     contact_points = row['contact']
        
    #     # Count the number of contact points
    #     num_contact_points = len(contact_points)
                
    #     # Extract information from each contact point if available
    #     if num_contact_points > 0:
    #         for contact in contact_points:
    #             position_on_a = contact['positionOnAInWS']
    #             normal_force = contact['normalForce']
    #             normal_direction = contact['contactNormalOnBInWS']
    #             print(f"Contact point at timestep {timestep}:")
    #             print(f"  Position on A: {position_on_a}")
    #             print(f"  Link Index on B: {contact['linkIndexB']}")
    #             print(f"  Normal Force: {normal_force}")
    #             print(f"  Normal Direction: {normal_direction}")
    #             print()  # Add a blank line for readability
    #     else:
    #         print("  No contact points available")
        
    plot_columns(position_action,'position_action', ["x action", "y action", "z action"])
    plot_columns(orientation_action,'orientation_action', ["x action", "y action", "z action"])
    plot_columns(positioning_reward,'positioning_reward', ["reward"])
    plot_combined_data(gripper_positions, block_positions, 'Positions', ["x", "y", "z"])
    plot_combined_data(gripper_orientations, block_orientations, 'Orientations', ["x", "y", "z", "w"])
    # remove the first row of gripper_linear_velocity and block_linear_velocity
    gripper_linear_velocity = gripper_linear_velocity[3:]
    block_linear_velocity = block_linear_velocity[3:]
    plot_combined_data(gripper_linear_velocity, block_linear_velocity, 'Linear Velocity', ["x", "y", "z"])

if __name__ == "__main__":
    main()
