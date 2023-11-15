import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import pybullet as p
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from robotiqGymEnv import robotiqGymEnv
import numpy as np
import csv

def load_model(file_path):
    """
    Load the model from the given file path.
    """
    return SAC.load(file_path)

def extract_data(env, model, obs, steps):
    """
    Extract data from the environment.
    """
    data = {
        "position_action": [],
        "angle_action": [],
        "gripper_position": [],
        "gripper_angle": [],
        "gripper_velocity": [],
        "gripper_angular_velocity": [],
        "target_position": [],
        "target_angle": [],
        "target_velocity": [],
        "target_angular_velocity": [],
        "closest_point": [],
        "contact_force": [],
        "rewards": [],
        "finger1_angles": [],
        "finger2_angles": [],
        "finger3_angles": [],
        "finger1_min_dists": [],
        "finger2_min_dists": [],
        "finger3_min_dists": [],
        "num_contact_points": [],
        "fingertip_num_contact_points": [],
        "joint_force": [],
        "accumulated_contact_force": [],
        "totalLateralFrictionForce": [],
        "finger1_link1_world_angles": [],
        "finger1_link2_world_angles": [],
        "finger1_link3_world_angles": [],
    }

    done = False
    first_time = True
    joint_force = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # action = [0,0,0,0,0,0]
        if (env._contactinfo()[4] > 0).all():
            pass
            # action[1] = -0.05
        obs, rewards, done, info = env.step(action)
        
        if rewards > 2:
            if first_time:
                for i in range(env._robotiq.num_joints):
                    #disable default constraint-based motors
                    p.setJointMotorControl2(env._robotiq.robotiq_uid, i, p.POSITION_CONTROL, targetPosition=0, force=0)
                first_time = False
                joint1_force = 0.001
                joint3_force = 0.0001
            if (env._contactinfo()[4] > 0).all() and joint_force < 0.1:
                # joint_force = joint_force + 0.01
                pass
            p.setJointMotorControlArray(env._robotiq.robotiq_uid, [1, 5, 9], p.TORQUE_CONTROL, forces=[joint1_force, joint1_force, joint1_force], positionGains=[0.03, 0.03, 0.03], velocityGains=[1, 1, 1])
            p.setJointMotorControlArray(env._robotiq.robotiq_uid, [2, 6, 10], p.POSITION_CONTROL, targetPositions=[0, 0, 0])
            p.setJointMotorControlArray(env._robotiq.robotiq_uid, [3, 7, 11], p.POSITION_CONTROL, targetPositions=[-0.0, -0.0, -0.0])

            # j1_angle = p.getJointState(env._robotiq.robotiq_uid, 1)[0]
            # j1_angle = j1_angle + 0.1
            
            # if (env._contactinfo()[4] > 0).all():
            #     p.setJointMotorControlArray(env._robotiq.robotiq_uid, [3, 7, 11], p.TORQUE_CONTROL, forces=[-0.001, -0.001, -0.001])#, positionGains=[0.03, 0.03, 0.03], velocityGains=[1, 1, 1])
            # else:
            # p.setJointMotorControlArray(env._robotiq.robotiq_uid, [3, 7, 11], p.POSITION_CONTROL, targetPositions=[-j1_angle, -j1_angle, -j1_angle])
            # if env._contactinfo()[3] > 0:
            #     p.setJointMotorControlArray(env._robotiq.robotiq_uid, [3, 7, 11], p.TORQUE_CONTROL, forces=[joint_force, joint_force, joint_force])#, positionGains=[0.03, 0.03, 0.03], velocityGains=[1, 1, 1])
        
        finger1_link1_world_angles = np.array(p.getEulerFromQuaternion(p.getLinkState(env._robotiq.robotiq_uid, 2)[1]))
        finger1_link2_world_angles = np.array(p.getEulerFromQuaternion(p.getLinkState(env._robotiq.robotiq_uid, 3)[1]))
        finger1_link3_world_angles = np.array(p.getEulerFromQuaternion(p.getLinkState(env._robotiq.robotiq_uid, 4)[1]))
        
        base_pos, base_orientation = p.getBasePositionAndOrientation(env._robotiq.robotiq_uid)
        target_pos, target_orientation = p.getBasePositionAndOrientation(env.blockUid)
        base_pos = np.array(base_pos)
        target_pos = np.array(target_pos)

        base_velocity, base_angular_velocity = p.getBaseVelocity(env._robotiq.robotiq_uid)
        target_velocity, target_angular_velocity = p.getBaseVelocity(env.blockUid)
        base_velocity = np.array(base_velocity)
        target_velocity = np.array(target_velocity)
        base_angular_velocity = np.array(base_angular_velocity)
        target_angular_velocity = np.array(target_angular_velocity)

        gripper_angle = p.getEulerFromQuaternion(base_orientation)
        target_angle = p.getEulerFromQuaternion(target_orientation)
        gripper_angle = np.array(gripper_angle)
        target_angle = np.array(target_angle)

        joint_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11]
        joint_states = p.getJointStates(env._robotiq.robotiq_uid, joint_indices)
        
        finger1_angle = np.array([np.float64(joint_states[0][0]), np.float64(joint_states[1][0]), np.float64(joint_states[2][0])])
        finger2_angle = np.array([np.float64(joint_states[3][0]), np.float64(joint_states[4][0]), np.float64(joint_states[5][0])])
        finger3_angle = np.array([np.float64(joint_states[6][0]), np.float64(joint_states[7][0]), np.float64(joint_states[8][0])])
        
        data["finger1_angles"].append(finger1_angle)
        data["finger2_angles"].append(finger2_angle)
        data["finger3_angles"].append(finger3_angle)
                
        dist_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11]
        min_dists = [p.getClosestPoints(env.blockUid, env._robotiq.robotiq_uid, 10, -1, i)[0][8] for i in dist_indices]
        min_dists = np.array(min_dists)
        
        
        data["finger1_min_dists"].append(min_dists[0:3])
        data["finger2_min_dists"].append(min_dists[3:6])
        data["finger3_min_dists"].append(min_dists[6:9])
        
        data["position_action"].append(action[0:3])
        data["angle_action"].append(action[3:6])
        data["gripper_position"].append(base_pos)
        data["gripper_angle"].append(gripper_angle)
        data["gripper_velocity"].append(base_velocity)
        data["gripper_angular_velocity"].append(base_angular_velocity)
        data["target_position"].append(target_pos)
        data["target_angle"].append(target_angle)
        data["target_velocity"].append(target_velocity)
        data["target_angular_velocity"].append(target_angular_velocity)
        data["closest_point"].append(p.getClosestPoints(env._robotiq.robotiq_uid, env.blockUid, 10, -1, -1)[0][8])
        data["contact_force"].append(env._contactinfo()[5])
        data["num_contact_points"].append(env._contactinfo()[3])
        data["fingertip_num_contact_points"].append(env._contactinfo()[4])
        data["rewards"].append(rewards)
        data["joint_force"].append(joint_force)
        data["accumulated_contact_force"].append(env._accumulated_contact_force)
        data["totalLateralFrictionForce"].append(env._contactinfo()[2])
        data["finger1_link1_world_angles"].append(finger1_link1_world_angles)
        data["finger1_link2_world_angles"].append(finger1_link2_world_angles)
        data["finger1_link3_world_angles"].append(finger1_link3_world_angles)
        
        env.render()
        
    return data

def plot_data(data, labels):
    """
    Plot the data using matplotlib.
    """
    plt.figure()
    for d, label in zip(data, labels):
        plt.plot(d, label=label)
        plt.legend()

def plot_combined_data(data_gripper, data_target, labels, upper_bound=None, lower_bound=None):
    """
    Plot gripper and target data on the same plot with the same color for each direction.
    Gripper data will be solid lines, target data will be dashed lines.
    """
    plt.figure()  # You can adjust the size if you want

    colors = ['b', 'g', 'r']  # blue, green, red for x, y, z respectively
    line_styles = ['-', '--']  # solid for gripper, dashed for target

    for i, (d_gripper, d_target, label) in enumerate(zip(data_gripper, data_target, labels)):
        color = colors[i % len(colors)]
        plt.plot(d_gripper, line_styles[0], color=color, label=f'Gripper {label}')
        plt.plot(d_target, line_styles[1], color=color, label=f'Target {label}')
        if upper_bound is not None and lower_bound is not None:
            plt.ylim(lower_bound, upper_bound)
        plt.legend()

def save_to_csv(filename, data):
    """
    Save data to CSV.
    """
    # Find the maximum length among all lists
    max_length = max(len(v) for v in data.values())

    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        headers = list(data.keys())
        csv_writer.writerow(headers)

        # Write rows
        for i in range(max_length):
            row = [data[header][i] if i < len(data[header]) else "" for header in headers]
            csv_writer.writerow(row)

def main():
    """
    Main function to run the program.
    """
    model_file = "models/20230316-03:42PM_SAC/best_model.zip"
    model = load_model(model_file)

    with robotiqGymEnv(records=True, renders=True) as env:
        obs = env.reset()
        steps = range(500)  # Define the number of steps here
        data = extract_data(env, model, obs, steps)

    # Plotting data
    plot_data(zip(*data["position_action"]), ["x action", "y action", "z action"])
    plot_data(zip(*data["angle_action"]), ["roll action", "pitch action", "yaw action"])
    plot_combined_data(zip(*data["gripper_position"]),zip(*data["target_position"]),["x", "y", "z"])
    plot_data(zip(*data["gripper_angle"]), ["x angle", "y angle", "z angle"])
    plot_combined_data(zip(*data["gripper_velocity"]),zip(*data["target_velocity"]),["x", "y", "z"], upper_bound=0.5, lower_bound=-0.5)
    plot_data([data["closest_point"]], ["closest distance"])
    plot_data([data["rewards"]], ["rewards"])
    plot_data([data["contact_force"]], ["contact force"])
    plot_data([data["accumulated_contact_force"]], ["accumulated contact force"])
    plot_data(zip(*data["finger1_angles"]), ["angle_1", "angle_2", "angle_3"])
    plot_data(zip(*data["finger2_angles"]), ["angle_1", "angle_2", "angle_3"])
    plot_data(zip(*data["finger3_angles"]), ["angle_1", "angle_2", "angle_3"])
    plot_data([data["num_contact_points"]], ["num_contact_points"])
    plot_data(zip(*data["fingertip_num_contact_points"]), ["fingertip_1", "fingertip_2", "fingertip_3"])
    plot_data([data["joint_force"]], ["joint_force"])
    plot_data([data["totalLateralFrictionForce"]], ["totalLateralFrictionForce"])
    plot_data(zip(*data["finger1_link1_world_angles"]), ["finger1_link1_world_roll", "finger1_link1_world_pitch", "finger1_link1_world_yaw"])
    plot_data(zip(*data["finger1_link2_world_angles"]), ["finger1_link2_world_roll", "finger1_link2_world_pitch", "finger1_link2_world_yaw"])
    plot_data(zip(*data["finger1_link3_world_angles"]), ["finger1_link3_world_roll", "finger1_link3_world_pitch", "finger1_link3_world_yaw"])
    # plot_data(zip(*data["finger2_min_dists"]), ["finger2_min_dist_1", "finger2_min_dist_2", "finger2_min_dist_3"])

    # Saving data to CSV
    save_to_csv("output_data.csv", data)


if __name__ == "__main__":
    main()
