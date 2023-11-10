# Fanuc Robotiq Grasp

## Introduction
Welcome to the Fanuc Robotiq Grasp project! This repository houses an innovative solution for robotic arm manipulation, specifically designed for the Fanuc Robotiq gripper. Leveraging advanced algorithms and robotics principles, this project aims to enhance the capabilities of robotic arms in complex manipulation tasks.

## Technologies
- **Programming Language:** Python
- **Key Libraries:** [List any specific robotics or machine learning libraries used]

## Installation and Setup
To get started with the Fanuc Robotiq Grasp, follow these steps:

1. **Clone the Repository:**
   ```
   git clone https://github.com/baha2r/Fanuc_Robotiq_Grasp.git
   cd Fanuc_Robotiq_Grasp
   ```

2. **Set Up the Environment:**
   - Ensure Python [version] is installed.
   - Install required packages:
     ```
     pip install -r requirements.txt
     ```

# Observation Space
The observation space for the robotic arm environment is represented by the configuration, Box(-inf, inf, (40,), float32). This space consists of a set of 40 continuous variables, each describing a distinct attribute related to the position, movement, and velocity of both the robotic gripper and its target. These variables embody an extensive range of information about the environment, capturing the dynamism and intricacies involved in the manipulative tasks of the robotic arm. 

The table provided below offers a comprehensive overview of each variable within the observation space. It outlines not only the variable itself, but also the corresponding limits and the unit of measurement used. This range from negative infinity to positive infinity underscores the continuous nature of these variables, further emphasizing the complexity of the tasks and movements this robotic arm is designed to perform.

| Num | Observation                                                  | Min  | Max  | Unit             |
|-----|--------------------------------------------------------------|------|------|------------------|
| 0   | Gripper position in X direction                              | -inf | inf  | Position (m)     |
| 1   | Gripper position in Y direction                              | -inf | inf  | Position (m)     |
| 2   | Gripper position in Z direction                              | -inf | inf  | Position (m)     |
| 3   | Gripper orientation (Roll)                                   | -inf | inf  | Angle (deg)      |
| 4   | Gripper orientation (Pitch)                                  | -inf | inf  | Angle (deg)      |
| 5   | Gripper orientation (Yaw)                                    | -inf | inf  | Angle (deg)      |
| 6   | Linear velocity of gripper in X direction                    | -inf | inf  | Linear_velocity (m/s)|
| 7   | Linear velocity of gripper in Y direction                    | -inf | inf  | Linear_velocity (m/s)|
| 8   | Linear velocity of gripper in Z direction                    | -inf | inf  | Linear_velocity (m/s)|
| 9   | Angular velocity of gripper in Roll                          | -inf | inf  | Angular_velocity (deg/s)|
| 10  | Angular velocity of gripper in Pitch                         | -inf | inf  | Angular_velocity (deg/s)|
| 11  | Angular velocity of gripper in Yaw                           | -inf | inf  | Angular_velocity (deg/s)|
| 12  | Position of target in X direction                            | -inf | inf  | Position (m)     |
| 13  | Position of target in Y direction                            | -inf | inf  | Position (m)     |
| 14  | Position of target in Z direction                            | -inf | inf  | Position (m)     |
| 15  | Relative position between gripper and target in X direction  | -inf | inf  | Position (m)     |
| 16  | Relative position between gripper and target in Y direction  | -inf | inf  | Position (m)     |
| 17  | Relative position between gripper and target in Z direction  | -inf | inf  | Position (m)     |
| 18  | Target orientation (Roll)                                    | -inf | inf  | Angle (deg)      |
| 19  | Target orientation (Pitch)                                   | -inf | inf  | Angle (deg)      |
| 20  | Target orientation (Yaw)                                     | -inf | inf  | Angle (deg)      |
| 21  | Relative orientation between gripper and target (Roll)       | -inf | inf  | Angle (deg)      |
| 22  | Relative orientation between gripper and target (Pitch)      | -inf | inf  | Angle (deg)      |
| 23  | Relative orientation between gripper and target (Yaw)        | -inf | inf  | Angle (deg)      |
| 24  | Target linear velocity in X direction                       | -inf | inf  | Linear_velocity (m/s)|
| 25  | Target linear velocity in Y direction                       | -inf | inf  | Linear_velocity (m/s)|
| 26  | Target linear velocity in Z direction                       | -inf | inf  | Linear_velocity (m/s)|
| 27  | Target angular velocity in Roll                             | -inf | inf  | Angular_velocity (deg/s)|
| 28  | Target angular velocity in Pitch                            | -inf | inf  | Angular_velocity (deg/s)|
| 29  | Target angular velocity in Yaw                              | -inf | inf  | Angular_velocity (deg/s)|
| 30  | Relative linear velocity between gripper and target in X     | -inf | inf  | Linear_velocity (m/s)|
| 31  | Relative linear velocity between gripper and target in Y     | -inf | inf  | Linear_velocity (m/s)|
| 32  | Relative linear velocity between gripper and target in Z     | -inf | inf  | Linear_velocity (m/s)|
| 33  | Relative angular velocity between gripper and target in Roll | -inf | inf  | Angular_velocity (deg/s)|
| 34  | Relative angular velocity between gripper and target in Pitch| -inf | inf  | Angular_velocity (deg/s)|
| 35  | Relative angular velocity between gripper and target in Yaw  | -inf | inf  | Angular_velocity (deg/s)|
| 36  | Closest distance between palm of gripper and target in X     | -inf | inf  | Position (m)     |
| 37  | Closest distance between palm of gripper and target in Y     | -inf | inf  | Position (m)     |
| 38  | Closest distance between palm of gripper and target in Z     | -inf | inf  | Position (m)     |
| 39  | Contact (tactile sensor) information                         | -inf | inf  | Force (N)             |


# Action Space
The action space is defined within a Box(-1.0, 1.0, (6,), float32), which encapsulates the absolute position and orientation of the 3f RobotiQ gripper when functioning as an end-effector. Control actions are enforced by modulating the physical motion of the gripper's base across six degrees of freedom (6dof). This comprises three translational (linear) and three rotational (angular) movements that are executed by the robotic manipulator through inverse kinematics. For compatibility purposes, control action inputs are scaled to a range between -1 and 1. The elements of the action array are as follows:

| Num | Action                                               | Control Min | Control Max | Angle Min  | Angle Max  | Name   | Joint | Unit        |
|-----|------------------------------------------------------|-------------|-------------|------------|------------|--------|-------|-------------|
| 0   | Linear translation of the gripper in x direction     | -1          | 1           | -0.01 (m)  | 0.01 (m)   | dx     | slide | position (m)|
| 1   | Linear translation of the gripper in y direction     | -1          | 1           | -0.01 (m)  | 0.01 (m)   | dy     | slide | position (m)|
| 2   | Linear translation of the gripper in z direction     | -1          | 1           | -0.01 (m)  | 0.01 (m)   | dz     | slide | position (m)|
| 3   | Rotate the arm around the X-axis                     | -1          | 1           | -0.1 (deg) | 0.1 (deg)  | droll  | rotate| angle (deg) |
| 4   | Rotate the arm around the Y-axis                     | -1          | 1           | -0.1 (deg) | 0.1 (deg)  | dpitch | rotate| angle (deg) |
| 5   | Rotate the arm around the Z-axis                     | -1          | 1           | -0.1 (deg) | 0.1 (deg)  | dyaw   | rotate| angle (deg) |

# Rewards
In this work, we introduce a novel reward function that integrates both dense and sparse rewards, aiming to address the challenge of precisely approaching to grasp a moving, floating object. The agent employs the dense reward component to ascertain the appropriate approach towards the target, while simultaneously maintaining its position and orientation. Subsequently, the sparse reward aspect of the reward function provides guidance to maintain an optimal posture, preserve a safe distance between the gripper and the target, and ultimately prevent contact with the target.

# Episode End
The episode will be truncated when the duration reaches a total of max_episode_steps which by default is set to 500 timesteps. The episode is never terminated since the task is continuing with an infinite horizon.



