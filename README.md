# Observation Space
The observation space for the robotic arm environment consists of 12 continuous variables representing various aspects of the robot's position and movement, as well as its velocity. The table below lists each variable in the observation space, along with its corresponding control and angle limits, the name of the variable as it appears in the XML file, the joint that the variable corresponds to, and the unit of measurement.

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
| 39  | Contact (tactile sensor) information                         | -inf | inf  | n/a              |


# Action Space
The action space is defined as a Box(-1.0, 1.0, (6,), float32), encompassing the absolute position and orientation of the 3f RobotiQ gripper. Control actions are applied by adjusting the real movement of the gripper's base in six degrees of freedom (6dof), consisting of three translational and three rotational movements. To ensure compatibility, the input of control actions is scaled to a range between -1 and 1. The elements of the action array are as follows:

| Num | Action                                               | Control Min | Control Max | Angle Min  | Angle Max  | Name   | Joint | Unit        |
|-----|------------------------------------------------------|-------------|-------------|------------|------------|--------|-------|-------------|
| 0   | Linear translation of the gripper in x direction     | -1          | 1           | -0.01 (m)  | 0.01 (m)   | dx     | slide | position (m)|
| 1   | Linear translation of the gripper in y direction     | -1          | 1           | -0.01 (m)  | 0.01 (m)   | dy     | slide | position (m)|
| 2   | Linear translation of the gripper in z direction     | -1          | 1           | -0.01 (m)  | 0.01 (m)   | dz     | slide | position (m)|
| 3   | Rotate the arm around the X-axis                     | -1          | 1           | -0.1 (deg) | 0.1 (deg)  | droll  | rotate| angle (rad) |
| 4   | Rotate the arm around the Y-axis                     | -1          | 1           | -0.1 (deg) | 0.1 (deg)  | dpitch | rotate| angle (rad) |
| 5   | Rotate the arm around the Z-axis                     | -1          | 1           | -0.1 (deg) | 0.1 (deg)  | dyaw   | rotate| angle (rad) |
