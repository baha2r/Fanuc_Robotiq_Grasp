Observation Space
The observation space for the robotic arm environment consists of 12 continuous variables representing various aspects of the robot's position and movement, as well as its velocity. The table below lists each variable in the observation space, along with its corresponding control and angle limits, the name of the variable as it appears in the XML file, the joint that the variable corresponds to, and the unit of measurement.

Num	Observation	Control Min	Control Max	Angle Min	Angle Max	Name (in corresponding XML file)	Joint	Unit
1	X-coordinate of robot base position	-inf	inf	-	-	-	-	m
2	Y-coordinate of robot base position	-inf	inf	-	-	-	-	m
3	Z-coordinate of robot base position	-inf	inf	-	-	-	-	m
4	Euler angle of robot base orientation (roll)	-inf	inf				

# Action Space
The action space is defined as a Box(-1.0, 1.0, (6,), float32), encompassing the absolute position and orientation of the 3f RobotiQ gripper. Control actions are applied by adjusting the real movement of the gripper's base in six degrees of freedom (6dof), consisting of three translational and three rotational movements. To ensure compatibility, the input of control actions is scaled to a range between -1 and 1. The elements of the action array are as follows:

| Num | Action                                               | Control Min | Control Max | Angle Min | Angle Max | Name (in corresponding XML file) | Joint | Unit        |
|-----|------------------------------------------------------|-------------|-------------|-----------|-----------|----------------------------------|-------|-------------|
| 0   | Linear translation of the gripper in x direction     | -1          | 1           | -0.1 (m)  | 0.1 (m)   | A_ARTx                           | slide | position (m)|
| 1   | Linear translation of the gripper in y direction     | -1          | 1           | -0.1 (m)  | 0.1 (m)   | A_ARTy                           | slide | position (m)|
| 2   | Linear translation of the gripper in z direction     | -1          | 1           | -0.1 (m)  | 0.1 (m)   | A_ARTz                           | slide | position (m)|
| 3   | Rotate the arm around the X-axis                     | -1          | 1           | -0.5 (deg)| 0.5 (deg) | A_ARTrotX                        | rotate| angle (rad) |
| 4   | Rotate the arm around the Y-axis                     | -1          | 1           | -0.5 (deg)| 0.5 (deg) | A_ARTrotY                        | rotate| angle (rad) |
| 5   | Rotate the arm around the Z-axis                     | -1          | 1           | -0.5 (deg)| 0.5 (deg) | A_ARTrotZ                        | rotate| angle (rad) |
