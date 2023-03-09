Observation Space
The observation space for the robotic arm environment consists of 12 continuous variables representing various aspects of the robot's position and movement, as well as its velocity. The table below lists each variable in the observation space, along with its corresponding control and angle limits, the name of the variable as it appears in the XML file, the joint that the variable corresponds to, and the unit of measurement.

Num	Observation	Control Min	Control Max	Angle Min	Angle Max	Name (in corresponding XML file)	Joint	Unit
1	X-coordinate of robot base position	-inf	inf	-	-	-	-	m
2	Y-coordinate of robot base position	-inf	inf	-	-	-	-	m
3	Z-coordinate of robot base position	-inf	inf	-	-	-	-	m
4	Euler angle of robot base orientation (roll)	-inf	inf					
