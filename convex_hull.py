import pybullet as p
import time
import pybullet_data
import numpy as np

# Connect to the physics server
client = p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1 --background_color_blue=1')

# add search path for loadURDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Load a URDF, set the start position and orientation
start_pos = [0,0,5]
start_orientation = p.getQuaternionFromEuler([0,0,0])
model = p.loadURDF("urdf/robotiq.urdf", start_pos, start_orientation)
blockUid = p.loadURDF("urdf/block.urdf", [0, 0, 10], start_orientation)
# change the color of the block
p.changeVisualShape(blockUid, -1, rgbaColor=[255, 255, 255, 1])

aabb_min, aabb_max = p.getAABB(blockUid)
x = np.linspace(aabb_min[0], aabb_max[0], 5)
y = np.linspace(aabb_min[1], aabb_max[1], 5)
z = np.linspace(aabb_min[2], aabb_max[2], 5)
red_point_dot_radius = 5 # Small sphere radius
red_point_dot_color = [1, 0, 0] # Red color: [r, g, b]
points = []
for i in range(len(x)):
    for j in range(len(y)):
        for k in range(len(z)):
            points.append([x[i], y[j], z[k]])
points = np.array(points)
red_point_dot_color = np.array([red_point_dot_color] * len(points))

p.addUserDebugPoints(pointPositions=points,
                        pointColorsRGB=red_point_dot_color,
                        pointSize=red_point_dot_radius)

# Create a large box to use as the background
# visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
#                                     halfExtents=[100, 100, 100],
#                                     rgbaColor=[1, 1, 1, 1])

# background_id = p.createMultiBody(baseMass=0,
#                                   baseInertialFramePosition=[0, 0, 0],
#                                   baseVisualShapeIndex=visualShapeId)

# Set the position and orientation of the box to cover the whole scene
# p.resetBasePositionAndOrientation(background_id, [0, 0, 0], [0, 0, 0, 1])

# Continue with your simulation
while True:
    p.stepSimulation()
