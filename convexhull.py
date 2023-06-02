import pybullet as p
import time
import pybullet_data

# Connect to the physics server
client = p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1 --background_color_blue=1')

# add search path for loadURDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Load a URDF, set the start position and orientation
start_pos = [0,0,5]
start_orientation = p.getQuaternionFromEuler([0,0,0])
model = p.loadURDF("urdf/robotiq.urdf", start_pos, start_orientation)

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
