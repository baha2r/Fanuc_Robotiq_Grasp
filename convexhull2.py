from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pybullet as p
import time
import pybullet_data
import numpy as np
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


# Example points in 3D
p.connect(p.DIRECT)
p.setGravity(0, 0, 0)
p.setRealTimeSimulation(1)
# planeId = p.loadURDF("plane.urdf")
startpos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
robotiqUid = p.loadURDF("urdf/robotiq.urdf", startpos, startOrientation)
numJoints = p.getNumJoints(robotiqUid)
gripper_link_pose = []
for i in range(numJoints):
    if i==0 or i==4 or i==8:
        continue
    state = p.getLinkState(robotiqUid, i)
    # state = p.getJointState(robotiqUid, i)
    pos = state[0]
    # print(pos)
    gripper_link_pose.append(pos)
gripper_link_pose = np.array(gripper_link_pose)

hull = ConvexHull(gripper_link_pose)

# To visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot defining corner points
ax.plot(gripper_link_pose.T[0], gripper_link_pose.T[1], gripper_link_pose.T[2], "ko")
# ax.scatter(gripper_link_pose[:,0], gripper_link_pose[:,1], gripper_link_pose[:,2])

# Create a collection of polygons for the facets
polygons = [gripper_link_pose[simplex] for simplex in hull.simplices]

# Plot the convex hull facets
ax.add_collection3d(Poly3DCollection(polygons, facecolors='cyan', linewidths=1, edgecolors='b', alpha=.25))
ax.grid(False)
ax.set_axis_off()

plt.show()
# 12 = 2 * 6 faces are the simplices (2 simplices per square face)

# for s in hull.simplices:
#     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
#     ax.plot(gripper_link_pose[s, 0], gripper_link_pose[s, 1], gripper_link_pose[s, 2], "r-")

# plt.show()
