import pybullet as p
import pybullet_data

# Load atlas\\atlas_v4_with_multisense.urdf and make sure it stands upright
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot_id = p.loadURDF("atlas/atlas_v4_with_multisense.urdf", [0, 0, 0], useFixedBase=False)

# for _ in range(100):  # Run a few simulation steps for settling
#     p.stepSimulation()

aabb_min, aabb_max = p.getAABB(robot_id)

# load plane.urdf at the bottom of the robot
plane_id = p.loadURDF("plane.urdf", [0, 0, aabb_min[2] - 0.82], useFixedBase=True)

for _ in range(100):  # Run a few simulation steps for settling
    p.stepSimulation()

# Set gravity
p.setGravity(0, 0, -9.81)

# Set simulation time step
p.setTimeStep(1/240)

# Set real-time simulation
p.setRealTimeSimulation(1)

# Run simulation for 10 seconds
p.setTimeOut(10)

# Keep simulation running
while p.isConnected():
    p.stepSimulation()