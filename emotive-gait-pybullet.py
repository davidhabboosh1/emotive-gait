import pybullet as p
import pybullet_data

# Load atlas\\atlas_v4_with_multisense.urdf and make sure it stands upright
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("atlas/atlas_v4_with_multisense.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))

# Load plane.urdf
p.loadURDF("plane.urdf")

# Set gravity
p.setGravity(0, 0, -9.81)

# Set simulation time step
p.setTimeStep(1/100)

# Set real-time simulation
p.setRealTimeSimulation(1)

# Run simulation for 10 seconds
p.setTimeOut(10)

# Keep simulation running
while p.isConnected():
    p.stepSimulation()