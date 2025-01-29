import pybullet as p
import pybullet_data

# Load the URDF file
p.connect(p.GUI)
# load plane from pybullet_data
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.loadURDF("nao.urdf", [0, 0, 0])

# Set the gravity
p.setGravity(0, 0, -9.8)

# Set the simulation time step
p.setTimeStep(1/240)

# Run the simulation
while True:
    p.stepSimulation()