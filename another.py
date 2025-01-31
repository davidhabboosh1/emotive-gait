import numpy as np
import mujoco

import numpy as np
import mujoco

def compute_linearized_dynamics(model, data, q_desired):
    """
    Computes the linearized system dynamics (A, B) at a given desired joint position q_desired.

    Parameters:
    - model: MuJoCo model
    - data: MuJoCo data
    - q_desired: Desired joint positions (numpy array)

    Returns:
    - A: (2*nq, 2*nq) state transition matrix
    - B: (2*nq, nq) control input matrix
    """
    nq, nv = model.nq, model.nv
    nx = 2 * nq  # Full state size (q, v)

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nv))

    # Reset state to desired position
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_desired
    data.qvel[:] = np.zeros(nv)  # Ensure zero initial velocity
    mujoco.mj_forward(model, data)  # Update derivatives at desired q

    # Construct A matrix
    A[:nq, nq:] = np.eye(nq)  # dq/dt = v

    # Compute M⁻¹ * C (this is acceleration related to damping)
    # Create a dummy force vector to solve for the effect of damping forces
    dummy_force = np.zeros(nv)  # Zero control torques for now
    M_inv_dummy = np.zeros_like(dummy_force)
    mujoco.mj_solveM(model, data, M_inv_dummy, dummy_force)  # Solve M⁻¹ * F (M⁻¹ * 0 here)

    # Construct B matrix (control input effect on acceleration)
    B[nq:, :] = np.eye(nv)  # Identity, because MuJoCo expects direct control torques

    return A, B

with open('scene.xml', 'r') as file:
    xml = file.read()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Assume q_desired (joint angles) and u_ff (feedforward torques) are precomputed
q_desired = np.zeros((10, model.nq))
u_ff = np.zeros((10, model.nv))

T = q_desired.shape[0]  # Number of timesteps
nv = model.nv  # Number of DoFs

# LQR cost matrices (tune these)
Q = np.eye(2 * nv) * 10  # Penalizes state deviation
R = np.eye(nv) * 0.1  # Penalizes control effort

# Storage for gains
K_t = np.zeros((T, nv, 2 * nv))

# Solve TVLQR backwards in time
P = np.copy(Q)  # Final cost
for t in reversed(range(T - 1)):
    A_t, B_t = compute_linearized_dynamics(model, data, q_desired[t])
    
    # Solve Riccati equation
    K_t[t] = np.linalg.solve(R + B_t.T @ P @ B_t, B_t.T @ P @ A_t)
    P = Q + A_t.T @ P @ A_t - K_t[t].T @ (R + B_t.T @ P @ B_t) @ K_t[t]

# Simulation loop with TVLQR stabilization
for t in range(T):
    mujoco.mj_forward(model, data)

    # Get current state
    q = data.qpos
    dq = data.qvel
    x = np.hstack([q, dq])
    
    # Compute control
    e = x - np.hstack([q_desired[t], np.zeros(nv)])
    u = u_ff[t] - K_t[t] @ e
    
    # Apply torques
    data.ctrl[:] = u
    mujoco.mj_step(model, data)