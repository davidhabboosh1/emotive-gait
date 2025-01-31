import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
ctrl0 = data.ctrl.copy()

nv = model.nv
nu = model.nu

R = np.eye(nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        print('step')
        
        qpos0 = data.qpos.copy()
        ctrl0 = data.ctrl.copy()
        
        A = np.zeros((2 * nv, 2 * nv))
        B = np.zeros((2 * nv, nu))
        mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
        
        mujoco.mj_resetData(model, data)
        data.qpos = qpos0
        mujoco.mj_forward(model, data)
        
        jac_com = np.zeros((3, nv))
        jac_contact = np.zeros((3, nv))
        
        mujoco.mj_jacSubtreeCom(model, data, jac_com, 0)
        
        contact_body_ids = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_body = model.geom_bodyid[contact.geom1]
            geom2_body = model.geom_bodyid[contact.geom2]
            contact_body_ids.add(geom1_body)
            contact_body_ids.add(geom2_body)
            
        num_contacts = len(contact_body_ids)
        if num_contacts > 0:
            jac_sum = np.zeros((3, nv))
            for body_id in contact_body_ids:
                temp_jac = np.zeros((3, nv))
                mujoco.mj_jacBodyCom(model, data, temp_jac, None, body_id)
                jac_sum += temp_jac
            jac_contact = jac_sum / num_contacts
            
        jac_diff = jac_com - jac_contact
        Qbalance = jac_diff.T @ jac_diff

        joint_names = [model.joint(i) for i in range(model.njnt)]
        
        root_dofs = range(6)
        body_dofs = range(6, nv)
        balance_dofs = body_dofs
        other_dofs = np.setdiff1d(body_dofs, balance_dofs)
        
        BALANCE_COST        = 1000
        BALANCE_JOINT_COST  = 3
        OTHER_JOINT_COST    = .3

        Qjoint = np.eye(nv)
        Qjoint[root_dofs, root_dofs] *= 0
        Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
        Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

        Qpos = BALANCE_COST * Qbalance + Qjoint

        Q = np.block([[Qpos, np.zeros((nv, nv))],
                    [np.zeros((nv, 2*nv))]])

        mujoco.mj_resetData(model, data)
        data.ctrl = ctrl0
        data.qpos = qpos0
        
        A = np.zeros((2 * nv, 2 * nv))
        B = np.zeros((2 * nv, nu))
        epsilon = 1e-6
        flg_centered = True
        mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)
        
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        
        mujoco.mj_resetData(model, data)
        data.qpos = qpos0
        
        dq = np.zeros(nv)
        
        mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T

        data.ctrl = ctrl0 - K @ dx
        
        mujoco.mj_step(model, data)

        viewer.sync()