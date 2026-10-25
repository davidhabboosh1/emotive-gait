import mujoco

from emotive_gait.robot.root import RootMode, to_root_mode


def test_root_modes_build_on_the_installed_mujoco():
    spec = mujoco.MjSpec()
    pelvis = spec.worldbody.add_body(name="pelvis")
    pelvis.add_geom(size=[0.1, 0, 0])  # a moving body needs mass to compile
    pelvis.add_freejoint()
    to_root_mode(spec, RootMode.Z_RAIL)
    model = spec.compile()
    assert model.njnt == 1 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE


def test_weld_removes_every_root_joint():
    spec = mujoco.MjSpec()
    pelvis = spec.worldbody.add_body(name="pelvis")
    pelvis.add_freejoint()
    to_root_mode(spec, RootMode.WELD)
    assert spec.compile().njnt == 0
