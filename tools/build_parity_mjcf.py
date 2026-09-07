"""Build the MuJoCo parity scene from the six-DOF training URDF.

The URDF is the source of truth for the tree approximation.  MuJoCo's URDF
loader intentionally does not add a floating base, ground, or actuators, so
this small adapter adds only those simulation concerns.  Motor gear is one:
the controls are joint torques, matching Isaac Gym's
``set_dof_actuation_force_tensor`` path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/wheel_leg_train.urdf"
DEFAULT_OUTPUT = ROOT / "assets/wheel_leg_train_parity.xml"
JOINTS = (
    ("left_leg_0", 40.0),
    ("left_leg_1", 40.0),
    ("left_wheel", 47.294118),
    ("right_leg_0", 40.0),
    ("right_leg_1", 40.0),
    ("right_wheel", 47.294118),
)


def build(source: Path = URDF, output: Path = DEFAULT_OUTPUT) -> Path:
    spec = mujoco.MjSpec.from_file(str(source))
    spec.modelname = "wheel_leg_train_parity"
    # The Isaac Gym actor has a free root and starts at z=0.4.  The URDF root
    # is fixed by default, so add the freejoint to the existing base body.
    base = spec.body("base_link")
    if base is None:
        raise ValueError("training URDF has no base_link")
    if not any(j.name == "robot_free" for j in base.joints):
        base.add_freejoint(name="robot_free")

    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0],
        size=[100.0, 100.0, 0.1],
        friction=[1.10, 0.05, 0.01],
        condim=3,
        contype=1,
        conaffinity=1,
    )
    for joint, limit in JOINTS:
        spec.add_actuator(
            name=f"{joint}_motor",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint,
            gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ctrllimited=True,
            ctrlrange=[-limit, limit],
            forcelimited=True,
            forcerange=[-limit, limit],
        )

    spec.option.timestep = 0.001
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 100
    spec.option.ls_iterations = 50
    # MjSpec normalizes URDF input during compilation; MuJoCo requires that
    # compilation before serializing the generated MJCF.
    spec.compile()
    output.parent.mkdir(parents=True, exist_ok=True)
    spec.to_file(str(output))

    model = mujoco.MjModel.from_xml_path(str(output))
    expected = {"nq": 13, "nv": 12, "nu": 6, "njnt": 7}
    actual = {key: int(getattr(model, key)) for key in expected}
    if actual != expected:
        raise RuntimeError(f"unexpected parity model dimensions: {actual}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.source, args.output)
    print(f"built={result}")
    print("dimensions=nq:13 nv:12 nu:6 njnt:7")


if __name__ == "__main__":
    main()
