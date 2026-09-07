#!/usr/bin/env python3
"""Build a six-DOF Fudan-compatible training URDF from the closed MJCF.

The resulting tree is an intentionally documented training approximation. The
closed-chain MJCF remains the source of truth for later MuJoCo sim2sim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "wheeled_infantry.xml"
MESH_DIR = ROOT / "meshes_mj"
OUT_DIR = ROOT / "assets"
OUT_URDF = OUT_DIR / "wheel_leg_train.urdf"
OUT_MANIFEST = OUT_DIR / "wheel_leg_train.json"

# Map the source mechanism into a right-handed control frame:
# canonical x=forward, y=left, z=up. The source root frame has x=wheel axle,
# y=up, and the rolling direction is the remaining horizontal axis.
FRAME = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

GROUP_BODIES = {
    "base": {"robot"},
    "left_leg_0": {"left_upper_leg", "left_active_link", "left_long_link", "left_parallel_link", "left_node_link"},
    "left_leg_1": {"left_lower_leg"},
    "left_wheel": {"left_wheel"},
    "right_leg_0": {"right_upper_leg", "right_active_link", "right_long_link", "right_parallel_link", "right_node_link"},
    "right_leg_1": {"right_lower_leg"},
    "right_wheel": {"right_wheel"},
}

MESH_GROUP = {
    "base_visual_aluminum_part1": "base",
    "base_visual_aluminum_part2": "base",
    "base_visual_aluminum_part3": "base",
    "base_visual_aluminum_part4": "base",
    "base_visual_aluminum_part5": "base",
    "base_visual_carbon": "base",
    "base_visual_plastic": "base",
    "l_active_aluminum": "left_leg_0",
    "l_long_aluminum": "left_leg_0",
    "l_node_aluminum": "left_leg_0",
    "l_parallel_aluminum": "left_leg_0",
    "l_upper_aluminum": "left_leg_0",
    "l_lower_aluminum": "left_leg_1",
    "left_wheel_complete": "left_wheel",
    "r_active_aluminum": "right_leg_0",
    "r_long_aluminum": "right_leg_0",
    "r_node_aluminum": "right_leg_0",
    "r_parallel_aluminum": "right_leg_0",
    "r_upper_aluminum": "right_leg_0",
    "r_lower_aluminum": "right_leg_1",
    "right_wheel_complete": "right_wheel",
}

P0 = {
    "left": np.array([0.177500, 0.000024, -0.000014]),
    "right": np.array([-0.177500, 0.000024, -0.000014]),
}
P1 = {
    "left": P0["left"] + np.array([0.017500, -0.142940, 0.153840]),
    "right": P0["right"] + np.array([-0.017500, -0.142940, 0.153840]),
}
P2 = {
    "left": P1["left"] + np.array([0.025489, -0.197014, -0.153817]),
    "right": P1["right"] + np.array([-0.025511, -0.197077, -0.153795]),
}


def vec(text: str | None, size: int = 3) -> np.ndarray:
    if not text:
        return np.zeros(size)
    return np.asarray([float(value) for value in text.split()], dtype=float)


def quat_wxyz_to_matrix(values: np.ndarray) -> np.ndarray:
    w, x, y, z = values
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rpy_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    if sy > 1.0e-8:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def rpy_for_z_axis(vector: np.ndarray) -> tuple[float, float, float]:
    vector = vector / max(float(np.linalg.norm(vector)), 1.0e-12)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, vector)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(z, vector))
    if sine < 1.0e-8:
        return (math.pi if cosine < 0 else 0.0, 0.0, 0.0)
    axis /= sine
    angle = math.atan2(sine, cosine)
    x, y, z_axis = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    rotation = np.array(
        [[t * x * x + c, t * x * y - s * z_axis, t * x * z_axis + s * y],
         [t * x * y + s * z_axis, t * y * y + c, t * y * z_axis - s * x],
         [t * x * z_axis - s * y, t * y * z_axis + s * x, t * z_axis * z_axis + c]]
    )
    return rpy_from_matrix(rotation)


def fmt(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def body_group(name: str) -> str:
    for group, names in GROUP_BODIES.items():
        if name in names:
            return group
    return "base"


def load_source(path: Path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    root_pos = np.array(data.xpos[root_id], dtype=float)
    root_rotation = np.asarray(data.xmat[root_id], dtype=float).reshape(3, 3)
    return model, data, root_id, root_pos, root_rotation


def aggregate_inertials(model, data, root_id: int, root_pos: np.ndarray, root_rotation: np.ndarray):
    inertials: dict[str, dict[str, np.ndarray | float]] = {}
    for group in GROUP_BODIES:
        inertials[group] = {"mass": 0.0, "com": np.zeros(3), "inertia_ref": np.zeros((3, 3))}

    for body_id in range(model.nbody):
        name = model.body(body_id).name
        group = body_group(name)
        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            continue
        com_old = root_rotation.T @ (np.asarray(data.xipos[body_id]) - root_pos)
        com = FRAME @ com_old
        body_rotation_old = root_rotation.T @ np.asarray(data.xmat[body_id]).reshape(3, 3)
        principal = quat_wxyz_to_matrix(np.asarray(model.body_iquat[body_id], dtype=float))
        inertia_body = principal @ np.diag(np.asarray(model.body_inertia[body_id], dtype=float)) @ principal.T
        inertia_old = body_rotation_old @ inertia_body @ body_rotation_old.T
        inertia = FRAME @ inertia_old @ FRAME.T
        inertials[group]["mass"] += mass
        inertials[group]["com"] += mass * com
        # Accumulate each body's inertia about the group's link-frame origin.
        ref = FRAME @ {
            "base": np.zeros(3),
            "left_leg_0": P0["left"], "left_leg_1": P1["left"], "left_wheel": P2["left"],
            "right_leg_0": P0["right"], "right_leg_1": P1["right"], "right_wheel": P2["right"],
        }[group]
        offset = com - ref
        inertials[group]["inertia_ref"] += inertia + mass * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )

    refs = {
        "base": np.zeros(3),
        "left_leg_0": P0["left"], "left_leg_1": P1["left"], "left_wheel": P2["left"],
        "right_leg_0": P0["right"], "right_leg_1": P1["right"], "right_wheel": P2["right"],
    }
    result = {}
    for group, item in inertials.items():
        mass = float(item["mass"])
        if mass <= 0.0:
            raise ValueError(f"group {group} has no mass")
        com_world = item["com"] / mass
        ref = FRAME @ refs[group]
        offset = com_world - ref
        inertia_ref = item["inertia_ref"]
        inertia_com = inertia_ref - mass * ((np.dot(offset, offset) * np.eye(3)) - np.outer(offset, offset))
        result[group] = {"mass": mass, "com": offset, "inertia": inertia_com}
    total = sum(float(item["mass"]) for item in result.values())
    source_total = float(np.sum(model.body_mass))
    if not math.isclose(total, source_total, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(f"aggregated mass mismatch: {total} != {source_total}")
    return result


def mesh_geometries(xml_path: Path, data, root_id: int, root_pos: np.ndarray, root_rotation: np.ndarray):
    xml_root = ET.parse(xml_path).getroot()
    body_ids = {body_id: name for body_id, name in enumerate([])}
    import mujoco
    body_name_to_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id): body_id
        for body_id in range(model.nbody)
    }
    geometries = defaultdict(list)
    for body in xml_root.findall(".//body"):
        body_name = body.get("name")
        if not body_name or body_name not in body_name_to_id:
            continue
        body_id = body_name_to_id[body_name]
        body_rotation_old = root_rotation.T @ np.asarray(data.xmat[body_id]).reshape(3, 3)
        body_pos_old = root_rotation.T @ (np.asarray(data.xpos[body_id]) - root_pos)
        for geom in body.findall("./geom"):
            mesh = geom.get("mesh")
            group = MESH_GROUP.get(mesh or "")
            if not group:
                continue
            geom_pos = vec(geom.get("pos"))
            geom_quat = quat_wxyz_to_matrix(vec(geom.get("quat"), 4) if geom.get("quat") else np.array([1.0, 0.0, 0.0, 0.0]))
            pos_old = body_pos_old + body_rotation_old @ geom_pos
            rotation = FRAME @ body_rotation_old @ geom_quat
            geometries[group].append((mesh, pos_old, rotation))
    return geometries


def inertial_xml(item: dict[str, np.ndarray | float]) -> str:
    inertia = np.asarray(item["inertia"], dtype=float)
    return (
        f'<inertial><origin xyz="{fmt(np.asarray(item["com"]))}" rpy="0 0 0"/>'
        f'<mass value="{float(item["mass"]):.12g}"/>'
        f'<inertia ixx="{inertia[0,0]:.12g}" ixy="{inertia[0,1]:.12g}" ixz="{inertia[0,2]:.12g}" '
        f'iyy="{inertia[1,1]:.12g}" iyz="{inertia[1,2]:.12g}" izz="{inertia[2,2]:.12g}"/></inertial>'
    )


def visual_xml(group: str, geometries, refs) -> str:
    lines = []
    ref = FRAME @ refs[group]
    for mesh, pos_old, rotation in geometries.get(group, []):
        pos = FRAME @ pos_old - ref
        rpy = rpy_from_matrix(rotation)
        lines.append(
            f'<visual><origin xyz="{fmt(pos)}" rpy="{fmt(rpy)}"/><geometry>'
            f'<mesh filename="../meshes_mj/{mesh}.stl" scale="0.001 0.001 0.001"/>'
            '</geometry></visual>'
        )
    return "".join(lines)


def collision_box(size: tuple[float, float, float], origin: np.ndarray | None = None, rpy=(0.0, 0.0, 0.0)) -> str:
    origin = np.zeros(3) if origin is None else origin
    return f'<collision><origin xyz="{fmt(origin)}" rpy="{fmt(rpy)}"/><geometry><box size="{fmt(size)}"/></geometry></collision>'


def collision_cylinder(radius: float, length: float, origin: np.ndarray, rpy) -> str:
    return f'<collision><origin xyz="{fmt(origin)}" rpy="{fmt(rpy)}"/><geometry><cylinder radius="{radius:.8g}" length="{length:.8g}"/></geometry></collision>'


def link_xml(name: str, group: str, item, geometries, refs, collision: str) -> str:
    return f'<link name="{name}">{inertial_xml(item)}{visual_xml(group, geometries, refs)}{collision}</link>'


def joint_xml(name, parent, child, origin, axis, effort, velocity, joint_type="revolute", limit=True, lower=-3.14, upper=3.14) -> str:
    if joint_type == "continuous":
        limits = f'<limit effort="{effort:.8g}" velocity="{velocity:.8g}"/>'
    elif limit:
        limits = f'<limit lower="{lower:.8g}" upper="{upper:.8g}" effort="{effort:.8g}" velocity="{velocity:.8g}"/>'
    else:
        limits = ""
    return f'<joint name="{name}" type="{joint_type}"><origin xyz="{fmt(origin)}" rpy="0 0 0"/><parent link="{parent}"/><child link="{child}"/><axis xyz="{fmt(axis)}"/>{limits}</joint>'


def build(xml_path: Path, out_urdf: Path, out_manifest: Path) -> None:
    global model
    model, data, root_id, root_pos, root_rotation = load_source(xml_path)
    inertials = aggregate_inertials(model, data, root_id, root_pos, root_rotation)
    geometries = mesh_geometries(xml_path, data, root_id, root_pos, root_rotation)
    refs_old = {
        "base": np.zeros(3), "left_leg_0": P0["left"], "left_leg_1": P1["left"], "left_wheel": P2["left"],
        "right_leg_0": P0["right"], "right_leg_1": P1["right"], "right_wheel": P2["right"],
    }
    # Use one canonical +Y pitch axis for all four leg joints.  Left/right
    # sign conventions belong to the action/deployment adapter, not to the
    # tree asset topology; mixed signs here caused asymmetric responses.
    axis_pitch = np.array([0.0, 1.0, 0.0])
    axis_wheel = FRAME @ np.array([1.0, 0.0, 0.0])
    joints = []
    links = []
    links.append(link_xml("base_link", "base", inertials["base"], geometries, refs_old, collision_box((0.36, 0.18, 0.16))))
    for side in ("left", "right"):
        prefix = f"{side}_"
        leg0 = prefix + "leg_0"
        leg1 = prefix + "leg_1"
        wheel = prefix + "wheel"
        p0, p1, p2 = P0[side], P1[side], P2[side]
        v1 = FRAME @ (p1 - p0)
        v2 = FRAME @ (p2 - p1)
        links.append(link_xml(leg0 + "_link", leg0, inertials[leg0], geometries, refs_old, collision_box((0.06, 0.06, float(np.linalg.norm(v1))), 0.5 * v1, rpy_for_z_axis(v1))))
        links.append(link_xml(leg1 + "_link", leg1, inertials[leg1], geometries, refs_old, collision_box((0.06, 0.06, float(np.linalg.norm(v2))), 0.5 * v2, rpy_for_z_axis(v2))))
        links.append(link_xml(wheel + "_link", wheel, inertials[wheel], geometries, refs_old, collision_cylinder(0.060, 0.040, np.zeros(3), (math.pi / 2, 0.0, 0.0))))
        joints.append(joint_xml(prefix + "leg_0", "base_link", leg0 + "_link", FRAME @ p0, axis_pitch, 40.0, 30.0))
        joints.append(joint_xml(prefix + "leg_1", leg0 + "_link", leg1 + "_link", v1, axis_pitch, 40.0, 30.0))
        # Isaac Gym's URDF parser is more reliable with a wide revolute joint
        # than with a continuous joint for wheel spin DOFs.
        joints.append(joint_xml(prefix + "wheel", leg1 + "_link", wheel + "_link", v2, axis_wheel, 47.2941176472, 120.0, "revolute", True, -1000.0, 1000.0))

    chain = {
        "left": {"p0_m": (FRAME @ P0["left"]).tolist(), "p1_m": (FRAME @ P1["left"]).tolist(), "p2_m": (FRAME @ P2["left"]).tolist(), "link_lengths_m": [float(np.linalg.norm(P1["left"] - P0["left"])), float(np.linalg.norm(P2["left"] - P1["left"]))]},
        "right": {"p0_m": (FRAME @ P0["right"]).tolist(), "p1_m": (FRAME @ P1["right"]).tolist(), "p2_m": (FRAME @ P2["right"]).tolist(), "link_lengths_m": [float(np.linalg.norm(P1["right"] - P0["right"])), float(np.linalg.norm(P2["right"] - P1["right"]))]},
    }
    # Keep base_link as the URDF root. Isaac Gym uses this as the floating
    # actor root; the complete MJCF remains the mass/closed-chain truth model.
    xml = '<?xml version="1.0"?>\n<robot name="wheel_leg_train">' + "".join(links) + "".join(joints) + "</robot>\n"
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_urdf.write_text(xml, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source_mjcf": str(xml_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "training_approximation": "six_dof_virtual_serial_chain_with_fixed_passive_attachments",
        "dof_names": ["left_leg_0", "left_leg_1", "left_wheel", "right_leg_0", "right_leg_1", "right_wheel"],
        "action_semantics": ["position", "position", "velocity", "position", "position", "velocity"],
        "canonical_frame": {"x": "forward", "y": "left", "z": "up"},
        "total_mass_kg": float(sum(float(item["mass"]) for item in inertials.values())),
        "serial_chain": chain,
        "notes": ["Passive five-bar and guide-wheel bodies are fixed at the nominal MJCF pose.", "Use the full closed-chain MJCF for sim2sim and actuator torque mapping."],
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"urdf": str(out_urdf), "manifest": str(out_manifest), "dof_count": 6, "total_mass_kg": manifest["total_mass_kg"], "chain": chain}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=MJCF)
    parser.add_argument("--urdf", type=Path, default=OUT_URDF)
    parser.add_argument("--manifest", type=Path, default=OUT_MANIFEST)
    args = parser.parse_args()
    build(args.source.resolve(strict=True), args.urdf.resolve(), args.manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
