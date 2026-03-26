#!/usr/bin/env python3
"""
Rerun-based robot teleoperation data visualization (LeRobot v2.1 format).

Replaces the PyRender/OpenCV pipeline with Rerun's interactive viewer.
Features:
  - 3D world view with EEF poses, trajectories, gripper and controller meshes
  - Camera and tactile image panels
  - Timeseries plots for EEF position and gripper width
  - Interactive timeline scrubbing across all episodes

Usage (run from any directory):
    python /path/to/src/viz_rerun.py /path/to/lerobot_dataset
    python /path/to/src/viz_rerun.py /path/to/dataset --episode 0
    python /path/to/src/viz_rerun.py /path/to/dataset --save output.rrd
"""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation
import rerun as rr
import rerun.blueprint as rrb

# ── project imports ──────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

# ── constants ─────────────────────────────────────────────────────────────────
ROBOT_IDS = [0, 1]
ROBOT_COLORS = {0: [220, 60, 60], 1: [60, 220, 60]}  # red / green


# ── mesh loading ──────────────────────────────────────────────────────────────
def _load_trimesh(path, scale=1.0, extra_transform=None):
    """Load an STL file via trimesh, returning vertex/face/normal arrays."""
    import trimesh
    mesh = trimesh.load(path)
    if scale != 1.0:
        mesh.apply_scale(scale)
    if extra_transform is not None:
        mesh.apply_transform(extra_transform)
    return {
        "vertices": mesh.vertices.astype(np.float32),
        "faces":    mesh.faces.astype(np.int32),
        "normals":  (mesh.vertex_normals.astype(np.float32)
                     if hasattr(mesh, "vertex_normals") else None),
    }


def load_static_meshes():
    """
    Load gripper and controller STL files once.
    Returns a dict with keys:
      gripper_left, gripper_right, controller_left, controller_right
    """
    meshes = {}

    # ── gripper ───────────────────────────────────────────────────────────────
    gripper_path = os.path.join(_SRC, "meshes", "夹爪.STL")
    if os.path.exists(gripper_path):
        try:
            import trimesh
            base = trimesh.load(gripper_path)
            base.apply_scale(0.001)                                   # mm → m
            center = (base.bounds[0] + base.bounds[1]) / 2
            base.apply_translation(-center)
            rot = np.eye(4)
            rot[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
            base.apply_transform(rot)

            # left finger
            meshes["gripper_left"] = {
                "vertices": base.vertices.astype(np.float32),
                "faces":    base.faces.astype(np.int32),
                "normals":  base.vertex_normals.astype(np.float32),
            }

            # right finger  – mirror across Y axis
            right = base.copy()
            mirror = np.eye(4); mirror[1, 1] = -1
            right.apply_transform(mirror)
            meshes["gripper_right"] = {
                "vertices": right.vertices.astype(np.float32),
                "faces":    right.faces.astype(np.int32),
                "normals":  right.vertex_normals.astype(np.float32),
            }
        except Exception as e:
            print(f"Warning: could not load gripper STL: {e}")

    # ── controllers ───────────────────────────────────────────────────────────
    for side, fname in [
        ("left",  "Oculus_Meta_Quest_Touch_Plus_Controller_Left.stl"),
        ("right", "Oculus_Meta_Quest_Touch_Plus_Controller_Right.stl"),
    ]:
        fpath = os.path.join(_SRC, "meshes", fname)
        if os.path.exists(fpath):
            try:
                meshes[f"controller_{side}"] = _load_trimesh(fpath, scale=0.0015)
            except Exception as e:
                print(f"Warning: could not load controller mesh ({side}): {e}")

    return meshes


# ── static scene logging ──────────────────────────────────────────────────────
def log_static_geometry(meshes):
    """Log world frame, axes, and mesh geometries that never change shape."""

    # world coordinate convention: Z-up, right-hand
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for r in ROBOT_IDS:
        # EEF coordinate axes (3 arrows in local EEF frame, inherit EEF transform)
        rr.log(
            f"world/robot{r}/eef/axes",
            rr.Arrows3D(
                vectors=np.eye(3, dtype=np.float32) * 0.05,
                origins=np.zeros((3, 3), dtype=np.float32),
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )

        # Controller mesh – geometry is static; only its parent EEF transform moves
        ctrl_key = f"controller_{'left' if r == 1 else 'right'}"
        if ctrl_key in meshes:
            m = meshes[ctrl_key]
            n = len(m["vertices"])
            ctrl_rot_quat = Rotation.from_euler("y", 90, degrees=True).as_quat()  # xyzw
            rr.log(
                f"world/robot{r}/eef/controller",
                rr.Transform3D(
                    translation=[0, 0, 0.05],
                    quaternion=rr.Quaternion(xyzw=ctrl_rot_quat),
                ),
                static=True,
            )
            rr.log(
                f"world/robot{r}/eef/controller/mesh",
                rr.Mesh3D(
                    vertex_positions=m["vertices"],
                    triangle_indices=m["faces"],
                    vertex_normals=m.get("normals"),
                    vertex_colors=np.full((n, 3), [100, 100, 200], dtype=np.uint8),
                ),
                static=True,
            )

        # Gripper finger meshes – shape is constant, only transforms change per frame
        for side in ("left", "right"):
            key = f"gripper_{side}"
            if key in meshes:
                m = meshes[key]
                n = len(m["vertices"])
                rr.log(
                    f"world/robot{r}/eef/gripper/{side}/mesh",
                    rr.Mesh3D(
                        vertex_positions=m["vertices"],
                        triangle_indices=m["faces"],
                        vertex_normals=m.get("normals"),
                        vertex_colors=np.full((n, 3), [180, 180, 180], dtype=np.uint8),
                    ),
                    static=True,
                )

            # Tactile sensor marker (small disc at gripper tip)
            sensor_color = [0, 220, 0] if side == "left" else [220, 0, 0]
            rr.log(
                f"world/robot{r}/eef/gripper/{side}/sensor",
                rr.Points3D(positions=[[0, 0, 0]], colors=[sensor_color], radii=[0.012]),
                static=True,
            )


# ── LeRobot parquet loader ────────────────────────────────────────────────────
def _pose6_to_mat(pose6):
    """6-dim pose (pos + rotvec) → 4×4 transform matrix."""
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose6[:3]
    T[:3, :3] = Rotation.from_rotvec(pose6[3:6]).as_matrix()
    return T


def _decode_image_bytes(img_dict):
    """Decode JPEG/PNG bytes stored in LeRobot parquet image column."""
    import cv2
    raw = img_dict.get("bytes") if isinstance(img_dict, dict) else img_dict
    if raw is None:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_lerobot_episode(dataset_path, episode_idx):
    """
    Load one episode from a LeRobot v2.1 parquet dataset.

    Returns data dict compatible with log_frame():
      data['robot0']['poses']         list of 4×4 np.float32
      data['robot0']['gripper']       list of float
      data['robot0']['visual']        list of HxWx3 uint8 (or [])
      data['robot0']['left_tactile']  ...
      data['robot0']['right_tactile'] ...
      data['robot0']['left_pc']       [] (not available)
      data['robot0']['right_pc']      []
    """
    import pandas as pd

    chunk = episode_idx // 1000
    parquet_path = os.path.join(
        dataset_path,
        f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet",
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)

    data = {f"robot{r}": {
        "poses": [], "gripper": [],
        "visual": [], "left_tactile": [], "right_tactile": [],
        "left_pc": [], "right_pc": [],
    } for r in ROBOT_IDS}

    # image column map  (lerobot column → robot / sensor)
    img_cols = {
        "observation.images.camera0":      ("robot0", "visual"),
        "observation.images.camera1":      ("robot1", "visual"),
        "observation.images.tactile_left_0":  ("robot0", "left_tactile"),
        "observation.images.tactile_right_0": ("robot0", "right_tactile"),
        "observation.images.tactile_left_1":  ("robot1", "left_tactile"),
        "observation.images.tactile_right_1": ("robot1", "right_tactile"),
    }
    available_img = {c: k for c, k in img_cols.items() if c in df.columns}

    for _, row in df.iterrows():
        # ── parse state vector (20-dim) ───────────────────────────────────────
        state = np.asarray(row["observation.state"], dtype=np.float32)

        # robot0 absolute pose
        robot0_mat = _pose6_to_mat(state[0:6])
        robot0_grip = float(state[6])

        # robot1 absolute pose derived from relative transform in state[14:20]
        rel_0to1_mat = _pose6_to_mat(state[14:20])
        robot1_mat = robot0_mat @ np.linalg.inv(rel_0to1_mat)
        robot1_grip = float(state[13])

        data["robot0"]["poses"].append(robot0_mat)
        data["robot0"]["gripper"].append(robot0_grip)
        data["robot1"]["poses"].append(robot1_mat)
        data["robot1"]["gripper"].append(robot1_grip)

        # ── images ────────────────────────────────────────────────────────────
        for col, (robot_key, sensor_key) in available_img.items():
            img = _decode_image_bytes(row[col])
            if img is not None:
                data[robot_key][sensor_key].append(img)

    return data


# ── per-frame logging ─────────────────────────────────────────────────────────
def log_frame(frame_idx, data, trajectories, global_frame):
    """Log one timestep of robot data to Rerun."""
    rr.set_time("frame", sequence=global_frame)

    for r in ROBOT_IDS:
        prefix = f"robot{r}"
        color  = ROBOT_COLORS[r]

        # ── EEF pose ──────────────────────────────────────────────────────────
        poses = data[prefix].get("poses", [])
        if poses and frame_idx < len(poses):
            T      = poses[frame_idx]
            pos    = T[:3, 3]
            quat   = Rotation.from_matrix(T[:3, :3]).as_quat()   # xyzw

            trajectories[r].append(pos.copy())

            rr.log(
                f"world/robot{r}/eef",
                rr.Transform3D(
                    translation=pos,
                    quaternion=rr.Quaternion(xyzw=quat),
                ),
            )

            # growing trajectory line
            traj = np.array(trajectories[r], dtype=np.float32)
            rr.log(
                f"world/robot{r}/trajectory",
                rr.LineStrips3D([traj], colors=[color], radii=[0.003]),
            )

            # timeseries channels for EEF position
            rr.log(f"timeseries/robot{r}/eef_x", rr.Scalars(float(pos[0])))
            rr.log(f"timeseries/robot{r}/eef_y", rr.Scalars(float(pos[1])))
            rr.log(f"timeseries/robot{r}/eef_z", rr.Scalars(float(pos[2])))

        # ── gripper width ─────────────────────────────────────────────────────
        gripper_data = data[prefix].get("gripper", [])
        if gripper_data and frame_idx < len(gripper_data):
            grip_width = float(gripper_data[frame_idx])
            rr.log(f"timeseries/robot{r}/gripper_width", rr.Scalars(grip_width))

            # update finger offset transforms (only the translation changes)
            offset = max(grip_width * 0.5, 0.03)
            for side, sign in [("left", -1), ("right", 1)]:
                rr.log(
                    f"world/robot{r}/eef/gripper/{side}",
                    rr.Transform3D(translation=[0.02, sign * offset, -0.04]),
                )

        # ── camera images ─────────────────────────────────────────────────────
        for sensor, cam_path in [
            ("visual",        f"cameras/robot{r}/visual"),
            ("left_tactile",  f"cameras/robot{r}/left_tactile"),
            ("right_tactile", f"cameras/robot{r}/right_tactile"),
        ]:
            imgs = data[prefix].get(sensor, [])
            if imgs and frame_idx < len(imgs):
                rr.log(cam_path, rr.Image(imgs[frame_idx]))

        # ── tactile point clouds ──────────────────────────────────────────────
        for pc_key, pc_color in [
            ("left_pc",  [0, 200, 100]),
            ("right_pc", [200, 100, 0]),
        ]:
            pcs = data[prefix].get(pc_key, [])
            if pcs and frame_idx < len(pcs):
                pts = pcs[frame_idx]
                if len(pts) > 0:
                    side = "left" if "left" in pc_key else "right"
                    rr.log(
                        f"world/robot{r}/tactile/{side}",
                        rr.Points3D(
                            positions=pts,
                            colors=np.tile(pc_color, (len(pts), 1)).astype(np.uint8),
                            radii=[0.001],
                        ),
                    )


# ── blueprint ─────────────────────────────────────────────────────────────────
def make_blueprint():
    """Configure the Rerun viewer layout."""
    return rrb.Blueprint(
        rrb.Horizontal(
            # left column: 3D world + timeseries
            rrb.Vertical(
                rrb.Spatial3DView(name="3D World", origin="world"),
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="Robot 0", origin="timeseries/robot0"),
                    rrb.TimeSeriesView(name="Robot 1", origin="timeseries/robot1"),
                ),
                row_shares=[3, 2],
            ),
            # right column: camera feeds + gripper timeseries
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="R0 Visual",  origin="cameras/robot0/visual"),
                    rrb.Spatial2DView(name="R0 L-Tact",  origin="cameras/robot0/left_tactile"),
                    rrb.Spatial2DView(name="R0 R-Tact",  origin="cameras/robot0/right_tactile"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="R1 Visual",  origin="cameras/robot1/visual"),
                    rrb.Spatial2DView(name="R1 L-Tact",  origin="cameras/robot1/left_tactile"),
                    rrb.Spatial2DView(name="R1 R-Tact",  origin="cameras/robot1/right_tactile"),
                ),
                rrb.TimeSeriesView(name="Gripper Widths", origin="timeseries"),
                row_shares=[2, 2, 1],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize LeRobot teleoperation data with Rerun.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/viz_rerun.py /path/to/lerobot_dataset\n"
            "  python src/viz_rerun.py /path/to/dataset --episode 3\n"
            "  python src/viz_rerun.py /path/to/dataset --save out.rrd\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="LeRobot dataset directory (must contain meta/info.json)",
    )
    parser.add_argument(
        "--episode", "-e", type=int, default=None,
        help="Single episode index to visualize (default: all episodes)",
    )
    parser.add_argument(
        "--save", "-s", type=str, default=None,
        help="Save recording to .rrd file instead of spawning the viewer",
    )
    args = parser.parse_args()

    # resolve to absolute path so the script works from any CWD
    dataset_path = os.path.abspath(args.dataset)

    if not os.path.exists(dataset_path):
        print(f"Error: path not found: {dataset_path}")
        sys.exit(1)

    if not os.path.exists(os.path.join(dataset_path, "meta", "info.json")):
        print("Error: not a LeRobot dataset (missing meta/info.json)")
        sys.exit(1)

    # ── initialise Rerun ──────────────────────────────────────────────────────
    blueprint = make_blueprint()
    rr.init("robot_visualization", spawn=(args.save is None))
    if args.save:
        rr.save(args.save, default_blueprint=blueprint)
    else:
        rr.send_blueprint(blueprint)

    # ── static geometry ───────────────────────────────────────────────────────
    print("Loading meshes...")
    meshes = load_static_meshes()
    log_static_geometry(meshes)
    print(f"  gripper: {'✓' if 'gripper_left' in meshes else '✗ (STL not found)'}  "
          f"controllers: {'✓' if 'controller_left' in meshes else '✗ (STL not found)'}")

    # ── episode data ──────────────────────────────────────────────────────────
    import json
    with open(os.path.join(dataset_path, "meta", "info.json")) as f:
        meta = json.load(f)
    n_episodes = meta["total_episodes"]
    print(f"LeRobot dataset: {meta['total_frames']:,} frames | {n_episodes} episodes")

    ep_indices = list(range(n_episodes)) if args.episode is None else [args.episode]

    global_frame = 0
    for ep_idx in ep_indices:
        rr.set_time("episode", sequence=ep_idx)
        print(f"  ep {ep_idx:3d}:", end="", flush=True)
        try:
            data = load_lerobot_episode(dataset_path, ep_idx)
        except FileNotFoundError as e:
            print(f"  skip (not found: {e})")
            continue
        trajectories = {r: [] for r in ROBOT_IDS}
        max_frames = len(data["robot0"]["poses"])
        print(f" {max_frames} frames", end="", flush=True)
        for frame_idx in range(max_frames):
            log_frame(frame_idx, data, trajectories, global_frame)
            global_frame += 1
        print("  ✓")

    print(f"\nDone — {global_frame} frames logged.")
    if args.save:
        print(f"Saved to: {args.save}")
    else:
        print("Rerun viewer opened. Use the timeline slider to scrub frames.")


if __name__ == "__main__":
    main()
