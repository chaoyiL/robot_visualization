#!/usr/bin/env python3
"""
Rerun-based visualization for EricChen06/raw_0407 (LeRobot v3.0, single arm).

Dataset differences from v2.1:
  - Single robot arm (7-DOF state: [x,y,z,rx,ry,rz, gripper])
  - Multiple episodes per parquet file: data/chunk-000/file-000.parquet
  - 6 image streams: camera0, camera1, tactile_left_0/1, tactile_right_0/1

Usage:
    # Download dataset first (requires huggingface_hub):
    python src/viz_raw_0407.py /path/to/raw_0407 --episode 0
    python src/viz_raw_0407.py /path/to/raw_0407 -e 0-5
    python src/viz_raw_0407.py /path/to/raw_0407 --save out.rrd

    # Or download on-the-fly from HuggingFace:
    python src/viz_raw_0407.py --hf EricChen06/raw_0407 -e 0
"""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation
import rerun as rr
import rerun.blueprint as rrb

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

ROBOT_COLOR = [60, 160, 220]   # single arm: blue


# ── mesh loading ──────────────────────────────────────────────────────────────
def _load_trimesh(path, scale=1.0, extra_transform=None):
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


def load_meshes():
    """Load gripper + controller STLs."""
    meshes = {}

    # ── gripper ───────────────────────────────────────────────────────────────
    gripper_path = os.path.join(_SRC, "meshes", "夹爪.STL")
    if os.path.exists(gripper_path):
        try:
            import trimesh
            base = trimesh.load(gripper_path)
            base.apply_scale(0.001)
            center = (base.bounds[0] + base.bounds[1]) / 2
            base.apply_translation(-center)
            rot = np.eye(4)
            rot[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
            base.apply_transform(rot)

            meshes["gripper_left"] = {
                "vertices": base.vertices.astype(np.float32),
                "faces":    base.faces.astype(np.int32),
                "normals":  base.vertex_normals.astype(np.float32),
            }
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


# ── static scene ──────────────────────────────────────────────────────────────
def log_static_geometry(meshes):
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/robot/eef/axes",
        rr.Arrows3D(
            vectors=np.eye(3, dtype=np.float32) * 0.05,
            origins=np.zeros((3, 3), dtype=np.float32),
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )

    # ── controller (right hand = robot arm) ───────────────────────────────────
    ctrl_key = "controller_right"
    if ctrl_key in meshes:
        m = meshes[ctrl_key]
        n = len(m["vertices"])
        ctrl_rot_quat = Rotation.from_euler("y", 90, degrees=True).as_quat()
        rr.log(
            "world/robot/eef/controller",
            rr.Transform3D(
                translation=[0, 0, 0.05],
                quaternion=rr.Quaternion(xyzw=ctrl_rot_quat),
            ),
            static=True,
        )
        rr.log(
            "world/robot/eef/controller/mesh",
            rr.Mesh3D(
                vertex_positions=m["vertices"],
                triangle_indices=m["faces"],
                vertex_normals=m.get("normals"),
                vertex_colors=np.full((n, 3), [100, 100, 200], dtype=np.uint8),
            ),
            static=True,
        )

    # ── gripper fingers ───────────────────────────────────────────────────────
    for side in ("left", "right"):
        key = f"gripper_{side}"
        if key in meshes:
            m = meshes[key]
            n = len(m["vertices"])
            rr.log(
                f"world/robot/eef/gripper/{side}/mesh",
                rr.Mesh3D(
                    vertex_positions=m["vertices"],
                    triangle_indices=m["faces"],
                    vertex_normals=m.get("normals"),
                    vertex_colors=np.full((n, 3), [180, 180, 180], dtype=np.uint8),
                ),
                static=True,
            )
        sensor_color = [0, 220, 0] if side == "left" else [220, 0, 0]
        rr.log(
            f"world/robot/eef/gripper/{side}/sensor",
            rr.Points3D(positions=[[0, 0, 0]], colors=[sensor_color], radii=[0.012]),
            static=True,
        )


# ── image decoding ────────────────────────────────────────────────────────────
def _decode_image(img_val):
    """
    Decode image from multiple possible storage formats in LeRobot parquet:
      - dict with "bytes" key (JPEG/PNG encoded)
      - bytes directly
      - numpy array [H, W, 3]
    """
    # raw numpy array
    if isinstance(img_val, np.ndarray):
        return img_val.astype(np.uint8) if img_val.dtype != np.uint8 else img_val

    # dict with "bytes" key (JPEG/PNG)
    raw = None
    if isinstance(img_val, dict):
        raw = img_val.get("bytes")
        # also handle {"path": ..., "bytes": ...} or PIL-style {"array": ...}
        if raw is None and "array" in img_val:
            arr = np.asarray(img_val["array"], dtype=np.uint8)
            return arr
    elif isinstance(img_val, (bytes, bytearray)):
        raw = img_val

    if raw is not None:
        import cv2
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return None


# ── LeRobot v3.0 parquet loader ───────────────────────────────────────────────
_EPISODE_INDEX_CACHE = {}

def _load_episode_index(dataset_path):
    """
    Read meta/episodes/chunk-*/file-*.parquet to build a map:
      episode_idx → (chunk_index, file_index)
    Cached after first load.
    """
    if dataset_path in _EPISODE_INDEX_CACHE:
        return _EPISODE_INDEX_CACHE[dataset_path]

    import pandas as pd, glob as _glob
    ep_dir = os.path.join(dataset_path, "meta", "episodes")
    files = sorted(_glob.glob(os.path.join(ep_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No episode index files found in {ep_dir}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=["episode_index", "data/chunk_index", "data/file_index"])
        dfs.append(df)
    idx = pd.concat(dfs, ignore_index=True)
    mapping = {
        int(row["episode_index"]): (int(row["data/chunk_index"]), int(row["data/file_index"]))
        for _, row in idx.iterrows()
    }
    _EPISODE_INDEX_CACHE[dataset_path] = mapping
    return mapping


def _find_episode_file(dataset_path, episode_idx):
    """Return path to the parquet file containing episode_idx."""
    mapping = _load_episode_index(dataset_path)
    if episode_idx not in mapping:
        raise FileNotFoundError(f"Episode {episode_idx} not in episode index")
    chunk_idx, file_idx = mapping[episode_idx]
    fpath = os.path.join(
        dataset_path, "data",
        f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet",
    )
    return fpath


def load_episode(dataset_path, episode_idx):
    """
    Load one episode from LeRobot v3.0 dataset.
    State: 7-DOF [x, y, z, rx, ry, rz, gripper_width]
    """
    import pandas as pd

    fpath = _find_episode_file(dataset_path, episode_idx)
    df = pd.read_parquet(fpath)
    df = df[df["episode_index"] == episode_idx].reset_index(drop=True)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    img_cols = [
        "observation.images.camera0",
        "observation.images.camera1",
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    ]
    available_imgs = [c for c in img_cols if c in df.columns]

    poses, grippers, images = [], [], {c: [] for c in available_imgs}

    for _, row in df.iterrows():
        state = np.asarray(row["observation.state"], dtype=np.float32)
        # 7-DOF: [x, y, z, rx, ry, rz, gripper]
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = state[0:3]
        T[:3, :3] = Rotation.from_rotvec(state[3:6]).as_matrix()
        poses.append(T)
        grippers.append(float(state[6]) if len(state) > 6 else 0.0)

        for col in available_imgs:
            img = _decode_image(row[col])
            if img is not None:
                images[col].append(img)

    return poses, grippers, images


# ── per-frame logging ─────────────────────────────────────────────────────────
def log_frame(frame_idx, poses, grippers, images, trajectory, global_frame):
    rr.set_time("frame", sequence=global_frame)

    if frame_idx < len(poses):
        T      = poses[frame_idx]
        pos    = T[:3, 3]
        quat   = Rotation.from_matrix(T[:3, :3]).as_quat()   # xyzw
        trajectory.append(pos.copy())

        rr.log("world/robot/eef",
               rr.Transform3D(translation=pos, quaternion=rr.Quaternion(xyzw=quat)))

        traj = np.array(trajectory, dtype=np.float32)
        rr.log("world/robot/trajectory",
               rr.LineStrips3D([traj], colors=[ROBOT_COLOR], radii=[0.003]))

        rr.log("timeseries/eef_x", rr.Scalars(float(pos[0])))
        rr.log("timeseries/eef_y", rr.Scalars(float(pos[1])))
        rr.log("timeseries/eef_z", rr.Scalars(float(pos[2])))

    if frame_idx < len(grippers):
        grip_width = float(grippers[frame_idx])
        rr.log("timeseries/gripper_width", rr.Scalars(grip_width))
        offset = max(grip_width * 0.5, 0.03)
        for side, sign in [("left", -1), ("right", 1)]:
            rr.log(f"world/robot/eef/gripper/{side}",
                   rr.Transform3D(translation=[0.02, sign * offset, -0.04]))

    cam_map = {
        "observation.images.camera0":         "cameras/camera0",
        "observation.images.camera1":         "cameras/camera1",
        "observation.images.tactile_left_0":  "cameras/tactile_left_0",
        "observation.images.tactile_right_0": "cameras/tactile_right_0",
        "observation.images.tactile_left_1":  "cameras/tactile_left_1",
        "observation.images.tactile_right_1": "cameras/tactile_right_1",
    }
    for col, path in cam_map.items():
        imgs = images.get(col, [])
        if imgs and frame_idx < len(imgs):
            rr.log(path, rr.Image(imgs[frame_idx]))


# ── blueprint ─────────────────────────────────────────────────────────────────
def make_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(name="3D World", origin="world"),
                rrb.TimeSeriesView(name="EEF & Gripper", origin="timeseries"),
                row_shares=[3, 2],
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Camera 0",   origin="cameras/camera0"),
                    rrb.Spatial2DView(name="Camera 1",   origin="cameras/camera1"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Tact L-0",   origin="cameras/tactile_left_0"),
                    rrb.Spatial2DView(name="Tact R-0",   origin="cameras/tactile_right_0"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Tact L-1",   origin="cameras/tactile_left_1"),
                    rrb.Spatial2DView(name="Tact R-1",   origin="cameras/tactile_right_1"),
                ),
            ),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


# ── episode selection ─────────────────────────────────────────────────────────
def _parse_episodes(spec, n_total):
    if spec is None:
        return list(range(n_total))
    indices = set()
    for token in spec:
        if "-" in token:
            parts = token.split("-")
            start, end = int(parts[0]), int(parts[1])
            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))
    result = sorted(i for i in indices if 0 <= i < n_total)
    out_of_range = sorted(i for i in indices if not (0 <= i < n_total))
    if out_of_range:
        print(f"Warning: out of range (0–{n_total-1}), skipping: {out_of_range}")
    return result


# ── HuggingFace download helper ───────────────────────────────────────────────
def download_from_hf(repo_id, local_dir, episode_indices=None):
    """
    Download required files from HuggingFace using huggingface_hub.
    Downloads meta/ always, plus data/chunk-000/ (all or just needed files).
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading metadata from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["meta/**"],
        ignore_patterns=["*.git*"],
    )

    print("Downloading data files (this may take a while for large datasets)...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["data/chunk-000/**"],
        ignore_patterns=["*.git*"],
    )
    print(f"Downloaded to: {local_dir}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Visualize EricChen06/raw_0407 (LeRobot v3.0) with Rerun.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/viz_raw_0407.py /path/to/raw_0407 -e 0\n"
            "  python src/viz_raw_0407.py /path/to/raw_0407 -e 0-5\n"
            "  python src/viz_raw_0407.py --hf EricChen06/raw_0407 -e 0\n"
            "  python src/viz_raw_0407.py /path/to/raw_0407 --save out.rrd\n"
        ),
    )
    parser.add_argument("dataset", nargs="?", default=None,
                        help="Local dataset directory")
    parser.add_argument("--hf", type=str, default=None,
                        help="HuggingFace repo id (e.g. EricChen06/raw_0407); "
                             "downloads to ~/.cache/raw_0407 if dataset not given")
    parser.add_argument("-e", "--episode", nargs="+", default=None, metavar="EP",
                        help="Episodes: single (-e 3), range (-e 0-10), mixed (-e 0-5 8)")
    parser.add_argument("--save", "-s", type=str, default=None,
                        help="Save to .rrd file instead of spawning viewer")
    args = parser.parse_args()

    if args.hf and args.dataset is None:
        args.dataset = os.path.expanduser(
            f"~/.cache/huggingface/datasets/{args.hf.replace('/', '_')}"
        )
        if not os.path.exists(os.path.join(args.dataset, "meta", "info.json")):
            download_from_hf(args.hf, args.dataset)

    if args.dataset is None:
        parser.error("Provide dataset path or --hf repo_id")

    dataset_path = os.path.abspath(args.dataset)
    info_path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.exists(info_path):
        print(f"Error: not a LeRobot dataset (missing meta/info.json): {dataset_path}")
        sys.exit(1)

    with open(info_path) as f:
        meta = json.load(f)
    n_episodes = meta["total_episodes"]
    print(f"Dataset: {meta.get('total_frames', '?'):,} frames | {n_episodes} episodes "
          f"| codebase_version={meta.get('codebase_version', '?')}")

    ep_indices = _parse_episodes(args.episode, n_episodes)
    if not ep_indices:
        print("No episodes to visualize.")
        sys.exit(0)

    # ── Rerun init ────────────────────────────────────────────────────────────
    blueprint = make_blueprint()
    rr.init("raw_0407_viz", spawn=(args.save is None))
    if args.save:
        rr.save(args.save, default_blueprint=blueprint)
    else:
        rr.send_blueprint(blueprint)

    print("Loading meshes...")
    meshes = load_meshes()
    log_static_geometry(meshes)
    print(f"  gripper: {'✓' if 'gripper_left' in meshes else '✗'}  "
          f"controller: {'✓' if 'controller_right' in meshes else '✗'}")

    global_frame = 0
    for ep_idx in ep_indices:
        rr.set_time("episode", sequence=ep_idx)
        print(f"  ep {ep_idx:4d}:", end="", flush=True)
        try:
            poses, grippers, images = load_episode(dataset_path, ep_idx)
        except Exception as e:
            print(f"  skip ({e})")
            continue

        n_frames = len(poses)
        print(f" {n_frames} frames", end="", flush=True)
        trajectory = []
        for fi in range(n_frames):
            log_frame(fi, poses, grippers, images, trajectory, global_frame)
            global_frame += 1
        print("  ✓")

    print(f"\nDone — {global_frame} frames logged.")
    if args.save:
        print(f"Saved to: {args.save}")
    else:
        print("Rerun viewer opened. Use the timeline to scrub frames.")


if __name__ == "__main__":
    main()
