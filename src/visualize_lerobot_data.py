#!/usr/bin/env python3
"""
LeRobot数据可视化适配器（修复版）

直接使用 HuggingFace datasets 加载，绕过 LeRobotDataset 的 bug

使用方法:
python src/visualize_lerobot_data.py --repo_id /home/rvsa/codehub/robot_visualization/data/datasets--liuchaoyi--lerobot_data_0118/snapshots/13c4d0afed2466af0eec729bad1e8ccaa9083ca7 --pose_mode action
python src/visualize_lerobot_data.py --repo_id /home/rvsa/codehub/robot_visualization/data/example --pose_mode action
"""

import sys
import os
import numpy as np
import argparse
import json
import scipy.spatial.transform as st

# 直接使用 HuggingFace datasets
try:
    from datasets import load_dataset
except ImportError:
    print("错误: 请先安装 datasets")
    print("安装命令: pip install datasets")
    sys.exit(1)

from replay_buffer import ReplayBuffer
from viz_3d_enhanced import Enhanced3DVisualizer

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def rot6d_to_mat(a1, a2):
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-1)
    return out

def action9_to_mat(action9: np.ndarray):
    if action9.shape[0] != 9:
        raise ValueError(f"action length must be 9, got {action9.shape[0]}")
    t = action9[:3].astype(np.float32)
    c1 = action9[3:6].astype(np.float32)
    c2 = action9[6:9].astype(np.float32)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot6d_to_mat(c1, c2)
    mat[:3, 3] = t
    return mat

class _LazyColumn:
    """按需读取的列视图，避免一次性加载全量数据。"""

    def __init__(self, adapter, key):
        self.adapter = adapter
        self.key = key

    def __len__(self):
        return self.adapter.n_steps

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.adapter.n_steps)
            return [self[i] for i in range(start, stop, step)]
        return self.adapter._get_value(self.key, int(idx))


class LeRobotReplayBufferAdapter:
    """LeRobot到ReplayBuffer的适配器（修复版）"""
    
    def __init__(self, repo_id: str, pose_mode: str = 'state'):
        print(f"\n{'='*70}")
        print(f"加载LeRobot数据集: {repo_id}")
        print(f"{'='*70}\n")
        if pose_mode not in ('state', 'action'):
            raise ValueError("pose_mode must be 'state' or 'action'")
        self.pose_mode = pose_mode
        
        # 处理本地路径
        repo_id = self._resolve_path(repo_id)
        
        try:
            # 直接用 datasets 库加载，绕过 LeRobotDataset 的 bug
            print("  使用 HuggingFace datasets 加载...")
            dataset_dict = load_dataset(repo_id)
            
            if 'train' in dataset_dict:
                self.dataset = dataset_dict['train']
            else:
                # 如果没有 train split，使用第一个可用的
                self.dataset = dataset_dict[list(dataset_dict.keys())[0]]
            
            self.repo_id = repo_id
            
            # 加载 meta 信息
            meta_path = os.path.join(repo_id, 'meta', 'info.json')
            with open(meta_path, 'r') as f:
                self.meta_info = json.load(f)
            
            # 加载 episodes 信息
            episodes_path = os.path.join(repo_id, 'meta', 'episodes.jsonl')
            self.episodes_info = []
            with open(episodes_path, 'r') as f:
                for line in f:
                    self.episodes_info.append(json.loads(line))
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            raise
        
        print(f"✓ 数据集信息:")
        print(f"  - Episodes: {self.meta_info['total_episodes']}")
        print(f"  - 总帧数: {self.meta_info['total_frames']}")
        print(f"  - FPS: {self.meta_info['fps']}")
        print(f"  - Features: {len(self.meta_info['features'])}")
        
        self._create_buffer_structure()
    
    def _resolve_path(self, path: str) -> str:
        """解析路径"""
        if path.startswith('~'):
            path = os.path.expanduser(path)
        
        if os.path.exists(path):
            path = os.path.abspath(path)
            print(f"  使用本地路径: {path}")
        
        return path
    
    def _create_buffer_structure(self):
        """创建ReplayBuffer兼容的懒加载结构（不预加载全量帧）"""
        self.n_episodes = self.meta_info['total_episodes']
        self.n_steps = self.meta_info['total_frames']
        self.dataset_columns = set(self.dataset.column_names)
        self._column_cache = {}
        self._last_idx = None
        self._last_item = None
        self._action_episode_cache = {}

        self._state_keys = {
            'robot0_eef_pos',
            'robot0_eef_rot_axis_angle',
            'robot0_gripper_width',
            'robot1_eef_pos',
            'robot1_eef_rot_axis_angle',
            'robot1_gripper_width',
        }

        self._column_key_map = {
            'camera0_rgb': 'observation.images.camera0',
            'camera1_rgb': 'observation.images.camera1',
            'camera0_left_tactile': 'observation.images.tactile_left_0',
            'camera0_right_tactile': 'observation.images.tactile_right_0',
            'camera1_left_tactile': 'observation.images.tactile_left_1',
            'camera1_right_tactile': 'observation.images.tactile_right_1',
        }

        self._episode_ends = []
        total_frames = 0
        for ep in sorted(self.episodes_info, key=lambda x: int(x['episode_index'])):
            total_frames += int(ep['length'])
            self._episode_ends.append(total_frames)

        self._available_keys = set()
        if 'observation.state' in self.dataset_columns:
            self._available_keys.update(self._state_keys)
        if 'actions' in self.dataset_columns:
            self._available_keys.update({'robot0_action', 'robot1_action'})
        for key, col in self._column_key_map.items():
            if col in self.dataset_columns:
                self._available_keys.add(key)

        # 同时允许直接访问原始列名，兼容不同上层调用习惯
        self._available_keys.update(self.dataset_columns)

        print("\n✓ 使用懒加载模式（按需读取帧数据）")
        print(f"✓ 位姿来源模式: {self.pose_mode}")

    def _get_episode_bounds_by_idx(self, idx: int):
        if idx < 0:
            idx += self.n_steps
        if idx < 0 or idx >= self.n_steps:
            raise IndexError(f"index {idx} out of range")
        ep_end = self._episode_ends[0]
        ep_start = 0
        for end in self._episode_ends:
            if idx < end:
                ep_end = end
                break
            ep_start = end
        return ep_start, ep_end

    def _build_action_episode_cache(self, idx: int):
        ep_start, ep_end = self._get_episode_bounds_by_idx(idx)
        cache_key = (ep_start, ep_end)
        if cache_key in self._action_episode_cache:
            return self._action_episode_cache[cache_key]

        if 'observation.state' not in self.dataset_columns or 'actions' not in self.dataset_columns:
            raise KeyError("action模式需要 observation.state 和 actions 两列")

        ep_len = ep_end - ep_start
        robot0_poses = [None] * ep_len
        robot1_poses = [None] * ep_len
        robot0_gripper = np.zeros((ep_len,), dtype=np.float32)
        robot1_gripper = np.zeros((ep_len,), dtype=np.float32)

        first_state = np.asarray(self.dataset[ep_start]['observation.state'], dtype=np.float32)
        rel_0to1_mat = pose_to_mat(first_state[14:20].copy())
        robot0_pose0 = pose_to_mat(first_state[0:6].copy())
        robot1_pose0 = robot0_pose0 @ np.linalg.inv(rel_0to1_mat)
        robot0_poses[0] = robot0_pose0.astype(np.float32)
        robot1_poses[0] = robot1_pose0.astype(np.float32)

        first_action = np.asarray(self.dataset[ep_start]['actions'], dtype=np.float32)
        robot0_gripper[0] = first_action[9]
        robot1_gripper[0] = first_action[19]

        for local_i in range(1, ep_len):
            abs_i = ep_start + local_i
            action = np.asarray(self.dataset[abs_i]['actions'], dtype=np.float32)

            # 数据定义：当前帧 -> 上一帧
            robot0_cur_to_prev = action9_to_mat(action[0:9].copy())
            robot1_cur_to_prev = action9_to_mat(action[10:19].copy())

            robot0_poses[local_i] = (robot0_poses[local_i - 1] @ robot0_cur_to_prev).astype(np.float32)
            robot1_poses[local_i] = (robot1_poses[local_i - 1] @ robot1_cur_to_prev).astype(np.float32)

            robot0_gripper[local_i] = action[9]
            robot1_gripper[local_i] = action[19]

        cache = {
            'ep_start': ep_start,
            'ep_end': ep_end,
            'robot0_poses': robot0_poses,
            'robot1_poses': robot1_poses,
            'robot0_gripper': robot0_gripper,
            'robot1_gripper': robot1_gripper,
        }
        self._action_episode_cache[cache_key] = cache
        return cache

    def _get_item(self, idx: int):
        if idx < 0:
            idx += self.n_steps
        if idx < 0 or idx >= self.n_steps:
            raise IndexError(f"index {idx} out of range")
        if self._last_idx != idx:
            self._last_item = self.dataset[idx]
            self._last_idx = idx
        return self._last_item

    def _get_value(self, key: str, idx: int):
        item = self._get_item(idx)

        if key in self.dataset_columns:
            return item[key]

        if key in self._state_keys:
            if self.pose_mode == 'action':
                cache = self._build_action_episode_cache(idx)
                local_i = idx - cache['ep_start']
                robot0_pose = mat_to_pose(cache['robot0_poses'][local_i])
                robot1_pose = mat_to_pose(cache['robot1_poses'][local_i])

                if key == 'robot0_eef_pos':
                    return robot0_pose[0:3].copy()
                if key == 'robot0_eef_rot_axis_angle':
                    return robot0_pose[3:6].copy()
                if key == 'robot0_gripper_width':
                    return np.array([cache['robot0_gripper'][local_i]], dtype=np.float32)
                if key == 'robot1_eef_pos':
                    return robot1_pose[0:3].copy()
                if key == 'robot1_eef_rot_axis_angle':
                    return robot1_pose[3:6].copy()
                if key == 'robot1_gripper_width':
                    return np.array([cache['robot1_gripper'][local_i]], dtype=np.float32)
            else:
                state = np.asarray(item['observation.state'], dtype=np.float32)
                rel_pose = state[14:20].copy()
                rel_0to1_mat = pose_to_mat(rel_pose)

                robot0_eef_mat = pose_to_mat(state[0:6].copy())
                robot1_eef_mat = robot0_eef_mat @ np.linalg.inv(rel_0to1_mat)
                robot1_eef_pose = mat_to_pose(robot1_eef_mat)
                
                if key == 'robot0_eef_pos':
                    return state[0:3].copy()
                if key == 'robot0_eef_rot_axis_angle':
                    return state[3:6].copy()
                if key == 'robot0_gripper_width':
                    return np.array([state[6]], dtype=np.float32)
                if key == 'robot1_eef_pos':
                    return robot1_eef_pose[0:3].copy()
                if key == 'robot1_eef_rot_axis_angle':
                    return robot1_eef_pose[3:6].copy()
                if key == 'robot1_gripper_width':
                    return np.array([state[13]], dtype=np.float32)

        if key in ('robot0_action', 'robot1_action'):
            action = np.asarray(item['actions'], dtype=np.float32)
            if key == 'robot0_action':
                return action[0:9].copy()
            return action[10:19].copy()

        column_name = self._column_key_map.get(key)
        if column_name and column_name in item:
            return np.asarray(item[column_name], dtype=np.uint8)

        raise KeyError(f"Key '{key}' not found")
    
    def get_episode_slice(self, episode_idx: int):
        if episode_idx == 0:
            start = 0
        else:
            start = self._episode_ends[episode_idx - 1]
        end = self._episode_ends[episode_idx]
        return slice(start, end)
    
    def keys(self):
        return self._available_keys
    
    def __getitem__(self, key):
        if key not in self._available_keys:
            raise KeyError(f"Key '{key}' not found")
        if key not in self._column_cache:
            self._column_cache[key] = _LazyColumn(self, key)
        return self._column_cache[key]


def main():
    parser = argparse.ArgumentParser(
        description='LeRobot数据可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化数据集
  python src/visualize_lerobot_data.py --repo_id ~/Downloads/_test_127
  
  # 录制视频
  python src/visualize_lerobot_data.py --repo_id ~/Downloads/_test_127 \\
      --record --record_episode 0 --output demo.mp4
        """
    )
    
    parser.add_argument('--repo_id', type=str, required=True, 
                       help='数据集路径或repo_id')
    parser.add_argument('--episode', type=int, default=0, 
                       help='起始episode')
    parser.add_argument('--record', '-r', action='store_true', 
                       help='录制视频')
    parser.add_argument('--record_episode', '-e', type=int, default=0, 
                       help='录制的episode')
    parser.add_argument('--output_video', '-o', type=str, default=None, 
                       help='输出视频文件名')
    parser.add_argument('--fps', type=int, default=30, 
                       help='视频FPS')
    parser.add_argument('--continue_after_record', '-c', action='store_true', 
                       help='录制后继续')
    parser.add_argument('--pose_mode', type=str, default='state', choices=['state', 'action'],
                       help='位姿来源模式: state(原始状态) 或 action(动作增量重建)')
    
    args = parser.parse_args()
    
    try:
        adapter = LeRobotReplayBufferAdapter(args.repo_id, pose_mode=args.pose_mode)
        
        if args.record and args.output_video is None:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_repo_id = os.path.basename(args.repo_id)
            args.output_video = f"lerobot_ep{args.record_episode}_{safe_repo_id}_{ts}.mp4"
        
        print(f"{'='*70}")
        print("启动可视化器...")
        print(f"{'='*70}\n")
        
        episodes = np.arange(adapter.n_episodes)
        
        visualizer = Enhanced3DVisualizer(
            replay_buffer=adapter,
            episodes=episodes,
            record_mode=args.record,
            record_episode=args.record_episode,
            output_video=args.output_video,
            record_fps=args.fps,
            continue_after_record=args.continue_after_record
        )
        
        print("\n✓ 完成")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()