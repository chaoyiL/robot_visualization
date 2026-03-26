# Robot Visualization

VR双臂机器人遥操作数据可视化工具，基于 [Rerun](https://rerun.io) 构建。

支持 LeRobot v2.1 数据集格式，提供：
- 交互式 3D 世界视图（EEF 位姿、轨迹、夹爪 / 控制器网格）
- 相机 & 触觉传感器图像面板
- EEF 位置 & 夹爪宽度时间序列曲线
- 时间轴自由拖动，支持跨 episode 浏览

## 安装

```bash
git clone https://github.com/chaoyiL/robot_visualization.git
cd robot_visualization
pip install -r requirements.txt
```

主要依赖：`rerun-sdk >= 0.16`、`pyarrow`、`trimesh`、`scipy`、`opencv-python`

## 使用

```bash
python src/viz_rerun.py /path/to/lerobot_dataset
```

运行后 Rerun viewer 会自动打开，数据实时流入。

### 常用选项

| 选项 | 说明 |
|------|------|
| `-e 3` | 只加载第 3 个 episode |
| `-e 0 3 7` | 加载第 0、3、7 个 episode |
| `-e 0-10` | 加载第 0 到 10 个 episode（含两端） |
| `-e 0-5 8 12-15` | 混合写法 |
| `--save out.rrd` / `-s out.rrd` | 保存为 `.rrd` 文件，不打开 viewer |

```bash
# 单个 episode
python src/viz_rerun.py /path/to/dataset -e 0

# 多个不连续 episode
python src/viz_rerun.py /path/to/dataset -e 0 3 7

# 范围
python src/viz_rerun.py /path/to/dataset -e 0-10

# 混合
python src/viz_rerun.py /path/to/dataset -e 0-5 8 12-15

# 保存录制文件，之后用 rerun 打开
python src/viz_rerun.py /path/to/dataset --save output.rrd
rerun output.rrd
```

## 数据格式

LeRobot v2.1 目录结构：

```
dataset/
├── meta/
│   ├── info.json          # 数据集元信息
│   └── episodes.jsonl
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        ├── episode_000001.parquet
        └── ...
```

每个 parquet 文件包含以下列：

| 列名 | 说明 |
|------|------|
| `observation.state` | 20 维状态向量（双臂 EEF 位姿 + 夹爪宽度） |
| `observation.images.camera0/1` | 双臂腕部相机（224×224 RGB） |
| `observation.images.tactile_left/right_0/1` | 四路触觉传感器图像 |
| `actions` | 20 维动作向量 |

## Viewer 界面

```
┌────────────────────────────┬────────────────────────────────┐
│                            │  R0-Visual │ R0-L-Tact │ R0-R  │
│       3D World             ├────────────┴───────────┴───────┤
│  (EEF + 轨迹 + 夹爪 mesh) │  R1-Visual │ R1-L-Tact │ R1-R  │
│                            ├────────────────────────────────┤
├──────────────┬─────────────┤       Gripper Widths           │
│  Robot 0 XYZ│  Robot 1 XYZ│       (timeseries)             │
└──────────────┴─────────────┴────────────────────────────────┘
```

拖动底部时间轴即可在帧之间自由跳转，3D 视图支持鼠标旋转 / 缩放 / 平移。
