# Robot Visualization

VR双臂机器人遥操作数据可视化工具

## 安装
```bash
git clone https://github.com/Jerryzhang258/robot_visualization.git
cd robot_visualization
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 使用
```bash
# 交互模式
python src/viz_3d_enhanced.py data/your_data.zarr.zip --record False

# 录制视频
python src/viz_3d_enhanced.py data/your_data.zarr.zip --record True --record_episode 1 --output_video demo.mp4
```

## 控制

- `A/D` - 前后帧
- `W/S` - 切换Episode  
- `P` - 自动播放
- `1-5` - 调速 (0.25x, 0.5x, 1x, 2x, 5x)
- `Q` - 退出

## 数据格式

Zarr格式 (.zarr.zip)，包含：
- robot0/1_eef_pos (位置)
- robot0/1_gripper_width (夹爪)
- robot0/1_visual (相机)
- robot0/1_left/right_tactile (触觉)
