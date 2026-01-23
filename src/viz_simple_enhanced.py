import cv2
import numpy as np
from viz_vb_data import *

class SimpleEnhancedVisualizer(CombinedVisualizer):
    """简化的增强版可视化"""
    
    def __init__(self, replay_buffer, episodes, record_mode=False, record_episode=0, 
                 output_video=None, record_fps=30, continue_after_record=False):
        # 先设置window_name，再调用父类初始化
        self.window_name = "GenRobot.AI - Enhanced Monitor"
        
        # 调用父类初始化
        super().__init__(replay_buffer, episodes, record_mode, record_episode, 
                        output_video, record_fps, continue_after_record)
    
    def create_styled_frame(self, frame_idx):
        """添加GenRobot风格的装饰"""
        # 使用原有的渲染
        self.frame_idx = frame_idx
        pc_images, world_image = self.get_frame_images()
        original_frame = create_combined_image(self.data, frame_idx, pc_images, world_image)
        
        if original_frame is None:
            return None
        
        # 添加深色边框
        border = 5
        h, w = original_frame.shape[:2]
        styled = np.zeros((h + border * 2 + 60, w + border * 2, 3), dtype=np.uint8)
        styled[:] = [20, 25, 35]  # 深色背景
        
        # 添加顶部标题栏
        cv2.rectangle(styled, (0, 0), (styled.shape[1], 50), (15, 20, 30), -1)
        cv2.putText(styled, "Gen", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (59, 130, 246), 2, cv2.LINE_AA)
        cv2.putText(styled, "Robot.AI", (85, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 状态指示灯（闪烁效果）
        pulse = int((frame_idx % 30) / 30 * 100) + 100
        cv2.circle(styled, (250, 25), 6, (pulse, 255, pulse), -1)
        cv2.putText(styled, "Monitor", (270, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        
        # 右侧信息
        ep_id = self.episodes[self.ep_idx]
        total_eps = len(self.episodes)
        info = f"Episode {ep_id}/{total_eps} | Frame {frame_idx}"
        cv2.putText(styled, info, (w - 350, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        
        # 放置原始帧
        styled[50+border:50+border+h, border:border+w] = original_frame
        
        # 底部状态栏
        bottom_y = styled.shape[0] - 55
        cv2.rectangle(styled, (0, bottom_y), (styled.shape[1], styled.shape[0]), 
                     (15, 20, 30), -1)
        
        # 机器人位姿信息
        info_x = 20
        for r in ROBOT_IDS:
            poses = self.data[f'robot{r}'].get('poses', [])
            if poses and frame_idx < len(poses):
                pos = poses[frame_idx][:3, 3]
                color = (100, 100, 255) if r == 0 else (100, 255, 100)
                pos_text = f"R{r}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                cv2.putText(styled, pos_text, (info_x, bottom_y + 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                info_x += 280
        
        # 控制提示
        controls = "[A/D]Frame [W/S]Episode [P]Play [R]Reset [C]Shot [Q]Quit"
        cv2.putText(styled, controls, (info_x, bottom_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
        
        return styled
    
    def render_frame(self, frame_idx):
        """渲染帧"""
        return self.create_styled_frame(frame_idx)
    
    def print_help(self):
        """打印帮助信息"""
        print("\n" + "=" * 70)
        print("  🎮 GenRobot.AI Enhanced Visualizer")
        print("=" * 70)
        print("  控制键:")
        print("    A/D      - 上一帧/下一帧")
        print("    W/S      - 上一个/下一个 Episode")
        print("    P        - 自动播放开关")
        print("    R        - 重置相机视角")
        print("    C        - 截图保存")
        print("    Q        - 退出程序")
        print("=" * 70 + "\n")
    
    def run(self):
        """运行可视化（添加自动播放）"""
        auto_play = False
        self.print_help()
        
        while True:
            frame = self.render_frame(self.frame_idx)
            
            if frame is not None:
                # 如果自动播放，显示标识
                if auto_play:
                    cv2.putText(frame, "▶ AUTO", (frame.shape[1] - 100, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2, cv2.LINE_AA)
                
                cv2.imshow(self.window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # 自动播放逻辑
            if auto_play:
                max_f = len(self.data['robot0']['poses'])
                if self.frame_idx < max_f - 1:
                    self.frame_idx += 1
                else:
                    auto_play = False
                    print("⏸️  到达末尾，自动播放停止")
            
            key = cv2.waitKey(30) & 0xFF
            
            if key in [ord('d'), ord('D')]:
                max_f = len(self.data['robot0']['poses'])
                if self.frame_idx < max_f - 1:
                    self.frame_idx += 1
                elif self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
                    auto_play = False
            elif key in [ord('a'), ord('A')]:
                if self.frame_idx > 0:
                    self.frame_idx -= 1
                    auto_play = False
            elif key in [ord('w'), ord('W')]:
                if self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
                    auto_play = False
            elif key in [ord('s'), ord('S')]:
                if self.ep_idx > 0:
                    self.ep_idx -= 1
                    self.load_episode()
                    auto_play = False
            elif key in [ord('p'), ord('P')]:
                auto_play = not auto_play
                status = "开启" if auto_play else "关闭"
                icon = "▶️" if auto_play else "⏸️"
                print(f"{icon}  自动播放: {status}")
            elif key in [ord('r'), ord('R')]:
                self.setup_camera_params()
                print("📷 重置视角")
            elif key in [ord('c'), ord('C')]:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ep_id = self.episodes[self.ep_idx]
                filename = f"screenshot_ep{ep_id}_frame{self.frame_idx}_{timestamp}.png"
                if frame is not None:
                    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"📸 截图保存: {filename}")
            elif key in [ord('q'), ord('Q')]:
                print("\n👋 退出程序\n")
                break
        
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🤖 GenRobot.AI Enhanced Visualizer - VR Robot Teleoperation Data Viewer')
    parser.add_argument('zarr_path', nargs='?', 
                       default='data/_0115_bi_pick_and_place_2ver.zarr.zip',
                       help='Path to zarr.zip data file')
    parser.add_argument('--record', type=lambda x: x.lower() == 'true', default=False,
                       help='Enable video recording mode')
    parser.add_argument('--record_episode', type=int, default=1,
                       help='Episode number to record')
    parser.add_argument('--output_video', type=str, default=None,
                       help='Output video filename')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video recording FPS')
    parser.add_argument('--continue_after_record', type=lambda x: x.lower() == 'true', 
                       default=True, help='Continue interactive mode after recording')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path):
        print(f"\n❌ 错误: 找不到文件")
        print(f"   路径: {args.zarr_path}\n")
        return
    
    print(f"\n📂 加载数据: {args.zarr_path}")
    
    store = ZipStore(args.zarr_path, mode='r')
    try:
        root = zarr.open_group(store=store, mode='r')
        rb = ReplayBuffer.create_from_group(root)
        print(f"✅ 加载成功")
        print(f"   总帧数: {rb.n_steps:,}")
        print(f"   Episodes: {rb.n_episodes}")
        
        if args.record:
            if args.record_episode >= rb.n_episodes:
                print(f"\n❌ 错误: Episode {args.record_episode} 超出范围")
                print(f"   可用范围: 0 ~ {rb.n_episodes - 1}\n")
                return
            
            if args.output_video is None:
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                args.output_video = f"genrobot_ep{args.record_episode}_{ts}.mp4"
            
            print(f"🎬 录制模式启动")
            print(f"   Episode: {args.record_episode}")
            print(f"   输出: {args.output_video}")

        SimpleEnhancedVisualizer(
            rb, 
            np.arange(rb.n_episodes), 
            args.record, 
            args.record_episode,
            args.output_video, 
            args.fps, 
            args.continue_after_record
        )
    except Exception as e:
        print(f"\n❌ 发生错误: {e}\n")
        import traceback
        traceback.print_exc()
    finally:
        store.close()

if __name__ == "__main__":
    main()
