"""
data_collection.py - 自动采集滞后数据用于 AI 训练
"""

import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

class DataCollector:
    """自动数据采集器"""
    
    def __init__(self, stage, state, get_tip_func):
        self.stage = stage
        self.state = state
        self.get_tip = get_tip_func
        self.data = []  # 存储每次采集的数据
        
    def collect_single_scan(self, start_x, end_x, steps, wait_time=0.05, label=""):
        """
        采集单次来回扫描的数据
        
        参数:
            start_x: 起始 X 位置 (um)
            end_x: 结束 X 位置 (um)
            steps: 步数
            wait_time: 每步等待时间 (s)
            label: 数据标签
        """
        print(f"\n--- Collecting: {label} ---")
        print(f"  Range: {start_x} -> {end_x} -> {start_x}")
        print(f"  Steps: {steps}, Wait time: {wait_time}s")
        
        # 重置定位器状态
        self.stage.reset(self.state.x, self.state.y)
        
        # 生成命令序列：上升 + 下降
        cmd_asc = np.linspace(start_x, end_x, steps)
        cmd_desc = np.linspace(end_x, start_x, steps)
        
        cmd_list = []
        actual_list = []
        dir_list = []
        
        # 上升阶段
        for cmd in cmd_asc:
            actual, _ = self.stage.move_to(cmd, 0)
            cmd_list.append(cmd)
            actual_list.append(actual)
            dir_list.append('ascending')
            time.sleep(wait_time)
        
        # 下降阶段
        for cmd in cmd_desc:
            actual, _ = self.stage.move_to(cmd, 0)
            cmd_list.append(cmd)
            actual_list.append(actual)
            dir_list.append('descending')
            time.sleep(wait_time)
        
        # 保存数据
        scan_data = {
            'label': label,
            'cmd_x': cmd_list,
            'actual_x': actual_list,
            'direction': dir_list
        }
        
        self.data.append(scan_data)
        print(f"  Collected {len(cmd_list)} points")
        
        return scan_data
    
    def collect_multi_configurations(self, configs, base_wait_time=0.05):
        """
        采集多种配置下的数据
        
        configs: 配置列表，每个配置包含:
            - start_x: 起始位置
            - end_x: 结束位置
            - steps: 步数
            - speed_factor: 速度因子 (wait_time = base_wait_time / speed_factor)
            - label: 标签
        """
        all_data = []
        for config in configs:
            wait_time = base_wait_time / config.get('speed_factor', 1)
            data = self.collect_single_scan(
                start_x=config['start_x'],
                end_x=config['end_x'],
                steps=config['steps'],
                wait_time=wait_time,
                label=config['label']
            )
            all_data.append(data)
        return all_data
    
    def save_all_to_csv(self, output_dir="collected_data"):
        """保存所有采集的数据到 CSV 文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        all_rows = []
        for scan_data in self.data:
            for cmd, actual, direction in zip(
                scan_data['cmd_x'], 
                scan_data['actual_x'],
                scan_data['direction']
            ):
                all_rows.append({
                    'label': scan_data['label'],
                    'cmd_x': cmd,
                    'actual_x': actual,
                    'direction': direction,
                    'error': actual - cmd
                })
        
        df = pd.DataFrame(all_rows)
        filename = os.path.join(output_dir, f"hysteresis_data_{timestamp}.csv")
        df.to_csv(filename, index=False)
        print(f"\nData saved to: {filename}")
        print(f"Total records: {len(df)}")
        
        # 打印统计信息
        self._print_statistics(df)
        
        return filename
    
    def _print_statistics(self, df):
        """打印数据统计信息"""
        print("\n" + "="*50)
        print("Data Statistics")
        print("="*50)
        print(f"Total records: {len(df)}")
        print(f"Command range: [{df['cmd_x'].min():.1f}, {df['cmd_x'].max():.1f}]")
        print(f"Actual range: [{df['actual_x'].min():.1f}, {df['actual_x'].max():.1f}]")
        print(f"Error range: [{df['error'].min():.2f}, {df['error'].max():.2f}]")
        print(f"Mean error: {df['error'].mean():.2f} um")
        print(f"Std error: {df['error'].std():.2f} um")
        
        # 按方向分组统计
        print("\nBy direction:")
        for direction, group in df.groupby('direction'):
            print(f"  {direction}: mean error = {group['error'].mean():.2f} um")
    
    def plot_collected_data(self):
        """绘制采集的数据"""
        if not self.data:
            print("No data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 子图1：所有数据的滞后环
        plt.subplot(2, 2, 1)
        for scan_data in self.data:
            cmd = scan_data['cmd_x']
            actual = scan_data['actual_x']
            plt.plot(cmd, actual, '-', alpha=0.5, label=scan_data['label'])
        plt.plot([0, max(cmd)], [0, max(cmd)], 'k--', label='Ideal')
        plt.xlabel('Command (um)')
        plt.ylabel('Actual (um)')
        plt.title('All Hysteresis Loops')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 子图2：误差 vs 命令
        plt.subplot(2, 2, 2)
        for scan_data in self.data:
            cmd = scan_data['cmd_x']
            error = np.array(scan_data['actual_x']) - np.array(cmd)
            plt.plot(cmd, error, '-', alpha=0.5, label=scan_data['label'])
        plt.xlabel('Command (um)')
        plt.ylabel('Error (um)')
        plt.title('Error vs Command')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 子图3：误差分布直方图
        plt.subplot(2, 2, 3)
        all_errors = []
        for scan_data in self.data:
            error = np.array(scan_data['actual_x']) - np.array(scan_data['cmd_x'])
            all_errors.extend(error)
        plt.hist(all_errors, bins=50, alpha=0.7, color='blue')
        plt.xlabel('Error (um)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # 子图4：不同速度对比（仅上升段）
        plt.subplot(2, 2, 4)
        for scan_data in self.data:
            if 'speed' in scan_data['label'].lower():
                cmd = scan_data['cmd_x']
                actual = scan_data['actual_x']
                n = len(cmd) // 2
                plt.plot(cmd[:n], actual[:n], '-', label=scan_data['label'])
        plt.xlabel('Command (um)')
        plt.ylabel('Actual (um)')
        plt.title('Speed Comparison (Ascending)')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def run_data_collection_demo():
    """独立运行数据采集演示"""
    print("="*60)
    print("Hysteresis Data Collection Demo")
    print("="*60)
    
    # 创建必要的对象（需根据实际情况调整）
    from afm_state import AFMState
    from sample_generation import sample, width_um, height_um
    from hysteresis import NanoPositioner
    from afm_utils import get_tip_position
    import matplotlib.pyplot as plt
    
    state = AFMState(sample, width_um, height_um)
    stage = NanoPositioner(log_file="temp_log.csv")
    
    # 模拟 get_tip 函数（实际应返回真实 tip 坐标）
    def dummy_get_tip():
        return state.x + 421, state.y + 315
    
    collector = DataCollector(stage, state, dummy_get_tip)
    
    # 定义采集配置
    configs = [
        {'start_x': 200, 'end_x': 600, 'steps': 80, 'speed_factor': 1, 'label': 'Range_400um_Slow'},
        {'start_x': 200, 'end_x': 600, 'steps': 80, 'speed_factor': 2, 'label': 'Range_400um_Fast'},
        {'start_x': 200, 'end_x': 1000, 'steps': 120, 'speed_factor': 1, 'label': 'Range_800um_Slow'},
        {'start_x': 200, 'end_x': 1000, 'steps': 120, 'speed_factor': 2, 'label': 'Range_800um_Fast'},
        {'start_x': 200, 'end_x': 1400, 'steps': 160, 'speed_factor': 1, 'label': 'Range_1200um_Slow'},
        {'start_x': 200, 'end_x': 1400, 'steps': 160, 'speed_factor': 2, 'label': 'Range_1200um_Fast'},
    ]
    
    # 采集数据
    collector.collect_multi_configurations(configs, base_wait_time=0.05)
    
    # 保存数据
    filename = collector.save_all_to_csv()
    
    # 绘制数据
    collector.plot_collected_data()
    
    print(f"\nData collection complete! File: {filename}")


if __name__ == "__main__":
    run_data_collection_demo()
