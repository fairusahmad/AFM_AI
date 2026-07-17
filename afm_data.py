import csv
import numpy as np
import matplotlib.pyplot as plt

class TrajectoryData:
    def __init__(self, max_length=2000):
        self.max_length = max_length
        self.ideal_tip_x = []
        self.ideal_tip_y = []
        self.ideal_cmd_x = []
        self.ideal_cmd_y = []
        self.hyst_tip_x = []
        self.hyst_tip_y = []
        self.hyst_cmd_x = []
        self.hyst_cmd_y = []
    
    def add_ideal(self, tip_x, tip_y, cmd_x, cmd_y):
        self.ideal_tip_x.append(tip_x)
        self.ideal_tip_y.append(tip_y)
        self.ideal_cmd_x.append(cmd_x)
        self.ideal_cmd_y.append(cmd_y)
        if len(self.ideal_tip_x) > self.max_length:
            self.ideal_tip_x.pop(0); self.ideal_tip_y.pop(0)
            self.ideal_cmd_x.pop(0); self.ideal_cmd_y.pop(0)
    
    def add_hyst(self, tip_x, tip_y, cmd_x, cmd_y):
        self.hyst_tip_x.append(tip_x)
        self.hyst_tip_y.append(tip_y)
        self.hyst_cmd_x.append(cmd_x)
        self.hyst_cmd_y.append(cmd_y)
        if len(self.hyst_tip_x) > self.max_length:
            self.hyst_tip_x.pop(0); self.hyst_tip_y.pop(0)
            self.hyst_cmd_x.pop(0); self.hyst_cmd_y.pop(0)
    
    def clear(self):
        self.ideal_tip_x = []; self.ideal_tip_y = []
        self.ideal_cmd_x = []; self.ideal_cmd_y = []
        self.hyst_tip_x = []; self.hyst_tip_y = []
        self.hyst_cmd_x = []; self.hyst_cmd_y = []
    
    def insert_nan(self):
        self.ideal_tip_x.append(np.nan); self.ideal_tip_y.append(np.nan)
        self.ideal_cmd_x.append(None); self.ideal_cmd_y.append(None)
        self.hyst_tip_x.append(np.nan); self.hyst_tip_y.append(np.nan)
        self.hyst_cmd_x.append(None); self.hyst_cmd_y.append(None)
    
    def save(self):
        with open("ideal_trajectory.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cmd_x_um", "cmd_y_um", "tip_x_um", "tip_y_um"])
            for cx, cy, tx, ty in zip(self.ideal_cmd_x, self.ideal_cmd_y, self.ideal_tip_x, self.ideal_tip_y):
                if cx is not None and not np.isnan(tx):
                    writer.writerow([f"{cx:.3f}", f"{cy:.3f}", f"{tx:.3f}", f"{ty:.3f}"])
        print("Ideal trajectory saved")
        
        with open("hysteresis_trajectory.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cmd_x_um", "cmd_y_um", "tip_x_um", "tip_y_um"])
            for cx, cy, tx, ty in zip(self.hyst_cmd_x, self.hyst_cmd_y, self.hyst_tip_x, self.hyst_tip_y):
                if cx is not None and not np.isnan(tx):
                    writer.writerow([f"{cx:.3f}", f"{cy:.3f}", f"{tx:.3f}", f"{ty:.3f}"])
        print("Hysteresis trajectory saved")
    
    def plot(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        if self.ideal_tip_x:
            plt.plot(self.ideal_tip_x, self.ideal_tip_y, 'g-', linewidth=1.5, label='Ideal')
        if self.hyst_tip_x:
            plt.plot(self.hyst_tip_x, self.hyst_tip_y, 'b-', linewidth=1.5, label='Hysteresis')
        plt.xlabel('Distance (um)')
        plt.ylabel('Distance (um)')
        plt.title('Tip Trajectory')
        plt.grid(True, alpha=0.3)
        if self.ideal_tip_x or self.hyst_tip_x:
            plt.legend()
        plt.axis('equal')
        
        plt.subplot(122)
        valid_cmd = [c for c, a in zip(self.hyst_cmd_x, self.hyst_tip_x) if c is not None and not np.isnan(a)]
        valid_act = [a for c, a in zip(self.hyst_cmd_x, self.hyst_tip_x) if c is not None and not np.isnan(a)]
        if valid_cmd:
            plt.plot(valid_cmd, valid_act, 'b.', markersize=2, label='Data')
            plt.plot([min(valid_cmd), max(valid_cmd)], [min(valid_cmd), max(valid_cmd)], 'r--', label='Ideal')
            plt.fill_between(valid_cmd, valid_act, valid_cmd, alpha=0.2)
            plt.legend()
        plt.xlabel('Command (um)')
        plt.ylabel('Tip (um)')
        plt.title('X-axis Hysteresis')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()