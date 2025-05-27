import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# --- 仿真参数 ---
AREA_WIDTH = 200  # meters
AREA_HEIGHT = 200 # meters
SIM_TIME = 60      # seconds
TIME_STEP = 0.1    # seconds (simulation update interval)
NUM_UES = 2
NUM_BS = 2

# --- 基站参数 ---
BS_TX_POWER_dBm = 40  # dBm
BS_ANTENNA_GAIN_dBi = 15 # dBi
UE_ANTENNA_GAIN_dBi = 0  # dBi
CARRIER_FREQ_GHz = 2.6 # GHz (e.g., mid-band 5G)

# --- 信道模型参数 ---
PATH_LOSS_EXPONENT = 3.5  # 路径损耗指数
SHADOWING_STD_DEV_dB = 8 # dB (阴影衰落标准差)
REFERENCE_DISTANCE_m = 1 # m (路径损耗参考距离)

# 路径损耗在参考距离处的损耗 (dB), 使用自由空间路径损耗公式
# L_FSPL(dB) = 20log10(d) + 20log10(f) + 20log10(4pi/c) - G_tx - G_rx
# 这里我们简化，假设 L_0 是在 d=1m 处的路径损耗基准
c = 3e8 # 光速
wavelength = c / (CARRIER_FREQ_GHz * 1e9)
L0_dB = 20 * np.log10(4 * np.pi * REFERENCE_DISTANCE_m / wavelength)


class BaseStation:
    def __init__(self, id, pos, tx_power_dBm, antenna_gain_dBi):
        self.id = id
        self.pos = np.array(pos)  # [x, y]
        self.tx_power_dBm = tx_power_dBm
        self.antenna_gain_dBi = antenna_gain_dBi
        self.eirp_dBm = tx_power_dBm + antenna_gain_dBi # 有效全向辐射功率

    def plot(self, ax):
        ax.plot(self.pos[0], self.pos[1], 'hb', markersize=10, label=f"BS {self.id}" if not ax.get_legend() else "")
        ax.text(self.pos[0] + 20, self.pos[1] + 20, f"BS{self.id}")

class UserEquipment:
    def __init__(self, id, pos, max_speed_mps):
        self.id = id
        self.pos = np.array(pos, dtype=float) # [x, y]
        self.max_speed_mps = max_speed_mps
        self.velocity = np.array([0.0, 0.0]) # [vx, vy]
        self.connected_bs_id = None
        self.rssi_dBm = -np.inf # 接收信号强度
        self.color = np.random.rand(3,) # 为UE分配一个随机颜色

    def update_position(self, dt):
        # 简单的随机游走模型 (带边界反弹)
        # 随机改变速度方向和大小
        if np.random.rand() < 0.1: # 10% 概率改变方向
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5 * self.max_speed_mps, self.max_speed_mps)
            self.velocity[0] = speed * np.cos(angle)
            self.velocity[1] = speed * np.sin(angle)

        self.pos += self.velocity * dt

        # 边界反弹
        if self.pos[0] < 0 or self.pos[0] > AREA_WIDTH:
            self.velocity[0] *= -1
            self.pos[0] = np.clip(self.pos[0], 0, AREA_WIDTH)
        if self.pos[1] < 0 or self.pos[1] > AREA_HEIGHT:
            self.velocity[1] *= -1
            self.pos[1] = np.clip(self.pos[1], 0, AREA_HEIGHT)

    def calculate_rssi(self, bs_list):
        best_rssi = -np.inf
        selected_bs_id = None

        for bs in bs_list:
            distance = np.linalg.norm(self.pos - bs.pos)
            if distance < 1e-6: # 避免除以零或log(0)
                distance = 1e-6

            # 路径损耗 (dB)
            path_loss_dB = L0_dB + 10 * PATH_LOSS_EXPONENT * np.log10(distance / REFERENCE_DISTANCE_m)
            
            # 阴影衰落 (dB)
            shadowing_loss_dB = np.random.normal(0, SHADOWING_STD_DEV_dB)
            
            total_loss_dB = path_loss_dB + shadowing_loss_dB
            
            # 接收功率 (dBm)
            rx_power_dBm = bs.eirp_dBm - total_loss_dB - UE_ANTENNA_GAIN_dBi # 减去UE增益因为已在EIRP中考虑BS增益

            if rx_power_dBm > best_rssi:
                best_rssi = rx_power_dBm
                selected_bs_id = bs.id
        
        self.rssi_dBm = best_rssi
        self.connected_bs_id = selected_bs_id
        return best_rssi, selected_bs_id

    def plot(self, ax):
        ax.plot(self.pos[0], self.pos[1], 'o', color=self.color, markersize=6)
        ax.text(self.pos[0] + 10, self.pos[1] + 10, f"U{self.id}({self.rssi_dBm:.0f}dBm)")
        if self.connected_bs_id is not None:
            # 找到连接的BS对象以获取其位置
            connected_bs_obj = next((bs for bs in sim.base_stations if bs.id == self.connected_bs_id), None)
            if connected_bs_obj:
                ax.plot([self.pos[0], connected_bs_obj.pos[0]],
                        [self.pos[1], connected_bs_obj.pos[1]],
                        '--', color=self.color, alpha=0.5)


class Simulation:
    def __init__(self, num_bs, num_ues):
        self.base_stations = []
        self.user_equipments = []
        self.current_time = 0.0

        # 初始化基站
        # 简单地将基站放置在区域的某些固定点
        bs_positions = [
            (AREA_WIDTH * 0.25, AREA_HEIGHT * 0.5),
            (AREA_WIDTH * 0.75, AREA_HEIGHT * 0.25),
            (AREA_WIDTH * 0.75, AREA_HEIGHT * 0.75)
        ]
        for i in range(min(num_bs, len(bs_positions))):
            self.base_stations.append(BaseStation(i, bs_positions[i], BS_TX_POWER_dBm, BS_ANTENNA_GAIN_dBi))
        # 如果需要更多BS，可以随机放置或按网格放置
        for i in range(len(bs_positions), num_bs):
             pos = (np.random.rand() * AREA_WIDTH, np.random.rand() * AREA_HEIGHT)
             self.base_stations.append(BaseStation(i, pos, BS_TX_POWER_dBm, BS_ANTENNA_GAIN_dBi))


        # 初始化UE
        for i in range(num_ues):
            pos = (np.random.rand() * AREA_WIDTH, np.random.rand() * AREA_HEIGHT)
            max_speed = np.random.uniform(10, 30) # m/s (36 to 108 km/h)
            self.user_equipments.append(UserEquipment(i, pos, max_speed))

    def step(self, dt):
        # 更新UE位置
        for ue in self.user_equipments:
            ue.update_position(dt)

        # UE计算RSSI并选择连接的BS
        for ue in self.user_equipments:
            ue.calculate_rssi(self.base_stations)
            
        self.current_time += dt

# --- 可视化 ---
fig, ax = plt.subplots(figsize=(10, 10))
sim = Simulation(NUM_BS, NUM_UES)

# 预先绘制一次BS，以便图例只显示一次
for bs in sim.base_stations:
    bs.plot(ax)
if sim.base_stations:
    ax.legend(loc="upper right")


# 用于存储UE轨迹点
ue_trails = {ue.id: {'x': [], 'y': []} for ue in sim.user_equipments}
trail_length = 50 # 显示轨迹的长度


def init_animation():
    ax.clear()
    ax.set_xlim(0, AREA_WIDTH)
    ax.set_ylim(0, AREA_HEIGHT)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"High-Speed Mobile Communication Simulation")
    ax.grid(True)
    # 重新绘制BS，因为clear会清除它们
    for bs_obj in sim.base_stations:
        bs_obj.plot(ax)
    if sim.base_stations:
        ax.legend(loc="upper right") # 保持图例
    return [] # 返回一个空的艺术家列表

def update_animation(frame_num):
    start_time = time.time()
    sim.step(TIME_STEP)
    
    # 清除之前的UE位置和连接线，但保留BS和标题等
    # 找到所有代表UE的 'line' 和 'text' 对象并移除它们
    artists_to_remove = []
    for child in ax.get_children():
        if isinstance(child, plt.Line2D) and (child.get_marker() == 'o' or child.get_linestyle() == '--'): # UE点和连接线
            artists_to_remove.append(child)
        elif isinstance(child, plt.Text) and child.get_text().startswith("U"): # UE标签
             artists_to_remove.append(child)
    
    for artist in artists_to_remove:
        artist.remove()

    # 绘制新的UE位置和连接
    current_artists = []
    for ue in sim.user_equipments:
        ue.plot(ax) # 这会重新绘制UE的点、标签和连接线
        
        # 更新并绘制轨迹
        ue_trails[ue.id]['x'].append(ue.pos[0])
        ue_trails[ue.id]['y'].append(ue.pos[1])
        if len(ue_trails[ue.id]['x']) > trail_length:
            ue_trails[ue.id]['x'].pop(0)
            ue_trails[ue.id]['y'].pop(0)
        
        # ax.plot返回一个包含Line2D对象的列表，我们将其添加到artists
        trail_line, = ax.plot(ue_trails[ue.id]['x'], ue_trails[ue.id]['y'], '-', color=ue.color, alpha=0.3, lw=1)
        current_artists.append(trail_line)


    # 更新标题显示时间
    ax.set_title(f"High-Speed Mobile Comm Sim - Time: {sim.current_time:.1f}s (Frame: {frame_num})")
    
    # 将从ue.plot()和轨迹绘制中新创建的Line2D和Text对象添加到返回列表
    # ue.plot() 内部会创建 Line2D 和 Text 对象，但我们没有直接捕获它们。
    # 最简单的方式是假设ue.plot()正确地将它们添加到ax。
    # 我们需要返回所有在这一帧*新创建或更新*的artists。
    # 由于我们清除了旧的UE artists，所有当前可见的UE相关的artists都是新绘制的。
    # matplotlib的animation会处理它们。我们主要返回轨迹线，因为它们是明确创建的。
    
    # print(f"Frame {frame_num}, Sim time: {sim.current_time:.1f}s, Render time: {time.time() - start_time:.3f}s")
    return current_artists # 返回新绘制的轨迹线

# 计算总帧数
num_frames = int(SIM_TIME / TIME_STEP)

# 创建动画
# blit=True 可以提高性能，但有时在清除和重绘复杂图形时会导致问题。
# 如果遇到问题，可以尝试 blit=False。
# init_func是必须的，以确保动画开始时有一个干净的画布。
ani = animation.FuncAnimation(fig, update_animation, frames=num_frames,
                              init_func=init_animation, blit=False, interval=int(TIME_STEP * 1000), repeat=False)

plt.tight_layout()
plt.show()

# 如果想保存动画 (需要ffmpeg或imagemagick等)
# print("Attempting to save animation...")
# ani.save('mobile_com_sim.mp4', writer='ffmpeg', fps=10)
# print("Animation saved as mobile_com_sim.mp4")