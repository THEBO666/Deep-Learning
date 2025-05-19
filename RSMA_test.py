import numpy as np

class RSMADownlinkSimulator:
    """
    单层下行链路RSMA（Rate-Splitting Multiple Access）仿真器。

    属性:
        num_users (int): 系统中的用户数量 (K)。
        num_bs_antennas (int): 基站的天线数量 (M)。
        total_power_dBm (float): 基站的总发射功率 (dBm)。
        noise_power_dBm (float): 用户端的噪声功率 (dBm)。
        seed (int, optional): 随机数生成器的种子，用于可复现性。

        total_power_linear (float): 线性尺度下的总发射功率。
        noise_power_linear (float): 线性尺度下的噪声功率。
        channels (np.ndarray): 形状为 (num_users, num_bs_antennas) 的信道矩阵, 每个元素是复数。
        p_c_dir (np.ndarray): 公共流预编码向量的方向 (归一化)。
        p_k_dirs (list[np.ndarray]): 私有流预编码向量的方向列表 (归一化)。
        
        p_c (np.ndarray): 实际的公共流预编码向量 (携带功率)。
        p_k_list (list[np.ndarray]): 实际的私有流预编码向量列表 (携带功率)。

        sinr_c_k_list (list[float]): 每个用户解码公共流时的SINR列表。
        sinr_p_k_list (list[float]): 每个用户解码其私有流时的SINR列表。
        R_c (float): 公共流的可达速率。
        R_p_k_list (list[float]): 每个用户的私有流可达速率列表。
        sum_rate (float): 系统总和速率。
    """

    def __init__(self, num_users: int, num_bs_antennas: int, 
                 total_power_dBm: float, noise_power_dBm: float = -90, 
                 seed: int = None):
        """
        初始化RSMA仿真器。

        参数:
            num_users (int): 用户数量 (K)。
            num_bs_antennas (int): 基站天线数量 (M)。
            total_power_dBm (float): 基站总发射功率 (dBm)。
            noise_power_dBm (float): 噪声功率 (dBm)。
            seed (int, optional): 随机数种子。
        """
        if num_users <= 0:
            raise ValueError("用户数量必须大于0。")
        if num_bs_antennas <= 0:
            raise ValueError("基站天线数量必须大于0。")

        self.num_users = num_users
        self.num_bs_antennas = num_bs_antennas
        self.total_power_dBm = total_power_dBm
        self.noise_power_dBm = noise_power_dBm
        
        if seed is not None:
            np.random.seed(seed)

        # 转换功率到线性尺度
        self.total_power_linear = 10**((self.total_power_dBm - 30) / 10)
        self.noise_power_linear = 10**((self.noise_power_dBm - 30) / 10)

        # 初始化信道和预编码器方向 (这些可以在仿真步骤中重新生成或设置)
        self.channels = None
        self.p_c_dir = None
        self.p_k_dirs = None
        
        self._initialize_system_components()

        # 结果占位符
        self.p_c = None
        self.p_k_list = None
        self.sinr_c_k_list = None
        self.sinr_p_k_list = None
        self.R_c = 0.0
        self.R_p_k_list = None
        self.sum_rate = 0.0

    def _initialize_system_components(self):
        """内部方法：初始化信道和预编码器方向。"""
        self.generate_channels()
        self._initialize_precoder_directions()

    def generate_channels(self, channel_type: str = 'rayleigh'):
        """
        生成用户信道。
        目前仅支持瑞利衰落信道。

        参数:
            channel_type (str): 信道类型 ('rayleigh')。
        """
        if channel_type.lower() == 'rayleigh':
            # 瑞利衰落信道: h_k ~ CN(0, I)
            # 形状: (num_users, num_bs_antennas)
            # 每一行 h_k^H 是一个用户的信道向量 (1 x M)
            # 这里我们存储 h_k (M x 1), 所以 self.channels[k] 是 h_k
            self.channels = (np.random.randn(self.num_users, self.num_bs_antennas) + 
                             1j * np.random.randn(self.num_users, self.num_bs_antennas)) / np.sqrt(2)
        else:
            raise ValueError(f"不支持的信道类型: {channel_type}")
        
        # 重置结果，因为信道变了
        self._reset_results()

    def _initialize_precoder_directions(self, strategy: str = 'random'):
        """
        初始化预编码器的方向 (归一化)。
        这些方向是固定的，功率将在其上分配。
        更高级的RSMA会联合优化预编码器和功率。

        参数:
            strategy (str): 预编码器方向初始化策略。
                            'random': 随机方向。
                            'mrt_like_private': 私有流采用类最大比传输(MRT)方向，公共流为用户MRT方向的平均。
        """
        if self.channels is None:
            raise ValueError("必须先生成信道才能初始化基于信道的预编码器方向。")

        if strategy == 'random':
            # 公共流预编码器方向
            pc_dir_unnormalized = (np.random.randn(self.num_bs_antennas) + 
                                   1j * np.random.randn(self.num_bs_antennas))
            self.p_c_dir = pc_dir_unnormalized / np.linalg.norm(pc_dir_unnormalized)

            # 私有流预编码器方向
            self.p_k_dirs = []
            for _ in range(self.num_users):
                pk_dir_unnormalized = (np.random.randn(self.num_bs_antennas) + 
                                       1j * np.random.randn(self.num_bs_antennas))
                self.p_k_dirs.append(pk_dir_unnormalized / np.linalg.norm(pk_dir_unnormalized))
        
        elif strategy == 'mrt_like_private':
            self.p_k_dirs = []
            sum_hk_norm = np.zeros(self.num_bs_antennas, dtype=complex)
            for k in range(self.num_users):
                hk = self.channels[k, :] # h_k^H (在我们的定义中，channels[k]是h_k)
                # 私有预编码器方向，匹配用户k的信道 (MRT)
                # 注意：这里的 hk 是 M x 1 的向量，我们直接使用它作为方向
                # 在实际MRT中 p_k = h_k / ||h_k||. 这里我们直接用 h_k 作为方向的基准，归一化后用。
                pk_dir_unnormalized = np.conjugate(hk) # M x 1
                self.p_k_dirs.append(pk_dir_unnormalized / np.linalg.norm(pk_dir_unnormalized))
                sum_hk_norm += (np.conjugate(hk) / np.linalg.norm(hk)) # 累加归一化的信道方向
            
            # 公共流预编码器方向：可以是所有用户MRT方向的某种组合，例如平均
            if self.num_users > 0:
                 # 避免除以零
                norm_sum_hk = np.linalg.norm(sum_hk_norm)
                if norm_sum_hk > 1e-9: # 避免除以一个非常小的值
                    self.p_c_dir = sum_hk_norm / norm_sum_hk
                else: # 如果所有信道都接近零或相互抵消，则退化为随机
                    pc_dir_unnormalized = (np.random.randn(self.num_bs_antennas) + 
                                       1j * np.random.randn(self.num_bs_antennas))
                    self.p_c_dir = pc_dir_unnormalized / np.linalg.norm(pc_dir_unnormalized)
            else: # 如果没有用户，公共预编码器是任意的
                pc_dir_unnormalized = (np.random.randn(self.num_bs_antennas) + 
                                   1j * np.random.randn(self.num_bs_antennas))
                self.p_c_dir = pc_dir_unnormalized / np.linalg.norm(pc_dir_unnormalized)

        else:
            raise ValueError(f"不支持的预编码器方向策略: {strategy}")
        
        self._reset_results()

    def set_custom_precoder_directions(self, p_c_dir: np.ndarray, p_k_dirs: list[np.ndarray]):
        """
        允许用户设置自定义的预编码器方向。
        这些方向必须是归一化的。

        参数:
            p_c_dir (np.ndarray): 归一化的公共流预编码器方向 (M x 1)。
            p_k_dirs (list[np.ndarray]): 归一化的私有流预编码器方向列表, 每个元素是 (M x 1)。
        """
        if not np.isclose(np.linalg.norm(p_c_dir), 1.0):
            raise ValueError("p_c_dir 必须是归一化的。")
        if p_c_dir.shape != (self.num_bs_antennas,):
            p_c_dir = p_c_dir.reshape(self.num_bs_antennas) # 尝试展平
            if p_c_dir.shape != (self.num_bs_antennas,):
                 raise ValueError(f"p_c_dir 的形状应为 ({self.num_bs_antennas},) 或 ({self.num_bs_antennas},1)。")


        if len(p_k_dirs) != self.num_users:
            raise ValueError("p_k_dirs 的长度必须等于用户数量。")
        for i, pk_dir in enumerate(p_k_dirs):
            if not np.isclose(np.linalg.norm(pk_dir), 1.0):
                raise ValueError(f"p_k_dirs[{i}] 必须是归一化的。")
            if pk_dir.shape != (self.num_bs_antennas,):
                pk_dir_reshaped = pk_dir.reshape(self.num_bs_antennas) # 尝试展平
                if pk_dir_reshaped.shape != (self.num_bs_antennas,):
                    raise ValueError(f"p_k_dirs[{i}] 的形状应为 ({self.num_bs_antennas},) 或 ({self.num_bs_antennas},1)。")
                p_k_dirs[i] = pk_dir_reshaped


        self.p_c_dir = p_c_dir.reshape(self.num_bs_antennas)
        self.p_k_dirs = [pkd.reshape(self.num_bs_antennas) for pkd in p_k_dirs]
        self._reset_results()

    def _form_precoders_with_power(self, common_power_fraction: float, 
                                   private_power_distribution: str = 'equal'):
        """
        根据功率分配因子形成实际的预编码向量。

        参数:
            common_power_fraction (float): 分配给公共流的功率占总功率的比例 (alpha, 0 <= alpha <= 1)。
            private_power_distribution (str): 剩余功率如何在私有流之间分配。
                                              'equal': 平均分配。
        """
        if not (0 <= common_power_fraction <= 1):
            raise ValueError("common_power_fraction 必须在 [0, 1] 之间。")
        if self.p_c_dir is None or self.p_k_dirs is None:
            raise RuntimeError("预编码器方向尚未初始化。请先调用 _initialize_precoder_directions() 或 set_custom_precoder_directions()。")

        power_c = common_power_fraction * self.total_power_linear
        power_private_total = (1 - common_power_fraction) * self.total_power_linear

        self.p_c = np.sqrt(power_c) * self.p_c_dir

        self.p_k_list = []
        if self.num_users > 0:
            if private_power_distribution == 'equal':
                power_per_private_stream = power_private_total / self.num_users
                for k_idx in range(self.num_users):
                    p_k = np.sqrt(power_per_private_stream) * self.p_k_dirs[k_idx]
                    self.p_k_list.append(p_k)
            # 未来可以扩展其他分配策略，例如基于信道质量的分配
            else:
                raise ValueError(f"不支持的私有功率分配策略: {private_power_distribution}")
        elif power_private_total > 1e-9: # 如果没有用户但分配了私有功率，发出警告或错误
             print("警告: common_power_fraction < 1 但 num_users = 0。私有功率将被浪费。")


    def calculate_sinr_and_rates(self):
        """
        计算SINR和可达速率。
        此方法假设 self.p_c 和 self.p_k_list 已经被正确设置（携带功率）。
        """
        if self.channels is None:
            raise RuntimeError("信道尚未生成。请先调用 generate_channels()。")
        if self.p_c is None or self.p_k_list is None:
            raise RuntimeError("预编码器尚未通过功率分配形成。请先调用 _form_precoders_with_power()。")
        
        # 确保p_k_list的长度与用户数匹配，即使没有用户，它也应该是空列表
        if len(self.p_k_list) != self.num_users:
             # 这种情况理论上不应该在 _form_precoders_with_power 正确执行后发生
             # 但作为防御性编程添加
            if self.num_users == 0 and not self.p_k_list: # 如果没有用户且列表为空，是OK的
                pass
            else:
                raise RuntimeError(f"p_k_list 长度 ({len(self.p_k_list)}) 与用户数 ({self.num_users}) 不匹配。")


        self.sinr_c_k_list = []
        self.sinr_p_k_list = []
        self.R_p_k_list = []

        # 计算每个用户解码公共流时的SINR (SINR_c,k)
        min_rate_c_k = float('inf') # 用于确定实际的 R_c

        if self.num_users == 0: # 如果没有用户，公共速率和私有速率都为0
            self.R_c = 0.0
            self.sum_rate = 0.0
            return

        for k in range(self.num_users):
            h_k = self.channels[k, :] # 用户k的信道向量 (1 x M) or (M,)
                                     # 我们在生成时是 (num_users, num_bs_antennas)
                                     # 所以 h_k 是 (num_bs_antennas,)
            
            # 解码公共流 s_c
            # 信号功率: |h_k^H p_c|^2
            # 在我们的定义中 h_k 是 (M,) 向量， p_c 是 (M,) 向量
            # h_k^H p_c = h_k.conj().T @ p_c
            signal_power_c = np.abs(np.vdot(h_k, self.p_c))**2 # vdot = conj(a) @ b

            # 干扰来自所有私有流: sum_j |h_k^H p_j|^2
            interference_power_c_from_private = 0
            for j in range(self.num_users):
                interference_power_c_from_private += np.abs(np.vdot(h_k, self.p_k_list[j]))**2
            
            sinr_c_k = signal_power_c / (interference_power_c_from_private + self.noise_power_linear)
            self.sinr_c_k_list.append(sinr_c_k)
            
            rate_c_k = np.log2(1 + sinr_c_k)
            if rate_c_k < min_rate_c_k:
                min_rate_c_k = rate_c_k
        
        # 公共流速率 R_c
        # 如果common_power_fraction为0，则p_c为0，signal_power_c为0，sinr_c_k为0，R_c为0
        # 如果没有用户，min_rate_c_k 保持 float('inf')，此时R_c应为0
        self.R_c = min_rate_c_k if self.num_users > 0 and np.linalg.norm(self.p_c) > 1e-9 else 0.0


        # 计算每个用户解码其私有流时的SINR (SINR_p,k) 和速率 (R_p,k)
        # (假设公共流已被完美解码和消除)
        current_sum_private_rate = 0.0
        for k in range(self.num_users):
            h_k = self.channels[k, :]
            
            # 信号功率: |h_k^H p_k|^2
            signal_power_pk = np.abs(np.vdot(h_k, self.p_k_list[k]))**2

            # 干扰来自其他私有流: sum_{j!=k} |h_k^H p_j|^2
            interference_power_pk_from_other_private = 0
            for j in range(self.num_users):
                if k == j:
                    continue
                interference_power_pk_from_other_private += np.abs(np.vdot(h_k, self.p_k_list[j]))**2
            
            sinr_p_k = signal_power_pk / (interference_power_pk_from_other_private + self.noise_power_linear)
            self.sinr_p_k_list.append(sinr_p_k)

            # 如果分配给该私有流的功率为0 (例如1-alpha=0，或者用户数很多导致平均功率很小)
            # 那么 signal_power_pk 会是0，sinr_p_k 是0，R_p_k 是0
            rate_p_k = np.log2(1 + sinr_p_k) if np.linalg.norm(self.p_k_list[k]) > 1e-9 else 0.0
            self.R_p_k_list.append(rate_p_k)
            current_sum_private_rate += rate_p_k
            
        # 系统总和速率
        self.sum_rate = self.R_c + current_sum_private_rate

    def run_simulation_step(self, common_power_fraction: float, 
                            private_power_distribution: str = 'equal',
                            precoder_init_strategy: str = 'random',
                            regenerate_channels: bool = False,
                            regenerate_precoder_dirs: bool = False) -> dict:
        """
        执行单次仿真步骤。

        参数:
            common_power_fraction (float): 分配给公共流的功率比例 (alpha)。
            private_power_distribution (str): 私有流功率分配策略。
            precoder_init_strategy (str): 如果需要，用于初始化预编码器方向的策略。
            regenerate_channels (bool): 是否为此次运行重新生成信道。
            regenerate_precoder_dirs (bool): 是否为此次运行重新生成预编码器方向。

        返回:
            dict: 包含SINR、速率等结果的字典。
        """
        if regenerate_channels or self.channels is None:
            self.generate_channels() # 使用默认的'rayleigh'
        
        if regenerate_precoder_dirs or self.p_c_dir is None or not self.p_k_dirs:
            self._initialize_precoder_directions(strategy=precoder_init_strategy)

        self._form_precoders_with_power(common_power_fraction, private_power_distribution)
        self.calculate_sinr_and_rates()

        return self.get_results()

    def get_results(self) -> dict:
        """
        获取当前仿真状态的结果。

        返回:
            dict: 包含SINR、速率等结果。
        """
        return {
            "common_power_fraction_alpha": np.linalg.norm(self.p_c)**2 / self.total_power_linear if self.total_power_linear > 0 else 0,
            "SINR_c_k (per user for common stream)": self.sinr_c_k_list,
            "SINR_p_k (per user for private stream)": self.sinr_p_k_list,
            "R_c (common rate)": self.R_c,
            "R_p_k_list (private rates)": self.R_p_k_list,
            "User_total_rates (R_c/K + R_p_k, example)": [self.R_c/self.num_users + rpk if self.num_users > 0 else 0 for rpk in self.R_p_k_list] if self.R_p_k_list else [], # 假设公共速率平均分配
            "Sum_Rate (R_c + sum(R_p_k))": self.sum_rate,
            "Power_p_c": np.linalg.norm(self.p_c)**2 if self.p_c is not None else 0,
            "Power_p_k_list": [np.linalg.norm(pk)**2 for pk in self.p_k_list] if self.p_k_list else [],
            "Total_Allocated_Power": (np.linalg.norm(self.p_c)**2 if self.p_c is not None else 0) + sum(np.linalg.norm(pk)**2 for pk in self.p_k_list if pk is not None)
        }
    
    def _reset_results(self):
        """重置仿真结果变量，当系统参数（如信道、预编码器方向）改变时调用。"""
        self.p_c = None
        self.p_k_list = None
        self.sinr_c_k_list = None
        self.sinr_p_k_list = None
        self.R_c = 0.0
        self.R_p_k_list = None
        self.sum_rate = 0.0

    def compare_with_sdma(self, precoder_init_strategy: str = 'mrt_like_private', regenerate_channels: bool = False, regenerate_precoder_dirs: bool = False) -> dict:
        """
        运行一个SDMA（等效于RSMA中 alpha=0）的仿真作为对比。
        在SDMA中，没有公共流，所有功率分配给私有流。
        预编码器通常采用迫零(ZF)或正则化迫零(RZF)等。
        这里为了简单对比，我们使用与RSMA相同的预编码器方向初始化策略，
        只是将所有功率分配给私有流。

        注意：这只是一个简化的SDMA，真正的SDMA优化预编码器可能不同。
        """
        # 保存当前的预编码器方向状态
        original_pc_dir = self.p_c_dir
        original_pk_dirs = self.p_k_dirs

        # 运行RSMA仿真，但 alpha = 0 (所有功率给私有流)
        # 使用相同的信道和预编码器方向（如果指定不重新生成）
        sdma_results = self.run_simulation_step(
            common_power_fraction=0.0, # SDMA: 没有公共流
            precoder_init_strategy=precoder_init_strategy,
            regenerate_channels=regenerate_channels,
            regenerate_precoder_dirs=regenerate_precoder_dirs 
        )
        
        # 恢复原始预编码器方向（如果它们被 run_simulation_step 中的 regenerate_precoder_dirs 修改了）
        # 如果用户不希望修改原始状态，他们应该在调用此方法前保存状态或重新初始化模拟器。
        # 为了确保此方法不意外更改状态，可以考虑深拷贝或重新设置。
        # 但鉴于run_simulation_step的逻辑，如果regenerate_precoder_dirs为False，它们不会改变。
        # 为安全起见，如果它们被改变了，我们恢复。
        if regenerate_precoder_dirs:
            self.p_c_dir = original_pc_dir
            self.p_k_dirs = original_pk_dirs
        
        # SDMA的和速率就是所有私有流速率之和，因为R_c会是0
        sdma_results["Sum_Rate (SDMA)"] = sdma_results["Sum_Rate (R_c + sum(R_p_k))"]
        return sdma_results

import matplotlib.pyplot as plt

# 1. 基本仿真设置
num_users = 2
num_bs_antennas = 4
total_power_dBm = 20  # 总功率 20 dBm
noise_power_dBm = -90 # 噪声功率 -90 dBm
random_seed = 42

# 2. 创建RSMA仿真器实例
rsma_sim = RSMADownlinkSimulator(
    num_users=num_users,
    num_bs_antennas=num_bs_antennas,
    total_power_dBm=total_power_dBm,
    noise_power_dBm=noise_power_dBm,
    seed=random_seed
)
print(f"仿真器初始化完毕。总功率: {rsma_sim.total_power_linear:.4f} W, 噪声功率: {rsma_sim.noise_power_linear:.2e} W")

# 可以查看初始化的信道和预编码器方向 (可选)
print("Channels (H_k^H for each user k):\n", rsma_sim.channels)
print("Common precoder direction (p_c_dir):\n", rsma_sim.p_c_dir)
print("Private precoder directions (p_k_dirs):\n", rsma_sim.p_k_dirs)

# 3. 运行单次仿真，指定公共功率分配比例 (alpha)
# 例如，将总功率的30%分配给公共流
alpha = 0.3 
results = rsma_sim.run_simulation_step(
    common_power_fraction=alpha,
    precoder_init_strategy='mrt_like_private', # 使用MRT类方向
    regenerate_channels=False, # 使用初始化的信道
    regenerate_precoder_dirs=True # 根据MRT策略重新生成预编码器方向
)

print(f"\n--- 仿真结果 (alpha = {alpha}, precoder_init_strategy='mrt_like_private') ---")
print(f"公共流速率 (R_c): {results['R_c (common rate)']:.4f} bps/Hz")
print(f"私有流速率 (R_p_k_list): {[f'{r:.4f}' for r in results['R_p_k_list (private rates)']]} bps/Hz")
print(f"系统和速率: {results['Sum_Rate (R_c + sum(R_p_k))']:.4f} bps/Hz")
print(f"分配给公共流的功率: {results['Power_p_c']:.4f} W")
print(f"分配给私有流的功率: {[f'{p:.4f}' for p in results['Power_p_k_list']]} W")
print(f"总分配功率: {results['Total_Allocated_Power']:.4f} W (应接近 {rsma_sim.total_power_linear:.4f} W)")

# 4. 对比SDMA (alpha = 0)
# 注意：这里为了对比，我们会使用相同的信道和预编码器方向生成逻辑
# SDMA通常有自己的预编码器设计（如ZF, RZF），这里的比较是基于RSMA框架的特例
sdma_results = rsma_sim.compare_with_sdma(
    precoder_init_strategy='mrt_like_private', # 与上面RSMA使用相同的策略
    regenerate_channels=False, # 使用相同的信道实例
    regenerate_precoder_dirs=False # 使用刚才RSMA步骤中已经设置好的方向（因为alpha=0，p_c_dir无关紧要）
                                     # 如果设为True，会重新随机或按MRT生成（与之前可能不同）
)
print(f"\n--- SDMA (alpha = 0) 对比结果 ---")
print(f"SDMA 和速率: {sdma_results['Sum_Rate (SDMA)']:.4f} bps/Hz")


# 5. 扫描不同的alpha值，观察和速率变化
alpha_values = np.linspace(0, 1, 21) # 从0到1，取21个点
sum_rates_rsma = []
sum_rates_sdma_baseline = sdma_results['Sum_Rate (SDMA)'] # SDMA速率作为基准

# 使用相同的信道和预编码器方向进行扫描
# 为此，我们先固定它们
rsma_sim.generate_channels() # 生成一次信道
rsma_sim._initialize_precoder_directions(strategy='mrt_like_private') # 初始化一次方向

for alpha_val in alpha_values:
    current_results = rsma_sim.run_simulation_step(
        common_power_fraction=alpha_val,
        regenerate_channels=False, # 不重新生成信道
        regenerate_precoder_dirs=False # 不重新生成方向
    )
    sum_rates_rsma.append(current_results['Sum_Rate (R_c + sum(R_p_k))'])

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, sum_rates_rsma, marker='o', linestyle='-', label='RSMA Sum Rate')
plt.axhline(y=sum_rates_sdma_baseline, color='r', linestyle='--', label=f'SDMA Sum Rate (alpha=0 baseline)')
plt.xlabel('Common Power Fraction (alpha)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title(f'RSMA Sum Rate vs. Alpha (K={num_users}, M={num_bs_antennas}, P_total={total_power_dBm}dBm)')
plt.legend()
plt.grid(True)
plt.show()

# 6. 多次信道实现下的平均性能 (蒙特卡洛仿真)
num_channel_realizations = 50
avg_sum_rates_rsma = np.zeros_like(alpha_values)
avg_sum_rates_sdma = 0

for i in range(num_channel_realizations):
    # 每次都重新生成信道和预编码器方向（基于MRT）
    # 如果只想变信道，则 regenerate_precoder_dirs=False，但MRT依赖信道，所以通常一起变
    
    # 先算这次信道下的SDMA
    sdma_res_mc = rsma_sim.compare_with_sdma(
        precoder_init_strategy='mrt_like_private',
        regenerate_channels=True, # 新信道
        regenerate_precoder_dirs=True # 基于新信道的新MRT方向
    )
    avg_sum_rates_sdma += sdma_res_mc['Sum_Rate (SDMA)']

    # 然后算这次信道和方向下的RSMA (alpha扫描)
    # 注意：run_simulation_step内部会使用当前rsma_sim对象中的信道和方向
    # compare_with_sdma中regenerate_channels=True, regenerate_precoder_dirs=True已经更新了它们
    for j, alpha_val in enumerate(alpha_values):
        current_results_mc = rsma_sim.run_simulation_step(
            common_power_fraction=alpha_val,
            regenerate_channels=False, # 不再重新生成，用刚才SDMA步骤生成的
            regenerate_precoder_dirs=False # 不再重新生成
        )
        avg_sum_rates_rsma[j] += current_results_mc['Sum_Rate (R_c + sum(R_p_k))']
    if (i+1) % 10 == 0:
        print(f"蒙特卡洛: 完成 {i+1}/{num_channel_realizations} 次信道实现")

avg_sum_rates_rsma /= num_channel_realizations
avg_sum_rates_sdma /= num_channel_realizations

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, avg_sum_rates_rsma, marker='s', linestyle='-', label='Avg. RSMA Sum Rate')
plt.axhline(y=avg_sum_rates_sdma, color='g', linestyle='--', label=f'Avg. SDMA Sum Rate (alpha=0)')
plt.xlabel('Common Power Fraction (alpha)')
plt.ylabel('Average Sum Rate (bps/Hz)')
plt.title(f'Avg. RSMA Sum Rate vs. Alpha (K={num_users}, M={num_bs_antennas}, P_total={total_power_dBm}dBm, {num_channel_realizations} realizations)')
plt.legend()
plt.grid(True)
plt.show()