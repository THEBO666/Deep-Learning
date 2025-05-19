import numpy as np

class SingleLayerRSMA:
    def __init__(self, num_users=2, num_tx_antennas=4, total_power=1.0, noise_power=1e-3, delta=0.0):
        """
        初始化RSMA系统。
        """
        self.K = num_users
        self.Nt = num_tx_antennas
        self.Pt = total_power
        self.noise_power = noise_power
        self.delta = delta
        self.H_true = None
        self.H_est = None

    def normalize(self, x):
        return x / np.linalg.norm(x)

    def generate_channels(self, seed=None):
        """
        生成真实信道 H_true 和扰动后的估计信道 H_est。
        """
        if seed is not None:
            np.random.seed(seed)
        H_true = (np.random.randn(self.K, self.Nt) + 1j * np.random.randn(self.K, self.Nt)) / np.sqrt(2)
        H_error = (np.random.randn(self.K, self.Nt) + 1j * np.random.randn(self.K, self.Nt)) / np.sqrt(2)
        H_est = H_true + self.delta * H_error / np.linalg.norm(H_error, axis=1, keepdims=True)
        self.H_true = H_true
        self.H_est = H_est
        return H_true, H_est

    def compute_sinr(self, alpha):
        """
        计算每个用户的 SINR，包括公共和私有。
        """
        Pc = alpha * self.Pt
        Pp = (1 - alpha) * self.Pt / self.K
        w_c = self.normalize(np.sum(self.H_est, axis=0).reshape(-1, 1))  # 公共波束
        w_p = np.array([self.normalize(self.H_est[k, :]) for k in range(self.K)]).T  # 私有波束矩阵

        sinr_c = []
        sinr_p = []
        for k in range(self.K):
            h = self.H_true[k, :].reshape(1, -1)
            s_c = np.abs(h @ w_c)**2 * Pc
            i_c = np.sum([np.abs(h @ w_p[:, j])**2 * Pp for j in range(self.K)])
            sinr_common = s_c / (i_c + self.noise_power)
            sinr_c.append(sinr_common)

            s_p = np.abs(h @ w_p[:, k])**2 * Pp
            i_p = np.sum([np.abs(h @ w_p[:, j])**2 * Pp for j in range(self.K) if j != k])
            sinr_private = s_p / (i_p + self.noise_power)
            sinr_p.append(sinr_private)

        return np.array(sinr_c), np.array(sinr_p)

    def compute_rates(self, alpha):
        """
        返回公共速率、私有速率数组、每用户总速率、系统总和速率。
        """
        sinr_c, sinr_p = self.compute_sinr(alpha)
        R_common = np.log2(1 + np.min(sinr_c))
        R_private = np.log2(1 + sinr_p)
        R_total = R_common + R_private
        R_sum = np.sum(R_total)
        return {
            "R_common": R_common,
            "R_private": R_private,
            "R_total_per_user": R_total,
            "R_sum": R_sum
        }

    def optimize_alpha(self, alpha_range=np.linspace(0.01, 0.99, 100)):
        """
        遍历搜索最优 alpha 以最大化系统总速率。
        """
        best_alpha = 0
        best_rate = -np.inf
        best_detail = None
        for alpha in alpha_range:
            rates = self.compute_rates(alpha)
            if rates["R_sum"] > best_rate:
                best_rate = rates["R_sum"]
                best_alpha = alpha
                best_detail = rates
        return {
            "best_alpha": best_alpha,
            "max_sum_rate": best_rate,
            "details": best_detail
        }

# 示例使用
rsma = SingleLayerRSMA(num_users=2, num_tx_antennas=4, total_power=1.0, noise_power=1e-3, delta=0.05)
rsma.generate_channels(seed=42)
result = rsma.optimize_alpha()
print(result)
