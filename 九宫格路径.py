import numpy as np
import matplotlib.pyplot as plt

# ---------- 参数设置 ----------
GRID_SIZE = 3
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
N_ACTIONS = len(ACTIONS)
Q = np.zeros((GRID_SIZE, GRID_SIZE, N_ACTIONS))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000

START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)


# ---------- 环境交互函数 ----------
def step(state, action_id):
    r, c = state
    dr, dc = ACTIONS[action_id]
    nr = np.clip(r + dr, 0, GRID_SIZE - 1)
    nc = np.clip(c + dc, 0, GRID_SIZE - 1)
    next_state = (nr, nc)
    reward = 10 if next_state == GOAL else -1
    return next_state, reward


# ---------- 训练过程 ----------
episode_steps = []
episode_returns = []

for _ in range(episodes):
    state = START
    total_reward = 0
    for t in range(50):
        if np.random.rand() < epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            action = np.argmax(Q[state])
        next_state, reward = step(state, action)
        total_reward += reward
        best_next = np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
        state = next_state
        if state == GOAL:
            break
    episode_steps.append(t + 1)
    episode_returns.append(total_reward)

# ---------- 提取最优路径 ----------
path = [START]
state = START
for _ in range(10):
    action = np.argmax(Q[state])
    state, _ = step(state, action)
    path.append(state)
    if state == GOAL:
        break

# 图1：最终路径图
fig, ax = plt.subplots(figsize=(4, 4))
for i in range(GRID_SIZE + 1):
    ax.plot([-0.5, GRID_SIZE - 0.5], [i - 0.5, i - 0.5], color="black")
    ax.plot([i - 0.5, i - 0.5], [-0.5, GRID_SIZE - 0.5], color="black")
for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
    ax.arrow(
        c1,
        r1,
        c2 - c1,
        r2 - r1,
        head_width=0.15,
        length_includes_head=True,
        color="orange",
    )
ax.scatter([START[1]], [START[0]], s=120, c="green", label="Start")
ax.scatter([GOAL[1]], [GOAL[0]], s=120, c="red", label="Goal")
ax.set_xlim(-0.5, GRID_SIZE - 0.5)
ax.set_ylim(GRID_SIZE - 0.5, -0.5)
ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))
ax.set_aspect("equal")
ax.set_title("Shortest Path Learned via Q-Learning")
ax.legend()
plt.show()

# 图2：训练过程曲线
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(episode_steps, lw=1)
ax[0].set_title("Episode Lengths")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("# Steps")
ax[1].plot(episode_returns, lw=1)
ax[1].set_title("Episode Returns")
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Return")
plt.suptitle("Learning Curve of Tabular Q-Learning")
plt.tight_layout()
plt.show()

# 图3：状态值热力图
state_values = np.max(Q, axis=2)
plt.figure(figsize=(4, 4))
plt.imshow(state_values, cmap="viridis", origin="upper")
plt.colorbar(label="V(s) = max_a Q(s,a)")
plt.xticks(range(GRID_SIZE))
plt.yticks(range(GRID_SIZE))
plt.title("State-Value Heatmap after Training")
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        plt.text(
            c,
            r,
            f"{state_values[r, c]:.1f}",
            ha="center",
            va="center",
            color="white" if state_values[r, c] < 0 else "black",
        )
plt.show()

# 图4：策略箭头图
action2arrow = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}
fig, ax = plt.subplots(figsize=(4, 4))
for i in range(GRID_SIZE + 1):
    ax.plot([-0.5, GRID_SIZE - 0.5], [i - 0.5, i - 0.5], color="black")
    ax.plot([i - 0.5, i - 0.5], [-0.5, GRID_SIZE - 0.5], color="black")
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if (r, c) == GOAL:
            continue
        a = np.argmax(Q[r, c])
        dx, dy = action2arrow[a]
        ax.arrow(c, r, dx, dy, head_width=0.12, length_includes_head=True)
ax.scatter([START[1]], [START[0]], c="green", s=120, label="Start")
ax.scatter([GOAL[1]], [GOAL[0]], c="red", s=120, label="Goal")
ax.set_xlim(-0.5, GRID_SIZE - 0.5)
ax.set_ylim(GRID_SIZE - 0.5, -0.5)
ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))
ax.set_title("Greedy Policy Arrows")
ax.legend()
plt.show()
