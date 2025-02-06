import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SelfBalancingCarEnv(gym.Env):
    """
    自平衡小车环境（类似倒立摆）——基于 Gymnasium API

    状态:
        - theta: 车体相对于竖直方向的偏离角度（弧度）
        - theta_dot: 角速度

    动作:
        - 扭矩，作用在车体上，用于调节平衡

    奖励:
        - 根据偏离角度、角速度和扭矩大小构造负奖励（目标为保持平衡，即 theta 接近 0）

    终止条件:
        - 当偏角超过 ±90° 时，认为系统失稳，episode 结束。
    """
    metadata = {"render_modes": ["human"], "name": "SelfBalancingCarEnv-v0"}

    def __init__(self, render_mode=None):
        super(SelfBalancingCarEnv, self).__init__()
        self.render_mode = render_mode

        # 定义状态空间：theta ∈ [-pi, pi]，theta_dot ∈ [-10, 10]
        high = np.array([np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # 定义动作空间：施加的扭矩在 [-max_torque, max_torque] 范围内
        self.max_torque = 2.0
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)

        # 环境参数
        self.dt = 0.02  # 时间步长
        self.mass = 1.0  # 小车质量
        self.length = 0.5  # 类似摆长
        self.gravity = 9.8

        self.state = None

    def step(self, action):
        theta, theta_dot = self.state

        # 限制动作
        torque = np.clip(action[0], -self.max_torque, self.max_torque)

        # 根据倒立摆动力学更新状态：
        # theta_double_dot = (g / L) * sin(theta) + torque/(m*L^2)
        theta_double_dot = (self.gravity / self.length) * np.sin(theta) + torque / (self.mass * self.length ** 2)

        # Euler 积分更新
        theta_dot += theta_double_dot * self.dt
        theta += theta_dot * self.dt

        self.state = np.array([theta, theta_dot], dtype=np.float32)

        # 当偏离角度过大时认为失败（终止条件）
        terminated = bool(abs(theta) > np.pi / 2)
        # 这里不区分因时间限制而截断，因此 truncated 保持 False
        truncated = False

        # 奖励设计：希望保持 theta 接近 0，同时惩罚过大的角速度和扭矩使用
        reward = - (theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * (torque ** 2))

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        # 设置随机种子（如果提供）
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random

        # 随机初始化状态，确保初始状态较接近平衡状态
        self.state = np.array([
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.05, 0.05)
        ], dtype=np.float32)
        # Gymnasium 的 reset 返回 (observation, info)
        return self.state, {}

    def render(self):
        # 简单的文本渲染
        theta, theta_dot = self.state
        print(f"theta: {theta:.2f} rad, theta_dot: {theta_dot:.2f} rad/s")

    def close(self):
        pass


# 单元测试环境（可选）
if __name__ == "__main__":
    env = SelfBalancingCarEnv()
    obs, _ = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print("Episode terminated!")
            break
