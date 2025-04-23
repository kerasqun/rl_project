import gym
import numpy as np
import os
import pybullet as p
import pybullet_data
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.utils import safe_mean
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path


class PyBulletCartPoleEnv(gym.Env):
    """自定义PyBullet倒立摆环境"""
    def __init__(self, gui=False, randomize_params=False):
        super().__init__()
        # 动作空间：施加在小车上的力，进一步减小动作范围
        self.action_space = gym.spaces.Box(low=-30, high=30, shape=(1,), dtype=np.float32)
        # 观察空间：[x, x_dot, theta, theta_dot]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # 连接到PyBullet
        self.client = p.connect(p.GUI if gui else p.DIRECT)  # GUI模式用于可视化
        
        # 是否随机化参数
        self.randomize_params = randomize_params
        
        # 记录上一步的杆角度，用于计算角度变化
        self.last_pole_angle = 0
        
        # 设置物理参数
        self.timestep = 1.0/240.0
        
    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timestep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载地面和倒立摆
        self.plane = p.loadURDF("plane.urdf")
        self.cartpole = p.loadURDF("cartpole.urdf", [0, 0, 0.1])
        
        # 禁用电机控制
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        
        # 重置关节状态，可以添加一些随机性
        initial_cart_pos = 0.0
        initial_pole_angle = 0.0
        
        if self.randomize_params:
            # 添加一些随机初始状态，帮助模型更好地泛化
            initial_cart_pos = np.random.uniform(-0.3, 0.3)  # 减小初始位置随机范围
            initial_pole_angle = np.random.uniform(-0.03, 0.03)  # 进一步减小初始角度随机范围
        
        p.resetJointState(self.cartpole, 0, targetValue=initial_cart_pos, targetVelocity=0)
        p.resetJointState(self.cartpole, 1, targetValue=initial_pole_angle, targetVelocity=0)
        
        # 初始化上一步的杆角度
        self.last_pole_angle = initial_pole_angle
        
        # 执行几步模拟，使系统稳定
        for _ in range(5):
            p.stepSimulation()
        
        return self._get_observation()
    
    def step(self, action):
        # 限制动作范围，增加稳定性
        action = np.clip(action, -30, 30)
        
        # 施加控制力
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=float(action[0]))
        p.stepSimulation()
        
        # 获取状态
        obs = self._get_observation()
        cart_pos, cart_vel = obs[0], obs[1]
        pole_angle, pole_vel = obs[2], obs[3]
        
        # 计算角度变化
        angle_change = abs(pole_angle - self.last_pole_angle)
        self.last_pole_angle = pole_angle
        
        # 计算奖励
        x_threshold = 2.4
        theta_threshold_radians = 0.21
        
        done = bool(
            cart_pos < -x_threshold
            or cart_pos > x_threshold
            or pole_angle < -theta_threshold_radians
            or pole_angle > theta_threshold_radians
        )
        
        # 简化的奖励函数，更加稳定
        if not done:
            # 主要奖励：杆的角度，越接近垂直越好
            angle_reward = 1.0 - abs(pole_angle) / theta_threshold_radians
            
            # 次要奖励：小车位置，越接近中心越好
            position_reward = 1.0 - abs(cart_pos) / x_threshold
            
            # 稳定性奖励：角速度和角度变化越小越好
            stability_reward = 1.0 - min(1.0, abs(pole_vel) / 5.0)
            
            # 组合奖励
            reward = 1.0 + 2.0 * angle_reward + 0.5 * position_reward + 0.5 * stability_reward
        else:
            # 失败惩罚
            reward = 0.0
        
        return obs, reward, done, {}
    
    def _get_observation(self):
        cart_state = p.getJointState(self.cartpole, 0)
        pole_state = p.getJointState(self.cartpole, 1)
        
        cart_pos, cart_vel = cart_state[0], cart_state[1]
        pole_angle, pole_vel = pole_state[0], pole_state[1]
        
        return np.array([cart_pos, cart_vel, pole_angle, pole_vel], dtype=np.float32)
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        p.disconnect(self.client)


def make_env(gui=False, randomize_params=False):
    """创建环境的辅助函数"""
    def _init():
        env = PyBulletCartPoleEnv(gui=gui, randomize_params=randomize_params)
        return env
    return _init


def validate_model(model_path, num_episodes=5, render=True):
    """
    加载并验证训练好的模型
    
    参数：
        model_path: 模型文件路径
        num_episodes: 验证轮数
        render: 是否渲染显示
    """
    # 创建GUI环境用于验证
    env = DummyVecEnv([lambda: PyBulletCartPoleEnv(gui=render)])
    # 添加归一化包装器，但只用于观测值
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    
    # 加载模型
    model = PPO.load(model_path)
    print(f"加载模型: {model_path}")
    
    # 设置环境的统计数据（如果保存了VecNormalize状态）
    normalize_path = model_path.replace(".zip", "") + "_vecnormalize.pkl"
    if os.path.exists(normalize_path):
        print(f"加载环境归一化状态: {normalize_path}")
        env = VecNormalize.load(normalize_path, env)
        # 不要更新统计数据
        env.training = False
        env.norm_reward = False
    
    # 进行多轮验证
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if render:
                time.sleep(1/240.0)  # 控制显示速度
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: 奖励 = {episode_reward}, 步数 = {episode_length}")
    
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print("\n验证结果:")
    print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"平均步数: {mean_length:.2f} +/- {std_length:.2f}")
    
    env.close()
    return mean_reward, std_reward


class CustomMetricsCallback(BaseCallback):
    """
    收集训练过程中的详细指标和诊断信息
    """
    def __init__(self, verbose=0, log_dir="./metrics", log_freq=1000, plot_freq=10000):
        super(CustomMetricsCallback, self).__init__(verbose)
        self.metrics = {
            "timesteps": [],
            "mean_reward": [],
            "mean_ep_length": [],
            "approx_kl": [],
            "entropy_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "explained_variance": [],
            "clip_fraction": [],
            "clip_range": [],
            "learning_rate": [],
            "n_updates": []
        }
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.plot_freq = plot_freq
        self.log_file = os.path.join(log_dir, "training_metrics.json")
        self.best_mean_reward = -np.inf
    
    def _on_training_start(self):
        # 确保日志目录存在
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # 记录初始配置
        model_config = {
            "policy_type": self.model.policy.__class__.__name__,
            "learning_rate": self.model.learning_rate if not callable(self.model.learning_rate) else "scheduled",
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "ent_coef": self.model.ent_coef,
            "vf_coef": self.model.vf_coef,
            "policy_kwargs": str(self.model.policy_kwargs)
        }
        
        with open(os.path.join(self.log_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=4)
            
        print(f"训练配置保存至 {os.path.join(self.log_dir, 'model_config.json')}")
        
    def _on_step(self):
        # 每个步骤收集数据
        if self.n_calls % self.log_freq == 0:
            # 获取训练相关指标
            approx_kl = float(self.model.logger.name_to_value.get("train/approx_kl", 0))
            entropy_loss = float(self.model.logger.name_to_value.get("train/entropy_loss", 0))
            policy_loss = float(self.model.logger.name_to_value.get("train/policy_gradient_loss", 0))
            value_loss = float(self.model.logger.name_to_value.get("train/value_loss", 0))
            explained_var = float(self.model.logger.name_to_value.get("train/explained_variance", 0))
            clip_fraction = float(self.model.logger.name_to_value.get("train/clip_fraction", 0))
            clip_range = float(self.model.logger.name_to_value.get("train/clip_range", 0))
            learning_rate = float(self.model.logger.name_to_value.get("train/learning_rate", 0))
            n_updates = int(self.model.logger.name_to_value.get("train/n_updates", 0))
            
            # 获取episode相关指标
            mean_reward = 0
            mean_ep_length = 0
            
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                mean_ep_length = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            # 更新指标字典
            self.metrics["timesteps"].append(self.num_timesteps)
            self.metrics["mean_reward"].append(mean_reward)
            self.metrics["mean_ep_length"].append(mean_ep_length)
            self.metrics["approx_kl"].append(approx_kl)
            self.metrics["entropy_loss"].append(entropy_loss)
            self.metrics["policy_loss"].append(policy_loss)
            self.metrics["value_loss"].append(value_loss)
            self.metrics["explained_variance"].append(explained_var)
            self.metrics["clip_fraction"].append(clip_fraction)
            self.metrics["clip_range"].append(clip_range)
            self.metrics["learning_rate"].append(learning_rate)
            self.metrics["n_updates"].append(n_updates)
            
            # 保存指标到文件
            with open(self.log_file, "w") as f:
                json.dump(self.metrics, f, indent=4)
            
            # 打印训练进度
            if self.verbose > 0:
                print(f"\nStep: {self.num_timesteps}")
                print(f"Mean Reward: {mean_reward:.2f}")
                print(f"Mean Episode Length: {mean_ep_length:.2f}")
                print(f"Approx KL: {approx_kl:.5f}")
                print(f"Value Loss: {value_loss:.2f}")
                print(f"Policy Loss: {policy_loss:.5f}")
                print(f"Explained Variance: {explained_var:.5f}")
            
            # 跟踪最佳奖励
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"新的最佳平均奖励: {self.best_mean_reward:.2f}")
            
            # 定期绘制学习曲线
            if self.n_calls % self.plot_freq == 0:
                self._plot_learning_curves()
        
        return True
    
    def _on_training_end(self):
        # 训练结束后保存最终指标和绘制学习曲线
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
        
        print(f"训练指标保存至 {self.log_file}")
        self._plot_learning_curves()
    
    def _plot_learning_curves(self):
        """绘制学习曲线并保存为图片"""
        if len(self.metrics["timesteps"]) < 2:
            return
        
        # 创建图表目录
        plots_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 绘制奖励和回合长度
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics["timesteps"], self.metrics["mean_reward"])
        plt.title("Mean Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics["timesteps"], self.metrics["mean_ep_length"])
        plt.title("Mean Episode Length")
        plt.xlabel("Timesteps")
        plt.ylabel("Length")
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics["timesteps"], self.metrics["value_loss"])
        plt.title("Value Loss")
        plt.xlabel("Timesteps")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics["timesteps"], self.metrics["policy_loss"])
        plt.title("Policy Loss")
        plt.xlabel("Timesteps")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "performance.png"))
        plt.close()
        
        # 绘制其他训练指标
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 2, 1)
        plt.plot(self.metrics["timesteps"], self.metrics["approx_kl"])
        plt.title("Approx KL Divergence")
        plt.xlabel("Timesteps")
        plt.ylabel("KL")
        plt.grid(True)
        
        plt.subplot(3, 2, 2)
        plt.plot(self.metrics["timesteps"], self.metrics["entropy_loss"])
        plt.title("Entropy Loss")
        plt.xlabel("Timesteps")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.subplot(3, 2, 3)
        plt.plot(self.metrics["timesteps"], self.metrics["explained_variance"])
        plt.title("Explained Variance")
        plt.xlabel("Timesteps")
        plt.ylabel("Variance")
        plt.grid(True)
        
        plt.subplot(3, 2, 4)
        plt.plot(self.metrics["timesteps"], self.metrics["clip_fraction"])
        plt.title("Clip Fraction")
        plt.xlabel("Timesteps")
        plt.ylabel("Fraction")
        plt.grid(True)
        
        plt.subplot(3, 2, 5)
        plt.plot(self.metrics["timesteps"], self.metrics["learning_rate"])
        plt.title("Learning Rate")
        plt.xlabel("Timesteps")
        plt.ylabel("Rate")
        plt.grid(True)
        
        plt.subplot(3, 2, 6)
        plt.plot(self.metrics["timesteps"], self.metrics["n_updates"])
        plt.title("Number of Updates")
        plt.xlabel("Timesteps")
        plt.ylabel("Updates")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "training_metrics.png"))
        plt.close()
        
        if self.verbose > 0:
            print(f"学习曲线保存至 {plots_dir}")


def main(mode='train'):
    if mode == 'train':
        # 创建目录
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./tensorboard", exist_ok=True)
        
        # 检查CUDA是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n使用设备: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            # 设置CUDA的工作方式以提高性能
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 增加并行环境数量以提高GPU利用率
        n_envs = 16  # 增加并行环境数量
        
        # 创建向量化环境，启用参数随机化以提高泛化能力
        env_fns = [make_env(randomize_params=True) for _ in range(n_envs)]
        # 使用SubprocVecEnv代替DummyVecEnv以实现真正的并行处理
        env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
        # 添加观察归一化以提高训练稳定性
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        
        eval_env = None

        try:
            # 创建用于评估的环境
            eval_env = DummyVecEnv([make_env()])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

            # 创建评估回调
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path="./models/",
                log_path="./logs/",
                eval_freq=10000,  # 评估频率
                deterministic=True,
                render=False,
                n_eval_episodes=5,  # 减少评估轮数，避免卡住
                verbose=1
            )
            
            # 创建自定义指标回调
            metrics_callback = CustomMetricsCallback(
                verbose=1,
                log_dir="./metrics",
                log_freq=1000,  # 每1000步记录一次
                plot_freq=10000  # 每10000步绘图一次
            )
            
            # 早停回调，如果模型表现足够好就停止训练
            from stable_baselines3.common.callbacks import BaseCallback
            
            class EarlyStoppingCallback(BaseCallback):
                """
                当达到指定奖励时停止训练的回调
                """
                def __init__(self, check_freq: int, min_reward: float, eval_callback: EvalCallback, min_steps: int = 100000, patience: int = 3, max_no_improvement_evals: int = 10, verbose: int = 1):
                    super(EarlyStoppingCallback, self).__init__(verbose)
                    self.check_freq = check_freq
                    self.min_reward = min_reward
                    self.min_steps = min_steps  # 最小训练步数
                    self.patience = patience  # 连续达到目标的次数
                    self.max_no_improvement_evals = max_no_improvement_evals  # 最大无改进评估次数
                    self.count_success = 0  # 计数器
                    self.no_improvement_count = 0  # 无改进计数器
                    self.best_mean_reward = -np.inf
                    self.eval_callback = eval_callback  # 保存评估回调的引用
                    self.last_eval_time = time.time()  # 上次评估时间
                    
                def _init_callback(self) -> None:
                    # 初始化回调时调用
                    pass
                    
                def _on_step(self) -> bool:
                    # 每一步调用，返回False会停止训练
                    # 确保至少训练了最小步数
                    if self.n_calls < self.min_steps:
                        if self.n_calls % (self.min_steps // 10) == 0 and self.verbose > 0:
                            print(f"训练进度: {self.n_calls}/{self.min_steps} 步 ({self.n_calls/self.min_steps*100:.1f}%)")
                        return True
                        
                    if self.n_calls % self.check_freq == 0:
                        # 检查评估是否超时
                        current_time = time.time()
                        if current_time - self.last_eval_time > 60:  # 如果评估超过60秒
                            print(f"警告: 评估似乎卡住了，已经运行了 {current_time - self.last_eval_time:.1f} 秒")
                            print("尝试继续训练...")
                            self.last_eval_time = current_time
                            return True
                            
                        # 从评估回调中获取最新的平均奖励
                        if hasattr(self.eval_callback, "last_mean_reward"):
                            mean_reward = self.eval_callback.last_mean_reward
                            self.last_eval_time = current_time  # 更新评估时间
                            
                            if self.verbose > 0:
                                print(f"当前平均奖励: {mean_reward:.2f}")
                                
                            # 检查是否有改进
                            if mean_reward > self.best_mean_reward:
                                self.best_mean_reward = mean_reward
                                self.no_improvement_count = 0
                            else:
                                self.no_improvement_count += 1
                                if self.verbose > 0:
                                    print(f"无改进评估次数: {self.no_improvement_count}/{self.max_no_improvement_evals}")
                                if self.no_improvement_count >= self.max_no_improvement_evals:
                                    if self.verbose > 0:
                                        print(f"早停: {self.max_no_improvement_evals}次评估无改进")
                                    return False
                            
                            # 如果达到目标奖励，增加计数器
                            if mean_reward >= self.min_reward:
                                self.count_success += 1
                                if self.verbose > 0:
                                    print(f"达到目标奖励 {mean_reward:.2f} >= {self.min_reward}，连续次数: {self.count_success}/{self.patience}")
                                # 如果连续达到目标次数达到patience，停止训练
                                if self.count_success >= self.patience:
                                    if self.verbose > 0:
                                        print(f"早停: 连续{self.patience}次达到目标奖励")
                                    return False
                            else:
                                # 如果没有达到目标，重置计数器
                                self.count_success = 0
                    return True

            # 设置运行ID（使用时间戳）
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_log = f"./tensorboard/PPO-PyBulletCartPole-{run_id}"

            # 学习率调度函数
            def linear_schedule(initial_value):
                def func(progress_remaining):
                    return progress_remaining * initial_value
                return func

            # 调整网络架构
            def create_optimized_policy_kwargs(network_type="medium"):
                if network_type == "small":
                    return dict(
                        net_arch=[dict(pi=[32, 32], vf=[32, 32])],
                        activation_fn=torch.nn.Tanh  # 对CartPole，Tanh通常比ReLU效果更好
                    )
                elif network_type == "medium":
                    return dict(
                        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        activation_fn=torch.nn.Tanh
                    )
                elif network_type == "large":
                    return dict(
                        net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])],
                        activation_fn=torch.nn.Tanh
                    )
                elif network_type == "separate":
                    # 完全分离的策略和值函数网络
                    return dict(
                        net_arch=dict(pi=[64, 64], vf=[64, 64]),
                        activation_fn=torch.nn.Tanh
                    )

            # 创建PPO模型，调整超参数以提高GPU利用率
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=linear_schedule(3e-4),  # 适当降低学习率以适应更大的批量
                n_steps=1024,  # 每个环境收集的步骤数
                batch_size=256,  # 增加批量大小以提高GPU利用率
                n_epochs=10,  # 增加epoch数，对每批数据多次训练
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=0.2,
                ent_coef=0.01,  # 增加熵系数以鼓励探索
                vf_coef=0.5,  # 增加值函数系数
                max_grad_norm=0.5,
                policy_kwargs=create_optimized_policy_kwargs(),
                verbose=1,
                tensorboard_log=tensorboard_log,
                device=device  # 明确指定设备
            )

            # 训练模型
            total_timesteps = 1000000  # 维持训练步数
            print("开始训练...")
            print(f"Tensorboard 日志目录: {tensorboard_log}")
            print(f"并行环境数量: {n_envs}")
            print(f"批量大小: {256}")
            
            # 组合回调
            early_stopping_callback = EarlyStoppingCallback(
                check_freq=10000, 
                min_reward=195,
                eval_callback=eval_callback,
                min_steps=100000,
                patience=3,
                max_no_improvement_evals=10
            )
            callbacks = [eval_callback, early_stopping_callback, metrics_callback]
            
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks
            )

            # 保存最终模型
            model.save("./models/ppo_cartpole_final")
            # 保存环境归一化状态
            env.save("./models/ppo_cartpole_final_vecnormalize.pkl")
            print("训练完成，模型和环境状态已保存")
            
        except KeyboardInterrupt:
            print("\n训练被用户中断...")
            if 'model' in locals() and model is not None:
                print("保存当前模型...")
                model.save("./models/ppo_cartpole_interrupted")
                env.save("./models/ppo_cartpole_interrupted_vecnormalize.pkl")
                print("模型和环境状态已保存为 './models/ppo_cartpole_interrupted'")
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            if 'model' in locals() and model is not None:
                print("保存当前模型...")
                model.save("./models/ppo_cartpole_error")
                try:
                    env.save("./models/ppo_cartpole_error_vecnormalize.pkl")
                except:
                    print("无法保存环境状态")
                print("模型已保存为 './models/ppo_cartpole_error'")
        finally:
            # 确保环境被正确关闭
            if 'env' in locals() and env is not None:
                env.close()
            if 'eval_env' in locals() and eval_env is not None:
                eval_env.close()
                
        # 返回最终或中断时保存的模型路径
        if os.path.exists("./models/ppo_cartpole_final.zip"):
            return "./models/ppo_cartpole_final"
        elif os.path.exists("./models/ppo_cartpole_interrupted.zip"):
            return "./models/ppo_cartpole_interrupted"
        elif os.path.exists("./models/ppo_cartpole_error.zip"):
            return "./models/ppo_cartpole_error"
        elif os.path.exists("./models/best_model.zip"):
            return "./models/best_model"
        return None
    
    elif mode == 'validate':
        # 验证最新训练的模型
        model_paths = [
            "./models/ppo_cartpole_final",
            "./models/ppo_cartpole_interrupted",
            "./models/ppo_cartpole_error",
            "./models/best_model"
        ]
        
        # 查找存在的模型文件
        model_path = None
        for path in model_paths:
            if os.path.exists(path + ".zip"):
                model_path = path
                break
                
        if model_path is None:
            print("错误：找不到训练好的模型文件！")
            return
        
        print(f"使用模型: {model_path}")
        validate_model(model_path, num_episodes=5, render=True)


def test_model_architectures():
    # 测试不同网络架构
    architectures = ["small", "medium", "large", "separate"]
    
    results = {}
    for arch in architectures:
        print(f"测试架构: {arch}")
        
        # 创建环境
        env = make_vec_env(lambda: PyBulletCartPoleEnv(randomize_params=True), n_envs=8)
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        
        # 创建并训练模型
        model = PPO(
            "MlpPolicy", 
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs=create_optimized_policy_kwargs(arch),
            verbose=1,
            tensorboard_log=f"./tensorboard/ppo_cartpole_{arch}"
        )
        
        # 添加自定义回调进行详细记录
        custom_callback = CustomCallback()
        
        # 训练较短时间来快速比较
        model.learn(total_timesteps=200000, callback=custom_callback)
        
        # 评估模型
        mean_reward, _ = evaluate_policy(model, make_vec_env(lambda: PyBulletCartPoleEnv(gui=False), n_envs=1), n_eval_episodes=20)
        results[arch] = mean_reward
        
        # 保存模型
        model.save(f"./models/ppo_cartpole_{arch}")
        env.save(f"./models/ppo_cartpole_{arch}_vecnormalize.pkl")
        
    # 输出比较结果
    print("\n架构性能比较:")
    for arch, reward in results.items():
        print(f"{arch}: 平均奖励 = {reward:.2f}")
    
    return results


def optimize_hyperparams_for_architecture(best_arch):
    # 创建参数网格
    param_grid = {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "n_steps": [512, 1024, 2048],
        "batch_size": [32, 64, 128],
        "gae_lambda": [0.9, 0.95, 0.99],
        "ent_coef": [0.0, 0.005, 0.01]
    }
    
    # 创建参数组合
    import itertools
    param_combos = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    best_reward = -np.inf
    best_params = None
    
    # 测试一部分组合以节省时间
    for i, combo in enumerate(param_combos[:10]):  # 限制测试数量
        params = dict(zip(param_names, combo))
        print(f"测试参数组合 {i+1}/10: {params}")
        
        # 创建环境
        env = make_vec_env(lambda: PyBulletCartPoleEnv(randomize_params=True), n_envs=8)
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        
        # 创建并训练模型
        model = PPO(
            "MlpPolicy", 
            env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=10,
            gamma=0.99,
            gae_lambda=params["gae_lambda"],
            ent_coef=params["ent_coef"],
            policy_kwargs=create_optimized_policy_kwargs(best_arch),
            verbose=0,
            tensorboard_log=f"./tensorboard/ppo_hyperparam_opt"
        )
        
        # 训练较短时间
        model.learn(total_timesteps=150000)
        
        # 评估
        mean_reward, _ = evaluate_policy(model, make_vec_env(lambda: PyBulletCartPoleEnv(gui=False), n_envs=1), n_eval_episodes=20)
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params
    
    print(f"\n最佳参数: {best_params}")
    print(f"最佳平均奖励: {best_reward:.2f}")
    
    return best_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PPO倒立摆训练与验证')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate'],
                      help='运行模式：train（训练）或validate（验证）')
    args = parser.parse_args()
    
    main(mode=args.mode) 