import gym
import numpy as np
import os
import pybullet as p
import pybullet_data
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import torch


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
    
    # 加载模型
    model = PPO.load(model_path)
    print(f"加载模型: {model_path}")
    
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


def main(mode='train'):
    if mode == 'train':
        # 创建目录
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./tensorboard", exist_ok=True)

        # 创建向量化环境，启用参数随机化以提高泛化能力
        env = DummyVecEnv([make_env(randomize_params=True)])
        eval_env = None

        try:
            # 创建用于评估的环境
            eval_env = DummyVecEnv([make_env()])

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

            # 创建PPO模型，调整超参数
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=linear_schedule(1e-3),  # 增大初始学习率
                n_steps=2048,  # 增加步数
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=0.2,  # 添加值函数裁剪
                ent_coef=0.005,  # 减小熵系数
                max_grad_norm=0.5,  # 梯度裁剪，增加稳定性
                policy_kwargs=dict(
                    net_arch=[dict(pi=[64, 32], vf=[64, 32])],  # 使用更简单的网络结构
                    activation_fn=torch.nn.ReLU  # 使用ReLU激活函数
                ),
                verbose=1,
                tensorboard_log=tensorboard_log
            )

            # 训练模型
            total_timesteps = 1000000  # 增加训练步数到100万
            print("开始训练...")
            print(f"Tensorboard 日志目录: {tensorboard_log}")
            
            # 组合回调
            early_stopping_callback = EarlyStoppingCallback(
                check_freq=10000, 
                min_reward=195,  # 降低目标奖励值，CartPole任务通常认为平均奖励达到195就算解决
                eval_callback=eval_callback,
                min_steps=100000,  # 确保至少训练10万步
                patience=3,  # 连续3次达到目标奖励才停止
                max_no_improvement_evals=10  # 最大无改进评估次数
            )
            callbacks = [eval_callback]
            
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks
            )

            # 保存最终模型
            model.save("./models/ppo_cartpole_final")
            print("训练完成，模型已保存")
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
        finally:
            # 确保环境被正确关闭
            if env is not None:
                env.close()
            if eval_env is not None:
                eval_env.close()
        
        return "./models/ppo_cartpole_final"
    
    elif mode == 'validate':
        # 验证最新训练的模型
        model_path = "./models/ppo_cartpole_final"
        if not os.path.exists(model_path + ".zip"):
            print("错误：找不到训练好的模型文件！")
            return
        
        validate_model(model_path, num_episodes=5, render=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PPO倒立摆训练与验证')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate'],
                      help='运行模式：train（训练）或validate（验证）')
    args = parser.parse_args()
    
    main(mode=args.mode) 