import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from self_balancing_car_env import SelfBalancingCarEnv

# 创建环境并进行检查
env = SelfBalancingCarEnv()
check_env(env, warn=True)

# 创建 PPO 模型，使用 MLP 策略网络
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型，总共训练 100,000 个时间步
model.learn(total_timesteps=100000)

# 保存训练后的模型
model.save("ppo_self_balancing_car")
print("Model saved.")

# 测试训练效果
obs, _ = env.reset()
for i in range(200):
    # 根据当前状态预测动作
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print("Episode ended after {} timesteps.".format(i+1))
        break
env.close()
