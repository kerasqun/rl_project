import gym
import numpy as np
from stable_baselines3 import DQN


def evaluate(model, env, n_eval_episodes=10, render=True):
    rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    mean_reward = np.mean(rewards)
    return mean_reward


def main():
    # 创建环境
    env = gym.make("CartPole-v1")

    # 创建模型
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

    total_timesteps = 50000
    eval_interval = 10000  # 每训练多少步评估一次
    best_mean_reward = -np.inf

    for timestep in range(0, total_timesteps, eval_interval):
        print(f"训练 {timestep} 到 {timestep + eval_interval} 步")
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)

        # 进行评估
        mean_reward = evaluate(model, env, n_eval_episodes=10)
        print(f"在 {timestep + eval_interval} 步时，平均奖励为: {mean_reward}")

        # 如果平均奖励达到或超过一定阈值，可以认为模型已收敛（例如 490）
        if mean_reward >= 490:
            print("模型已收敛，达到收敛标准！")
            break

        # 保存最佳模型
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            model.save("best_dqn_cartpole")

    env.close()


if __name__ == "__main__":
    main() 