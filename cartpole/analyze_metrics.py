import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_metrics(metrics_file):
    """加载训练指标文件"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return metrics


def smooth_curve(points, factor=0.8):
    """平滑曲线，用于更清晰地展示趋势"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def analyze_metrics(metrics_file, output_dir=None, smooth=True):
    """分析训练指标并生成诊断图表"""
    # 加载指标
    print(f"加载指标文件: {metrics_file}")
    metrics = load_metrics(metrics_file)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(metrics_file), "analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"分析结果将保存到: {output_dir}")
    
    # 提取关键指标
    timesteps = np.array(metrics["timesteps"])
    mean_rewards = np.array(metrics["mean_reward"])
    mean_ep_lengths = np.array(metrics["mean_ep_length"])
    value_losses = np.array(metrics["value_loss"])
    policy_losses = np.array(metrics["policy_loss"])
    explained_vars = np.array(metrics["explained_variance"])
    entropies = np.array(metrics["entropy_loss"])
    kl_divs = np.array(metrics["approx_kl"])
    
    # 打印基本统计信息
    print("\n===== 训练统计信息 =====")
    print(f"总训练步数: {timesteps[-1]}")
    print(f"最终平均奖励: {mean_rewards[-1]:.2f}")
    print(f"最高平均奖励: {np.max(mean_rewards):.2f}")
    print(f"最终回合长度: {mean_ep_lengths[-1]:.2f}")
    print(f"最终值函数损失: {value_losses[-1]:.2f}")
    print(f"最终策略损失: {policy_losses[-1]:.5f}")
    print(f"最终解释方差: {explained_vars[-1]:.5f}")
    
    # 检测训练问题
    print("\n===== 训练诊断 =====")
    
    # 检查奖励趋势
    reward_trend = np.polyfit(range(len(mean_rewards[-10:])), mean_rewards[-10:], 1)[0]
    if reward_trend < 0:
        print("⚠️ 警告: 奖励在训练末期呈下降趋势")
    elif reward_trend > 0 and mean_rewards[-1] < 195:
        print("ℹ️ 信息: 奖励在训练末期仍在上升，可能需要更长时间训练")
        
    # 检查值函数损失
    if value_losses[-1] > 1000:
        print("⚠️ 警告: 值函数损失较高，表明值函数预测不准确")
        
    # 检查解释方差
    if explained_vars[-1] < 0:
        print("⚠️ 警告: 解释方差为负，表明值函数预测质量差")
    elif explained_vars[-1] < 0.5:
        print("⚠️ 警告: 解释方差较低，值函数可能需要改进")
        
    # 检查KL散度
    if np.mean(kl_divs[-10:]) > 0.05:
        print("⚠️ 警告: KL散度较大，策略更新可能过于激进")
    elif np.mean(kl_divs[-10:]) < 0.001:
        print("⚠️ 警告: KL散度较小，学习可能过于保守")
        
    # 检查熵损失
    if entropies[-1] > -0.5:
        print("⚠️ 警告: 熵较高，策略可能过于不确定")
    elif entropies[-1] < -2:
        print("⚠️ 警告: 熵较低，策略可能过早收敛")
        
    # 绘制诊断图表
    plt.figure(figsize=(15, 10))
    
    # 奖励曲线
    plt.subplot(2, 2, 1)
    if smooth:
        plt.plot(timesteps, smooth_curve(mean_rewards), label='平滑')
    plt.plot(timesteps, mean_rewards, alpha=0.3, label='原始')
    plt.grid(True)
    plt.title('平均奖励')
    plt.xlabel('训练步数')
    plt.ylabel('奖励')
    plt.legend()
    
    # 回合长度曲线
    plt.subplot(2, 2, 2)
    if smooth:
        plt.plot(timesteps, smooth_curve(mean_ep_lengths), label='平滑')
    plt.plot(timesteps, mean_ep_lengths, alpha=0.3, label='原始')
    plt.grid(True)
    plt.title('平均回合长度')
    plt.xlabel('训练步数')
    plt.ylabel('步数')
    plt.legend()
    
    # 损失曲线
    plt.subplot(2, 2, 3)
    if smooth:
        plt.plot(timesteps, smooth_curve(value_losses), label='值函数损失(平滑)')
        plt.plot(timesteps, smooth_curve(policy_losses), label='策略损失(平滑)')
    else:
        plt.plot(timesteps, value_losses, label='值函数损失')
        plt.plot(timesteps, policy_losses, label='策略损失')
    plt.grid(True)
    plt.title('损失曲线')
    plt.xlabel('训练步数')
    plt.ylabel('损失')
    plt.legend()
    
    # 解释方差
    plt.subplot(2, 2, 4)
    plt.plot(timesteps, explained_vars)
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('解释方差')
    plt.xlabel('训练步数')
    plt.ylabel('方差')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_metrics.png'))
    
    # 绘制其他指标
    plt.figure(figsize=(15, 8))
    
    # KL散度
    plt.subplot(2, 2, 1)
    plt.plot(timesteps, kl_divs)
    plt.grid(True)
    plt.title('KL散度')
    plt.xlabel('训练步数')
    plt.ylabel('KL')
    
    # 熵损失
    plt.subplot(2, 2, 2)
    plt.plot(timesteps, entropies)
    plt.grid(True)
    plt.title('熵损失')
    plt.xlabel('训练步数')
    plt.ylabel('熵')
    
    # 剪切分数
    plt.subplot(2, 2, 3)
    plt.plot(timesteps, metrics["clip_fraction"])
    plt.grid(True)
    plt.title('剪切分数')
    plt.xlabel('训练步数')
    plt.ylabel('比例')
    
    # 学习率
    plt.subplot(2, 2, 4)
    plt.plot(timesteps, metrics["learning_rate"])
    plt.grid(True)
    plt.title('学习率')
    plt.xlabel('训练步数')
    plt.ylabel('学习率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'additional_metrics.png'))
    
    # 生成优化建议
    print("\n===== 优化建议 =====")
    
    # 基于奖励分析
    if np.max(mean_rewards) < 150:
        print("🔄 考虑增加训练步数，当前奖励水平较低")
        
    # 基于值函数分析
    if explained_vars[-1] < 0.5:
        print("🔄 考虑调整网络架构，增强值函数网络能力")
        print("   - 增加值函数网络的宽度和深度")
        print("   - 考虑使用不同的激活函数，如LeakyReLU")
        print("   - 调整vf_coef参数，增加值函数的更新权重")
    
    # 基于KL散度分析
    if np.mean(kl_divs[-10:]) > 0.05:
        print("🔄 策略更新可能过于激进")
        print("   - 减小学习率")
        print("   - 增大clip_range")
    elif np.mean(kl_divs[-10:]) < 0.001:
        print("🔄 策略更新可能过于保守")
        print("   - 增大学习率")
        print("   - 减小clip_range")
    
    # 基于熵分析
    if entropies[-1] > -0.5:
        print("🔄 策略可能过于不确定")
        print("   - 减小ent_coef，降低探索程度")
    elif entropies[-1] < -2:
        print("🔄 策略可能过早收敛")
        print("   - 增大ent_coef，增加探索")
    
    print("\n分析完成！")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='分析PPO训练指标')
    parser.add_argument('--metrics_file', type=str, default='./metrics/training_metrics.json',
                        help='训练指标JSON文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='分析结果输出目录')
    parser.add_argument('--no_smooth', action='store_true',
                        help='不平滑曲线')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics_file):
        print(f"错误: 找不到指标文件 {args.metrics_file}")
        return
    
    analyze_metrics(args.metrics_file, args.output_dir, not args.no_smooth)


if __name__ == "__main__":
    main() 