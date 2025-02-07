import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are

# =============================================================================
# 控制器基类（未来可扩展其它控制方法，如强化学习控制器）
# =============================================================================
class BaseController:
    def update(self, cart_state, pole_state, dt):
        """
        根据当前状态更新控制力，返回施加于小车的控制力。
        cart_state: (位置, 速度)
        pole_state: (角度, 角速度)
        dt: 时间步长
        """
        raise NotImplementedError("必须实现 update 方法")

    def reset(self):
        """重置控制器状态（如积分项）"""
        pass

# =============================================================================
# PID 控制器（倒立摆 PD + 小车 PI）
# =============================================================================
class PIDController(BaseController):
    def __init__(self, kp=100, kd=10, ki=0.05, desired_cart_pos=0, integral_limit=10):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.desired_cart_pos = desired_cart_pos
        self.cart_integral_error = 0.0
        self.integral_limit = integral_limit

    def update(self, cart_state, pole_state, dt):
        # 提取状态信息
        cart_pos, cart_vel = cart_state[0], cart_state[1]
        pole_angle, pole_vel = pole_state[0], pole_state[1]

        # 1. 倒立摆 PD 控制部分（目标：pole_angle=0）
        pole_error = -pole_angle      # 若 pole_angle > 0 则表示偏离竖直，需施加反向力
        pole_d_error = -pole_vel

        # 2. 小车 PI 控制部分（目标：cart_pos = desired_cart_pos）
        cart_error = self.desired_cart_pos - cart_pos
        self.cart_integral_error += cart_error * dt
        self.cart_integral_error = max(min(self.cart_integral_error, self.integral_limit), -self.integral_limit)

        # 3. 组合控制信号（注意各部分正负号需与物理系统一致）
        control_force = self.kp * pole_error + self.kd * pole_d_error + self.ki * self.cart_integral_error
        return control_force

    def reset(self):
        self.cart_integral_error = 0.0

# =============================================================================
# LQR 控制器设计
# =============================================================================
class LQRController(BaseController):
    def __init__(self, A, B, Q, R):
        """
        A, B: 状态空间模型矩阵
        Q, R: LQR 设计中的状态与输入权重矩阵
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        # 求解连续时间 Riccati 方程：AᵀP + PA - PBR⁻¹BᵀP + Q = 0
        P = solve_continuous_are(A, B, Q, R)
        # 计算最优反馈增益：K = R⁻¹BᵀP
        self.K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
        # 对于 LQR，控制律为 u = -Kx。
        # 由于在仿真中我们施加力时调用了 p.setJointMotorControl2(..., force=-control_force)
        # 因此我们在 update() 中返回 np.dot(K, x)，这样外部施加的力为 -Kx。
        print("LQR 增益矩阵 K = ", self.K)

    def update(self, cart_state, pole_state, dt):
        # 组合状态向量：x = [x, x_dot, theta, theta_dot]
        x = np.array([cart_state[0], cart_state[1], pole_state[0], pole_state[1]])
        # 根据标准 LQR 控制律 u = -Kx，而外部会再取负值，所以这里返回 np.dot(K, x)
        control_force = np.dot(self.K, x)
        return control_force

    def reset(self):
        pass

# =============================================================================
# PyBullet 仿真初始化
# =============================================================================

# 初始化 PyBullet 仿真（GUI 模式）
is_render = True
sim_mode = p.GUI if is_render else p.DIRECT
physicsClient = p.connect(sim_mode)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 设置重力和时间步长
p.setGravity(0, 0, -9.8)
timeStep = 1. / 240.
p.setTimeStep(timeStep)

# 加载地面和 Cart-Pole 模型
planeId = p.loadURDF("plane.urdf")
# 为避免与地面碰撞，cartpole 模型初始位置稍微抬高
cartpoleId = p.loadURDF("cartpole.urdf", [0, 0, 0.1])

# 关闭默认的电机控制，使我们可以手动施加控制力
# 这里假设：
#   - 关节0 为小车沿水平方向的平移关节（prismatic joint）
#   - 关节1 为倒立摆的铰链关节（hinge joint）
p.setJointMotorControl2(cartpoleId, 0, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(cartpoleId, 1, controlMode=p.VELOCITY_CONTROL, force=0)

# ---------------------------
# 添加用户调试参数（仅用于 PID 控制器调参）
kp_slider = p.addUserDebugParameter("kp", 0, 200, 100)
kd_slider = p.addUserDebugParameter("kd", 0, 100, 10)
ki_slider = p.addUserDebugParameter("ki", 0, 10, 0.05)
# 添加 Reset 滑动条（范围 0 到 1，初始值为 0）
reset_slider = p.addUserDebugParameter("Reset", 0, 1, 0)

print("拖动 Reset 滑块（超过0.5）可重置环境。")

def reset_cartpole():
    """
    重置 cartpole 状态：将小车位置、倒立摆角度与角速度均归零，
    同时将机器人基座位置恢复到初始位置 [0,0,0.1]。
    """
    p.resetBasePositionAndOrientation(cartpoleId, [0, 0, 0.1], [0, 0, 0, 1])
    p.resetJointState(cartpoleId, 0, targetValue=0, targetVelocity=0)
    p.resetJointState(cartpoleId, 1, targetValue=0, targetVelocity=0)
    print("环境已重置。")

# =============================================================================
# 选择使用的控制器：True 为 LQR，False 为 PID
# =============================================================================
use_lqr = True

if use_lqr:
    # 设定倒立摆系统参数（需与实际仿真模型匹配）
    M = 1.0   # 小车质量
    m = 0.1   # 摆质量
    l = 0.5   # 摆长
    g = 9.8   # 重力加速度

    # 构造线性化状态空间模型：
    # 状态 x = [x, x_dot, theta, theta_dot]
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -m*g/M, 0],
                  [0, 0, 0, 1],
                  [0, 0, (M+m)*g/(M*l), 0]])
    B = np.array([[0],
                  [1/M],
                  [0],
                  [-1/(M*l)]])
    # 定义 LQR 的权重矩阵 Q 与 R，可根据设计要求调整
    Q = np.diag([10, 1, 100, 1])
    R = np.array([[0.1]])
    controller = LQRController(A, B, Q, R)
else:
    controller = PIDController()

# =============================================================================
# 主程序入口：仿真循环
# =============================================================================

# 用于记录仿真数据
time_list = []
cart_error_list = []
pole_angle_error_list = []
pole_vel_error_list = []

# 定义仿真总时长和步数
simulation_duration = 5.0  # 秒
steps = int(simulation_duration / timeStep)

# 记录是否已经施加初始扰动
initial_push_applied = False
# 初始扰动力大小（例如 50 牛顿，沿 x 轴正方向）
push_force = 50

for i in range(steps):
    # 前 20 步施加扰动，模拟外界干扰
    if i < 20:
        p.applyExternalForce(objectUniqueId=cartpoleId,
                             linkIndex=0,
                             forceObj=[push_force, 0, 0],
                             posObj=[0, 0, 0],
                             flags=p.WORLD_FRAME)
        initial_push_applied = True

    # 如果使用 PID 控制器，则实时读取调参滑块的值
    if not use_lqr:
        controller.kp = p.readUserDebugParameter(kp_slider)
        controller.kd = p.readUserDebugParameter(kd_slider)
        controller.ki = p.readUserDebugParameter(ki_slider)

    # 检测 Reset 滑块，若其值大于 0.5，则重置环境
    reset_val = p.readUserDebugParameter(reset_slider)
    if reset_val > 0.5:
        reset_cartpole()
        controller.reset()
        time.sleep(0.2)
        p.removeUserDebugItem(reset_slider)
        reset_slider = p.addUserDebugParameter("Reset", 0, 1, 0)

    # 获取关节状态：
    # 关节0：小车平移状态 (位置, 速度, ...)
    cart_state = p.getJointState(cartpoleId, 0)
    # 关节1：倒立摆状态 (角度, 角速度, ...)
    pole_state = p.getJointState(cartpoleId, 1)

    cart_pos = cart_state[0]    # 小车水平位置
    cart_vel = cart_state[1]    # 小车水平速度
    pole_angle = pole_state[0]  # 倒立摆偏离竖直角度（单位：弧度）
    pole_vel = pole_state[1]    # 倒立摆角速度（单位：弧度/秒）

    # 使用所选控制器计算控制力
    control_force = controller.update((cart_pos, cart_vel), (pole_angle, pole_vel), timeStep)
    # 注意：由于原代码施加控制力时使用了负号，此处不论 PID 或 LQR 均调用相同方式
    p.setJointMotorControl2(cartpoleId, 0, controlMode=p.TORQUE_CONTROL, force=-control_force)

    # 执行仿真一步
    p.stepSimulation()

    # 记录数据（便于后续绘图）
    current_time = i * timeStep
    time_list.append(current_time)
    cart_error_list.append(0 - cart_pos)   # 目标小车位置为 0
    pole_angle_error_list.append(0 - pole_angle)  # 目标摆角为 0
    pole_vel_error_list.append(0 - pole_vel)

    if is_render:
        time.sleep(timeStep)

# 仿真结束后断开连接
p.disconnect()

# 绘制误差曲线
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time_list, cart_error_list, label="Cart Position Error")
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_list, pole_angle_error_list, label="Pole Angle Error", color='orange')
plt.xlabel("Time (s)")
plt.ylabel("Error (rad)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_list, pole_vel_error_list, label="Pole Angular Velocity Error", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Error (rad/s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
