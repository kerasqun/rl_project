import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt

# ---------------------------
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
# 加载 cartpole 模型，初始位置稍微抬高以避免与地面碰撞
cartpoleId = p.loadURDF("cartpole.urdf", [0, 0, 0.1])

# 关闭默认的电机控制，使我们可以手动施加控制力
# 这里假设：
#   - 关节0 为小车沿水平方向的平移关节（prismatic joint）
#   - 关节1 为倒立摆的铰链关节（hinge joint）
p.setJointMotorControl2(cartpoleId, 0, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(cartpoleId, 1, controlMode=p.VELOCITY_CONTROL, force=0)

# ---------------------------
# 添加用户调试参数：kp、kd、ki 和 reset 按钮
# kp, kd 用于倒立摆的 PD 控制；ki 用于小车位置误差的积分控制
kp_slider = p.addUserDebugParameter("kp", 0, 200, 100)
kd_slider = p.addUserDebugParameter("kd", 0, 100, 10)
ki_slider = p.addUserDebugParameter("ki", 0, 10, 0.05)
# 添加一个 Reset 滑动条（范围 0 到 1，初始值为 0）
reset_slider = p.addUserDebugParameter("Reset", 0, 1, 0)

print("调节滑动条可以实时调整 kp、kd、ki；拖动 Reset 滑块（超过0.5）重置环境。")


def reset_cartpole():
    """
    重置 cartpole 机器人状态，将小车位置、倒立摆角度与角速度均归零，
    同时将机器人基座位置恢复到初始位置 [0,0,0.1]。
    """
    p.resetBasePositionAndOrientation(cartpoleId, [0, 0, 0.1], [0, 0, 0, 1])
    p.resetJointState(cartpoleId, 0, targetValue=0, targetVelocity=0)
    p.resetJointState(cartpoleId, 1, targetValue=0, targetVelocity=0)
    print("环境已重置。")


# 用于记录仿真数据的数组
time_list = []
cart_error_list = []
pole_angle_error_list = []
pole_vel_error_list = []

# 初始化小车位置误差积分
cart_integral_error = 0
desired_cart_pos = 0  # 目标小车 x 坐标

# 定义仿真步数，例如运行 5000 步
simulation_duration = 5.0  # 秒
steps = int(simulation_duration / timeStep)

# 定义一个变量，标识是否已经施加过初始外力
initial_push_applied = False
# 设置初始推力大小，例如 50 牛顿，方向为 x 轴正方向
push_force = 50

# ---------------------------
# 主控制循环
for i in range(steps):
    # 在第一个 step 时施加初始推力（也可以选择连续几个 step 施加）
    if not initial_push_applied:
        # 注意：这里 linkIndex 的选择需要依据你的 URDF 文件，
        # 如果 cartpole 的基座 linkIndex 为 -1，则表示整个物体，否则可以使用实际 link 的索引（例如 0）。
        p.applyExternalForce(objectUniqueId=cartpoleId,
                             linkIndex=0,
                             forceObj=[push_force, 0, 0],
                             posObj=[0, 0, 0],
                             flags=p.WORLD_FRAME)
        initial_push_applied = True

    # 实时读取控制参数
    kp = p.readUserDebugParameter(kp_slider)
    kd = p.readUserDebugParameter(kd_slider)
    ki = p.readUserDebugParameter(ki_slider)

    # 检测 Reset 滑块，若其值大于0.5则重置环境
    reset_val = p.readUserDebugParameter(reset_slider)
    if reset_val > 0.5:
        reset_cartpole()
        # 重置积分误差
        cart_integral_error = 0
        time.sleep(0.2)  # 短暂等待
        # 移除并重新添加 Reset 控件以恢复初始值
        p.removeUserDebugItem(reset_slider)
        reset_slider = p.addUserDebugParameter("Reset", 0, 1, 0)

    # 获取关节状态：
    # 关节0（小车平移）返回 (position, velocity, reaction forces, torque)
    cart_state = p.getJointState(cartpoleId, 0)
    # 关节1（倒立摆）返回 (angle, angular velocity, reaction forces, torque)
    pole_state = p.getJointState(cartpoleId, 1)

    cart_pos = cart_state[0]  # 小车的水平位置
    cart_vel = cart_state[1]  # 小车水平速度
    pole_angle = pole_state[0]  # 倒立摆偏离竖直方向的角度（单位：弧度）
    pole_vel = pole_state[1]  # 倒立摆角速度（单位：弧度/秒）

    print(f'cart_pos: {cart_pos}')
    # ---------------------------
    # 设计控制器
    # 1. 倒立摆 PD 控制部分（目标：保持 pole_angle 为 0）
    pole_error = -pole_angle  # 倒立摆角度误差
    pole_d_error = -pole_vel  # 倒立摆角速度

    # 2. 小车位置 PI 控制部分（目标：使小车位置回到 0）
    cart_error = -(desired_cart_pos - cart_pos)
    cart_integral_error += cart_error * timeStep

    # 3. 组合控制力（注意这里各部分的正负号需与物理系统的定义一致）
    #    此处的设计思路：若倒立摆向右倾（pole_angle>0），则小车应向右运动；
    #    同时，若小车位置偏右（cart_error<0），则加上负的积分项以推动小车向左回归。
    control_force = (kp * pole_error + kd * pole_d_error) + (ki * cart_integral_error)

    # 将计算得到的控制力以 TORQUE_CONTROL 模式施加到小车关节（关节0）上
    p.setJointMotorControl2(cartpoleId, 0, controlMode=p.TORQUE_CONTROL, force=-control_force)

    # 执行仿真一步
    p.stepSimulation()

    # 记录每一步的时间和误差值
    current_time = i * timeStep
    time_list.append(current_time)
    cart_error_list.append(cart_error)
    pole_angle_error_list.append(pole_error)
    pole_vel_error_list.append(pole_d_error)

    if is_render:
        time.sleep(timeStep)


# 仿真结束后断开连接
p.disconnect()

# ---------------------------
# 绘制误差曲线
plt.figure(figsize=(12, 10))

# 小车位置误差曲线
plt.subplot(3, 1, 1)
plt.plot(time_list, cart_error_list, label="Cart Position Error")
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.legend()
plt.grid(True)

# 倒立摆角度误差曲线
plt.subplot(3, 1, 2)
plt.plot(time_list, pole_angle_error_list, label="Pole Angle Error", color='orange')
plt.xlabel("Time (s)")
plt.ylabel("Error (rad)")
plt.legend()
plt.grid(True)

# 倒立摆角速度误差曲线
plt.subplot(3, 1, 3)
plt.plot(time_list, pole_vel_error_list, label="Pole Angular Velocity Error", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Error (rad/s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()