
from math import sin
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import time

import numpy as np
from scipy.spatial.transform import Rotation as R


target_angle = 1.413
kp = 12
kd = 2.001
time_data = []
position_data = []
orientation_data = []

xml_file = 'model_actuator.xml'
model = mujoco.MjModel.from_xml_path(xml_file)
data = mujoco.MjData(model)

def pd (current_angle, target_angle, prev_error, dt):
    error = target_angle - current_angle
    d_error = (error - prev_error) / dt
    control_signal = kp * error + kd * d_error
    return control_signal, error

mujoco.mj_forward(model, data)
simulation_time = 10
timestep = model.opt.timestep

prev_error_R1 = 0
prev_error_R2 = 0

R1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R1")
R2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R2")
A_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A")
B_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "B")
end_ef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "s8")


with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    total_simulation_time = start_time 
    
    while viewer.is_running() and time.time() - start_time < simulation_time:
        mujoco.mj_step(model, data)


    
        for t in np.arange(0, simulation_time, timestep) :
            R1 = data.qpos[R1_id]
            R2 = data.qpos[R2_id]

            control_R1, prev_error_R1 = pd(R1, target_angle, prev_error_R1, timestep)
            control_R2, prev_error_R2 = pd(R2, target_angle, prev_error_R1, timestep)

            data.ctrl[A_id] = control_R1
            data.ctrl[B_id] = control_R1

            mujoco.mj_step(model, data)
            time_data.append(t)
            position_data.append(data.site_xpos[end_ef].copy())

            rotation_matrix = data.site_xmat[end_ef].reshape(3, 3)
            r = R.from_matrix(rotation_matrix)
            orientation = r.as_euler('xyz', degrees=False)
            orientation_data.append(orientation)

    viewer.sync()


position_data = np.array(position_data)
orientation_data = np.array(orientation_data)





# Визуализация данных
plt.figure(figsize=(12, 6))


plt.plot(time_data, position_data[0], label='X Position')
plt.plot(time_data, position_data[1], label='Y Position')
plt.plot(time_data, position_data[2], label='Z Position')
plt.title('Position of End Effector Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()

# График ориентации
plt.subplot(2, 1, 2)
plt.plot(time_data, orientation_data[:, 0], label='Roll (X)')
plt.plot(time_data, orientation_data[:, 1], label='Pitch (Y)')
plt.plot(time_data, orientation_data[:, 2], label='Yaw (Z)')
plt.title('Orientation of End Effector Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Orientation (radians)')
plt.legend()
plt.grid()

# Показать графики
plt.tight_layout()
plt.show()