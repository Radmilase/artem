import numpy as np
import time
import os
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

model = mujoco.MjModel.from_xml_path("D:/Itmo/IMRS/artem/artem/lab4/model.xml")
data = mujoco.MjData(model)

start_time = time.time()

max_left = -0.32
limit_time = 30
angle_theta=[]
actuator_pos=[]

def euler_angles(rotation_matrix):
    rotations = R.from_matrix(rotation_matrix)
    euler = rotations.as_euler('xyz', degrees=True)
    return euler

def plot(actuator_pos, y_angles):
    plt.figure(figsize=(10,6))
    plt.tight_layout()
    plt.plot(actuator_pos, y_angles, label="Theta")
    plt.xlabel('Actuator position')
    plt.ylabel('Angle(Degrees)')
    plt.legend()
    plt.grid(True)
    plt.show()

def simulation():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time=time.time()
        dx = -0.000001
        while viewer.is_running() and time.time() - start_time < limit_time:
            mujoco.mj_step(model, data)
            viewer.sync()
            if data.ctrl[0] >= max_left:
                data.ctrl[0] += dx
                actuator_pos.append(data.body('').xpos[0] * -1)
                rotation_matrix = np.array(data.body('Link1').xmat).reshape((3,3))
                angles = euler_angles(rotation_matrix)
                cur_angle = angles[1]
                angle_theta.append(cur_angle * -1 + 90)
                
            
            
        plot(np.array(actuator_pos), np.array(angle_theta))
        print(angle_theta.index(min(angle_theta)))

if __name__ == '__main__':
    simulation()