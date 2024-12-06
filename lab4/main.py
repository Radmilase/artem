import json
from math import sin
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import time

import numpy as np

model = mujoco.MjModel.from_xml_path("/D:/Itmo/IMRS/robot_simulation/ex4/solution/model.xml")
data = mujoco.MjData(model)

start_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:

  while viewer.is_running() and (time.time() - start_time) < 100:
    data.ctrl = [sin(time.time() - start_time)]

    mujoco.mj_step(model, data)
    viewer.sync()  

plt.plot 