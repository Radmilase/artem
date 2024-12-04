import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

xml_file = 'model.xml'

m = mujoco.MjModel.from_xml_path(xml_file)
d = mujoco.MjData(m)

time_data = []
position_data = []

with mujoco.viewer.launch_passive(m, d) as viewer:
    start_time = time.time()
    total_simulation_time = start_time + 30 
    
    while viewer.is_running() and time.time() < total_simulation_time:
        mujoco.mj_step(m, d)


    viewer.sync()





