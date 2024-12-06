import numpy as np
import matplotlib.pyplot as plt

# Парметрый 

def mass_spring_system(state:list):
    b = 0.5
    m = 1 
    g = 9.8
    k = 20
    x, dx = state
    ddx = - 1/m * (b*dx + k*x)

    return np.array([dx, ddx])

# Явный метод Эйлера
def forward_euler(fun:object, x0:np.ndarray, Tf:float, h:float) -> tuple:
    t = np.arange(0, Tf + h, h)

    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0

    for k in range(len(t)-1):
        x_hist[:, k + 1] = x_hist[:, k] + h * fun(x_hist[:, k])
    
    return x_hist, t

# Решение
x0 = np.array([0.24, 0])
x_hist_fwr, t_fwr = forward_euler(mass_spring_system, x0, 10, 0.01)

# Вывод результатов
plt.plot(t_fwr, x_hist_fwr[0,:], label="Forward Euler")
plt.plot(t_fwr, x_hist_fwr[1,:], label="Forward Euler")


plt.xlabel('Time, [sec]') 
plt.ylabel('x') 
plt.legend()
plt.grid()
plt.show()