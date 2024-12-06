import numpy as np
import matplotlib.pyplot as plt

def mass_spring_system(state:list):
    x, dx = state
    b = 0.03
    m = 1 
    g = 9.8
    k = 18.8
    ddx = - 1/m * (b*dx + k*x)
    return np.array([dx, ddx])

def fun_pendulum_rk4(xk, h):
    f1 = mass_spring_system(xk)
    f2 = mass_spring_system(xk + 0.5*h*f1)
    f3 = mass_spring_system(xk + 0.5*h*f2)
    f4 = mass_spring_system(xk + h*f3)
    return xk + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)

def pendulum_rk4(fun, x0, Tf, h):
    t = np.arange(0, Tf, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = fun_pendulum_rk4(x_hist[:, k], h)
    
    return x_hist, t

x0 = np.array([0.24, 0])
x_hist, t_hist = fun_pendulum_rk4(mass_spring_system, x0, 15, 0.1)

plt.plot(t_hist, x_hist[0,:], label="$x$")
plt.plot(t_hist, x_hist[1,:], label="$dx$")
plt.xlabel('Time, [sec]')
plt.ylabel('state')
plt.legend()
plt.grid()
plt.show()