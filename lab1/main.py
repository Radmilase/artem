import numpy as np
import matplotlib.pyplot as plt

def mass_spring_system(state:list):
    theta, dtheta = state
    b = 0.015
    m = 0.6
    g = 9.8
    k = 10
    l = 0.6

    ddtheta = - 1/m*1**2* (b*dtheta + m*g*1*np.sin(theta) + k*theta)
    return np.array([dtheta, ddtheta])

def runge_kutta_4(fun:object, x0:np.ndarray, tf:float, h:float) -> tuple:
    t = np.arange(0, tf+h, h)
    theta_hist = np.zeros((len(theta0), len(t)))
    theta_hist[:, 0] = theta0
    for n in range(len(t)-1):
        k1 = fun(theta_hist[:, n])
        k2 = fun(theta_hist[:, n] + h/2*k1)
        k3 = fun(theta_hist[:, n] + h/2*k2)
        k4 = fun(theta_hist[:, n] + h*k3)
        theta_hist[:, n+1] = theta_hist[:, n] + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return theta_hist, t
    

def pendulum_bckw_euler(fun:object, x0:np.ndarray, t_f:float, h:float) -> tuple:
 
    t = np.arange(0, t_f+h, h)

    theta_hist = np.zeros((len(theta0), len(t)))
    theta_hist[:, 0] = x0

    for n in range(len(t)-1):
        e = 1 # an initial error value
        theta_hist[:, n+1] = theta_hist[:, n] 
        while e > 1e-8:
            theta_n = theta_hist[:, n] + h*fun(theta_hist[:, n+1])
            e = np.linalg.norm(theta_n - theta_hist[:,n+1]) # an error value
            theta_hist[:, n+1] = theta_n
 
    return theta_hist, t


def forward_euler(fun:object, x0:np.ndarray, Tf:float, h:float) -> tuple:
    t = np.arange(0, Tf + h, h)

    theta_hist = np.zeros((len(theta0), len(t)))
    theta_hist[:, 0] = x0

    for n in range(len(t)-1):
        theta_hist[:, n + 1] = theta_hist[:, n] + h * fun(theta_hist[:, n])
    
    return theta_hist, t

# Решение
theta0 = np.array([0.4869619855, 0])
theta_hist1, t_hist1 = forward_euler(mass_spring_system, theta0, 10, 0.01)
theta_hist3, t_hist3 = runge_kutta_4(mass_spring_system, theta0, 10, 0.1)
theta_hist2, t_hist2 = pendulum_bckw_euler(mass_spring_system, theta0, 10, .01)



plt.plot(t_hist1, theta_hist1[1, :], label="$dtheta-Forward\_Euler$")

# plt.plot(t_hist1, theta_hist1[0, :], label="$theta-Forward\_Euler$")
# plt.plot(t_hist2, theta_hist2[0, :], label="$theta-Back\_Euler$")
plt.plot(t_hist2, theta_hist2[1, :], label="$dtheta-Back\_Euler$")
# plt.plot(t_hist3, theta_hist3[0,:], label="$theta-RK4$")
plt.plot(t_hist3, theta_hist3[1,:], label="$dtheta-RK4$")


plt.xlabel('Time, [sec]')
plt.ylabel('state')
plt.legend()
plt.grid()
plt.show()