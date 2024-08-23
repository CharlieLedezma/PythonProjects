import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Pendulum rod lengths (m), bob masses (kg).
k1, k2 = 5, 5
m1, m2 = 1, 1
#     V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0
#     T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
#             2*L1*L2*th1d*th2d*np.cos(th1-th2))
u , b = 3, 1
# The gravitational acceleration (m.s-2).
g = 9.81
#     V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
#     T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
#             2*L1*L2*th1d*th2d*np.cos(th1-th2))*L2*g*np.cos(th2)
#     T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
#             2*L1*L2*th1d*th2d*np.cos(th1-th2))
    q1, z1, q2, z2 = y
#     V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
#     T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
#             2*L1*L2*th1d*th2d*np.cos(th1-th2))

    # c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    
    q1_dot = z1
    z1_dot = (u -b*z1 - k1*q1 - k2*(q1-q2))/m1
    q2_dot = z2
    z2_dot = (k2*(q1-q2))/m2
#     V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
#     T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
#             2*L1*L2*th1d*th2d*np.cos(th1-th2))2*s*(L1*z1**2*c + L2*z2**2) -
    #          (m1+m2)*g*np.sin(theta1)) / (L1*(m1 + m2*s**2))
    # theta2dot = z2
    # z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
    #          m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return q1_dot, z1_dot, q2_dot, z2_dot

# def calc_E(y):
#     """Return the total energy of the system."""

#     th1, th1d, th2, th2d = y.T
#     V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
#     T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
#             2*L1*L2*th1d*th2d*np.cos(th1-th2))
#     return T + V

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 30, 0.01
t = np.arange(0, tmax+dt, dt)
# Initial conditions: q1, dq1/dt, q2, dq1/dt.
y0 = np.array([0, 0, 0, 0])

# Do the numerical integration of the equations of motion
y = odeint(deriv, y0, t, args=(k1, k2, m1, m2, u, b))

# # Check that the calculation conserves total energy to within some tolerance.
# EDRIFT = 0.05
# # Total energy from the initial conditions
# E = calc_E(y0)
# if np.max(np.sum(np.abs(calc_E(y) - E))) > EDRIFT:
#     sys.exit('Maximum energy drift of {} exceeded.'.format(EDRIFT))

# # Unpack z and theta as a function of time
# theta1, theta2 = y[:,0], y[:,2]

# # Convert to Cartesian coordinates of the two bob positions.
# x1 = L1 * np.sin(theta1)
# y1 = -L1 * np.cos(theta1)
# x2 = x1 + L2 * np.sin(theta2)
# y2 = y1 - L2 * np.cos(theta2)
# plot the 3 sets
# Unpack z and theta as a function of time
q1, q2 = y[:,0], y[:,2]

plt.plot(t,q1,label='Disp q1')
plt.plot(t,q2,label='Disp q2')
# plt.plot(x,y3, label='plot 3')

# call with no parameters
plt.legend()

plt.show()