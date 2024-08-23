#https://scipython.com/blog/the-double-pendulum/
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1

# The gravitational acceleration (m.s-2).
g = 9.81

# def deriv(y, t, L1, L2, m1, m2):
#     """Return the first derivatives of y = theta1, z1, theta2, z2."""
#     theta1, z1, theta2, z2 = y
#     c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

#     theta1dot = z1
#     z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
#              (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)

#     theta2dot = z2
#     z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
#              m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
#     return theta1dot, z1dot, theta2dot, z2dot



def deriv(y, t, L1, L2, m1, m2):
    theta1 = y[0]
    z1 = y[1]
    theta2 = y[2]
    z2 = y[3]
    
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    #theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)

    #theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return z1, z1dot, z2, z2dot





#  def calc_E(y):
#      """Return the total energy of the system."""

#      th1, th1d, th2, th2d = y.T
#      V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
#      T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
#              2*L1*L2*th1d*th2d*np.cos(th1-th2))
#      return T + V

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 30, 0.01
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0])

# Do the numerical integration of the equations of motion
y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

q1, dq1 = y[:,0], y[:,1]
q2, dq2 = y[:,2], y[:,3]

fig,ax=plt.subplots(2,1)
#ax[0].text(0.3,0.5,"1st Subplot")
ax[0].plot(t,q1)
ax[0].plot(t,dq1)
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Y")

ax[1].plot(t,q2)
ax[1].plot(t,dq2)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Y")

plt.tight_layout()
#plt.plot(t,q1,label='Disp q1 e dq1')
#plt.plot(t,q2,label='Disp q2 e dq2')
#plt.legend()

plt.show()




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
