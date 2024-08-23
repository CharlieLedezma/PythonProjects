#simulation of motionof pendulum
#https://skill-lync.com/student-projects/Solving-second-order-ODE-in-python-36864

import math
import matplotlib.pyplot as plt
import numpy as np
#import scipy.integrate import odeint
from scipy.integrate import odeint


#odeint allow solve
#Function that return dz/dt
def model(theta,t,b,g,l,m):
    theta1 = theta [0]
    theta2 = theta [1]
    dtheta1_dt = theta2
    dtheta2_dt = -(b/m)*theta2 - (g/l)*math.sin(theta1)
    dtheta_dt = [dtheta1_dt,dtheta2_dt]
    return dtheta_dt

#input
b=0.05  #damping coeficient.
g=9.81  #gravity in m/s2  
l=1     #length of the pendulum in m
m=0.1   #mass of the ball in kg

#Initial condition
theta_0 = [0,3]

#Time points
t = np.linspace(0,20,150)

#Solve ODE
theta = odeint(model,theta_0,t,args=(b,g,l,m))
#theta = odeint(model,theta_0,t,args=(b,g))

#Plot
plt.plot(t,theta[:,0],'b--',label= r'$\frac{d\theta_1}{dt} = \theta_2 $')
plt.plot(t,theta[:,1],'r--',label= r'$\frac{d\theta_2}{dt} = -\frac{b}{m}\theta_2 -\frac{g}{l}sin\theta_1 $')
plt.xlabel('Time')
plt.ylabel('Plot')
plt.xlim([0,12])
plt.legend(loc = 'best')
plt.show()

#Animation
#THETA3=[]
#for theta3 in theta[:,0]:
#    THETA3.append(theta3)

#ct=1
# for THETA4 in THETA3:
#     x0=0
#     y0=0

#     x1=l*math.sin(THETA4)
#     y1=-l*math.cos(THETA4)
#     filename='Pendulum%05d.png'%ct
#     ct=ct+1

#     plt.plot([-0.2,0.2],[0,0])      #Horizontal support or pendulum
#     plt.plot([x0,x1],[y0,y1])       #String of pendulum
#     plt.plot(x1,y1,'-o')            #Bob tied at end
#     plt.xlim([-1.5,1.5])
#     plt.ylim([-1.5,1])
#     plt.title('Motion of Pendulum')
#     plt.savefig(filename)
#     plt.show()










