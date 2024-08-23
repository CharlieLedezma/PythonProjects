import numpy as np
import control
import matplotlib.pyplot as plt

T_delay = 5
n_pade = 10
( num_pade , den_pade ) = control.pade ( T_delay , n_pade )
H_pade = control.tf (num_pade , den_pade)
#Generating transfer function without time delay :
num = np.array ([1])
den = np.array ([10 , 1])
H_without_delay = control.tf ( num , den )

#Generating transfer function with time delay :
H_with_delay = control.series( H_pade , H_without_delay )
#Simulation of step response :
t = np.linspace (0 , 40 , 100)

(t , y ) = control.step_response ( H_with_delay , t )
plt.plot (t , y )
plt.xlabel ('t[s]')
plt . grid ()
# Generating pdf file of the plotting figure :
plt.savefig ('pade_approx . pdf')