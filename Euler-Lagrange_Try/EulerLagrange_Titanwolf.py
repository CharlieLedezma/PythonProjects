#https://titanwolf.org/Network/Articles/Article?AID=1c71215f-5d49-47c2-a020-6f7e3b333eac#gsc.tab=0
#Sympy the import
import sympy as sp

# Set the time of the variable
t = sp.symbols('t')

# set the variable position of the mass point
# (x Adding a cls = sp.Function the option to set the function of T)
x = sp.symbols('x ', cls=sp.Function)
#setting  mass and spring constant
m, k = sp.symbols('m k')

# Lagrangian L the set
L =sp.Function("L")
L = (m*(x(t).diff(t))**2)/2 -(k*x(t)**2)/2

# An array to the time derivative of the coordinates and the coordinates of the variables create
# additional variables here when extended to multivariable
pos = [x(t)]
vel =[x(t).diff(t)]

# Define the functions for solving the Euler Lagrange equation
def EulerLagrange(L,pos,vel):
    EQ_list =[]
    for i in range(len(pos)):
        EQ_list.append(sp.simplify(sp.Eq(L.diff(vel[i]).diff(t) - L.diff(pos[i]),0)))
    return EQ_list

# Equation of motion
f =EulerLagrange(L,pos,vel)[0]
print(f)