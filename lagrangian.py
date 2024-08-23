import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
import sympy as sym
import sympy.physics.mechanics as mech
from sympy.calculus.euler import euler_equations
mech.init_vprinting()

#Constants


R,m1,m2,g,L = sym.symbols('R, m_1, m_2, g, L', positive=True)
t = sym.symbols('t')
x = sym.Function('x')(t)
y = -x + (L-sym.pi*R)

#Defining energies
T = sym.Rational(1,2)*m1*x.diff(t)**2 + sym.Rational(1,2)*m2*y.diff(t)**2
T.simplify()

U = m1*g*x + m2*g*y
print(U.simplify())

#Lagrangian
Lagr = T-U
print(Lagr.simplify())

#Lagrange Equations
rhs = Lagr.diff(x)
rhs

lhs = Lagr.diff(x.diff(t)).diff(t)
lhs

solution = sym.solve(sym.Eq(rhs,lhs),x.diff(t,2))
print(solution)