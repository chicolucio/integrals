from sympy import Symbol
from integral import Integral
import matplotlib.pyplot as plt


# ------------------------------ Example 1 -------------------------
x = Symbol('x')
expr = 1 / (1 + x**2)
example = Integral(expr, (-0.5, 5.5), (0, 5), x)
ani = example.animation(save=False)
plt.show()


# ------------------------------ Example 2 -------------------------
# from sympy import pi, sin
# x = Symbol('x')
# expr = sin(x)
# example = Integral(expr, (0, pi/2), (0, pi/2), x, 10)
# ani = example.animation(save=False, filename='animation2.gif')
# plt.show()

# ------------------------------ Example 3 -------------------------
# x = Symbol('x')
# expr = 4 / (1 + x**2)
# example = Integral(expr, (0, 1), (0, 1), x, 10)
# ani = example.animation(save=False, filename='animation3.gif')
# plt.show()

# ------------------------------ Example 4 -------------------------
# x = Symbol('x')
# expr = 1 / x
# example = Integral(expr, (1, 2), (1, 2), x, 10)
# ani = example.animation(save=False, filename='animation4.gif')
# plt.show()
