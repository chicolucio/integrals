import matplotlib.pyplot as plt
from sympy import Symbol, plot, lambdify, integrate, N
from helpers import move_sympy_plot_to_axes
import numpy as np


x = Symbol('x')
expr = 1 / (1 + x**2)


def definite_integral_fill_plot(sympy_expression, x_range, interval,
                                nb_points=100, figsize=(8, 6)):
    # TODO update the docs

    # TODO change behavior so that the function can receive an axes instead of
    # a figsize

    fig, ax = plt.subplots(figsize=figsize)

    p = plot(sympy_expression, (x, *x_range), show=False)
    move_sympy_plot_to_axes(p, ax)

    x_fill = np.linspace(*interval, nb_points)
    f = lambdify(x, sympy_expression, 'numpy')
    ax.fill_between(x_fill, f(x_fill))

    # ax.legend(loc='best')
    plt.show()


# definite_integral_fill_plot(expr, (-1, 6), (0, 5))


def riemann_plot(sympy_expression, x_range, interval, bars=10):
    # TODO update the docs
    fig, arr = plt.subplots(nrows=1, ncols=3, figsize=(15, 5),
                            constrained_layout=True, facecolor=(1, 1, 1))

    p = plot(sympy_expression, (x, *x_range), show=False)

    for ax in arr:
        move_sympy_plot_to_axes(p, ax)

    x_values = np.linspace(*interval, bars+1)
    f = lambdify(x, sympy_expression, 'numpy')

    x_left = x_values[:-1]
    y_left = f(x_values)[:-1]
    x_right = x_values[1:]
    y_right = f(x_values)[1:]
    x_mid = (x_left + x_right)/2
    y_mid = f(x_mid)

    arr[0].scatter(x_left, y_left, s=10)
    arr[0].bar(x_left, y_left, width=(interval[1]-interval[0])/bars,
               align='edge', alpha=0.2, edgecolor='orange')
    arr[1].scatter(x_mid, y_mid, s=10)
    arr[1].bar(x_mid, y_mid, width=(interval[1]-interval[0])/bars,
               align='center', alpha=0.2, edgecolor='orange')
    arr[2].scatter(x_right, y_right, s=10)
    arr[2].bar(x_right, y_right, width=-(interval[1]-interval[0])/bars,
               align='edge', alpha=0.2, edgecolor='orange')

    plt.show()


# riemann_plot(expr, (-0.5, 5.5), (0, 5), bars=10)


def riemann_sum(sympy_expression, interval, bars, method='midpoint'):
    # TODO update the docs

    x_values = np.linspace(*interval, bars+1)
    f = lambdify(x, sympy_expression, 'numpy')

    x_left = x_values[:-1]
    x_right = x_values[1:]
    x_mid = (x_left + x_right)/2

    width = (interval[1] - interval[0])/bars

    if method == 'left':
        return np.sum(f(x_left)*width)
    elif method == 'right':
        return np.sum(f(x_right)*width)
    elif method == 'midpoint':
        return np.sum(f(x_mid)*width)
    else:
        raise ValueError("Method must be 'left', 'right' or 'midpoint'.")


for method in ('left', 'midpoint', 'right'):
    print(f'{method}: {riemann_sum(expr, (0,5), 10, method=method)}')


def exact_integral_value(sympy_expression, interval, num_eval=False, digits=5):
    symbolic_answer = integrate(sympy_expression, (x, *interval))
    if num_eval:
        return N(symbolic_answer, digits)
    return symbolic_answer


print(exact_integral_value(expr, (0, 5)))
print(exact_integral_value(expr, (0, 5), num_eval=True))
