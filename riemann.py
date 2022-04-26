import matplotlib.pyplot as plt
from sympy import Symbol, plot, lambdify, integrate, N
from helpers import move_sympy_plot_to_axes
import numpy as np


class Integral:

    def __init__(self, sympy_expression, domain, interval, variable, bars=10):
        self._sympy_expression = sympy_expression
        self._domain = domain
        self._interval = interval
        self._x = variable
        self.bars = bars

    @property
    def sympy_expression(self):
        return self._sympy_expression

    @property
    def domain(self):
        return self._domain

    @property
    def interval(self):
        return self._interval

    @property
    def x(self):
        return self._x

    @property
    def bars(self):
        return self._bars

    @bars.setter
    def bars(self, value):
        if not isinstance(value, int):
            raise ValueError('Must be a positive integer')
        elif value < 0:
            raise ValueError('Must be a positive integer')
        else:
            self._bars = value

    def definite_integral_fill_plot(self, nb_points=100, figsize=(8, 6)):
        # TODO update the docs

        # TODO change behavior so that the function can receive an axes instead
        # of a figsize

        fig, ax = plt.subplots(figsize=figsize)

        p = plot(self.sympy_expression, (self.x, *self.domain), show=False)
        move_sympy_plot_to_axes(p, ax)

        x_fill = np.linspace(*self.interval, nb_points)
        f = lambdify(self.x, self.sympy_expression, 'numpy')
        ax.fill_between(x_fill, f(x_fill))

        # ax.legend(loc='best')
        plt.show()

    def riemann_plot(self):
        # TODO update the docs
        fig, arr = plt.subplots(nrows=1, ncols=3, figsize=(15, 5),
                                constrained_layout=True, facecolor=(1, 1, 1))

        p = plot(self.sympy_expression, (self.x, *self.domain), show=False)

        for ax in arr:
            move_sympy_plot_to_axes(p, ax)

        x_values = np.linspace(*self.interval, self.bars+1)
        f = lambdify(self.x, self.sympy_expression, 'numpy')

        x_left = x_values[:-1]
        y_left = f(x_values)[:-1]
        x_right = x_values[1:]
        y_right = f(x_values)[1:]
        x_mid = (x_left + x_right)/2
        y_mid = f(x_mid)

        arr[0].scatter(x_left, y_left, s=10)
        arr[0].bar(x_left, y_left,
                   width=(self.interval[1]-self.interval[0])/self.bars,
                   align='edge', alpha=0.2, edgecolor='orange')
        arr[1].scatter(x_mid, y_mid, s=10)
        arr[1].bar(x_mid, y_mid,
                   width=(self.interval[1]-self.interval[0])/self.bars,
                   align='center', alpha=0.2, edgecolor='orange')
        arr[2].scatter(x_right, y_right, s=10)
        arr[2].bar(x_right, y_right,
                   width=-(self.interval[1]-self.interval[0])/self.bars,
                   align='edge', alpha=0.2, edgecolor='orange')

        plt.show()

    def riemann_sum(self, method='midpoint'):
        # TODO update the docs

        x_values = np.linspace(*self.interval, self.bars+1)
        f = lambdify(self.x, self.sympy_expression, 'numpy')

        x_left = x_values[:-1]
        x_right = x_values[1:]
        x_mid = (x_left + x_right)/2

        width = (self.interval[1] - self.interval[0])/self.bars

        if method == 'left':
            return np.sum(f(x_left)*width)
        elif method == 'right':
            return np.sum(f(x_right)*width)
        elif method == 'midpoint':
            return np.sum(f(x_mid)*width)
        else:
            raise ValueError("Method must be 'left', 'right' or 'midpoint'.")

    def exact_integral_value(self, num_eval=False, digits=5):
        symbolic_answer = integrate(
            self.sympy_expression, (self.x, *self.interval))
        if num_eval:
            return N(symbolic_answer, digits)
        return symbolic_answer


if __name__ == "__main__":

    x = Symbol('x')
    expr = 1 / (1 + x**2)

    example = Integral(expr, (-0.5, 5.5), (0, 5), x)
    example.definite_integral_fill_plot()
    example.riemann_plot()

    for method in ('left', 'midpoint', 'right'):
        print(f'{method}: {example.riemann_sum(method=method)}')

    print(example.exact_integral_value())
    print(example.exact_integral_value(num_eval=True))
