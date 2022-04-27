import matplotlib.pyplot as plt
from sympy import Symbol, plot, lambdify, integrate, N, latex
from helpers import move_sympy_plot_to_axes
import numpy as np
from collections import namedtuple


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

    @property
    def _latex_strings(self):
        Latex = namedtuple('Latex', ('expr', 'lower_lim', 'upper_lim'))
        return Latex(f'${latex(self.sympy_expression)}$',
                     f'${latex(self.interval[0])}$',
                     f'${latex(self.interval[1])}$')

    def definite_integral_fill_plot(self, nb_points=100, figsize=(8, 6)):
        # TODO update the docs

        # TODO change behavior so that the function can receive an axes instead
        # of a figsize

        fig, ax = plt.subplots(figsize=figsize)
        Latex = self._latex_strings

        p = plot(self.sympy_expression, (self.x, *self.domain), show=False,
                 label=Latex.expr)
        move_sympy_plot_to_axes(p, ax)

        try:
            x_fill = np.linspace(*self.interval, nb_points)
        except TypeError:
            x_fill = np.linspace(float(self.interval[0]), float(
                self.interval[1]), nb_points)
        f = lambdify(self.x, self.sympy_expression, 'numpy')
        ax.fill_between(x_fill, f(x_fill))

        main_title = f'Plot for {Latex.expr}\n'
        area_title = f'Area from {Latex.lower_lim} to {Latex.upper_lim}'
        title = main_title + area_title

        # ax.legend(loc='best', fontsize=14)
        plt.title(title, fontsize=16)
        plt.show()

    @property
    def riemann_x_y(self):
        try:
            x_values = np.linspace(*self.interval, self.bars+1)
        except TypeError:
            x_values = np.linspace(float(self.interval[0]), float(
                self.interval[1]), self.bars+1)
        f = lambdify(self.x, self.sympy_expression, 'numpy')
        return x_values, f

    @property
    def riemann_calculations(self):
        x_values, f = self.riemann_x_y

        x_left = x_values[:-1]
        y_left = f(x_values)[:-1]
        x_right = x_values[1:]
        y_right = f(x_values)[1:]
        x_mid = (x_left + x_right)/2
        y_mid = f(x_mid)
        bar_width = (x_values[-1] - x_values[0])/self.bars

        Riemann = namedtuple('Riemann',
                             ['x_left', 'y_left', 'x_right', 'y_right',
                              'x_mid', 'y_mid', 'bar_width'])
        return Riemann(x_left, y_left, x_right, y_right,
                       x_mid, y_mid, bar_width)

    def riemann_plot(self):
        # TODO update the docs
        Latex = self._latex_strings
        fig, arr = plt.subplots(nrows=1, ncols=3, figsize=(15, 5),
                                facecolor=(1, 1, 1))

        p = plot(self.sympy_expression, (self.x, *self.domain), show=False)

        for ax in arr:
            move_sympy_plot_to_axes(p, ax)

        Riemann = self.riemann_calculations

        arr[0].scatter(Riemann.x_left, Riemann.y_left, s=10)
        arr[0].bar(Riemann.x_left, Riemann.y_left,
                   width=Riemann.bar_width,
                   align='edge', alpha=0.2, edgecolor='orange')
        arr[0].set_title(f'Left Riemann Sum - N = {self.bars}')
        arr[0].annotate(f'Exact value: {self.exact_integral_value()}',
                        xy=(0.5, 0.), xytext=(0, 30),
                        xycoords=('axes fraction', 'figure fraction'),
                        textcoords='offset points',
                        ha='center', va='bottom', size=12)
        arr[0].annotate(f'Approximate value: {self.riemann_sum.left}',
                        xy=(0.5, 0.), xytext=(0, 15),
                        xycoords=('axes fraction', 'subfigure fraction'),
                        textcoords='offset points',
                        ha='center', va='bottom', size=12)
        arr[0].annotate(f'Error: {self.riemann_errors.left}',
                        xy=(0.5, 0.), xytext=(0, 0),
                        xycoords=('axes fraction', 'figure fraction'),
                        textcoords='offset points',
                        ha='center', va='bottom', size=12)

        arr[1].scatter(Riemann.x_mid, Riemann.y_mid, s=10)
        arr[1].bar(Riemann.x_mid, Riemann.y_mid,
                   width=Riemann.bar_width,
                   align='center', alpha=0.2, edgecolor='orange')
        arr[1].set_title(f'Midpoint Riemann Sum - N = {self.bars}')
        arr[1].annotate(f'Exact value: {self.exact_integral_value()}',
                        xy=(0.5, 0.), xytext=(0, 30),
                        xycoords=('axes fraction', 'figure fraction'),
                        textcoords='offset points',
                        ha='center', va='bottom', size=12)
        arr[1].annotate(f'Approximate value: {self.riemann_sum.mid}',
                        xy=(0.5, 0.), xytext=(0, 15),
                        xycoords=('axes fraction', 'subfigure fraction'),
                        textcoords='offset points',
                        ha='center', va='bottom', size=12)
        arr[1].annotate(f'Error: {self.riemann_errors.mid}',
                        xy=(0.5, 0.), xytext=(0, 0),
                        xycoords=('axes fraction', 'figure fraction'),
                        textcoords='offset points',
                        ha='center', va='bottom', size=12)

        arr[2].scatter(Riemann.x_right, Riemann.y_right, s=10)
        arr[2].bar(Riemann.x_right, Riemann.y_right,
                   width=-Riemann.bar_width,
                   align='edge', alpha=0.2, edgecolor='orange')
        arr[2].set_title(f'Right Riemann Sum - N = {self.bars}')
        arr[2].annotate(f'Exact value: {self.exact_integral_value()}',
                        xy=(0.5, 0.), xytext=(0, 30),
                        xycoords=('axes fraction', 'figure fraction'),
                        textcoords='offset points',
                        ha='center', va='bottom', size=12)
        arr[2].annotate(f'Approximate value: {self.riemann_sum.right}',
                        xy=(0.5, 0.), xytext=(0, 15),
                        xycoords=('axes fraction', 'subfigure fraction'),
                        textcoords='offset points',
                        ha='center', va='bottom', size=12)
        arr[2].annotate(f'Error: {self.riemann_errors.right}',
                        xy=(0.5, 0.), xytext=(0, 0),
                        xycoords=('axes fraction', 'figure fraction'),
                        textcoords='offset points',
                        ha='center', va='bottom', size=12)

        main_title = f'Plot for {Latex.expr}\n'
        area_title = f'Area from {Latex.lower_lim} to {Latex.upper_lim}'
        title = main_title + area_title
        fig.suptitle(title, fontsize=18)
        fig.tight_layout(rect=[0, 0.07, 1, 1.0])
        plt.show()

    @property
    def riemann_sum(self):
        # TODO update the docs
        _, f = self.riemann_x_y
        Riemann = self.riemann_calculations

        RiemannSum = namedtuple('RiemannSum', ('left', 'right', 'mid'))

        left = np.sum(f(Riemann.x_left)*Riemann.bar_width)
        right = np.sum(f(Riemann.x_right)*Riemann.bar_width)
        mid = np.sum(f(Riemann.x_mid)*Riemann.bar_width)

        return RiemannSum(left, right, mid)

    def exact_integral_value(self, num_eval=False, digits=5):
        symbolic_answer = integrate(
            self.sympy_expression, (self.x, *self.interval))
        if num_eval:
            return N(symbolic_answer, digits)
        return symbolic_answer

    @property
    def riemann_errors(self):
        RiemannErrors = namedtuple('RiemannErrors', ('left', 'right', 'mid'))
        exact = float(self.exact_integral_value(num_eval=True))
        calc = (value - exact for value in self.riemann_sum)
        return RiemannErrors(*calc)


if __name__ == "__main__":

    x = Symbol('x')
    expr = 1 / (1 + x**2)

    example = Integral(expr, (-0.5, 5.5), (0, 5), x)
    example.definite_integral_fill_plot()
    example.riemann_plot()

    for i, method in enumerate(example.riemann_sum._fields):
        print(f'{method}: {example.riemann_sum[i]}')

    print(example.exact_integral_value())
    print(example.exact_integral_value(num_eval=True))
