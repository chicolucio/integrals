import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy import Symbol, plot, lambdify, integrate, N, latex
from helpers import move_sympy_plot_to_axes
import numpy as np
from collections import namedtuple


class Integral:
    """A class used to represent an Integral. It has methods to calculate
    the exact integral value and approximate values through Riemann sums.
    It has methods to plot graphs: an exact area; and an approximate area
    with Riemann sums.
    """

    def __init__(self, sympy_expression, domain, interval, variable, bars=10):
        """
        Parameters
        ----------
        sympy_expression : SymPy expression
            SymPy expression representing the function
        domain : tuple
            Function domain
        interval : tuple
            Integral lower and upper limits. It can be the same as the domain
        variable : SymPy symbol
            Variable SymPy symbol used in the sympy_expression
        bars : int, optional
            The number of Riemann rectangles to be considered, by default 10
        """

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
        """Returns LaTeX strings to be used on graphs

        Returns
        -------
        tuple
            Tuple of strings:
            (expression, integral lower limit, integral upper limit)
        """

        Latex = namedtuple('Latex', ('expr', 'lower_lim', 'upper_lim'))
        return Latex(f'${latex(self.sympy_expression)}$',
                     f'${latex(self.interval[0])}$',
                     f'${latex(self.interval[1])}$')

    def definite_integral_fill_plot(self, nb_points=100, figsize=(8, 6)):
        """Plots the function graph with a shaded area representing the
        exact integral

        Parameters
        ----------
        nb_points : int, optional
            Number of points to be used to create the area, by default 100
        figsize : tuple, optional
            Figure size, by default (8, 6)

        Returns
        -------
        Matplotlib axis
            Axis with the Matplotlib objects.
        """

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
        plt.title(title, fontsize=16)
        return ax

    @property
    def riemann_x_y(self):
        """Generates the coordinates to be used in Riemann calculations
        and plots.

        Returns
        -------
        tuple
            Coordinates.
        """

        try:
            x_values = np.linspace(*self.interval, self.bars+1)
        except TypeError:
            x_values = np.linspace(float(self.interval[0]), float(
                self.interval[1]), self.bars+1)
        f = lambdify(self.x, self.sympy_expression, 'numpy')
        return x_values, f

    @property
    def riemann_calculations(self):
        """Riemann calculations for each method: left, midpoint, right. The
        methods consider endpoints (left and right) or the midpoint of each
        subinterval.

        Returns
        -------
        tuple
            NamedTuple with x and y values for each method.
        """

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

    def _riemann_plot_setup(self):
        fig, arr = plt.subplots(nrows=1, ncols=3, figsize=(15, 5),
                                facecolor=(1, 1, 1), tight_layout=True)
        return fig, arr

    def riemann_plot(self):
        """Plots each Riemann method.

        Returns
        -------
        Tuple
            Matplotlib axes, one for each method.
        """

        try:
            self.fig
        except AttributeError:
            self.fig, self.arr = self._riemann_plot_setup()
        Latex = self._latex_strings

        p = plot(self.sympy_expression, (self.x, *self.domain), show=False)

        for ax in self.arr:
            move_sympy_plot_to_axes(p, ax)

        Riemann = self.riemann_calculations
        exact = self.exact_integral_value(num_eval=True)

        self.arr[0].scatter(Riemann.x_left, Riemann.y_left, s=10)
        self.arr[0].bar(Riemann.x_left, Riemann.y_left,
                        width=Riemann.bar_width,
                        align='edge', alpha=0.2, edgecolor='orange')
        self.arr[0].set_title(f'Left Riemann Sum - N = {self.bars}')
        self.arr[0].annotate(f'Exact value: {exact}',
                             xy=(0.5, 0.), xytext=(0, 30),
                             xycoords=('axes fraction', 'figure fraction'),
                             textcoords='offset points',
                             ha='center', va='bottom', size=12)
        self.arr[0].annotate(f'Approximate value: {self.riemann_sum.left:.5f}',
                             xy=(0.5, 0.), xytext=(0, 15),
                             xycoords=('axes fraction', 'subfigure fraction'),
                             textcoords='offset points',
                             ha='center', va='bottom', size=12)
        self.arr[0].annotate(f'Error: {self.riemann_errors.left:.5E}',
                             xy=(0.5, 0.), xytext=(0, 0),
                             xycoords=('axes fraction', 'figure fraction'),
                             textcoords='offset points',
                             ha='center', va='bottom', size=12)

        self.arr[1].scatter(Riemann.x_mid, Riemann.y_mid, s=10)
        self.arr[1].bar(Riemann.x_mid, Riemann.y_mid,
                        width=Riemann.bar_width,
                        align='center', alpha=0.2, edgecolor='orange')
        self.arr[1].set_title(f'Midpoint Riemann Sum - N = {self.bars}')
        self.arr[1].annotate(f'Exact value: {exact}',
                             xy=(0.5, 0.), xytext=(0, 30),
                             xycoords=('axes fraction', 'figure fraction'),
                             textcoords='offset points',
                             ha='center', va='bottom', size=12)
        self.arr[1].annotate(f'Approximate value: {self.riemann_sum.mid:.5f}',
                             xy=(0.5, 0.), xytext=(0, 15),
                             xycoords=('axes fraction', 'subfigure fraction'),
                             textcoords='offset points',
                             ha='center', va='bottom', size=12)
        self.arr[1].annotate(f'Error: {self.riemann_errors.mid:.5E}',
                             xy=(0.5, 0.), xytext=(0, 0),
                             xycoords=('axes fraction', 'figure fraction'),
                             textcoords='offset points',
                             ha='center', va='bottom', size=12)

        self.arr[2].scatter(Riemann.x_right, Riemann.y_right, s=10)
        self.arr[2].bar(Riemann.x_right, Riemann.y_right,
                        width=-Riemann.bar_width,
                        align='edge', alpha=0.2, edgecolor='orange')
        self.arr[2].set_title(f'Right Riemann Sum - N = {self.bars}')
        self.arr[2].annotate(f'Exact value: {exact}',
                             xy=(0.5, 0.), xytext=(0, 30),
                             xycoords=('axes fraction', 'figure fraction'),
                             textcoords='offset points',
                             ha='center', va='bottom', size=12)
        self.arr[2].annotate(f'Approximate value: {self.riemann_sum.right:.5f}',  # NoQA
                             xy=(0.5, 0.), xytext=(0, 15),
                             xycoords=('axes fraction', 'subfigure fraction'),
                             textcoords='offset points',
                             ha='center', va='bottom', size=12)
        self.arr[2].annotate(f'Error: {self.riemann_errors.right:.5E}',
                             xy=(0.5, 0.), xytext=(0, 0),
                             xycoords=('axes fraction', 'figure fraction'),
                             textcoords='offset points',
                             ha='center', va='bottom', size=12)

        main_title = f'Plot for {Latex.expr}\n'
        area_title = f'Area from {Latex.lower_lim} to {Latex.upper_lim}'
        title = main_title + area_title
        self.fig.suptitle(title, fontsize=18)
        self.fig.tight_layout(rect=[0, 0.07, 1, 1.0])
        return self.arr

    def _animate(self, bars):
        for ax in self.arr:
            ax.autoscale_view()
            ax.clear()
        self.fig.canvas.draw()
        self.bars = bars
        return self.riemann_plot()

    def animation(self, frames=(1, 5, 10, 20, 50, 100, 200), interval=150,
                  save=False, filename='animation.gif', fps=1):
        """Animates the Riemann plots

        Parameters
        ----------
        frames : tuple, optional
            values for the number of rectangles, by default
            (1, 5, 10, 20, 50, 100, 200)
        interval : int, optional
            Time in ms between each frame, by default 150
        save : bool, optional
            If the animation is going to be saved, by default False
        filename : str, optional
            GIF filename, by default 'animation.gif'
        fps : int, optional
            GIF frames per second, by default 1

        Returns
        -------
        Matplotlib animation
            Matplotlib animation
        """

        self.fig, self.arr = self._riemann_plot_setup()
        ani = animation.FuncAnimation(self.fig, self._animate,
                                      frames=frames,
                                      interval=interval,
                                      repeat_delay=150,
                                      )
        if save:
            ani.save(filename, 'imagemagick', fps=fps)
        return ani

    @property
    def riemann_sum(self):
        """Calculates the Riemann sum for each method

        Returns
        -------
        tuple
            NamedTuple with the values for each method
        """

        _, f = self.riemann_x_y
        Riemann = self.riemann_calculations

        RiemannSum = namedtuple('RiemannSum', ('left', 'right', 'mid'))

        left = np.sum(f(Riemann.x_left)*Riemann.bar_width)
        right = np.sum(f(Riemann.x_right)*Riemann.bar_width)
        mid = np.sum(f(Riemann.x_mid)*Riemann.bar_width)

        return RiemannSum(left, right, mid)

    def exact_integral_value(self, num_eval=False, digits=5):
        """Calculates the exact integral value. Symbolic or numeric.

        Parameters
        ----------
        num_eval : bool, optional
            If it will return a numeric evaluation, by default False
        digits : int, optional
            The number of digits of the numeric evaluation, by default 5

        Returns
        -------
        float or SymPy expression
            A standard Python float if num_eval. Else, a SymPy expression.
        """

        symbolic_answer = integrate(
            self.sympy_expression, (self.x, *self.interval))
        if num_eval:
            return N(symbolic_answer, digits)
        return symbolic_answer

    @property
    def riemann_errors(self):
        """Calculates the Riemann sum errors for each method.

        Returns
        -------
        tuple
            NamedTuple with the values for each method.
        """

        RiemannErrors = namedtuple('RiemannErrors', ('left', 'right', 'mid'))
        exact = float(self.exact_integral_value(num_eval=True))
        calc = (value - exact for value in self.riemann_sum)
        return RiemannErrors(*calc)


if __name__ == "__main__":

    # ------------------------------ Example 1 -------------------------
    x = Symbol('x')
    expr = 1 / (1 + x**2)
    example = Integral(expr, (-0.5, 5.5), (0, 5), x)
    example.definite_integral_fill_plot()
    plt.show()
    example.riemann_plot()
    plt.show()

    for i, method in enumerate(example.riemann_sum._fields):
        print(f'{method}: {example.riemann_sum[i]}')

    print(example.exact_integral_value())
    print(example.exact_integral_value(num_eval=True))

    # ------------------------------ Example 2 -------------------------
    # from sympy import pi, sin
    # x = Symbol('x')
    # expr = sin(x)
    # example = Integral(expr, (0, pi/2), (0, pi/2), x, 10)
    # example.definite_integral_fill_plot()
    # example.riemann_plot()
    # plt.show()

    # ------------------------------ Example 3 -------------------------
    # x = Symbol('x')
    # expr = 4 / (1 + x**2)
    # example = Integral(expr, (0, 1), (0, 1), x, 10)
    # example.definite_integral_fill_plot()
    # example.riemann_plot()
    # plt.show()

    # ------------------------------ Example 4 -------------------------
    # x = Symbol('x')
    # expr = 1 / x
    # example = Integral(expr, (1, 2), (1, 2), x, 10)
    # example.definite_integral_fill_plot()
    # example.riemann_plot()
    # plt.show()
