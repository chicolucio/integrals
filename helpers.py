import matplotlib.pyplot as plt


def move_sympy_plot_to_axes(sympy_plot, plt_ax):
    """Moves a SymPy plot to a Matplotlib axes

    Adapted from: https://stackoverflow.com/a/46813804/8706250

    Parameters
    ----------
    sympy_plot : SymPy plot
    plt_ax : Matplotlib axes
    """

    backend = sympy_plot.backend(sympy_plot)
    backend.ax = plt_ax
    backend._process_series(backend.parent._series, plt_ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['top'].set_color('none')
    plt.close(backend.fig)
