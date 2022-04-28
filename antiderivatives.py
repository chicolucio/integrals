import matplotlib.pyplot as plt
from sympy import Symbol, plot
from helpers import move_sympy_plot_to_axes


x = Symbol('x')

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(-2, 3):
    expr = x**2 + i
    text_label = rf"$x^2 {i:+g}$"
    if i == 0:
        text_label = r"$x^2$"
    p = plot(expr, (x, -3, 3), ylim=(-2.5, 4), label=text_label, show=False)
    move_sympy_plot_to_axes(p, ax)

ax.legend(loc='best', fontsize=13)
ax.set_title('Antiderivatives of $2x$\n', fontsize=16)
plt.show()
fig.savefig('images/antiderivatives.png', 
            dpi=fig.dpi, bbox_inches='tight', pad_inches=0.1)
