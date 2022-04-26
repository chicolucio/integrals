from sympy import Symbol, atan
import numpy as np
from integral import Integral
import pytest


class TestArctan5:

    x = Symbol('x')
    expr = 1 / (1 + x**2)
    example = Integral(expr, (-0.5, 5.5), (0, 5), x)

    def test_symbolic_answer(self):
        assert self.example.exact_integral_value() == atan(5)

    def test_numeric_evaluation_answer(self):
        result = self.example.exact_integral_value(num_eval=True)
        assert np.isclose(float(result), 1.3734)

    @pytest.mark.parametrize('method, result', (('left', 1.6134886),
                                                ('midpoint', 1.3735434),
                                                ('right', 1.1327194)))
    def test_riemann_sums(self, method, result):
        assert np.isclose(self.example.riemann_sum(method), result)