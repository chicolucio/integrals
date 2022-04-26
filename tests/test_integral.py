'''
Test values from
https://personal.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/
'''

from sympy import Symbol, atan, sin, pi, log
import numpy as np
from integral import Integral


class TestArctan5:

    x = Symbol('x')
    expr = 1 / (1 + x**2)
    example = Integral(expr, (-0.5, 5.5), (0, 5), x)

    def test_symbolic_answer(self):
        assert self.example.exact_integral_value() == atan(5)

    def test_numeric_evaluation_answer(self):
        result = self.example.exact_integral_value(num_eval=True)
        assert np.isclose(float(result), 1.3734)

    def test_riemann_sums(self):
        assert np.allclose((self.example.riemann_sum),
                           (1.6134886, 1.1327194, 1.3735434))

    def test_riemann_errors(self):
        assert np.allclose((self.example.riemann_errors),
                           (0.2400870, -0.2406821, 0.0001417),
                           atol=1E-6, rtol=0)


class TestSin:

    x = Symbol('x')
    expr = sin(x)
    example = Integral(expr, (0, pi/2), (0, pi/2), x, 100)

    def test_symbolic_answer(self):
        assert self.example.exact_integral_value() == 1

    def test_numeric_evaluation_answer(self):
        result = self.example.exact_integral_value(num_eval=True)
        assert np.isclose(float(result), 1)

    def test_riemann_sums(self):
        assert np.allclose((self.example.riemann_sum),
                           (0.992125456, 1.00783341, 1.0000102))


class TestPi:
    x = Symbol('x')
    expr = 4 / (1 + x**2)
    example = Integral(expr, (0, 1), (0, 1), x, 130000)

    def test_symbolic_answer(self):
        assert self.example.exact_integral_value() == pi

    def test_numeric_evaluation_answer(self):
        result = self.example.exact_integral_value(num_eval=True)
        assert np.isclose(float(result), float(pi.n()))

    def test_riemann_right(self):
        assert np.isclose(self.example.riemann_sum.right,
                          3.1415849612722386,
                          rtol=0, atol=1E-5)


class TestLn2:
    x = Symbol('x')
    expr = 1 / x
    example = Integral(expr, (1, 2), (1, 2), x, 2887)

    def test_symbolic_answer(self):
        assert self.example.exact_integral_value() == log(2)

    def test_numeric_evaluation_answer(self):
        result = self.example.exact_integral_value(num_eval=True)
        assert np.isclose(float(result), float(log(2).n()))

    def test_riemann_mid(self):
        assert np.isclose(self.example.riemann_sum.mid,
                          0.6931471768105913,
                          rtol=0, atol=1E-8)
