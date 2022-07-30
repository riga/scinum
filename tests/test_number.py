# coding: utf-8


__all__ = ["TestCase"]


import sys
import math
import decimal
import operator
import unittest

from scinum import (
    Number, Correlation, DeferredResult, ops, HAS_NUMPY, HAS_UNCERTAINTIES, split_value,
    match_precision, calculate_uncertainty, round_uncertainty, round_value, infer_si_prefix,
)

if HAS_NUMPY:
    import numpy as np

if HAS_UNCERTAINTIES:
    from uncertainties import ufloat

UP = Number.UP
DOWN = Number.DOWN


def if_numpy(func):
    return func if HAS_NUMPY else (lambda self: None)


def if_uncertainties(func):
    return func if HAS_UNCERTAINTIES else (lambda self: None)


def ptgr(*args):
    return sum([a ** 2. for a in args]) ** 0.5


class TestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCase, self).__init__(*args, **kwargs)

        self.num = Number(2.5, {
            "A": 0.5,
            "B": (1.0,),
            "C": (1.0, 1.5),
            "D": (Number.REL, 0.1),
            "E": (Number.REL, 0.1, 0.2),
            "F": (1.0, Number.REL, 0.2),
            "G": (Number.REL, 0.3, Number.ABS, 0.3),
            "H": (0.3j, 0.3),
        })

    def test_constructor(self):
        num = Number(42, 5)

        self.assertIsInstance(num.nominal, float)
        self.assertEqual(num.nominal, 42.)

        unc = num.get_uncertainty(Number.DEFAULT)
        self.assertIsInstance(unc, tuple)
        self.assertEqual(len(unc), 2)
        self.assertEqual(unc, (5., 5.))

        num = Number(42, {"foo": 5})

        self.assertEqual(num.get_uncertainty("foo"), (5, 5))
        self.assertEqual(num.get_uncertainty("NOT_EXISTING", default=(1, 2)), (1, 2))

        with self.assertRaises(KeyError):
            num.get_uncertainty("NOT_EXISTING")

        with self.assertRaises(ValueError):
            num.get_uncertainty("foo", direction="UNKNOWN")

    @if_numpy
    def test_constructor_numpy(self):
        num = Number(np.array([5, 27, 42]), 5)

        self.assertTrue(num.is_numpy)
        self.assertEqual(num.shape, (3,))

        unc = num.u()
        self.assertTrue(all(u.shape == num.shape for u in unc))

        with self.assertRaises(TypeError):
            num.nominal = 5

        num = Number(5, 2)
        self.assertFalse(num.is_numpy)
        self.assertIsNone(num.shape)

        num.nominal = np.arange(5)
        self.assertTrue(num.is_numpy)
        self.assertEqual(num.shape, (5,))
        self.assertEqual(num.u(direction=UP).shape, (5,))

        num.set_uncertainty("A", np.arange(5, 10))
        with self.assertRaises(ValueError):
            num.set_uncertainty("B", np.arange(5, 9))

    @if_uncertainties
    def test_constructor_ufloat(self):
        num = Number(ufloat(42, 5))
        self.assertEqual(num.nominal, 42.)
        self.assertEqual(num.get_uncertainty(Number.DEFAULT), (5., 5.))

        with self.assertRaises(ValueError):
            Number(ufloat(42, 5), uncertainties={"other_error": 123})

        num = Number(ufloat(42, 5, tag="foo"))
        self.assertEqual(num.get_uncertainty("foo"), (5., 5.))

        num = Number(ufloat(42, 5) + ufloat(2, 2))
        self.assertEqual(num.nominal, 44.)
        self.assertEqual(num.get_uncertainty(Number.DEFAULT), (7., 7.))

        num = Number(ufloat(42, 5, tag="foo") + ufloat(2, 2, tag="bar"))
        self.assertEqual(num.nominal, 44.)
        self.assertEqual(num.get_uncertainty("foo"), (5., 5.))
        self.assertEqual(num.get_uncertainty("bar"), (2., 2.))

        num = Number(ufloat(42, 5, tag="foo") + ufloat(2, 2, tag="bar") + ufloat(1, 1, tag="bar"))
        self.assertEqual(num.nominal, 45.)
        self.assertEqual(num.get_uncertainty("foo"), (5., 5.))
        self.assertEqual(num.get_uncertainty("bar"), (3., 3.))

    def test_copy(self):
        num = self.num.copy()
        self.assertFalse(num is self.num)

        num = num.copy(nominal=123, uncertainties=1)
        self.assertEqual(num.nominal, 123)
        self.assertEqual(len(num.uncertainties), 1)

    @if_numpy
    def test_copy_numpy(self):
        num = self.num.copy()
        num.nominal = np.array([3, 4, 5])
        self.assertFalse(num is self.num)

        num = num.copy(uncertainties=1)
        self.assertEqual(tuple(num.nominal), (3, 4, 5))
        self.assertEqual(len(num.uncertainties), 1)
        self.assertEqual(num.u(direction=UP).shape, (3,))

    def test_string_formats(self):
        self.assertEqual(len(self.num.str()), 105)
        self.assertEqual(len(self.num.str("%.3f")), 129)
        self.assertEqual(len(self.num.str(lambda n: "%s" % n)), 105)
        self.assertEqual(len(self.num.str(lambda n: "X%s" % n)), 119)
        self.assertEqual(len(self.num.repr().split(" ", 3)[-1]), 108)
        self.assertEqual(len(self.num.str(2)), 115)
        self.assertEqual(len(self.num.str(3)), 129)
        self.assertEqual(len(self.num.str(4)), 143)
        self.assertEqual(len(self.num.str(-1)), 101)
        self.assertEqual(len(self.num.str("pub")), 129)
        self.assertEqual(len(self.num.str("publication")), 129)
        self.assertEqual(len(self.num.str("pdg")), 115)
        self.assertEqual(len(self.num.str("pdg+1")), 129)

        with self.assertRaises(ValueError):
            self.num.str("foo")

        num = self.num.copy()
        num.uncertainties = {}

        self.assertEqual(len(num.str()), 22)
        self.assertTrue(num.str().endswith(" (no uncertainties)"))
        self.assertEqual(len(num.repr().split(" ", 3)[-1]), 25)

    def test_string_flags(self):
        n = Number(8848, {"stat": (30, 20)})
        n.set_uncertainty("syst", (Number.REL, 0.5))

        self.assertEqual(n.str(), "8848.0 +30.0-20.0 (stat) +- 4424.0 (syst)")
        self.assertEqual(n.str(scientific=True), "8.848 +0.03-0.02 (stat) +- 4.424 (syst) x 1E3")
        self.assertEqual(n.str(scientific=True, unit="m"),
            "8.848 +0.03-0.02 (stat) +- 4.424 (syst) x 1E3 m")
        self.assertEqual(n.str(si=True), "8.848 +0.03-0.02 (stat) +- 4.424 (syst) k")
        self.assertEqual(n.str(si=True, unit="m"), "8.848 +0.03-0.02 (stat) +- 4.424 (syst) km")
        self.assertEqual(n.str("%.2f", si=True), "8.85 +0.03-0.02 (stat) +- 4.42 (syst) k")
        self.assertEqual(n.str(-2, si=True), "8.85 +0.03-0.02 (stat) +- 4.42 (syst) k")

        self.assertEqual(n.str("%.3f", si=True), "8.848 +0.030-0.020 (stat) +- 4.424 (syst) k")
        self.assertEqual(n.str(-3, si=False), "8848.000 +30.000-20.000 (stat) +- 4424.000 (syst)")
        self.assertEqual(n.str(-3, si=True), "8.848 +0.030-0.020 (stat) +- 4.424 (syst) k")
        self.assertEqual(n.str(3, si=False), "8848.0 +30.0-20.0 (stat) +- 4424.0 (syst)")
        self.assertEqual(n.str("pdg", si=False), "8848 +30-20 (stat) +- 4000 (syst)")

    def test_uncertainty_parsing(self):
        uncs = {}
        for name in "ABCDEFGH":
            unc = uncs[name] = self.num.get_uncertainty(name)

            self.assertIsInstance(unc, tuple)
            self.assertEqual(len(unc), 2)

        self.assertEqual(uncs["A"], (0.5, 0.5))
        self.assertEqual(uncs["B"], (1.0, 1.0))
        self.assertEqual(uncs["C"], (1.0, 1.5))
        self.assertEqual(uncs["D"], (0.25, 0.25))
        self.assertEqual(uncs["E"], (0.25, 0.5))
        self.assertEqual(uncs["F"], (1.0, 0.5))
        self.assertEqual(uncs["G"], (0.75, 0.3))
        self.assertEqual(uncs["H"], (0.75, 0.3))

        num = self.num.copy()
        num.set_uncertainty("I", (Number.REL, 0.5, Number.ABS, 0.5))
        self.assertEqual(num.get_uncertainty("I"), (1.25, 0.5))
        num.set_uncertainty("J", (0.5j, 0.5))
        self.assertEqual(num.get_uncertainty("J"), (1.25, 0.5))

    def test_uncertainty_combination(self):
        nom = self.num.nominal

        allUp = ptgr(0.5, 1.0, 1.0, 0.25, 0.25, 1.0, 0.75, 0.75)
        allDown = ptgr(0.5, 1.0, 1.5, 0.25, 0.5, 0.5, 0.3, 0.3)
        self.assertEqual(self.num.get(UP), nom + allUp)
        self.assertEqual(self.num.get(DOWN), nom - allDown)

        self.assertEqual(self.num.get(UP, ("A", "B")), nom + ptgr(1, 0.5))
        self.assertEqual(self.num.get(DOWN, ("B", "C")), nom - ptgr(1, 1.5))
        self.assertEqual(self.num.get(DOWN), nom - allDown)

        for name in "ABCDEFGH":
            unc = self.num.get_uncertainty(name)

            self.assertEqual(self.num.get(names=name), nom)

            self.assertEqual(self.num.get(UP, name), nom + unc[0])
            self.assertEqual(self.num.get(DOWN, name), nom - unc[1])

            self.assertEqual(self.num.u(name, UP), unc[0])
            self.assertEqual(self.num.u(name, DOWN), unc[1])

            self.assertEqual(self.num.get(UP, name, factor=True), (nom + unc[0]) / nom)
            self.assertEqual(self.num.get(DOWN, name, factor=True), (nom - unc[1]) / nom)

            self.assertEqual(self.num.get(UP, name, unc=True, factor=True), unc[0] / nom)
            self.assertEqual(self.num.get(DOWN, name, unc=True, factor=True), unc[1] / nom)

            self.assertEqual(self.num.get((UP, DOWN), name, unc=True, factor=True),
                (unc[0] / nom, unc[1] / nom))

    @if_numpy
    def test_uncertainty_combination_numpy(self):
        num = Number(np.array([2, 4, 6]), 2)
        arr = np.array([1, 2, 3])
        num2 = Number(arr, 1)

        # fully correlated division
        d = num / num2
        self.assertEqual(tuple(d()), (2., 2., 2.))
        self.assertEqual(tuple(d.u(direction=UP)), (0., 0., 0.))

        # uncorrelated division
        d = num.div(num2, rho=0., inplace=False)
        self.assertEqual(tuple(d()), (2., 2., 2.))
        self.assertAlmostEqual(d.u(direction=UP)[0], 2. / 1. * 2.**0.5, 6)
        self.assertAlmostEqual(d.u(direction=UP)[1], 2. / 2. * 2.**0.5, 6)
        self.assertAlmostEqual(d.u(direction=UP)[2], 2. / 3. * 2.**0.5, 6)

        # division by plain array
        d = num / arr
        self.assertEqual(tuple(d()), (2., 2., 2.))
        self.assertAlmostEqual(d.u(direction=UP)[0], 2.0, 6)
        self.assertAlmostEqual(d.u(direction=UP)[1], 1.0, 6)
        self.assertAlmostEqual(d.u(direction=UP)[2], 0.666667, 6)

    def test_uncertainty_propagation(self):
        # ops with constants
        num = self.num + 2
        self.assertEqual(num(), 4.5)
        self.assertEqual(num.u("A", UP), self.num.get_uncertainty("A", UP))
        self.assertEqual(num.u("C", UP), self.num.get_uncertainty("C", UP))
        self.assertEqual(num.u("C", DOWN), self.num.get_uncertainty("C", DOWN))

        num = self.num - 2
        self.assertEqual(num(), 0.5)
        self.assertEqual(num.u("A", UP), self.num.get_uncertainty("A", UP))
        self.assertEqual(num.u("C", UP), self.num.get_uncertainty("C", UP))
        self.assertEqual(num.u("C", DOWN), self.num.get_uncertainty("C", DOWN))

        num = self.num * 3
        self.assertEqual(num(), 7.5)
        self.assertEqual(num.u("A", UP), self.num.get_uncertainty("A", UP) * 3.)
        self.assertEqual(num.u("C", UP), self.num.get_uncertainty("C", UP) * 3.)
        self.assertEqual(num.u("C", DOWN), self.num.get_uncertainty("C", DOWN) * 3.)

        num = self.num / 5
        self.assertEqual(num(), 0.5)
        self.assertEqual(num.u("A", UP), self.num.get_uncertainty("A", UP) / 5.)
        self.assertEqual(num.u("C", UP), self.num.get_uncertainty("C", UP) / 5.)
        self.assertEqual(num.u("C", DOWN), self.num.get_uncertainty("C", DOWN) / 5.)

        num = self.num ** 2
        self.assertAlmostEqual(num(), 6.25)
        self.assertAlmostEqual(num.u("A", UP),
            2 * self.num() * self.num.get_uncertainty("A", UP))
        self.assertAlmostEqual(num.u("B", UP),
            2 * self.num() * self.num.get_uncertainty("B", UP))
        self.assertAlmostEqual(num.u("C", UP),
            2 * self.num() * self.num.get_uncertainty("C", UP))
        self.assertAlmostEqual(num.u("C", DOWN),
            2 * self.num() * self.num.get_uncertainty("C", DOWN))

        num = 2. ** self.num
        self.assertAlmostEqual(num(), 2 ** 2.5)
        self.assertAlmostEqual(num.u("A", UP),
            math.log(2.) * num() * self.num.get_uncertainty("A", UP))
        self.assertAlmostEqual(num.u("B", UP),
            math.log(2) * num() * self.num.get_uncertainty("B", UP))
        self.assertAlmostEqual(num.u("C", UP),
            math.log(2) * num() * self.num.get_uncertainty("C", UP))
        self.assertAlmostEqual(num.u("C", DOWN),
            math.log(2) * num() * self.num.get_uncertainty("C", DOWN))

        num = self.num * 0
        self.assertEqual(num(), 0)
        self.assertEqual(num(UP), 0)
        self.assertEqual(num(DOWN), 0)

        # ops with other numbers
        num2 = Number(5, {"A": 2.5, "C": 1})
        num = self.num + num2
        self.assertEqual(num(), 7.5)
        self.assertEqual(num.u("A", UP), 3.0)
        self.assertEqual(num.u("B", UP), 1.0)
        self.assertEqual(num.u("C", UP), 2.0)
        self.assertEqual(num.u("C", DOWN), 2.5)

        num = self.num - num2
        self.assertEqual(num(), -2.5)
        self.assertEqual(num.u("A", UP), 2.0)
        self.assertEqual(num.u("B", UP), 1.0)
        self.assertEqual(num.u("C", UP), 0.)
        self.assertEqual(num.u("C", DOWN), 0.5)

        num = self.num * num2
        self.assertEqual(num(), 12.5)
        self.assertEqual(num.u("A", UP), 8.75)
        self.assertEqual(num.u("B", UP), 5.0)
        self.assertAlmostEqual(num.u("C", UP), 7.5)
        self.assertEqual(num.u("C", DOWN), 10.)

        num = self.num / num2
        self.assertEqual(num(), 0.5)
        self.assertAlmostEqual(num.u("A", UP), 0.15)
        self.assertEqual(num.u("B", UP), 0.2)
        self.assertEqual(num.u("C", UP), 0.1)
        self.assertEqual(num.u("C", DOWN), 0.2)

        num = self.num ** num2
        self.assertAlmostEqual(num(), self.num() ** 5)
        self.assertAlmostEqual(num.u("A", UP), 321.3600420)
        self.assertAlmostEqual(num.u("B", UP), 195.3125)
        self.assertAlmostEqual(num.u("C", DOWN), 382.4502668)

        num = self.num * Number(0, 0)
        self.assertEqual(num(), 0)
        self.assertEqual(num(UP), 0)
        self.assertEqual(num(DOWN), 0)

    def test_ops_registration(self):
        self.assertTrue("exp" in ops)

        self.assertFalse("foo" in ops)

        @ops.register(ufuncs="absolute")
        def foo(x, a, b, c):
            return a + b * x + c * x ** 2

        self.assertTrue("foo" in ops)
        self.assertEqual(ops.get_operation("foo"), foo)
        self.assertIsNone(foo.derivative)

        if HAS_NUMPY:
            self.assertEqual(foo.ufuncs[0], np.abs)
            self.assertEqual(foo.ufuncs[0].__name__, "absolute")
            self.assertEqual(ops.get_ufunc_operation("abs"), foo)

        @foo.derive
        def foo(x, a, b, c):
            return b + 2 * c * x

        self.assertTrue(callable(foo.derivative))

    @if_numpy
    def test_ufuncs(self):
        num = np.multiply(self.num, 2)
        self.assertAlmostEqual(num(), self.num() * 2.)
        self.assertAlmostEqual(num.u("A", UP), 1.)
        self.assertAlmostEqual(num.u("B", UP), 2.)
        self.assertAlmostEqual(num.u("C", DOWN), 3.)

        num = np.exp(Number(np.array([1., 2.]), 3.))
        self.assertAlmostEqual(num.nominal[0], 2.71828, 4)
        self.assertAlmostEqual(num.nominal[1], 7.38906, 4)
        self.assertAlmostEqual(num.get(UP)[0], 10.87313, 4)
        self.assertAlmostEqual(num.get(UP)[1], 29.55623, 4)

        a = Number(np.array([1.0, 2.0]), np.array([0.5, 0.5]))
        b = Number(np.array([2.0, 3.0]), np.array([0.5, 0.5]))
        for op in [operator.truediv, np.divide]:
            c = op(a, b)
            self.assertAlmostEqual(c.nominal[0], 0.5, 5)
            self.assertAlmostEqual(c.nominal[1], 2. / 3., 5)
            self.assertAlmostEqual(c.get(UP)[0], 0.625, 5)
            self.assertAlmostEqual(c.get(UP)[1], 0.722222, 5)
            self.assertAlmostEqual(c.get(DOWN)[0], 0.375, 5)

    def test_op_pow(self):
        num = ops.pow(self.num, 2)
        self.assertEqual(num(), self.num() ** 2.)
        self.assertEqual(num.u("A", UP),
            2. * num() * self.num(UP, "A", unc=True, factor=True))

    def test_op_exp(self):
        num = ops.exp(self.num)
        self.assertEqual(num(), math.exp(self.num()))
        self.assertEqual(num.u("A", UP), self.num.u("A", UP) * num())

    def test_op_log(self):
        num = ops.log(self.num)
        self.assertEqual(num(), math.log(self.num()))
        self.assertEqual(num.u("A", UP), self.num(UP, "A", unc=True, factor=True))

        num = ops.log(self.num, 2.)
        self.assertEqual(num(), math.log(self.num(), 2))
        self.assertAlmostEqual(num.u("A", UP),
            self.num(UP, "A", unc=True, factor=True) / math.log(2))

    def test_op_sin(self):
        num = ops.sin(self.num)
        self.assertEqual(num(), math.sin(self.num()))
        self.assertAlmostEqual(num.u("A", UP),
            self.num.u("A", UP) * abs(math.cos(self.num())))

    def test_op_cos(self):
        num = ops.cos(self.num)
        self.assertEqual(num(), math.cos(self.num()))
        self.assertAlmostEqual(num.u("A", UP),
            self.num.u("A", UP) * abs(math.sin(self.num())))

    def test_op_tan(self):
        num = ops.tan(self.num)
        self.assertEqual(num(), math.tan(self.num()))
        self.assertAlmostEqual(num.u("A", UP),
            self.num.u("A", UP) / abs(math.cos(self.num())) ** 2)

    def test_split_value(self):
        self.assertEqual(split_value(1), (1., 0))
        self.assertEqual(split_value(0.123), (1.23, -1))
        self.assertEqual(split_value(42.5), (4.25, 1))
        self.assertEqual(split_value(0), (0., 0))

    @if_numpy
    def test_split_value_numpy(self):
        sig, mag = split_value(np.array([1., 0.123, -42.5, 0.]))
        self.assertEqual(tuple(sig), (1., 1.23, -4.25, 0.))
        self.assertEqual(tuple(mag), (0, -1, 1, 0))

    def test_match_precision(self):
        self.assertEqual(match_precision(1.234, ".1"), "1.2")
        self.assertEqual(match_precision(1.234, "1."), "1")
        self.assertEqual(match_precision(1.234, ".1", rounding=decimal.ROUND_UP), "1.3")
        self.assertEqual(match_precision(1.0, "1"), "1")
        self.assertEqual(match_precision(1.0, "1.0"), "1.0")
        self.assertEqual(match_precision(1.0, 1.2), "1")
        self.assertEqual(match_precision(1.0, 1.2, force_float=True), "1.0")

    @if_numpy
    def test_match_precision_numpy(self):
        a = np.array([1., 0.123, -42.5, 0.])
        self.assertEqual(tuple(match_precision(a, "1.")), (b"1", b"0", b"-43", b"0"))
        self.assertEqual(tuple(match_precision(a, ".1")), (b"1.0", b"0.1", b"-42.5", b"0.0"))
        self.assertEqual(tuple(match_precision(a, 1.2)), (b"1", b"0", b"-43", b"0"))
        self.assertEqual(tuple(match_precision(a, 1)), (b"1", b"0", b"-43", b"0"))
        self.assertEqual(tuple(match_precision(a, 0.01)), (b"1.00", b"0.12", b"-42.50", b"0.00"))

    def test_calculate_uncertainty(self):
        self.assertEqual(calculate_uncertainty([(3, 0.5), (4, 0.5)]), 2.5)
        self.assertEqual(calculate_uncertainty([(3, 0.5), (4, 0.5)], rho=1), 3.5)
        self.assertEqual(calculate_uncertainty([(3, 0.5), (4, 0.5)], rho={(0, 1): 1}), 3.5)
        self.assertEqual(calculate_uncertainty([(3, 0.5), (4, 0.5)], rho={(1, 2): 1}), 2.5)

    def test_round_uncertainty(self):
        self.assertEqual(round_uncertainty(0.352), ("4", -1, 1))
        self.assertEqual(round_uncertainty(0.352, 1), ("4", -1, 1))
        self.assertEqual(round_uncertainty(0.352, 2), ("35", -2, 2))
        self.assertEqual(round_uncertainty(0.352, "pdg"), ("35", -2, 2))
        self.assertEqual(round_uncertainty(0.352, "pdg+1"), ("352", -3, 3))
        self.assertEqual(round_uncertainty(0.352, "publication"), ("352", -3, 3))
        self.assertEqual(round_uncertainty(0.352, "pub"), ("352", -3, 3))

        self.assertEqual(round_uncertainty(0.835), ("8", -1, 1))
        self.assertEqual(round_uncertainty(0.835, 1), ("8", -1, 1))
        self.assertEqual(round_uncertainty(0.835, 2), ("84", -2, 2))
        self.assertEqual(round_uncertainty(0.835, "pdg"), ("8", -1, 1))
        self.assertEqual(round_uncertainty(0.835, "pdg+1"), ("84", -2, 2))
        self.assertEqual(round_uncertainty(0.835, "publication"), ("84", -2, 2))
        self.assertEqual(round_uncertainty(0.835, "pub"), ("84", -2, 2))

        self.assertEqual(round_uncertainty(0.962), ("1", 0, 1))
        self.assertEqual(round_uncertainty(0.962, 1), ("1", 0, 1))
        self.assertEqual(round_uncertainty(0.962, 2), ("96", -2, 2))
        self.assertEqual(round_uncertainty(0.962, "pdg"), ("10", -1, 2))
        self.assertEqual(round_uncertainty(0.962, "pdg+1"), ("100", -2, 3))
        self.assertEqual(round_uncertainty(0.962, "publication"), ("96", -2, 2))
        self.assertEqual(round_uncertainty(0.962, "pub"), ("96", -2, 2))

        # enforce precision after rounding
        self.assertEqual(round_uncertainty(0.352, 1, 1), ("4", -1, 1))
        self.assertEqual(round_uncertainty(0.352, 1, 2), ("40", -2, 2))
        self.assertEqual(round_uncertainty(0.352, 2, 1), ("4", -1, 1))
        self.assertEqual(round_uncertainty(0.352, 2, 2), ("35", -2, 2))
        self.assertEqual(round_uncertainty(0.962, "pdg", 2), ("10", -1, 2))
        self.assertEqual(round_uncertainty(0.962, "pdg", 3), ("100", -2, 3))
        self.assertEqual(round_uncertainty(0.962, "pdg+1", 3), ("100", -2, 3))
        self.assertEqual(round_uncertainty(0.962, "pdg+1", 4), ("1000", -3, 4))
        self.assertEqual(round_uncertainty(0.962, "publication", 2), ("96", -2, 2))
        self.assertEqual(round_uncertainty(0.962, "publication", 3), ("960", -3, 3))
        self.assertEqual(round_uncertainty(0.962, "pub", 2), ("96", -2, 2))
        self.assertEqual(round_uncertainty(0.962, "pub", 3), ("960", -3, 3))
        self.assertEqual(round_uncertainty(962, "pub", 2), ("96", 1, 2))
        self.assertEqual(round_uncertainty(962, "pub", 3), ("960", 0, 3))

        with self.assertRaises(ValueError):
            round_uncertainty(0.962, "foo")
        with self.assertRaises(ValueError):
            round_uncertainty(0.962, -1)

    @if_numpy
    def test_round_uncertainty_numpy(self):
        a = np.array([0.123, 0.456, 0.987])

        digits, mag, prec = round_uncertainty(a)
        self.assertEqual(tuple(digits), (b"1", b"5", b"1"))
        self.assertEqual(tuple(mag), (-1, -1, 0))

        digits, mag, prec = round_uncertainty(a, 1)
        self.assertEqual(tuple(digits), (b"1", b"5", b"1"))
        self.assertEqual(tuple(mag), (-1, -1, 0))

        digits, mag, prec = round_uncertainty(a, 2)
        self.assertEqual(tuple(digits), (b"12", b"46", b"99"))
        self.assertEqual(tuple(mag), (-2, -2, -2))

        digits, mag, prec = round_uncertainty(a, "pub")
        self.assertEqual(tuple(digits), (b"123", b"46", b"99"))
        self.assertEqual(tuple(mag), (-3, -2, -2))

        digits, mag, prec = round_uncertainty(a, "publication")
        self.assertEqual(tuple(digits), (b"123", b"46", b"99"))
        self.assertEqual(tuple(mag), (-3, -2, -2))

        digits, mag, prec = round_uncertainty(a, "pdg")
        self.assertEqual(tuple(digits), (b"12", b"5", b"10"))
        self.assertEqual(tuple(mag), (-2, -1, -1))

        digits, mag, prec = round_uncertainty(a, "pdg+1")
        self.assertEqual(tuple(digits), (b"123", b"46", b"100"))
        self.assertEqual(tuple(mag), (-3, -2, -2))

    def test_round_value(self):
        self.assertEqual(round_value(1.23, 0.456), ("1", "0", 0))
        self.assertEqual(round_value(1.23, 0.456, 0), ("1", "0", 0))
        self.assertEqual(round_value(1.23, 0.456, 1), ("12", "5", -1))
        self.assertEqual(round_value(1.23, 0.456, 2), ("123", "46", -2))
        self.assertEqual(round_value(1.23, 0.456, -1), ("12", "5", -1))
        self.assertEqual(round_value(1.23, 0.456, -2), ("123", "46", -2))
        self.assertEqual(round_value(1.23, 0.456, "pub"), ("123", "46", -2))
        self.assertEqual(round_value(1.23, 0.456, "publication"), ("123", "46", -2))
        self.assertEqual(round_value(1.23, 0.456, "pdg"), ("12", "5", -1))
        self.assertEqual(round_value(1.23, 0.456, "pdg+1"), ("123", "46", -2))

        self.assertEqual(round_value(1.23), ("1", None, 0))
        self.assertEqual(round_value(1.23, method=0), ("1", None, 0))
        self.assertEqual(round_value(1.23, method=1), ("1", None, 0))
        self.assertEqual(round_value(1.23, method=2), ("12", None, -1))
        self.assertEqual(round_value(1.23, method=-1), ("12", None, -1))
        self.assertEqual(round_value(1.23, method=-2), ("123", None, -2))
        with self.assertRaises(ValueError):
            round_value(1.23, method="pub"), ("123", "46", -2)
        with self.assertRaises(ValueError):
            round_value(1.23, method="publication"), ("123", "46", -2)
        with self.assertRaises(ValueError):
            round_value(1.23, method="pdg"), ("12", "5", -1)
        with self.assertRaises(ValueError):
            round_value(1.23, method="pdg+1"), ("123", "46", -2)

        num = Number(1.23, 0.456)
        val_str, unc_strs, mag = round_value(num, method=-2)
        self.assertEqual(val_str, "123")
        self.assertEqual(unc_strs[0], ("46", "46"))
        self.assertEqual(mag, -2)

    def test_round_value_list(self):
        val_str, unc_strs, mag = round_value(1.23, [0.333, 0.45678, 0.078, 0.951], "pub")
        self.assertEqual(val_str, "1230")
        self.assertEqual(tuple(unc_strs), ("333", "460", "78", "950"))
        self.assertEqual(mag, -3)

        val_str, unc_strs, mag = round_value(1.23, [0.333, 0.45678, 0.078, 0.951], "pdg")
        self.assertEqual(val_str, "123")
        self.assertEqual(tuple(unc_strs), ("33", "50", "8", "100"))
        self.assertEqual(mag, -2)

        val_str, unc_strs, mag = round_value(1.23, [0.333, 0.45678, 0.078, 0.951], "pdg",
            align_precision=False)
        self.assertEqual(val_str, "123")
        self.assertEqual(tuple(unc_strs), ("33", "46", "8", "95"))
        self.assertEqual(mag, -2)

    @if_numpy
    def test_round_value_numpy(self):
        val_str, unc_strs, mag = round_value(np.array([1.23, 4.56, 10]),
            np.array([0.45678, 0.078, 0.998]), "pub")
        self.assertEqual(tuple(val_str), (b"123", b"4560", b"100"))
        self.assertEqual(tuple(unc_strs), (b"46", b"78", b"10"))
        self.assertEqual(tuple(mag), (-2, -3, -1))

        val_str, unc_strs, mag = round_value(np.array([1.23, 4.56, 10]), 1, "pub")
        self.assertEqual(tuple(val_str), (b"123", b"456", b"1000"))
        self.assertEqual(tuple(unc_strs), (b"100", b"100", b"100"))
        self.assertTrue(np.all(mag == -2))

        with self.assertRaises(ValueError):
            round_value(np.array([1.23, 4.56, 10]), method="pub")

    def test_infer_si_prefix(self):
        self.assertEqual(infer_si_prefix(0), ("", 0))
        self.assertEqual(infer_si_prefix(2), ("", 0))
        self.assertEqual(infer_si_prefix(20), ("", 0))
        self.assertEqual(infer_si_prefix(200), ("", 0))
        self.assertEqual(infer_si_prefix(2000), ("k", 3))

        for n in range(-18, 19, 3):
            self.assertEqual(infer_si_prefix(10 ** n)[1], n)

    def test_correlation(self):
        c = Correlation(1.5, foo=0.5)
        self.assertEqual(c.default, 1.5)
        self.assertEqual(c.get("foo"), 0.5)
        self.assertEqual(c.get("bar"), 1.5)
        self.assertEqual(c.get("bar", 0.75), 0.75)

        self.assertEqual(Correlation().default, 1)

        with self.assertRaises(Exception):
            Correlation(1, 1)

    def test_deferred_result(self):
        c = Correlation(1.5, A=0.5)
        d = self.num * c
        self.assertIsInstance(d, DeferredResult)
        self.assertEqual(d.number, self.num)
        self.assertEqual(d.correlation, c)

        self.assertIsInstance(c * self.num, DeferredResult)

        with self.assertRaises(ValueError):
            self.num + c

        if sys.version_info.major >= 3:
            eval("self.num @ c")

    def test_deferred_resolution(self):
        n = (self.num * Correlation(A=1)) + self.num
        self.assertEqual(n.u("A"), (1.0, 1.0))
        self.assertEqual(n.u("B"), (2.0, 2.0))

        n = (self.num * Correlation(A=0)) + self.num
        self.assertEqual(n.u("A"), (0.5**0.5, 0.5**0.5))
        self.assertEqual(n.u("B"), (2.0, 2.0))
