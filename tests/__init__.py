# -*- coding: utf-8 -*-


__all__ = ["TestCase"]


import os
import sys
import math
import decimal
import unittest

# adjust the path to import scinum
base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)
from scinum import Number, Operation, ops, HAS_NUMPY, split_value, match_precision, \
    round_uncertainty, round_value, infer_si_prefix

if HAS_NUMPY:
    import numpy as np

UP = Number.UP
DOWN = Number.DOWN


def if_numpy(func):
    return func if HAS_NUMPY else (lambda self: None)


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
            "G": (Number.REL, 0.3, Number.ABS, 0.3)
        })

    def test_constructor(self):
        num = Number(42, 5)

        self.assertIsInstance(num.nominal, float)
        self.assertEqual(num.nominal, 42.)

        unc = num.get_uncertainty(Number.DEFAULT)
        self.assertIsInstance(unc, tuple)
        self.assertEqual(len(unc), 2)
        self.assertEqual(unc, (5., 5.))

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

    def test_strings(self):
        self.assertEqual(len(self.num.str()), 97)
        self.assertEqual(len(self.num.str("%.3f")), 109)

        self.assertEqual(len(self.num.repr().split(" ", 3)[-1]), 100)

        num = self.num.copy()
        num.uncertainties = {}

        self.assertEqual(len(num.str()), 23)
        self.assertTrue(num.str().endswith(" (no uncertainties)"))
        self.assertEqual(len(num.repr().split(" ", 3)[-1]), 26)

    def test_uncertainty_parsing(self):
        uncs = {}
        for name in "ABCDEFG":
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

        num = self.num.copy()
        num.set_uncertainty("H", (Number.REL, 0.5, Number.ABS, 0.5))
        self.assertEqual(num.get_uncertainty("H"), (1.25, 0.5))

    def test_uncertainty_combination(self):
        nom = self.num.nominal

        allUp = ptgr(0.5, 1.0, 1.0, 0.25, 0.25, 1.0, 0.75)
        allDown = ptgr(0.5, 1.0, 1.5, 0.25, 0.5, 0.5, 0.3)
        self.assertEqual(self.num.get(UP), nom + allUp)
        self.assertEqual(self.num.get(DOWN), nom - allDown)

        self.assertEqual(self.num.get(UP, ("A", "B")), nom + ptgr(1, 0.5))
        self.assertEqual(self.num.get(DOWN, ("B", "C")), nom - ptgr(1, 1.5))
        self.assertEqual(self.num.get(DOWN), nom - allDown)

        for name in "ABCDEFG":
            unc = self.num.get_uncertainty(name)

            self.assertEqual(self.num.get(names=name), nom)

            self.assertEqual(self.num.get(UP, name), nom + unc[0])
            self.assertEqual(self.num.get(DOWN, name), nom - unc[1])

            self.assertEqual(self.num.u(name, UP), unc[0])
            self.assertEqual(self.num.u(name, DOWN), unc[1])

            self.assertEqual(self.num.get(UP, name, factor=True), (nom + unc[0]) / nom)
            self.assertEqual(self.num.get(DOWN, name, factor=True), (nom - unc[1]) / nom)

            self.assertEqual(self.num.get(UP, name, diff=True, factor=True), unc[0] / nom)
            self.assertEqual(self.num.get(DOWN, name, diff=True, factor=True), unc[1] / nom)

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

        # ops with other numbers
        num2 = Number(5, {"A": 2.5, "C": 1})
        num = self.num + num2
        self.assertEqual(num(), 7.5)
        self.assertEqual(num.u("A", UP), ptgr(0.5, 2.5))
        self.assertEqual(num.u("B", UP), 1.0)
        self.assertEqual(num.u("C", UP), ptgr(1.0, 1.0))
        self.assertEqual(num.u("C", DOWN), ptgr(1.0, 1.5))

        num = self.num - num2
        self.assertEqual(num(), -2.5)
        self.assertEqual(num.u("A", UP), ptgr(0.5, 2.5))
        self.assertEqual(num.u("B", UP), 1.0)
        self.assertEqual(num.u("C", UP), ptgr(1.0, 1.0))
        self.assertEqual(num.u("C", DOWN), ptgr(1.0, 1.5))

        num = self.num * num2
        self.assertEqual(num(), 12.5)
        self.assertEqual(num.u("A", UP), num() * ptgr(0.5, 0.2))
        self.assertEqual(num.u("B", UP), num() * 0.4)
        self.assertEqual(num.u("C", UP), num() * ptgr(0.2, 0.4))
        self.assertEqual(num.u("C", DOWN), num() * ptgr(0.2, 0.6))

        num = self.num / num2
        self.assertEqual(num(), 0.5)
        self.assertEqual(num.u("A", UP), num() * ptgr(0.5, 0.2))
        self.assertEqual(num.u("B", UP), num() * 0.4)
        self.assertEqual(num.u("C", UP), num() * ptgr(0.2, 0.4))
        self.assertEqual(num.u("C", DOWN), num() * ptgr(0.2, 0.6))

        num = self.num ** num2
        self.assertAlmostEqual(num(), self.num() ** 5)
        self.assertAlmostEqual(num.u("A", UP), ptgr(
                5 * self.num() ** 4 * self.num.get_uncertainty("A", UP),
                num() * math.log(self.num()) * num2.get_uncertainty("A", UP)))
        self.assertAlmostEqual(num.u("B", UP),
                5 * self.num() ** 4 * self.num.get_uncertainty("B", UP))
        self.assertAlmostEqual(num.u("C", DOWN), ptgr(
                5 * self.num() ** 4 * self.num.get_uncertainty("C", DOWN),
                num() * math.log(self.num()) * num2.get_uncertainty("C", DOWN)))

    def test_ops_registration(self):
        self.assertTrue("exp" in ops)

        self.assertFalse("foo" in ops)

        @ops.register
        def foo(x, a, b, c):
            return a + b * x + c * x ** 2

        self.assertTrue("foo" in ops)
        self.assertEqual(ops.get_operation("foo"), foo)
        self.assertIsNone(foo.derivative)

        @foo.derive
        def foo(x, a, b, c):
            return b + 2 * c * x

        self.assertTrue(callable(foo.derivative))

    def test_op_pow(self):
        num = ops.pow(self.num, 2)
        self.assertEqual(num(), self.num() ** 2.)
        self.assertEqual(num.u("A", UP),
                2. * num() * self.num(UP, "A", diff=True, factor=True))

    def test_op_exp(self):
        num = ops.exp(self.num)
        self.assertEqual(num(), math.exp(self.num()))
        self.assertEqual(num.u("A", UP), self.num.u("A", UP) * num())

    def test_op_log(self):
        num = ops.log(self.num)
        self.assertEqual(num(), math.log(self.num()))
        self.assertEqual(num.u("A", UP), self.num(UP, "A", diff=True, factor=True))

        num = ops.log(self.num, 2.)
        self.assertEqual(num(), math.log(self.num(), 2))
        self.assertAlmostEqual(num.u("A", UP),
                self.num(UP, "A", diff=True, factor=True) / math.log(2))

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
        self.assertEqual(match_precision(1.234, ".1", decimal.ROUND_UP), "1.3")

    @if_numpy
    def test_match_precision_numpy(self):
        a = np.array([1., 0.123, -42.5, 0.])
        self.assertEqual(tuple(match_precision(a, "1.")), ("1", "0", "-42", "0"))
        self.assertEqual(tuple(match_precision(a, ".1")), ("1.0", "0.1", "-42.5", "0.0"))

    def test_round_uncertainty(self):
        self.assertEqual(round_uncertainty(0.352, "pdg"), ("35", -2))
        self.assertEqual(round_uncertainty(0.352, "pub"), ("352", -3))
        self.assertEqual(round_uncertainty(0.352, "one"), ("4", -1))

        self.assertEqual(round_uncertainty(0.835, "pdg"), ("8", -1))
        self.assertEqual(round_uncertainty(0.835, "pub"), ("84", -2))
        self.assertEqual(round_uncertainty(0.835, "one"), ("8", -1))

        self.assertEqual(round_uncertainty(0.962, "pdg"), ("10", -1))
        self.assertEqual(round_uncertainty(0.962, "pub"), ("962", -3))
        self.assertEqual(round_uncertainty(0.962, "one"), ("10", -1))

        self.assertEqual(round_uncertainty(0.532, "pdg"), ("5", -1))
        self.assertEqual(round_uncertainty(0.532, "pub"), ("53", -2))
        self.assertEqual(round_uncertainty(0.532, "one"), ("5", -1))

        self.assertEqual(round_uncertainty(0.895, "pdg"), ("9", -1))
        self.assertEqual(round_uncertainty(0.895, "pub"), ("90", -2))
        self.assertEqual(round_uncertainty(0.895, "one"), ("9", -1))

    @if_numpy
    def test_round_uncertainty_numpy(self):
        digits, mag = round_uncertainty(np.array([0.123, 0.456, 0.987]))
        self.assertEqual(tuple(digits), ("123", "46", "987"))
        self.assertEqual(tuple(mag), (-3, -2, -3))

    def test_round_value(self):
        val_str, unc_strs, mag = round_value(1.23, 0.456)
        self.assertEqual(val_str, "123")
        self.assertEqual(tuple(unc_strs), ("46",))
        self.assertEqual(mag, -2)

        num = Number(1.23, 0.456)
        val_str, unc_strs, mag = round_value(num)
        self.assertEqual(val_str, "123")
        self.assertEqual(tuple(unc_strs), ("46", "46"))
        self.assertEqual(mag, -2)

        with self.assertRaises(ValueError):
            round_value(1.23)

    def test_round_value_list(self):
        val_str, unc_strs, mag = round_value(1.23, [0.45678, 0.078, 0.998])
        self.assertEqual(val_str, "1230")
        self.assertEqual(tuple(unc_strs[0]), ("457", "78", "998"))
        self.assertEqual(mag, -3)

    @if_numpy
    def test_round_value_numpy(self):
        val_str, unc_strs, mag = round_value(np.array([1.23, 4.56, 10]), np.array([0.45678, 0.078, 0.998]))
        self.assertEqual(tuple(val_str), ("1230", "4560", "10000"))
        self.assertEqual(tuple(unc_strs[0]), ("457", "78", "998"))
        self.assertEqual(mag, -3)

    def test_infer_si_prefix(self):
        self.assertEqual(infer_si_prefix(0), ("", 0))
        self.assertEqual(infer_si_prefix(2), ("", 0))
        self.assertEqual(infer_si_prefix(20), ("", 0))
        self.assertEqual(infer_si_prefix(200), ("", 0))
        self.assertEqual(infer_si_prefix(2000), ("k", 3))

        for n in range(-18, 19, 3):
            self.assertEqual(infer_si_prefix(10 ** n)[1], n)
