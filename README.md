![scinum logo](https://media.githubusercontent.com/media/riga/scinum/master/assets/logo250.png "scinum logo")

[![Lint and test](https://github.com/riga/scinum/actions/workflows/lint_and_test.yml/badge.svg)](https://github.com/riga/scinum/actions/workflows/lint_and_test.yml)
[![Documentation Status](https://readthedocs.org/projects/scinum/badge/?version=latest)](http://scinum.readthedocs.org/en/latest/?badge=latest)
[![Cover coverage](https://codecov.io/gh/riga/scinum/branch/master/graph/badge.svg?token=bvykpaUaHQ)](https://codecov.io/gh/riga/scinum)
[![Package Status](https://img.shields.io/pypi/v/scinum.svg?style=flat)](https://pypi.python.org/pypi/scinum)
[![License](https://img.shields.io/github/license/riga/scinum.svg)](https://github.com/riga/scinum/blob/master/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/scinum.svg)](https://pypi.python.org/pypi/scinum)

scinum provides a simple `Number` class that wraps plain floats or [NumPy](http://www.numpy.org/) arrays and adds support for multiple uncertainties, automatic (gaussian) error propagation, and scientific rounding.


### Usage

The following examples demonstrate the most common use cases.
For more info, see the [API documentation](http://scinum.readthedocs.org/en/latest/?badge=latest) or open the [example.ipynb](https://github.com/riga/scinum/blob/master/example.ipynb) notebook on colab or binder.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riga/scinum/blob/master/example.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/riga/scinum/master?filepath=example.ipynb)


###### Number definition

```python
from scinum import Number, UP, DOWN

Number.default_format = "%.2f"

num = Number(5, (2, 1))
print(num)                    # -> 5.00 +2.00-1.00

# get the nominal value
print(num.nominal)            # -> 5.0
print(num.n)                  # -> 5.0 (shorthand)
print(num())                  # -> 5.0 (shorthand)

# get uncertainties
print(num.get_uncertainty())  # -> (2.0, 1.0)
print(num.u())                # -> (2.0, 1.0) (shorthand)
print(num.u(direction=UP))    # -> 2.0

# get shifted values
print(num.get())              # -> 5.0 (no shift)
print(num.get(UP))            # -> 7.0 (up shift)
print(num(UP))                # -> 7.0 (up shift, shorthand)
print(num.get(DOWN))          # -> 4.0 (down shift)
print(num(DOWN))              # -> 4.0 (down shift, shorthand)
```


###### Multiple uncertainties

Use single values to denote symmetric uncertainties, and tuples for asymmetric ones.
Float values refer to absolute values whereas complex numbers (only their imaginary part) define relative effects.

```python
from scinum import Number, ABS, REL

num = Number(2.5, {
    "sourceA": 0.5,              # absolute 0.5, both up and down
    "sourceB": (1.0, 1.5),       # absolute 1.0 up, 1.5 down
    "sourceC": 0.1j,             # relative 10%, both up and down
    "sourceD": (0.1j, 0.2j),     # relative 10% up, relative 20% down
    "sourceE": (1.0, 0.2j),      # absolute 1.0 up, relative 20% down
    "sourceF": (0.3j, 0.3),      # relative 30% up, absolute 0.3 down
    # the old 'marker' syntax
    "sourceG": (REL, 0.1, 0.2),       # relative 10% up, relative 20% down
    "sourceH": (REL, 0.1, ABS, 0.2),  # relative 10% up, absolute 0.2 down
})
```


###### Correlation handling

When two numbers are combined by means of an operator, the correlation between equally named uncertainties is assumed to be 1.
The example above shows how to configure this correlation coefficient `rho` when used with explicit operator methods defined on a number, such as `num.add()` or `num.mul()`.

However, it is probably more convenient to use `Correlation` objects:

```python
from scinum import Number, Correlation

num = Number(2, 5)
print(num * num)  # -> '4.0 +-20.0', fully correlated by default
# same as
# print(num**2)
# print(num.pow(2, inplace=False))

print(num * Correlation(0) * num)  # -> '4.0 +-14.14', no correlation
# same as
# print(num.pow(2, rho=0, inplace=False))
```

The correlation object is combined with a number through multiplication, resulting in a `DeferredResult` object.
The deferred result is used to resolve the actual uncertainty combination once it is applied to another number instance which happens in a second step.
Internally, the above example is handled as

```python
deferred = num * Correlation(0)
print(deferred * num)
```

and similarly, adding two numbers without correlation can be expressed as

```python
(num * Correlation(0)) + num
```

When combining numbers with multiple, named uncertainties, correlation coefficients can be controlled per uncertainty by passing names to the `Correlation` constructor.

```python
Correlation(1, sourceA=0)  # zero correlation for sourceA, all others default to 1
Correlation(sourceA=0)     # zero correlation for sourceA, no default
```

###### Formatting and rounding

`Number.str()` provides some simple formatting tools, including `latex` and `root latex` support, as well as scientific rounding rules:

```python
# output formatting
n = Number(8848, 10)
n.str(unit="m")                          # -> "8848.0 +-10.0 m"
n.str(unit="m", force_asymmetric=True)   # -> "8848.0 +10.0-10.0 m"
n.str(unit="m", scientific=True)         # -> "8.848 +-0.01 x 1E3 m"
n.str(unit="m", si=True)                 # -> "8.848 +-0.01 km"
n.str(style="fancy")                     # -> "$8848.0 ±10.0$"
n.str(unit="m", style="fancy")           # -> "$8848.0 ±10.0\,m$"
n.str(unit="m", style="latex")           # -> "$8848.0 \pm 10.0\,m$"
n.str(unit="m", style="latex", si=True)  # -> "8.848 \pm 0.01\,km"
n.str(unit="m", style="root")            # -> "8848.0 #pm 10.0 m"
n.str(unit="m", style="root", si=True)   # -> "8.848 #pm 0.01 km"

# output rounding
n = Number(17.321, {"a": 1.158, "b": 0.453})
n.str()               # -> '17.321 +-1.158 (a) +-0.453 (b)'
n.str("%.1f")         # -> '17.3 +-1.2 (a) +-0.5 (b)'
n.str("publication")  # -> '17.32 +-1.16 (a) +-0.45 (b)'
n.str("pdg")          # -> '17.3 +-1.2 (a) +-0.5 (b)'
```

For situations that require more sophisticated rounding and formatting rules, you might want to checkout:

- [`sn.split_value()`](http://scinum.readthedocs.io/en/latest/#split-value)
- [`sn.match_precision()`](http://scinum.readthedocs.io/en/latest/#match-precision)
- [`sn.round_uncertainty()`](http://scinum.readthedocs.io/en/latest/#round-uncertainty)
- [`sn.round_value()`](http://scinum.readthedocs.io/en/latest/#round-value)
- [`sn.infer_si_prefix()`](http://scinum.readthedocs.io/en/latest/#infer-si-prefix)


###### Uncertainty propagation

```python
from scinum import Number

num = Number(5, 1)
print(num + 2)  # -> '7.0 +-1.0'
print(num * 3)  # -> '15.0 +-3.0'

num2 = Number(2.5, 1.5)
print(num + num2)  # -> '7.5 +-2.5'
print(num * num2)  # -> '12.5 +-10.0'

# add num2 to num and consider their uncertainties to be fully uncorrelated, i.e. rho = 0
num.add(num2, rho=0)
print(num)  # -> '7.5 +-1.80277563773'
```


###### Math operations

As a drop-in replacement for the `math` module, scinum provides an object `ops` that contains math operations that are aware of gaussian error propagation.

```python
from scinum import Number, ops

num = ops.log(Number(5, 2))
print(num)  # -> 1.60943791243 +-0.4

num = ops.exp(ops.tan(Number(5, 2)))
print(num)  # -> 0.0340299245972 +-0.845839754815
print(num.str("%.2f"))  # -> 0.03 +-0.85
```


###### Custom operations

There might be situations where a specific operation is not (yet) contained in the `ops` object.
In this case, you can easily register a new one via:

```python
from scinum import Number, ops

@ops.register
def my_op(x):
    return x * 2 + 1

@my_op.derive
def my_op(x):
    return 2

num = ops.my_op(Number(5, 2))
print(num)  # -> 11.00 (+4.00, -4.00)
```

Please note that there is no need to register *simple* functions like in the particular example above as most of them are just composite operations whose propagation rules (derivatives) are already known.


###### NumPy arrays

```python
from scinum import Number
import numpy as np

num = Number(np.array([3, 4, 5]), 2)
print(num)
# [ 3.  4.  5.]
# + [ 2.  2.  2.]
# - [ 2.  2.  2.]

num = Number(np.array([3, 4, 5]), {
    "sourceA": (np.array([0.1, 0.2, 0.3]), 0.5j),  # absolute values for up, 50% down
})
print(num)
# [ 3.  4.  5.]
# + sourceA [ 0.1  0.2  0.3]
# - sourceA [ 1.5  2.   2.5]
```


### Installation and dependencies

Via [pip](https://pypi.python.org/pypi/scinum)

```bash
pip install scinum
```

or by simply copying the file into your project.

Numpy is an optional dependency.


### Contributing

If you like to contribute, I'm happy to receive pull requests.
Just make sure to add a new test cases and run them via:

```bash
> python -m unittest tests
```


##### Testing

In general, tests should be run for different environments:

- Python 2.7
- Python 3.X (X ≥ 5)


##### Docker

To run tests in a docker container, do:

```bash
git clone https://github.com/riga/scinum.git
cd scinum

docker run --rm -v `pwd`:/scinum -w /scinum python:3.8 python -m unittest tests
```


### Development

- Source hosted at [GitHub](https://github.com/riga/scinum)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/riga/scinum/issues)
