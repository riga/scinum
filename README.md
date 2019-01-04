<img src="https://raw.githubusercontent.com/riga/scinum/master/logo.png" alt="scinum logo" width="250"/>

[![Build Status](https://travis-ci.org/riga/scinum.svg?branch=master)](https://travis-ci.org/riga/scinum) [![Documentation Status](https://readthedocs.org/projects/scinum/badge/?version=latest)](http://scinum.readthedocs.org/en/latest/?badge=latest) [![Package Status](https://badge.fury.io/py/scinum.svg)](https://badge.fury.io/py/scinum) [![License](https://img.shields.io/github/license/riga/scinum.svg)](https://github.com/riga/scinum/blob/master/LICENSE)

scinum provides a simple `Number` class that wraps plain floats or [NumPy](http://www.numpy.org/) arrays and adds support for multiple uncertainties, automatic (gaussian) error propagation, and scientific rounding.


### Usage

The following examples demonstrate the most common use cases. For more info, see the [API documentation](http://scinum.readthedocs.org/en/latest/?badge=latest).


###### Number definition

```python
from scinum import Number, UP, DOWN

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

```python
from scinum import Number, ABS, REL

num = Number(2.5, {
    "sourceA": 0.5,                  # absolute 0.5, both up and down
    "sourceB": (1.0, 1.5),           # absolute 1.0 up, 1.5 down
    "sourceC": (REL, 0.1),           # relative 10%, both up and down
    "sourceD": (REL, 0.1, 0.2),      # relative 10% up, 20% down
    "sourceE": (1.0, REL, 0.2),      # absolute 1.0 up, relative 20% down
    "sourceF": (REL, 0.3, ABS, 0.3)  # relative 30% up, absolute 0.3 down
})
```


###### Formatting and rounding

`Number.str()` provides some simple formatting tools, including `latex` and `root latex` support, as well as scientific rounding rules:

```python
# output formatting
n = Number(8848, 10)
n.str(unit="m")                          # -> "8848.0 +- 10.0 m"
n.str(unit="m", force_asymmetric=True)   # -> "8848.0 +10.0-10.0 m"
n.str(unit="m", scientific=True)         # -> "8.848 +- 0.01 x 1E3 m"
n.str(unit="m", si=True)                 # -> "8.848 +- 0.01 km"
n.str(unit="m", style="latex")           # -> "$8848.0\;\pm\;10.0\;m$"
n.str(unit="m", style="latex", si=True)  # -> "$8.848\;\pm\;0.01\;km$"
n.str(unit="m", style="root")            # -> "8848.0 #pm 10.0 m"
n.str(unit="m", style="root", si=True)   # -> "8.848 #pm 0.01 km"

# output rounding
n = Number(17.321, {"a": 1.158, "b": 0.453})
n.str()               # -> '17.321 +- 1.158 (a) +- 0.453 (b)'
n.str("%.1f")         # -> '17.3 +- 1.2 (a) +- 0.5 (b)'
n.str("publication")  # -> '17.32 +- 1.16 (a) +- 0.45 (b)'
n.str("pdg")          # -> '17.3 +- 1.2 (a) +- 0.5 (b)'
```

For situations that require more sophisticated rounding and formatting rules, you might want to checkout:

- [`sn.split_value()`](http://scinum.readthedocs.io/en/latest/#split-value)
- [`sn.match_precision()`](http://scinum.readthedocs.io/en/latest/#match-precision)
- [`sn.round_uncertainty()`](http://scinum.readthedocs.io/en/latest/#round-uncertainty)
- [`sn.round_value()`](http://scinum.readthedocs.io/en/latest/#round-value)
- [`sn.infer_si_prefix()`](http://scinum.readthedocs.io/en/latest/#infer-si-prefix)


###### NumPy arrays

```python
from scinum import Number, ABS, REL
import numpy as np

num = Number(np.array([3, 4, 5]), 2)
print(num)
# [ 3.  4.  5.]
# + [ 2.  2.  2.]
# - [ 2.  2.  2.]

num = Number(np.array([3, 4, 5]), {
    "sourceA": (np.array([0.1, 0.2, 0.3]), REL, 0.5)  # absolute values for up, 50% down
})
print(num)
# [ 3.  4.  5.]
# + sourceA [ 0.1  0.2  0.3]
# - sourceA [ 1.5  2.   2.5]
```


###### Uncertainty propagation

```python
from scinum import Number

num = Number(5, 1)
print(num + 2)  # -> '7.0 +- 1.0'
print(num * 3)  # -> '15.0 +- 3.0'

num2 = Number(2.5, 1.5)
print(num + num2)  # -> '7.5 +- 1.80277563773'
print(num * num2)  # -> '12.5 +- 7.90569415042'

# add num2 to num and consider their uncertainties to be fully correlated, i.e. rho = 1
num.add(num2, rho=1)
print(num)  # -> '7.5 +- 2.5'
```


###### Math operations

As a drop-in replacement for the `math` module, scinum provides an object `ops` that contains math operations that are aware of guassian error propagation.

```python
from scinum import Number, ops

num = ops.log(Number(5, 2))
print(num)  # -> 1.61 (+0.40, -0.40)

num = ops.exp(ops.tan(Number(5, 2)))
print(num)  # -> 0.03 (+0.85, -0.85)
```


###### Custom operations

There might be situations where a specific operation is not (yet) contained in the `ops` object. In this case, you can easily register a new one via:

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


### Installation and dependencies

Via [pip](https://pypi.python.org/pypi/scinum)

```bash
pip install scinum
```

or by simply copying the file into your project.

Numpy is an optional dependency.


### Contributing

If you like to contribute, I'm happy to receive pull requests. Just make sure to add a new test cases and run them via:

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

docker run --rm -v `pwd`:/root/scinum -w /root/scinum python:3.6 python -m unittest tests
```


### Development

- Source hosted at [GitHub](https://github.com/riga/scinum)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/riga/scinum/issues)


### Contributors

- [Marcel R.](https://github.com/riga) (author)
