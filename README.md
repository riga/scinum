<img src="https://raw.githubusercontent.com/riga/scinum/master/logo.png" alt="scinum logo" width="250"/>

[![Build Status](https://travis-ci.org/riga/scinum.svg?branch=master)](https://travis-ci.org/riga/scinum) [![Documentation Status](https://readthedocs.org/projects/scinum/badge/?version=latest)](http://scinum.readthedocs.org/en/latest/?badge=latest) [![Package Status](https://badge.fury.io/py/scinum.svg)](https://badge.fury.io/py/scinum)

scinum provides a simple `Number` class that wraps plain floats or [NumPy](http://www.numpy.org/) arrays and adds support for multiple uncertainties, automatic (gaussian) error propagation, and scientific rounding.


### Usage

The following examples demonstrate the most common use cases. For more info, see the [API documentation](http://scinum.readthedocs.org/en/latest/?badge=latest).


###### Number definition

```python
from scinum import Number
UP = Number.UP

num = Number(5, (2, 1))
print(num)                    # -> 5.00 +2.00-1.00
print(num.nominal)            # -> 5.0
print(num.n)                  # -> 5.0 (shorthand)
print(num.get_uncertainty())  # -> (2.0, 1.0)
print(num.u())                # -> (2.0, 1.0) (shorthand)
print(num.u(direction=UP))    # -> 2.0
```


###### Multiple uncertainties

```python
from scinum import Number
ABS, REL = Number.ABS, Number.REL

num = Number(2.5, {
    "sourceA": 0.5,                  # absolute 0.5, both up and down
    "sourceB": (1.0, 1.5),           # absolute 1.0 up, 1.5 down
    "sourceC": (REL, 0.1),           # relative 10%, both up and down
    "sourceD": (REL, 0.1, 0.2),      # relative 10% up, 20% down
    "sourceE": (1.0, REL, 0.2),      # absolute 1.0 up, relative 20% down
    "sourceF": (REL, 0.3, ABS, 0.3)  # relative 30% up, absolute 0.3 down
})
```


###### Formatting

`Number.str()` provides some simple formatting tools, including `latex` and `root latex` support:

```python
n = Number(17.321, {"a": 1.158, "b": 0.453})
n.str()               # -> '17.32 +1.16-1.16 (a), +0.45-0.45 (b)'
n.str("%.3f")         # -> '17.321 +1.158-1.158 (a), +0.453-0.453 (b)'
n.str("publication")  # -> '17.32 +1.16-1.16 (a) +0.45-0.45 (b)'
n.str("pdg")          # -> '17.3 +1.2-1.2 (a) +0.5-0.5 (b)'

n = Number(8848, 10)
n.str(unit="m")                          # -> "8848.00 +-10.00 m"
n.str(unit="m", force_asymmetric=True)   # -> "8848.00 +10.00-10.00 m"
n.str(unit="m", scientific=True)         # -> "8.85 +-0.01 x 1E3 m"
n.str(unit="m", si=True)                 # -> "8.85 +-0.01 km"
n.str(unit="m", style="latex")           # -> "$8848.00\;\pm10.00\;m$"
n.str(unit="m", style="latex", si=True)  # -> "$8.85\;\pm0.01\;km$"
n.str(unit="m", style="root")            # -> "8848.00 #pm 10.00 m"
n.str(unit="m", style="root", si=True)   # -> "8.85 #pm 0.01 km"
```


###### NumPy arrays

```python
from scinum import Number
ABS, REL = Number.ABS, Number.REL
import numpy as np

num = Number(np.array([3, 4, 5]), 2)
print(num)
# [ 3.  4.  5.]
# + [ 2.  2.  2.]
# - [ 2.  2.  2.]

num = Number(np.array([3, 4, 5]), {
    "sourceA": (np.array([0.1, 0.2, 0.3]), REL, 0.5)
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
print(num + 2)  # -> 7.00 (+1.00, -1.00)
print(num * 3)  # -> 15.00 (+3.00, -3.00)

num2 = Number(2.5, 1.5)
print(num + num2)  # -> 7.50 (+1.80, -1.80)
print(num * num2)  # -> 12.50 (+7.91, -7.91)

num.add(num2, rho=1)
print(num)  # -> 7.5 (+2.50, -2.50)
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

Please note that there is no need to register *simple* functions as in the particular example above as most of them are just composite operations whose propagation rules (derivatives) are already known.


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
- Python 3.5
- Python 3.6
- Python 3.X + NumPy


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


### Authors

- [Marcel Rieger](https://github.com/riga)


### License

The MIT License (MIT)

Copyright (c) 2017-2018 Marcel Rieger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
