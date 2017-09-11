# -*- coding: utf-8 -*-

"""
Scientific numbers with multiple uncertainties and correlation-aware, gaussian propagation.
"""


__author__     = "Marcel Rieger"
__email__      = "python-scinum@googlegroups.com"
__copyright__  = "Copyright 2017, Marcel Rieger"
__credits__    = ["Marcel Rieger"]
__contact__    = "https://github.com/riga/scinum"
__license__    = "MIT"
__status__     = "Development"
__version__    = "0.0.1"
__all__        = ["Number", "Operation", "ops"]


import sys
import math
import functools
import operator
import types

# optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


# version related adjustments
if sys.version_info > (3, 0):
    basestring = str


# metaclass decorator from six package, credits to Benjamin Peterson
def add_metaclass(metaclass):
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get("__slots__")
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop("__dict__", None)
        orig_vars.pop("__weakref__", None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper


class typed(property):
    """
    Shorthand for the most common property definition. Can be used as a decorator to wrap around
    a single function. Example:

    .. code-block:: python

         class MyClass(object):

            def __init__(self):
                self._foo = None

            @typed
            def foo(self, foo):
                if not isinstance(foo, str):
                    raise TypeError("not a string: '%s'" % foo)
                return foo

        myInstance = MyClass()
        myInstance.foo = 123   -> TypeError
        myInstance.foo = "bar" -> ok
        print(myInstance.foo)  -> prints "bar"

    In the exampe above, set/get calls target the instance member ``_foo``, i.e. "_<function_name>".
    The member name can be configured by setting *name*. If *setter* (*deleter*) is *True* (the
    default), a setter (deleter) method is booked as well. Prior to updating the member when the
    setter is called, *fparse* is invoked which may implement sanity checks.
    """

    def __init__(self, fparse=None, setter=True, deleter=True, name=None):
        self._flags = (setter, deleter)

        # only register the property if fparse is set
        if fparse is not None:
            self.fparse = fparse

            # build the default name
            if name is None:
                name = "_" + fparse.__name__

            # call the super constructor with generated methods
            property.__init__(self,
                functools.wraps(fparse)(self._fget(name)),
                self._fset(name) if setter else None,
                self._fdel(name) if deleter else None
            )

    def __call__(self, fparse):
        return self.__class__(fparse, *self._flags)

    def _fget(self, name):
        """
        Build and returns the property's *fget* method for the member defined by *name*.
        """
        def fget(inst):
            return getattr(inst, name)
        return fget

    def _fset(self, name):
        """
        Build and returns the property's *fdel* method for the member defined by *name*.
        """
        def fset(inst, value):
            # the setter uses the wrapped function as well
            # to allow for value checks
            value = self.fparse(inst, value)
            setattr(inst, name, value)
        return fset

    def _fdel(self, name):
        """
        Build and returns the property's *fdel* method for the member defined by *name*.
        """
        def fdel(inst):
            delattr(inst, name)
        return fdel


class Number(object):
    """ __init__(nominal=0.0, uncertainties={})
    Implementation of a scientific number, i.e., a *nominal* value with named *uncertainties*.
    *uncertainties* mist be a dict or convertable to a dict with strings as keys. If a value is an
    int or float, it is interpreted as an absolute, symmetric uncertainty. If it is a tuple, it is
    interpreted in different ways. Examples:

    .. code-block:: python

        num = Number(2.5, {
            "sourceA": 0.5,                               # absolute 0.5, both up and down
            "sourceB": (1.0, 1.5),                        # absolute 1.0 up, 1.5 down
            "sourceC": (Number.REL, 0.1),                 # relative 10%, both up and down
            "sourceD": (Number.REL, 0.1, 0.2),            # relative 10% up, 20% down
            "sourceE": (1.0, Number.REL, 0.2),            # absolute 1.0 up, relative 20% down
            "sourceF": (Number.REL, 0.3, Number.ABS, 0.3) # relative 30% up, absolute 0.3 down
        })

        # get the nominal value via direct access 
        num.nominal # => 2.5

        # get the nominal value via __call__() (same as get())
        num()                    # => 2.5
        num(direction="nominal") # => 2.5

        # get uncertainties
        num.get_uncertainty("sourceA") # => (0.5, 0.5)
        num.get_uncertainty("sourceB") # => (1.0, 1.5)
        num.get_uncertainty("sourceC") # => (0.25, 0.25)
        num.get_uncertainty("sourceD") # => (0.25, 0.5)
        num.get_uncertainty("sourceE") # => (1.0, 0.5)
        num.get_uncertainty("sourceF") # => (0.75, 0.3)

        # get shifted values via __call__() (same as get())
        num(Number.UP, "sourceA")              # => 3.0
        num(Number.DOWN, "sourceB")            # => 1.0
        num(Number.UP, ("sourceC", "sourceD")) # => 2.854...
        num(Number.UP, Number.ALL)             # => 4.214...

        # get only the uncertainty (unsigned)
        num(Number.DOWN, ("sourceE", "sourceF"), diff=True) # => 0.583...

        # get the uncertainty factor (unsigned)
        num(Number.DOWN, ("sourceE", "sourceF"), factor=True) # => 1.233...

        # combined
        num(Number.DOWN, ("sourceE", "sourceF"), diff=True, factor=True) # => 0.233...

    When *uncertainties* is not a dictionary, it is interpreted as the *default* uncertainty, named
    ``Number.DEFAULT``.

    This class re-defines most of Python's magic functions to allow transparent use in standard
    operations like ``+``, ``*``, etc. Uncertainties are propagated automatically. When operations
    connect two number instances, their uncertainties are combined assuming there is no correlation.
    For correlation-aware operations, please refer to methods such as :py:meth:`add` or
    :py:meth:`mul` below. Examples:

    .. code-block:: python

        num = Number(5, 1)
        print(num + 2) # -> 7.00 (+1.00, -1.00)
        print(num * 3) # -> 15.00 (+3.00, -3.00)

        num2 = Number(2.5, 1.5)
        print(num + num2) # -> 7.50 (+1.80, -1.80)
        print(num * num2) # -> 12.50 (+7.91, -7.91)

        num.add(num2, rho=1)
        print(num) # -> 7.5 (+2.50, -2.50)

    .. py:attribute:: DEFAULT
       classmember

       Constant that denotes the default uncertainty (``"default"``).

    .. py:attribute:: ALL
       classmember

       Constant that denotes all unceratinties (``"ALL"``).

    .. py:attribute:: REL
       classmember

       Constant that denotes relative errors (``"REL"``).

    .. py:attribute:: ABS
       classmember

       Constant that denotes absolute errors (``"ABS"``).

    .. py:attribute:: NOMINAL
       classmember

       Constant that denotes the nominal value (``"NOMINAL"``).

    .. py:attribute:: UP
       classmember

       Constant that denotes the up direction (``"UP"``).

    .. py:attribute:: DOWN
       classmember

       Constant that denotes the down direction (``"DOWN"``).

    .. py:attribute:: nominal
       type: float

       The nominal value.

    .. py:attribute:: n
       type: float

       Shorthand for :py:attr:`nominal`.

    .. py:attribute:: uncertainties
       type: dictionary

       The uncertainty dictionary that maps names to 2-tuples holding absolute up/down effects.

    .. py:attribute:: is_numpy
       type: bool
       read-only

       Whether or not a NumPy array is wrapped.

    .. py:attribute:: shape
       type: tuple

       The shape of the wrapped NumPy array or *None*, depending on what type is wrapped.

    .. py:attribute:: dtype
       type: type

       The default dtype to use when a NumPy array is wrapped. The initial value is
       ``numpy.float32`` when NumPy is available, *None* otherwise.
    """

    # uncertainty flags
    DEFAULT = "DEFAULT"
    ALL = "ALL"

    # uncertainty types
    REL = "REL"
    ABS = "ABS"

    # uncertainty directions
    NOMINAL = "NOMINAL"
    UP = "UP"
    DOWN = "DOWN"

    def __init__(self, nominal=0.0, uncertainties=None):
        super(Number, self).__init__()

        # wrapped values
        self._nominal = None
        self._uncertainties = {}

        # numpy settings
        self.dtype = np.float32 if HAS_NUMPY else None

        # set initial values
        self.nominal = nominal
        if uncertainties is not None:
            self.uncertainties = uncertainties

    @typed
    def nominal(self, nominal):
        # parser for the typed member holding the nominal value
        if isinstance(nominal, (int, float)):
            if self.uncertainties and is_numpy(list(self.uncertainties.values())[0][0]):
                raise TypeError("cannot set nominal to plain value when uncertainties are arrays")
            nominal = float(nominal)
        elif is_numpy(nominal):
            # check and adjust uncertainties
            if self.uncertainties:
                first_unc = list(self.uncertainties.values())[0][0]
                # convert to arrays
                if not is_numpy(first_unc):
                    for name, unc in self.uncertainties.items():
                        unc = tuple(u * np.ones(nominal.shape, dtype=self.dtype) for u in unc)
                        self._uncertainties[name] = unc
                # compare shape if already an array
                elif nominal.shape != first_unc.shape:
                    raise ValueError("shape not matching uncertainty shape: %s" % nominal.shape)
            nominal = nominal.astype(self.dtype)
        else:
            raise TypeError("invalid nominal value: %s" % nominal)

        return nominal

    @property
    def n(self):
        return self.nominal

    @n.setter
    def n(self, n):
        self.nominal = n

    @typed
    def uncertainties(self, uncertainties):
        # parser for the typed member holding the uncertainties
        if not isinstance(uncertainties, dict):
            try:
                uncertainties = dict(uncertainties)
            except:
                uncertainties = {self.DEFAULT: uncertainties}

        _uncertainties = {}
        for name, val in uncertainties.items():
            # check the name
            if not isinstance(name, basestring):
                raise TypeError("invalid uncertainty name: %s" % name)

            # parse the value type
            if isinstance(val, (int, float)) or is_numpy(val):
                val = (val, val)
            elif isinstance(val, list):
                val = tuple(val)
            elif not isinstance(val, tuple):
                raise TypeError("invalid uncertainty type: %s" % val)

            # parse the value itself
            utype, up, down = self.ABS, None, None
            for v in val:
                # check if the uncertainty type is changed
                if isinstance(v, basestring):
                    if v not in (self.ABS, self.REL):
                        raise ValueError("unknown uncertainty type: %s" % v)
                    utype = v
                    continue

                # parse the value
                if isinstance(v, (int, float)):
                    v = float(v)
                    # convert to array when nominal is in array
                    if self.is_numpy:
                        v *= np.ones(self.shape, dtype=self.dtype)
                elif is_numpy(v):
                    # check the shape
                    if v.shape != self.shape:
                        raise ValueError("shape not matching nominal shape: %s" % v.shape)
                    v = v.astype(self.dtype)
                else:
                    raise TypeError("invalid uncertainty value: %s" % v)

                # convert to abs
                if utype == self.REL:
                    v *= self.nominal

                # store the value
                if up is None:
                    up = v
                else:
                    down = v
                    break

            # down defaults to up
            if down is None:
                down = up

            _uncertainties[str(name)] = (up, down)

        return _uncertainties

    def get_uncertainty(self, name=DEFAULT, direction=None, **kwargs):
        """ get_uncertainty(name=DEFAULT, direction=None, default=None)
        Returns the *absolute* up and down variaton in a 2-tuple for an uncertainty *name*. When
        *direction* is set, the particular value is returned instead of a 2-tuple. In case no
        uncertainty was found and *default* is given, that value is returned.
        """
        if direction not in (None, self.UP, self.DOWN):
            raise ValueError("unknown direction: %s" % direction)

        unc = self.uncertainties.get(name, *kwargs.values())

        if direction is None:
            return unc
        else:
            return unc[0 if direction == self.UP else 1]

    def u(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`get_uncertainty`.
        """
        return self.get_uncertainty(*args, **kwargs)

    def set_uncertainty(self, name, value):
        """
        Sets the uncertainty *value* for an uncertainty *name*. *value* should have one of the
        formats as described in :py:meth:`uncertainties`.
        """
        uncertainties = self.__class__.uncertainties.fparse(self, {name: value})
        self._uncertainties.update(uncertainties)

    def str(self, format="%.2f", **kwargs):
        """
        Returns a readable string representiation of the number. *format* is used to format
        non-NumPy nominal and uncertainty values. It can be a string such as ``"%d"`` or a function
        that is passed the value to format. In case of NumPy objects, all *kwargs* are passed to
        `numpy.array2string
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array2string.html>`_.
        """
        if not self.is_numpy:
            if callable(format):
                fmt = format
            else:
                fmt = lambda x: format % x

            # nominal text
            nom_text = fmt(self.nominal)

            # uncertainty text
            unc_text = ""
            if len(self.uncertainties) == 0:
                unc_text += ", no uncertainties"
            elif len(self.uncertainties) == 1 and self.uncertainties.keys()[0] == self.DEFAULT:
                up, down = self.get_uncertainty()
                unc_text = " (+%s, -%s)" % (fmt(up), fmt(down))
            else:
                for name, (up, down) in self.uncertainties.items():
                    unc_text += ", %s: (+%s, -%s)" % (name, fmt(up), fmt(down))
        else:
            # nominal text
            nom_text = np.array2string(self.nominal, **kwargs)

            # uncertainty text
            unc_text = ""
            if len(self.uncertainties) == 0:
                unc_text += ", no uncertainties"
            elif len(self.uncertainties) == 1 and self.uncertainties.keys()[0] == self.DEFAULT:
                up, down = self.get_uncertainty()
                unc_text += "\n+ %s" % np.array2string(up, **kwargs)
                unc_text += "\n- %s" % np.array2string(down, **kwargs)
            else:
                for name, (up, down) in self.uncertainties.items():
                    unc_text += "\n+ %s %s" % (name, np.array2string(up, **kwargs))
                    unc_text += "\n- %s %s" % (name, np.array2string(down, **kwargs))

        return nom_text + unc_text

    def repr(self):
        """
        Returns the unique string representation of the number.
        """
        if not self.is_numpy:
            nom_text = str(self.nominal)
        else:
            nom_text = np.array2string(self.nominal)

        tpl = (self.__class__.__name__, hex(id(self)), nom_text, len(self.uncertainties))
        return "<%s at %s, nominal: %s, uncertaintie(s): %i>" % tpl

    def copy(self, nominal=None, uncertainties=None):
        """
        Returns a deep copy of the number instance. When *nominal* or *uncertainties* are set, they
        overwrite the fields of the copied instance.
        """
        if nominal is None:
            nominal = self.nominal
        if uncertainties is None:
            uncertainties = self.uncertainties
        return self.__class__(nominal, uncertainties=uncertainties)

    def get(self, direction=NOMINAL, names=ALL, diff=False, factor=False):
        """ get(direction=NOMINAL, names=ALL, diff=False, factor=False)
        Returns different representations of the contained value(s). *direction* should be any of
        *NOMINAL*, *UP* or *DOWN*. When not *NOMINAL*, *names* decides which uncertainties to take
        into account for the combination. When *diff* is *True*, only the unsigned, combined
        uncertainty is returned. When *False*, the nominal value plus or minus the uncertainty is
        returned. When *factor* is *True*, the ratio w.r.t. the nominal value is returned.
        """
        if direction == self.NOMINAL:
            value = self.nominal

        elif direction in (self.UP, self.DOWN):
            # find uncertainties to take into account
            if names == self.ALL:
                names = self.uncertainties.keys()
            else:
                names = make_list(names)
                if any(name not in self.uncertainties for name in names):
                    unknown = list(set(names) - set(self.uncertainties.keys()))
                    raise ValueError("unknown uncertainty name(s): %s" % names)

            # calculate the combined uncertainty without correlation
            idx = int(direction == self.DOWN)
            uncs = [self.uncertainties[name][idx] for name in names]
            unc = sum(u ** 2. for u in uncs) ** 0.5

            # determine the output value
            if diff:
                value = unc
            elif direction == self.UP:
                value = self.nominal + unc
            else:
                value = self.nominal - unc

        else:
            raise ValueError("unknown direction: %s" % direction)

        return value if not factor else value / self.nominal

    @property
    def is_numpy(self):
        return is_numpy(self.nominal)

    @property
    def shape(self):
        return None if not self.is_numpy else self.nominal.shape

    def add(self, *args, **kwargs):
        """ add(other, rho=0, inplace=True)
        Adds an *other* number instance. The correlation coefficient *rho* can be configured per
        uncertainty when passed as a dict. When *inplace* is *False*, a new instance is returned.
        """
        return self._apply(operator.add, *args, **kwargs)

    def sub(self, *args, **kwargs):
        """ sub(other, rho=0, inplace=True)
        Subtracts an *other* number instance. The correlation coefficient *rho* can be configured
        per uncertainty when passed as a dict. When *inplace* is *False*, a new instance is
        returned.
        """
        return self._apply(operator.sub, *args, **kwargs)

    def mul(self, *args, **kwargs):
        """ mul(other, rho=0, inplace=True)
        Multiplies by an *other* number instance. The correlation coefficient *rho* can be
        configured per uncertainty when passed as a dict. When *inplace* is *False*, a new instance
        is returned.
        """
        return self._apply(operator.mul, *args, **kwargs)

    def div(self, *args, **kwargs):
        """ div(other, rho=0, inplace=True)
        Divides by an *other* number instance. The correlation coefficient *rho* can be configured
        per uncertainty when passed as a dict. When *inplace* is *False*, a new instance is
        returned.
        """
        return self._apply(operator.truediv, *args, **kwargs)

    def pow(self, *args, **kwargs):
        """ pow(other, rho=0, inplace=True)
        Raises by the power of an *other* number instance. The correlation coefficient *rho* can be
        configured per uncertainty when passed as a dict. When *inplace* is *False*, a new instance
        is returned.
        """
        return self._apply(operator.pow, *args, **kwargs)

    def _apply(self, op, other, rho=0., inplace=True):
        num = self if inplace else self.copy()
        other = ensure_number(other)

        # calculate the nominal value
        nom = op(num.nominal, other.nominal)

        # propagate uncertainties
        uncs = {}
        dflt = (0., 0.)
        for name in set(num.uncertainties.keys()) | set(other.uncertainties.keys()):
            _rho = rho if not isinstance(rho, dict) else rho.get(name, 0.)

            num_unc = num.get_uncertainty(name, default=dflt)
            other_unc = other.get_uncertainty(name, default=dflt)

            uncs[name] = tuple(combine_uncertainties(op, num_unc[i], other_unc[i],
                    nom1=num.nominal, nom2=other.nominal, rho=_rho) for i in range(2))

        # store values
        num.nominal = nom
        num.uncertainties = uncs

        return num

    def __call__(self, *args, **kwargs):
        # shorthand for get
        return self.get(*args, **kwargs)

    def __float__(self):
        # extract nominal value
        return self.nominal

    def __str__(self):
        # forward to default str
        return self.str()

    def __repr__(self):
        # forward to default repr
        return self.repr()

    def __contains__(self, name):
        # check whether name is an unceratinty
        return name in self.uncertainties

    def __nonzero__(self):
        # forward to self.nominal
        return self.nominal.__nonzero__()

    def __eq__(self, other):
        # compare nominal values
        if not self.is_numpy:
            return self.nominal == ensure_nominal(other)
        else:
            # element-wise
            return np.equal(self.nominal, ensure_nominal(other))

    def __ne__(self, other):
        # opposite of __eq__
        return not self.__eq__(other)

    def __lt__(self, other):
        # compare nominal values
        # numpy: element-wise
        return self.nominal < ensure_nominal(other)

    def __le__(self, other):
        # compare nominal values
        # numpy: element-wise
        return self.nominal <= ensure_nominal(other)

    def __gt__(self, other):
        # compare nominal values
        # numpy: element-wise
        return self.nominal > ensure_nominal(other)

    def __ge__(self, other):
        # compare nominal values
        # numpy: element-wise
        return self.nominal >= ensure_nominal(other)

    def __pos__(self):
        # simply copy
        return self.copy()

    def __neg__(self):
        # simply copy and flip the nominal value
        return self.copy(nominal=-self.nominal)

    def __abs__(self):
        # make nominal absolute
        if not is_numpy:
            nominal = abs(self.nominal)
        else:
            nominal = np.abs(self.nominal)
        return self.copy(nominal=nominal)

    def __add__(self, other):
        return self.add(other, inplace=False)

    def __radd__(self, other):
        return ensure_number(other).add(self, inplace=False)

    def __iadd__(self, other):
        return self.add(other, inplace=True)

    def __sub__(self, other):
        return self.sub(other, inplace=False)

    def __rsub__(self, other):
        return ensure_number(other).sub(self, inplace=False)

    def __isub__(self, other):
        return self.sub(other, inplace=True)

    def __mul__(self, other):
        return self.mul(other, inplace=False)

    def __rmul__(self, other):
        return ensure_number(other).mul(self, inplace=False)

    def __imul__(self, other):
        return self.mul(other, inplace=True)

    def __div__(self, other):
        return self.div(other, inplace=False)

    def __rdiv__(self, other):
        return ensure_number(other).div(self, inplace=False)

    def __idiv__(self, other):
        return self.div(other, inplace=True)

    def __truediv__(self, other):
        return self.div(other, inplace=False)

    def __rtruediv__(self, other):
        return ensure_number(other).div(self, inplace=False)

    def __itruediv__(self, other):
        return self.div(other, inplace=True)

    def __pow__(self, other):
        return self.pow(other, inplace=False)

    def __rpow__(self, other):
        return ensure_number(other).pow(self, inplace=False)

    def __ipow__(self, other):
        return self.pow(other, inplace=True)


class Operation(object):
    """
    Wrapper around a function and its derivative.

    .. py:attribute:: function
       type: function

       The wrapped function.

    .. py:attribute:: derivative
       type: function

       The wrapped derivative.

    .. py:attribute:: name
       type: string
       read-only

       The name of the operation.
    """

    def __init__(self, function, derivative=None, name=None):
        super(Operation, self).__init__()

        self.function = function
        self.derivative = derivative
        self._name = name or function.__name__

        # decorator for setting the derivative
        def derive(derivative):
            self.derivative = derivative
            return self
        self.derive = derive

    @property
    def name(self):
        return self._name

    def __repr__(self):
        tpl = (self.__class__.__name__, self.name, hex(id(self)))
        return "<%s '%s' at %s>" % tpl


class OpsMeta(type):

    def __contains__(cls, name):
        return name in cls._instances


@add_metaclass(OpsMeta)
class ops(object):
    """
    Number-aware replacement for the global math (or numpy) module. The purpose of the class is to
    provide operations (e.g. `pow`, `cos`, `sin`, etc.) that automatically propagate the
    uncertainties of a :py:class:`Number` instance through the derivative of the operation. Example:

    .. code-block:: python

        num = ops.pow(Number(5, 1), 2)
        print(num) # -> 25.00 (+10.00, -10.00)
    """

    _instances = {}

    @classmethod
    def register(cls, function=None, name=None):
        """
        Registers a new math function *function* with *name* and returns an :py:class:`Operation`
        instance. A math function expects a :py:class:`Number` as its first argument, followed by
        optional (keyword) arguments. When *name* is *None*, the name of the *function* is used. The
        returned object can be used to set the derivative (similar to *property*). Example:

        .. code-block:: python

            @ops.register
            def my_op(x):
                return x * 2 + 1

            @my_op.derive
            def my_op(x):
                return 2

            num = Number(5, 2)
            print(num) # -> 5.00 (+2.00, -2.00)

            num = ops.my_op(num)
            print(num) # -> 11.00 (+4.00, -4.00)

        Please note that there is no need to register *simple* functions as in the particular
        example above as most of them are just composite operations whose derivatives are already
        known.
        """
        def register(function):
            op = Operation(function, name=name)

            @functools.wraps(function)
            def wrapper(num, *args, **kwargs):
                if op.derivative is None:
                    raise Exception("cannot run operation '%s', no derivative registered" % op.name)

                # ensure we deal with a number instance
                num = ensure_number(num)

                # apply to the nominal value
                nominal = op.function(num.nominal, *args, **kwargs)

                # apply to all uncertainties via
                # unc_f = derivative_f(x) * unc_x
                x = abs(op.derivative(num.nominal, *args, **kwargs))
                uncertainties = {}
                for name in num.uncertainties.keys():
                    up, down = num.get_uncertainty(name)
                    uncertainties[name] = (x * up, x * down)

                # create and return the new number
                return num.__class__(nominal, uncertainties)

            # actual registration
            cls._instances[op.name] = op
            setattr(cls, op.name, staticmethod(wrapper))

            return op

        if function is None:
            return register
        else:
            return register(function)

    @classmethod
    def get_operation(cls, name):
        """
        Returns an operation that was previously registered with *name*.
        """
        return cls._instances[name]

    @classmethod
    def op(cls, *args, **kwargs):
        """
        Shorthand for :py:meth:`get_operation`.
        """
        return cls.get_operation(*args, **kwargs)


#
# Pre-registered operations.
#

@ops.register
def add(x, n):
    """ add(x, n)
    Addition function.
    """
    return x + n

@add.derive
def add(x, n):
    return 1.


@ops.register
def sub(x, n):
    """ sub(x, n)
    Subtraction function.
    """
    return x - n

@sub.derive
def sub(x, n):
    return 1.


@ops.register
def mul(x, n):
    """ mul(x, n)
    Multiplication function.
    """
    return x * n

@mul.derive
def mul(x, n):
    return n


@ops.register
def div(x, n):
    """ div(x, n)
    Division function.
    """
    return x / n

@div.derive
def div(x, n):
    return 1. / n


@ops.register
def pow(x, n):
    """ pow(x, n)
    Power function.
    """
    return x ** n

@pow.derive
def pow(x, n):
    return n * x ** (n - 1.)


@ops.register
def exp(x):
    """ exp(x)
    Exponential function.
    """
    return infer_math(x).exp(x)

exp.derivative = exp.function


@ops.register
def log(x, base=None):
    """ log(x, base=e)
    Logarithmic function.
    """
    _math = infer_math(x)
    if base is None:
        return _math.log(x)
    elif _math == math:
        return _math.log(x, base)
    else:
        # numpy has no option to set a base
        return _math.log(x) / _math.log(base)

@log.derive
def log(x, base=None):
    if base is None:
        return 1. / x
    else:
        return 1. / (x * infer_math(x).log(base))


@ops.register
def sqrt(x):
    """ sqrt(x)
    Square root function.
    """
    return infer_math(x).sqrt(x)

@sqrt.derive
def sqrt(x):
    return 1. / (2 * infer_math(x).sqrt(x))


@ops.register
def sin(x):
    """ sin(x)
    Trigonometric sin function.
    """
    return infer_math(x).sin(x)

@sin.derive
def sin(x):
    return infer_math(x).cos(x)


@ops.register
def cos(x):
    """ cos(x)
    Trigonometric cos function.
    """
    return infer_math(x).cos(x)

@cos.derive
def cos(x):
    return -infer_math(x).sin(x)


@ops.register
def tan(x):
    """ tan(x)
    Trigonometric tan function.
    """
    return infer_math(x).tan(x)

@tan.derive
def tan(x):
    return 1. / infer_math(x).cos(x) ** 2


@ops.register
def asin(x):
    """ asin(x)
    Trigonometric arc sin function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.asin(x)
    else:
        return _math.arcsin(x)

@asin.derive
def asin(x):
    return 1. / infer_math(x).sqrt(1 - x ** 2.)


@ops.register
def acos(x):
    """ acos(x)
    Trigonometric arc cos function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.acos(x)
    else:
        return _math.arccos(x)

@acos.derive
def acos(x):
    return -1. / infer_math(x).sqrt(1 - x ** 2.)


@ops.register
def atan(x):
    """ tan(x)
    Trigonometric arc tan function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.atan(x)
    else:
        return _math.arctan(x)

@atan.derive
def atan(x):
    return 1. / (1 + x ** 2.)


@ops.register
def sinh(x):
    """ sinh(x)
    Hyperbolic sin function.
    """
    return infer_math(x).sinh(x)

@sinh.derive
def sinh(x):
    return infer_math(x).cosh(x)


@ops.register
def cosh(x):
    """ cosh(x)
    Hyperbolic cos function.
    """
    return infer_math(x).cosh(x)

@cosh.derive
def cosh(x):
    return infer_math(x).sinh(x)


@ops.register
def tanh(x):
    """ tanh(x)
    Hyperbolic tan function.
    """
    return infer_math(x).tanh(x)

@tanh.derive
def tanh(x):
    return 1. / infer_math(x).cosh(x) ** 2


@ops.register
def asinh(x):
    """ asinh(x)
    Hyperbolic arc sin function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.asinh(x)
    else:
        return _math.arcsinh(x)


@ops.register
def acosh(x):
    """ acosh(x)
    Hyperbolic arc cos function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.acosh(x)
    else:
        return _math.arccosh(x)

asinh.derivative = acosh.function
acosh.derivative = asinh.function


@ops.register
def atanh(x):
    """ atanh(x)
    Hyperbolic arc tan function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.atanh(x)
    else:
        return _math.arctanh(x)

@atanh.derive
def atanh(x):
    return 1. / (1. - x ** 2.)



#
# helper functions
#

_op_map = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
           "**": operator.pow}
_op_map_reverse = dict(zip(_op_map.values(), _op_map.keys()))

def combine_uncertainties(op, unc1, unc2, nom1=None, nom2=None, rho=0.):
    """
    Combines two uncertainties *unc1* and *unc2* according to an operator *op* which must be either
    ``"+"``, ``"-"``, ``"*"``, ``"/"``, or ``"**"``. The three latter operators require that you
    also pass the nominal values *nom1* and *nom2*, respectively. The correlation can be configured
    via *rho*.
    """
    # operator valid?
    if op in _op_map:
        f = _op_map[op]
    elif op in _op_map_reverse:
        f = op
        op = _op_map_reverse[op]
    else:
        raise ValueError("unknown operator: %s" % op)

    # prepare values for combination, depends on operator
    if op in ("*", "/", "**"):
        if nom1 is None or nom2 is None:
            raise ValueError("operator '%s' requires nominal values" % op)
        # numpy-safe conversion to float
        nom1 *= 1.
        nom2 *= 1.
        # convert uncertainties to relative values
        unc1 /= nom1
        unc2 /= nom2
        # determine
        nom = abs(f(nom1, nom2))
    else:
        nom = 1.

    # combined formula
    if op == "**":
        return nom * abs(nom2) * (unc1 ** 2 + (math.log(nom1) * unc2) ** 2 \
                                  + 2 * rho * math.log(nom1) * unc1 * unc2) ** 0.5
    else:
        # flip rho for sub and div
        if op in ("-", "/"):
            rho = -rho
        return nom * (unc1 ** 2. + unc2 ** 2. + 2. * rho * unc1 * unc2) ** 0.5


def ensure_number(num, *args, **kwargs):
    """
    Returns *num* again if it is an instance of :py:class:`Number`, or uses all passed arguments to
    create one and returns it.
    """
    return num if isinstance(num, Number) else Number(num, *args, **kwargs)


def ensure_nominal(nominal):
    """
    Returns *nominal* again if it is not an instance of :py:class:`Number`, or returns its nominal
    value.
    """
    return nominal if not isinstance(nominal, Number) else nominal.nominal


def is_numpy(x):
    """
    Returns *True* when numpy is available on your system and *x* is a numpy type.
    """
    return HAS_NUMPY and type(x).__module__ == np.__name__


def infer_math(x):
    """
    Returns the numpy module when :py:func:`is_numpy` for *x* is *True*, and the math module
    otherwise.
    """
    return np if is_numpy(x) else math


def make_list(obj, cast=True):
    """
    Converts an object *obj* to a list and returns it. Objects of types *tuple* and *set* are
    converted if *cast* is *True*. Otherwise, and for all other types, *obj* is put in a new list.
    """
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, types.GeneratorType):
        return list(obj)
    if isinstance(obj, (tuple, set)) and cast:
        return list(obj)
    else:
        return [obj]
