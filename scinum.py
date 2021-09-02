# coding: utf-8

"""
Scientific numbers with multiple uncertainties and correlation-aware, gaussian propagation and numpy
support.
"""


__author__ = "Marcel Rieger"
__email__ = "python-scinum@googlegroups.com"
__copyright__ = "Copyright 2017-2021, Marcel Rieger"
__credits__ = ["Marcel Rieger"]
__contact__ = "https://github.com/riga/scinum"
__license__ = "BSD-3-Clause"
__status__ = "Development"
__version__ = "1.2.0"
__all__ = [
    "Number", "Correlation", "DeferredResult", "Operation",
    "ops", "style_dict",
    "REL", "ABS", "NOMINAL", "UP", "DOWN",
]


import sys
import math
import functools
import operator
import types
import decimal
from collections import defaultdict

# optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import uncertainties as _uncs
    HAS_UNCERTAINTIES = True
except ImportError:
    _uncs = None
    HAS_UNCERTAINTIES = False


# version related adjustments
string_types = (str,)
if sys.version_info.major < 3:
    string_types += (basestring,)  # noqa

integer_types = (int,)
if sys.version_info.major < 3:
    integer_types += (long,)  # noqa


# metaclass decorator from six package, credits to Benjamin Peterson
def with_metaclass(meta, *bases):
    class metaclass(type):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

        @classmethod
        def __prepare__(cls, name, this_bases):
            return meta.__prepare__(name, bases)

    return type.__new__(metaclass, "temporary_class", (), {})


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
                    raise TypeError("not a string: {}".format(foo))
                return foo

        myInstance = MyClass()
        myInstance.foo = 123    # -> TypeError
        myInstance.foo = "bar"  # -> ok
        print(myInstance.foo)   # -> prints "bar"

    In the exampe above, set/get calls target the instance member ``_foo``, i.e. "_<function_name>".
    The member name can be configured by setting *name*. If *setter* (*deleter*) is *True* (the
    default), a setter (deleter) method is booked as well. Prior to updating the member when the
    setter is called, *fparse* is invoked which may implement sanity checks.
    """

    def __init__(self, fparse=None, setter=True, deleter=True, name=None):
        self._args = (setter, deleter, name)

        # only register the property if fparse is set
        if fparse is not None:
            self.fparse = fparse

            # build the default name
            if name is None:
                name = fparse.__name__
            self.__name__ = name

            # the name of the wrapped member
            m_name = "_" + name

            # call the super constructor with generated methods
            property.__init__(self,
                functools.wraps(fparse)(self._fget(m_name)),
                self._fset(m_name) if setter else None,
                self._fdel(m_name) if deleter else None
            )

    def __call__(self, fparse):
        return self.__class__(fparse, *self._args)

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

        from scinum import Number, REL, ABS, UP, DOWN

        num = Number(2.5, {
            "sourceA": 0.5,                  # absolute 0.5, both up and down
            "sourceB": (1.0, 1.5),           # absolute 1.0 up, 1.5 down
            "sourceC": (REL, 0.1),           # relative 10%, both up and down
            "sourceD": (REL, 0.1, 0.2),      # relative 10% up, 20% down
            "sourceE": (1.0, REL, 0.2),      # absolute 1.0 up, relative 20% down
            "sourceF": (REL, 0.3, ABS, 0.3)  # relative 30% up, absolute 0.3 down
        })

        # get the nominal value via direct access
        num.nominal # => 2.5

        # get the nominal value via __call__() (same as get())
        num()                     # => 2.5
        num(direction="nominal")  # => 2.5

        # get uncertainties
        num.get_uncertainty("sourceA")  # => (0.5, 0.5)
        num.get_uncertainty("sourceB")  # => (1.0, 1.5)
        num.get_uncertainty("sourceC")  # => (0.25, 0.25)
        num.get_uncertainty("sourceD")  # => (0.25, 0.5)
        num.get_uncertainty("sourceE")  # => (1.0, 0.5)
        num.get_uncertainty("sourceF")  # => (0.75, 0.3)

        # get shifted values via __call__() (same as get())
        num(UP, "sourceA")               # => 3.0
        num(DOWN, "sourceB")             # => 1.0
        num(UP, ("sourceC", "sourceD"))  # => 2.854...
        num(UP)                          # => 4.214... (all uncertainties)

        # get only the uncertainty (unsigned)
        num(DOWN, ("sourceE", "sourceF"), diff=True)  # => 0.583...

        # get the uncertainty factor (unsigned)
        num(DOWN, ("sourceE", "sourceF"), factor=True)  # => 1.233...

        # combined
        num(DOWN, ("sourceE", "sourceF"), diff=True, factor=True)  # => 0.233...

    When *uncertainties* is not a dictionary, it is interpreted as the *default* uncertainty, named
    ``Number.DEFAULT``.

    This class redefines most of Python's magic functions to allow transparent use in standard
    operations like ``+``, ``*``, etc. Gaussian uncertainty propagation is applied automatically.
    When operations connect two number instances, their uncertainties are combined assuming there is
    no correlation. For correlation-aware operations, please refer to methods such as :py:meth:`add`
    or :py:meth:`mul` below. Examples:

    .. code-block:: python

        num = Number(5, 1)
        print(num + 2)  # -> '7.0 +- 1.0'
        print(num * 3)  # -> '15.0 +- 3.0'

        num2 = Number(2.5, 1.5)
        print(num + num2)  # -> '7.5 +- 1.80277563773'
        print(num * num2)  # -> '12.5 +- 7.90569415042'

        num.add(num2, rho=1)
        print(num)  # -> '7.5 +- 2.5'

    See :py:meth:`str` for information on string formatting.

    .. py:classattribute:: default_format

       The default format string (``"%s"``) that is used in :py:meth:`str()` when no format string
       was passed.

    .. py:classattribute:: DEFAULT

       Constant that denotes the default uncertainty (``"default"``).

    .. py:classattribute:: ALL

       Constant that denotes all uncertainties (``"all"``).

    .. py:classattribute:: REL

       Constant that denotes relative errors (``"rel"``).

    .. py:classattribute:: ABS

       Constant that denotes absolute errors (``"abs"``).

    .. py:classattribute:: NOMINAL

       Constant that denotes the nominal value (``"nominal"``).

    .. py:classattribute:: UP

       Constant that denotes the up direction (``"up"``).

    .. py:classattribute:: DOWN

       Constant that denotes the down direction (``"down"``).

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
    DEFAULT = "default"
    ALL = "all"

    # uncertainty types
    REL = "rel"
    ABS = "abs"

    # uncertainty directions
    NOMINAL = "nominal"
    UP = "up"
    DOWN = "down"

    default_format = "%s"

    def __init__(self, nominal=0.0, uncertainties=None, default_format=None):
        super(Number, self).__init__()

        # wrapped values
        self._nominal = None
        self._uncertainties = {}

        # numpy settings
        self.dtype = np.float32 if HAS_NUMPY else None

        # prepare conversion from uncertainties.ufloat
        if is_ufloat(nominal):
            # uncertainties must not be set
            if uncertainties:
                raise ValueError("uncertainties must not be set when converting a ufloat")
            # extract nominal value and uncertainties
            nominal, uncertainties = parse_ufloat(nominal)

        # set initial values
        self.nominal = nominal
        if uncertainties is not None:
            self.uncertainties = uncertainties

        self.default_format = default_format

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
                    raise ValueError("shape not matching uncertainty shape: {}".format(
                        nominal.shape))
            nominal = nominal.astype(self.dtype)
        else:
            raise TypeError("invalid nominal value: {}".format(nominal))

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
            if not isinstance(name, string_types):
                raise TypeError("invalid uncertainty name: {}".format(name))

            # parse the value type
            if isinstance(val, (int, float)) or is_numpy(val):
                val = (val, val)
            elif isinstance(val, list):
                val = tuple(val)
            elif not isinstance(val, tuple):
                raise TypeError("invalid uncertainty type: {}".format(val))

            # parse the value itself
            utype, up, down = self.ABS, None, None
            for v in val:
                # check if the uncertainty type is changed
                if isinstance(v, string_types):
                    if v not in (self.ABS, self.REL):
                        raise ValueError("unknown uncertainty type: {}".format(v))
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
                        raise ValueError("shape not matching nominal shape: {}".format(v.shape))
                    v = v.astype(self.dtype)
                else:
                    raise TypeError("invalid uncertainty value: {}".format(v))

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

    def get_uncertainty(self, name=DEFAULT, direction=None, default=None):
        """
        Returns the *absolute* up and down variaton in a 2-tuple for an uncertainty *name*. When
        *direction* is set, the particular value is returned instead of a 2-tuple. In case no
        uncertainty was found and *default* is not *None*, that value is returned.
        """
        if direction not in (None, self.UP, self.DOWN):
            raise ValueError("unknown direction: {}".format(direction))

        if name not in self.uncertainties and default is not None:
            return default

        unc = self.uncertainties[name]

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

    def clear(self, nominal=None, uncertainties=None):
        """
        Removes all uncertainties and sets the nominal value to zero (float). When *nominal* and
        *uncertainties* are given, these new values are set on this instance.
        """
        self.uncertainties = {}
        self.nominal = 0.

        if nominal is not None:
            self.nominal = nominal
        if uncertainties is not None:
            self.uncertainties = uncertainties

    def str(self, format=None, unit=None, scientific=False, si=False, labels=True, style="plain",
            styles=None, force_asymmetric=False, **kwargs):
        r"""
        Returns a readable string representiation of the number. *format* is used to format
        non-NumPy nominal and uncertainty values. It can be a string such as ``"%d"``, a function
        that is called with the value to format, or a rounding function as accepted by
        :py:func:`round_value`. When *None* (the default), :py:attr:`default_format` is used. All
        keyword arguments except wildcard *kwargs* are only used to format non-NumPy values. In case
        of NumPy objects, *kwargs* are passed to `numpy.array2string
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array2string.html>`_.

        When *unit* is set, it is appended to the end of the string. When *scientific* is *True*,
        all values are represented by their scientific notation. When *scientific* is *False* and
        *si* is *True*, the appropriate SI prefix is used. *labels* controls whether uncertainty
        labels are shown in the string. When *True*, uncertainty names are used, but it can also
        be a list of labels whose order should match the uncertainty dict traversal order. *style*
        can be ``"plain"``, ``"latex"``, or ``"root"``. *styles* can be a dict with fields
        ``"space"``, ``"label"``, ``"unit"``, ``"sym"``, ``"asym"``, ``"sci"`` to customize every
        aspect of the format style on top of :py:attr:`style_dict`. Unless *force_asymmetric* is
        *True*, an uncertainty is quoted symmetric if it yields identical values in both directions.

        Examples:

        .. code-block:: python

            n = Number(17.321, {"a": 1.158, "b": 0.453})
            n.str()               # -> '17.321 +- 1.158 (a) +- 0.453 (b)'
            n.str("%.1f")         # -> '17.3 +- 1.2 (a) +- 0.5 (b)'
            n.str("publication")  # -> '17.32 +- 1.16 (a) +- 0.45 (b)'
            n.str("pdg")          # -> '17.3 +- 1.2 (a) +- 0.5 (b)'

            n = Number(8848, 10)
            n.str(unit="m")                           # -> "8848.0 +- 10.0 m"
            n.str(unit="m", force_asymmetric=True)    # -> "8848.0 +10.0-10.0 m"
            n.str(unit="m", scientific=True)          # -> "8.848 +- 0.01 x 1E3 m"
            n.str("%.2f", unit="m", scientific=True)  # -> "8.85 +- 0.01 x 1E3 m"
            n.str(unit="m", si=True)                  # -> "8.848 +- 0.01 km"
            n.str("%.2f", unit="m", si=True)          # -> "8.85 +- 0.01 km"
            n.str(unit="m", style="latex")            # -> "8848.0 \pm 10.0\,m"
            n.str(unit="m", style="latex", si=True)   # -> "8.848 \pm 0.01\,km"
            n.str(unit="m", style="root")             # -> "8848.0 #pm 10.0 m"
            n.str(unit="m", style="root", si=True)    # -> "8.848 #pm 0.01 km"
        """
        if format is None:
            format = self.default_format or self.__class__.default_format

        if not self.is_numpy:
            # check style
            style = style.lower()
            if style not in style_dict.keys():
                raise ValueError("unknown style '{}'".format(style))
            d = style_dict[style]

            # extend by custom styles
            if styles:
                d.update(styles)

            # scientific or SI notation?
            prefix = ""
            transform = lambda x: x
            if scientific or si:
                if scientific:
                    mag = 0 if self.nominal == 0 else int(math.floor(math.log10(abs(self.nominal))))
                else:
                    prefix, mag = infer_si_prefix(self.nominal)
                transform = lambda x: x * 10.**(-mag)

            # gather and transform values
            nominal = transform(self.nominal)
            names, ups, downs = [], [], []
            for name, (up, down) in self.uncertainties.items():
                names.append(name)
                ups.append(transform(up))
                downs.append(transform(down))

            # special formats implemented by round_value
            if format in ("pub", "publication", "pdg", "one", "onedigit"):
                # complain when no uncertainties exist
                if not ups:
                    raise Exception("cannot apply format '{}' when no uncertainties exist".format(
                        format))
                nominal, (ups, downs), _mag = round_value(self.nominal, ups, downs, method=format)
                fmt = lambda x: match_precision(float(x) * 10.**_mag, 10.**_mag)

            # string formatting
            elif not callable(format):
                fmt = lambda x: format % x

            # helper to build the ending consisting of scientific notation or SI prefix, and unit
            def ending():
                e = ""
                if scientific and mag:
                    e += d["space"] + d["sci"].format(mag=mag)
                _unit = (prefix or "") + (unit or "")
                if _unit:
                    e += d["unit"].format(unit=_unit)
                return e

            # start building the text
            text = fmt(nominal, **kwargs)

            # no uncertainties
            if len(names) == 0:
                text += ending()
                if style == "plain" and labels:
                    text += d["space"] + d["label"].format(label="no uncertainties")

            # one ore more uncertainties
            else:
                # special case: only the default uncertainty
                if len(names) == 1 and names[0] == self.DEFAULT:
                    labels = False

                for i, (name, up, down) in enumerate(zip(names, ups, downs)):
                    up = str(fmt(up))
                    down = str(fmt(down))

                    if up == down and not force_asymmetric:
                        text += d["space"] + d["sym"].format(unc=up)
                    else:
                        text += d["space"] + d["asym"].format(up=up, down=down)

                    if labels:
                        label = labels[i] if isinstance(labels, (list, tuple)) else name
                        text += d["space"] + d["label"].format(label=label)

                text += ending()

            return text

        else:
            # we are dealing with a numpy array here
            # start with nominal text
            text = np.array2string(self.nominal, **kwargs)

            # uncertainty text
            uncs = self.uncertainties
            if len(uncs) == 0:
                text += " (no uncertainties)"
            elif len(uncs) == 1 and list(uncs.keys())[0] == self.DEFAULT:
                up, down = self.get_uncertainty()
                text += "\n+ {}".format(np.array2string(up, **kwargs))
                text += "\n- {}".format(np.array2string(down, **kwargs))
            else:
                for name, (up, down) in uncs.items():
                    text += "\n+ {} {}".format(name, np.array2string(up, **kwargs))
                    text += "\n- {} {}".format(name, np.array2string(down, **kwargs))

            return text

    def repr(self, *args, **kwargs):
        """
        Returns the unique string representation of the number.
        """
        if not self.is_numpy:
            text = "'" + self.str(*args, **kwargs) + "'"
        else:
            text = "numpy array, shape {}, {} uncertainties".format(self.shape,
                len(self.uncertainties))

        return "<{} at {}, {}>".format(self.__class__.__name__, hex(id(self)), text)

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
                    raise ValueError("unknown uncertainty name(s): {}".format(unknown))

            # calculate the combined uncertainty without correlation
            idx = int(direction == self.DOWN)
            uncs = [self.uncertainties[name][idx] for name in names]
            unc = sum(u**2. for u in uncs)**0.5

            # determine the output value
            if diff:
                value = unc
            elif direction == self.UP:
                value = self.nominal + unc
            else:
                value = self.nominal - unc

        else:
            raise ValueError("unknown direction: {}".format(direction))

        return value if not factor else value / self.nominal

    @property
    def is_numpy(self):
        return is_numpy(self.nominal)

    @property
    def shape(self):
        return None if not self.is_numpy else self.nominal.shape

    def add(self, *args, **kwargs):
        """ add(other, rho=1.0, inplace=True)
        Adds an *other* :py:class:`Number` or :py:class:`DeferredResult` instance, propagating all
        uncertainties. Uncertainties with the same name are combined with the correlation *rho*,
        which can either be a :py:class:`Correlation` instance, a dict with correlations defined per
        uncertainty, or a plain float. When *inplace* is *False*, a new instance is returned.
        """
        return self._apply(operator.add, *args, **kwargs)

    def sub(self, *args, **kwargs):
        """ sub(other, rho=1.0, inplace=True)
        Subtracts an *other* :py:class:`Number` or :py:class:`DeferredResult` instance, propagating
        all uncertainties. Uncertainties with the same name are combined with the correlation *rho*,
        which can either be a :py:class:`Correlation` instance, a dict with correlations defined per
        uncertainty, or a plain float. When *inplace* is *False*, a new instance is returned.
        """
        return self._apply(operator.sub, *args, **kwargs)

    def mul(self, *args, **kwargs):
        """ mul(other, rho=1.0, inplace=True)
        Multiplies by an *other* :py:class:`Number` or :py:class:`DeferredResult` instance,
        propagating all uncertainties. Uncertainties with the same name are combined with the
        correlation *rho*, which can either be a :py:class:`Correlation` instance, a dict with
        correlations defined per uncertainty, or a plain float. When *inplace* is *False*, a new
        instance is returned.

        Unlike the other operations, *other* can also be a :py:class:`Correlation` instance, in
        which case a :py:class:`DeferredResult` is returned to resolve the combination of
        uncertainties later on.
        """
        return self._apply(operator.mul, *args, **kwargs)

    def div(self, *args, **kwargs):
        """ div(other, rho=1.0, inplace=True)
        Divides by an *other* :py:class:`Number` or :py:class:`DeferredResult` instance, propagating
        all uncertainties. Uncertainties with the same name are combined with the correlation *rho*,
        which can either be a :py:class:`Correlation` instance, a dict with correlations defined per
        uncertainty, or a plain float. When *inplace* is *False*, a new instance is returned.
        """
        return self._apply(operator.truediv, *args, **kwargs)

    def pow(self, *args, **kwargs):
        """ pow(other, rho=1.0, inplace=True)
        Raises by the power of an *other* :py:class:`Number` or :py:class:`DeferredResult` instance,
        propagating all uncertainties. Uncertainties with the same name are combined with the
        correlation *rho*, which can either be a :py:class:`Correlation` instance, a dict with
        correlations defined per uncertainty, or a plain float. When *inplace* is *False*, a new
        instance is returned.
        """
        return self._apply(operator.pow, *args, **kwargs)

    def _apply(self, op, other, rho=1., inplace=True):
        # when other is a correlation object and op is mul, return a deferred result that is
        # resolved in the next operation
        if isinstance(other, Correlation):
            if op != operator.mul:
                raise ValueError("cannot apply correlation object {} via operator {}, only mul is "
                    "supported".format(other, op.__name__))
            return DeferredResult(self, other)

        # when other is a deferred result, use its number of correlation
        if isinstance(other, DeferredResult):
            rho = other.correlation
            other = other.number

        # prepare the number to update and the other number to apply
        num = self if inplace else self.copy()
        other = ensure_number(other)

        # calculate the nominal value
        nom = op(num.nominal, other.nominal)

        # propagate uncertainties
        uncs = {}
        default = (0., 0.)
        for name in set(num.uncertainties.keys()) | set(other.uncertainties.keys()):
            # get the correlation coefficient for this uncertainty
            if isinstance(rho, Correlation):
                _rho = rho.get(name, 1. if rho.default is None else rho.default)
            elif isinstance(rho, dict):
                _rho = rho.get(name, 1.)
            else:
                _rho = rho

            # get uncertainty components
            num_unc = num.get_uncertainty(name, default=default)
            other_unc = other.get_uncertainty(name, default=default)

            # combine them
            uncs[name] = tuple(combine_uncertainties(op, num_unc[i], other_unc[i],
                nom1=num.nominal, nom2=other.nominal, rho=_rho) for i in range(2))

        # store values
        num.clear(nom, uncs)

        return num

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # only direct calls of the ufunc are supported
        if method != "__call__":
            return NotImplemented

        # try to find the proper op for that ufunc
        op = ops.get_ufunc_operation(ufunc)
        if op is None:
            return NotImplemented

        # extract kwargs
        out = kwargs.pop("out", None)

        # apply the op
        result = op(*inputs, **kwargs)

        # insert in-place to an "out" object?
        if out is not None:
            out = out[0]
            out.clear(result.nominal, result.uncertainties)
            result = out

        return result

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

    def _repr_latex_(self):
        return self.repr() if self.is_numpy else "${}$".format(self.str(style="latex"))

    def __contains__(self, name):
        # check whether name is an uncertainty
        return name in self.uncertainties

    def __nonzero__(self):
        # forward to self.nominal
        return self.nominal.__nonzero__()

    def __eq__(self, other):
        # compare nominal values
        if isinstance(other, Number):
            other = ensure_nominal(other)

        if self.is_numpy and is_numpy(other):
            # element-wise
            try:
                return np.equal(self.nominal, other)
            except ValueError:
                return False
        elif self.is_numpy or is_numpy(other):
            return (self.nominal == other).all()
        else:
            return self.nominal == other

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
        if isinstance(other, DeferredResult):
            return other.number.add(self, rho=other.correlation, inplace=False)
        else:
            return ensure_number(other).add(self, inplace=False)

    def __iadd__(self, other):
        return self.add(other, inplace=True)

    def __sub__(self, other):
        return self.sub(other, inplace=False)

    def __rsub__(self, other):
        if isinstance(other, DeferredResult):
            return other.number.sub(self, rho=other.correlation, inplace=False)
        else:
            return ensure_number(other).sub(self, inplace=False)

    def __isub__(self, other):
        return self.sub(other, inplace=True)

    def __mul__(self, other):
        return self.mul(other, inplace=False)

    def __rmul__(self, other):
        if isinstance(other, Correlation):
            return self.mul(other, inplace=False)
        elif isinstance(other, DeferredResult):
            return other.number.mul(self, rho=other.correlation, inplace=False)
        else:
            return ensure_number(other).mul(self, inplace=False)

    def __imul__(self, other):
        return self.mul(other, inplace=True)

    def __div__(self, other):
        return self.div(other, inplace=False)

    def __rdiv__(self, other):
        if isinstance(other, DeferredResult):
            return other.number.rdiv(self, rho=other.correlation, inplace=False)
        else:
            return ensure_number(other).div(self, inplace=False)

    def __idiv__(self, other):
        return self.div(other, inplace=True)

    def __truediv__(self, other):
        return self.div(other, inplace=False)

    def __rtruediv__(self, other):
        if isinstance(other, DeferredResult):
            return other.number.div(self, rho=other.correlation, inplace=False)
        else:
            return ensure_number(other).div(self, inplace=False)

    def __itruediv__(self, other):
        return self.div(other, inplace=True)

    def __pow__(self, other):
        return self.pow(other, inplace=False)

    def __rpow__(self, other):
        if isinstance(other, DeferredResult):
            return other.number.rpow(self, rho=other.correlation, inplace=False)
        else:
            return ensure_number(other).pow(self, inplace=False)

    def __ipow__(self, other):
        return self.pow(other, inplace=True)


# module-wide shorthands for Number flags
REL = Number.REL
ABS = Number.ABS
NOMINAL = Number.NOMINAL
UP = Number.UP
DOWN = Number.DOWN


class Correlation(object):
    """ Correlation([default], **rhos)
    Container class describing correlations to be applied to equally named uncertainties when
    combining two :py:class:`Number` instances through an operator.

    A correlation object is therefore applied to a number by means of multiplication, resulting in a
    :py:class:`DeferredResult` object which is used subsequently by the actual combination operation
    with an other number. See :py:class:`DeferredResult` for more examples.
    """

    def __init__(self, *args, **rhos):
        super(Correlation, self).__init__()

        # at most one positional argument is accepted
        if len(args) >= 2:
            raise Exception("only one default value is accepted, got {} instead".format(args))
        # at least one argument at all is expected
        if not args and not rhos:
            raise Exception("at least one correlation coefficient or a default value is expected")

        # store attributes
        self.default = float(args[0]) if len(args) == 1 else None
        self.rhos = rhos

    def __repr__(self):
        parts = []
        if self.default is not None:
            parts.append(str(self.default))
        parts.extend("{}={}".format(*tpl) for tpl in self.rhos.items())
        return "<{} '{}' at {}>".format(self.__class__.__name__, ", ".join(parts), hex(id(self)))

    def get(self, name, default=None):
        """
        Returns a correlation coefficient rho named *name*. When no coefficient with that name
        exists and *default* is set, which itself defaults to :py:attr:`default`, this value is
        returned instead. Otherwise, a *KeyError* is raised.
        """
        if default is None:
            default = self.default

        return self.rhos[name] if default is None else self.rhos.get(name, default)


class DeferredResult(object):
    """
    Class that wraps a :py:class:`Number` instance *number* and a :py:class:`Correlation` instance
    *correlation* that is automatically produced as a result of a multiplication between the two.
    Internally, this is used for the deferred resolution of uncertainty correlations when combined
    with an other :py:class:`Number`. Example:

    .. code-block:: python

        n = Number(2, 5)

        n * Correlation(1) * n
        # -> '25.0 +- 20.0' (the default)

        n * Correlation(0) * n
        # -> '25.00 +- 14.14'

        # note the multiplication n * c, which creates the DeferredResult
        n**(n * c)
        # -> '3125.00 +- 11842.54'

    .. py:attribute:: number
       type: Number

       The wrapped number object.

    .. py:attribute:: correlation
       type: Correlation

       The wrapped correlation object.
    """

    def __init__(self, number, correlation):
        super(DeferredResult, self).__init__()

        # store attributes
        self.number = number
        self.correlation = correlation


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

    .. py:attribute:: ufuncs
       type: list
       read-only

       List of ufunc objects that this operation handles.
    """

    def __init__(self, function, derivative=None, name=None, ufuncs=None):
        super(Operation, self).__init__()

        self.function = function
        self.derivative = derivative
        self._name = name or function.__name__
        self._ufuncs = ufuncs or []

        # decorator for setting the derivative
        def derive(derivative):
            self.derivative = derivative
            return self
        self.derive = derive

    @property
    def name(self):
        return self._name

    @property
    def ufuncs(self):
        return self._ufuncs

    def __repr__(self):
        return "<{} '{}' at {}>".format(self.__class__.__name__, self.name, hex(id(self)))

    def __call__(self, num, *args, **kwargs):
        if self.derivative is None:
            raise Exception("cannot run operation '{}', no derivative registered".format(
                self.name))

        # ensure we deal with a number instance
        num = ensure_number(num)

        # apply to the nominal value
        nominal = self.function(num.nominal, *args, **kwargs)

        # apply to all uncertainties via
        # unc_f = derivative_f(x) * unc_x
        x = abs(self.derivative(num.nominal, *args, **kwargs))
        uncertainties = {}
        for name in num.uncertainties:
            up, down = num.get_uncertainty(name)
            uncertainties[name] = (x * up, x * down)

        # create and return the new number
        return num.__class__(nominal, uncertainties)


class OpsMeta(type):

    def __contains__(cls, name):
        return name in cls._instances


class ops(with_metaclass(OpsMeta, object)):
    """
    Number-aware replacement for the global math (or numpy) module. The purpose of the class is to
    provide operations (e.g. `pow`, `cos`, `sin`, etc.) that automatically propagate the
    uncertainties of a :py:class:`Number` instance through the derivative of the operation. Example:

    .. code-block:: python

        num = ops.pow(Number(5, 1), 2)
        print(num) # -> 25.00 (+10.00, -10.00)
    """

    # registered operations mapped to their names
    _instances = {}

    # mapping of ufunc to operation names for faster lookup
    _ufuncs = {}

    @classmethod
    def register(cls, function=None, name=None, ufunc=None):
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

        To comply with numpy's ufuncs (https://numpy.org/neps/nep-0013-ufunc-overrides.html) that
        are dispatched by :py:meth:`Number.__array_ufunc__`, an operation might register the *ufunc*
        object that it handles. When *ufunc* is a string, it is interpreted as a name of a numpy
        function. It can also be a list to signalize that it handles more than one function.
        """
        # prepare ufuncs
        ufuncs = []
        if ufunc is not None:
            for u in (ufunc if isinstance(ufunc, (list, tuple)) else [ufunc]):
                if isinstance(u, string_types):
                    if not HAS_NUMPY:
                        continue
                    u = getattr(np, u)
                ufuncs.append(u)

        def register(function):
            op = Operation(function, name=name, ufuncs=ufuncs)

            # save as class attribute and also in _instances
            cls._instances[op.name] = op
            setattr(cls, op.name, op)

            # add ufuncs to mapping
            for ufunc in op.ufuncs:
                cls._ufuncs[ufunc] = op.name

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

    @classmethod
    def get_ufunc_operation(cls, ufunc):
        """
        Returns an operation that was previously registered to handle a NumPy *ufunc*, which can be
        a string or the function itself. *None* is returned when no operation was found to handle
        the function.
        """
        if isinstance(ufunc, string_types):
            if not HAS_NUMPY:
                return None
            ufunc = getattr(np, ufunc)

        if ufunc not in cls._ufuncs:
            return None

        op_name = cls._ufuncs[ufunc]
        return cls.get_operation(op_name)

    @classmethod
    def rebuilt_ufunc_cache(cls):
        """
        Rebuilts the internal cache of ufuncs.
        """
        cls._ufuncs.clear()
        for name, op in cls._instances.items():
            for ufunc in op.ufuncs:
                cls._ufuncs[ufunc] = name


#
# pre-registered operations
#

@ops.register(ufunc="add")
def add(x, n):
    """ add(x, n)
    Addition function.
    """
    return x + n


@add.derive
def add(x, n):
    return 1.


@ops.register(ufunc="subtract")
def sub(x, n):
    """ sub(x, n)
    Subtraction function.
    """
    return x - n


@sub.derive
def sub(x, n):
    return 1.


@ops.register(ufunc="multiply")
def mul(x, n):
    """ mul(x, n)
    Multiplication function.
    """
    return x * n


@mul.derive
def mul(x, n):
    return n


@ops.register(ufunc="divide")
def div(x, n):
    """ div(x, n)
    Division function.
    """
    return x / n


@div.derive
def div(x, n):
    return 1. / n


@ops.register(ufunc="power")
def pow(x, n):
    """ pow(x, n)
    Power function.
    """
    return x**n


@pow.derive
def pow(x, n):
    return n * x**(n - 1.)


@ops.register(ufunc="exp")
def exp(x):
    """ exp(x)
    Exponential function.
    """
    return infer_math(x).exp(x)


exp.derivative = exp.function


@ops.register(ufunc="log")
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


@ops.register(ufunc="log10")
def log10(x):
    """ log10(x)
    Logarithmic function with base 10.
    """
    return log.function(x, base=10.)


@log10.derive
def log10(x):
    return log.derivative(x, base=10.)


@ops.register(ufunc="log2")
def log2(x):
    """ log2(x)
    Logarithmic function with base 2.
    """
    return log.function(x, base=2.)


@log2.derive
def log2(x):
    return log.derivative(x, base=2.)


@ops.register(ufunc="sqrt")
def sqrt(x):
    """ sqrt(x)
    Square root function.
    """
    return infer_math(x).sqrt(x)


@sqrt.derive
def sqrt(x):
    return 1. / (2 * infer_math(x).sqrt(x))


@ops.register(ufunc="sin")
def sin(x):
    """ sin(x)
    Trigonometric sin function.
    """
    return infer_math(x).sin(x)


@sin.derive
def sin(x):
    return infer_math(x).cos(x)


@ops.register(ufunc="cos")
def cos(x):
    """ cos(x)
    Trigonometric cos function.
    """
    return infer_math(x).cos(x)


@cos.derive
def cos(x):
    return -infer_math(x).sin(x)


@ops.register(ufunc="tan")
def tan(x):
    """ tan(x)
    Trigonometric tan function.
    """
    return infer_math(x).tan(x)


@tan.derive
def tan(x):
    return 1. / infer_math(x).cos(x)**2.


@ops.register(ufunc="arcsin")
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
    return 1. / infer_math(x).sqrt(1 - x**2.)


@ops.register(ufunc="arccos")
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
    return -1. / infer_math(x).sqrt(1 - x**2.)


@ops.register(ufunc="arctan")
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
    return 1. / (1 + x**2.)


@ops.register(ufunc="sinh")
def sinh(x):
    """ sinh(x)
    Hyperbolic sin function.
    """
    return infer_math(x).sinh(x)


@sinh.derive
def sinh(x):
    return infer_math(x).cosh(x)


@ops.register(ufunc="cosh")
def cosh(x):
    """ cosh(x)
    Hyperbolic cos function.
    """
    return infer_math(x).cosh(x)


@cosh.derive
def cosh(x):
    return infer_math(x).sinh(x)


@ops.register(ufunc="tanh")
def tanh(x):
    """ tanh(x)
    Hyperbolic tan function.
    """
    return infer_math(x).tanh(x)


@tanh.derive
def tanh(x):
    return 1. / infer_math(x).cosh(x)**2.


@ops.register(ufunc="arcsinh")
def asinh(x):
    """ asinh(x)
    Hyperbolic arc sin function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.asinh(x)
    else:
        return _math.arcsinh(x)


@ops.register(ufunc="arccosh")
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


@ops.register(ufunc="arctanh")
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
    return 1. / (1. - x**2.)


#
# helper functions
#

def try_float(x):
    """
    Tries to convert a value *x* to float and returns it on success, and *None* otherwise.
    """
    try:
        return float(x)
    except:
        return None


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


def is_ufloat(x):
    """
    Returns *True* when the "uncertainties" package is available on your system and *x* is a
    ``ufloat``.
    """
    return HAS_UNCERTAINTIES and isinstance(x, _uncs.core.AffineScalarFunc)


def parse_ufloat(x, default_tag=Number.DEFAULT):
    """
    Takes a ``ufloat`` object *x* from the "uncertainties" package and returns a tuple with two
    elements containing its nominal value and a dictionary with its uncertainties. When the error
    components of *x* contain multiple uncertainties with the same name, they are combined under the
    assumption of full correlation. When an error component is not tagged, *default_tag* is used.
    """
    # store error components to be combined per tag
    components = defaultdict(list)
    for comp, value in x.error_components().items():
        name = comp.tag if comp.tag is not None else default_tag
        components[name].append((x.derivatives[comp], value))

    # combine components to uncertainties, assume full correlation
    uncertainties = {
        name: calculate_uncertainty(terms, rho=1.)
        for name, terms in components.items()
    }

    return x.nominal_value, uncertainties


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


def split_value(val):
    """
    Splits a value *val* into its significand and decimal exponent (magnitude) and returns them in a
    2-tuple. *val* might also be a numpy array. Example:

    .. code-block:: python

        split_value(1)     # -> (1.0, 0)
        split_value(0.123) # -> (1.23, -1)
        split_value(-42.5) # -> (-4.25, 1)

        a = np.array([1, 0.123, -42.5])
        split_value(a) # -> ([1., 1.23, -4.25], [0, -1, 1])

    The significand will be a float while magnitude will be an integer. *val* can be reconstructed
    via ``significand * 10**magnitude``.
    """
    val = ensure_nominal(val)

    if not is_numpy(val):
        # handle 0 separately
        if val == 0:
            return (0., 0)

        mag = int(math.floor(math.log10(abs(val))))
        sig = float(val) / (10.**mag)

    else:
        log = np.zeros(val.shape)
        np.log10(np.abs(val), out=log, where=(val != 0))
        mag = np.floor(log).astype(int)
        sig = val.astype(float) / (10.**mag)

    return (sig, mag)


def _match_precision(val, ref, *args, **kwargs):
    if isinstance(ref, float) and ref >= 1:
        ref = int(ref)
    val = decimal.Decimal(str(val))
    ref = decimal.Decimal(str(ref))
    return str(val.quantize(ref, *args, **kwargs))


def match_precision(val, ref, *args, **kwargs):
    """
    Returns a string version of a value *val* matching the significant digits as given in *ref*.
    *val* might also be a numpy array. All remaining *args* and *kwargs* are forwarded to
    ``Decimal.quantize``. Example:

    .. code-block:: python

        match_precision(1.234, ".1") # -> "1.2"
        match_precision(1.234, "1.") # -> "1"
        match_precision(1.234, ".1", decimal.ROUND_UP) # -> "1.3"

        a = np.array([1.234, 5.678, -9.101])
        match_precision(a, ".1") # -> ["1.2", "5.7", "-9.1"]
    """
    val = ensure_nominal(val)

    if not is_numpy(val):
        ret = _match_precision(val, ref, *args, **kwargs)

    else:
        # strategy: map into a flat list, create chararray with max itemsize, reshape
        strings = [_match_precision(v, r, *args, **kwargs) for v, r in np.nditer([val, ref])]
        ret = np.chararray(len(strings), itemsize=max(len(s) for s in strings))
        ret[:] = strings
        ret = ret.reshape(val.shape)

    return ret


def _infer_precision(unc, sig, mag, method):
    prec = 1
    if method not in ("one", "onedigit"):
        first_three = int(round(sig * 100))
        if first_three <= 354:
            prec += 1
        elif first_three >= 950:
            prec += 1
            if method == "pdg":
                sig = 1.0
                mag += 1
        if method in ("pub", "publication"):
            prec += 1

    return prec, sig, mag


_op_map = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
}

_op_map_reverse = dict(zip(_op_map.values(), _op_map.keys()))


def calculate_uncertainty(terms, rho=0.):
    """
    Generically calculates the uncertainty of a quantity that depends on multiple *terms*. Each term
    is expected to be a 2-tuple containing the derivative and the uncertainty of the term.
    Correlations can be defined via *rho*. When *rho* is a numner, all correlations are set to this
    value. It can also be a mapping of a 2-tuple, the two indices of the terms to describe, to their
    correlation coefficient. In case the indices of two terms are not included in this mapping, they
    are assumed to be uncorrelated. Example:

    .. code-block:: python

        calculate_uncertainty([(3, 0.5), (4, 0.5)])
        # uncorrelated
        # -> 2.5

        calculate_uncertainty([(3, 0.5), (4, 0.5)], rho=1)
        # fully correlated
        # -> 3.5

        calculate_uncertainty([(3, 0.5), (4, 0.5)], rho={(0, 1): 1})
        # fully correlated
        # -> 3.5

        calculate_uncertainty([(3, 0.5), (4, 0.5)], rho={(1, 2): 1})
        # no rho value defined for pair (0, 1), assumes zero correlation
        # -> 2.5
    """
    # sum over squaresall single terms
    variance = sum((derivative * uncertainty)**2. for derivative, uncertainty in terms)

    # add second order terms of all pairs
    for i in range(len(terms) - 1):
        for j in range(i + 1, len(terms)):
            _rho = rho.get((i, j), 0.) if isinstance(rho, dict) else rho
            variance += 2. * terms[i][0] * terms[j][0] * _rho * terms[i][1] * terms[j][1]

    return variance**0.5


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
        raise ValueError("unknown operator: {}".format(op))

    # when numpy arrays, the shapes of unc and nom must match
    if is_numpy(unc1) and is_numpy(nom1) and unc1.shape != nom1.shape:
        raise ValueError("the shape of unc1 and nom1 must be equal, found {} and {}".format(
            unc1.shape, nom1.shape))
    if is_numpy(unc2) and is_numpy(nom2) and unc2.shape != nom2.shape:
        raise ValueError("the shape of unc2 and nom2 must be equal, found {} and {}".format(
            unc2.shape, nom2.shape))

    # prepare values for combination, depends on operator
    if op in ("*", "/", "**"):
        if nom1 is None or nom2 is None:
            raise ValueError("operator '{}' requires nominal values".format(op))
        # numpy-safe conversion to float
        nom1 *= 1.
        nom2 *= 1.
        # convert uncertainties to relative values, taking into account zeros
        if is_numpy(nom1) and is_numpy(unc1):
            unc1 = np.array(unc1)
            non_zero = nom1 != 0
            unc1[non_zero] = unc1[non_zero] / nom1[non_zero]
            unc1[~non_zero] = 0.
        elif nom1:
            unc1 = unc1 / nom1
        else:
            unc1 = 0.
        if is_numpy(nom2) and is_numpy(unc2):
            unc2 = np.array(unc2)
            non_zero = nom2 != 0
            unc2[non_zero] = unc2[non_zero] / nom2[non_zero]
            unc2[~non_zero] = 0.
        elif nom2:
            unc2 = unc2 / nom2
        else:
            unc2 = 0.
        # determine the nominal value
        nom = abs(f(nom1, nom2))
    else:
        nom = 1.

    # combined formula
    if op == "**":
        return nom * abs(nom2) * (unc1**2. + (math.log(nom1) * unc2)**2. + 2 * rho *
            math.log(nom1) * unc1 * unc2)**0.5
    else:
        # flip rho for sub and div
        if op in ("-", "/"):
            rho = -rho
        return nom * (unc1**2. + unc2**2. + 2. * rho * unc1 * unc2)**0.5


def round_uncertainty(unc, method="publication"):
    """
    Rounds an uncertainty *unc* following a specific *method* and returns a 2-tuple containing the
    significant digits as a string, and the decimal magnitude that is required to recover the
    uncertainty. *unc* might also be a numpy array. Rounding methods:

    - ``"pdg"``: Rounding rules as defined by the `PDG
      <http://pdg.lbl.gov/2011/reviews/rpp2011-rev-rpp-intro.pdf#page=13>`_.
    - ``"publication"``, ``"pub``: Like ``"pdg"`` with an extra significant digit for results that
      need to be combined later.
    - ``"onedigit"``, ``"one"``: Forces one single significant digit. This is useful when there are
      multiple uncertainties that vary by more than a factor 10 among themselves.

    Example:

    .. code-block:: python

        round_uncertainty(0.123, "pub") # -> ("123", -3)
        round_uncertainty(0.123, "pdg") # -> ("12", -2)
        round_uncertainty(0.123, "one") # -> ("1", -1)

        round_uncertainty(0.456, "pub") # -> ("46", -2)
        round_uncertainty(0.456, "pdg") # -> ("5", -1)
        round_uncertainty(0.456, "one") # -> ("5", -1)

        round_uncertainty(0.987, "pub") # -> ("987", -3)
        round_uncertainty(0.987, "pdg") # -> ("10", -1)
        round_uncertainty(0.987, "one") # -> ("10", -1)

        a = np.array([0.123, 0.456, 0.987])
        round_uncertainty(a, "pub") # -> (["123", "46", "987"], [-3, -2, -3])
    """
    # validate the method
    meth = method.lower()
    if meth not in ("pub", "publication", "pdg", "one", "onedigit"):
        raise ValueError("unknown rounding method: {}".format(method))

    # split the uncertainty
    sig, mag = split_value(unc)

    # infer the precision based on the method and get updated significand and magnitude
    if not is_numpy(unc):
        prec, sig, mag = _infer_precision(unc, sig, mag, meth)
        replace_args = (".", "")
    else:
        prec = np.ones(unc.shape).astype(int)
        for p, u, s, m in np.nditer([prec, unc, sig, mag], op_flags=["readwrite"]):
            p[...], s[...], m[...] = _infer_precision(u, s, m, meth)
        replace_args = (b".", b"")

    # determine the significant digits and the decimal magnitude that would reconstruct the value
    digits = match_precision(sig, 10.**(1 - prec)).replace(*replace_args)
    mag -= prec - 1

    return (digits, mag)


def round_value(val, unc=None, unc_down=None, method="publication"):
    """
    Rounds a number *val* with a single symmetric uncertainty *unc* or asymmetric uncertainties
    *unc* (interpreted as the *up* variation) and *unc_down*, and calculates the orders of their
    magnitudes. They both can be a float or a list of floats for simultaneous evaluation. When *val*
    is a :py:class:`Number` instance, its combined uncertainty is used instead. Returns a 3-tuple
    containing:

    - The string representation of the central value.
    - The string representations of the uncertainties in a list. For the symmetric case, this list
      contains only one element.
    - The decimal magnitude.

    Examples:

    .. code-block:: python

        round_value(1.23, 0.456)        # -> ("123", ["46"], -2)
        round_value(1.23, 0.456, 0.987) # -> ("123", ["46", "99"], -2)

        round_value(1.23, [0.456, 0.312]) # -> ("123", [["456", "312"]], -3)

        vals = np.array([1.23, 4.56])
        uncs = np.array([0.45678, 0.078])
        round_value(vals, uncs) # -> (["1230", "4560"], [["457", "78"]], -3)
    """
    if isinstance(val, Number):
        unc, unc_down = val.get_uncertainty()
        val = val.nominal

    # prepare unc values
    asym = unc_down is not None
    unc_up = unc
    if not asym:
        unc_down = unc_up

    if not is_numpy(val):
        # treat as lists for simultaneous rounding
        passed_list = isinstance(unc_up, (list, tuple)) or isinstance(unc_down, (list, tuple))
        unc_up = make_list(unc_up)
        unc_down = make_list(unc_down)

        # sanity checks
        if len(unc_up) != len(unc_down):
            raise ValueError("uncertainties should have same length when passed as sequences")
        for u in unc_up + unc_down:
            if not try_float(u):
                raise TypeError("uncertainties must be convertible to float: {}".format(u))
            if u < 0:
                raise ValueError("uncertainties must be positive: {}".format(u))

        # to determine the precision, use the uncertainty with the smallest magnitude
        ref_mag = min(round_uncertainty(u, method=method)[1] for u in unc_up + unc_down)

        # convert the uncertainty and central value to match the reference magnitude
        scale = 1. / 10.**ref_mag
        val_str = match_precision(scale * val, "1")
        up_strs = [match_precision(scale * u, "1") for u in unc_up]
        down_strs = [match_precision(scale * u, "1") for u in unc_down]

        if passed_list:
            return (val_str, [up_strs, down_strs] if asym else [up_strs], ref_mag)
        else:
            return (val_str, [up_strs[0], down_strs[0]] if asym else [up_strs[0]], ref_mag)

    else:
        # check uncertainties and cast to arrays when plain numbers
        if not is_numpy(unc_up):
            if not try_float(unc_up):
                raise TypeError("uncertainty is neither an array nor float compatible: {}".format(
                    unc_up))
            unc_up = unc_up * np.ones_like(val)
        if not is_numpy(unc_down):
            if not try_float(unc_down):
                raise TypeError("uncertainty is neither an array nor float compatible: {}".format(
                    unc_down))
            unc_down = unc_down * np.ones_like(val)

        # sanity checks
        if (unc_up < 0).any():
            raise ValueError("uncertainties must be positive: {}".format(unc_up))
        elif (unc_down < 0).any():
            raise ValueError("uncertainties must be positive: {}".format(unc_down))

        # to determine the precision, use the uncertainty with the smallest magnitude
        ref_mag_up = round_uncertainty(unc_up, method=method)[1]
        ref_mag_down = round_uncertainty(unc_down, method=method)[1]
        ref_mag = min(ref_mag_up.min(), ref_mag_down.min())

        scale = 1. / 10.**ref_mag
        val_str = match_precision(scale * val, "1")
        up_str = match_precision(scale * unc_up, "1")
        down_str = match_precision(scale * unc_down, "1")

        return (val_str, [up_str, down_str] if asym else [up_str], ref_mag)


si_refixes = dict(zip(
    range(-18, 18 + 1, 3),
    ["a", "f", "p", "n", r"\mu", "m", "", "k", "M", "G", "T", "P", "E"]
))


def infer_si_prefix(f):
    """
    Infers the SI prefix of a value *f* and returns the string label and decimal magnitude in a
    2-tuple. Example:

    .. code-block:: python

        infer_si_prefix(1)     # -> ("", 0)
        infer_si_prefix(25)    # -> ("", 0)
        infer_si_prefix(4320)  # -> ("k", 3)
    """
    if f == 0:
        return "", 0
    else:
        mag = 3 * int(math.log10(abs(float(f))) // 3)
        return si_refixes[mag], mag


#: Dictionaly containing formatting styles for ``"plain"``, ``"latex"`` and ``"root"`` styles which
#: are used in :py:meth:`Number.str`. Each style dictionary contains 6 fields: ``"space"``,
#: ``"label"``, ``"unit"``, ``"sym"``, ``"asym"``, and ``"sci"``. As an example, the plain style is
#: configured as
#:
#: .. code-block:: python
#:
#:     {
#:         "space": " ",
#:         "label": "({label})",
#:         "unit": " {unit}",
#:         "sym": "+- {unc}",
#:         "asym": "+{up}-{down}",
#:         "sci": "x 1E{mag}",
#:     }
style_dict = {
    "plain": {
        "space": " ",
        "label": "({label})",
        "unit": " {unit}",
        "sym": "+- {unc}",
        "asym": "+{up}-{down}",
        "sci": "x 1E{mag}",
    },
    "latex": {
        "space": r" ",
        "label": r"\left(\text{{{label}}}\right)",
        "unit": r"\,{unit}",
        "sym": r"\pm {unc}",
        "asym": r"\,^{{+{up}}}_{{-{down}}}",
        "sci": r"\times 10^{{{mag}}}",
    },
    "root": {
        "space": " ",
        "label": "#left({label}#right)",
        "unit": " {unit}",
        "sym": "#pm {unc}",
        "asym": "^{{+{up}}}_{{-{down}}}",
        "sci": "#times 10^{{{mag}}}",
    },
}
