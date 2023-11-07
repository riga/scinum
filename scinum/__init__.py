# coding: utf-8

"""
Scientific numbers with multiple uncertainties and correlation-aware, gaussian propagation and numpy
support.
"""

from __future__ import annotations


__author__ = "Marcel Rieger"
__copyright__ = "Copyright 2017-2023, Marcel Rieger"
__credits__ = ["Marcel Rieger"]
__contact__ = "https://github.com/riga/scinum"
__license__ = "BSD-3-Clause"
__status__ = "Development"
__version__ = "2.0.2"
__all__ = [
    "Number", "Correlation", "DeferredResult", "Operation",
    "ops", "style_dict",
    "NOMINAL", "UP", "DOWN", "N", "U", "D",
]


import math
import re
import functools
import operator
import types
import decimal
from collections import defaultdict, OrderedDict
from types import ModuleType
from typing import TypeVar, Callable, Any, Sequence

T = TypeVar("T")

# optional imports
try:
    import numpy as np
    HAS_NUMPY = True
    np_array = np.array
except ImportError:
    np = None
    HAS_NUMPY = False
    np_array = None

try:
    import uncertainties as _uncs
    HAS_UNCERTAINTIES = True
    unc_variable = _uncs.core.Variable
except ImportError:
    _uncs = None
    HAS_UNCERTAINTIES = False
    unc_variable = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


# operators that enable correlation attachments
correlation_ops = (operator.mul, operator.matmul)


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

    def __init__(
        self: typed,
        fparse: Callable[[T], T | None] | None = None,
        *,
        setter: bool = True,
        deleter: bool = True,
        name: str | None = None,
    ) -> None:
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
            super().__init__(
                functools.wraps(fparse)(self._fget(m_name)),
                self._fset(m_name) if setter else None,
                self._fdel(m_name) if deleter else None,
            )

    def __call__(self: typed, fparse: Callable[[T], T | None]) -> typed:
        return self.__class__(fparse, *self._args)

    def _fget(self: typed, name: str) -> Callable[[typed], Any]:
        """
        Build and returns the property's *fget* method for the member defined by *name*.
        """
        def fget(inst: typed) -> Any:
            return getattr(inst, name)
        return fget

    def _fset(self: typed, name: str) -> Callable[[typed, Any], None]:
        """
        Build and returns the property's *fdel* method for the member defined by *name*.
        """
        def fset(inst: typed, value: Any) -> None:
            # the setter uses the wrapped function as well
            # to allow for value checks
            value = self.fparse(inst, value)
            setattr(inst, name, value)
        return fset

    def _fdel(self: typed, name: str) -> Callable[[typed], None]:
        """
        Build and returns the property's *fdel* method for the member defined by *name*.
        """
        def fdel(inst: typed) -> None:
            delattr(inst, name)
        return fdel


class Number(object):
    """ __init__(nominal=0.0, uncertainties={})
    Implementation of a scientific number, i.e., a *nominal* value with named *uncertainties*.
    *uncertainties* should be a dict or convertable to a dict with strings as keys, and the
    corresponding uncertainties as values. Whereas different formats are accepted for values to
    denote whether the passed value is relative or absolute, it should be noted that after some
    initial parsing, they are always stored as absolute numbers represented by floats internally.

    Uncertainty values can be normal floats to denote absolute, or complex numbers to denote
    relative values. In the latter case, only the imaginary part is used, meaning that one only
    needs to append the complex ``j`` (e.g. ``0.3j`` for a 30% effect). Asymmetric uncertainties can
    be defined by passing a 2-tuple of the above values, describing the up and down effect.

    Examples:

    .. code-block:: python

        from scinum import Number, UP, DOWN

        num = Number(2.5, {
            "sourceA": 0.5,           # absolute 0.5, both up and down
            "sourceB": (1.0, 1.5),    # absolute 1.0 up, 1.5 down
            "sourceC": 0.1j,          # relative 10%, both up and down
            "sourceD": (0.1j, 0.2j),  # relative 10% up, relative 20% down
            "sourceE": (1.0, 0.2j),   # absolute 1.0 up, relative 20% down
            "sourceF": (0.3j, 0.3),   # relative 30% up, absolute 0.3 down
        })

        # get the nominal value via direct access
        num.nominal  # => 2.5

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
        num((UP, DOWN), "sourceA")       # => (3.0, 2.0)

        # get only the uncertainty (unsigned)
        num(DOWN, ("sourceE", "sourceF"), unc=True)  # => 0.583...

        # get the uncertainty factor (unsigned)
        num(DOWN, ("sourceE", "sourceF"), factor=True)  # => 1.233...

        # combined
        num(DOWN, ("sourceE", "sourceF"), unc=True, factor=True)  # => 0.233...

    When *uncertainties* is not a dictionary, it is interpreted as the *default* uncertainty, named
    ``Number.DEFAULT``.

    This class redefines most of Python's magic functions to allow transparent use in standard
    operations like ``+``, ``*``, etc. Gaussian uncertainty propagation is applied automatically.
    When operations connect two number instances, their uncertainties are combined assuming there is
    no correlation. For correlation-aware operations, please refer to methods such as :py:meth:`add`
    or :py:meth:`mul` below. Examples:

    .. code-block:: python

        num = Number(5, 1)
        print(num + 2)  # -> '7.0 +-1.0'
        print(num * 3)  # -> '15.0 +-3.0'

        num2 = Number(2.5, 1.5)
        print(num + num2)  # -> '7.5 +-1.80277563773'
        print(num * num2)  # -> '12.5 +-7.90569415042'

        num.add(num2, rho=1)
        print(num)  # -> '7.5 +-2.5'

    See :py:meth:`str` for information on string formatting.

    .. py:classattribute:: default_format

        type: string

        The default format string (``"%s"``) that is used in :py:meth:`str()` when no format string
        was passed.

    .. py:classattribute:: default_style

        type: string

        The default style name (``"plain"``) that is used in :py:meth:`str()` when no style argument
        was passed.

    .. py:classattribute:: DEFAULT

        type: string

        Constant that denotes the default uncertainty (``"default"``).

    .. py:classattribute:: ALL

        type: string

        Constant that denotes all uncertainties (``"all"``).

    .. py:classattribute:: NOMINAL

        type: string

        Constant that denotes the nominal value (``"nominal"``).

    .. py:classattribute:: UP

        type: string

        Constant that denotes the up direction (``"up"``).

    .. py:classattribute:: DOWN

        type: string

        Constant that denotes the down direction (``"down"``).

    .. py:classattribute:: N

        type: string

        Shorthand for :py:attr:`NOMINAL`.

    .. py:classattribute:: U

        type: string

        Shorthand for :py:attr:`UP`.

    .. py:classattribute:: D

        type: string

        Shorthand for :py:attr:`DOWN`.

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

        type: bool (read-only)

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

    # uncertainty directions
    NOMINAL = "nominal"
    UP = "up"
    DOWN = "down"

    # aliases
    N = NOMINAL
    U = UP
    D = DOWN

    default_format = "%s"
    default_style = "plain"

    def __init__(
        self: Number,
        nominal: float | np_array = 0.0,
        uncertainties:
            float | np_array | tuple[float | np_array] |
            dict[str, float | np_array | tuple[float | np_array]] | None = None,
        *,
        default_format: str | None = None,
        default_style: str | None = None,
    ) -> None:
        super().__init__()

        # wrapped values
        self._nominal = None
        self._uncertainties = OrderedDict()

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
        self.default_style = default_style

    def _init_kwargs(self: Number) -> dict[str, str | None]:
        return {
            "default_format": self.default_format,
            "default_style": self.default_style,
        }

    @typed
    def nominal(self: Number, nominal: float | np_array) -> float | np_array:
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
                    raise ValueError(f"shape not matching uncertainty shape: {nominal.shape}")
            nominal = nominal.astype(self.dtype)
        else:
            raise TypeError(f"invalid nominal value: {nominal}")

        return nominal

    @property
    def n(self: Number) -> float | np_array:
        return self.nominal

    @n.setter
    def n(self: Number, n: float | np_array) -> None:
        self.nominal = n

    @typed
    def uncertainties(
        self: Number,
        uncertainties:
            float | np_array | tuple[float | np_array] |
            dict[str, float | np_array | tuple[float | np_array]],
    ) -> dict[str, tuple[float | np_array]]:
        # parser for the typed member holding the uncertainties
        if not isinstance(uncertainties, dict):
            try:
                uncertainties = dict(uncertainties)
            except:
                uncertainties = {self.DEFAULT: uncertainties}

        _uncertainties = OrderedDict()
        for name, val in uncertainties.items():
            # check the name
            if not isinstance(name, str):
                raise TypeError(f"invalid uncertainty name: {name}")

            # parse the value type
            if isinstance(val, (int, float, complex)) or is_numpy(val):
                val = (val, val)
            elif isinstance(val, list):
                val = tuple(val)
            elif not isinstance(val, tuple):
                raise TypeError(f"invalid uncertainty type: {val}")

            # check the length
            if len(val) == 0:
                continue
            if len(val) == 1:
                val = 2 * val
            if len(val) != 2:
                raise ValueError(f"invalid uncertainty format: {val}")

            # parse the value itself
            utype, up, down = "abs", None, None
            for v in val:
                # interpret complex numbers as relative uncertainties
                _utype = utype
                if isinstance(v, complex):
                    _utype = "rel"
                    v = v.imag

                # parse the value
                if isinstance(v, (int, float)):
                    v = float(v)
                    # convert to array when nominal is in array
                    if self.is_numpy:
                        v *= np.ones(self.shape, dtype=self.dtype)
                elif is_numpy(v):
                    # check the shape
                    if v.shape != self.shape:
                        raise ValueError(f"shape not matching nominal shape: {v.shape}")
                    v = v.astype(self.dtype)
                else:
                    raise TypeError(f"invalid uncertainty value: {v}")

                # convert to abs
                if _utype == "rel":
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

    def get_uncertainty(
        self: Number,
        name: str = DEFAULT,
        direction: str = NOMINAL,
        *,
        default: T | None = None,
    ) -> float | np_array | tuple[float | np_array] | T:
        """
        Returns the *absolute* up and down variaton in a 2-tuple for an uncertainty *name*. When
        *direction* is set, the particular value is returned instead of a 2-tuple. In case no
        uncertainty was found and *default* is not *None*, that value is returned.
        """
        if direction not in (self.UP, self.DOWN, self.NOMINAL):
            raise ValueError(f"unknown direction: {direction}")

        if name not in self.uncertainties:
            if default is not None:
                return default
            raise KeyError(f"no uncertainty '{name}' in {self!r}")

        unc = self.uncertainties[name]

        if direction == self.NOMINAL:
            return unc

        return unc[0 if direction == self.UP else 1]

    def u(
        self: Number,
        name: str = DEFAULT,
        direction: str = NOMINAL,
        *,
        default: T | None = None,
    ) -> float | np_array | tuple[float | np_array] | T:
        """
        Shorthand for :py:meth:`get_uncertainty`.
        """
        return self.get_uncertainty(name=name, direction=direction, default=default)

    def set_uncertainty(
        self: Number,
        name: str,
        value: float | np_array | tuple[float | np_array],
    ) -> None:
        """
        Sets the uncertainty *value* for an uncertainty *name*. *value* should have one of the
        formats as described in :py:meth:`uncertainties`.
        """
        uncertainties = self.__class__.uncertainties.fparse(self, {name: value})
        self._uncertainties.update(uncertainties)

    def combine_uncertaintes(
        self: Number,
        combine: str | dict[str, Sequence[str]] = ALL,
    ) -> Number:
        """ combine_uncertaintes(combine_uncs=ALL)
        Returns a copy of this number with certain uncertainties combined. *combine* can be a
        dictionary of keys mapping to strings denoting uncertainties that should be combined with
        keys refering to new names of the combined uncertainties.

        When :py:attr:`ALL`, all uncertainties are combined.
        """
        # create a map that contains all uncertainties
        combine_map = OrderedDict()
        if combine == self.ALL:
            combine_map[self.DEFAULT] = list(self.uncertainties.keys())
        elif isinstance(combine, dict):
            seen_uncs = set()
            for new_name, names in combine.items():
                if names == self.ALL:
                    combine_map[new_name] = list(self.uncertainties.keys())
                elif isinstance(names, (list, tuple)):
                    combine_map[new_name] = list(names)
                else:
                    raise ValueError(f"expected uncertainty names sequence, got '{names}'")
                seen_uncs |= set(combine_map[new_name])
            # add remaining uncertainties
            for name in self.uncertainties:
                if name not in seen_uncs:
                    combine_map[name] = None
        else:
            raise TypeError(f"cannot interpret uncertainties to combine from '{combine}'")

        # create combined uncertainties
        uncs = OrderedDict()
        for new_name, names in combine_map.items():
            uncs[new_name] = (
                self.uncertainties[new_name]
                if names is None
                else self.get(direction=(self.UP, self.DOWN), names=names, unc=True)
            )

        return self.__class__(self.nominal, uncs, **self._init_kwargs())

    def clear(
        self: Number,
        nominal: float | np_array | None = None,
        uncertainties:
            float | np_array | tuple[float | np_array] |
            dict[str, float | np_array | tuple[float | np_array]] | None = None,
    ) -> None:
        """
        Removes all uncertainties and sets the nominal value to zero (float). When *nominal* and
        *uncertainties* are given, these new values are set on this instance.
        """
        self.uncertainties.clear()
        self.nominal = 0.0

        if nominal is not None:
            self.nominal = nominal
        if uncertainties is not None:
            self.uncertainties = uncertainties

    def str(
        self: Number,
        format: str | None = None,
        combine_uncs: str | dict[str, Sequence[str]] | None = None,
        unit: str | None = None,
        scientific: bool = False,
        si: bool = False,
        labels: bool = True,
        style: str | None = None,
        styles: dict[str, str] | None = None,
        force_asymmetric: bool = False,
        **kwargs,
    ) -> str:
        r"""
        Returns a readable string representiation of the number. *format* is used to format
        non-NumPy nominal and uncertainty values. It can be a string such as ``"%d"``, a function
        that is called with the value to format, or a rounding function as accepted by
        :py:func:`round_value`. When *None* (the default), :py:attr:`default_format` is used. All
        keyword arguments except wildcard *kwargs* are only used to format non-NumPy values. In case
        of NumPy objects, *kwargs* are passed to `numpy.array2string
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array2string.html>`_.

        When *combine_uncs* is set, uncertainties are reduced via :py:meth:`combine_uncertaintes`.
        When *unit* is set, it is appended to the end of the string. When *scientific* is *True*,
        all values are represented by their scientific notation. When *scientific* is *False* and
        *si* is *True*, the appropriate SI prefix is used.

        *labels* controls whether uncertainty labels are shown in the string. When *True*,
        uncertainty names are used, but it can also be a list of labels whose order should match the
        uncertainty dict traversal order. *style* can be ``"plain"``, ``"fancy"``, ``"latex"``, or
        ``"root"``. When *None* (the default), :py:attr:`default_style` is used.

        *styles* can be a dict with fields ``"space"``, ``"label"``, ``"unit"``, ``"sym"``,
        ``"asym"``, ``"sci"`` to customize every aspect of the format style on top of
        :py:attr:`style_dict`. Unless *force_asymmetric* is *True*, an uncertainty is quoted
        symmetric if it yields identical values in both directions.

        Examples:

        .. code-block:: python

            n = Number(17.321, {"a": 1.158, "b": 0.453})
            n.str()                    # -> '17.321 +-1.158 (a) +-0.453 (b)'
            n.str("%.1f")              # -> '17.3 +-1.2 (a) +-0.5 (b)'
            n.str("publication")       # -> '17.32 +-1.16 (a) +-0.45 (b)'
            n.str("pdg")               # -> '17.3 +-1.2 (a) +-0.5 (b)'
            n.str(combine_uncs="all")  # -> 'TODO'

            n = Number(8848, 10)
            n.str(unit="m")                           # -> "8848.0 +-10.0 m"
            n.str(unit="m", force_asymmetric=True)    # -> "8848.0 +10.0-10.0 m"
            n.str(unit="m", scientific=True)          # -> "8.848 +-0.01 x 1E3 m"
            n.str("%.2f", unit="m", scientific=True)  # -> "8.85 +-0.01 x 1E3 m"
            n.str(unit="m", si=True)                  # -> "8.848 +-0.01 km"
            n.str("%.2f", unit="m", si=True)          # -> "8.85 +-0.01 km"
            n.str(style="fancy")                      # -> "8848.0 ±10.0"
            n.str(unit="m", style="fancy")            # -> "8848.0 ±10.0 m"
            n.str(unit="m", style="latex")            # -> "8848.0 \pm10.0\,m"
            n.str(unit="m", style="latex", si=True)   # -> "8.848 \pm0.01\,km"
            n.str(unit="m", style="root")             # -> "8848.0 #pm10.0 m"
            n.str(unit="m", style="root", si=True)    # -> "8.848 #pm0.01 km"
        """
        if format is None:
            format = self.default_format or self.__class__.default_format
        if style is None:
            style = self.default_style or self.__class__.default_style

        # when uncertainties should be combined, create a new instance and forward to its formatting
        if combine_uncs:
            return self.combine_uncertaintes(combine=combine_uncs).str(
                format=format,
                unit=unit,
                scientific=scientific,
                si=si,
                labels=labels,
                style=style,
                styles=styles,
                force_asymmetric=force_asymmetric,
                **kwargs,
            )

        if not self.is_numpy:
            # check style
            style = style.lower()
            if style not in style_dict.keys():
                raise ValueError(f"unknown style '{style}'")
            d = style_dict[style]

            # extend by custom styles
            if styles:
                d.update(styles)

            # scientific or SI notation
            prefix = ""
            transform = lambda x: x
            if scientific or si:
                if scientific:
                    mag = split_value(self.nominal)[1]
                else:
                    prefix, mag = infer_si_prefix(self.nominal)
                transform = lambda x: x * 10.0**(-mag)

            # gather and transform values
            nominal = transform(self.nominal)
            uncs = [tuple(map(transform, unc)) for unc in self.uncertainties.values()]
            names = list(self.uncertainties)

            # prepare formatting
            if callable(format):
                fmt = format
            elif isinstance(format, str) and "%" in format:
                # string formatting
                fmt = lambda x: format % x
            else:
                # special formatting implemented by round_value
                nominal, uncs, _mag = round_value(nominal, uncs, method=format, **kwargs)

                def fmt(x, **kwargs):
                    return match_precision(float(x) * 10.0**_mag, 10.0**_mag, **kwargs)

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
                if style in ("plain", "fancy") and labels:
                    text += d["space"] + d["label"].format(label="no uncertainties")

            # one ore more uncertainties
            else:
                # special case: only the default uncertainty
                if len(names) == 1 and names[0] == self.DEFAULT:
                    labels = False

                for i, (name, (up, down)) in enumerate(zip(names, uncs)):
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
                text += f"\n+ {np.array2string(up, **kwargs)}"
                text += f"\n- {np.array2string(down, **kwargs)}"
            else:
                for name, (up, down) in uncs.items():
                    text += f"\n+ {name} {np.array2string(up, **kwargs)}"
                    text += f"\n- {name} {np.array2string(down, **kwargs)}"

            return text

    def repr(self: Number, *args, **kwargs) -> str:
        """
        Returns the unique string representation of the number, forwarding all *args* and *kwargs*
        to :py:meth:`str`.
        """
        if not self.is_numpy:
            text = "'" + self.str(*args, **kwargs) + "'"
        else:
            text = f"numpy array, shape {self.shape}, {len(self.uncertainties)} uncertainties"

        return f"<{self.__class__.__name__} at {hex(id(self))}, {text}>"

    def copy(
        self: Number,
        nominal: float | np_array | None = None,
        uncertainties:
            float | np_array | tuple[float | np_array] |
            dict[str, float | np_array | tuple[float | np_array]] | None = None,
    ) -> Number:
        """
        Returns a deep copy of the number instance. When *nominal* or *uncertainties* are set, they
        overwrite the fields of the copied instance.
        """
        if nominal is None:
            nominal = self.nominal
        if uncertainties is None:
            uncertainties = self.uncertainties

        return self.__class__(nominal, uncertainties=uncertainties)

    def get(
        self: Number,
        direction: str = NOMINAL,
        names: str | Sequence[str] = ALL,
        unc: bool = False,
        factor: bool = False,
    ) -> float | np_array:
        """
        Returns different representations of the contained value(s). *direction* should be any of
        *NOMINAL*, *UP* or *DOWN*, or a tuple containing a combination of them. When not *NOMINAL*,
        *names* decides which uncertainties to take into account for the combination. When *unc* is
        *True*, only the unsigned, combined uncertainty is returned. When *False*, the nominal value
        plus or minus the uncertainty is returned. When *factor* is *True*, the ratio w.r.t. the
        nominal value is returned.
        """
        if isinstance(direction, tuple) and all(d in (NOMINAL, UP, DOWN) for d in direction):
            return tuple(
                self.get(direction=d, names=names, unc=unc, factor=factor)
                for d in direction
            )

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
                    raise ValueError(f"unknown uncertainty name(s): {unknown}")

            # calculate the combined uncertainty without correlation
            idx = int(direction == self.DOWN)
            uncs = [self.uncertainties[name][idx] for name in names]
            combined_unc = sum(u**2.0 for u in uncs)**0.5

            # determine the output value
            if unc:
                value = combined_unc
            elif direction == self.UP:
                value = self.nominal + combined_unc
            else:
                value = self.nominal - combined_unc

        else:
            raise ValueError(f"unknown direction: {direction}")

        return value if not factor else value / self.nominal

    @property
    def is_numpy(self: Number) -> bool:
        return is_numpy(self.nominal)

    @property
    def shape(self: Number) -> None | tuple[int]:
        return self.nominal.shape if self.is_numpy else None

    def add(
        self: Number,
        other: Number | DeferredResult,
        rho: float | Correlation | dict[str, float] = 1.0,
        inplace: bool = True,
    ) -> Number:
        """
        Adds an *other* :py:class:`Number` or :py:class:`DeferredResult` instance, propagating all
        uncertainties. Uncertainties with the same name are combined with the correlation *rho*,
        which can either be a :py:class:`Correlation` instance, a dict with correlations defined per
        uncertainty, or a plain float. When *inplace* is *False*, a new instance is returned.
        """
        return self._apply(operator.add, other=other, rho=rho, inplace=inplace)

    def sub(
        self: Number,
        other: Number | DeferredResult,
        rho: float | Correlation | dict[str, float] = 1.0,
        inplace: bool = True,
    ) -> Number:
        """
        Subtracts an *other* :py:class:`Number` or :py:class:`DeferredResult` instance, propagating
        all uncertainties. Uncertainties with the same name are combined with the correlation *rho*,
        which can either be a :py:class:`Correlation` instance, a dict with correlations defined per
        uncertainty, or a plain float. When *inplace* is *False*, a new instance is returned.
        """
        return self._apply(operator.sub, other=other, rho=rho, inplace=inplace)

    def mul(
        self: Number,
        other: Number | DeferredResult | Correlation,
        rho: float | Correlation | dict[str, float] = 1.0,
        inplace: bool = True,
    ) -> Number:
        """
        Multiplies by an *other* :py:class:`Number` or :py:class:`DeferredResult` instance,
        propagating all uncertainties. Uncertainties with the same name are combined with the
        correlation *rho*, which can either be a :py:class:`Correlation` instance, a dict with
        correlations defined per uncertainty, or a plain float. When *inplace* is *False*, a new
        instance is returned.

        Unlike the other operations, *other* can also be a :py:class:`Correlation` instance, in
        which case a :py:class:`DeferredResult` is returned to resolve the combination of
        uncertainties later on.
        """
        return self._apply(operator.mul, other=other, rho=rho, inplace=inplace)

    def div(
        self: Number,
        other: Number | DeferredResult,
        rho: float | Correlation | dict[str, float] = 1.0,
        inplace: bool = True,
    ) -> Number:
        """
        Divides by an *other* :py:class:`Number` or :py:class:`DeferredResult` instance, propagating
        all uncertainties. Uncertainties with the same name are combined with the correlation *rho*,
        which can either be a :py:class:`Correlation` instance, a dict with correlations defined per
        uncertainty, or a plain float. When *inplace* is *False*, a new instance is returned.
        """
        return self._apply(operator.truediv, other=other, rho=rho, inplace=inplace)

    def pow(
        self: Number,
        other: Number | DeferredResult,
        rho: float | Correlation | dict[str, float] = 1.0,
        inplace: bool = True,
    ) -> Number:
        """
        Raises by the power of an *other* :py:class:`Number` or :py:class:`DeferredResult` instance,
        propagating all uncertainties. Uncertainties with the same name are combined with the
        correlation *rho*, which can either be a :py:class:`Correlation` instance, a dict with
        correlations defined per uncertainty, or a plain float. When *inplace* is *False*, a new
        instance is returned.
        """
        return self._apply(operator.pow, other=other, rho=rho, inplace=inplace)

    def _apply(
        self: Number,
        op: str | Operation,
        other: Number | DeferredResult | Correlation,
        rho: float | Correlation | dict[str, float] = 1.0,
        inplace: bool = True,
    ) -> Number:
        # get the python op
        py_op = op
        if isinstance(op, Operation):
            py_op = op.py_op
        if not py_op:
            raise RuntimeError(
                f"cannot apply operation using {op} intance that is not configured "
                "to combine uncertainties of two operations",
            )

        # when other is a correlation object and op is (mat)mul, return a deferred result that is to
        # be resolved in the next operation
        if isinstance(other, Correlation):
            if py_op not in correlation_ops:
                names = ",".join(o.__name__ for o in correlation_ops)
                raise ValueError(
                    f"cannot apply correlation object {other} via operator {py_op.__name__}, "
                    f"supported operators are: {names}",
                )
            return DeferredResult(self, other)

        # when other is a deferred result, use its number of correlation
        if isinstance(other, DeferredResult):
            rho = other.correlation
            other = other.number

        # prepare the number to update and the other number to apply
        num = self if inplace else self.copy()
        other = ensure_number(other)

        # calculate the nominal value
        nom = py_op(num.nominal, other.nominal)

        # propagate uncertainties
        uncs = {}
        default = (0.0, 0.0)
        for name in set(num.uncertainties.keys()) | set(other.uncertainties.keys()):
            # get the correlation coefficient for this uncertainty
            if isinstance(rho, Correlation):
                _rho = rho.get(name, 1.0 if rho.default is None else rho.default)
            elif isinstance(rho, dict):
                _rho = rho.get(name, 1.0)
            else:
                _rho = rho

            # get uncertainty components
            num_unc = num.get_uncertainty(name, default=default)
            other_unc = other.get_uncertainty(name, default=default)

            # combine them
            uncs[name] = tuple(combine_uncertainties(py_op, num_unc[i], other_unc[i],
                nom1=num.nominal, nom2=other.nominal, rho=_rho) for i in range(2))

        # store values
        num.clear(nom, uncs)

        return num

    def __array_ufunc__(
        self: Number,
        ufunc: str,
        method: str,
        *inputs,
        **kwargs,
    ) -> Number:
        # only direct calls of the ufunc are supported
        if method != "__call__":
            return NotImplemented

        # try to find the proper op for that ufunc
        op = ops.get_ufunc_operation(ufunc)
        if op is None:
            return NotImplemented

        # extract kwargs
        out = kwargs.pop("out", None)

        # make sure all inputs are numbers
        inputs = tuple(map(ensure_number, inputs))

        # when the operation combines uncertainties of two numbers, use _apply, otherwise, just
        # let the operation itself handle the uncerainty update
        if op.has_py_op():
            if len(inputs) != 2:
                raise RuntimeError(
                    "the operation '{}' is configured to combine uncertainties of "
                    "two operands, but received only {}: {}".format(op, len(inputs), inputs),
                )
            result = inputs[0]._apply(op, inputs[1], inplace=False, **kwargs)
        else:
            result = op(*inputs, **kwargs)

        # insert in-place to out when set
        if out is not None:
            out = out[0]
            out.clear(result.nominal, result.uncertainties)
            result = out

        return result

    def __call__(
        self: Number,
        direction: str = NOMINAL,
        names: str | Sequence[str] = ALL,
        unc: bool = False,
        factor: bool = False,
    ) -> float | np_array:
        # shorthand for get
        return self.get(direction=direction, names=names, unc=unc, factor=factor)

    def __float__(self) -> float | np_array:
        # extract nominal value
        return self.nominal

    def __str__(self: Number) -> str:
        # forward to default str
        return self.str()

    def __repr__(self: Number) -> str:
        # forward to default repr
        return self.repr()

    def _repr_latex_(self: Number) -> str:
        return self.repr() if self.is_numpy else "${}$".format(self.str(style="latex"))

    def __contains__(self: Number, name: str) -> bool:
        # check whether name is an uncertainty
        return name in self.uncertainties

    def __nonzero__(self: Number) -> bool:
        # forward to self.nominal
        return self.nominal.__nonzero__()

    def __eq__(self: Number, other: Number | float | np_array) -> bool:
        # compare nominal values
        if isinstance(other, Number):
            other = ensure_nominal(other)

        if self.is_numpy and is_numpy(other):
            # element-wise
            try:
                return np.equal(self.nominal, other)
            except ValueError:
                return False

        if self.is_numpy or is_numpy(other):
            return (self.nominal == other).all()

        return self.nominal == other

    def __ne__(self: Number, other: Number | float | np_array) -> bool:
        # opposite of __eq__
        return not self.__eq__(other)

    def __lt__(self: Number, other: Number | float | np_array) -> bool:
        # compare nominal values
        # numpy: element-wise
        return self.nominal < ensure_nominal(other)

    def __le__(self: Number, other: Number | float | np_array) -> bool:
        # compare nominal values
        # numpy: element-wise
        return self.nominal <= ensure_nominal(other)

    def __gt__(self: Number, other: Number | float | np_array) -> bool:
        # compare nominal values
        # numpy: element-wise
        return self.nominal > ensure_nominal(other)

    def __ge__(self: Number, other: Number | float | np_array) -> bool:
        # compare nominal values
        # numpy: element-wise
        return self.nominal >= ensure_nominal(other)

    def __pos__(self: Number) -> Number:
        # simply copy
        return self.copy()

    def __neg__(self: Number) -> Number:
        # simply copy and flip the nominal value
        return self.copy(nominal=-self.nominal)

    def __abs__(self: Number) -> Number:
        # make nominal absolute
        if not is_numpy:
            nominal = abs(self.nominal)
        else:
            nominal = np.abs(self.nominal)

        return self.copy(nominal=nominal)

    def __add__(self: Number, other: Number | float | np_array) -> Number:
        return self.add(other, inplace=False)

    def __radd__(self: Number, other: Number | DeferredResult | float | np_array) -> Number:
        if isinstance(other, DeferredResult):
            return other.number.add(self, rho=other.correlation, inplace=False)

        return ensure_number(other).add(self, inplace=False)

    def __iadd__(self: Number, other: Number | float | np_array) -> Number:
        return self.add(other, inplace=True)

    def __sub__(self: Number, other: Number | float | np_array) -> Number:
        return self.sub(other, inplace=False)

    def __rsub__(self: Number, other: Number | DeferredResult | float | np_array) -> Number:
        if isinstance(other, DeferredResult):
            return other.number.sub(self, rho=other.correlation, inplace=False)

        return ensure_number(other).sub(self, inplace=False)

    def __isub__(self: Number, other: Number | float | np_array) -> Number:
        return self.sub(other, inplace=True)

    def __mul__(self: Number, other: Number | float | np_array) -> Number:
        return self.mul(other, inplace=False)

    def __rmul__(
        self: Number,
        other: Number | DeferredResult | Correlation | float | np_array,
    ) -> Number:
        if isinstance(other, Correlation):
            return self.mul(other, inplace=False)

        if isinstance(other, DeferredResult):
            return other.number.mul(self, rho=other.correlation, inplace=False)

        return ensure_number(other).mul(self, inplace=False)

    def __matmul__(self: Number, other: Correlation) -> Number:
        # only supported for correlations
        if not isinstance(other, Correlation):
            raise NotImplementedError

        return self.mul(other, inplace=False)

    def __rmatmul__(self: Number, other: Correlation) -> Number:
        return self.__matmul__(other)

    def __imul__(
        self: Number,
        other: Number | DeferredResult | Correlation | float | np_array,
    ) -> Number:
        return self.mul(other, inplace=True)

    def __div__(self: Number, other: Number | float | np_array) -> Number:
        return self.div(other, inplace=False)

    def __rdiv__(self, other):
        if isinstance(other, DeferredResult):
            return other.number.rdiv(self, rho=other.correlation, inplace=False)

        return ensure_number(other).div(self, inplace=False)

    def __idiv__(self: Number, other: Number | float | np_array) -> Number:
        return self.div(other, inplace=True)

    def __truediv__(self: Number, other: Number | float | np_array) -> Number:
        return self.div(other, inplace=False)

    def __rtruediv__(self: Number, other: Number | DeferredResult | float | np_array) -> Number:
        if isinstance(other, DeferredResult):
            return other.number.div(self, rho=other.correlation, inplace=False)

        return ensure_number(other).div(self, inplace=False)

    def __itruediv__(self: Number, other: Number | float | np_array) -> Number:
        return self.div(other, inplace=True)

    def __pow__(self: Number, other: Number | float | np_array) -> Number:
        return self.pow(other, inplace=False)

    def __rpow__(self: Number, other: Number | DeferredResult | float | np_array) -> Number:
        if isinstance(other, DeferredResult):
            return other.number.rpow(self, rho=other.correlation, inplace=False)

        return ensure_number(other).pow(self, inplace=False)

    def __ipow__(self: Number, other: Number | float | np_array) -> Number:
        return self.pow(other, inplace=True)


# module-wide shorthands for Number flags
NOMINAL = Number.NOMINAL
UP = Number.UP
DOWN = Number.DOWN
N = Number.N
U = Number.U
D = Number.D


class Correlation(object):
    """ Correlation([default], **rhos)
    Container class describing correlations to be applied to equally named uncertainties when
    combining two :py:class:`Number` instances through an operator.

    A correlation object is therefore applied to a number by means of multiplication or matrix
    multiplication (i.e. ``*`` or ``@``), resulting in a :py:class:`DeferredResult` object which is
    used subsequently by the actual combination operation with an other number. See
    :py:class:`DeferredResult` for more examples.

    Correlation coefficients can be defined per named source of uncertainty via *rhos*. When a
    coefficient is retrieved (by :py:meth:`get`) with a name that was not defined before, a
    *default* value is used, which itself defaults to one.
    """

    def __init__(self: Correlation, *args, **rhos) -> None:
        super().__init__()

        # at most one positional argument is accepted
        if len(args) >= 2:
            raise Exception("only one default value is accepted: {}".format(args))

        # store attributes
        self.default: int = float(args[0]) if len(args) == 1 else 1.0
        self.rhos: dict[str, float] = rhos

    def __repr__(self: Correlation) -> str:
        parts = [str(self.default)] + [f"{k}={v}" for k, v in self.rhos.items()]
        return f"<{self.__class__.__name__} '{', '.join(parts)}' at {hex(id(self))}>"  # noqa

    def get(self: Correlation, name: str, default: T | None = None) -> float | T:
        """
        Returns a correlation coefficient rho named *name*. When no coefficient with that name
        exists and *default* is set, which itself defaults to :py:attr:`default`, this value is
        returned instead. Otherwise, a *KeyError* is raised.
        """
        if default is None:
            default = self.default

        return self.rhos.get(name, default)


class DeferredResult(object):
    """
    Class that wraps a :py:class:`Number` instance *number* and a :py:class:`Correlation` instance
    *correlation* that is automatically produced as a result of a multiplication or matrix
    multiplication between the two. Internally, this is used for the deferred resolution of
    uncertainty correlations when combined with an other :py:class:`Number`. Example:

    .. code-block:: python

        n = Number(2, 5)

        n * Correlation(1) * n
        # -> '25.0 +-20.0' (the default)

        n * Correlation(0) * n
        # -> '25.00 +-14.14'

        # note the multiplication n * c, which creates the DeferredResult
        n**(n * c)
        # -> '3125.00 +-11842.54'

    .. py:attribute:: number

        type: Number

        The wrapped number object.

    .. py:attribute:: correlation

        type: Correlation

        The wrapped correlation object.
    """

    def __init__(self: DeferredResult, number: Number, correlation: Correlation) -> None:
        super().__init__()

        # store attributes
        self.number: Number = number
        self.correlation: Correlation = correlation


# python ops for which uncertainty propagation combining two operands is implemented
# (propagation through all other ops is straight forward using derivatives)
_py_ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
}

_py_ops_reverse = dict(zip(_py_ops.values(), _py_ops.keys()))


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

        type: string (read-only)

        The name of the operation.

    .. py:attribute:: py_op

        type: None, string (read-only)

        The symbol referring to an operation that implements uncertainty propagation combining two
        operands.

    .. py:attribute:: ufuncs

        type: list (read-only)

        List of ufunc objects that this operation handles.
    """

    def __init__(
        self: Operation,
        function: Callable,
        derivative: Callable | None = None,
        name: str | None = None,
        py_op: str | None = None,
        ufuncs: list[str] | None = None,
    ) -> None:
        super().__init__()

        # check that combined op is known
        if py_op and py_op not in _py_ops and py_op not in _py_ops_reverse:
            raise ValueError(f"unknown py_op: {py_op}")

        # store attributes
        self.function = function
        self.derivative = derivative
        self._name = name or function.__name__
        self._py_op = py_op
        self._ufuncs = ufuncs or []

        # decorator for setting the derivative
        def derive(derivative: Callable) -> Operation:
            self.derivative = derivative
            return self

        self.derive = derive

    @property
    def name(self: Operation) -> str:
        return self._name

    @property
    def py_op(self: Operation) -> str | None:
        if self._py_op in _py_ops:
            return _py_ops[self._py_op]

        if self._py_op in _py_ops_reverse:
            return self._py_op

        return None

    def has_py_op(self: Operation) -> bool:
        return self.py_op is not None

    @property
    def ufuncs(self: Operation) -> list[str]:
        return self._ufuncs

    def __repr__(self: Operation) -> str:
        return f"<{self.__class__.__name__} '{self.name}' at {hex(id(self))}>"

    def __call__(self: Operation, num: Number, *args, **kwargs) -> Number:
        if self.derivative is None:
            raise Exception(f"cannot run operation '{self.name}', no derivative registered")

        # ensure we deal with a number instance
        num = ensure_number(num)

        # all operations are designed to run on raw values (floats or NumPy arrays) so ensure we
        # take only those from all inputs
        if args:
            args = tuple(map(ensure_nominal, args))
        if kwargs:
            kwargs = {k: ensure_nominal(v) for k, v in kwargs.items()}

        # apply to the nominal value
        nominal = self.function(num.nominal, *args, **kwargs)

        # apply to all uncertainties via
        # unc_f = derivative_f(x) * unc_x
        dx = abs(self.derivative(num.nominal, *args, **kwargs))
        uncertainties = {}
        for name in num.uncertainties:
            up, down = num.get_uncertainty(name)
            uncertainties[name] = (dx * up, dx * down)

        # create and return the new number
        return num.__class__(nominal, uncertainties)


class OpsMeta(type):

    def __contains__(cls: type, name: str) -> bool:
        return name in cls._instances


class ops(object, metaclass=OpsMeta):
    """
    Number-aware replacement for the global math (or numpy) module. The purpose of the class is to
    provide operations (e.g. `pow`, `cos`, `sin`, etc.) that automatically propagate the
    uncertainties of a :py:class:`Number` instance through the derivative of the operation. Example:

    .. code-block:: python

        num = ops.pow(Number(5, 1), 2)
        print(num) # -> 25.00 (+10.00, -10.00)
    """

    # registered operations mapped to their names
    _instances: dict[str, Operation] = {}

    # mapping of ufunc to operation names for faster lookup
    _ufuncs: dict[str, Callable] = {}

    @classmethod
    def register(
        cls: OpsMeta,
        function: Callable | None = None,
        name: str | None = None,
        py_op: str | None = None,
        ufuncs: list[str] | None = None,
    ) -> Operation:
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

        When the registered operation is a member of ``operators`` and thus capable of propagating
        uncertainties with two operands, *py_op* should be set to the symbol of the operation (e.g.
        ``"*"``, see ``_py_ops``).

        To comply with NumPy's ufuncs (https://numpy.org/neps/nep-0013-ufunc-overrides.html) that
        are dispatched by :py:meth:`Number.__array_ufunc__`, an operation might register the
        *ufuncs* objects that it handles. When strings, they are interpreted as a name of a NumPy
        function.
        """
        # prepare ufuncs
        _ufuncs = []
        if ufuncs is not None:
            for u in (ufuncs if isinstance(ufuncs, (list, tuple)) else [ufuncs]):
                if isinstance(u, str):
                    if not HAS_NUMPY:
                        continue
                    u = getattr(np, u)
                _ufuncs.append(u)

        def register(function: Callable) -> Operation:
            op = Operation(function, name=name, py_op=py_op, ufuncs=_ufuncs)

            # save as class attribute and also in _instances
            cls._instances[op.name] = op
            setattr(cls, op.name, op)

            # add ufuncs to mapping
            for ufunc in op.ufuncs:
                cls._ufuncs[ufunc] = op.name

            return op

        if function is None:
            return register

        return register(function)

    @classmethod
    def get_operation(cls: OpsMeta, name: str) -> Operation:
        """
        Returns an operation that was previously registered with *name*.
        """
        return cls._instances[name]

    @classmethod
    def op(cls: OpsMeta, name: str) -> Operation:
        """
        Shorthand for :py:meth:`get_operation`.
        """
        return cls.get_operation(name)

    @classmethod
    def get_ufunc_operation(cls: OpsMeta, ufunc: str | Callable) -> Operation:
        """
        Returns an operation that was previously registered to handle a NumPy *ufunc*, which can be
        a string or the function itself. *None* is returned when no operation was found to handle
        the function.
        """
        if isinstance(ufunc, str):
            if not HAS_NUMPY:
                return None
            ufunc = getattr(np, ufunc)

        if ufunc not in cls._ufuncs:
            return None

        op_name = cls._ufuncs[ufunc]
        return cls.get_operation(op_name)

    @classmethod
    def rebuilt_ufunc_cache(cls: OpsMeta) -> None:
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

@ops.register(py_op="+", ufuncs="add")
def add(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> Number | float | np_array:
    """ add(x, n)
    Addition function.
    """
    return x + n


@add.derive
def add(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> float:
    return 1.0


@ops.register(py_op="-", ufuncs="subtract")
def sub(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> Number | float | np_array:
    """ sub(x, n)
    Subtraction function.
    """
    return x - n


@sub.derive
def sub(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> float:
    return 1.0


@ops.register(py_op="*", ufuncs="multiply")
def mul(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> Number | float | np_array:
    """ mul(x, n)
    Multiplication function.
    """
    return x * n


@mul.derive
def mul(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> Number | float | np_array:
    return n


@ops.register(py_op="/", ufuncs="divide")
def div(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> Number | float | np_array:
    """ div(x, n)
    Division function.
    """
    return x / n


@div.derive
def div(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> Number | float | np_array:
    return 1.0 / n


@ops.register(py_op="**", ufuncs="power")
def pow(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> Number | float | np_array:
    """ pow(x, n)
    Power function.
    """
    return x**n


@pow.derive
def pow(
    x: Number | float | np_array,
    n: Number | float | np_array,
) -> Number | float | np_array:
    return n * x**(n - 1.0)


@ops.register(ufuncs="exp")
def exp(x: Number | float | np_array) -> Number | float | np_array:
    """ exp(x)
    Exponential function.
    """
    return infer_math(x).exp(x)


# :)
exp.derivative = exp.function


@ops.register(ufuncs="log")
def log(
    x: Number | float | np_array,
    base: Number | float | np_array | None = None,
) -> Number | float | np_array:
    """ log(x, base=e)
    Logarithmic function.
    """
    _math = infer_math(x)
    if base is None:
        return _math.log(x)
    return _math.log(x) / _math.log(base)


@log.derive
def log(
    x: Number | float | np_array,
    base: Number | float | np_array | None = None,
) -> Number | float | np_array:
    if base is None:
        return 1.0 / x
    return 1.0 / (x * infer_math(x).log(base))


@ops.register(ufuncs="log10")
def log10(x: Number | float | np_array) -> Number | float | np_array:
    """ log10(x)
    Logarithmic function with base 10.
    """
    return log.function(x, base=10.0)


@log10.derive
def log10(x: Number | float | np_array) -> Number | float | np_array:
    return log.derivative(x, base=10.0)


@ops.register(ufuncs="log2")
def log2(x: Number | float | np_array) -> Number | float | np_array:
    """ log2(x)
    Logarithmic function with base 2.
    """
    return log.function(x, base=2.0)


@log2.derive
def log2(x: Number | float | np_array) -> Number | float | np_array:
    return log.derivative(x, base=2.0)


@ops.register(ufuncs="sqrt")
def sqrt(x: Number | float | np_array) -> Number | float | np_array:
    """ sqrt(x)
    Square root function.
    """
    return infer_math(x).sqrt(x)


@sqrt.derive
def sqrt(x: Number | float | np_array) -> Number | float | np_array:
    return 1.0 / (2.0 * infer_math(x).sqrt(x))


@ops.register(ufuncs="sin")
def sin(x: Number | float | np_array) -> Number | float | np_array:
    """ sin(x)
    Trigonometric sin function.
    """
    return infer_math(x).sin(x)


@sin.derive
def sin(x: Number | float | np_array) -> Number | float | np_array:
    return infer_math(x).cos(x)


@ops.register(ufuncs="cos")
def cos(x: Number | float | np_array) -> Number | float | np_array:
    """ cos(x)
    Trigonometric cos function.
    """
    return infer_math(x).cos(x)


@cos.derive
def cos(x: Number | float | np_array) -> Number | float | np_array:
    return -infer_math(x).sin(x)


@ops.register(ufuncs="tan")
def tan(x: Number | float | np_array) -> Number | float | np_array:
    """ tan(x)
    Trigonometric tan function.
    """
    return infer_math(x).tan(x)


@tan.derive
def tan(x: Number | float | np_array) -> Number | float | np_array:
    return 1.0 / infer_math(x).cos(x)**2.0


@ops.register(ufuncs="arcsin")
def asin(x: Number | float | np_array) -> Number | float | np_array:
    """ asin(x)
    Trigonometric arc sin function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.asin(x)
    return _math.arcsin(x)


@asin.derive
def asin(x: Number | float | np_array) -> Number | float | np_array:
    return 1.0 / infer_math(x).sqrt(1 - x**2.0)


@ops.register(ufuncs="arccos")
def acos(x: Number | float | np_array) -> Number | float | np_array:
    """ acos(x)
    Trigonometric arc cos function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.acos(x)
    return _math.arccos(x)


@acos.derive
def acos(x: Number | float | np_array) -> Number | float | np_array:
    return -1.0 / infer_math(x).sqrt(1 - x**2.0)


@ops.register(ufuncs="arctan")
def atan(x: Number | float | np_array) -> Number | float | np_array:
    """ tan(x)
    Trigonometric arc tan function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.atan(x)
    return _math.arctan(x)


@atan.derive
def atan(x: Number | float | np_array) -> Number | float | np_array:
    return 1.0 / (1.0 + x**2.0)


@ops.register(ufuncs="sinh")
def sinh(x: Number | float | np_array) -> Number | float | np_array:
    """ sinh(x)
    Hyperbolic sin function.
    """
    return infer_math(x).sinh(x)


@sinh.derive
def sinh(x: Number | float | np_array) -> Number | float | np_array:
    return infer_math(x).cosh(x)


@ops.register(ufuncs="cosh")
def cosh(x: Number | float | np_array) -> Number | float | np_array:
    """ cosh(x)
    Hyperbolic cos function.
    """
    return infer_math(x).cosh(x)


@cosh.derive
def cosh(x: Number | float | np_array) -> Number | float | np_array:
    return infer_math(x).sinh(x)


@ops.register(ufuncs="tanh")
def tanh(x: Number | float | np_array) -> Number | float | np_array:
    """ tanh(x)
    Hyperbolic tan function.
    """
    return infer_math(x).tanh(x)


@tanh.derive
def tanh(x: Number | float | np_array) -> Number | float | np_array:
    return 1.0 / infer_math(x).cosh(x)**2.0


@ops.register(ufuncs="arcsinh")
def asinh(x: Number | float | np_array) -> Number | float | np_array:
    """ asinh(x)
    Hyperbolic arc sin function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.asinh(x)
    return _math.arcsinh(x)


@ops.register(ufuncs="arccosh")
def acosh(x: Number | float | np_array) -> Number | float | np_array:
    """ acosh(x)
    Hyperbolic arc cos function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.acosh(x)
    return _math.arccosh(x)


asinh.derivative = acosh.function
acosh.derivative = asinh.function


@ops.register(ufuncs="arctanh")
def atanh(x: Number | float | np_array) -> Number | float | np_array:
    """ atanh(x)
    Hyperbolic arc tan function.
    """
    _math = infer_math(x)
    if _math is math:
        return _math.atanh(x)
    return _math.arctanh(x)


@atanh.derive
def atanh(x: Number | float | np_array) -> Number | float | np_array:
    return 1.0 / (1.0 - x**2.0)


#
# helper functions
#

def try_float(x: Any) -> float | None:
    """
    Tries to convert a value *x* to float and returns it on success, and *None* otherwise.
    """
    try:
        return float(x)
    except:
        return None


def ensure_number(num: Any, *args, **kwargs) -> Number:
    """
    Returns *num* again if it is an instance of :py:class:`Number`, or uses all passed arguments to
    create one and returns it.
    """
    return num if isinstance(num, Number) else Number(num, *args, **kwargs)


def ensure_nominal(nominal: Number | float | np_array) -> float | np_array:
    """
    Returns *nominal* again if it is not an instance of :py:class:`Number`, or returns its nominal
    value.
    """
    return nominal if not isinstance(nominal, Number) else nominal.nominal


def is_numpy(x: Any) -> bool:
    """
    Returns *True* when numpy is available on your system and *x* is a numpy type.
    """
    return HAS_NUMPY and type(x).__module__ == np.__name__


def is_ufloat(x: Any) -> bool:
    """
    Returns *True* when the "uncertainties" package is available on your system and *x* is a
    ``ufloat``.
    """
    return HAS_UNCERTAINTIES and isinstance(x, _uncs.core.AffineScalarFunc)


def parse_ufloat(
    x: unc_variable,
    default_tag: str = Number.DEFAULT,
) -> tuple[float, dict[str, float]]:
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
        name: calculate_uncertainty(terms, rho=1.0)
        for name, terms in components.items()
    }

    return x.nominal_value, uncertainties


def infer_math(x: ModuleType) -> ModuleType:
    """
    Returns the numpy module when :py:func:`is_numpy` for *x* is *True*, and the math module
    otherwise.
    """
    return np if is_numpy(x) else math


def make_list(obj: Any, cast: bool = True) -> list:
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
    return [obj]


def calculate_uncertainty(
    terms: Sequence[float | np_array],
    rho: float | dict[str, float] = 0.0,
) -> float | np_array:
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
    # sum over squares of all single terms
    variance = sum((derivative * uncertainty)**2.0 for derivative, uncertainty in terms)

    # add second order terms of all pairs if they are correlated
    for i in range(len(terms) - 1):
        for j in range(i + 1, len(terms)):
            _rho = rho.get((i, j), 0.0) if isinstance(rho, dict) else rho
            variance += 2.0 * terms[i][0] * terms[j][0] * _rho * terms[i][1] * terms[j][1]

    return variance**0.5


def combine_uncertainties(
    op: Operation | str,
    unc1: float | np_array,
    unc2: float | np_array,
    nom1: float | np_array | None = None,
    nom2: float | np_array | None = None,
    rho: float = 0.0,
) -> float | np_array:
    """
    Combines two uncertainties *unc1* and *unc2* according to an operator *op* which must be either
    ``"+"``, ``"-"``, ``"*"``, ``"/"``, or ``"**"``. The three latter operators require that you
    also pass the nominal values *nom1* and *nom2*, respectively. The correlation can be configured
    via *rho*.
    """
    # handle Operation instances
    if isinstance(op, Operation) and op.has_py_op():
        op = op.py_op

    # operator valid?
    if op in _py_ops:
        f = _py_ops[op]
    elif op in _py_ops_reverse:
        f = op
        op = _py_ops_reverse[op]
    else:
        raise ValueError(f"unknown operator: {op}")

    # when numpy arrays, the shapes of unc and nom must match
    if is_numpy(unc1) and is_numpy(nom1) and unc1.shape != nom1.shape:
        raise ValueError(
            f"the shape of unc1 and nom1 must be equal, found {unc1.shape} and {nom1.shape}",
        )
    if is_numpy(unc2) and is_numpy(nom2) and unc2.shape != nom2.shape:
        raise ValueError(
            f"the shape of unc2 and nom2 must be equal, found {unc2.shape} and {nom2.shape}",
        )

    # prepare values for combination, depends on operator
    if op in ("*", "/", "**"):
        if nom1 is None or nom2 is None:
            raise ValueError(f"operator '{op}' requires nominal values")
        # numpy-safe conversion to float
        nom1 = nom1 * 1.0
        nom2 = nom2 * 1.0

        # ensure none or both values are arrays
        def ensure_numpy(nom, unc):
            nom_numpy, unc_numpy = is_numpy(nom), is_numpy(unc)
            if nom_numpy and not unc_numpy:
                unc = np.ones_like(nom, float) * unc
            elif not nom_numpy and unc_numpy:
                nom = np.ones_like(unc, float) * nom
            return (nom_numpy or unc_numpy), nom, unc

        is_numpy1, nom1, unc1 = ensure_numpy(nom1, unc1)
        is_numpy2, nom2, unc2 = ensure_numpy(nom2, unc2)

        # convert uncertainties to relative values, taking into account zeros
        if is_numpy1:
            unc1 = np.array(unc1)
            non_zero = nom1 != 0
            unc1[non_zero] = unc1[non_zero] / nom1[non_zero]
            unc1[~non_zero] = 0.0
        elif nom1:
            unc1 = unc1 / nom1
        else:
            unc1 = 0.0
        if is_numpy2:
            unc2 = np.array(unc2)
            non_zero = nom2 != 0
            unc2[non_zero] = unc2[non_zero] / nom2[non_zero]
            unc2[~non_zero] = 0.0
        elif nom2:
            unc2 = unc2 / nom2
        else:
            unc2 = 0.0

        # determine the nominal value
        nom = abs(f(nom1, nom2))
    else:
        nom = 1.0

    # combined formula
    if op == "**":
        return (
            nom *
            abs(nom2) *
            (unc1**2.0 + (math.log(nom1) * unc2)**2.0 + 2 * rho * math.log(nom1) * unc1 * unc2)**0.5
        )

    # flip rho for sub and div
    if op in ("-", "/"):
        rho = -rho

    return nom * (unc1**2.0 + unc2**2.0 + 2.0 * rho * unc1 * unc2)**0.5


def split_value(val: float | np_array) -> tuple[float, float] | tuple[np_array, np_array]:
    """
    Splits a value *val* into its significand and decimal exponent (magnitude) and returns them in a
    2-tuple. *val* might also be a numpy array. Example:

    .. code-block:: python

        split_value(1)     # -> (1.0, 0)
        split_value(0.123) # -> (1.23, -1)
        split_value(-42.5) # -> (-4.25, 1)

        a = np.array([1, 0.123, -42.5])
        split_value(a) # -> ([1.0, 1.23, -4.25], [0, -1, 1])

    The significand will be a float while magnitude will be an integer. *val* can be reconstructed
    via ``significand * 10**magnitude``.
    """
    val = ensure_nominal(val)

    if not is_numpy(val):
        # handle 0 separately
        if val == 0:
            return (0.0, 0)

        mag = int(math.floor(math.log10(abs(val))))
        sig = float(val) / (10.0**mag)

    else:
        log = np.zeros(val.shape)
        np.log10(np.abs(val), out=log, where=(val != 0))
        mag = np.floor(log).astype(int)
        sig = val.astype(float) / (10.0**mag)

    return (sig, mag)


def _match_precision(val: float, ref: float, **kwargs) -> str:
    # extract settings not meant for quantize
    force_float = kwargs.pop("force_float", False)

    # default settings for qunatize
    kwargs.setdefault("rounding", decimal.ROUND_HALF_UP)

    # maybe cast to int
    if not force_float and isinstance(ref, float) and ref >= 1:
        ref = int(ref)

    val = decimal.Decimal(str(val))
    ref = decimal.Decimal(str(ref))

    return str(val.quantize(ref, **kwargs))


def match_precision(val: float | np_array, ref: float | np_array, **kwargs) -> str | np_array:
    """ match_precision(val, ref, force_float=False, **kwargs)
    Returns a string version of a value *val* matching the significant digits as given in *ref*.
    *val* might also be a numpy array. Unless *force_float* is *True*, the returned string might
    represent an integer in case the decimal digits are removed. All remaining *kwargs* are
    forwarded to ``Decimal.quantize``. Example:

    .. code-block:: python

        match_precision(1.234, "0.1") # -> "1.2"
        match_precision(1.234, "1.0") # -> "1"
        match_precision(1.234, "0.1", decimal.ROUND_UP) # -> "1.3"

        a = np.array([1.234, 5.678, -9.101])
        match_precision(a, "0.1") # -> ["1.2", "5.7", "-9.1"]
    """
    val = ensure_nominal(val)

    if not is_numpy(val):
        ret = _match_precision(val, ref, **kwargs)

    else:
        # strategy: map into a flat list, create chararray with max itemsize, reshape
        strings = [
            _match_precision(float(v), float(r), **kwargs)
            for v, r in np.nditer([val, ref])
        ]
        ret = np.chararray(len(strings), itemsize=max(len(s) for s in strings))
        ret[:] = strings
        ret = ret.reshape(val.shape)

    return ret


def infer_uncertainty_precision(sig: float | np_array, mag: float | np_array, method: str | int):
    """
    Infers the precision of a number given its significand *sig* and mangnitude *mag* for a certain
    *method*. The precision corresponds to the amount of significant digits to keep and, in
    particular, does not refer to the number of digits after the decimal point A 3-tuple with
    (precision, significand, magnitude) is returned.

    The *method* can either be a positive integer which directly translates to the precision, or a
    string. In the later case, see :py:func:`round_uncertainty` for details.
    """
    _is_numpy = is_numpy(sig)

    if isinstance(method, int):
        if method <= 0:
            raise ValueError(f"cannot infer precision for non-positive method value '{method}'")

        prec = method
        if _is_numpy:
            prec = np.ones(sig.shape, int) * prec

    elif method in ["pdg", "pdg+1", "publication", "pub"]:
        # default precision
        prec = 1 if method == "pdg" else 2

        if not _is_numpy:
            # make all decisions based on the three leading digits
            first_three = int(round(sig * 100))
            is_small = first_three <= 354
            is_large = first_three >= 950
            if is_small:
                prec += 1
            elif is_large and method in ["pdg", "pdg+1"]:
                # ceil and increase the magnitude
                sig = 1.0
                mag += 1
                prec += 1

        else:  # is_numpy
            if not is_numpy(mag) or sig.shape != mag.shape:
                raise ValueError(
                    "sig and mag must both be NumPy arrays with the same shape, got\n"
                    f"{sig}\nand\n{mag}",
                )

            prec = np.ones(sig.shape, int) * prec

            # make all decisions based on the three leading digits
            first_three = np.round(sig * 100).astype(int)
            is_small = first_three <= 354
            is_large = first_three >= 950
            prec[is_small] += 1
            if method in ["pdg", "pdg+1"]:
                # ceil and increase the magnitude
                sig[is_large] = 1.0
                mag[is_large] += 1
                prec[is_large] += 1

    else:
        raise ValueError(f"unknown method for inferring precision: {method}")

    return prec, sig, mag


# names of methods that are purely based on uncertainties
infer_uncertainty_precision.uncertainty_methods = ["pdg", "pdg+1", "publication", "pub"]


def round_uncertainty(
    unc: float | np_array,
    method: int | str = 1,
    precision: int | np_array | None = None,
    **kwargs,
) -> tuple[float, int, int] | np_array:
    """
    Rounds an uncertainty *unc* following a specific *method* and returns a 3-tuple containing the
    significant digits as a string, the decimal magnitude that is required to recover the
    uncertainty, and the precision (== number of significant digits). *unc* might also be a numpy
    array. Possible values for the rounding *method* are:

    - ``"pdg"``: Rounding rules as defined by the `PDG
      <https://pdg.lbl.gov/2021/reviews/rpp2021-rev-rpp-intro.pdf#page=18>`_.
    - ``"pdg+1"``: Same rules as for ``"pdg"`` with an additional significant digit.
    - ``"publication"``, ``"pub"``: Same rules as for``"pdg+1"`` but without the rounding of the
      first three significant digits above 949 to 1000.
    - positive integer: Enforces a fixed number of significant digits.

    By default, the target *precision* is derived from the rounding method itself. However, a value
    can be defined to enfore a certain number of significant digits **after** the rounding took
    place. This is only useful for methods that include fixed rounding thresholds (``"pdg"``). All
    remaining *kwargs* are forwarded to :py:func:`match_precision` which is performing the rounding
    internally.

    Example:

    .. code-block:: python

        round_uncertainty(0.123, 1)      # -> ("1", -1, 1)
        round_uncertainty(0.123, "pub")  # -> ("123", -3, 3)
        round_uncertainty(0.123, "pdg")  # -> ("12", -2, 2)

        round_uncertainty(0.456, 1)      # -> ("5", -1, 1)
        round_uncertainty(0.456, "pub")  # -> ("46", -2, 2)
        round_uncertainty(0.456, "pdg")  # -> ("5", -1, 1)

        round_uncertainty(9.87, 1)      # -> ("1", 1, 1)
        round_uncertainty(9.87, "pub")  # -> ("99", -1, 2)
        round_uncertainty(9.87, "pdg")  # -> ("10", 0, 2)

        # enfore higher precision
        round_uncertainty(0.987, "pub", precision=3)  # -> ("990", -3, 3)
        round_uncertainty(0.987, "pdg", precision=3)  # -> ("100", -2, 3)

        # numpy array support
        a = np.array([0.123, 0.456, 0.987])
        round_uncertainty(a, "pub")  # -> (["123", "46", "987"], [-3, -2, -3])
    """
    # split the uncertainty
    sig, mag = split_value(unc)

    # infer the precision based on the method and get updated significand and magnitude
    prec, sig, mag = infer_uncertainty_precision(sig, mag, method)

    # apply the rounding and determine the decimal magnitude that would reconstruct the value
    digits = match_precision(sig * 10.0**(prec - 1), "1", **kwargs)
    mag -= prec - 1

    # the number of digits is now equal to the precision, except for cases where the rounding raised
    # the value to the next order of magnitude, and we rather want to encode this in the magnitude
    _is_numpy = is_numpy(digits)
    if not _is_numpy:
        if len(digits) > prec:
            digits = digits[:-1]
            mag += 1
    else:
        mask = np.char.str_len(digits) > prec
        mag[mask] += 1
        digits_flat = digits.reshape(-1)
        prec_flat = prec.reshape(-1)
        digits_flat[:] = [(d[:-1] if len(d) > p else d) for d, p in zip(digits_flat, prec_flat)]
        digits = digits_flat.reshape(digits.shape)

    # when a custom precision is set, update the digits and magnitude
    if precision is not None:
        if _is_numpy:
            if not is_numpy(precision):
                precision = np.ones(digits.shape, int) * precision
            if np.any(precision <= 0):
                raise ValueError("precision must be positive: {}".format(precision))
        elif precision <= 0:
            raise ValueError("precision must be positive: {}".format(precision))

        digits_float = np.array(digits, float) if _is_numpy else float(digits)
        digits = match_precision(digits_float * 10.0**(precision - prec), "1", **kwargs)
        mag -= precision - prec

    return (digits, mag, len(digits) if not _is_numpy else np.char.str_len(digits))


def round_value(
    val: Number | float | np_array,
    unc: float | np_array | tuple[float | np_array] | None = None,
    method: str | None = 0,
    align_precision: bool = True,
    **kwargs,
) -> tuple[str, str | list | tuple | None, int] | tuple[np_array, np_array | None, np_array]:
    """
    Rounds a number *val* with an uncertainty *unc* which can be a single float or array (symmetric)
    or a 2-tuple (asymmetric up / down) of floats or arrays. It also supports a list of these values
    for simultaneous evaluation. When *val* is a :py:class:`Number` instance, its uncertainties are
    used in their default iteration order. Returns a 3-tuple containing:

    - The string representation of the central value.
    - The string representation(s) of uncertainties. The structure is identical to the one passed on
      *unc*.
    - The decimal magnitude.

    *method* controls the behavior of the rounding:

    1. When ``"pdg"``, ``"pdg+1"``, ``"publication"``, or ``"pub"``, uncertainties are required and
       internally :py:func:`round_uncertainty` is used to infer the precision based on the smallest
       uncertainty.
    2. When a formatting string is passed, it should have the (default) pattern ``"%*.<N>f"``, and
       *N* is interpreted as the number of digits after the decimal point.
    3. When a negative integer or zero (the default) is passed, the value is interpreted as the
       number of digits after the decimal point (similar to passing a format string).
    4. When a positive number is passed, it is interpreted as the amount of significant digits to
       keep, evaluated on the smallest number among either the nominal or uncertainty values.

    In case multiple uncertainties are given and the rounding *method* is uncertainty-based (1.
    above), the precision is derived based on the smallest uncertainty as a reference and then
    enforced to the nominal value and all other uncertainties when *align_precision* is *True*.
    Otherwise, values are allowed to have different precisions. All remaining *kwargs* are forwarded
    to :py:func:`match_precision` which is performing the rounding internally.

    Examples:

    .. code-block:: python

        # differnt uncertainty structures
        round_value(1.23, 0.456, 1)             # -> ("12", "5", -1)
        round_value(1.23, [0.456], 1)           # -> ("12", ["5"], -1)
        round_value(1.23, (0.456, 0.987), 1)    # -> ("12", ("5", "10"), -1)
        round_value(1.23, [(0.456, 0.987)], 1)  # -> ("12", [("5", "10")], -1)
        round_value(1.23, [0.456, 0.987], 1)    # -> ("12", ["5", "10"], -1)

        # different rounding methods
        round_value(125.09, (0.56, 0.97))          # -> ("125", ("1", "1"), 0)
        round_value(125.09, (0.56, 0.97), "pub")   # -> ("12509", ("56", "97"), -2)
        round_value(125.09, (0.56, 0.97), "%.2f")  # -> ("12509", ("56", "97"), -2)
        round_value(125.09, (0.56, 0.97), -2)      # -> ("12509", ("56", "97"), -2)
        round_value(125.09, (0.56, 0.97), 3)       # -> ("125090", ("560", "970"), -3)

        # without uncertainties
        round_value(125.09, method=2)       # -> ("13", None, 1)
        round_value(125.09, method=-2)      # -> ("12509", None, -2)
        round_value(125.09, method="%.2f")  # -> ("12509", None, -2)
        round_value(125.09, method="pdg")   # -> Exception, "pdg" is uncertainty based

        # array support
        vals = np.array([1.23, 4.56])
        uncs = np.array([0.45678, 0.078])
        round_value(vals, uncs, 2)  # -> (["123", "4560"], ["46", "78"], [-2, -3])
    """
    if isinstance(val, Number):
        unc = list(val.uncertainties.values()) or None
        val = val.nominal

    # treat uncertainties as lists for simultaneous rounding and run checks
    has_unc = unc is not None
    _is_numpy = is_numpy(val)
    if has_unc:
        multi = isinstance(unc, list)
        if not multi:
            unc = [unc]
        flat_unc = []

        for i, u in enumerate(list(unc)):
            asym = isinstance(u, tuple)
            if asym and len(u) != 2:
                raise ValueError(f"asymmetric uncertainties must provided as 2-tuple: {u}")

            _us = list(u) if asym else [u]
            if not _is_numpy:
                for _u in _us:
                    if not try_float(_u):
                        raise TypeError(f"uncertainties must be convertible to float: {_u}")
                    if _u < 0:
                        raise ValueError(f"uncertainties must be positive: {_u}")

            else:
                for j, _u in enumerate(list(_us)):
                    if not is_numpy(_u):
                        if not try_float(_u):
                            raise TypeError(f"uncertainty is neither array nor float: {_u}")
                        _us[j] = _u = _u * np.ones_like(val)
                    if (_u < 0).any():
                        raise ValueError(f"uncertainties must be positive: {_u}")
                unc[i] = tuple(_us) if asym else _us[0]

            # store in flat list of uncertainty values
            flat_unc.extend(_us)

    # determine the formatting or precision, based on the rounding method
    if method in infer_uncertainty_precision.uncertainty_methods:
        # uncertainty based rounding
        if not has_unc:
            raise ValueError(
                f"cannot perform uncertainty based rounding with method '{method}' "
                f"without uncertainties on value {val}",
            )

        # use the uncertainty with the smallest magnitude
        get_mag = lambda u: round_uncertainty(u, method=method)[1]
        if not _is_numpy:
            ref_mag = min(map(get_mag, flat_unc))
        else:
            ref_mag = np.min(np.stack([
                np.minimum(*map(get_mag, u)) if isinstance(u, tuple) else get_mag(u)
                for u in unc
            ], axis=0), axis=0)

        # if requested, enforce rounding of the nominal value and all uncertainties according to
        # the selected method resulting in consistent precisions
        if align_precision:
            def rnd(u):
                digits, mag, _ = round_uncertainty(u, method=method)
                return (np.array(digits, float) if _is_numpy else float(digits)) * 10.0**mag
            unc = [
                (tuple(map(rnd, u)) if isinstance(u, tuple) else rnd(u))
                for u in unc
            ]

    elif isinstance(method, int) and method > 0:
        # positive integer passed, interpret as number of significant digits of smallest value,
        # either in nominal value or uncertainties
        ref = val
        if has_unc:
            if not _is_numpy:
                ref = min([val] + flat_unc)
            else:
                ref = np.min(np.stack([val] + flat_unc, axis=0), axis=0)
        ref_mag = split_value(ref)[1] - (method - 1)

    elif ((isinstance(method, int) and method <= 0) or
            (isinstance(method, str) and method.startswith("%"))):
        # negative number of format string, interpret as number of digits after decimal point
        if isinstance(method, str):
            m = re.match(r"^\%.*\.(\d+)f$", method)
            if not m:
                raise ValueError(f"format string should end with '.<int>f': {method}")
            method = -int(m.group(1))

        # trivial case
        if not _is_numpy:
            ref_mag = method
        else:
            ref_mag = np.ones_like(val, int) * method

    else:
        raise ValueError(f"unknown method for rounding value: {method}")

    # round the central value and uncertainties
    apply_rounding = lambda v: match_precision(v * 10.0**(-ref_mag), "1", **kwargs)
    val_str = apply_rounding(val)
    if has_unc:
        unc_strs = [
            (tuple(map(apply_rounding, u)) if isinstance(u, tuple) else apply_rounding(u))
            for u in unc
        ]

    return (val_str, (unc_strs if multi else unc_strs[0]) if has_unc else None, ref_mag)


def format_multiplicative_uncertainty(
    num: Number,
    unc: str | Sequence[str] = Number.DEFAULT,
    digits: int = 3,
    asym_threshold: float = 0.2,
) -> str:
    """
    Creates an inline representation of an uncertainty named *unc* of a :py:class:`Number` *num* and
    returns it. The representation makes use of the mulitiplicative factors that would scale the
    nominal to the up/down varied values. Example:

    .. code-block:: python

        format_uncertainty_inline(Number(1.4, 0.15j))  # 15% relative uncertainty
        # -> "1.150"  # symmetric

        format_uncertainty_inline(Number(1.4, (0.15j, 0.1j)))  # +15%/-10% relative uncertainty
        # -> "1.150/0.900"  # asymmetric

    When the uncertainty is either symmetric within a certain number of *digits* and the smallest
    effect is below *asym_threshold*, the symmetric representation is used (first example). In any
    other case, the asymmetric version us returned.
    """
    # get both multiplicative factors
    f_u = num("up", unc, factor=True)
    f_d = num("down", unc, factor=True)

    # if at least one absolute effect is large, consider them asymmetric,
    # if their effects are opposite and similar, consider them symmetric
    mag_u = abs(1.0 - f_u)
    mag_d = abs(1.0 - f_d)
    sym = (
        max(mag_u, mag_d) < asym_threshold and
        round(f_u, digits) == round(2.0 - f_d, digits)
    )

    # format and return
    tmpl = "{{:.{}f}}".format(digits)
    return tmpl.format(f_u) if sym else (tmpl + "/" + tmpl).format(f_u, f_d)


si_refixes = dict(zip(
    range(-18, 18 + 1, 3),
    ["a", "f", "p", "n", r"\mu", "m", "", "k", "M", "G", "T", "P", "E"],
))


def infer_si_prefix(f: float) -> tuple[str, int]:
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

    mag = 3 * int(math.log10(abs(float(f))) // 3)
    return si_refixes[mag], mag


def create_hep_data_representer(
    method: str | int | None = None,
    force_asymmetric: bool = False,
    force_float: bool = False,
    **kwargs,
) -> Callable:
    """
    Creates a PyYAML representer function that encodes a :py:class:`Number` as a data structure that
    is compatible to the `HEPData
    <https://hepdata-submission.readthedocs.io/en/latest/data_yaml.html>`_ format for values in data
    files.

    .. code-block:: python

        import yaml
        import scinum as sn

        yaml.add_representer(sn.Number, sn.create_hep_data_representer())

    For documentation of the rounding *method*, see :py:func:`round_uncertainty`. When *None*, the
    *default_format* of the number instance is used in case it is not a python format string.
    Otherwise ``3`` is assumed, i.e., 3 significant digits. When the up and down variations of an
    uncertainty are identical after rounding, they are encoded as a symmetric uncertainty unless
    *force_asymmetric* is *True*. Also, when all decimal digits are removed during rounding, the
    final value is encoded as an integer value unless *force_float* is *True*.

    All remaining *kwargs* are forwarded to :py:func:`match_precision` which is performing the
    rounding internally.
    """
    if not HAS_YAML:
        raise RuntimeError(
            "create_hep_data_representer requires PyYAML (https://pyyaml.org) to be installed on " +
            "your system",
        )

    # yaml node factories
    y_map = lambda value: yaml.MappingNode(tag="tag:yaml.org,2002:map", value=value)
    y_seq = lambda value: yaml.SequenceNode(tag="tag:yaml.org,2002:seq", value=value)
    y_str = lambda value: yaml.ScalarNode(tag="tag:yaml.org,2002:str", value=str(value))
    y_int = lambda value: yaml.ScalarNode(tag="tag:yaml.org,2002:int", value=str(value))
    y_float = lambda value: yaml.ScalarNode(tag="tag:yaml.org,2002:float", value=str(value))
    y_int_or_float = lambda value: y_float(value) if "." in str(value) else y_int(value)

    def representer(dumper, num):
        """
        Produced node structure:
          value: float
          errors:
            - symerror: float
              label: str
            - asymerror:
                plus: float
                minus: float
              label: str
        """
        if num.is_numpy:
            raise NotImplementedError("create_hep_data_representer does not support NumPy arrays")

        # apply the rounding method
        nom = num.nominal
        uncs = list(num.uncertainties.values())
        _method = method or num.default_format or 3
        nom, uncs, mag = round_value(nom, uncs, method=_method, **kwargs)
        def fmt(x, sign=1.0):
            return match_precision(
                sign * float(x) * 10.0**mag,
                10.0**mag,
                force_float=force_float,
                **kwargs,
            )

        # build error nodes
        error_nodes = []
        for name, (up, down) in zip(num.uncertainties, uncs):
            if up == down and not force_asymmetric:
                node = y_map([
                    (y_str("label"), y_str(name)),
                    (y_str("symerror"), y_int_or_float(fmt(up))),
                ])
            else:
                node = y_map([
                    (y_str("label"), y_str(name)),
                    (y_str("asymerror"), y_map([
                        (y_str("plus"), y_int_or_float(fmt(up))),
                        (y_str("minus"), y_int_or_float(fmt(down, -1.0))),
                    ])),
                ])
            error_nodes.append(node)

        # build the value node
        value_node_items = [(y_str("value"), y_int_or_float(fmt(nom)))]
        if error_nodes:
            value_node_items.append((y_str("errors"), y_seq(error_nodes)))
        value_node = y_map(value_node_items)

        return value_node

    return representer


#: Dictionaly containing formatting styles for ``"plain"``, ``"fancy"``, ``"latex"`` and ``"root"``
#: styles which are used in :py:meth:`Number.str`. Each style dictionary contains 6 fields:
#: ``"space"``, ``"label"``, ``"unit"``, ``"sym"``, ``"asym"``, and ``"sci"``. As an example, the
#: plain style is configured as
#:
#: .. code-block:: python
#:
#:     {
#:         "space": " ",
#:         "label": "({label})",
#:         "unit": " {unit}",
#:         "sym": "+-{unc}",
#:         "asym": "+{up}-{down}",
#:         "sci": "x 1E{mag}",
#:     }
style_dict = {
    "plain": {
        "space": " ",
        "label": "({label})",
        "unit": " {unit}",
        "sym": "+-{unc}",
        "asym": "+{up}-{down}",
        "sci": "x 1E{mag}",
    },
    "fancy": {
        "space": " ",
        "label": "({label})",
        "unit": " {unit}",
        "sym": "±{unc}",
        "asym": "+{up}-{down}",
        "sci": "x 1E{mag}",
    },
    "latex": {
        "space": r" ",
        "label": r"\left(\text{{{label}}}\right)",
        "unit": r"\,{unit}",
        "sym": r"\pm{unc}",
        "asym": r"\,^{{+{up}}}_{{-{down}}}",
        "sci": r"\times 10^{{{mag}}}",
    },
    "root": {
        "space": " ",
        "label": "#left({label}#right)",
        "unit": " {unit}",
        "sym": "#pm{unc}",
        "asym": "^{{+{up}}}_{{-{down}}}",
        "sci": "#times 10^{{{mag}}}",
    },
}
