scinum
======

.. centered:: This page contains only API docs. For more info, visit `scinum on GitHub <https://github.com/riga/scinum>`_ or open the `example.ipynb <https://github.com/riga/scinum/blob/master/example.ipynb>`__ notebook on colab or binder: |colab| |binder|

.. contents::

.. automodule:: scinum


Classes
^^^^^^^

``Number``
----------

.. autoclass:: Number
   :member-order: bysource
   :members:


``Correlation``
---------------

.. autoclass:: Correlation
   :member-order: bysource
   :members:


``DeferredResult``
------------------

.. autoclass:: DeferredResult
   :member-order: bysource
   :members:


``ops``
-------

.. autoclass:: ops
   :member-order: bysource
   :members:


``Operation``
-------------

.. autoclass:: Operation
   :member-order: bysource
   :members:


``typed``
---------

.. autoclass:: typed
   :member-order: bysource
   :members:


Functions
^^^^^^^^^

``combine_uncertainties``
-------------------------

.. autofunction:: combine_uncertainties


``calculate_uncertainty``
-------------------------

.. autofunction:: calculate_uncertainty


``ensure_number``
-----------------

.. autofunction:: ensure_number


``ensure_nominal``
------------------

.. autofunction:: ensure_nominal


``is_numpy``
------------

.. autofunction:: is_numpy


``is_ufloat``
-------------

.. autofunction:: is_ufloat


``parse_ufloat``
----------------

.. autofunction:: parse_ufloat


``infer_math``
--------------

.. autofunction:: infer_math


``make_list``
-------------

.. autofunction:: make_list


``split_value``
---------------

.. autofunction:: split_value


``match_precision``
-------------------

.. autofunction:: match_precision


``round_uncertainty``
---------------------

.. autofunction:: round_uncertainty


``round_value``
---------------

.. autofunction:: round_value


``format_multiplicative_uncertainty``
-------------------------------------

.. autofunction:: format_multiplicative_uncertainty


``infer_si_prefix``
-------------------

.. autofunction:: infer_si_prefix


``create_hep_data_representer``
-------------------------------

.. autofunction:: create_hep_data_representer


Other attributes
^^^^^^^^^^^^^^^^

.. py:attribute:: style_dict
   type: dict

   Dictionaly containing formatting styles for ``"plain"``, ``"latex"`` and ``"root"`` styles which
   are used in :py:meth:`Number.str`. Each style dictionary contains 6 fields: ``"space"``,
   ``"label"``, ``"unit"``, ``"sym"``, ``"asym"``, and ``"sci"``. As an example, the plain style is
   configured as

   .. code-block:: python

       {
           "space": " ",
           "label": "({label})",
           "unit": " {unit}",
           "sym": "+- {unc}",
           "asym": "+{up}-{down}",
           "sci": "x 1E{mag}",
       }

.. py:attribute:: HAS_NUMPY
   type: bool

   A flag that is *True* when NumPy is available on your system, *False* otherwise.

.. py:attribute:: HAS_UNCERTAINTIES
   type: bool

   A flag that is *True* when the uncertainties package is available on your system, *False*
   otherwise.

.. py:attribute:: HAS_YAML
   type: bool

   A flag that is *True* when PyYAML is available on your system, *False* otherwise.

.. py:attribute:: NOMINAL
   type: string

   Shorthand for :py:attr:`Number.NOMINAL`.

.. py:attribute:: UP
   type: string

   Shorthand for :py:attr:`Number.UP`.

.. py:attribute:: DOWN
   type: string

   Shorthand for :py:attr:`Number.DOWN`.

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/riga/scinum/blob/master/example.ipynb
   :alt: Open in colab

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/riga/scinum/master?filepath=example.ipynb
   :alt: Open in binder
