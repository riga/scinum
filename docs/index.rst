scinum
======

.. centered:: This page contains only API docs. For more info, visit `scinum on GitHub <https://github.com/riga/scinum>`_ or open the `example.ipynb <https://github.com/riga/scinum/blob/master/example.ipynb>`__ notebook in binder: |binder|

.. contents::

.. automodule:: scinum


Classes
^^^^^^^

``Number``
----------

.. autoclass:: Number
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


``ensure_number``
-----------------

.. autofunction:: ensure_number


``ensure_nominal``
------------------

.. autofunction:: ensure_nominal


``is_numpy``
------------

.. autofunction:: is_numpy


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


``infer_si_prefix``
-------------------

.. autofunction:: infer_si_prefix


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

   A flag that is *True* when numpy is available on your system, *False* otherwise.

.. py:attribute:: NOMINAL
   type: string

   Shorthand for :py:attr:`Number.NOMINAL`.

.. py:attribute:: UP
   type: string

   Shorthand for :py:attr:`Number.UP`.

.. py:attribute:: DOWN
   type: string

   Shorthand for :py:attr:`Number.DOWN`.

.. py:attribute:: REL
   type: string

   Shorthand for :py:attr:`Number.REL`.

.. py:attribute:: ABS
   type: string

   Shorthand for :py:attr:`Number.ABS`.


.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/riga/scinum/master?filepath=example.ipynb
   :alt: Open in binder
