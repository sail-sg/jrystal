Installation
============


.. _installation:

Installation Guide
-----------------

.. warning::
   Before installing ``jrystal``, make sure you have the latest version of ``JAX`` installed. 
   For ``JAX`` installation instructions, please refer to the `JAX Installation Guide <https://docs.jax.dev/en/latest/installation.html>`_.


.. note::
   We strongly recommend user to install the cuda version of ``JAX`` for better performance, if you have a GPU. ``jrystal`` can be run on CPU, but it is not optimized for CPU and not recommended.


Install from PyPI
~~~~~~~~~~~~~~~~

The easiest way to install ``jrystal`` is via pip:

.. code-block:: console

   $ pip install jrystal

Install from Source
~~~~~~~~~~~~~~~~~

For the latest development version, you can install from source:

.. code-block:: console

   $ git clone https://github.com/sail-sg/jrystal.git
   $ cd jrystal
   $ pip install -e .

Verify Installation
~~~~~~~~~~~~~~~~~

To verify that jrystal is installed correctly, you can run:

.. code-block:: console

   $ python -c "import jrystal; print(jrystal.__version__)"

Troubleshooting
~~~~~~~~~~~~~

If you encounter any installation issues, please:

1. Ensure your Python version is compatible (Python 3.7 or higher recommended)
2. Update pip to the latest version: ``pip install --upgrade pip``
3. Check our `GitHub Issues <https://github.com/sail-sg/jrystal/issues>`_ page
4. If the problem persists, please `open a new issue <https://github.com/sail-sg/jrystal/issues/new>`_