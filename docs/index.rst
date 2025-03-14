Jrystal
=======================================================================
.. rst-class:: project-subtitle
A JAX-based Differentiable Density Functional Theory Framework for Materials.


Core Features
--------

.. raw:: html

  <div style="display: flex; align-items: center; justify-content: center; margin-left: 10px; margin-right: 40px; padding: 10px; border: 0px solid #ddd; border-radius: 8px;
  margin-bottom: 10px; margin-top: 10px;">
    <div style="flex: 1; padding-right: 30px; display: flex; justify-content: center;">
      <img src="_static/images/feature-AD.png" alt="feature-AD" style="width: 80%; align-self: center;">
    </div>
    <div style="flex: 4; padding-right: 10px; display: flex; align-items: center;">
      <p style="font-size: 18px; margin: 0;"> <b>Differentiable:</b> Leveraging JAX's automatic differentiation for efficient gradient computation of quantum properties, enabling straightforward optimization workflows. </p>
    </div>
  </div>

  <div style="display: flex; align-items: center; justify-content: center; margin-left: 10px; margin-right: 40px; padding: 10px; border: 0px solid #ddd; border-radius: 8px;
  margin-bottom: 10px; margin-top: 10px;">
    <div style="flex: 1; padding-right: 30px; display: flex; justify-content: center;">
      <img src="_static/images/feature-gpu.png" alt="feature-AD" style="width: 80%; align-self: center;">
    </div>
    <div style="flex: 4; padding-right: 10px; display: flex; align-items: center;">
      <p style="font-size: 18px; margin: 0;"> <b>GPU-Accelerated:</b> Optimized for modern GPU architectures, delivering high-performance quantum calculations with automatic hardware acceleration. </p>
    </div>
  </div>
    
  <div style="display: flex; align-items: center; justify-content: center; margin-left: 10px; margin-right: 40px; padding: 10px; border: 0px solid #ddd; border-radius: 8px;
  margin-bottom: 10px; margin-top: 10px;">
    <div style="flex: 1; padding-right: 30px; display: flex; justify-content: center;">
      <img src="_static/images/feature-solid.png" alt="feature-solid" style="width: 80%; align-self: center;">
    </div>
    <div style="flex: 4; padding-right: 10px; display: flex; align-items: center;">
      <p style="font-size: 18px; margin: 0;"> <b>Solid-State Calculation:</b> Full-featured framework for periodic systems using plane wave basis sets, supporting precise electronic structure calculations of crystalline materials. </p>
    </div>
  </div>
  
  <div style="display: flex; align-items: center; justify-content: center; margin-left: 10px; margin-right: 40px; padding: 10px; border: 0px solid #ddd; border-radius: 8px;
  margin-bottom: 10px; margin-top: 10px;">
    <div style="flex: 1; padding-right: 30px; display: flex; justify-content: center;">
      <img src="_static/images/feature-total.png" alt="feature-solid" style="width: 80%; align-self: center;">
    </div>
    <div style="flex: 4; padding-right: 10px; display: flex; align-items: center;">
      <p style="font-size: 18px; margin: 0;"> <b>Direct Optimization:</b> A direct minimization approach that avoids SCF iterations, enabling smooth convergence and natural integration of machine learning methods, and advanced quantum chemistry methods into density functional theory calculations. </p>
    </div>
  </div>






Call Graph
------------
.. raw:: html

    <!-- Container for the graph -->
    <div id="graph" style="text-align: center;"></div>

    <!-- Load required libraries (using CDN here) -->
    <script src="//d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/@hpcc-js/wasm@2.20.0/dist/graphviz.umd.js"></script>
    <script src="https://unpkg.com/d3-graphviz@5.6.0/build/d3-graphviz.js"></script>
    
    <!-- Add MathJax with SVG output -->
    <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$']],
        displayMath: [['$$', '$$']]
      },
      svg: {
        fontCache: 'global'
      }
    };
    </script>
    <script id="MathJax-script" async 
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
    </script>

    <!-- Render the DOT graph -->
    <script src="/_static/graph.js?v={{ now() }}"></script>


This project is a `JAX <https://github.com/google/jax/>`_-based package for differantiable density functional theory computation of solids.


Installation
------------

You may install directly from GitHub, using the following command. This can be used to obtain the most recent version of Optax:


.. code-block:: console

   pip install git+https://github.com/sail-sg/jrystal.git

If you want to install the package in development mode, you can use the following command. This allows you to modify the source code and have the changes take effect without you having to rebuild and reinstall.

.. code-block:: console
  
  git clone git@github.com:sail-sg/jrystal.git
  cd jrystal
  pip install -e .

For more installation instructions, please refer to the :doc:`installation <installation>` page.



Run with Command Line
---------------------

Once ``jrystal`` is installed, you can run it with the following command.

.. code-block:: bash

  jrystal -m energy -c config.yaml


The ``-m`` or ``--mode`` option is used to specify the mode to run. Currently, we support two modes: ``energy`` and ``band``.

- ``energy``: compute the ground state energy of a system.
- ``band``: compute the band structure of a system.

The ``-c`` or ``--config`` option is used to specify the configuration file. If it is not provided, the program will look for the file named ``config.yaml`` in the current directory. You may modify the ``config.yaml`` file to customize the calculation.



Band Structure Benchmark
------------------------

We provide a benchmark for the band structure calculation of bulk silicon, graphene, aluminum, and sodium with Quantum ESPRESSO for both all-electron and norm-conserving pseudopotentials.

All-electron calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. figure:: _static/images/band_ae/si_ae.png
          :width: 100%
          :align: center
          :alt: Image 1

          Silicon (Si)

     - .. figure:: _static/images/band_ae/al_ae.png
          :width: 100%
          :align: center
          :alt: Image 2

          Aluminum (Al)

   * - .. figure:: _static/images/band_ae/graphene_ae.png
          :width: 100%
          :align: center
          :alt: Image 3

          Graphene

     - .. figure:: _static/images/band_ae/na_ae.png
          :width: 100%
          :align: center
          :alt: Image 4

          Sodium (Na)


Norm-conserving Psuedopotential calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 50 50
   :header-rows: 0
  
   * - .. figure:: _static/images/band_nc/si.png
          :width: 100%
          :align: center
          :alt: Image 1

          Silicon (Si)

     - .. figure:: _static/images/band_nc/al.png
          :width: 100%
          :align: center
          :alt: Image 2

          Aluminum (Al)



The Team
--------

This project is developed by `SEA AI LAB (SAIL) <https://sail.sea.com/>`_. We are also grateful to researchers from `NUS I-FIM <https://ifim.nus.edu.sg/>`_ for contributing ideas and theoretical support.  

.. image:: images/sail_logo.png
   :width: 300
   :alt: Alternative text

.. image:: images/ifim_logo.png
   :width: 300
   :alt: Alternative text


Citation
--------

If you find this project useful, please cite this repo:


.. code-block:: console
  
  @software{jrystal,
    author = {Li, Tianbo and Shi, Zekun and Zhao, Jiaxi and Dale, Stephen Gregory and Vignale, Giovanni and Neto, AH Castro and Novoselov, Kostya S and Lin, Min},
    title = {Jrystal: A JAX-based Differentiable Density Functional Theory Framework for Materials},
    year = {2025},
    url = {https://github.com/sail-sg/jrystal}
  }

We also have two preprints that you may find helpful:

.. code-block:: console
  
  @inproceedings{ml4ps2024,
    title = {Jrystal: A JAX-based Differentiable Density Functional Theory Framework for Materials},
    author = {Li, Tianbo and Shi, Zekun and Dale, Stephen Gregory and Vignale, Giovanni and Lin, Min},
    booktitle = {Machine Learning and the Physical Sciences Workshop at NeurIPS 2024},
    year = {2024},
  }

and

.. code-block:: console
  
  @article{li2024diagonalization,
    title={Diagonalization without Diagonalization: A Direct Optimization Approach for Solid-State Density Functional Theory},
    author={Li, Tianbo and Lin, Min and Dale, Stephen and Shi, Zekun and Neto, AH Castro and Novoselov, Kostya S and Vignale, Giovanni},
    journal={arXiv preprint arXiv:2411.05033},
    year={2024}
  }


License
-------

This project is licensed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.


Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   examples/index
   tutorial/index
   api/index
   roadmap
   glossary
