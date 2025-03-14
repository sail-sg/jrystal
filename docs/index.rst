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


Getting Started
###############

Currently we support the following calculation methods:

* vanilla self-consistent field method.
* stochastic self-consistent field method.
* stochasic gradient-based optimizers supported by `Optax <https://optax.readthedocs.io/en/latest/>`_.


Examples
########


Theory
########
We include a detailed mathematical derivation of density functional theory from scratch, which is useful for those who do not have related background. 


Support 
-------


The Team
########

This project is developed by `SEA AI LAB (SAIL) <https://sail.sea.com/>`_. We are also grateful to researchers from `NUS I-FIM <https://ifim.nus.edu.sg/>`_ for contributing ideas and theoretical support.  

.. image:: images/sail_logo.png
   :width: 300
   :alt: Alternative text

.. image:: images/ifim_logo.png
   :width: 300
   :alt: Alternative text


Citation
########

.. code-block:: console

TODO: add citation here



License
-------



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
