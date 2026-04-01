Scikit-Explain Documentation
==================================

**scikit-explain** is a user-friendly Python module for machine learning explainability.
For a comprehensive tutorial, see `Flora et al. (2024) <https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0018.1.xml>`_.

Current explainability products include:

* Feature Importance:
    * Single- and Multi-pass Permutation Importance (`Breiman et al. 2001 <https://link.springer.com/article/10.1023/A:1010933404324>`_; `Lakshmanan et al. 2015 <https://journals.ametsoc.org/view/journals/atot/32/6/jtech-d-13-00205_1.xml>`_; `McGovern et al. 2019 <https://journals.ametsoc.org/view/journals/bams/100/11/bams-d-18-0195.1.xml>`_)
    * First-order PD/ALE Variance (`Greenwell et al. 2018 <https://arxiv.org/abs/1805.04755>`_)
    * Grouped Permutation Importance (`Au et al. 2021 <https://arxiv.org/abs/2104.11688>`_)

* Feature Effects/Attributions:
    * `Partial Dependence <https://christophm.github.io/interpretable-ml-book/pdp.html>`_ (PD)
    * `Accumulated Local Effects <https://christophm.github.io/interpretable-ml-book/ale.html>`_ (ALE)
    * Individual Conditional Expectations (ICE)
    * `SHapley Additive Explanations <https://christophm.github.io/interpretable-ml-book/shap.html>`_ (SHAP)
    * `Local Interpretable Model-Agnostic Explanations <https://christophm.github.io/interpretable-ml-book/lime.html>`_ (LIME)
    * Random forest-based feature contributions (`TreeInterpreter <http://blog.datadive.net/interpreting-random-forests/>`_)

* Feature Interactions:
    * Second-order PD/ALE
    * Interaction Strength (IAS) and Main Effect Complexity (MEC) (`Molnar et al. 2019 <https://arxiv.org/abs/1904.03867>`_)
    * Friedman H-statistic (`Friedman and Popescu 2008 <https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Predictive-learning-via-rule-ensembles/10.1214/07-AOAS148.full>`_)
    * Sobol Indices

A primary feature of scikit-explain is the accompanying plotting methods, which are designed to be
easy to use while producing publication-quality figures. Computations leverage parallelization when possible.

The package is under active development. Feel free to raise issues!


Citation
==================

If you employ scikit-explain in your research, please cite:

.. code-block:: bibtex

   @article{Flora_2024,
     author  = {Flora, Montgomery L. and McGovern, Amy and Handler, Shawn},
     title   = {A Machine Learning Explainability Tutorial for Atmospheric Sciences},
     journal = {Artificial Intelligence for the Earth Systems},
     volume  = {3},
     number  = {1},
     pages   = {e230018},
     year    = {2024},
     doi     = {10.1175/AIES-D-23-0018.1},
     url     = {https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0018.1.xml}
   }


Installation
==================

**pip** (PyPI):

.. code-block:: bash

   pip install scikit-explain

**conda** (conda-forge):

.. code-block:: bash

   conda install -c conda-forge scikit-explain

**Development version** (most up-to-date):

.. code-block:: bash

   git clone https://github.com/monte-flora/scikit-explain.git
   cd scikit-explain
   pip install -e .


Tutorials
==================

.. toctree::
    :maxdepth: 2

    notebooks/01_quickstart
    ExplainToolkit API <explain_toolkit>
    Feature Importance <importance>
    Feature Effects <effects>
    Feature Attributions <attributions>
    Feature Interactions <interactions>
    Multiclass Classification <multiclass>
    Plot Configuration <plot_config>


Contribute
-----------

- Issue Tracker: https://github.com/monte-flora/scikit-explain/issues
- Source Code: https://github.com/monte-flora/scikit-explain


Support
----------

If you are having issues, please let us know.
We have a mailing list located at: monte.flora@noaa.gov


License
----------

The project is licensed under the BSD license.
