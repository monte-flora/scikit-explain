Scikit-Explain Documentation
==================================

**scikit-explain** is a user-friendly Python module for machine learning explainability. Current explainability products include:

* Feature Importance:
    * Single- and Multi-pass Permutation Importance (`Breiman et al. 2001 <https://link.springer.com/article/10.1023/A:1010933404324>`_, `Lakshmanan et al. 2015 <https://journals.ametsoc.org/view/journals/atot/32/6/jtech-d-13-00205_1.xml?rskey=hlSyXu&result=2>`_)
    * First-order PD/ALE Variance (`Greenwell et al. 2018 <https://arxiv.org/abs/1805.04755>`_)
    * Grouped Permutation Importance (`Au et al. 2021 <https://arxiv.org/abs/2104.11688>`_)

* Feature Effects/Attributions:
    * `Partial Dependence <https://christophm.github.io/interpretable-ml-book/pdp.html>`_ (PD)
    * `Accumulated Local Effects <https://christophm.github.io/interpretable-ml-book/ale.html>`_ (ALE)
    * Individual Conditional Expectations (ICE)
    * `SHAP <https://christophm.github.io/interpretable-ml-book/shap.html>`_
    * `LIME <https://christophm.github.io/interpretable-ml-book/lime.html>`_
    * Random forest-based feature contributions (`TreeInterpreter <http://blog.datadive.net/interpreting-random-forests/>`_)

* Feature Interactions:
    * Second-order PD/ALE
    * Interaction Strength and Main Effect Complexity (IAS/MEC; `Molnar et al. 2019 <https://arxiv.org/abs/1904.03867>`_)
    * Friedman H-statistic (`Friedman and Popescu 2008 <https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Predictive-learning-via-rule-ensembles/10.1214/07-AOAS148.full>`_)
    * Sobol Indices

A primary feature of scikit-explain is the accompanying plotting methods, which are designed to be
easy to use while producing publication-quality figures. Computations leverage parallelization when possible.

The package is under active development. Feel free to raise issues!
If you employ scikit-explain in your research, please cite this GitHub and the relevant packages listed above.


Installation
==================
.. code-block:: bash

   pip install scikit-explain


Tutorials
==================

.. toctree::
    :maxdepth: 2

    Quickstart <quickstart>
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
