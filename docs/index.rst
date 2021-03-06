Scikit-Explain Documentation
==================================

**scikit-explain** is a user-friendly Python module for machine learning explainability. Current explainability products includes

* Feature importance: 
    * Single- and Multi-pass Permutation Importance (`Brieman et al. 2001 <https://link.springer.com/article/10.1023/A:1010933404324>`_ , `Lakshmanan et al. 2015 <https://journals.ametsoc.org/view/journals/atot/32/6/jtech-d-13-00205_1.xml?rskey=hlSyXu&result=2>`_)
    * `SHAP <https://christophm.github.io/interpretable-ml-book/shap.html>`_ 
    * First-order PD/ALE Variance (`Greenwell et al. 2018 <https://arxiv.org/abs/1805.04755>`_ )    
    * Grouped Permutation Importance (`Au et al. 2021 <https://arxiv.org/abs/2104.11688>`_)

* Feature Effects/Attributions: 
    * `Partial Dependence <https://christophm.github.io/interpretable-ml-book/pdp.html>`_ (PD), 
    * `Accumulated local effects <https://christophm.github.io/interpretable-ml-book/ale.html>`_ (ALE), 
    * Random forest-based feature contributions (`treeinterpreter <http://blog.datadive.net/interpreting-random-forests/>`_)
    * `SHAP <https://christophm.github.io/interpretable-ml-book/shap.html>`_ 
    * Main Effect Complexity (MEC; `Molnar et al. 2019 <https://arxiv.org/abs/1904.03867>`_)

* Feature Interactions:
    * Second-order PD/ALE 
    * Interaction Strength and Main Effect Complexity (IAS; `Molnar et al. 2019 <https://arxiv.org/abs/1904.03867>`_)
    * Second-order PD/ALE Variance (`Greenwell et al. 2018 <https://arxiv.org/abs/1805.04755>`_) 
    * Second-order Permutation Importance (`Oh et al. 2019 <https://www.mdpi.com/2076-3417/9/23/5191>`_)
    * Friedman H-statistic (`Friedman and Popescu 2008 <https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Predictive-learning-via-rule-ensembles/10.1214/07-AOAS148.full>`_)

These explainability methods are discussed at length in Christoph Molnar's `Interpretable Machine Learning <https://christophm.github.io/interpretable-ml-book/>`_. A primary feature of scikit-learn is the accompanying plotting methods, which are desgined to be easy to use while producing publication-level quality figures. Lastly, computations in scikit-explain do leverage parallelization when possible. 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues! If you employ scikit-explain in your research, please cite this github and the relevant packages listed above. 


Installation
==================
pip install scikit-explain

Documentation
==================

.. toctree::
    :maxdepth: 2
    
    ExplainToolkit <explain_toolkit>
    Accumulated Local Effects <ale>
    Partial Dependence <pd>
    Feature Attributions <feature_attributions> 
    SHAP-Style <shap>
    Permutation Importance <pimp>

Contribute
-----------

- Issue Tracker: github.com/monte-flora/scikit-explain/issues
- Source Code: github.com/monte-flora/scikit-explain


Support
----------

If you are having issues, please let us know.
We have a mailing list located at: monte.flora@noaa.gov


License
----------

The project is licensed under the BSD license.

