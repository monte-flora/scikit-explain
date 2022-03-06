PyMint Documentation
==================================

scikit-explain is a user-friendly Python module for machine learning explainability. Current explainability products includes
* Feature importance: 
  * Single- and Multi-pass Permutation Importance (:ref:`Brieman et al. 2001 <https://link.springer.com/article/10.1023/A:1010933404324>`_ , :ref:`Lakshmanan et al. 2015 <https://journals.ametsoc.org/view/journals/atot/32/6/jtech-d-13-00205_1.xml?rskey=hlSyXu&result=2>`_)
  * :ref:`SHAP <https://christophm.github.io/interpretable-ml-book/shap.html>`_ 
  * First-order PD/ALE Variance (:ref:`Greenwell et al. 2018 <https://arxiv.org/abs/1805.04755>`_ )    
  * Grouped Importance 

* Feature Effects/Attributions: 
  * :ref:`Partial Dependence <https://christophm.github.io/interpretable-ml-book/pdp.html>`_ (PD), 
  * :ref:`Accumulated local effects <https://christophm.github.io/interpretable-ml-book/ale.html>`_ (ALE), 
  * Random forest-based feature contributions (:ref:`treeinterpreter <http://blog.datadive.net/interpreting-random-forests/>`_)
  * :ref:`SHAP <https://christophm.github.io/interpretable-ml-book/shap.html>`_ 
  * Main Effect Complexity (MEC; :ref:`Molnar et al. 2019 <https://arxiv.org/abs/1904.03867>`_)

* Feature Interactions:
  * Second-order PD/ALE 
  * Interaction Strength and Main Effect Complexity (IAS; :ref:`Molnar et al. 2019 <https://arxiv.org/abs/1904.03867>`_)
  * Second-order PD/ALE Variance (:ref:`Greenwell et al. 2018 <https://arxiv.org/abs/1805.04755>`_) 
  * Second-order Permutation Importance (:ref:`Oh et al. 2019 <https://www.mdpi.com/2076-3417/9/23/5191>`_)
  * Friedman H-statistic (:ref:`Friedman and Popescu 2008 <https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Predictive-learning-via-rule-ensembles/10.1214/07-AOAS148.full>`_)

These explainability methods are discussed at length in Christoph Molnar's :ref:`Interpretable Machine Learning <https://christophm.github.io/interpretable-ml-book/>`_. The primary feature of this package is the accompanying built-in plotting methods, which are desgined to be easy to use while producing publication-level quality figures. The computations do leverage parallelization when possible. 

The package is under active development and will likely contain bugs or errors. Feel free to raise issues! If you employ scikit-explain in your research, please cite this github and the relevant packages listed above. 


Installation
==================
pip install scikit-explain

Documentation
==================

.. automodule:: skexplain.main.interpret_toolkit
    :members:
    
.. toctree::
    :maxdepth: 2


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


Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
