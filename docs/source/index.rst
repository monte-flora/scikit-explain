PyMint Documentation
==================================

PyMint (Python-based Model INTerpretations) is designed to be a user-friendly package for computing and plotting machine learning interpretation output in Python. Current computation includes partial dependence (PD), accumulated local effects (ALE), random forest-based feature contributions (treeinterpreter), single- and multiple-pass permutation importance, and Shapley Additive Explanations (SHAP). All of these methods are discussed at length in Christoph Molnar's interpretable ML book (https://christophm.github.io/interpretable-ml-book/). Most calculations can be performed in parallel when multi-core processing is available. The primary feature of this package is the accompanying built-in plotting methods, which are desgined to be easy to use while producing publication-level quality figures. 


Installation
==================
pip install py-mint

Documentation
==================

.. automodule:: pymint.main.interpret_toolkit
    :members:
    
.. toctree::
    :maxdepth: 2


Contribute
-----------

- Issue Tracker: github.com/monte-flora/py-mint/issues
- Source Code: github.com/monte-flora/py-mint


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
