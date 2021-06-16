The cost of untracked diversity in brain-imaging prediction
===========================================================

.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. image:: https://mybinder.org/badge_logo.svg
     :target: https://mybinder.org/v2/gh/OualidBenkarim/ps_diversity/main?urlpath=https%3A%2F%2Fgithub.com%2FOualidBenkarim%2Fps_diversity%2Fblob%2Fmain%2Fexample.ipynb

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

|

This repo contains code for our work on the role of diversity in prediction accuracy.

Paper
-----
Benkarim O, Paquola C, Park B, Kebets V, Hong SJ, Vos de Wael R, Zhang S, Yeo BTT, Eickenberg M, Ge T,
Poline JB, Bernhardt B, and Bzdok D. (2021) *The Cost of Untracked Diversity in Brain-Imaging Prediction*. TBA


Code
----
There are 2 main functions to evaluate the impact of diversity:

.. code-block:: python

    from evaluation import evaluate_diversity, evaluate_sampling

    # Compare prediction performance of contiguous vs diverse sampling schemes
    coef, score = evaluate_sampling(...)

    # Evaluate prediction performance vs diversity
    df_match, coef, score = evaluate_diversity(...)




All the functions are documented. To run the code, check the `example <https://github.com/OualidBenkarim/ps_diversity/blob/main/example.ipynb>`_ notebook.
You can also run it in `Binder <https://mybinder.org/v2/gh/OualidBenkarim/ps_diversity/main?urlpath=https%3A%2F%2Fgithub.com%2FOualidBenkarim%2Fps_diversity%2Fblob%2Fmain%2Fexample.ipynb>`_.
The example uses the sample data provided with the code. For dependencies, check `requirements.txt <https://github.com/OualidBenkarim/ps_diversity/blob/main/requirements.txt>`_.

The code was tested in Python 3.6-3.8.

License
-------

The source code is available under the `BSD (3-Clause) license <https://github.com/OualidBenkarim/ps_diversity/blob/main/LICENSE>`_.