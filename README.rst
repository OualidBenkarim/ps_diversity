The cost of untracked diversity in brain-imaging prediction
===========================================================


.. image:: https://mybinder.org/badge_logo.svg
     :target: https://mybinder.org/v2/gh/OualidBenkarim/ps_diversity/main?urlpath=https%3A%2F%2Fgithub.com%2FOualidBenkarim%2Fps_diversity%2Fblob%2Fmain%2Fexample.ipynb

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

..


 | This repo contains code for our work on the impact of diversity on prediction accuracy:

Benkarim O, Paquola C, Park B, Kebets V, Hong SJ, Vos de Wael R, Zhang S, Yeo BTT, Eickenberg M, Ge T,
Poline JB, Bernhardt B, and Bzdok D. (2021) *The Cost of Untracked Diversity in Brain-Imaging Prediction*. TBA

Code
~~~~

There are 2 main functions to evaluate the impact of diversity:

.. code-block:: python

    from evaluation import evaluate_diversity, evaluate_sampling

    # Compare prediction performance of contiguous vs diverse sampling schemes
    coef, score = evaluate_sampling(...)

    # Evaluate prediction performance vs diversity
    df_match, coef, score = evaluate_diversity(...)


To run the code, check the notebook `example.ipynb <https://github.com/OualidBenkarim/ps_diversity/blob/main/example.ipynb>`_.
You can also run it in `Binder <https://mybinder.org/v2/gh/OualidBenkarim/ps_diversity/main?urlpath=https%3A%2F%2Fgithub.com%2FOualidBenkarim%2Fps_diversity%2Fblob%2Fmain%2Fexample.ipynb>`_.


License
-------

The source code is available under the `BSD (3-Clause) license <https://github.com/OualidBenkarim/ps_diversity/blob/main/LICENSE>`_.