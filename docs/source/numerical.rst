Generation methods
===================


All presented methods can be used for accurrate generation of TFBM trajectories, but for certain parameter sets, each of them may not be able to generate an accurate trajectory.
The conditions for applying a given method are checked automatically by the package.
If the process can not be generated using Davies-Harte/Wood-Chan methods, then Cholesky method is used.

If all exact methods fail, then approximate version of Wood-Chan method can be used.

Simulation of the trajectorey can be divieded into two steps: preparation step and generation step.
Preparation stes is performed only once, while generation step is performed for each generated trajectory.
Therefore generation multiple trajectories at once can significantly reduce the average time of generating a single trajectory, as the preparation step is performed only once.


Cholesky Decomposition Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method directly uses the Cholesky decomposition :math:`\Sigma = LL^T` of the covariance matrix:

1. Compute the covariance matrix :math:`\Sigma`
2. Find its Cholesky decomposition :math:`L`
3. Generate the process as :math:`X = L Z` where :math:`Z` is standard Gaussian

**Preparation time**: :math:`O(N^3)`
**Generation time**: :math:`O(N^2)`

**Limitations**: Covariance matrix need to be positive defined.


Davies-Harte Method
~~~~~~~~~~~~~~~~~~~

The Davies-Harte[2] method is based on the circulant embedding of the covariance matrix.

**Preparation time**: :math:`O(N \log N)`
**Generation time**: :math:`O(N)`

**Limitations**: Can only be applied if the embedding of the covariance matrix into circulant matrix is positive definite


Wood-Chan Method
~~~~~~~~~~~~~~~~

The Wood-Chan [1] is generalization of the Davies-Harte method. Next embedding of the covariance matrix are created until embedding is positive definite or the maximum embedding size is reached.
Wood-Chan method is exact if the covariance matrix can be embedded in a positive definite circulant matrix (with reasonable embedding size), but it can also be used as an
approximation method if the embedding is not positive definite.

**Preparation time**: :math:`O(N \log N)`
**Generation time**: :math:`O(N)`

Assymptoics above do not take into account siz of final embedding. Size of the embedding can significantly affect the performance of the method.

**Limitations**: Embedding need to be positive defined for exact generation.    


References
----------

1. Wood, A. T. A., & Chan, G. (1994). "Simulation of Stationary Gaussian Processes in [0, 1] d". *Journal of Computational and Graphical Statistics*, 3(4), 409–432

2. Davies, R. B., Harte, D. S. (1987). "Tests for Hurst effect". *Biometrika*, 74(1), 95-101.

3. Asmussen, S. (1998). "Stochastic simulation with a view towards stochastic processes." University of Aarhus. Centre for Mathematical Physics and Stochastics (MaPhySto)[MPS].
