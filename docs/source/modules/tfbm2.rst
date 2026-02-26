TFBM2 Class
============

Overview
--------

The :class:`tfbm.TFBM2.TFBM2` class implements tempered fractional Brownian motion of the second kind (TFBM II).

Mathematical Definition
-----------------------

The stochastic processes :math:`B^{II}_{H,\lambda} = \{B^{II}_{H,\lambda}(t)\}_{t\in\mathbb{R}}` defined by the Wiener integral

.. math::

   B^{II}_{H,\lambda}(t) := \int_{\mathbb{R}} g^{II}_{H,\lambda,t}(s) \,dB_{s},

where

.. math::

   g^{II}_{H,\lambda,t}(s) := (t - s)_{+}^{H- \frac{1}{2}} e^{\lambda(t-s)_{+}} - (-s)_{+}^{H- \frac{1}{2}} e^{\lambda(-s)_{+}} + \lambda \int_{0}^{t} (u - s)_{+}^{H- \frac{1}{2}} e^{\lambda(u-s)_{+}} du, \quad s \in \mathbb{R},

is called a tempered fractional Brownian motion of the second kind (TFBM II).

.. automodule:: tfbm.TFBM2
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:


Numerical Stability
-------------------

All methods for generating TFBM II samples require computing the covariance of the process, which is given by equations (2.19) and (2.20) in [1].
Unfortunately, direct computation of formula (2.19) involving generalized hypergeometric functions may result in significant numerical errors
when the product of time :math:`t` and tempering parameter :math:`\lambda` is large. Furthermore, the formula is undefined when :math:`H` is an integer or equals 0.5.
To address these issues, the package uses a heuristic to determine when to use direct numerical integration instead. If :math:`\lambda T > 25`, numerical integration is used; otherwise, the formula with hypergeometric functions is used.

Computing covariance using numerical integration is problematic for small values of :math:`H`. Users can choose between two numerical integration methods:
the "fast" method uses :py:func:`scipy.integrate.quad`, which is faster but may be less accurate for small :math:`H`, while the "strict" method uses :py:func:`mpmath.quadsubdiv`,
which is more accurate but slower. Users can also select the "flexible" strategy, which uses a heuristic to determine the appropriate numerical integration method.
By default (using the "flexible" strategy), the fast method is used for :math:`H > 0.75`, while the more accurate method is used for :math:`H \leq 0.75`.

These heuristics are based on numerical experiments. The table below shows the simplified relationship between simulation performance and process parameters for the "flexible" strategy.

================================  =========================  ==========================
            Parameters             :math:`\lambda  T < 25`    :math:`\lambda  T \geq 25` 
================================  =========================  ==========================
:math:`H > 0.75`                  FAST                       FAST
:math:`H \leq 0.75`               MODERATE                   FAST
:math:`H \approx 0.5, 1, 2, ...`  SLOW                       SLOW
================================  =========================  ==========================


Generation Methods
------------------

The TFBM class allows for generation of TFBM samples using Cholesky, Davies-Harte, Wood-Chan methods.
Details on these methods can be found in :doc:`../numerical`.

References
----------

1. Sabzikar F., Surgailis D. (2018). "Tempered fractional Brownian and stable motions of second kind" *Statistics & Probability Letters*, 132(10), 2269-2275.
2. Piessens, R.; de Doncker-Kapenga, E.; Überhuber, C. W.; Kahaner, D. (1983). "QUADPACK: A subroutine package for automatic integration" Springer-Verlag. ISBN 978-3-540-12553-2.
3. McCullough, T.; Phillips, K. (1973). "Foundations of Analysis in the Complex Plane". Holt Rinehart Winston. ISBN 0-03-086370-8
4. Virtanen, P. et al., 2020. "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python". *Nature Methods*, 17, pp.261–272.