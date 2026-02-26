TFBM3 Class
============

Overview
--------

The :class:`tfbm.TFBM3.TFBM3` class implements tempered fractional Brownian motion of the third kind (TFBM III). This variant arises from physical considerations of particle motion in viscous media and is defined through a velocity process with specific autocorrelation properties.

Mathematical Definition
-----------------------

A stochastic process :math:`B^{III}_{H,\lambda} = \{B^{III}_{H,\lambda}(t)\}_{t\in\mathbb{R}}` is called a tempered fractional Brownian motion of the third kind (TFBM III) if it satisfies the differential equation: 

.. math::

   \frac{dB^{III}_{H,\lambda}(t)}{dt} = \nu(t),

where :math:`\nu(t)` represents a velocity process with the autocorrelation function given by:

.. math::

   \gamma_{H}(\tau) = \frac{1}{\Gamma(2H - 1)} \tau^{2H-2}e^{-\tau /\lambda}, \quad \tau > 0,

where :math:`\lambda > 0` is a characteristic crossover time scale, and the Hurst parameter
satisfies :math:`\frac{1}{2} \leq H < 1`.

.. automodule:: tfbm.TFBM3
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Parameter Constraints
---------------------

TFBM III has specific parameter constraints:

1. **Hurst parameter**: :math:`0.5 \leq H < 1`
2. **Tempering parameter**: :math:`0 < \lambda`


Generation Methods
------------------

The TFBM class allows for generation of TFBM samples using Cholesky, Davies-Harte, Wood-Chan methods.
Details on these methods can be found in :doc:`../numerical`.

References
----------

1. Molina-Garcia, D., Sandev, T., Safdari, H., Pagnini, G., Chechkin, A., Metzler, R. (2018). "Crossover from anomalous to normal diffusion: truncated power-law noise correlations and applications to dynamics in lipid bilayers." *New Journal of Physics*, 20