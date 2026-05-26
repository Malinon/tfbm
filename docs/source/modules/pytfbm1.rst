TFBM1 Class  
============

Overview
--------

The :class:`tfbm.TFBM1.TFBM1` class implements tempered fractional Brownian motion of the first kind (TFBM I).

Mathematical Definition
-----------------------

TFBM I is defined as the Gaussian process:

.. math::

   B^I_{H,\lambda}(t) = \int_{\mathbb{R}} g^I_{H,\lambda,t}(s) \, dB_s

where the kernel function is:
    
.. math::

   g^I_{H,\lambda,t}(s) = (t-s)_+^{H-1/2} e^{-\lambda(t-s)_+} - (-s)_+^{H-1/2} e^{-\lambda(-s)_+}

.. automodule:: tfbm.TFBM1
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Parameter Constraints
---------------------

TFBM I has specific parameter constraints:

1. **Hurst parameter**: :math:`0 < H`
2. **Tempering parameter**: :math:`0 < \lambda`


Generation Methods
------------------

The TFBM class allows for generation of TFBM samples using Cholesky, Davies-Harte, Wood-Chan methods.
Details on these methods can be found in :doc:`../numerical`.

References
----------

The TFBM1 implementation is based on:

* Sabzikar, F., Meerschaert, M. M. (2013). "Tempered fractional Brownian motion." *Statistics & Probability Letters*, 83(10), 2269-2275.