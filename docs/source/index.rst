.. TFBM documentation master file, created by
   sphinx-quickstart on Thu Feb 19 11:34:42 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TFBM: Tempered Fractional Brownian Motion
==========================================

Package for simulating and detecting tempered fractional Brownian motions.

Overview
--------

This package provides implementations for three types of Tempered Fractional Brownian Motion (TFBM):

* **TFBM I**: Tempered fractional Brownian motion of the first kind [1]
* **TFBM II**: Tempered fractional Brownian motion of the second kind [2] 
* **TFBM III**: Tempered fractional Brownian motion of the third kind [3]

Mathematical Definitions
------------------------

TFBM I
~~~~~~

The stochastic processes :math:`B^{I}_{H,\lambda} = \{B^{I}_{H,\lambda}(t)\}_{t\in\mathbb{R}}` defined by the Wiener integral

.. math::

   B^{I}_{H,\lambda}(t) := \int_{\mathbb{R}} g^{I}_{H,\lambda,t}(s) \,dB_{s},

where

.. math::

   g^{I}_{H,\lambda,t}(s) := \left[(t - s)_{+}^{H- \frac{1}{2}} e^{\lambda(t-s)_{+}} - (-s)_{+}^{H- \frac{1}{2}} e^{\lambda(-s)_{+}}\right], \quad s \in \mathbb{R}

is called a tempered fractional Brownian motion (TFBM I).

TFBM II  
~~~~~~~

The stochastic processes :math:`B^{II}_{H,\lambda} = \{B^{II}_{H,\lambda}(t)\}_{t\in\mathbb{R}}` defined by the Wiener integral

.. math::

   B^{II}_{H,\lambda}(t) := \int_{\mathbb{R}} g^{II}_{H,\lambda,t}(s) \,dB_{s},

where

.. math::

   g^{II}_{H,\lambda,t}(s) := (t - s)_{+}^{H- \frac{1}{2}} e^{\lambda(t-s)_{+}} - (-s)_{+}^{H- \frac{1}{2}} e^{\lambda(-s)_{+}} + \lambda \int_{0}^{t} (u - s)_{+}^{H- \frac{1}{2}} e^{\lambda(u-s)_{+}} du, \quad s \in \mathbb{R},

is called a tempered fractional Brownian motion of the second kind (TFBM II).

TFBM III
~~~~~~~~

A stochastic process :math:`B^{III}_{H,\lambda} = \{B^{III}_{H,\lambda}(t)\}_{t\in\mathbb{R}}` is called a tempered fractional Brownian motion of the third kind (TFBM III) if it satisfies the differential equation:

 

.. math::

   \frac{dB^{III}_{H,\lambda}(t)}{dt} = \nu(t),

where :math:`\nu(t)` represents a velocity process with the autocorrelation function given by:

.. math::

   \gamma_{H}(\tau) = \frac{1}{\Gamma(2H - 1)} \tau^{2H-2}e^{-\tau /\tau^{*}}, \quad \tau > 0,

where :math:`\tau^* > 0` is a characteristic crossover time scale, and the Hurst parameter
satisfies :math:`\frac{1}{2} \leq H < 1`.

Installation
------------

You can install the package using the following commands:

.. code-block:: bash

   git clone https://github.com/Malinon/tfbm
   pip install ./tfbm

Usage Example
--------------

.. code-block:: python

   from tfbm import TFBM1
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Create TFBM1 process
   H = 0.7        # Hurst parameter
   lambd = 0.5    # Tempering parameter  
   T = 1.0        # Time horizon
   N = 100       # Number of steps
   
   tfbm1 = TFBM1(T, N, H, lambd, method="davies-harte")
   
   # Generate single sample path
   path = tfbm1.generate_samples(num_of_samples=1)[0]
   
   # Plot result
   plt.plot(tfbm1.ts, path)
   plt.xlabel('Time')
   plt.ylabel('TFBM I Value')
   plt.title(f'TFBM I (H={H}, λ={lambd})')
   plt.show()

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:
   
   installation
   api
   theory
   numerical

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API:
   
   modules/tfbm
   modules/tfbm1
   modules/tfbm2  
   modules/tfbm3
   modules/tests


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
----------

1. Sabzikar, F., Meerschaert, M. M. (2013). "Tempered fractional Brownian motion." *Statistics & Probability Letters*, 83(10), 2269-2275.
2. Sabzikar F., Surgailis D. (2018). "Tempered fractional Brownian and stable motions of second kind" *Statistics & Probability Letters*, 132(10), 2269-2275.
3. Molina-Garcia, D., Sandev, T., Safdari, H., Pagnini, G., Chechkin, A., Metzler, R. (2018). "Crossover from anomalous to normal diffusion: truncated power-law noise correlations and applications to dynamics in lipid bilayers." *New Journal of Physics*, 20