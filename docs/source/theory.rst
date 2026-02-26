Mathematical Theory
===================

This section provides the mathematical background behind tempered fractional Brownian motions and statistical tests for discriminating between TFBM and other stochastic processes.

Let :math:`B = \{B_s, s \in \mathbb{R}\}` be a two-sided Wiener process.

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


Statistical Tests for Detecting TFBM
--------------------------------------------

The package includes statistical tests based on quadratic form statistics two-sided test with the following hypotheses:

*Null hypothesis:* :math:`H_0: B^J_{H,\lambda}` is a trajectory of TFBM of type :math:`J` withparameters :math:`H` and :math:`\lambda`

*Alternative hypothesis:* :math:`H_1: B^J_{H,\lambda}` is not a trajectory of TFBM of type :math:`J` with parameters :math:`H` and :math:`\lambda`.

3 specific tests for TFBM corresponding to time average mean-squared displacement, sample autocovariance function and detrended moving average are implemented
Definitions of these statistics are presented below:

.. math::
   
   \text{TAMSD}(n) = \frac{1}{N-n} \sum_{i=1}^{N-n} (X_{i+n} - X_i)^2

   \text{ACF}(n) = \frac{1}{N-n} \sum_{i=1}^{N-n} (X_{i+n} - \bar{X})(X_i - \bar{X})

   \text{DMA}(n) = \frac{1}{N-n} \sum_{i=1}^{N-n} (X_i - \tilde{X}_i(n))^2

where :math:`\tilde{X}_i(n)` is the moving average of the process defined as:

.. math::

   \tilde{X}_i(n) = \frac{1}{n} \sum_{j=0}^{n-1} X_{i+j}

Discussion of these test methodology and their performance in general can be found in [4]. Article [5] provides detailed discussion of the performance of these tests for TFBM.

References
----------

1. Sabzikar, F., Meerschaert, M. M. (2013). "Tempered fractional Brownian motion." *Statistics & Probability Letters*, 83(10), 2269-2275.
2. Sabzikar F., Surgailis D. (2018). "Tempered fractional Brownian and stable motions of second kind" *Statistics & Probability Letters*, 132(10), 2269-2275.
3. Molina-Garcia, D., Sandev, T., Safdari, H., Pagnini, G., Chechkin, A., Metzler, R. (2018). "Crossover from anomalous to normal diffusion: truncated power-law noise correlations and applications to dynamics in lipid bilayers." *New Journal of Physics*, 20
4. Balcerek M., Burnecki K., Sikora G., and Wyłomańska A. (2021). "Discriminating Gaussian processes via quadratic form statistics." *Chaos*, 31(6):063101
5. Macioszek K. Sabzikar F., Burnecki K. (2025). "Testing of tempered fractional Brownian motions" *BioPhysMath*. 1-23