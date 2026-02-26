TFBM Base Class
===============

Overview
--------

The :class:`tfbm.TFBM.TFBM` class serves as the abstract base class for all tempered fractional Brownian motion implementations. It provides common functionality and defines the interface that specific TFBM types must implement.

.. automodule:: tfbm.TFBM
   :members:
   :undoc-members:


Generation Methods
------------------

The TFBM class allows for generation of TFBM samples using Cholesky, Davies-Harte, Wood-Chan methods.
Details on these methods can be found in :doc:`../numerical`.


Class Hierarchy
---------------

The TFBM class hierarchy is structured as follows:

.. code-block:: text

   TFBM (abstract base class)
   ├── TFBM1 (implements TFBM Type I)
   ├── TFBM2 (implements TFBM Type II)  
   └── TFBM3 (implements TFBM Type III)

Subclass Requirements
~~~~~~~~~~~~~~~~~~~~~

Each TFBM subclass must:

   1. implement **ct_2(self, t)**: The function used for calculating covariance according to formula cov(X(t), X(s)) = 0.5 * (ct_2(t) + ct_2(s) - ct_2(|t-s|))
   2. define **cov_matrices_dir**: Directory for storing covariance matrices (if save_cov_matrix is True)
