# tfbm

Package for simulating tempered fractional Brownian motions described in [1], [2], [3].



## TFBM I

The stochastic processes $B^{I}_{H,\lambda} = \{B^{I}_{H,\lambda}(t)\}_{t\in\R}$ defined by the Wiener integral
$$
B^{I}_{H,\lambda}(t) := \int_{\R} g^{I}_{H,\lambda,t}(s) \,dB_{s},
$$
where
$$
g^{I}_{H,\lambda,t}(s) := \left[(t - s)_{+}^{H- \frac{1}{2}} e^{\lambda(t-s)_{+}} - (-s)_{+}^{H- \frac{1}{2}} e^{\lambda(-s)_{+}}\right], \quad s \in \R
$$
is called a tempered fractional Brownian motion (TFBM I)

## TFBM II
The stochastic processes $B^{II}_{H,\lambda} = \{B^{II}_{H,\lambda}(t)\}_{t\in\R}$ defined by the Wiener integral
$$
B^{II}_{H,\lambda}(t) := \int_{\R} g^{II}_{H,\lambda,t}(s) \,dB_{s},
$$
where
$$
g^{II}_{H,\lambda,t}(s) := (t - s)_{+}^{H- \frac{1}{2}} e^{\lambda(t-s)_{+}} - (-s)_{+}^{H- \frac{1}{2}} e^{\lambda(-s)_{+}} + \lambda \int_{0}^{t} (u - s)_{+}^{H- \frac{1}{2}} e^{\lambda(u-s)_{+}} du, \quad s \in \R,
$$
is called a tempered fractional Brownian motion of the second kind (TFBM II).

## TFBM III

We consider the overdamped stochastic equation of motion of a particle in a viscous medium under the influence of a stochastic force $\xi(t)$. A stochastic process $B^{III}_{H,\lambda} = \{B^{III}_{H,\lambda}(t)\}_{t\in\R}$ is called a tempered fractional Brownian motion of the third kind (TFBM III) if it satisfies the differential equation:
$$
\frac{dB^{III}_{H,\lambda}(t)}{dt} = \frac{\xi(t)}{m\eta} = \nu(t),
$$
where $m$ is the particle mass, $\eta$ the friction coefficient and $\nu(t)$ represents a velocity process with the autocorrelation function given by:
$$
\gamma_{H}(\tau) = \frac{1}{\Gamma(2H - 1)} \tau^{2H-2}e^{-\tau /\tau^{*}}, \quad \tau > 0,
$$
where $\tau^* > 0$ is a characteristic crossover time scale, and the Hurst parameter
satisfies $\frac{1}{2} \leq H < 1$. 

## Installation

You can install the package using following commands:

```bash
git clone https://github.com/Malinon/tfbm
pip install ./tfbm
```

## Example usage
```python
import tfbm

# Create generator of TFBM I 
tfbm1 = tfbm.TFBM1(H=10, T=10, N=500, lambd=0.5)
trajectories, increments = tfbm1.generate_samples(num_of_samples=100, get_increments=True)
```



## References

[1] M. M. Meerschaert and F. Sabzikar. Tempered fractional Brownian motion. Statistics & Probability Letters, 83(10):2269–2275, 2013. doi:10.1016/j.spl.2013.06.016 <br>
[2] F. Sabzikar and D. Surgailis. Tempered fractional Brownian and stable motions of second kind. Statistics & Probability Letters, 132:17–27, 2018. doi: 10.1016/j.spl.2017.08.015. <br>
[3] D. Molina-Garcia, T. Sandev, H. Safdari, G. Pagnini, A. Chechkin, and R. Metzler. Crossover from anomalous to normal diffusion: truncated power-law noise correlations and applications to dynamics in lipid bilayers. New Journal of Physics 20, page 103027, 2018. doi: 10.1088/1367-2630/aae4b2. URL https://doi.org/10.1088/1367-2630/aae4b2.
