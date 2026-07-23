
from setuptools import setup, find_packages

setup(
    name='pytfbm',
    version='1.0.0',
    packages=find_packages(),
    install_requires=["numpy", "scipy", "mpmath"],
    description='Package for simulating tempered fractional Brownian motion',
    author='Jakub Malinowski',
    author_email='jakub.malinowski@pwr.edu.pl',
    url='https://github.com/Malinon/tfbm',
    license='MIT',
)
