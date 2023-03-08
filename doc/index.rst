demcmc
======

A package for estimating differential emission measures (DEMs) from observations of the Sun using Monte Carlo methods.

``demcmc`` provides an interface for collecting information on emission lines together in one place (a list of `EmissionLine` objects), and running a DEM inversion on them (using :func:`~demcmc.predict_dem_emcee`).
Each `EmissionLine` object contains the intensity of an observed line, and the contribution function of that line.
The inversion is carried out using Markov chain Monte carlo (MCMC) methods, with the `emcee package <https://emcee.readthedocs.io/en/stable/>`__ used to run the sampling.

``demcmc`` does **not** make opinionated choices on how to run the the MCMC algorithm.
Users are forced to choose the number of MCMC steps and walkers, and are given the outputs needed to evaluate whether their choices are appropriate.
Before using this package it is highly recommended to read the `emcee paper <https://doi.org/10.48550/arXiv.1202.3665>`__ to understand how the sampling works, and what to choose for the MCMC parameters.



.. toctree::
   :maxdepth: 1

   installing
   _auto_examples/plot_real
   api/index
   explanation/dem_theory
   explanation/other_tools
