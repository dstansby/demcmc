# DEM theory

This page gives a short theoretical overview of differential emission measures (DEMs).

## Background
The background theory is taken from [Del Zanna & Mason].
The intensity if light detected from an optically thin plasma is ([Del Zanna & Mason] eq. 5):

```{math}
I \left ( \lambda \right ) = \int_{s} Ab(Z) C \left ( n_{e}, T, \lambda \right )~n_{e}n_{H}~ds
```
where

- {math}`I \left ( \lambda \right )` is the observed intensity as a function of wavelength {math}`\lambda`.
- {math}`C \left ( n_{e}, T, \lambda \right )` is the contribution function of a plasma at a given temperature. This hides all the atomic physics.
- {math}`Ab(Z) = n_{Z} / n_{H}` is the abundance of the element that is emitting, relative to the hydrogen abundance.
- {math}`n_{e}` is the electron density.
- {math}`n_{H}` is the hydrogen density.
- The integral is taken along a line of sight {math}`ds`.

If there is a unique mapping from {math}`n_{e}` and {math}`T` then the differential emission measure {math}`DEM(T)` can be defined as ([Del Zanna & Mason] eq. 88)
```{math}
\int_{T} DEM(T) dT = \int_{h} n_{e}n_{H} dh
```

Substituting this into the equation for intensity and assuming the chemical abundance is constant along the line of sight we get
```{math}
I \left ( \lambda \right ) = Ab(Z) \int_{T} C \left ( n_{e}, T, \lambda \right )~DEM(T)~dT
```

If we had the perfect telescope that just observed a single wavelength this equation would be helpful.
Sometimes this is a good approximation for diffraction spectrometers (e.g. Hinode EIS), but in general and in other cases (e.g. SDO AIA) we have to take into account that the intensity measured is an integral of the full spectrum multiplied by the transmission function {math}`T(\lambda)`.
The intensity measured by a given telescope or channel is

```{math}
I = \int_{\lambda} T (\lambda) I(\lambda) d\lambda \\
I =  \int_{T} \left [ \int_{\lambda} T (\lambda) Ab(Z) C \left ( n_{e}, T, \lambda \right ) d\lambda \right ]~DEM(T)~dT
```
The term in square brackets is a function of {math}`T` only, and called the temperature response function of the telescope or specific channel.
Calculating it requires adding up the contribution from every line that falls in areas of high transmission function values.

[Del Zanna & Mason]: https://link.springer.com/article/10.1007/s41116-018-0015-3

## Estimating a DEM
To estimate {math}`DEM(T)` we therefore need to invert the previous equation.
To do that we need two inputs as data:
- {math}`I \left ( \lambda \right )`: the intensity as a function of wavelength for different atomic transitions.
- {math}`C \left ( \lambda; n_{e}, T \right )`: the contribution function to a given emission line as a function of density and temperature.


## MCMC methods

This page gives a brief overfiew of MCMC methods used to estimate a DEM.
The theory is taken from [Kashyap & Drake 1998](https://iopscience.iop.org/article/10.1086/305964/pdf).

The general idea is to obtain the most probable set of DEM values that describe a set of observed intensities.
This method starts at Bayes theorem.
For a set of observed intensities, {math}`D = \{d_{i}, i=1...n\}`, the probability of a given set of DEM values, {math}`\Theta = \{ \theta_{j}, j=1...m\}` is

```{math}
p \left ( \Theta | D \right ) = p \left (\Theta \right) ~~ \Pi_{i=1}^{n} p \left (d_{i} | \Theta \right )
```
where
- {math}`p \left ( \Theta \right )` is the prior probability distribution of the model values.
- {math}`p \left (d_{i} | \Theta \right )` is the probability of observing a given intensity for a set of model values.

For an observed intensity of {math}`I_{i} \pm \sigma_{i}` the probability of observing is modelled as a Gaussian

```{math}
p \left (d_{i} | \Theta \right ) = \exp -  \left [ (I_{i} - I_{pred, i}(\Theta ))^{2} / \sigma_{i}^{2} \right ]
```

The predicted intensity is given by
```{math}
I_{pred, i} =  \int_{T} \left [ Ab(Z_{i}) C_{i} \left ( n_{e}, T \right ) \right ]~DEM(\Theta(T))~dT
```

In this equation the electron density and elemental abundances are kept fixed.
