# GeoGals: Geostatistic methods for the analysis of Galaxies

Benjamin Metha

Jaime Blackwell, Tree Smith, Qihan Zou

## Summary

* Data is getting better

## Statement of Need

1. Context
2. Benefits
3. Novelty

CONTEXT 

- spatially resolved data has never been better
- Programs like PHANGS, MAD, and TYPHOON are capable of resolving down to tens of parsec scales, probing the small and intermediate scale nature of the interstellar medium of galaxies. Understanding the physics of gas on these scales is crucial to understanding the physics and nature of stellar feedback — that is, the effects that the supernovae of massive stars have on preventing (or triggering) future generations of star formation. Understanding the effects that these small scale processes can have on the larger scale properties of a galaxy is an aspect of galaxy evolution that is still not well understood, and is a key focus of astronomical research.
- For this reason, it is desirable to use data modelling techniques that can extract the maximum amount of useful information about a galaxy, in particular about its behaviour at different spatial scales. 
- Methods from geostatistics (def) have been shown to be powerful data processing tools for galactic chemical data.
- Born in the 1950s, an integral pillar of spatial statistics (Cressie 93), and foundational for the (invention/theoretical underpinning) of Gaussian Processes (Rasmussen+Williams06), a collection of methods that bridge the gap between spatial statistics and machine learning (Hogg24?). As they are more interpretable than many other machine learning methods, their importance continues to be relevant to this day. 
- For data visualisation, semivariogram has proven to be a valuable tool. Its use has been demonstrated in Geogals1, in which a semivariogram fit to galaxy metallicity data from the —- survey revealed that these galaxies have chemical fluctuations of ~est dex, 100?s of times greater than the size of fluctuations estimated by an analytical model, suggesting an over-efficiency of the mixing processes that are occurring in this model.
- For fitting models that capture the two-(only 2?) dimensional structure within our data, geostatistical methods 
- inform physical processes that govern the spatial structure. 

%For example, fitting the best shape of the kernel informs on the nature of turbulence (add kernels and tutorial about this? honestly I should write a paper about this)

Many other Gaussian process modelling packages exist, some of which are even written by astronomers. **take notes from PyHiiExtractor intro**



Novelty:

* A fast semivariogram implementation **Figure: speed comparison. Q: Any existing semivariogram code?**
* Designed to deal with file conventions unique to astronomy, such as ‘.fits’ files (are they unique?). Corrects for effects of inclination using affine metrics, which is not addressed in other Python packages.
* Designed to be able to analyse a variety of spatially resolved galaxy data, such as star formation rates, gas densities, or chemical enrichment. 
* Uniquely placed to take advantage of kriging in order to fit models between different data gathered with different methods.

In addition to the general tools that we present in this package, we also present two subpackages specified to deal with two real-world data reduction challenges -- one for observational data captured by the PHANGS team\footnote{available for public download at link} and one for simulated data produced by the FIRE consortium.

This first release contains a single model for the mean, and for 