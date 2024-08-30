# SPIM_model
Tools for performing 1D ray tracing, 3D field propagation, and wavefront analysis.

- Rays are explicitly traced and refracted at each element surface.
- The field propagation model uses the exact transfer function method (Goodman, pg. 140).
- Includes wavefront fitting and pupil decomposition.

## Installation
To install the package, use the following code snippet. If you want to edit the package after installing, include the `-e` option.

```
git clone
cd SPIM_model
pip install .
```
```
git clone
cd SPIM_model
pip install -e .
```
## Intended use
Ray tracing and field propagation can be used separately or combined. The typical workflow involves setting up an optical train, ray tracing, extracting aberrations, propagating, and calculating the 3D electric field distribution. Example code is available in the extFOV_SPIM directory.
### Ray Tracing
Available optical elements include 2-surface thick lenses, 3-surface doublet lenses, and perfect lenses for simulating objectives. Rays are characterized by their distance from the optical axis, angle off the optical axis, position along the optical axis, and optical path length. Convenient helper functions are available for determining focal planes, longitudinal spherical aberration, and plotting results.
### Propagation
Ray tracing and propagation are separate processes. The propagation package requires an NxN electric field array. To propagate the results from ray tracing, the ray's parameters are converted to an amplitude and phase, which are interpolated into an NxN field array.

Common issues with field propagation:
1. Scratchy or low-intensity regions: This could indicate poor ray flux sampling. Try increasing the number of initial rays or changing the amplitude binning method.
2. Spotty or uniform caustic or phase angle: This may occur if the propagation grid is not near the focus or if the sampling in k-space is incorrect. For large aberrations, there may be a significant difference between the midpoint focal plane and the diffraction focal plane. This can cause the propagated region to miss the focal plane area. Try using the diffraction focal plane and increasing the search parameters.


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12752313.svg)](https://doi.org/10.5281/zenodo.12752313)
