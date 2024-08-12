# SPIM_model
Model tools to perform 1d ray tracing and field propagation.
- Rays are explicitly traced and refracted at each element surface.
- The field propagation model applies the exact transfer function method (Goodman, pg140).

## Installation
Model tools can be used by manually importing *raytracing* and *propagation* scripts or by installing as a python package.
To install as a package use the following code snippet. In order to edit the package after installing, include *-e* option.

TODO: Modify package name and update documentation.
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
Ray Tracing and field propagation can be used seperately or combined. The intended workflow is to setup an optical train, raytrace, extract aberrations, propagate and calculate 3-d electric field distribution.

### Ray Tracing
Optics are classes and compiled using surface and material properties. Rays are characterized by the distance from the optical axis, angle off the optical axis, position along optical axis and the optical path length.

### Propagation
Ray tracing in not required to propagate an electric field, it works for any NxN field array. In order to propagate the results from ray tracing, the ray's parameters are converted to an amplitude and phase which is interpolated to an NxN field array.

Common issues with field propagation are: (should include examples)
1. If the intensity profile is scratchy or has regions of no intensity, this could be a sign of poor sampling, try generating more rays.
2. If the resulting caustic or phase angle appears spotty and uniform, the propagation grid is not near the focus or the sampling in kspace is wrong.


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12752313.svg)](https://doi.org/10.5281/zenodo.12752313)
