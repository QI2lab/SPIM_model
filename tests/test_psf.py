'''
Test PSF simulation.

Iterate through NAs and compare to theoretical width values at the focus.

06/09/2023
Steven Sheppard
'''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from model_tools.analytic_forms import airydisk_2d, fwhm, gaussian_intensity
from model_tools.analysis import fit_gaussian_1d
import model_tools.propagation as pt
import model_tools.raytrace as rt

try:
    import cupy as cp
except ImportError:
    cp = np

showfig = True

# Unit conversion, um -> mm: * , mm -> um: /
mm = 1e-3

root_dir = Path('/home/sjshepp2/Documents/github/raytracing_sjs/scripts/data') # server
savedir = rt.get_unique_dir(root_dir, Path('test_psf'))


#------------------------------------------------------------------------------#
# Setup for propagation
wl = 0.561 * mm
ko = 2 * np.pi / wl
initial_propagation_diameter = 200 * mm
propagation_length = 150 * mm
dx = wl / 2
dz = wl * 2
n_xy = int(initial_propagation_diameter // dx)
n_zx = int(propagation_length // dz)

# Check if n_grid is odd
if n_xy%2 == 0:
    n_xy += 1
if n_zx%2 == 0:
    n_zx += 1

#------------------------------------------------------------------------------#
# Setup psf params
nas = [0.14, 0.2]
pupil_radii = [5.6, 4.0]
fs = [pr / na  for na, pr in zip(nas, pupil_radii)]

#------------------------------------------------------------------------------#
# simulate psf and plot results
for na, f in zip(nas, fs):
    psf_field, psf_grid_params = pt.field_grid(num_xy=n_xy, num_zx=n_zx, dx=dx, dz=dz)
    x, radius_xy, extent_xy, z, extent_zx = psf_grid_params

    psf_field = pt.model_psf(na=na,
                             f=f,
                             wl=wl,
                             grid_params=psf_grid_params,
                             beam_type='gaussian',
                             power=0.1)
    zx_projection = pt.xz_projection(psf_field)

    pt.plot_field(fields=[psf_field[n_zx//2]],
                  field_labels=[f'focal plane {na}'],
                  grid_params=[psf_grid_params],
                  save_path=savedir / Path('field.png'),
                  showfig=showfig)

    pt.plot_xz_projection(fields=[psf_field],
                          field_labels=[f'{na}'],
                          grid_params=[psf_grid_params],
                          save_path=savedir / Path('projection.png'),
                          showfig=showfig)

    # Cacluate analytic psf, air disk
    airy_ffp_psf = airydisk_2d(radius_xy, na, wl)
    airy_slice = airy_ffp_psf[int(n_xy//2)]
    airy_fwhm = (0.51 * wl) / na

    # fit 2 gaussian and calculate FWHM
    gauss_fit = fit_gaussian_1d(data=np.abs(psf_field[int(n_zx//2),
                                                      int(n_xy//2), :])**2,
                                r_idxs=x,
                                DEBUG_fit=False)['fit_params']

    fwhm_fit = fwhm(gauss_fit[0]/mm) # Um

    # Grab intensity slice that was fit, and normalize
    psf_ffp_intensity = np.abs(psf_field[int(n_zx//2), int(n_xy//2), :])**2
    psf_ffp_intensity = psf_ffp_intensity / np.max(psf_ffp_intensity)

    # Create figure fo
    fig, ax = plt.subplots(1,1,figsize=(10,6), tight_layout=True)
    fig.suptitle(f'PSF Fit comparison, NA:{na:.1f}')
    ax.set_title(f'Airy FWHM:{(airy_fwhm/mm):.2f}um, gaussian fit FWHM:{fwhm_fit:.2f}um')
    ax.set_ylabel('I(r)')
    ax.set_xlabel(r'radius ($\mu m$)')
    ax.set_xlim(-4*fwhm_fit, 4*fwhm_fit)

    w, Io, mu, offset = gauss_fit
    ax.plot(x/mm,
            psf_ffp_intensity/np.max(psf_ffp_intensity),
            c='b', linestyle='-', label='Data')
    ax.plot(x/mm,
            gaussian_intensity(x, w, Io, mu, offset)/np.max(gaussian_intensity(x, w, Io, mu, offset) ),
            c='r', linestyle='--', marker='x', label='Gaussian fit')
    ax.plot(x / mm, airy_slice, c='g', linestyle='--', label='Airy')
    ax.axhline(y=0.5, c='k', linestyle='--')
    ax.legend()

    plt.savefig(savedir / Path(f'psf_slice_fit_results_na_{na:.2f}.pdf'))

if showfig:
    plt.show()
else:
    plt.close('all')
