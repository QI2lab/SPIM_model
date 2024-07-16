# -*- coding: utf-8 -*-
'''
Compare analytic Gaussian field to numerical propagation results.
The numerical model starts with the same analytic Gaussian electric field and propagates there after using the
exact-transfer method. At each z-plane, the analytic counterpart is also calculated.
To validate the numerical propagation, we plot the resulting intensities and their difference.

Steven Sheppard
04/05/2023
'''
# model imports
import model_tools.analytic_forms as gt
import model_tools.propagation as pt
import model_tools.raytrace as rt
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

root_dir = Path('/home/steven/Documents/qi2lab/github/raytracing_sjs/data')
save_dir = Path('gaussian_prop')
savedir = rt.get_unique_dir(root_dir, save_dir)

# Toggle figure saving and showing.
savefigs = True
showfigs = True

# Toggle plots
plot_phase = True
plot_projections = True
plot_comparison = False

#------------------------------------------------------------------------------#
# Model parameters
wl = 0.5
NA = 0.14
Io = 1.
ri = 1.
ko = 2 * np.pi / wl

# Must specify the desired waist.
wo = gt.gauss_waist(wl=wl, ri=1.0, na=NA)
zR = gt.gauss_rayleigh(wl=wl, wo=wo, ri=ri)

z_initial = -4 * zR
initial_propagation_diameter = 250
propagation_length = 5*np.abs(zR)
dx = wl / 2
dz = wl
n_xy = int(initial_propagation_diameter // dx)
n_zx = int(propagation_length // dz)

# Check if n_grid is odd
if n_xy%2 == 0:
    n_xy += 1
if n_zx%2 == 0:
    n_zx += 1

field_numerical, mask_params = pt.field_grid(n_xy, n_zx, dx, dz)
x, radius_xy, extent_xy, z, extent_zx = mask_params

#------------------------------------------------------------------------------#
# Model Gaussian beam
# Create empty field for analytic solution.
field_analytical = np.copy(field_numerical)

# Create the initial field using the analytic Gaussian electric field
initial_gaussian_field = gt.gaussian_field(radius_xy, z_initial, wl, wo, ri, Io)

# Propagate using exact transfer function
field_numerical = pt.get_3d_field(initial_gaussian_field,
                                  z=z+z_initial,
                                  wl=wl,
                                  dx=dx,
                                  ri=1.0)
zx_numerical = pt.xz_projection(field_numerical)

# Calculate fields using analytic form
for ii, zpos in enumerate(tqdm(z, desc='Calculating analytic field')):
    field_analytical[ii] = gt.gaussian_field(radius_xy, zpos, wl, wo, ri, Io)
zx_analytical = pt.xz_projection(field_analytical)

#------------------------------------------------------------------------------#
# Plot results
plot_phase = True
plot_projections = True
plot_comparison = False

# Plot numerical gaussian
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), sharex='all', sharey='all')
fig.suptitle('ZX Projections', fontsize=16)

ax.set_ylabel(r'x ($\mu m$)')
im = ax.imshow(zx_numerical,
                cmap='hot',
                extent=extent_zx,
                origin='lower',
                # aspect='auto',
                aspect=dx/dz,
                interpolation='gaussian')
ax.tick_params(axis='both', direction='inout')

# Only add to bottom axes plot.
ax.set_xlabel(r'z ($\mu m$)', fontsize=13)

# Add colorbar, shared for each axes
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cbar_ax)

if plot_projections:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex='all', sharey='all')
    fig.suptitle('ZX Projections', fontsize=16)

    for ax, caust, label in zip(axes.flat,
                                [zx_analytical, zx_numerical],
                                ['Analytic model', f'Gaussian beam, $\omega_0=${wo}$\mu m$, $z_R=${np.round(zR, 2)}$\mu m$']):

        ax.set_title(label, fontsize=14)
        ax.set_ylabel(r'x ($\mu m$)')
        im = ax.imshow(caust,
                        cmap='hot',
                        extent=extent_zx,
                        origin='lower',
                        aspect='auto',
                        interpolation=None)

        ax.tick_params(axis='both', direction='inout')
        ax.set_ylim((-5*wo, 5*wo))

    # Only add to bottom axes plot.
    ax.set_xlabel(r'z ($\mu m$)', fontsize=13)

    # Add colorbar, shared for each axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.1, 0.03, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    if savefigs:
        plt.savefig(savedir / Path('zx_projections'))

if plot_comparison:
    # Plot difference in analytic and numerical intensities
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex='all', sharey='all')
    fig.suptitle('Analytic vs Numerical model comparison', fontsize=16)

    ax = axs[0]
    ax.set_title('Analytic Gaussian', fontsize=14)
    ax.set_ylabel(r'x ($\mu m$)')
    im = ax.imshow(zx_analytical,
                cmap='hot',
                extent=extent_zx,
                origin='lower',
                aspect='auto',
                interpolation=None)
    fig.colorbar(im, ax=ax)

    ax = axs[1]
    ax.set_title('Analytics - Numerical', fontsize=14)
    ax.set_ylabel(r'y ($\mu m$)', fontsize=13)
    ax.set_xlabel(r'x ($\mu m$)', fontsize=13)
    im = ax.imshow(np.abs(zx_analytical - zx_numerical),
                cmap='hot',
                extent=extent_zx,
                origin='lower',
                aspect='auto',
                interpolation=None)
    fig.colorbar(im, ax=ax)

    if savefigs:
        plt.savefig(savedir / Path('zx_difference_comparison'))

    # Compare intensities at symmetric field planes
    fig, axs = plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(8, 10),
                            sharex='all', sharey='all',
                            tight_layout=True)
    fig.suptitle('Difference in symmetric planes', fontsize=16)

    ax = axs[0]
    ax.set_title('I(-z)', fontsize=14)
    ax.set_ylabel(r'y ($\mu m$)', fontsize=13)
    im = ax.imshow(np.abs(initial_gaussian_field)**2,
                cmap='hot',
                extent=extent_xy,
                origin='lower')
    fig.colorbar(im, ax=ax)

    ax = axs[1]
    ax.set_title('I(-z) - I(z)', fontsize=14)
    ax.set_ylabel(r'y ($\mu m$)', fontsize=13)
    ax.set_xlabel(r'x ($\mu m$)', fontsize=13)

    im = ax.imshow((np.abs(initial_gaussian_field)**2 - np.abs(field_numerical[-1])**2),
                cmap='hot',
                extent=extent_xy,
                origin='lower',
                aspect=dx/dz)
    fig.colorbar(im, ax=ax)
    if savefigs:
        plt.savefig(savedir / Path('xy_difference_comparison'))

if plot_phase:
    # ZX phase angle, depicts wavefront
    fig, axs = plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(10, 8),
                            sharex='all', sharey='all',
                            tight_layout=True)
    fig.suptitle('Field phase angle', fontsize=16)

    ax= axs[0]
    ax.set_title('Numerical propagation', fontsize=14)
    ax.set_ylabel(r'y $\mu m$', fontsize=13)
    im = ax.imshow(np.angle(field_numerical[:, n_xy//2, :].T),
                cmap='hot',
                extent=extent_zx,
                aspect='auto',
                origin='lower')

    fig.colorbar(im, ax=ax)

    ax = axs[1]
    ax.set_title('Analytic solution', fontsize=14)
    ax.set_ylabel(r'y $\mu m$', fontsize=13)
    ax.set_xlabel(r'x ($\mu m$)', fontsize=13)
    im = ax.imshow(np.angle(field_analytical[:, n_xy//2, :].T),
                cmap='hot',
                extent=extent_zx,
                aspect='auto',
                origin='lower')

    fig.colorbar(im, ax=ax)
    if savefigs:
        plt.savefig(savedir / Path('zx_phase'))

# show figures
if showfigs:
    plt.show()
else:
    plt.close('all')
