'''
Validate lens lens.

Compare the optical path length (OPL) at lens second vertex to the analytic OPL for plano-convex lens.

Analytic OPL is for a plane-wave ray at initial height h.

04/05/2023
Steven Sheppard
'''
import model_tools.raytrace as rt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


import matplotlib as mpl
from_list = mpl.colors.LinearSegmentedColormap.from_list
mpl.use("TkAgg")

import gc as gc
from pathlib import Path
import numpy as np
from matplotlib.widgets import Slider

root_dir = Path(r'C:\Users\Steven\Documents\qi2lab\github\SPIM_model\data')
save_dir = Path('lens_testing')
savedir = rt.get_unique_dir(root_dir, save_dir)

savefig = True
showfig = True

#------------------------------------------------------------------------#
# analytic functions for OPL
#------------------------------------------------------------------------#

def plano_convex_path_lenght(h, lens):
    '''
    :param float h: initial height
    :param class lens: Thick_lens() plano-convex
    :return float l: Plano-convex path length at V2
    '''
    R = np.abs(lens.r2)
    to = lens.t0
    rmax = lens.aperture
    n = lens.ri

    l = n * (R - np.sqrt(R**2 - rmax**2) + to) \
          - n * (R - np.sqrt(R**2 - h**2)) \
          + (R - np.sqrt(R**2 - h**2)) / (np.sqrt(1 - (n*h/R)**2) * np.sqrt(1 - (h/R)**2) + n*(h/R)**2)
    return l


def plano_convex_pathlength_quad(h, lens):
    '''
    :param float h: initial height
    :param class lens: Thick_lens() initiated using create_lens()
    :return float opl: Plano-convex optical path length at V2 up to quadratic term
    '''
    R = np.abs(lens.r2)
    to = lens.t0
    rmax = lens.aperture
    n = lens.ri

    opl = n * (R - np.sqrt(R**2 - rmax**2) + to) - (n-1) * h**2 / (2*R)
    return opl


def lens_opl_spherical(h, lens):
    '''
    :param float h: initial height
    :param class lens: Thick_lens() initiated using create_lens()
    :return float opl: Plano-convex optical path length at V2
    '''
    R = lens.r2
    to = lens.t0
    rmax = lens.aperture
    n = lens.ri

    opl = (n-1) * h**4 * (1/(8*R**3) - (n-1) / (4*R**3))
    return opl

def model(lens_radius=-100,
          lens_ri=1.3,
          lens_min_thickness=5,
          lens_diameter=50.8):

    #------------------------------------------------------------------------#
    # setup lens model
    #------------------------------------------------------------------------#

    # lens defocus is inversely proportional to the focal length
    lens_dpt =  - (lens_ri-1) * 10**3 / lens_radius # from lens formula

    # Create lens lens
    lens = rt.create_etl(dpt=lens_dpt, z1=0.0, d=lens_diameter, ri=lens_ri, t0=lens_min_thickness, ri_in=1.0, ri_out=1.0)
    lens.label = "lens"

    #------------------------------------------------------------------------#
    # raytrace lens
    n_rays = 51
    rays = rt.create_rays(type='flat_top',
                          source='infinity',
                          n_rays=n_rays,
                          diameter=lens_diameter)

    rays = lens.raytrace(rays)

    # raytrace to lens second vertex
    rays = rt.intersect_plane(rays, zf=lens.z2, ri_in=1.0, refract=False)

    return rays, lens


etl_ri = 1.3
etl_min_thickness = 5.0
etl_radius = -100.0
rays, lens = model(lens_radius = etl_radius,
                   lens_ri=etl_ri,
                   lens_diameter=25,
                   lens_min_thickness=etl_min_thickness)

# fit at lens V2
wf_fit_v2 = rt.ray_opl_polynomial(rays, 1, method='opl')

# extract radius of curvature from wf_fit coefficients
# reference the analytic form for OPL through plano-convex lens
radius_from_c2 = (etl_ri - 1) / (2*wf_fit_v2[2])
radius_from_c4 = np.cbrt((etl_ri - 1) / (8*wf_fit_v2[4]))

# Compile string for plot titles
fit_str = ''
for fit in wf_fit_v2:
    fit_str += f' {fit:.1e} '
fit_labels = [f"$C_{ii}$" for ii in range(9)]

#------------------------------------------------------------------------#
# Plot results
#------------------------------------------------------------------------#

# Create 1x4 grid with padding
fig = plt.figure(figsize=(22, 10))
grid = fig.add_gridspec(nrows=2, ncols=7, width_ratios=[1, 0.01, 1, 0.01, 1, 0.01, 1], height_ratios=[0.01,1], wspace=0.2)
fig.suptitle(f'lens R={etl_radius} test\n'
             + f'Curvature guess: C2~{np.round(radius_from_c2, 2)}, C4~{np.round(radius_from_c4, 2)}\n'
             + f'Fit results:{fit_str}')

# Plot ray trace to fit plane
ax = fig.add_subplot(grid[1,0])
ax.set_title('lens raytrace')
ax.set_ylabel('r (mm)')
ax.set_xlabel('z (mm)')

# Plot rays
n_plot = 25
skp_idx = int(rays[0].shape[0] // n_plot)
ax.plot(rays[:, ::skp_idx, 2], rays[:, ::skp_idx, 0], 'r')

# show lens surfaces
x = np.linspace(lens.z1+lens.t0, lens.z2, 1000)
a = lens.r2 * np.sqrt(1 - (lens.aperture / lens.r2)**2) + lens.t0
lens_curvature = np.sqrt(lens.r2**2 - (x - a)**2)

ax.plot(np.linspace(lens.z1, (lens.z1+lens.t0), 100), [lens.aperture]*100, c='k')
ax.plot(np.linspace(lens.z1, (lens.z1+lens.t0), 100), [-lens.aperture]*100, c='k')
ax.plot([lens.z1]*2, [-lens.aperture, lens.aperture], c='k')
ax.plot(x, lens_curvature, c='k', label='lens surface')
ax.plot(x, -lens_curvature, c='k')
ax.axvline(x=lens.z2, c='c', linestyle='--', label='Fit plane')
ax.legend(loc='lower center', framealpha=0.95)
ax.set_aspect('equal')

# Plot fit summary at v2
ax = fig.add_subplot(grid[1,2])
ax.set_title('Fit summary')
ax.set_ylabel(r'$|C_i|$')
ax.set_xlabel('Fit Coeff', rotation='horizontal')
ax.set_yscale('log')
ax.bar((np.arange(len(wf_fit_v2))), np.abs(wf_fit_v2), color='m', tick_label=fit_labels)

# Plot fit vs raw opld
# Calculate the fit
fit = 0
for ii in range(len(wf_fit_v2)):
    fit += rays[-1,:,0]**ii * wf_fit_v2[ii]

# subtract off the spherical contribution
fit_no_spher = fit - wf_fit_v2[4] * rays[-1,:,0]**4

# Calculate fit error from raytracing OPL
fit_error = fit - rays[-1,:,3]
fit_sqerror = np.square(fit_error) # squared errors
fit_mnsqerror = np.mean(fit_sqerror) # mean squared errors
fit_rmse = np.sqrt(fit_mnsqerror) # Root Mean Squared Error, RMSE
fit_rsq = 1.0 - (np.var(fit_error) / np.var(rays[-1,:,3]))

sph_error = (rays[-1,:,3] - fit_no_spher) - wf_fit_v2[4] * rays[-1,:,0]**4
sph_sqerror = np.square(sph_error) # squared errors
sph_mnsqerror = np.mean(sph_sqerror) # mean squared errors
sph_rmse = np.sqrt(sph_mnsqerror) # Root Mean Squared Error, RMSE
sph_rsq = 1.0 - (np.var(sph_error) / np.var((rays[-1,:,3] - fit_no_spher)))

# show fit vs raw opl
ax = fig.add_subplot(grid[1,4])
ax.set_title(f'Fit vs Raw OPLD\nRMSE:{fit_rmse:.2e}, R2:{fit_rsq:.2e}', fontsize=11)
ax.set_xlabel(xlabel='Initial Ray height (mm)', fontsize=11)
ax.set_ylabel(ylabel='OPL (mm)')
ax.plot(rays[-0,:,0], rays[-1,:,3], c='b', marker='x', label='Raw')
ax.plot(rays[-0,:,0], fit, c='r', marker='+', label='Fit')
ax.plot(rays[-0,:,0], plano_convex_path_lenght(rays[0, :, 0], lens), c='g', label='Analytic')
ax.legend()

# show spherical fit vs raw opl - fit with no spherical
ax = fig.add_subplot(grid[1,6], sharex=ax)
ax.set_title(f'Spherical Contribution\nRMSE:{sph_rmse:.2e}, R2:{sph_rsq:.2e}', fontsize=11)
ax.set_xlabel('Initial Ray height (mm)', fontsize=11)
ax.set_ylabel('Spherical')
ax.plot(rays[-1,:,0], rays[-1,:,3] - plano_convex_pathlength_quad(rays[0, :, 0], lens), 'bx', label="OPL - quadratic OPL")
ax.plot(rays[-1,:,0], lens_opl_spherical(rays[0, :, 0], lens), 'g', label="Analytic Spherical OPL")
ax.legend()

plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.865)

if savedir:
    fig.savefig(savedir / Path('plano_convex_summary.pdf'))

if showfig:
    plt.show()
else:
    plt.close('all')


#------------------------------------------------------------------------#
# Plot interactive ray tracing
#------------------------------------------------------------------------#
n_rays = 25
min_radius = -1000
max_radius = -30

def update_lens(val):
    curvature = radius_slider.val

    rays, lens = model(lens_radius=curvature,
                       lens_ri=etl_ri,
                       lens_min_thickness=etl_min_thickness,
                       lens_diameter=50.0)

    focal_planes, rays = rt.ray_focal_plane(rays=rays, ri=1.0, method='all', return_rays=True)

    axial_extent = focal_planes[2] - focal_planes[0]

    # Clear the current plot
    ax.clear()

    # Plot the ray tracing
    rt.plot_rays(rays=rays,
                 n_rays_to_plot=n_rays,
                 title=lens.label,
                 optical_train=[lens],
                 planes_of_interest={"paraxial f.p.": focal_planes[0],
                                     "midpoint f.p.": focal_planes[1],
                                     "marginal f.p.": focal_planes[2]},
                 show_focal_planes=False,
                 ax=ax)

    ax.set_title(f"R={lens.r2:.2f}mm, axial extent={axial_extent:.3f}mm")

    # ax.set_xlim(560, 620)
    # ax.set_ylim(-25, 25)

    # Update the plot
    plt.draw()


# Create figure and the initial plot
fig = plt.figure(figsize=(20,8))

grid = fig.add_gridspec(nrows=2,
                        ncols=1,
                        height_ratios=[1,0.1],
                        hspace=0.3)
ax = fig.add_subplot(grid[0])
ax.set_aspect("equal")
# Add a slider for the cuvette position
ax_slider = fig.add_subplot(grid[1])
radius_slider = Slider(ax=ax_slider,
                       label='R2',
                       valmin=min_radius,
                       valmax=max_radius,
                       valinit=-500,
                       valstep=-1)

radius_slider.on_changed(update_lens)

update_lens(-500)

plt.show()
