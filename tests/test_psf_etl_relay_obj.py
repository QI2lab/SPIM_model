# -*- coding: utf-8 -*-
'''
Test simulating a relay and objective optical train.

Ray tracing units are mm, propagation units are um.

TODO: update to use the current light sheet analysis.

Steven Sheppard
06/21/2023
'''
from pathlib import Path
import numpy as np

import model_tools.propagation as pt
import model_tools.raytrace as rt
import model_tools.analytic_forms as af
from model_tools.analysis import analyze_lightsheet_zx


root_dir = Path('/mnt/tilingspim/extFOV_results/debugging')
save_dir = Path('psf_relay_obj')
savedir = rt.get_unique_dir(root_dir, save_dir)

showfig = True

#------------------------------------------------------------------------------#
# Setup parameters
# Setup for propagation
wl = 0.000561
ko = 2 * np.pi / wl
initial_propagation_diameter = 0.350
propagation_length = 0.150
dx = wl / 2
dz = wl * 4
n_xy = int(initial_propagation_diameter // dx)
n_zx = int(propagation_length // dz)

# Check if n_grid is odd
if n_xy%2 == 0:
    n_xy += 1
if n_zx%2 == 0:
    n_zx += 1

# objective parameters
na=0.14
f = 40

# estimated parameters based on paraxial gaussian equation
best_waist = af.gauss_waist(wl=wl, ri=1.0, na=na)
best_length = 2 * af.gauss_rayleigh(wl=wl, na=na)

# setup a dictionary of function arguements for fitting the light sheet.
analysis_params = {"max_peak_ratio":0.1,
                   "peak_height_ratio":0.2,
                   "peak_distance":5,
                   "z_window":1,
                   "w_window":1,
                   "w_filter":[best_waist, 10],
                   "mu_filter": 0.1,
                   "plot_results":True,
                   "DEBUG_zstack":False,
                   "DEBUG_fits":False,
                   "showfig":showfig}

clear_aperture_508 = 45.72
clear_aperture_508 = 50.8

#------------------------------------------------------------------------------#
# Create ETL, using Optotune
etl_t0 = 5
etl_d = 16
etl_dpt = 0.5
etl_params = [-etl_t0, etl_dpt, etl_d, etl_t0, 1.3, 1.0, 1.0]
etl = rt.create_etl(*etl_params)

# Create lenses based on Thorlabs specification
# relay 1
ac508_180_a = rt.Doublet_lens(z1=0,
                              r1=109.7,
                              t1=12.0,
                              ri1= 1.5180,
                              r2=-80.7,
                              t2=2.0,
                              ri2= 1.6757,
                              r3=-238.5,
                              aperture_radius=(clear_aperture_508/2),
                              ri_in=1.0, ri_out=1.0)
ac508_180_a.label="ac508-180-A"

# relay 2 is backwards with infinity towards the objective
ac508_100_a_b = rt.Doublet_lens(z1=0,
                              r1=363.1,
                              t1=4.0,
                              ri1=1.7320,
                              r2=44.2,
                              t2=16.0,
                              ri2=1.6721,
                              r3=-71.1,
                              aperture_radius=(clear_aperture_508/2),
                              ri_in=1.0, ri_out=1.0)
ac508_100_a_b.label="ac508-100-A bckwrd"

# recreate lenses, reference the previous objects for calculated focal planes
relay1_z = np.abs(ac508_180_a.f1)
relay1 = rt.Doublet_lens(z1=relay1_z,
                         r1=ac508_180_a.r1,
                         t1=ac508_180_a.t1,
                         ri1=ac508_180_a.ri1,
                         r2=ac508_180_a.r2,
                         t2=ac508_180_a.t2,
                         ri2=ac508_180_a.ri2,
                         r3=ac508_180_a.r3,
                         aperture_radius=ac508_180_a.aperture,
                         ri_in=1.0, ri_out=1.0)
relay1.label = 'relay1 f180'

relay2_z = (relay1.ffp + np.abs(ac508_100_a_b.f1))
relay2 = rt.Doublet_lens(z1=relay2_z,
                         r1=363.1,
                         t1=4.0,
                         ri1=1.7320,
                         r2=44.2,
                         t2=16.0,
                         ri2=1.6721,
                         r3=-71.1,
                         aperture_radius=(clear_aperture_508/2),
                         ri_in=1.0, ri_out=1.0)
relay2.type = 'relay2 f100'

obj = rt.Perfect_lens(z1=relay2.ffp + f, f=f, na=na, ri_in=1.0, ri_out=1.0)

# optical train is a list
ot = [etl, relay1, relay2, obj]

#------------------------------------------------------------------------------#
# run simulation
# generate rays
n_rays = 1e6
initial_ray_diameter = 8
rays = rt.create_rays(type='gaussian',
                      source='infinity',
                      n_rays=1e6,
                      diameter=initial_ray_diameter)

rays[-1, :, 2] = -100

for lens in ot:
    rays = lens.raytrace(rays)

# find the focal plane using ray tracing paraxial and marginal ray focii.
fp_paraxial, fp_midpoint, fp_marginal = rt.ray_focal_plane(rays=rays, ri=1.0, method='all')

# Prepare electric field grid
grid, grid_params = pt.field_grid(num_xy=n_xy, num_zx=n_zx, dx=dx, dz=dz)
x, radius_xy, extent_xy, z, extent_zx = grid_params

# offset z to the focal plane using paraxial focus
z_real = z + fp_paraxial

# ray trace to propagation plane
rays_to_field_z = rt.rays_to_field_plane(rays=rays,
                                           x_max=x.max(),
                                           padding=0.050)

# The propagate() takes the distance from initial field to the field plane
z_prop = z_real - rays_to_field_z

# how far is the initial field from the midpoint focus?
initial_field_distance_to_focus = fp_paraxial - rays_to_field_z

# ray trace to the rays -> field plane
rays = rt.intersect_plane(rays, rays_to_field_z, ri_in=1.0, refract=False)

# generate simulation electric field using ray trace result to electric field
initial_field_simulation = rt.rays_to_field(mask_radius=radius_xy,
                                            rays=rays,
                                            ko=ko,
                                            binning=n_xy//5,
                                            amp_type='power',
                                            phase_type='opld',
                                            results='field',
                                            power=0.01,
                                            DEBUG=False,
                                            showfig=showfig)

# Propagate the initial fields to get a 3d distribution about the focus
field_simulation = pt.get_3d_field(initial_field_simulation,
                                   z=z_prop,
                                   wl=wl,
                                   dx=dx,
                                   ri=1.0,
                                   DEBUG=False,
                                   savedir=None,
                                   showfig=showfig)

psf_simulation = np.abs(field_simulation)**2

# light sheet analysis, fit each z plane for 3 gaussian model
ls_analysis_sim = analyze_lightsheet_zx(data=psf_simulation[:, n_xy//2, n_xy//2-100:n_xy//2+100],
                                        dx=dx * 1e3,
                                        dz=np.abs(dz) * 1e3,
                                        max_peak_ratio=analysis_params["max_peak_ratio"],
                                        peak_distance=analysis_params["peak_distance"],
                                        z_window=analysis_params["z_window"],
                                        w_window=analysis_params["w_window"],
                                        w_filter = analysis_params["w_filter"],
                                        mu_filter = analysis_params["mu_filter"],
                                        plot_results=analysis_params["plot_results"],
                                        DEBUG_zstack=analysis_params["DEBUG_zstack"],
                                        DEBUG_fits=analysis_params["DEBUG_fits"],
                                        showfig=analysis_params["showfig"],
                                        savedir=savedir,
                                        filename_label='simulation_analysis')

#------------------------------------------------------------------------------#
# Plotting
rt.plot_rays(rays=rt.intersect_plane(rays=rays, zf=obj.ffp+10, ri_in=1.0),
             n_rays_to_plot=51,
             title="Relay + Obj.",
             optical_train=ot,
             show_focal_planes=True,
             savedir=savedir / Path("simulation_raytrace.png"),
             showfig=showfig)

pt.plot_field(fields=[initial_field_simulation],
              field_labels=['Relay + Obj.'],
              grid_params=[grid_params],
              fig_title='Initial field comparison',
              gamma_scale=1.0,
              savedir=savedir / Path('simulation_field'),
              showfig=showfig)

pt.plot_xz_projection(fields=[field_simulation],
                      field_labels=["Relay + Obj."],
                      grid_params=[grid_params],
                      gamma_scale=1.0,
                      savedir=savedir / Path('psf_zx_projection'),
                      showfig=showfig)
