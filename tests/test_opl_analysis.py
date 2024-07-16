'''
Test Strehl ratio calculations

'''
# model imports
import model_tools.raytrace as rt
import model_tools.propagation as pt

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


#------------------------------------------------------------------------------#
# Define model parameters
#------------------------------------------------------------------------------#
plotrays=False

root_dir = Path(
    "/home/steven/Documents/qi2lab/github/raytracing_sjs/extFOV_model/data"
    )
savedir = rt.get_unique_dir(root_dir, 'strehl_testing')

# Unit conversion: mm to um /, um to mm *
mm = 1e-3

# Create set of rays to use for simulations
rays_params = {"type":"flat_top",
               "source":"point",
               "n_rays":4000,
               "diameter":10,
               "wl":0.000561}

initial_rays = rt.create_rays(*rays_params.values())

# Field model parameters
wl = 0.5 * mm
ko = 2 * np.pi / wl

# Create perfect objective
obj_z = 1.0
obj_f = 40.0
obj_na = 0.3

# Define defocus distances
dz_max = 100
n_model = 15
dz_positions = np.linspace(-dz_max, dz_max, n_model)

#------------------------------------------------------------------------------#
figshape = (10,5)
width_ratios = [1,1,1]
height_ratios = [1]
fig = plt.figure(figsize=figshape)
grid = fig.add_gridspec(nrows=len(height_ratios),
                        ncols=len(width_ratios),
                        width_ratios=width_ratios,
                        height_ratios=height_ratios,
                        wspace=0.4,
                        hspace=0.1)

ax_wf = fig.add_subplot(grid[0])
ax_rms = fig.add_subplot(grid[1])
ax_str = fig.add_subplot(grid[2])

ax_wf.set(title='Wavefront aberration',
          xlabel=r'$\rho$',
          ylabel=r'$\Delta W(\rho)$')
ax_rms.set(title='RMS',
           xlabel=r'$\Delta z$',
           ylabel=r'$RMS$')
ax_str.set(title='Strehl Ratios',
           xlabel=r'$\Delta z$',
           ylabel='Strehl')

for ii, dz in enumerate(tqdm(dz_positions,
                             desc='Iterating over dzs...',
                             leave=True)):
    # Offset point source'
    rays = initial_rays.copy()

    # Create 4f relay
    obj1 = rt.Perfect_lens(z1=(obj_f + dz),
                           f=obj_f,
                           na=obj_na,
                           ri_in=1.0,
                           ri_out=1.0,
                           type="obj"
                           )
    obj2 = rt.Perfect_lens(z1=obj1.ffp + obj1.f,
                           f=obj1.f,
                           na=obj_na,
                           ri_in=1.0,
                           ri_out=1.0,
                           type="obj")
    # Define relay lens 1 parameters
    relay1_lens_params = [0,
                        109.7, 12.0, 1.5180,
                        -80.7, 2.0, 1.6757,
                        -238.5,
                        (50/2),
                        1.0, 1.0,
                        "relay1"
                        ]
    # reference relay 1
    relay1_temp = rt.Doublet_lens(*relay1_lens_params)


    # ot = [relay1_temp] #, obj2]
    ot = [obj1, obj2]
    results = rt.raytrace_ot(optical_train=ot,
                             rays=initial_rays.copy(),
                             return_rays="all",
                             fit_plane="midpoint",
                             fit_method="opld",
                             plot_raytrace_results=True,
                             save_path=savedir / Path(f"raytracing_{ii}.png"))

    strehl = results["strehl"]
    rms = results["rms"]

    pupil_rays = results["rays"]
    pupil_radius = 1

    rho, wf = rt.get_ray_wf(pupil_rays, pupil_radius, "opld")

    # Fit opl to subtract piston C0 term.
    wf_fit = rt.ray_opl_polynomial(pupil_rays, pupil_radius, "opld")
    wf_aberration = wf - wf_fit[0]*rho

    ax_str.plot(dz, strehl, 'r.')
    ax_rms.plot(dz, rms, "b.")
    ax_wf.plot(rho, wf_aberration, "r.")
