'''
extFOV SPIM light sheet simulation.

Simulate experimental light sheet remote focus pathway.
Apt -> ETL -> AC508-150-A -> AC508-100-A -> NA OBJ -> AIR/WATER

Simulate light sheets by modeling without cylindrical lens and then
multiply their intensities at the focal plane.

01/10/2023
Steven Sheppard
'''
import model_tools.propagation as pt
import model_tools.raytrace as rt
from model_tools.analysis import light_sheet_analysis
from model_tools.analytic_forms import gaussian_intensity_no_offset
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc as gc
from pathlib import Path
import numpy as np
import time
from scipy.interpolate import interp1d
import dask
from dask.diagnostics import ProgressBar

t_initial = time.time()

# Debugging / plotting flags
DEBUG = False
showfig = False
plot_raytracing = False
plot_propagation = True
plot_light_sheet_slices = True

# light_sheet_simulations_f180f150_r0_80_f20_test_range
# where to save results
root_dir = Path("/mnt/server1/extFOV/light_sheet/simulations")
s_str = f'gaussian_light_sheet_model'
# s_str = f'for_fig_image'
# s_str = "f20_f180f150_r60"
savedir = rt.get_unique_dir(root_dir, Path(s_str))

#------------------------------------------------------------------------#
# Define model parameters
#------------------------------------------------------------------------#
# Propagation constants
wl = 0.000561
ko = 2 * np.pi / wl
dx = 0.000100
x_max = 0.500
x_padding = 0.050

rays_params = {"type":"gaussian",
               "source":"infinity",
               "n_rays":4e6,
               "diameter":100,
               "sampling_limit":16/100
               }

initial_rays = rt.create_rays(*rays_params.values())

#------------------------------------------------------------------------#
# Create remote focus / illumination pathway
# To create Relay doublets:
# 1. Define temporary lens at z=0
# 2. Use the temporary lens to place conjugate with the previous lens.
# How lens Doublet lens parameters are defined.
# _params = [z1,
#            r1, t1, ri1,
#            r2, t2, ri2,
#            r3,
#            aperture_radius,
#            ri_in, ri_out,
#            type]# Define the clear aperture for Thorlabs AC508-XXX-ML
#------------------------------------------------------------------------#
clear_aperture_508 = 50.8
# Define the cropping aperture
aperture_params = [0,
                   np.inf, 1.0, 1.0,
                   np.inf,
                   clear_aperture_508,
                   1.0, 1.0,
                   "aperture"
                   ]
# Define parameters for the ETL (plano-CURVED)
etl_z0 = 100
etl_ri = 1.3
etl_t0 = 10
etl_d = 16
etl_dpt_min = -10
etl_dpt_max = 10

# Define relay lens 1 parameters
relay1_lens_params = [0,
                      83.2, 12.0, 1.5180,
                      -72.1, 3.0, 1.6757,
                      -247.7,
                      (clear_aperture_508/2),
                      1.0, 1.0,
                      "relay1"]
# Define relay lens 2 parameters
relay2_lens_params = [0,
                      363.1, 4.0, 1.7320,
                      44.2, 16.0, 1.6721,
                      -71.1,
                      (clear_aperture_508/2),
                      1.0, 1.0,
                      "relay2"]


# Define cuvette / ri-mismatch parameters
immersion_ri = 1.33
cuvette_path_length = 20.0
cuvette_wall_thickness = 1.25
cuvette_height = 45.0
cuvette_ri = 1.4585
cuvette_offset = 6.52

#------------------------------------------------------------------------#
aperture = rt.Thick_lens(*aperture_params)
# flat ETL
flat_etl = rt.create_etl(z1=etl_z0, dpt=0, d=etl_d, ri=etl_ri, t0=etl_t0)
# reference relay 1
relay1_temp = rt.Doublet_lens(*relay1_lens_params)
# Update the z1 position of the lens
relay1_lens_params[0] = flat_etl.z2 + relay1_temp.f1
relay1 = rt.Doublet_lens(*relay1_lens_params)
# reference relay2
relay2_temp = rt.Doublet_lens(*relay2_lens_params)
# Update the z1 position of the lens
relay2_lens_params[0] = relay1.ffp + relay2_temp.f1
relay2 = rt.Doublet_lens(*relay2_lens_params)

# Create the excitation objectives to simulate
# Nikon 0.3 PlanFluor
nikon_f = 20
nikon_na = 0.3
nikon_mag = 10
nikon_wd = 16

exc_obj = rt.Perfect_lens(z1=relay2.ffp + nikon_f,
                          f=nikon_f,
                          na=nikon_na,
                          wd=nikon_wd,
                          fov=25,
                          mag=nikon_mag,
                          ri_in=1.0, ri_out=1.0,
                          type="exc_obj"
                          )

index_mismatch =  rt.Thick_lens(z1=((exc_obj.ffp-exc_obj.wd) + cuvette_offset),
                                r1=np.inf,
                                t=cuvette_wall_thickness,
                                ri=cuvette_ri,
                                r2=np.inf,
                                aperture_radius=cuvette_height/2,
                                ri_in=1.00,
                                ri_out=immersion_ri,
                                type = "cuvette")

# Define remote focus distances
etl_dpts = np.linspace(-3.0, 3.0, 11)
# etl_dpts = np.array([0])
# Define "aperture" conjugate to BFP for cropping initial rays
full_aperture = 6.0 # radius, experimental=6.0
apt_scales = np.array([1.00, 0.88, 0.80, 0.70, 0.65,
                       0.58, 0.50, 0.45, 0.41, 0.38])
# Define the experimental distance between the obj. surface and cuvette
cuvette_offset = 6.5

#------------------------------------------------------------------------#
# Dictionary of results to be appended to and ultimately saved
# Create dict. for simulations
results_dicts = [{"label":"Model in air",
                  "n_image":1.00,
                  "wl":wl,
                  "dx":dx,
                  "xy_extent":x_max,
                  "dzs":[0.001, 0.001, 0.001, 0.002, 0.002,
                         0.002, 0.002, 0.002, 0.002, 0.002],
                  "z_extent":[0.150, 0.180, 0.200, 0.220, 0.240,
                              0.260, 0.300, 0.350, 0.380, 0.400]
                  },
                 {"label":"Model in water",
                  "n_image":1.333,
                  "wl":wl,
                  "dx":dx,
                  "xy_extent":x_max,
                  "dzs":[0.001, 0.001, 0.001, 0.002, 0.002,
                         0.002, 0.002, 0.002, 0.002, 0.002],
                  "z_extent":[0.160, 0.180, 0.200, 0.220, 0.250,
                              0.300, 0.350, 0.380, 0.400, 0.450]
                  }
                 ]

# loop through experimental setups and run model
for _result in results_dicts:
    # Create new directory for saving plots
    plot_dir = savedir / Path(f"{_result['label']}_plots")
    plot_dir.mkdir(exist_ok=True)

    # Compile optical trains to simulate,
    optical_trains = []

    # assign the native focusing OT
    if 'water' in _result['label']:
        native_ot = [aperture,
                     flat_etl,
                     relay1,
                     relay2,
                     exc_obj
                     ]
    else:
        native_ot = [aperture,
                     flat_etl,
                     relay1,
                     relay2,
                     exc_obj,
                     index_mismatch
                     ]

    for ii, scale in enumerate(apt_scales):
        # Update aperture
        aperture_params[5] = full_aperture * scale
        aperture = rt.Thick_lens(*aperture_params)
        # Create list of optical trains
        temp_ot = []
        for dpt in etl_dpts:
            # Update ETL dpt
            etl = rt.create_etl(z1=flat_etl.z1,
                                dpt=dpt,
                                d=etl_d,
                                ri=etl_ri,
                                t0=etl_t0
                                )
            if 'water' in _result['label']:
                temp_ot.append([aperture,
                                etl,
                                relay1,
                                relay2,
                                exc_obj,
                                index_mismatch]
                                )
            else:
                temp_ot.append([aperture,
                                etl,
                                relay1,
                                relay2,
                                exc_obj]
                               )
        optical_trains.append(temp_ot)

    #------------------------------------------------------------------------#
    # Simulate optiucal trains
    for ii, ots in enumerate(optical_trains):
        print(f"\nAperture {ii+1} / {len(optical_trains)}",
              f"\nRaytracing {len(ots)} optical trains . . .  ")

        # rt_dir = plot_dir / Path("raytrace_plots")
        # rt_dir.mkdir(exist_ok=True)

        # Raytrace
        with ProgressBar():
            tasks = [dask.delayed(rt.raytrace_ot)(optical_train=ot,
                                                  rays=initial_rays.copy(),
                                                  fit_plane="midpoint",
                                                  fit_method="opld",
                                                  wl=wl,
                                                  return_rays="all",
                                                  plot_raytrace_results=False
                                                  )
                    for jj, ot in enumerate(ots)]
            raytrace_results = dask.compute(*tasks)

        #---------------------------------------------------------------------#
        # Create electric fields from ray tracing results
        # rt_field_dir = plot_dir / Path("rays_to_field_plots")
        # rt_field_dir.mkdir(exist_ok=True)

        # Update model grid params
        dz = _result["dzs"][ii]
        dx = _result["dx"]
        n_xy = int(_result["xy_extent"] // dx)
        n_zx = int(_result["z_extent"][ii] // dz)
        # enforce odd grid numbers
        if n_xy%2 == 0:
            n_xy += 1
        if n_zx%2 == 0:
            n_zx += 1
        # Create field grid to interpolate ray tracing results
        initial_field_params = pt.field_grid(num_xy=n_xy,
                                             num_zx=1,
                                             dx=dx,
                                             dz=dz,
                                             return_field=False
                                             )
        print(f"\nGenerating initial fields . . . ")

        initial_field = []
        for _rt in tqdm(raytrace_results):
            initial_field.append(
                rt.raytrace_to_field(
                    results=_rt,
                    grid_params=initial_field_params,
                    wl=wl,
                    amp_binning=n_xy//10,# "doane",
                    grid_padding=x_padding,
                    plot_rays_to_field=plot_propagation,
                    plot_raytrace_results=plot_raytracing,
                    label=(f"{full_aperture*apt_scales[ii]:.1f}apt_"
                            + f"{_rt['optical_train'][1].dpt:.2f}dpt_"
                            + f"{_result['label']}"),
                    savedir=plot_dir)
                                  )
        #---------------------------------------------------------------------#
        print(f"\nGenerating 3d electric fields . . . ")
        # field_dir = plot_dir / Path("field_plots")
        # field_dir.mkdir(exist_ok=True)
        # Create field grid to use for field propagation
        shared_grid_params = pt.field_grid(num_xy=n_xy,
                                           num_zx=n_zx,
                                           dx=dx,
                                           dz=dz,
                                           return_field=False
                                           )
        # Calculate light sheets
        sim_dicts = []
        for jj, (_rt, _init_field) in enumerate(zip(raytrace_results,
                                                    initial_field)):
            # Create label for current simulation
            label = f"{_result['label']}"
            # initialize grid params for propagation
            x, radius_xy, extent_xy, z, extent_zx = shared_grid_params

            #-----------------------------------------------------------------#
            # Generate the 3d field for the fast axis
            # Offset z to the focal plane using midpoint focal plane
            z_real = z + _rt["midpoint_focal_plane"]
            extent_zx[0] += _rt["midpoint_focal_plane"]
            extent_zx[1] += _rt["midpoint_focal_plane"]

            # Grab the propagation media RI from raytracing results
            prop_ri = _rt['optical_train'][-1].ri_out
            # Define propagation coordinates
            z_prop = z_real - _init_field[1]
            # Redefine propagation grid params
            grid_params = [x, radius_xy, extent_xy, z, extent_zx]

            print(f"Aperture: {ii+1} / {len(apt_scales)}, ",
                  f"Defocus OT: {jj+1} / {len(raytrace_results)}",
                  f"\nlabel: {label:s} ",
                  f"\nelectric field shape: {n_zx:d}x{n_xy:d}x{n_xy:d}")

            # Calculate 3d electric field
            field = pt.get_3d_field(_init_field[0],
                                     z=z_prop,
                                     wl=wl,
                                     dx=dx,
                                     ri=prop_ri,
                                     DEBUG=False
                                     )
            # Crop down fields to speed up calculations
            n_crop = int(0.040 // dx)
            # enforce odd grid numbers
            if n_crop%2 == 0:
                n_crop += 1
            # Update coordinates
            x_crop = x[n_xy//2-n_crop:n_xy//2+n_crop+1]
            y_crop = x_crop
            xx_crop, yy_crop = np.meshgrid(x_crop, y_crop)
            eval_radius = np.sqrt((xx_crop*np.sqrt(2))**2
                                  + (yy_crop*np.sqrt(2))**2)
            # Crop down fields before performing operations
            field = field[:,
                          n_xy//2-n_crop:n_xy//2+n_crop+1,
                          n_xy//2-n_crop:n_xy//2+n_crop+1]

            # normalize intensities
            intensity_f = np.abs(field)**2/np.max(np.abs(field)**2)
            intensity_s_xy = gaussian_intensity_no_offset(r=eval_radius,
                                                          w=20.00,
                                                          Io=1.0,
                                                          mu=0)
            joint_intensity = np.zeros(intensity_f.shape)

            # evaluate I(r) at different radial coordinates than native data
            for z_idx in range(n_zx):
                f_intensity_interp = interp1d(x_crop,
                                              intensity_f[z_idx, n_crop, :],
                                              kind="linear",
                                              bounds_error=False,
                                              fill_value=0
                                              )
                joint_intensity[z_idx] = np.sqrt(f_intensity_interp(eval_radius))*np.sqrt(intensity_s_xy)

            # grab central slice
            light_sheet_slice = joint_intensity[:, n_crop, :]

            if plot_light_sheet_slices:
                # Update grid params for plotting
                t_n_zx, t_n_xy, = light_sheet_slice.shape
                temp_grid_params = pt.field_grid(num_xy=t_n_xy,
                                                 num_zx=t_n_zx,
                                                 dx=dx,
                                                 dz=dz,
                                                 return_field=False)
                temp_grid_params[-1][0] += _rt["midpoint_focal_plane"]
                temp_grid_params[-1][1] += _rt["midpoint_focal_plane"]

                fig, ax = plt.subplots(1,1, figsize=(7,5))
                ax.set(title=(f"Light sheet slice intensity, " +
                              f"RMS={_rt['rms']:.3f}mm, Strehl={_rt['strehl']:.3f}"),
                       ylabel='r (um)',
                       xlabel='z (um)'
                       )
                im = ax.imshow(np.rot90(light_sheet_slice,1),
                               cmap='hot',
                               extent=temp_grid_params[-1],
                               aspect='equal',
                               origin='lower',
                               interpolation=None
                               )
                ax.axvline(x=_rt["midpoint_focal_plane"],
                           label="mipoint",
                           linestyle="--",
                           color="g"
                           )
                ax.axvline(x=_rt["paraxial_focal_plane"],
                           label="paraxial",
                           linestyle="--",
                           color="limegreen"
                           )
                ax.axvline(x=_rt["marginal_focal_plane"],
                           label="marginal",
                           linestyle="--",
                           color="red"
                           )
                ax.legend(fontsize=6,
                          framealpha=0.1,
                          labelcolor="white"
                          )
                plt.colorbar(im)

                f_str = "".join([f"ls_slice_{_result['label']}_",
                                 f"{_rt['optical_train'][0].aperture:.1f}mmapt_",
                                 f"{_rt['optical_train'][1].dpt:.2f}dpt.png"]
                                )
                fig.savefig(plot_dir / Path(f_str))


            # Run light sheet analysis
            sim_dict = {'light_sheet_slice':light_sheet_slice,
                        'dx':dx * 1e3, # Um
                        'dz':dz * 1e3, # Um
                        'stage_positions':z_real * 1e3, # Um
                        'etl_dpt':_rt['optical_train'][1].dpt,
                        'aperture':_rt['optical_train'][0].aperture,
                        'label':label,
                        'paraxial_focal_plane':_rt['paraxial_focal_plane']*1e3,
                        'midpoint_focal_plane':_rt["midpoint_focal_plane"]*1e3,
                        'marginal_focal_plane':_rt["marginal_focal_plane"]*1e3,
                        'rays_to_field_z':_init_field[1]*1e3,
                        'optical_train': _rt["optical_train"]
                        }
            a_str = "".join([f"{_result['label']}_",
                             f"{_rt['optical_train'][0].aperture:.1f}mmapt_",
                             f"{_rt['optical_train'][1].dpt:.2f}dpt"]
                            )
            light_sheet_analysis(results=sim_dict,
                                 plot_waist_fit=False,
                                 plot_width_calcs=False,
                                 plot_acq_results=True,
                                 label=a_str,
                                 savedir=plot_dir
                                 )
            sim_dicts.append(sim_dict)

            del (field,
                 intensity_f,
                 joint_intensity,
                 sim_dict)
            gc.collect()

        save_dict = {"results":sim_dicts,
                     "label": _result["label"]
                     }
        s_str = "".join([f"{_result['label']}_",
                         f"{_rt['optical_train'][0].aperture:.1f}mm_apt.npy"]
                        )
        save_path = savedir / Path(s_str)
        np.save(save_path, save_dict, allow_pickle=True)

        del save_dict, sim_dicts
        gc.collect()

print(f"Total run time: {(time.time() - t_initial)/60} minutes")
