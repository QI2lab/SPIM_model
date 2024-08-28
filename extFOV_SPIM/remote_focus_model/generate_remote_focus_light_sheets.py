"""
Open ETL remote focus optimization results (from generating heatmap).

Specify models to generate fields and display for paper figure.

08/20/2024
Steven Sheppard
"""
# Imports
import model_tools.raytrace as rt
import model_tools.propagation as pt
from model_tools.analytic_forms import (get_reference_sphere_opl,
                                        gaussian_intensity_no_offset)
from scipy.interpolate import interp1d
from pathlib import Path
import numpy as np
import gc as gc
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap, TwoSlopeNorm
import time
from tqdm import tqdm, trange
t_start = time.time()

# Define a method for generating the figure plots
def plot_summary(model: np.ndarray,
                    save_path: Path=None,
                    showfig: bool=False):
    """
    """
    # Create figure
    fig = plt.figure(figsize=(5.5, 3.0))
    grid = fig.add_gridspec(nrows=1,
                            ncols=6,
                            width_ratios=[0.25,0.1,1,0.075,1,0.075],
                            wspace=0.05,
                            hspace=0.05)

    # Plot a bar plot of the fit results
    ax = fig.add_subplot(grid[0])
    fit = model["opl_fit"]
    bar_heights = 0.5

    if plot_polynomial:
        ax.set_title("Polynomial Coeff.")
        fit_labels = [f"$C_{ii}$" for ii in range(len(fit))]

        # Separate positive and negative fit values for coloring
        colors = ['red' if coeff < 0 else 'blue' for coeff in fit]
        magnitudes = np.abs(fit)

        ax.barh(np.arange(len(fit)),
                magnitudes,
                tick_label=fit_labels,
                color=colors,
                height=bar_heights)
        ax.set_xlabel(r"$|C_i|$", labelpad=5, rotation="horizontal", fontsize=12)

        ax.tick_params("both", labelsize=11)
        ax.set_xscale("log")
        ax.set_xlim(left=1e-8)
    else:
        ax.set_title("Zernike Coeff.")

        zern_labels = [r"$Z_0$",r"$Z_3$",r"$Z_8$"]
        fit_labels = [f"$C_{ii}$" for ii in range(len(fit))]
        zernikes = rt.get_zernike_from_fit(fit)

        # Separate positive and negative fit values for coloring
        colors = ['red' if coeff < 0 else 'blue' for coeff in zernikes]
        magnitudes = np.abs(zernikes)

        ax.barh(np.arange(len(zernikes)),
                magnitudes,
                tick_label=zern_labels,
                color=colors,
                height=bar_heights)
        ax.set_xlabel(r"$|Z_i|$", labelpad=5, rotation="horizontal", fontsize=12)

        ax.tick_params("both", labelsize=11)
        ax.set_xscale("log")

    # plot the wavefront using fit coeffiecients
    ax = fig.add_subplot(grid[2])
    ax.set_title("Field Phase")
    # Define custom colormap symmetric about black
    cdict = {
            'red':   [(0.0, 1.0, 1.0),  # Red at the start (for negative values)
                    (0.5, 0.0, 0.0),  # Black in the center
                    (1.0, 0.0, 0.0)], # No red at the end (for positive values)

            'green': [(0.0, 0.0, 0.0),  # No green at the start
                    (0.5, 0.0, 0.0),  # Still no green in the center (black)
                    (1.0, 0.0, 0.0)], # No green at the end

            'blue':  [(0.0, 0.0, 0.0),  # No blue at the start (for negative values)
                    (0.5, 0.0, 0.0),  # Black in the center
                    (1.0, 1.0, 1.0)]  # Blue at the end (for positive values)
            }

    black_centered_cmap = LinearSegmentedColormap('BlackCentered',
                                                    segmentdata=cdict)

    # Define the wavefront grid to evaluate on
    num_xy = 2001
    dx = 2*np.sqrt(2)/num_xy
    x, radius_xy, extent_xy = pt.field_grid(num_xy=num_xy, num_zx=1, dx=dx, return_field=False)
    wf = rt.opl_polynomial(radius_xy,fit)

    # clip at pupil edge
    wf[radius_xy>1]=0

    # Apply gamma correction and center around zero using TwoSlopeNorm
    phase_max = np.max(np.abs(wf))
    norm = PowerNorm(gamma=0.5)  # Apply gamma scaling
    center_zero_norm = TwoSlopeNorm(vmin=-phase_max, vcenter=0, vmax=phase_max)

    im = ax.imshow(wf,
                   cmap=black_centered_cmap,
                   norm=center_zero_norm,
                   extent=extent_xy,
                   aspect="equal",
                   origin="lower")

    ax.set_xlabel(r"$\rho$", fontsize=12, labelpad=8, rotation="horizontal")
    ax.set_ylabel(r"$\rho$", fontsize=12, labelpad=5, rotation="horizontal")
    ax.set_xticks([-1.0, 0, 1.0])
    ax.set_yticks([-1.0, 0, 1.0])
    ax.tick_params("both", labelsize=11)

    # Cbar axes
    cax = fig.add_subplot(grid[3])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("OPL(mm)", rotation="horizontal", labelpad=25, fontsize=12)
    cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

    # plot the maxprojection along side
    ax = fig.add_subplot(grid[4])
    ax.set_title("Max Projection")

    # Define the wavefront grid to evaluate on
    max_projection = pt.xz_projection(model["field"])
    x, radius_xy, extent_xy, z, extent_zx = model["grid_params"]

    # Convert from mm to um
    x = [1e3 * xx for xx in x]
    z = [1e3 * zz for zz in z]
    extent_zx = [1e3 * xt for xt in extent_zx]

    x = np.asarray(x)
    z = np.asarray(z)
    dx = x[1] - x[0]
    dz = z[1] - z[0]

    x_max = 50
    # Optional, crop projection
    if x_max:
        idx_x = [(np.abs(x + x_max)).argmin(), (np.abs(x - x_max)).argmin()]
        max_projection = max_projection[idx_x[0]:idx_x[1], :]
        extent_zx[2:] =  [x[idx_x[0]] - dx/2, x[idx_x[1]] + dx/2]

    # Define maximum for symmetric colorbar
    im = ax.imshow(max_projection/np.max(max_projection),
                   cmap="hot",
                   vmin=0,
                   vmax=1,
                   extent=extent_zx,
                   aspect="equal",
                   origin="lower")

    ax.set_xlabel((r"x ($\mu m$)"), fontsize=12, labelpad=8, rotation="horizontal")
    ax.set_ylabel(r"z ($\mu m$)", fontsize=12, labelpad=5, rotation="horizontal")
    ax.tick_params("both", labelsize=11)

    # Cbar axes
    cax = fig.add_subplot(grid[5])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("A.U.", rotation="horizontal", labelpad=25, fontsize=12)
    cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

    # Adjust the layout to align all elements properly
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.2, right=0.85, left=0.05)
    plt.show()


    if save_path:
        fig.savefig(save_path)
    if showfig:
        fig.show()
    else: fig.close()

data_path = Path(r"/mnt/server1/extFOV/remote_focus_data/20240821_145235_remote_focus_results/0.14NA_in_ri_mismatch_results.npy")
plot_dir = data_path.parent / Path("plots/")

plot_polynomial = True
plot_light_sheet_slices = True

# Define which model parameters to generate fields and plot raytrace results
# [cuvette_offset, etl_dpt]
models = [[35, 4], [35, 10], [35, 16],
          [25, 4], [25, 10], [25, 16],
          [15, 4], [15, 10], [15, 16]]

# Define model parameters
wl = 0.000488
ko = 2 * np.pi / wl
dx = wl/2
dz = 0.002
x_max = 0.500
z_max = 0.500
x_padding = 0.050
n_xy = int(x_max // dx)
n_zx = int(z_max // dz)

# enforce odd grid numbers
if n_xy%2 == 0:
    n_xy += 1
if n_zx%2 == 0:
    n_zx += 1

# locate the models to simulate in results
# load list of result_dicts
print("Loading heatmap data . . . ")
tl_start = time.time()
results_list = np.load(data_path, allow_pickle=True)
print(f"Loading all heatmap results took {(time.time()-tl_start):.2f} seconds")

print("Finding matching remote focus model configurations")
models_to_keep = []
for params in tqdm(models, "Iterating over RF model heatmap", leave=True):
    target_cuvette_offset, target_focus_shift = params

    # Step 1: Find all indices with the closest cuvette_offset
    cuvette_offset = min(
        results_list,
        key=lambda d: abs(d['cuvette_offset'] - target_cuvette_offset)
    )['cuvette_offset']

    params[0] = round(cuvette_offset, 3)

    candidate_indices = [
        i for i, d in enumerate(results_list)
        if d['cuvette_offset'] == cuvette_offset
    ]

    # Step 2: Among those, find the index with the closest etl_dpt
    best_idx = min(
        candidate_indices,
        key=lambda i: abs(results_list[i]['focus_shift'] - target_focus_shift)
    )

    params[0] = round(results_list[best_idx]['cuvette_offset'], 3)
    params[1] = round(results_list[best_idx]['focus_shift'], 3)

    # Append the index of the best match to the params list
    models_to_keep.append(results_list[best_idx])

results_list = None
del results_list
# TODO come back and get the ETL dpt values for the desired focus shift and set up run_etl_.. to only run those values if and option is selected.

for ii, _r in enumerate(models_to_keep):
    print(f"Completing simulation {ii+1}/{len(models_to_keep)}...", end="\r")
    initial_field_params = pt.field_grid(num_xy=n_xy,
                                            num_zx=1,
                                            dx=dx,
                                            dz=dz,
                                            return_field=False
                                            )
    initial_field = rt.raytrace_to_field(results=_r,
                                         grid_params=initial_field_params,
                                         wl=wl,
                                         amp_binning=n_xy//2,
                                         focal_plane="midpoint",
                                         grid_padding=x_padding,
                                         plot_rays_to_field=True,
                                         plot_raytrace_results=True,
                                         label=(f"raytrace_to_field_{_r['cuvette_offset']:.1f}offset_{_r['focus_shift']:.1f}shift.pdf"),
                                         savedir=plot_dir,
                                         showfig=True)

    # plot the fit results:
    rt.plot_fit_summary(_r["opl_fit"],
                        fig_title=f"dc={_r['cuvette_offset']:.2f}mm, df={_r['focus_shift']:.2f}mm, Strehl={_r['strehl']:.3f}, RMS={_r['rms']*1e3:.3f}um",
                        showfig = True,
                        save_path=plot_dir / Path(f"wavefront_fit_{_r['cuvette_offset']:.1f}offset_{_r['focus_shift']:.1f}shift.pdf"))

    #---------------------------------------------------------------------#
    print(f"\nGenerating 3d electric fields . . . ")
    # Create field grid to use for field propagation
    shared_grid_params = pt.field_grid(num_xy=n_xy,
                                        num_zx=n_zx,
                                        dx=dx,
                                        dz=dz,
                                        return_field=False
                                        )

    # initialize grid params for propagation
    x, radius_xy, extent_xy, z, extent_zx = shared_grid_params

    #-----------------------------------------------------------------#
    # Generate the 3d field for the fast axis
    z_real = z + _r["midpoint_focal_plane"]
    extent_zx[0] += _r["midpoint_focal_plane"]
    extent_zx[1] += _r["midpoint_focal_plane"]

    # Grab the propagation media RI from raytracing results
    prop_ri = _r['optical_train'][-1].ri_out
    # Define propagation coordinates
    z_prop = z_real - initial_field[1]
    # Redefine propagation grid params
    grid_params = [x, radius_xy, extent_xy, z, extent_zx]

    # Calculate 3d electric field
    field = pt.get_3d_field(initial_field[0],
                            z=z_prop,
                            wl=wl,
                            dx=dx,
                            ri=prop_ri,
                            DEBUG=False
                            )

    joint_intensity = pt.get_light_sheet(field, x, None, 20.0)

    # grab central slice
    light_sheet_slice = joint_intensity[:, joint_intensity.shape[1]//2, :]
    _r["light_sheet_slice"] = light_sheet_slice
    _r["initial_field"] = initial_field
    _r["field"] = field
    _r["grid_params"] = shared_grid_params

    if plot_light_sheet_slices:
        # Update grid params for plotting
        t_n_zx, t_n_xy, = light_sheet_slice.shape
        temp_grid_params = pt.field_grid(num_xy=t_n_xy,
                                         num_zx=t_n_zx,
                                         dx=dx,
                                         dz=dz,
                                         return_field=False)
        temp_grid_params[-1][0] += _r["midpoint_focal_plane"]
        temp_grid_params[-1][1] += _r["midpoint_focal_plane"]

        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.set(title=(f"Light sheet slice intensity, " +
                        f"RMS={_r['rms']:.3f}mm, Strehl={_r['strehl']:.3f}"),
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
        ax.axvline(x=_r["midpoint_focal_plane"],
                    label="mipoint",
                    linestyle="--",
                    color="g"
                    )
        ax.axvline(x=_r["paraxial_focal_plane"],
                    label="paraxial",
                    linestyle="--",
                    color="limegreen"
                    )
        ax.axvline(x=_r["marginal_focal_plane"],
                    label="marginal",
                    linestyle="--",
                    color="red"
                    )
        ax.legend(fontsize=6,
                    framealpha=0.1,
                    labelcolor="white"
                    )
        plt.colorbar(im)

        fig.savefig(plot_dir / Path(f"light_sheet_slice_{_r['cuvette_offset']:.1f}offset_{_r['focus_shift']:.1f}shift.pdf"))

print("Saving simulated positions . . . ")
ts_start = time.time()
np.save(data_path.parent / Path("models_for_figure.npy"), models_to_keep, allow_pickle=True)
print(f"saving took {(time.time() - ts_start):.2f} seconds")

print("Plotting maximum intensity projections . . . ")
pt.plot_xz_projection(fields = [_r["field"] for _r in models_to_keep],
                      field_labels= [f"Cuvette Offset:{_p[0]}, focus shift:{_p[1]}, Strehl: {_r['strehl']:.3f}, RMS: {_r['rms']:.3f}" for _p, _r in zip(models, models_to_keep)],
                      grid_params=[_r["grid_params"] for _r in models_to_keep],
                      x_max=20,
                      save_path=data_path.parent / Path("projections.png"), showfig=True)

print(f"Total runtime: {(time.time() - t_start):.2f} seconds")