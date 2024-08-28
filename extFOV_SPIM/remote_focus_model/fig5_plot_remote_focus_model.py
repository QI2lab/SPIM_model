
# Imports
import model_tools.raytrace as rt
import model_tools.propagation as pt
from pathlib import Path
import numpy as np
import gc as gc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, CenteredNorm
import time
from tqdm import tqdm
zern_max = 1e-2
zern_min = 1e-8
phase_max = 40e-5

# Define a method for generating the figure plots
def plot_summary(model: np.ndarray,
                    save_path: Path=None,
                    showfig: bool=False):
    """
    """
    # Create figure
    fig = plt.figure(figsize=(6.5,3.0))
    fig.suptitle(f"Strehl={model['strehl']:.3f}, RMS={model['rms']:.6f}")

    grid = fig.add_gridspec(nrows=1,
                            ncols=7,
                            width_ratios=[0.3,0.15, 0.7, 0.07,0.30, 0.3, 0.07],
                            wspace=0.1,
                            hspace=0)

    # Plot a bar plot of the fit results
    ax = fig.add_subplot(grid[0])
    fit = model["opl_fit"]
    bar_heights = 0.5

    # ax.set_title("Zernike Coeff.")

    zern_labels = [r"Piston ($Z_0$)",r"Defocus ($Z_3$)",r"Primary Sph. ($Z_8$)", r"Secondary Sph. ($Z_{15}$)", r"Tertiary Sph. ($Z_{24}$)"]
    zernikes = rt.get_zernike_from_fit(fit)

    # Separate positive and negative fit values for coloring
    # colors = ['red' if coeff < 0 else 'blue' for coeff in zernikes]
    # magnitudes = np.abs(zernikes)
    ax.axvline(x=0, c='k')
    ax.barh(np.arange(len(zernikes)),
            zernikes,
            tick_label=zern_labels,
            color="b",
            height=bar_heights)
    ax.set_xlabel(r"$Z_i$", labelpad=5, rotation="horizontal", fontsize=11)
    ax.tick_params("both", labelsize=10)
    # ax.set_xscale("log")
    # ax.set_xlim(zern_min, zern_max)
    ax.set_xlim(-1.8e-4, 1.8e-4)
    ax.set_xticks([-1e-4, 1e-4])
    ax.set_xticklabels(["-1e-4", "1e-4"])

    # plot the wavefront using fit coeffiecients
    ax = fig.add_subplot(grid[2])
    # ax.set_title("Field Phase")
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
    # center_zero_norm = TwoSlopeNorm(vmin=-phase_max, vcenter=0, vmax=phase_max)
    gamma = 0.5
    tick_labels = ["-5e-4","-2e-4", "-1e-5", "1e-5", "2e-4","5e-4"]
    tick_values = [np.abs(float(tl))**gamma*np.sign(float(tl)) for tl in tick_labels]
    wf_max = np.max(tick_values)
    print(tick_labels, tick_values)
    im = ax.imshow(np.abs(wf)**gamma*np.sign(wf),
                   cmap=black_centered_cmap,
                   vmin=-wf_max,
                   vmax=(wf_max**1),
                   extent=extent_xy,
                   aspect="equal",
                   origin="lower")

    ax.set_xlabel(r"$\rho$", fontsize=11, labelpad=8, rotation="horizontal")
#     ax.set_ylabel(r"$\rho$", fontsize=11, labelpad=5, rotation="horizontal")
    ax.set_xticks([-1.0, 0, 1.0])
    ax.set_yticks([-1.0, 0, 1.0])
    ax.tick_params("both", labelsize=10)

    # Cbar axes
    cax = fig.add_subplot(grid[3])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_yticks(tick_values)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.set_ylabel("OPL(mm)", rotation="vertical", labelpad=2, fontsize=11)

    # plot the maxprojection along side
    ax = fig.add_subplot(grid[5])

    # Define the wavefront grid to evaluate on
    # ls_slice = pt.xz_projection(model["field"])
    ls_slice = model["light_sheet_slice"]
    x, radius_xy, extent_xy, z, extent_zx = model["grid_params"]

    # Convert from mm to um
    x = [1e3 * xx for xx in x]
    z = [1e3 * zz for zz in z]
    extent_zx = [1e3 * xt for xt in extent_zx]
    x = np.asarray(x)
    z = np.asarray(z)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    x_max = 30
    z_max = 150

    # Optional, crop projection
    if x_max:
        idx_x = [(np.abs(x + x_max)).argmin(), (np.abs(x - x_max)).argmin()]
        print(idx_x, ls_slice.shape)
        ls_slice = ls_slice[:, idx_x[0]:idx_x[1]]
        extent_zx[2:] =  [x[idx_x[0]] - dx/2, x[idx_x[1]] + dx/2]
    if z_max:
        z_temp = z - np.mean(z)
        idx_z = [(np.abs(z_temp + z_max)).argmin(), (np.abs(z_temp - z_max)).argmin()]
        ls_slice = ls_slice[idx_z[0]:idx_z[1], :]
        extent_zx[:2] =  [z[idx_z[0]] - dz/2, z[idx_z[1]] + dz/2]

    ls_slice=ls_slice/ls_slice.max()
    # Define maximum for symmetric colorbar
    im = ax.imshow(ls_slice,
                   cmap="hot",
                   norm=PowerNorm(gamma=0.75,vmin=0,vmax=1,),
                #    extent=extent_zx,
                #    vmin=0,
                #    vmax=1,
                   aspect = dz/dx)

    ax.tick_params("both",
                   labelbottom=False, labelleft=False, labeltop=False,
                   bottom=False, left=False,
                   labelsize=10)

    # Cbar axes
    cax = fig.add_subplot(grid[6])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("A.U.", rotation="vertical", labelpad=5, fontsize=11)

    # Adjust the layout to align all elements properly
    plt.subplots_adjust(top=0.75, bottom=0.28, right=0.90, left=0.25)
    plt.show()

    if save_path:
        fig.savefig(save_path)
    if showfig:
        fig.show()
    else: plt.close()


data_path = Path(
    r"/mnt/server1/extFOV/remote_focus_data/20240821_145235_remote_focus_results/models_for_figure.npy"
    )

# # load list of result_dicts
# print("Opening selected heatmap results . . .")
# tl_start = time.time()
# models = np.load(data_path, allow_pickle=True)
# print(f"Loading data takes: {(time.time() - tl_start):.2f} seconds")


for _r in tqdm(models[:3], desc="Generating wavefront summaries"):
    plot_summary(_r,
                 data_path.parent / Path("plots/") / Path(f"summary_{_r['cuvette_offset']:.2f}mm_ df{_r['focus_shift']:.2f}mm.pdf"),
                 showfig=True)
