"""
Load experimental data imaging light sheet using agar embedded beads.
Localize and fit beads using localize_psf package.

Steven Sheppard 06/03/2024
"""

# Imports
import numpy as np
from pathlib import Path
from ndtiff import Dataset
from scipy.ndimage import median_filter
from model_tools.analysis import mean_convolution
import time

# localize_psf imports
from localize_psf.fit_psf import gaussian3d_psf_model, average_exp_psfs
from localize_psf.localize import (localize_beads_generic,
                                   get_param_filter,
                                   filter,
                                   get_coords,
                                   plot_fit_roi,
                                   plot_bead_locations)
from localize_psf.fit import fit_model

# Model tools imports
from model_tools.raytrace import get_unique_dir
import dask

# Napari / plotting imports
import matplotlib.pyplot as plt
import napari
from napari.settings import get_settings
# Force program to wait for interactive windows to close
n_settings = get_settings()
n_settings.application.ipy_interactive = False

import logging
mat_logger = logging.getLogger('matplotlib')
mat_logger.setLevel(logging.WARNING)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# Plotting flags
DEBUG = False
showfig = False

t_initial = time.time()

#------------------------------------------------------------------------------#
# create a custom filter for localize_psf
class boundary_filter(filter):
    """
    Filter based on distance to boundary and fit sigma_z
    """
    def __init__(self,
                 coords,
                 sf = 2):
        """
        Define a filter based on the bead position and z-size.
        """
        self.condition_names = ["z position and PSF too small",
                                "z position and PSF too large"]
        self.coords = coords
        self.sf = sf

    def filter(self,
               fit_params: np.ndarray,
               *args,
               **kwargs):

        z, y, x = self.coords
        conditions = np.stack((fit_params[:, 3] >= fit_params[:, 5]*self.sf + z.min(),
                               fit_params[:, 3] <= z.max() - fit_params[:, 5]*self.sf + z.max()),
                              axis=1)

        return conditions


#------------------------------------------------------------------------------#
# Manage data and saving paths.
root_path = Path(r"E:\SPIM_data\bead_data\SJS\optimization")
savedir = get_unique_dir(root_path,
                         "paper_results")

# Data was taken such that each cuvette position is it's own root name.
cuv_pos_dirs = [_d for _d in root_path.iterdir() if "cuv" in _d.name]
cuv_positions = [
    int(p.stem.split('_')[-1].replace('mm', '')) for p in cuv_pos_dirs
    ]

position_mapping = {}
for pos in cuv_positions:
    position_mapping[f"{pos}"] = []

#------------------------------------------------------------------------------#
# Define mapping between cuvette positions and experimental offsets
cuvette_mapping = {"0":17., "1":20., "2":25., "3":30.,
                   "4":35., "5":10., "6":15.
                   }
etl_mapping = {"10":{"7":6.75,"9":6.2,"10":6.2,"11":5.73,"12":5.73},
               "15":{"1":5.7,"2":5.7,"3":6.34,"4":6.34,"5":5.3,"6":5.3},
               "25":{"1":5.13,"2":5.13,"3":4.79,"4":4.79,"6":4.3},
               "20":{"1":5.74,"2":5.74,"3":5.3,"4":5.3,"5":4.75,"6":4.75},
               "30":{"1":4.22,"2":4.22,"3":4.75,"4":4.75,"5":3.8,"6":3.8},
               "35":{"1":4.25,"2":4.25,"3":3.75,"4":3.75,"5":3.33,"6":3.33}
               }

# Optionally bin over Z. Make sure to apply to
# Create a figure for displaying the light sheet and cuvette positions.
figh_pos = plt.figure()
grid_pos = figh_pos.add_gridspec(nrows=1,
                                 ncols=1,
                                 width_ratios=[1],
                                 height_ratios=[1],
                                 hspace=0.15,
                                 wspace=0.15
                                 )
ax_pos = figh_pos.add_subplot(grid_pos[0])
ax_pos.set_title("Compiled cuvette offsets and ETL volts")
ax_pos.set_xlabel("ETL volts (V)")
ax_pos.set_ylabel("Cuvette Offset (mm)")
cuv_colors = plt.get_cmap("tab10").colors[:len(cuv_positions)]

for cuv_idx, cuv_offset in enumerate(etl_mapping.keys()):
    for volt in etl_mapping[cuv_offset].values():
        ax_pos.plot(volt, float(cuv_offset),
                    color=cuv_colors[int(cuv_idx)],
                    linestyle="none",
                    markersize=3,
                    marker="x")

# Show/save figure showing stage positions
if savedir:
    figh_pos.savefig(savedir / Path("experimental_positions.png"))
if showfig:
    figh_pos.show()
else:
    plt.close(figh_pos)

#------------------------------------------------------------------------------#
# Plotting and anaylsis helper functions.
def plot_average_psf(r,
                     save_path: Path=None,
                     label: str = "",
                     figsize: tuple[int, int] = (20,20),
                     showfig: bool = False):
    """
    Plot a summary PSF by averating PSF and showing fit results
    """
    to_keep = r["to_keep"]
    fit_params = r["fit_params"]
    dz = r["dz"]
    dxy = r["dxy"]
    centers = r["fit_params"][to_keep][:, (3, 2, 1)]
    img = r["data"]
    z, y, x = get_coords(img.shape, (dz, dxy, dxy))

    # Get and plot experimental PSFs
    psf_percentiles = (1, 5, 10, 50, 99)
    nps = len(psf_percentiles)
    fit_roi_size_pix = np.round(np.array(fit_roi_size) / np.array([dz, dxy, dxy])).astype(int)
    fit_roi_size_pix += (1 - np.mod(fit_roi_size_pix, 2))

    psfs_real = np.zeros((nps,) + tuple(fit_roi_size_pix))
    fit_params_real = np.zeros((nps, model.nparams))
    psf_coords = None

    for ii in range(len(psf_percentiles)):
        # only keep smallest so many percent of spots
        sigma_max = np.percentile(fit_params[:, 4][to_keep],
                                  psf_percentiles[ii])
        to_use = np.logical_and(to_keep, fit_params[:, 4] <= sigma_max)

        # get centers
        centers = np.stack((fit_params[:, 3][to_use],
                            fit_params[:, 2][to_use],
                            fit_params[:, 1][to_use]), axis=1)

        # find average experimental psf/otf
        psfs_real[ii], psf_coords = average_exp_psfs(
            img,
            (z, y, x),
            centers,
            fit_roi_size_pix,
            backgrounds=fit_params[:, 5][to_use],
            return_psf_coords=True
            )

        # fit average experimental psf
        def fn(p): return model.model(psf_coords, p)
        init_params = model.estimate_parameters(psfs_real[ii], psf_coords)

        results = fit_model(psfs_real[ii], fn, init_params, jac='3-point', x_scale='jac')
        fit_params_real[ii] = results["fit_params"]

        figh = plot_fit_roi(fit_params_real[ii],
                            [0, psfs_real[ii].shape[0],
                            0, psfs_real[ii].shape[1],
                            0, psfs_real[ii].shape[2]],
                            psfs_real[ii],
                            psf_coords,
                            model=model,
                            string=f"smallest {psf_percentiles[ii]:.0f} percent,"
                                f" {type(model)}, sf={model.sf}",
                            vmin=0,
                            vmax=1,
                            gamma=0.5,
                            figsize=figsize)

        if save_path:
            figh.savefig(Path(save_path) / Path(label + f"_psf_smallest_{psf_percentiles[ii]:.2f}.png"))

        if showfig:
            plt.show(figh)
        else:
            plt.close(figh)


def plot_localization_fit_summary(r,
                                  save_path: Path = None,
                                  showfig: bool = False):
    """
    Helper function for plotting localize_beads output
    """
    to_keep = r["to_keep"]
    sz = r["fit_params"][to_keep, 5]
    sxy = r["fit_params"][to_keep, 4]
    amp = r["fit_params"][to_keep, 0]
    bg = r["fit_params"][to_keep, 6]
    centers = r["fit_params"][to_keep][:, (3, 2, 1)]
    cx = centers[:,2]
    cy = centers[:,1]
    cy_mf = r["cy_mf"]
    sz_mf = r["sz_mf"]

    width_ratios=[1,0.7,0.1,0.1,0.1]
    height_ratios=[1,1,0.1,0.5,0.5,0.5,0.5,0.5]
    figh_sum = plt.figure(figsize=(8,18))
    grid_sum = figh_sum.add_gridspec(nrows=len(height_ratios),
                                     ncols=len(width_ratios),
                                     width_ratios=width_ratios,
                                     height_ratios=height_ratios,
                                     hspace=0.2,
                                     wspace=0.3
                                     )

    # Plot maximum projection with heatmap for sz
    ax_proj_sz = figh_sum.add_subplot(grid_sum[0,:2])
    ax_cmap_i_sz = figh_sum.add_subplot(grid_sum[0,2])
    ax_cmap_sz = figh_sum.add_subplot(grid_sum[0,4])
    figh_sum = plot_bead_locations(img_filtered,
                                    centers,
                                    weights=[r["fit_params"][to_keep, 5]],
                                    color_lists=["autumn"],
                                    color_limits=[[1,15.0]],
                                    cbar_labels=["$\sigma_z$"],
                                    title="Max intensity projection with Sz",
                                    coords=coords_2d,
                                    gamma=0.5,
                                    axes=[ax_proj_sz, ax_cmap_i_sz, ax_cmap_sz]
                                    )
    ax_proj_sxy = figh_sum.add_subplot(grid_sum[1,:2])
    ax_cmap_i_sxy = figh_sum.add_subplot(grid_sum[1,2])
    ax_cmap_sxy = figh_sum.add_subplot(grid_sum[1,4])
    figh_sum = plot_bead_locations(img_filtered,
                                    centers,
                                    weights=[r["fit_params"][to_keep, 4]],
                                    color_lists=["autumn"],
                                    color_limits=[[1.0,6.0]],
                                    cbar_labels=[r"$\sigma_{xy}$"],
                                    title="Max intensity projection with Sxy",
                                    coords=coords_2d,
                                    gamma=0.5,
                                    axes=[ax_proj_sxy, ax_cmap_i_sxy, ax_cmap_sxy]
                                    )
    ax_proj_sz.tick_params(labelbottom=False)
    ax_proj_sxy.set_title("")

    # Create axes for plotting x, y specific results
    ax_sz_cx = figh_sum.add_subplot(grid_sum[3,0])
    ax_sz_cy = figh_sum.add_subplot(grid_sum[3,1:],sharey=ax_sz_cx)
    ax_sxy_cx = figh_sum.add_subplot(grid_sum[4,0],sharex=ax_sz_cx)
    ax_sxy_cy = figh_sum.add_subplot(grid_sum[4,1:],
                                     sharey=ax_sxy_cx,sharex=ax_sz_cy)
    ax_amp_cx = figh_sum.add_subplot(grid_sum[5,0],sharex=ax_sz_cx)
    ax_amp_cy = figh_sum.add_subplot(grid_sum[5,1:],
                                     sharey=ax_amp_cx,sharex=ax_sz_cy)
    ax_bg_cx = figh_sum.add_subplot(grid_sum[6,0],sharex=ax_sz_cx)
    ax_bg_cy = figh_sum.add_subplot(grid_sum[6,1:],
                                    sharey=ax_bg_cx,sharex=ax_sz_cy)
    ax_sxy_sz = figh_sum.add_subplot(grid_sum[7,0])
    ax_amp_sz = figh_sum.add_subplot(grid_sum[7,1:], sharex=ax_sxy_sz)

    ax_sz_cx.set_ylabel("$\sigma_z$ ($\mu m$)")
    ax_sxy_cx.set_ylabel("$\sigma_{xy}$ ($\mu m$)")
    ax_amp_cx.set_ylabel("amplitude")
    ax_bg_cx.set_ylabel("background")
    ax_bg_cx.set_xlabel("$C_x$")
    ax_bg_cy.set_xlabel("$C_y$")
    ax_sxy_sz.set_ylabel("$\sigma_{xy}$")
    ax_sxy_sz.set_xlabel("$\sigma_z$")
    ax_amp_sz.set_ylabel("amplitude")
    ax_amp_sz.set_xlabel("$\sigma_z$")
    for ax in [ax_sz_cy,ax_sxy_cy,ax_amp_cy,ax_bg_cy]:
        ax.tick_params(labelleft=False)
    for ax in [ax_sz_cx,ax_sz_cy,ax_sxy_cx,ax_sxy_cy,ax_amp_cx,ax_amp_cy]:
        ax.tick_params(labelbottom=False)

    # Set limits for visualizing sz
    ax_sz_cx.set_ylim(0,20)
    ax_sz_cy.set_ylim(0,20)
    ax_sxy_cx.set_ylim(0,8)
    ax_sxy_cy.set_ylim(0,8)
    ax_amp_cx.set_ylim(0, 1.25 * max(amp))
    ax_amp_cy.set_ylim(0, 1.25 * max(amp))
    ax_bg_cx.set_ylim(0, 1.5 * max(bg))
    ax_bg_cy.set_ylim(0,1.5 * max(bg))
    ax_sxy_sz.set_ylim(0,8)
    ax_amp_sz.set_ylim(0,8)

    # Plot directional results
    ax_sz_cx.plot(cx, sz, c="b", marker=".", markersize=2, linestyle="none")
    ax_sz_cy.plot(cy, sz, c="b", marker=".", markersize=2, linestyle="none")
    ax_sz_cy.plot(cy_mf, sz_mf,
                  c="r", marker="x", markersize=1, linestyle="--", linewidth=1)
    ax_sxy_cx.plot(cx, sxy, c="b", marker=".", markersize=3, linestyle="none")
    ax_sxy_cy.plot(cy, sxy, c="b", marker=".", markersize=3, linestyle="none")
    ax_amp_cx.plot(cx, amp, c="b", marker=".", markersize=3, linestyle="none")
    ax_amp_cy.plot(cy, amp, c="b", marker=".", markersize=3, linestyle="none")
    ax_bg_cx.plot(cx, bg, c="b", marker=".", markersize=3, linestyle="none")
    ax_bg_cy.plot(cy, bg, c="b", marker=".", markersize=3, linestyle="none")
    ax_sxy_sz.plot(sz, sxy, c="b", marker=".", markersize=3, linestyle="none")
    ax_amp_sz.plot(sz, amp, c="b", marker=".", markersize=3, linestyle="none")

    if showfig:
        figh_sum.show()
    else:
        plt.close(figh_sum)
    if save_path:
        figh_sum.savefig(save_path, dpi=150)


def apply_filters(r,
                  median_width: int = 20,
                  mean_width: int = 5):
    """
    Crop to ROI and apply median filter to sz
    Return the smoothed signal and the matching centers.
    """
    # Compile results to plot
    # Grab values of interest
    to_keep = r["to_keep"]
    sz = r["fit_params"][to_keep, 5]
    centers = r["fit_params"][to_keep][:, (3, 2, 1)]
    cx = centers[:,2]
    cy = centers[:,1]

    # First sort along x axis to crop down the hieght
    order = np.argsort(cx)
    cx = cx[order]
    cy = cy[order]
    sz = sz[order]

    # Re-order along y for smoothing
    order = np.argsort(cy)
    cy = cy[order]
    sz = sz[order]

    # smooth sz using mean convolution kernel
    sz_mf = median_filter(sz, size=median_width)
    sz_mf = mean_convolution(sz_mf, window_width=mean_width)
    r["sz_mf"] = sz_mf
    r["cy_mf"] = cy

    return r


def plot_fit_rois(r,
                  num_localizations_to_plot: int = None,
                  figsize: tuple[int, int] = (10,7.5),
                  savedir: Path = None):

    to_keep = r["to_keep"]
    dz = r["dz"]
    dxy = r["dxy"]
    coords_3d = get_coords(img.shape, (dz, dxy, dxy))

    if num_localizations_to_plot is not None:
        ind_to_plot = np.arange(len(to_keep), dtype=int)[to_keep][:num_localizations_to_plot]
    else:
        ind_to_plot = np.arange(len(to_keep), dtype=int)[to_keep]

    prefix = f"{r['cuvette_offset']}mm_{r['etl_volt']}V"
    delayed = []
    for ind in ind_to_plot:
        delayed.append(dask.delayed(plot_fit_roi)(
            r["fit_params"][ind],
            r["rois"][ind],
            r["data"],
            coords_3d,
            r["init_params"][ind],
            figsize=figsize,
            prefix=prefix + f"_localization_roi_{ind}",
            string=("filter conditions = "
                    + " ".join(["%d," % c for c in r["conditions"][ind]])),
            save_dir=savedir
            )
                       )

    results = dask.compute(*delayed)


#------------------------------------------------------------------------------#
# Acquisition parameters
wl = 0.473
dxy = 4.25 / 2
na = 0.5

# Setup filtering parameters
filter_sigma_small = None # (1.0, 2., 2.)
filter_sigma_large = None # (7.0, 6.0, 6.0)
min_spot_sep = (4., 4.)

# set fit rejection thresholds
threshold = 2000
amp_bounds = (1000, np.inf)
sxy_bounds = (1.5, 8.0)
sz_bounds = (0.5, 15)
fit_dist_max_err = (15.0, 10.0)
fit_roi_size = (35.0, 25, 25)
dist_boundary_min = (1.0, 1.0)
axial_crop = 400
top_crop = 1000
bot_crop = 800
model = gaussian3d_psf_model()

#------------------------------------------------------------------------------#
# Run loading and localizations
results = np.full((len(etl_mapping),6), None)
for cuv_idx, cuv_pos_dir in enumerate(cuv_pos_dirs):
    # define the cuvette offset distance
    cuv_offset = cuv_positions[cuv_idx]

    # Grab the data directories and their position keys
    position_dirs = [_d for _d in cuv_pos_dir.iterdir() if "pos" in _d.name]
    pos_dir_keys = [_dir.name.split('_')[1] for _dir in position_dirs]

    print(f"\rRunning cuvette offset: {cuv_idx}/{len(cuv_pos_dirs)}", end="\n")
    for pos_idx, data_path in enumerate(position_dirs):
        print(f"\rLight sheet offset {pos_idx}/{len(position_dirs)}", end="")
        t_pos = time.time()

        #----------------------------------------------------------------------#
        # Open dataset, load data and imaging parameters
        dataset = Dataset(str(data_path))

        # There were atleast 2 acquisitions with only 1 channel, this deals with
        # any differences in shape and loads the raw data into a numpy array.
        if len(dataset.axes["channel"])==2:
            raw_img = np.asarray(dataset.as_array(axes=["position",
                                                        "time",
                                                        "channel",
                                                        "z"])
                             )[0,0,:,:,:]
            ch_id = np.argmax([raw_img[0,0,:,:].max(), raw_img[1,0,:,:].max()])
            raw_img = raw_img[ch_id,:,:,:]
        else:
            ch_id = 0
            raw_img = np.asarray(dataset.as_array(axes=["position",
                                                        "time",
                                                        "channel",
                                                        "z"])
                             )[0,0,0,:]

        # Catch any case where the above does not return a z-stack.
        if len(raw_img.shape)!=3:
            print(f"Raw data shape is not as expected: {raw_img.shape}")
        elif raw_img.max()<500:
            print(f"img maximum < 500, check selected channel: {ch_id}")

        # Crop down img for computation savings and avoiding out of focus edges
        temp = raw_img - raw_img.mean()
        x_sum = temp[0,:,:].sum(axis=0)
        ls_height_extent = np.where(np.diff(np.sign(x_sum)))[0]
        img = raw_img[:,
                      axial_crop:-axial_crop:,
                      min(ls_height_extent)+top_crop:max(ls_height_extent)-bot_crop]
        # Compile imaging parameters
        metadata = dataset.read_metadata(channel=ch_id,
                                         z=0,
                                         time=0,
                                         position=0)
        dz = dataset.summary_metadata["z-step_um"]
        etl_volt = etl_mapping[str(cuv_offset)][pos_dir_keys[pos_idx]]

        if DEBUG==True:
            # Plot the cropping results
            figh_crop, ax = plt.subplots(1, 1, figsize=(6,4))
            ax.set_title("Cropping to light sheet extent")
            ax.set_xlabel("y index")
            ax.set_ylabel("Sum over x")
            ax.plot(x_sum, c="b", ls="-", label="sum")
            ax.axvline(x=min(ls_height_extent), c="r", label="crop index")
            ax.axvline(x=max(ls_height_extent), c="r")
            ax.legend(fontsize=10)
            if savedir:
                figh_crop.savefig(savedir / Path(
                    f"debug_with_cropping_{cuv_offset}_{etl_volt}."
                    )
                                  )
            if showfig:
                figh_crop.show()
            else:
                plt.close(figh_crop)

            # Show the raw data and cropped results in Napari.
            viewer = napari.Viewer()
            viewer.add_image(img,
                                name="cropped img",
                                interpolation2d="nearest",
                                interpolation3d="nearest")
            viewer.add_image(raw_img,
                                name="raw img",
                                interpolation2d="nearest",
                                interpolation3d="nearest")
            viewer.add_image(img_filtered,
                                name="filtered img",
                                interpolation2d="nearest",
                                interpolation3d="nearest")

            napari.run()

        del raw_img, temp
        dataset = None

        #----------------------------------------------------------------------#
        print("\nStarting localization...")
        # Define coordinates to pass to localization
        coords_3d = get_coords(img.shape, (dz, dxy, dxy))
        coords_2d = get_coords(img.shape[1:], (dxy, dxy))

        # Prepare filter for localization
        param_filter = get_param_filter(coords_3d,
                                    fit_dist_max_err=fit_dist_max_err,
                                    min_spot_sep=min_spot_sep,
                                    amp_bounds=amp_bounds,
                                    dist_boundary_min=dist_boundary_min,
                                    sigma_bounds=((sz_bounds[0],
                                                    sxy_bounds[0]),
                                                    (sz_bounds[1],
                                                    sxy_bounds[1]))
                                    )
        sigma_pos_filter = boundary_filter(coords=coords_3d,
                                           sf=1.5
                                           )
        filter = sigma_pos_filter + param_filter

        # Run localization function
        _, r, img_filtered = localize_beads_generic(
            img,
            (dz, dxy, dxy),
            threshold=threshold,
            roi_size=fit_roi_size,
            filter_sigma_small=filter_sigma_small,
            filter_sigma_large=filter_sigma_large,
            min_spot_sep=min_spot_sep,
            model=model,
            filter=filter,
            max_nfit_iterations=50,
            use_gpu_fit=False,
            use_gpu_filter=False,
            return_filtered_images=True,
            fit_filtered_images=True,
            verbose=True
            )

        # Add experimental params to localization results
        r["cuvette_offset"] = cuv_offset
        r["etl_volt"] = etl_volt
        r["data"] = img_filtered
        r["dxy"] = dxy
        r["dz"] = dz

        #----------------------------------------------------------------------#
        print("\nLocalization complete, smoothing sz results")
        r = apply_filters(r, median_width=15, mean_width=5)

        # Include position results in results array
        results[cuv_idx, pos_idx] = r

        #----------------------------------------------------------------------#
        print("\nPlotting Results. . . ")
        summary_save_path = Path(savedir) / Path(
            f"fit_summary_{cuv_offset}mm_{etl_volt}.png"
            )
        rois_savedir = savedir / Path(
            f"{cuv_offset}mm_{etl_volt}V_roi_localization_plots"
            )
        rois_savedir.mkdir(exist_ok=True)

        plot_localization_fit_summary(r, summary_save_path, showfig)
        plot_fit_rois(r, 30, (20,20), rois_savedir)
        print(f"\nThis position took {(time.time() - t_pos)/(60):2f} minutes")

#------------------------------------------------------------------------------#
print("\nAnalysis complete, summarizing results... ")
print(f"\nAnalysis took {(time.time() - t_initial)/(60):2f} minutes")

# save results as numpy pickle object.
np.save(savedir / Path("localization_results.npy"),
        results,
        allow_pickle=True)
