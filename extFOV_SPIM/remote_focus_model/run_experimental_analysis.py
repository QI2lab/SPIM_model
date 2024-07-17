"""
Calculate the focal region using localization results of bead data.
Compile results and plot summary figure.

Steven Sheppard 06/03/2024
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
from matplotlib import colormaps

from pathlib import Path
from localize_psf.localize import get_coords
from model_tools.analysis import mean_convolution
from scipy.ndimage import median_filter

logging.getLogger('matplotlib').setLevel(logging.WARNING)
fig_version = True

#------------------------------------------------------------------------------#
# Manage results and saving directories
result_path = Path(
    r"E:\SPIM_data\bead_data\SJS\optimization\20240716_114052_paper_results\localization_results.npy"
    )
save_dir = result_path.parent

#------------------------------------------------------------------------------#
# Load results.
results = np.load(result_path, allow_pickle=True)

# Compile the number of figure elements.
non_none_count = sum(r is not None for cuv in results for r in cuv)
num_cuv_pos = len(results)
num_etl_pos = 3
dxy = 4.25 / 2

#------------------------------------------------------------------------------#
# Setup figure.
fig_height = 1 + 4.0 * num_cuv_pos
fig_width = 6.5 * num_etl_pos
height_ratios = [1] * num_cuv_pos
width_ratios=[1,0.5,0.1] * num_etl_pos
pos_mapping = [0, 3, 6]
figh = plt.figure(figsize=(fig_width, fig_height))
grid = figh.add_gridspec(nrows=len(height_ratios),
                         ncols=len(width_ratios),
                         width_ratios=width_ratios,
                         height_ratios=height_ratios,
                         hspace=0.3,
                         wspace=0.3)

#------------------------------------------------------------------------------#
# Iterate through results and plot the bead image and sz results.
median_width = 35
mean_width = 7
row_idx = 0
cuv_results = []
for cuv_ii in range(len(results)):
    #--------------------------------------------------------------------------#
    # Compile the ETL volts to identify unique positions.
    etl_volts = []
    for etl_ii, etl_r in enumerate(results[cuv_ii]):
        if etl_r is not None:
            etl_volts.append(etl_r["etl_volt"])
    etl_volts = np.array(etl_volts)
    etl_order = np.argsort(etl_volts)

    # Create an empty dictionary to append ETL volt results.
    # The dictionary keys are the unique ETL volts,
    # and are used later to identify the desired key
    etl_volt_keys = np.unique(etl_volts[etl_order])[::-1]
    etl_dict = {}
    for _v in etl_volt_keys:
        etl_dict[_v] = {}
        etl_dict[_v]["sz"] = []
        etl_dict[_v]["cy"] = []
        etl_dict[_v]["sz_smooth"] = None
        etl_dict[_v]["cy_smooth"] = None
        etl_dict[_v]["length"] = None
        etl_dict[_v]["data"] = None

    #--------------------------------------------------------------------------#
    # Loop through results and compile [sz, cy] results for each UNIQUE ETL volt.
    for etl_ii in range(len(etl_volts)):
        r = results[cuv_ii][etl_ii]
        etl_volt = r["etl_volt"]
        to_keep = r["to_keep"]
        sz_raw = r["fit_params"][to_keep, 5]
        centers = r["fit_params"][to_keep][:, (3, 2, 1)]
        cy_raw = centers[:,1]
        etl_dict[etl_volt]["sz"].append(sz_raw)
        etl_dict[etl_volt]["cy"].append(cy_raw)

    #--------------------------------------------------------------------------#
    # Loop through etl_dict and combine results to single array to display
    for pos_ii, etl_volt in enumerate(etl_volt_keys):
        cy_raw = np.concatenate(etl_dict[etl_volt]["cy"])
        sz_raw = np.concatenate(etl_dict[etl_volt]["sz"])

        # re-order results before smoothing
        order = np.argsort(cy_raw)
        cy = cy_raw[order]
        sz = sz_raw[order]
        sz = median_filter(sz, size=median_width)
        sz = mean_convolution(sz, window_width=mean_width)

        etl_dict[etl_volt]["sz_smooth"] = sz
        etl_dict[etl_volt]["cy_smooth"] = cy

        #----------------------------------------------------------------------#
        # Estimate the range of in focus beads.
        fp = np.argmin(sz)
        max_sz = (np.max(sz)-np.min(sz)) * 0.5 + np.min(sz)

        try:
            left = cy[np.max(np.where(sz[:fp] >= max_sz))]
        except Exception as e:
            print(f"Cuv. {cuv_ii}, volt:{etl_volt} edge not found ")
            left = cy[0]
        try:
            right = cy[fp + np.min(np.where(sz[fp:] >= max_sz))]
        except Exception as e:
            print(f"Cuv. {cuv_ii}, volt:{etl_volt} edge not found ")
            right = cy[-1]

        length = np.abs(right - left)
        length_um = length * dxy

        etl_dict[etl_volt]["length"] = length_um

        #----------------------------------------------------------------------#
        # load results representing one of the ETL positions, to use max proj.
        etl_ii = np.where(etl_volts==etl_volt)[0][0]
        r = results[cuv_ii][etl_ii]
        cuv_pos = r["cuvette_offset"]
        img = r["data"]
        yy, xx = get_coords(img.shape[1:], (dxy, dxy))

        #----------------------------------------------------------------------#
        # Plot in the first column + colorbar.
        pos_ax = pos_mapping[pos_ii]
        title_str = "".join([f"$dz_c$={cuv_pos:.0f}mm, ",
                             f"ETL power={etl_volt:.2f}V ",
                             f"Extent = {length_um/1e3:.2f}mm"])
        ax_im = figh.add_subplot(grid[cuv_ii,pos_ax + 0])
        ax_cbar = figh.add_subplot(grid[cuv_ii,pos_ax + 2])

        # Define the image coordinates to show real extent
        dx = xx[0, 1] - xx[0, 0]
        dy = yy[1, 0] - yy[0, 0]
        extent_xy = [xx.min() - 0.5 * dx, xx.max() + 0.5 * dx,
                     yy.max() + 0.5 * dy, xx.min() - 0.5 * dy]

        # Plot maximum projection
        img_max_proj = np.nanmax(img, axis=0)
        vmin = np.percentile(img_max_proj, 0.01)
        vmax = np.percentile(img_max_proj, 99.9)
        im = ax_im.imshow(img_max_proj,
                            norm=PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax),
                            cmap="bone",
                            extent=extent_xy,
                            aspect="equal")

        xlim = ax_im.get_xlim()
        ylim = ax_im.get_ylim()

        # Plot bead centers on max projection
        vmin = 0
        vmax = 15
        cmap_color = colormaps.get_cmap("autumn")
        cs = cmap_color((sz - vmin) / (vmax - vmin))
        ax_im.scatter(centers[:, 2],
                      centers[:, 1],
                      facecolor='none',
                      edgecolor=cs,
                      marker='o')
        ax_im.set_title(title_str)
        ax_im.tick_params(axis="both", labelsize=10)

        # Create colorbar for sz.
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                    cmap=cmap_color),
            cax=ax_cbar)
        cbar.ax.set_ylabel("$\sigma_z$")

        # Re-enforce im limits
        ax_im.set_xlim(xlim)
        ax_im.set_ylim(ylim)

        #----------------------------------------------------------------------#
        # Plot sz vs position in second column
        ax_sz = figh.add_subplot(grid[cuv_ii, pos_ax + 1],
                                 sharey=ax_im)
        ax_sz.plot(sz_raw, cy_raw,
                    c="r", marker=".", markersize=1, linestyle="none")
        ax_sz.plot(mean_convolution(sz, 15), cy,
                    c="b", marker=".", markersize=2, linestyle="-")
        ax_sz.axhline(y=left, c="limegreen")
        ax_sz.axhline(y=right, c="limegreen")

        ax_sz.set_xlabel("$\sigma_z(y)$")
        ax_sz.set_xlim(0,15)

    cuv_results.append(etl_dict)

if fig_version:
    figh.savefig(save_dir / Path("summary_combined.png"), dpi=300)
else:
    figh.savefig(save_dir / Path("summary_combined.png"))
