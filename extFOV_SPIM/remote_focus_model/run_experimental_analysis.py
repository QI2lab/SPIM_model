"""
Run analysis on remote foucs optimization data.
- Beads imaged with static light sheet.
- Results come from running localization using locoalize_psf package.

To complete analysis:
1. Combine psf fit results for each experimental configuration.
2. Smooth results using median and mean filter.
3. Calculate the focal extent using FWHM of the min PSF sz.
4. Plot results and save as *.npy.

Steven Sheppard 08/29/2024
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
import time
from tqdm import tqdm, trange

logging.getLogger('matplotlib').setLevel(logging.WARNING)
fig_version = True
t_start = time.time()
#------------------------------------------------------------------------------#
# Manage results and saving directories
result_path = Path(
    r"C:\Users\Steven\Documents\qi2lab\github\SPIM_model\data\paper_results\localization_results.npy"
    )
save_dir = result_path.parent

#------------------------------------------------------------------------------#
# Load results.
print("Loading localization results . . .")
tload = time.time()
results = np.load(result_path, allow_pickle=True)
print(f"Loading results took: {(time.time()-tload):.2f} seconds")

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
median_width = 25
mean_width = 5
row_idx = 0
cuv_results = []
for cuv_ii in trange(len(results), desc="Iterating over cuvette offsets", leave=True):
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
    cuv_dict = {}
    for _v in etl_volt_keys:
        cuv_dict[_v] = {}
        cuv_dict[_v]["sz"] = []
        cuv_dict[_v]["cy"] = []
        cuv_dict[_v]["cx"] = []
        cuv_dict[_v]["data"] = []
        cuv_dict[_v]["dc"] = None
        cuv_dict[_v]["sz_smooth"] = None
        cuv_dict[_v]["length"] = None
        cuv_dict[_v]["dc"] = None
        cuv_dict[_v]["left"] = None
        cuv_dict[_v]["right"] = None

    #--------------------------------------------------------------------------#
    # Loop through results and compile [sz, cy] results for each UNIQUE ETL volt.
    for etl_ii in range(len(etl_volts)):
        _r = results[cuv_ii][etl_ii]
        etl_volt = _r["etl_volt"]
        to_keep = _r["to_keep"]
        sz_raw = _r["fit_params"][to_keep, 5]
        centers = _r["fit_params"][to_keep][:, (3, 2, 1)]
        cy = centers[:,1]
        cx = centers[:,2]
        cuv_dict[etl_volt]["sz"].append(sz_raw)
        cuv_dict[etl_volt]["cy"].append(cy)
        cuv_dict[etl_volt]["cx"].append(cx)

    #--------------------------------------------------------------------------#
    # Loop through cuv_dict and combine results to single array to display
    for pos_ii, etl_volt in tqdm(enumerate(etl_volt_keys),
                                 desc="Loading localization results for each ETL position:",
                                 leave=False):

        cy = np.concatenate(cuv_dict[etl_volt]["cy"])
        cx = np.concatenate(cuv_dict[etl_volt]["cx"])
        sz_raw = np.concatenate(cuv_dict[etl_volt]["sz"])

        # re-order results before smoothing
        order = np.argsort(cy)
        cy = cy[order]
        cx = cx[order]
        sz_raw = sz_raw[order]

        # Smooth by applying median and mean filters
        sz = median_filter(sz_raw, size=median_width)
        sz = mean_convolution(sz, window_width=mean_width)
        cuv_dict[etl_volt]["sz_smooth"] = sz
        cuv_dict[etl_volt]["sz"] = sz_raw
        cuv_dict[etl_volt]["cx"] = cx
        cuv_dict[etl_volt]["cy"] = cy

        #----------------------------------------------------------------------#
        # Estimate the range of in focus beads.
        fp = np.argmin(sz)
        max_sz = (np.max(sz) - np.min(sz))*0.5 + np.min(sz)

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

        # calculate focal extent
        length = np.abs(right - left)
        length_um = length * dxy
        cuv_dict[etl_volt]["length"] = length_um
        cuv_dict[etl_volt]["left"] = left
        cuv_dict[etl_volt]["right"] = right

        #----------------------------------------------------------------------#
        # load results representing one of the ETL positions, to use max proj.
        matching_etl_idx = np.where(etl_volts==etl_volt)[0]

        etl_ii = np.where(etl_volts==etl_volt)[0][-1]
        r = results[cuv_ii][etl_ii]
        cuv_pos = _r["cuvette_offset"]
        img = _r["data"]
        yy, xx = get_coords(img.shape[1:], (dxy, dxy))

        cuv_dict[etl_volt]["data"] = img
        cuv_dict[etl_volt]["dc"] = cuv_pos

        #----------------------------------------------------------------------#
        # Plot in the first column + colorbar.
        pos_ax = pos_mapping[pos_ii]
        title_str = "".join([f"$dz_c$={cuv_pos:.0f}mm, ",
                             f"ETL power={etl_volt:.2f}V ",
                             f"Extent = {length_um/1e3:.2f}mm",
                             f"min sz = {np.min(sz_raw):.2f}"])
        ax_im = figh.add_subplot(grid[cuv_ii,pos_ax + 0])
        ax_cbar = figh.add_subplot(grid[cuv_ii,pos_ax + 2])

        # Define the image coordinates to show real extent
        dx = xx[0, 1] - xx[0, 0]
        dy = yy[1, 0] - yy[0, 0]
        extent_xy = [xx.min() - 0.5 * dx, xx.max() - 0.5 * dx,
                     yy.max() - 0.5 * dy, yy.min() - 0.5 * dy]

        # Plot maximum projection
        img_max_proj = np.nanmax(img, axis=0)
        vmin = np.percentile(img_max_proj, 0.01)
        vmax = np.percentile(img_max_proj, 99.9)
        im = ax_im.imshow(img_max_proj,
                            norm=PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax),
                            cmap="bone",
                            origin="upper",
                            extent=extent_xy,
                            aspect="equal")

        xlim = ax_im.get_xlim()
        ylim = ax_im.get_ylim()

        # Plot bead centers on max projection
        vmin = 1
        vmax = 10
        cmap_color = colormaps.get_cmap("autumn")
        cs = cmap_color((sz - vmin) / (vmax - vmin))
        ax_im.scatter(cx,
                      cy,
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
        ax_sz.plot(sz_raw, cy,
                    c="r", marker=".", markersize=1, linestyle="none")
        ax_sz.plot(sz, cy,
                    c="b", marker=".", markersize=2, linestyle="-")
        ax_sz.axhline(y=left, c="limegreen")
        ax_sz.axhline(y=right, c="limegreen")

        ax_sz.set_xlabel("$\sigma_z(y)$")
        ax_sz.set_xlim(0,15)

    cuv_results.append(cuv_dict)

print("Saving updated analysis results . . . ")
# saving updated results with lengths
np.save(save_dir / Path("combined_zstack_results.npy"), cuv_results, allow_pickle=True)
figh.savefig(save_dir / Path("summary_combined.pdf"))

print(f"Localization analysis took {(time.time()- t_start):.2f} seconds")