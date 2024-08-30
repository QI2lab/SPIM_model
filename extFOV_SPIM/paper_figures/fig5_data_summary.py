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
import time

logging.getLogger('matplotlib').setLevel(logging.WARNING)
fig_version = True
t_start = time.time()
#------------------------------------------------------------------------------#
# Manage results and saving directories
result_path = Path(
    r"C:\Users\Steven\Documents\qi2lab\github\SPIM_model\data\paper_results\combined_zstack_results.npy"
    )
save_dir = Path(r"C:\Users\Steven\Documents\qi2lab\illustrator_files\SPIM_model")

#------------------------------------------------------------------------------#
# Load results.
print("Loading localization results")
results = np.load(result_path, allow_pickle=True)

# Preselected dc position for figure
dcs_to_use = [35]

# Define experimental parameters
cuv_pos_to_plot = []
num_cuv_pos = len(dcs_to_use)
num_etl_pos = 3
dxy = 4.25 / 2
dz = 1.0

#------------------------------------------------------------------------------#
# Setup figure.
fig_height = 1.6
fig_width = 4.25
row_mapping = [0, 3, 6]

# Create figure.
figh = plt.figure(figsize=(fig_width, fig_height))
grid = figh.add_gridspec(nrows=2,
                         ncols=4,
                         width_ratios=[1,1,1,0.1],
                         height_ratios=[1,0.5],
                         hspace=0.1,
                         wspace=0.15)

# setup color map settings
sz_min = 1
sz_max = 10
cmap_color = colormaps.get_cmap("autumn")

#------------------------------------------------------------------------------#
# Iterate through results and plot the bead image and sz results.
row_idx = 0
for _r in results[::-1]:
    # Dict. keys are the etl volts
    etl_pos_keys = [_k for _k in _r.keys()]
    dc = _r[etl_pos_keys[0]]["dc"]

    if dc in dcs_to_use:
        pos_ax = row_mapping[row_idx]
        #----------------------------------------------------------------------#
        # Loop through ETL volts and plot fig row
        for jj, etl_v in enumerate(etl_pos_keys):
            #------------------------------------------------------------------#
            # Grab values from results
            cy = _r[etl_v]["cy"]
            cx = _r[etl_v]["cx"]
            sz_raw = _r[etl_v]["sz"]
            sz = _r[etl_v]["sz_smooth"]
            left = _r[etl_v]["left"]
            right = _r[etl_v]["right"]
            length = _r[etl_v]["length"]
            img = _r[etl_v]["data"]
            dc = _r[etl_v]["dc"]

            print(f"Coords y:{cy.min()} {cy.max()}")
            # define coords of interest
            fp = np.argmin(sz)
            max_sz = (np.max(sz)-np.min(sz)) * 0.5 + np.min(sz)
            yy, xx = get_coords(img.shape[1:], (dxy, dxy))

            #------------------------------------------------------------------#
            # Plot in the first column + colorbar.
            # create axes for image and colorbar
            ax_im = figh.add_subplot(grid[pos_ax, jj])
            ax_sz = figh.add_subplot(grid[pos_ax+1, jj], sharex=ax_im)

            # configure axes setting
            ax_sz.tick_params(axis="both", labelsize=12,
                              bottom=False, left=False,
                              labelbottom=False, labelleft=False)
            ax_im.tick_params(axis="both", labelsize=12,
                              bottom=False, left=False,
                              labelbottom=False, labelleft=False)

            # Plot bead centers on max projection
            cs = cmap_color((sz - sz_min) / (sz_max - sz_min))
            ax_im.scatter(cy, cx, sz,
                          facecolor='none', edgecolor=cs,
                          marker='o', linewidths=0.5, alpha=0.85)
            # ax_im.set_aspect(dz / dxy)
            ax_im.set_facecolor("k")

            # # Re-enforce im limits
            ax_im.set_xlim([0, 4800])

            #----------------------------------------------------------------------#
            # plot the bead axial positions and sigma z results.
            # ax_sz.plot(cy, sz_raw, c="r", marker=".", markersize=.5, linestyle="none")
            ax_sz.plot(cy, sz, c="b", marker=".", markersize=0.5, linestyle="-")
            ax_sz.axvline(x=left, c="k")
            ax_sz.axvline(x=right, c="k")

            ax_sz.set_ylim(0,18)
            ax_sz.set_xlim([0, 4800])

        row_idx += 1

# Create colorbar for sz.
ax_cbar =  figh.add_subplot(grid[:, -1])
cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=Normalize(vmin=sz_min, vmax=sz_max),
                        cmap=cmap_color),
    cax=ax_cbar)
cbar.set_ticks([2, 4, 6, 8, 10])
cbar.set_label(r"$\sigma_z$ ($\mu m$)")
ax_cbar.tick_params("both", labelsize=14)

# Force plot to fit.
figh.subplots_adjust(left=0.01, right=0.85,
                     top=0.95, bottom=0.05)
print("Saving updated analysis results . . . ")

figh.savefig(save_dir / Path("fig5_beadlocalizations.pdf"))
