
import model_tools.propagation as pt
from model_tools.analytic_forms import gaussian_field
from pathlib import Path
import numpy as np
import gc as gc
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

savedir_path = Path(r"C:\Users\Steven\Documents\qi2lab\illustrator_files\SPIM_model")

# Define model parameters
wl = 0.561
wo = 4
ri = 1.00
Io = 1.0
dz = 2.0
dx = 1.0
n_xy = int(200 // dx)
n_zx = int(2509.557 // dz)

# enforce odd grid numbers
if n_xy%2 == 0:
    n_xy += 1
if n_zx%2 == 0:
    n_zx += 1

grid, shared_grid_params = pt.field_grid(num_xy=n_xy,
                                         num_zx=n_zx,
                                         dx=dx,
                                         dz=dz,
                                         return_field=True
                                         )

x, radius_xy, extent_xy, z, extent_zx = shared_grid_params

# Generate the single light sheet
single_light_sheet = np.zeros(grid.shape)
scanned_light_sheet = np.zeros(grid.shape)
for ii in range(z.shape[0]):

    single_light_sheet[ii] = gaussian_field(radius_xy,
                                            z[ii],
                                            wl,
                                            wo,
                                            ri,
                                            Io)

    scanned_light_sheet[ii] = gaussian_field(radius_xy,
                                            0,
                                            wl,
                                            wo,
                                            ri,
                                            Io)

single_light_sheet = np.abs(single_light_sheet)**2
scanned_light_sheet = np.abs(scanned_light_sheet)**2
single_projection = np.max(single_light_sheet, axis=1).T
scanned_projection = np.max(scanned_light_sheet, axis=1).T

# setup figure
cmap = "hot"
aspect = "auto" # dx/dz #
label_fontsize = 12

fig = plt.figure(figsize=(10,4))
grid = fig.add_gridspec(nrows=2,
                        ncols=2,
                        width_ratios=[1,0.2],
                        height_ratios=[1.0,1.0],
                        wspace=0.2,
                        hspace=0.3)

# Plot single light sheet
ax = fig.add_subplot(grid[0,0])
im = ax.imshow(single_projection,
               norm=PowerNorm(gamma=0.5),
               cmap=cmap,
               aspect=aspect,
               extent=extent_zx)
ax.set_xlabel("")


# Plot scanned light sheet
ax = fig.add_subplot(grid[1,0])
im = ax.imshow(scanned_projection,
               norm=PowerNorm(gamma=0.5),
               cmap=cmap,
               aspect=aspect,
               extent=extent_zx)

fig.savefig(savedir_path / Path("single_and_scanned_lightsheet_projections.png"), dpi=300)