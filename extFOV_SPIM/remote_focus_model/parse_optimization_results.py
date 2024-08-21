"""
Open ETL remote focus optimization results (from generating heatmap).

Specify models to generate fields and display for paper figure.

08/20/2024
Steven Sheppard
"""
# Imports
import model_tools.raytrace as rt
import model_tools.propagation as pt
from pathlib import Path
import numpy as np
import gc as gc
import matplotlib.pyplot as plt

data_path = Path(r"C:\Users\Steven\Documents\qi2lab\github\SPIM_model\data\20240820_172317_remote_focus_results\0.14NA_in_ri_mismatch_results.npy")

# load list of result_dicts
results_list = np.load(data_path, allow_pickle=True)

# Define which model parameters to generate fields and plot raytrace results
# [cuvette_offset, etl_dpt]
models = [[35, 4], [35, 10], [35,16], [25,4], [25,10], [25,16], [15, 4], [15, 10], [15, 16], [10, 4], [10, 10], [10, 16]]

# Define model parameters
wl = 0.000488
ko = 2 * np.pi / wl
dx = wl/2
dz = 0.005
x_max = 0.300
z_max = 0.200
x_padding = 0.010
n_xy = int(x_max // dx)
n_zx = int(z_max // dz)

# enforce odd grid numbers
if n_xy%2 == 0:
    n_xy += 1
if n_zx%2 == 0:
    n_zx += 1

# locate the models to simulate in results
models_to_keep = []
for params in models:
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

for ii, _r in enumerate(models_to_keep):
    initial_field_params = pt.field_grid(num_xy=n_xy,
                                            num_zx=1,
                                            dx=dx,
                                            dz=dz,
                                            return_field=False
                                            )
    initial_field = rt.raytrace_to_field(results=_r,
                                         grid_params=initial_field_params,
                                         wl=wl,
                                         amp_binning=n_xy,
                                         grid_padding=x_padding,
                                         plot_rays_to_field=True,
                                         plot_raytrace_results=True,
                                         label=(f"raytrace_to_field_{ii}"),
                                         savedir=data_path.parent,
                                         showfig=True)

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
    # Offset z to the focal plane using midpoint focal plane
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

    _r["initial_field"] = initial_field
    _r["field"] = field
    _r["grid_params"] = grid_params

np.save(data_path.parent / Path("models_to_simulate.npy"), models_to_keep, allow_pickle=True)
pt.plot_xz_projection(fields = [_r["field"] for _r in models_to_keep],
                      field_labels= [f"Cuvette Offset:{_p[0]}, focus shift:{_p[1]}, Strehl: {_r['strehl']:.3f}, RMS: {_r['rms']:.3f}" for _p, _r in zip(models, models_to_keep)],
                      grid_params=[_r["grid_params"] for _r in models_to_keep],
                      x_max=20,
                      save_path=data_path.parent / Path("projections.png"), showfig=True)
