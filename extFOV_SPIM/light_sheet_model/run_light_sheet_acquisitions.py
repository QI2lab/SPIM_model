"""
Extract light sheet "slices" from experimental data.

- Open light sheet acquisition datasets
- load data and rotate
- save light sheet slices with metadata for analysis

2024/01/22 Steven Sheppard
"""
# imports
from model_tools.raytrace import get_unique_dir
from model_tools.analysis import (load_light_sheet_acquisition,
                                  light_sheet_analysis)
import numpy as np
from time import time
from pathlib import Path

debug_analysis = False
plot_light_sheet_loading = False
plot_light_sheet_slice = False
plot_light_sheet_analysis = True

#-----------------------------------------------------------------------------#
# Data path
data_path = Path("/mnt/tilingspim/extFOV_results/light_sheet_acq/WIP")

# Where to save light sheets results
save_dir_path = Path("/mnt/server1/extFOV/light_sheet/acquisitions")
savedir = get_unique_dir(save_dir_path,
                         Path("acquisition_results"))

# Compile the path for each acquisition directory
setup_dirs = [_dir for _dir in data_path.iterdir() if(_dir.is_dir()
                                                      and "lightsheet" in _dir.name)
              ]

#-----------------------------------------------------------------------------#
# Load experimental data
t_start = time()
for setup_idx, setup_dir in enumerate(setup_dirs):
    # Create label for experimental setup
    if "_MISMATCH" in setup_dir.name:
        medium = "water"
    elif "_noMISMATCH" in setup_dir.name:
        medium = "air"

    label = f"Exp. data in {medium}"

    # Compile directories containing data
    acq_dirs =  [_dir for _dir in setup_dir.iterdir() if _dir.is_dir() and    "sheet_acq_" in _dir.name]

    # The directory name has information about the settings
    acq_etl_amps = np.array([float(_dir.name.split("na")[1].split("_etl")[1].split("_")[0]) for _dir in acq_dirs])
    acq_apertures = np.array([float(_dir.name.split("na")[1].split("_etl")[0]) * 0.01 * 6.0 for _dir in acq_dirs])

    # Sort acquisitions by aperture and loop through remote focus positions
    apertures = np.unique(acq_apertures)
    aperture_idxs = [np.where(acq_apertures==_apt)[0] for _apt in apertures]

    # Load light sheet arrays from data
    print(f"Loading light sheet slices  ",
          f"{label},  {setup_idx} / {len(setup_dirs)-1} . . .  \n")

    # save one file per aperture
    for ii, load_idxs in enumerate(aperture_idxs):
        print(f"Loading aperture {ii+1}/{len(aperture_idxs)}")

        # Create a subdirectory for loading plots
        f_str = f"{label}_{apertures[ii]:.1f}_light_sheet_plots"
        plot_dir = savedir / Path(f_str)
        plot_dir.mkdir(exist_ok=True)

        acq_dicts=[]
        for jj in load_idxs:
            acq_dict = load_light_sheet_acquisition(acq_dir=acq_dirs[jj],
                                                    acq_amps=acq_etl_amps[jj],
                                                    acq_apt=acq_apertures[jj],
                                                    plot_light_sheet=plot_light_sheet_loading,
                                                    label=label,
                                                    savedir=plot_dir
                                                    )

            light_sheet_analysis(results_dict=acq_dict,
                                 plot_waist_fit=debug_analysis,
                                 plot_width_calcs=debug_analysis,
                                 plot_acq_results=plot_light_sheet_analysis,
                                 label=(f"{label}_{apertures[ii]:.1f}apt_",
                                        f"{acq_etl_amps[jj]}etlamps"),
                                 savedir=plot_dir
                                 )

            acq_dicts.append(acq_dict)

        # Save results
        save_dict = {"results":acq_dicts, "label":label}
        save_path = savedir / Path(f"{label}_{apertures[ii]:.1f}mm_apt.npy")
        np.save(save_path, save_dict, allow_pickle=True)

print(f"Total run time: {(time() - t_start) / 60} minutes")