"""
Verify that rays focused at the perfect lens principle plane (PP) do not have any action.

03/11/2023
Steven Sheppard
"""
import model_tools.raytrace as rt
import matplotlib.pyplot as plt

import gc as gc
from pathlib import Path
import numpy as np

# Debugging / plotting flags
showfig = True
plot_raytracing = True
plot_fitting = True
plot_opld = True


#------------------------------------------------------------------------#
#  Setup save paths
#------------------------------------------------------------------------#
root_dir = Path("/home/steven/Documents/qi2lab/github/raytracing_sjs/extFOV_model/data")
savedir_path = rt.get_unique_dir(root_dir, Path(f"etl_and_pl_testing_wip"))


#------------------------------------------------------------------------#
#  Model parameters
#------------------------------------------------------------------------#
n_rays = 1001
ray_diameter = 20
rays_params = {"type":"flat_top",
               "source":"infinity",
               "n_rays":n_rays,
               "diameter":ray_diameter,
               "wl":0.000561}

#------------------------------------------------------------------------#
#  Test ETL ray cropping
#------------------------------------------------------------------------#
f = 200
dpt = 1/f * 1e3
pf_lens_aperture = 50
pf_lens_na = np.sin(np.arctan(pf_lens_aperture/f)) * 1.0

etl = rt.Perfect_lens(z1= f,
                      f=f,
                      na=pf_lens_na,
                      wd=f)

pf_lens = rt.Perfect_lens(z1= etl.ffp,
                          f=f,
                          na=pf_lens_na,
                          wd=f)

#------------------------------------------------------------------------#
#  Test perfect lens action
#------------------------------------------------------------------------#
# Compare rays focusing on the PL PP with and with out ray tracing the PL
ray_no_pl = rt.create_rays(*rays_params.values())
ray_w_pl = rt.create_rays(*rays_params.values())

# ray trace ETL
ray_no_pl = etl.raytrace(ray_no_pl)

# ray trace ETL + PL
for lens in [etl, pf_lens]:
    ray_w_pl = lens.raytrace(ray_w_pl)

# Compare both OPLd at the pl.ffp
ray_no_pl = rt.intersect_plane(ray_no_pl, zf=pf_lens.ffp, refract=False)
ray_w_pl = rt.intersect_plane(ray_w_pl, zf=pf_lens.ffp, refract=False)

# Plot ray tracing
if plot_raytracing:
    rt.plot_rays(rays=ray_no_pl,
                 n_rays_to_plot=21,
                 optical_train=[etl, pf_lens],
                 planes_of_interest=None,
                 title='Raytracing ETL only',
                 show_focal_planes=True,
                 save_path=savedir_path / Path("raytrace_without_pf_lens.png"),
                 showfig=showfig,
                 ax=None,
                 figsize=(30,10))

    rt.plot_rays(rays=ray_w_pl,
                 n_rays_to_plot=51,
                 optical_train=[etl, pf_lens],
                 planes_of_interest=None,
                 title='Raytracing ETL + Perfect lens',
                 show_focal_planes=True,
                 save_path=savedir_path / Path("raytrace_with_pf_lens.png"),
                 showfig=showfig,
                 ax=None,
                 figsize=(30,10))

if plot_fitting:
    # Fit and compare their wavefronts at the PL ffp
    fit_w_pl = rt.ray_opl_polynomial(rays=ray_w_pl,
                                           method='opl')

    fit_no_pl = rt.ray_opl_polynomial(rays=ray_no_pl,
                                            method='opl')

    # Plot fit coefficients
    rt.plot_fit_summary(fit=fit_w_pl,
                        axes_title='ETL and no perfect lens',
                        wl=None,
                        save_path=savedir_path / Path("fit_summary_noperfectlens.png"),
                        showfig=showfig)

    rt.plot_fit_summary(fit=fit_w_pl,
                        axes_title='ETL with perfect lens',
                        wl=None,
                        save_path=savedir_path / Path("fit_summary_withperfectlens.png"),
                        showfig=showfig)

if plot_opld:
    # Plot OPLd
    rt.plot_opld(rays=[ray_no_pl, ray_w_pl],
                rays_labels=["No perfect pens", "With perfect lens"],
                axes_title='Optical length difference',
                num_to_plot=1001,
                units="um",
                save_path=savedir_path / Path("pathlength_comparison.png"),
                showfig=showfig)

plt.show()
