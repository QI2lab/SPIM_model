"""
ETL remote focus optimization.

1. Simulate ETL operating range +/- 10 diopters in air
2. Simulate all possible remote focus configurations, looking for an optimum.

Optionally, display interactive plot with ray tracing.

04/19/2024
Steven Sheppard
"""
#
import model_tools.raytrace as rt
from pathlib import Path
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar

import numpy as np
import gc as gc
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import matplotlib as mpl
# mpl.use("TkAgg")
if __name__ == '__main__':

    # Plotting flags
    showfig = False
    plot_interactive_results = False
    run_mitu = True
    run_nikon = False
    fig_version = True
    run_in_parallel = True
    n_rays = 3e6
    num_samples = 31
    save_name = "models_for_figure"
    save_data = True

    # Setup directories
    root_dir = Path("/mnt/server1/extFOV/remote_focus_data")
    savedir_path = rt.get_unique_dir(root_dir, "remote_focus_results")
    plots_dir = savedir_path / Path("plots")
    plots_dir.mkdir(exist_ok=True)


    """
    Calculate focal shift with respect to:
      "cuv" (last cuvette surface, exp.)
      "native" (0 dpt focal plane)
      "dpt" (show etl dpts instead of focal shift)
    """
    focus_shift_plane = "cuv"

    # Setup figure constraints
    grid_widths = [0.85, 0.1, 0.5, 0.85, 0.1]
    trans_max = 50
    axial_max = 300
    xlim = 20
    axial_yticks = [10, 25, 35]
    axial_cbar_ticks = [-200, -50, 50, 200]
    trans_cbar_ticks = [0, 10, 20]
    axes_label_ftsize = 12
    axes_tick_ftsize = 10
    figsize = (3.75, 1.8)

    #------------------------------------------------------------------------------#
    # Define model parameters
    #------------------------------------------------------------------------------#
    # Create set of rays to use for simulations
    rays_params = {"type":"flat_top",
                "source":"infinity",
                "n_rays":n_rays,
                "diameter":18,
                "wl":0.000488}

    initial_rays = rt.create_rays(*rays_params.values())

    # Define the clear aperture for Thorlabs AC508-XXX-ML
    clear_aperture_508 = 50.8

    # Define parameters for the ETL (plano-CURVED)
    etl_ri = 1.3
    etl_t0 = 10
    etl_d = 16
    etl_dpt_min = -10
    etl_dpt_max = 10

    """
    How lens Doublet lens parameters are defined.
    _params = [z1,
               r1, t1, ri1,
               r2, t2, ri2,
               r3,
               aperture_radius,
               ri_in, ri_out,
               type]
    """
    # Define relay lens 1 parameters
    relay1_lens_params = [0,
                        109.7, 12.0, 1.5180,
                        -80.7, 2.0, 1.6757,
                        -238.5,
                        (clear_aperture_508/2),
                        1.0, 1.0,
                        "relay1"
                        ]
    # Define relay lens 2 parameters
    relay2_lens_params = [0,
                        247.7, 3.0, 1.6757,
                        72.1, 12.0, 1.5180,
                        -83.2,
                        (clear_aperture_508/2),
                        1.0, 1.0,
                        "relay2"
                        ]

    # Define cuvette / ri-mismatch parameters
    immersion_ri = 1.33
    cuvette_path_length = 20.0
    cuvette_wall_thickness = 1.25
    cuvette_height = 45.0
    cuvette_ri = 1.4585

    #------------------------------------------------------------------------------#
    # Create remote focus / illumination pathway
    # To create Relay doublets:
    # 1. Define temporary lens at z=0
    # 2. Use the temporary lens to place conjugate with the previous lens.
    #------------------------------------------------------------------------------#

    # flat ETL
    flat_etl = rt.create_etl(z1=0, dpt=0, d=etl_d, ri=etl_ri, t0=etl_t0)

    # reference relay 1
    relay1_temp = rt.Doublet_lens(*relay1_lens_params)

    # Update the z1 position of the lens
    relay1_lens_params[0] = flat_etl.z2 + relay1_temp.f1
    relay1 = rt.Doublet_lens(*relay1_lens_params)

    # reference relay2
    relay2_temp = rt.Doublet_lens(*relay2_lens_params)

    # Update the z1 position of the lens
    relay2_lens_params[0] = relay1.ffp + relay2_temp.f1
    relay2 = rt.Doublet_lens(*relay2_lens_params)

    # Create the excitation objectives to simulate
    # Mitutoyo 0.14 Long working distance
    mitu_f = 40
    mitu_na = 0.14
    mitu_mag = 5
    mitu_wd = 34
    mitutoyo = rt.Perfect_lens(z1=relay2.ffp + mitu_f,
                            f=mitu_f,
                            na=mitu_na,
                            wd=mitu_wd,
                            fov=25,
                            mag=mitu_mag,
                            ri_in=1.0, ri_out=1.0,
                            type="mitutoyo obj."
                            )
    # Olympus 0.3 PlanFluor
    nikon_f = 20
    nikon_na = 0.3
    nikon_mag = 10
    nikon_wd = 16
    nikon = rt.Perfect_lens(z1=relay2.ffp + nikon_f,
                            f=nikon_f,
                            na=nikon_na,
                            wd=nikon_wd,
                            fov=25,
                            mag=nikon_mag,
                            ri_in=1.0, ri_out=1.0,
                            type="nikon obj."
                            )

    #------------------------------------------------------------------------------#
    # Method for running simulations
    #------------------------------------------------------------------------------#

    def model_illumination_pathway(exc_obj,
                                dpt: float = 0,
                                cuvette_offset : float = None):
        """
        Calculate the focal extent of the illumination pathway
        for the given exc. objective. Optionally, include
        the cuvette_offset to include an index mismatch using Thick_lens.
        Calcalutes the illumination pathway focus shift,
        with or without the cuvette, with respect to the flat ETL pathway.

        Args:
            exc_obj (_type_): Perfect_lens to use in illumination pathway
            dpt (float, optional): ETL diopter to use, defaults to 0.
            cuvette_offset (float, optional): to include the cuvette mismatch.
        Returns:
            _r (dict): Dictionary of results
        """

        # Create the current ETL
        _etl = rt.create_etl(z1=flat_etl.z1,
                            dpt=dpt,
                            d=etl_d,
                            t0=etl_t0)

        if cuvette_offset:
            # add the cuvette to optical pathway
            _cuvette = rt.Thick_lens(z1=((exc_obj.ffp-exc_obj.wd)
                                        + cuvette_offset),
                                    r1=np.inf,
                                    t=cuvette_wall_thickness,
                                    ri=cuvette_ri,
                                    r2=np.inf,
                                    aperture_radius=cuvette_height/2,
                                    ri_in=1.00,
                                    ri_out=immersion_ri)
        else:
            # Make the cuvette refractive index match air,
            # Include in OT to avoid distinguishing between air/mismatched models
            _cuvette = rt.Thick_lens(z1=exc_obj.ffp-exc_obj.wd,
                                    r1=np.inf,
                                    t=cuvette_wall_thickness,
                                    ri=1.00,
                                    r2=np.inf,
                                    aperture_radius=cuvette_height/2,
                                    ri_in=1.00,
                                    ri_out=1.00)
        # Create optical trains
        illumation_pathway = [_etl, relay1, relay2, exc_obj, _cuvette]
        native_pathway = [flat_etl, relay1, relay2, exc_obj, _cuvette]

        # Get the flat etl results for the given exc obj
        native_r = rt.raytrace_ot(optical_train=native_pathway,
                                rays=initial_rays.copy(),
                                fit_plane="midpoint",
                                return_rays="all",
                                plot_raytrace_results=False
                                )

        # ray trace and calculate focal extent for the optical train of interest
        _r = rt.raytrace_ot(optical_train=illumation_pathway,
                            rays=initial_rays.copy(),
                            fit_plane="paraxial",
                            return_rays="all",
                            plot_raytrace_results=False,
                            save_path=plots_dir / Path(f"{dpt:.2f}dpt_{cuvette_offset:.2f}offset.png")
                            )

        # Calculate the ray tracing focal extent
        rays = rt.intersect_plane(_r["rays"],
                                  _r["midpoint_focal_plane"],
                                  ri_in=immersion_ri,
                                  refract=False
                                  )


        # Calculate the focal extent and shift
        _r["transverse_extent"] = np.abs((np.nanmax(rays[-1, :, 0])
                                        - np.nanmin(rays[-1, :, 0])))
        _r["axial_extent"] = _r["paraxial_focal_plane"] -_r["marginal_focal_plane"]
        _r["cuvette_offset"] = cuvette_offset
        _r["etl_dpt"] = dpt
        _r["label"] = f"Cuvette Offset = {cuvette_offset}, ETL power: {dpt} diopters"
        _r["fail_reason"] = None

        # calculate focus shift, set at top of script
        if focus_shift_plane=="cuv":
            _r["focus_shift"] = (_r["midpoint_focal_plane"] - _cuvette.z2)
        elif focus_shift_plane=="dpt":
            _r["focus_shift"] = dpt
        else:
            _r["focus_shift"] = (_r["midpoint_focal_plane"]
                                - native_r["midpoint_focal_plane"])

        # Check to make sure the focus is with in the cuvette volume
        if cuvette_offset:
            oa_crossings = rt.get_ray_oa_intersects(rays)
            if oa_crossings[-1]==11 or oa_crossings[-1]==12:
                _r["axial_extent"] = np.nan
                _r["transverse_extent"] = np.nan
                _r["focus_shift"] = np.nan
                _r["fail_reason"] = "focus before or in cuvette"

        return _r


    def generate_heatmap_data(exc_obj,
                              num_samples,
                              model_label: str =None,
                              savedata:bool = False):
        """
        Simulate illumation pathway using the given excitation objective.
        Run ETL diopters \pm10 and over the physically possible cuvette positions.

        Args:
            exc_obj (Perfect_lens): excitation objective to use.
            num_samples (int): Number of simulated dpts and cuv dz.
            fig_shape (tuple, optional): Figure shape, defaults to (3.5, 1.75).
            axes_label_fontsize (int, optional): Defaults to 8.
            axes_tick_fontsize (int, optional): Defaults to 8.
            grid_width_ratios (list, optional): Defaults to [1, 0.075, 0.3, 1, 0.075].
            axial_max (int, optional): Defaults to 150.
            transverse_max (int, optional): Defaults to 20.
            transverse_cbar_ticks (list, optional): Defaults to [0, 10, 20].
            axial_cbar_ticks (list, optional): Defaults to [0, 10, 20].
            xlim (int, optional): Focus shift limit, uses \pm, defaults to 10.
            cmap_axial (str, optional): Defaults to "coolwarm".
            cmap_trans (str, optional): Defaults to "Reds".
            showfig (bool, optional): Return figure and show, defaults to True.
            final_version (bool, optional): Save as .svg for vectorized rendering.

        Returns:
            fig (optional): fig
        """
        # Define the ETL diopter and cuvette offset range to run all simulations
        etl_dpt_samples = np.linspace(etl_dpt_min, etl_dpt_max, num_samples)
        cuvette_offsets = np.linspace(0.1, exc_obj.f, num_samples)
        cuvette_offsets = np.array([35])

        #--------------------------------------------------------------------------#
        # simulate exc_obj over cuvette positions and ETL diopters
        if run_in_parallel:
            n_processes = 20
            # Function to be parallelized
            @delayed
            def simulate_model(dpt, offset):
                return model_illumination_pathway(exc_obj=exc_obj, dpt=dpt, cuvette_offset=offset)

            # Create list to hold all the delayed tasks
            tasks = []
            for offset in cuvette_offsets:
                tasks += [simulate_model(dpt, offset) for dpt in etl_dpt_samples]


            print(f"Parallizing computation for {len(tasks)} optical trains")
            # Use Dask's ProgressBar to monitor the parallel execution
            with ProgressBar():
                _r = compute(*tasks, scheduler="processes", num_workers=n_processes)
        else:
            _r = []
            for ii, offset in enumerate(cuvette_offsets):
                for jj, dpt in enumerate(etl_dpt_samples):
                    print(f"Running model: offset{ii+1}/{num_samples} and dpt {jj}/{num_samples}", end="\r")
                    _r.append(model_illumination_pathway(exc_obj=exc_obj,
                                                         dpt=dpt,
                                                         cuvette_offset=offset
                                                         ))

        # save results to propagat and plot specific configurations
        if not model_label:
            model_label = f"{exc_obj.na:.2f}NA_in_ri_mismatch_results"

        if savedata:
            print(f"Saving model label:{model_label}")
            np.save(savedir_path / Path(f"{model_label}.npy"), _r, allow_pickle=True)

        return _r


    def create_heatmap_figure(results,
                              exc_obj,
                             fig_shape: tuple = (3.5, 1.75),
                             axes_label_fontsize: int = 8,
                             axes_tick_fontsize: int = 8,
                             axial_max: float = 150,
                             xlim: float = 10,
                             cmap_axial: str = "coolwarm",
                             showfig: bool = True):

        # Compile 1d arrays of results to plot
        cuv_offsets = np.array([_["cuvette_offset"] for _ in results])
        focus_shift = np.array([_["focus_shift"] for _ in results])
        axial_extents = np.array([_["axial_extent"]*1e3 for _ in results])
        # c4s = np.array([_["opl_fit"][4] for _ in results])
        # c2s = np.array([_["opl_fit"][2] for _ in results])


        #--------------------------------------------------------------------------#
        # Create figure to plot results
        fig = plt.figure(figsize=fig_shape)
        grid = fig.add_gridspec(nrows=1,
                                ncols=2,
                                width_ratios=[1.0,0.1],
                                wspace=0.15)

        # Create the axial and transvers extent axis
        ax = fig.add_subplot(grid[0,0])
        ax_cbar = fig.add_subplot(grid[0,1])
        #--------------------------------------------------------------------------#

        # Create scatter plots
        a_scat = ax.scatter(focus_shift,
                            cuv_offsets,
                            c=axial_extents,
                            vmin=-axial_max,
                            vmax=axial_max,
                            cmap=cmap_axial,
                            s=1,
                            marker='o'
                            )
        # a_scat = ax.scatter(focus_shift,
        #                     cuv_offsets,
        #                     c=c4s,
        #                     vmin=-0.0005,
        #                     vmax=0.0005,
        #                     # vmin=-np.max(np.abs(c4s)),
        #                     # vmax=np.max(np.abs(c4s)),
        #                     cmap=cmap_axial,
        #                     s=1,
        #                     marker='o'
        #                     )
        # Create colorbars
        axial_cbar = plt.colorbar(a_scat, cax=ax_cbar)

        # setup the axes labels
        if focus_shift_plane == "dpt":
            x_label = "ETL dpt ($m^{-1}$)"
        else:
            x_label = "df (mm)"
        ax.set_ylabel("dc (mm)", fontsize=axes_label_fontsize)
        ax.set_xlabel(x_label, fontsize=axes_label_fontsize)
        axial_cbar.set_label("LSA ($\mu m$)",
                            loc="center",
                            fontsize=axes_label_fontsize,
                            labelpad=4
                            )
        ax_cbar.tick_params(axis="y", labelsize=axes_tick_fontsize)
        ax.tick_params(axis="both", labelsize=axes_tick_fontsize)
        ax.set_facecolor('k')

        # setup the axis limits
        if focus_shift_plane=="dpt":
            xlims = (-10, 10)
        elif focus_shift_plane=="cuv":
            xlims = (0, xlim)
        else:
            xlims = (-xlim, xlim)
        ax.set_xlim(xlims)
        ax.set_xticks(np.array([0,10,20]))
        plt.subplots_adjust(top=0.92, bottom=0.28, right=0.80, left=0.15)
        fig.savefig(savedir_path / Path(f"{exc_obj.type}_heatmap.pdf"))


        if showfig:
            fig.show()
            return fig
        else:
            return None


    #------------------------------------------------------------------------------#
    # Run simulations
    #------------------------------------------------------------------------------#
    if run_mitu:
        mitu_heatmap_data = generate_heatmap_data(mitutoyo, num_samples, save_name, save_data)

        create_heatmap_figure(mitu_heatmap_data,
                              mitutoyo,
                              fig_shape=figsize,
                              axes_label_fontsize=axes_label_ftsize,
                              axes_tick_fontsize=axes_tick_ftsize,
                              axial_max=axial_max,
                              xlim=xlim,
                              cmap_axial="RdBu",
                              showfig=True)



    # if run_nikon:
    #     fig_nikon = create_heatmap_figure(exc_obj=nikon,
    #                                     num_samples=num_samples,
    #                                     fig_shape=figsize,
    #                                     axes_label_fontsize=axes_label_ftsize,
    #                                     axes_tick_fontsize=axes_tick_ftsize,
    #                                     axial_max=axial_max,
    #                                     xlim=xlim,
    #                                     cmap_axial="RdBu",
    #                                     showfig=True,
    #                                     model_label=save_name)


    if plot_interactive_results:
        def update_mismatched_fig(val):
            """Update the figure model"""
            etl_dpt = dpt_mis_slider.val
            cuvette_offset = offset_mis_slider.val
            fov = ckbox.get_status()[0]
            use_nikon = ckbox_obj.get_status()[0]

            # Update exc obj.
            if use_nikon:
                exc_obj = nikon
            else:
                exc_obj = mitutoyo

            # Run model
            _r = model_illumination_pathway(exc_obj=exc_obj,
                                            dpt=etl_dpt,
                                            cuvette_offset=cuvette_offset
                                            )
            rays = _r["rays"]
            rays = rt.intersect_optical_axis(rays, ri=immersion_ri)

            # Update the figure axes
            ax_mis.clear()

            # Define the planes of interest to plot
            planes_to_show = {"obj ffp":exc_obj.ffp,
                            "obj bfp":exc_obj.bfp,
                            "par. fp":_r["paraxial_focal_plane"],
                            "mar. fp":_r["marginal_focal_plane"],
                            "mid. fp":_r["midpoint_focal_plane"],
                            }

            # Plot the updated rays
            rt.plot_rays(rays=rays,
                        n_rays_to_plot=15,
                        optical_train=_r["optical_train"],
                        planes_of_interest=planes_to_show,
                        show_focal_planes=False,
                        ax=ax_mis)

            # Update figure title
            if _r["axial_extent"] is not None:
                title_str = "".join([f"Focus shift = {_r['focus_shift']:.3f}mm, ",
                                    f"axial/trans. extent = ",
                                    f"{_r['axial_extent']:.3f}/",
                                    f"{_r['transverse_extent']:.3f}mm ",
                                    f"Fail code: {_r['fail_reason']}"])
            ax_mis.set_title(title_str)

            # adjust the axes limits
            ax_mis.set_ylim(-30, 30)
            if fov:
                ax_mis.set_xlim(exc_obj.ffp-40,exc_obj.ffp+40 )

            # Update the plot
            plt.draw()

        #-------------------------------------------------------------------------#
        # Create mismatched figure and the initial plot
        fig_mis = plt.figure(figsize=(8,4))
        grid_mis = fig_mis.add_gridspec(nrows=3,
                                        ncols=3,
                                        height_ratios=[1,0.1, 0.1],
                                        width_ratios=[0.2, 0.2, 1],
                                        hspace=0.3
                                        )
        ax_mis = fig_mis.add_subplot(grid_mis[0,:])

        # add dpt slider
        ax_dpt_mis = fig_mis.add_subplot(grid_mis[1,2])
        dpt_mis_slider = Slider(ax=ax_dpt_mis,
                                label='ETL dpt',
                                valmin=-10,
                                valmax=10,
                                valinit=0,
                                valstep=0.2
                                )

        # add cuv offset slider
        ax_mis_offset = fig_mis.add_subplot(grid_mis[2,2])
        offset_mis_slider = Slider(ax=ax_mis_offset,
                                label='Cuvette dz',
                                valmin=0.01,
                                valmax=mitutoyo.wd-cuvette_wall_thickness,
                                valinit=0,
                                valstep=0.2
                                )

        # add check box for FOV and exc objective
        ax_ckbox = fig_mis.add_subplot(grid_mis[1:,0])
        ckbox = CheckButtons(ax_ckbox,
                            labels=['view FOV'],
                            actives=[False])
        ax_ckbox_obj = fig_mis.add_subplot((grid_mis[1:,1]))
        ckbox_obj = CheckButtons(ax_ckbox_obj,
                                labels=["nikon 0.3"],
                                actives=[False])

        # Update actions
        dpt_mis_slider.on_changed(update_mismatched_fig)
        offset_mis_slider.on_changed(update_mismatched_fig)
        ckbox.on_clicked(update_mismatched_fig)
        ckbox_obj.on_clicked(update_mismatched_fig)

        update_mismatched_fig(None)
        fig_mis.show()

    if showfig:
        plt.show()

    else:
        plt.close("all")
