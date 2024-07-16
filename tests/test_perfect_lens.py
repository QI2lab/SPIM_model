'''
Test perfect lens.
1. Focus collimated rays, check strehl ratio, wavefront fit and axial spread of rays.
2. Propagate field and compare to theoretical beam parameters

Steven Sheppard
09/27/2022
'''

# model imports
from operator import truediv
import model_tools.raytrace as rt
import model_tools.propagation as pt

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import curve_fit


root_dir = Path('/home/steven/Documents/qi2lab/github/raytracing_sjs/data')
save_dir = Path('perfect_lens_testing')

savedir = rt.get_unique_dir(root_dir, save_dir)

# Unit conversion: mm to um /, um to mm *
mm = 1e-3

validate_pl_opl = False
test_defocus = True
test_focalplanes = True
test_strehl_ratios = True
test_pnt_source = True

wl = 0.0005

def validate_opl(obj_na: float,
                 obj_f: float,
                 ri_in: float,
                 ri_out: float,
                 dz: float,
                 dx: float = 0,
                 savedir: Path = None,
                 showfig: bool = True):
    '''
    This model validates the OPL calculation

    1. Test OPL of Perfect_lens by comparing the dOPL some dz away from perfect focus.
        - Compare raytracing results to geometric solution.

    2. Sample defocused point source and confirm the OPL fit C2 term predicts dz.
        - Extract the defocus distance, dz from Botcherby axial defocus equation (10)

    '''
    def defocused_opld(r, dz, ri):
        '''
        Analytic optical path length difference for ray with radius r, dz from a perfect focus.

        :param float r: Ray radius or height
        :param float dz: Distance from perfect focus
        :param float ri: Propagation media refractive index
        '''
        return  ri * (np.sqrt(r**2 + dz**2) - dz)

    #--------------------------------------------------------------------------#
    # model collimated rays
    #--------------------------------------------------------------------------#

    # Create perfect lens to test
    obj1 = rt.Perfect_lens(z1=obj_f*ri_in,
                           f=obj_f,
                           na=obj_na,
                           ri_in=ri_in,
                           ri_out=ri_out)

    # Create uniform plane-wave rays
    n_rays = 1e6
    rays1 = rt.create_rays(type='flat_top',
                           source='infinity',
                           n_rays=n_rays,
                           diameter=2*obj1.pupil_radius)

    # raytrace perfect lens and intersect defocused plane
    rays1 = obj1.raytrace(rays1, final_plane='ffp')
    rays1 = rt.intersect_plane(rays1,
                               zf=obj1.ffp + np.abs(dz)*obj1.ri_out,
                               ri_in=obj1.ri_out,
                               refract=False)

    # grab raytracing OPLD
    opld1 = rays1[-1, :, 3] - rays1[-1, int(n_rays//2), 3]

    # Calculate geometric dOPL for given dz
    analytic_opld = defocused_opld(rays1[-1, :, 0], np.abs(dz), obj1.ri_out)

    # fit analytic form to extract ri and
    wf_fit1 = rt.ray_opl_polynomial(rays1,
                                    pupil_radius=obj1.pupil_radius,
                                    method='opl')

    fit1 = 0
    for ii in range(len(wf_fit1)):
        fit1 += rays1[-1,:,0]**ii * wf_fit1[ii]

    # Calculate fit error from raytracing OPL
    fit_error1 = analytic_opld - opld1
    fit_sqerror1 = np.square(fit_error1) # squared errors
    fit_mnsqerror1 = np.mean(fit_sqerror1) # mean squared errors
    fit_rmse1 = np.sqrt(fit_mnsqerror1) # Root Mean Squared Error, RMSE
    fit_rsq1 = 1.0 - (np.var(fit_error1) / np.var(opld1))

    #--------------------------------------------------------------------------#
    # Model defocused pnt source
    # Create perfect lens with bfp offset by dz
    obj2 = rt.Perfect_lens(z1=obj_f*ri_in - dz,
                           f=obj_f,
                           na=obj_na,
                           ri_in=ri_in,
                           ri_out=ri_out)

    # Create uniform plane-wave rays
    n_rays = 1e3
    rays2 = rt.create_rays(type='flat_top',
                           source='point',
                           n_rays=n_rays,
                           diameter=2*obj1.pupil_radius,
                           na=obj2.na,
                           offset=dx)

    # raytrace perfect lens and intersect defocused plane
    rays2 = obj2.raytrace(rays2, final_plane='ffp')

    # rho = rays2[-1, :, 0] / obj2.pupil_radius
    # opld2 = rays2[-1, :, 3] - rays2[-1, int(n_rays//2), 3]

    # fit wavefront at FFP
    wf_fit = rt.ray_opl_polynomial(rays2,
                                   pupil_radius=obj2.pupil_radius,
                                   method='opld')
    # strehl = rt.ray_opl_strehl(pupil_rays=rays2,
    #                            wl=wl,
    #                            pupil_radius=obj2.pupil_radius)
    # wf_curvature = 1 / (2*wf_fit[2])

    fit_labels = [f"$C_{ii}$" for ii in range(9)]
    fit_str = ''
    for ft in wf_fit:
        fit_str += f' {ft:.2e} '

    # Use eq. 9 in the botcherby paper.
    # Dz_guess_botcherby = opld2 / np.sqrt(obj2.ri_in**2 - (rho * obj2.na**2))
    dz_guess_botcherby_coeff = [2 * wf_fit[2] * obj2.ri_in / obj2.na**2,
                                8 * wf_fit[4] * obj2.ri_in**3 / obj2.na**4]

    if dx:
        dx_guess_botcherby_coeff =wf_fit[1] / obj2.na

    #--------------------------------------------------------------------------#
    # Plot results
    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(nrows=3,
                            ncols=2,
                            width_ratios=[1,0.5],
                            height_ratios=[1,0.1, 1],
                            hspace=0.3 ,
                            wspace=0.3)
    fig.suptitle(f'Perfect lens $\Delta OPL$ Validation: dz={dz}, dx={dx} ri in/out:{ri_in}/{ri_out}')
    sk_idx = int(n_rays // 20)

    # Plot raytracing for comparing geometric defocus
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title(f'Geometric defocus comparison')
    ax.set_ylabel('radius, mm', labelpad=40, rotation=0)
    ax.set_xlabel('optical axis, mm')
    ax.plot(rays1[:, ::sk_idx, 2], rays1[:, ::sk_idx, 0], 'g')

    # add obj focal planes to plot
    ax.axvline(x=obj1.ffp, c='k', label='obj focal plane')
    ax.axvline(x=obj1.bfp, c='k')

    # show fit plane
    ax.axvline(x=rays1[-1,0,2], c='r', linestyle='--', label='fit plane')
    ax.legend()

    ### Plot results to show matching dOPL ###
    ax = fig.add_subplot(grid[0, 1])
    ax.set_title(r'$\Delta OPL$' +f'\nFit vs Data: RMSE={fit_rmse1:.2e}, R2={fit_rsq1:.2f}')
    ax.set_xlabel('radius, mm')
    ax.set_ylabel(r'$\Delta OPL$', labelpad=35, rotation=0)
    ax.plot(rays1[-1,:, 0],
            analytic_opld,
            marker='x', alpha=0.5, label=r'analytic $\Delta OPL$')
    ax.plot(rays1[-1,:, 0], opld1, label=r'raw $\Delta OPL$')
    ax.legend()

    # Plot raytracing for comparing geometric defocus
    ax = fig.add_subplot(grid[2, 0])
    ax.set_title(f'Defocused point source')
    ax.set_ylabel('radius, mm', labelpad=30, rotation=0)
    ax.set_xlabel('optical axis, mm')
    ax.plot(rays2[:, ::sk_idx, 2], rays2[:, ::sk_idx, 0], 'g')

    # add obj focal planes to plot
    ax.axvline(x=obj2.ffp, c='k', label='obj focal plane')
    ax.axvline(x=obj2.bfp, c='k')

    # show fit plane
    ax.axvline(x=obj2.ffp, c='r', linestyle='--', label='fit plane')
    ax.legend()

    ax = fig.add_subplot(grid[2,1])
    if dx:
        ax.set_title(f'Botcherby comparison\ndx=C1:{dx_guess_botcherby_coeff:.5f}, dz= C2:{dz_guess_botcherby_coeff[0]:.5f}, C4:{dz_guess_botcherby_coeff[1]:.5f}')
    else:
        ax.set_title(f'Botcherby comparison \ndz= C2:{dz_guess_botcherby_coeff[0]:.5f}, C4:{dz_guess_botcherby_coeff[1]:.5f}')

    ax.bar((np.arange(len(wf_fit))),
           np.abs(wf_fit),
           color='m', tick_label=fit_labels)
    ax.set_yscale('log')
    ax.set_ylabel(r'$|C_i|$', labelpad=12, rotation='horizontal')
    ax.set_xlabel('Fit Coeff', rotation='horizontal')

    if savedir:
        fig.savefig(savedir / Path('validate_perfectlens_opl.pdf'))

    if showfig:
        plt.show()


def test_ri_and_strehl(obj_na=0.3, obj_f=20, ri_in=1.0, ri_out=2.0, plot_raytracing=True, showfig=True, savedir=savedir):
    '''
    Show that the perfect lens Strehl ratio does not depend on the propagation media
    '''
    # field param
    wl = 0.5 * mm
    ko = 2*np.pi / wl

    # RI test values
    n_ris = 10
    ri_sweep = np.linspace(1, 2, n_ris)

    # Keep strehl ratio results
    strehl_ratios = np.empty((n_ris, n_ris))
    strehl_ratios_wamp = np.empty((n_ris, n_ris))

    for ii, ri_in in enumerate(ri_sweep):
        for jj, ri_out in enumerate(ri_sweep):
            # number of rays
            rays_density = 1e3

            # Create perfect lens to test
            obj = rt.Perfect_lens(z1=obj_f,
                                  f=obj_f,
                                  na=obj_na,
                                  ri_in=ri_in,
                                  ri_out=ri_out)

            # Create Gaussian plane-wave rays
            n_rays = int(rays_density * (2*obj.pupil_radius))
            rays = rt.create_rays(type='gaussian', source='infinity', n_rays=n_rays, diameter=2*obj.pupil_radius, sampling_limit=3.5)

            # raytrace perfect lens
            rays = obj.raytrace(rays, final_plane='ffp')

            # focal plane from analytic matrix methods and raytracing results
            # focal_plane_analytic = rt.get_mismatched_focal_plane(obj, n1=1.0, d1=0, n_cs=1.0, d_cs=0, n2=ri_out)
            focal_plane_paraxial = rt.ray_focal_plane(rays,
                                                      ri=obj.ri_out,
                                                      method='paraxial')
            focal_plane_midpoint = rt.ray_focal_plane(rays,
                                                      ri=obj.ri_out,
                                                      method='midpoint')


            # Place  perfect lens for fitting conjugate to ffp and place in ri_out
            fit_obj =  rt.Perfect_lens(z1=obj.ffp + obj.f2, f=obj.f, na=obj.na, ri_in=obj.ri_out, ri_out =obj.ri_out)
            fit_rays = fit_obj.raytrace(rays, final_plane='ffp')

            wf_fit = rt.ray_opl_polynomial(fit_rays, fit_obj.pupil_radius)
            strehl = rt.ray_opl_strehl(pupil_rays=fit_rays,
                                       wl=wl,
                                       pupil_radius=fit_obj.pupil_radius)
            strehl_wamp = rt.ray_opl_strehl_with_amp(pupil_rays=fit_rays,
                                                     ko=2*np.pi / wl,
                                                     pupil_radius=fit_obj.pupil_radius,
                                                     binning='doane')

            strehl_ratios[ii, jj] = strehl
            strehl_ratios_wamp[ii, jj] = strehl_wamp

            # Plot ray tracing of perfect lens and the fitting obj.
            sk_idx = int(n_rays // 20)

            if plot_raytracing:
                # extend fit rays beyond ffp
                plot_rays = rt.intersect_optical_axis(rays, ri=obj.ri_out)
                plot_fit = rt.intersect_plane(fit_rays, fit_obj.ffp +10)

                # Configure plot
                fig, ax = plt.subplots(1,1, figsize=(20,8))
                fig.suptitle(f'Perfect lens propagation')
                ax.set_title(f'Strehl results: {strehl}, Wavefront polynomial fit: {wf_fit}')
                ax.set_ylabel('radius (mm)')
                ax.set_xlabel('optical axis (mm)')

                # Plot results
                ax.plot(plot_rays[:, ::sk_idx, 2],
                        plot_rays[:, ::sk_idx, 0], 'k')
                ax.plot(plot_fit[-5:, ::sk_idx, 2],
                        plot_fit[-5:, ::sk_idx, 0], 'b')

                # add obj focal planes to plot
                ax.axvline(x=obj.z1, c='k', label='obj plane')
                ax.axvline(x=obj.ffp, c='k')
                ax.axvline(x=obj.bfp, c='k')

                # add fit obj focal planes
                ax.axvline(x=fit_obj.z1, c='b', label='fit obj plane')
                ax.axvline(x=fit_obj.ffp, c='b')
                ax.axvline(x=fit_obj.bfp, c='b')

                # add calculated focal planes
                # ax.axvline(focal_plane_analytic, c='r', linestyle='--', label='analytic')
                ax.axvline(focal_plane_midpoint,
                           c='g', linestyle='--', label='midpoint')
                ax.axvline(focal_plane_paraxial,
                           c='m', linestyle='--', label='paraxial')

                ax.legend()

                if savedir:
                    plt.savefig(savedir / Path(f'strehl_test_raytracing_{ri_in}_{ri_out}.pdf'))

    # Plot strehl ratio heatmap
    strehl_extent = [ri_sweep[0], ri_sweep[-1], ri_sweep[0], ri_sweep[-1]]
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    fig.suptitle('Testing perfect lens for arb. RI in/out')
    ax.set_title('Strehl Ratios')
    ax.set_ylabel('RI in')
    ax.set_xlabel('RI out')
    ax.imshow(strehl_ratios,
                extent=strehl_extent,
                vmin=0, vmax=1,
                aspect='auto',
                origin='lower')

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    fig.suptitle('Testing perfect lens for arb. RI in/out')
    ax.set_title('Strehl Ratios, with amp')
    ax.set_ylabel('RI in')
    ax.set_xlabel('RI out')
    ax.imshow(strehl_ratios_wamp,
                extent=strehl_extent,
                vmin=0, vmax=1,
                aspect='auto',
                origin='lower')
    if savedir:
        plt.savefig(savedir / Path(f'strehl_testing_map.pdf'))
    if showfig:
        plt.show()


def simulate_4f(f1: float,
                f2: float,
                na1: float,
                na2: float,
                dz: float,
                offset: float,
                showfig: bool = False):
    z1=f1+dz
    z2=z1+f1+f2
    obj1 = rt.Perfect_lens(z1=z1, f=f1, na=na1)
    obj2 = rt.Perfect_lens(z1=z2, f=f2, na=na2)

    rays = rt.create_rays(type='flat_top',
                          source='point',
                          na=na1,
                          offset=offset,
                          n_rays=100)

    rays = obj1.raytrace(rays)
    rays = obj2.raytrace(rays, final_plane='ffp')

    fig, ax = plt.subplots(1,1,figsize=(20,8))
    ax.set_title(f'Geometric defocus comparison, obj 1 f={obj1.f}, obj 2 f={obj2.f}')
    ax.set_ylabel('radius, mm', labelpad=30, rotation=0)
    ax.set_xlabel('optical axis, mm')
    ax.plot(rays[:, :, 2], rays[:, :, 0], 'g')

    # add obj focal planes to plot
    ax.axvline(x=obj1.ffp, c='k', label='obj1')
    ax.axvline(x=obj1.bfp, c='k')
    ax.axvline(x=obj2.ffp, ls='--', c='r', label='obj2 focal plane')
    ax.axvline(x=obj2.bfp, ls='--', c='r')

    ax.legend()

    if showfig:
        plt.show()


# Run different tests.
simulate_4fs = False
if simulate_4fs:
    simulate_4f(f1=30,
                f2=20,
                na1=0.2,
                na2=0.3,
                dz=0,
                offset=0.5)

if validate_pl_opl:
    validate_opl(obj_na=0.14,
                 obj_f=40,
                 ri_in=1.0,
                 ri_out=1.0,
                 dz=0.0,
                 dx=0.,
                 savedir=savedir,
                 showfig=True)

if test_strehl_ratios:
    test_ri_and_strehl(obj_na=0.3,
                       obj_f=20,
                       ri_in=1.2,
                       ri_out=1.5,
                       plot_raytracing=False,
                       savedir=savedir,
                       showfig=True)
