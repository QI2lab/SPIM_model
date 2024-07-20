"""
Function for running analysis or processing experimental data

Steven Sheppard
2024/04/24
"""
# Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from skimage import transform
from ndtiff import Dataset

import dask

from dask.diagnostics import ProgressBar
import model_tools.propagation as pt
from model_tools.analytic_forms import (gaussian_intensity,
                                        gaussian_mixture,
                                        gaussian_intensity_no_offset)

DEBUG=False


def bin_zstack(raw_data: np.ndarray,
               bin_width: int) -> np.ndarray:
    """
    Bin zstack data by summing over planes.
    Crops the beginning of the array when necessary.

    :param np.ndarray raw_data: 3d array containing z-stack dataset
    :param int bin_width: number of z planes to bin over.
    """
    if bin_width<2:
        return raw_data

    else:
        n_z = raw_data.shape[0]
        n_z_new = n_z//bin_width
        n_crop = n_z - (bin_width*n_z_new)

        if len(raw_data.shape)==2:
            raw_data_new = np.zeros((n_z_new,
                                     raw_data.shape[1]))
        elif len(raw_data.shape)==3:
            raw_data_new = np.zeros((n_z_new,
                                     raw_data.shape[1],
                                     raw_data.shape[2]))

        for ii in range(bin_width):
            raw_data_new += raw_data[int(ii + n_crop)::bin_width]

        return raw_data_new


def mean_convolution(data: np.ndarray,
                     window_width: float = 3) -> np.ndarray:
    """
    Apply a mean filter to 1d data

    :param np.ndarray data: 1d data
    :param float window_width: number of samples to average over.
    """

    pad_width = 3 * window_width
    convolved = np.convolve(np.pad(data,
                                   pad_width=pad_width,
                                   mode="constant",
                                   constant_values=(data[0],
                                                    data[-1])),
                            np.ones(window_width)/window_width,
                            mode="same")

    return convolved[pad_width:-pad_width]


def fit_gaussian_1d(data: np.ndarray,
                    r_idxs: np.ndarray,
                    offset: bool = False,
                    DEBUG_fit: bool = False,
                    savedir: Path = None,
                    showfig: bool = False,
                    label: str = '') -> dict:
    """
    Fit 1-d slice for Gaussian intensity distribution.
    unit: pixel

    dict key:value
    fit_params:popt,
    fit_keys:['w', 'Io', 'mu', 'bg'],
    fit_errors:np.sqrt(np.diag(pcov)),
    initial_params:fit_ip}

    :param nd.array data: 1-d array containing pixel intensity values
    :return dict results: Dict. containing results and fit parameters
    """
    # initial guess
    fit_ip = np.zeros((4))

    # generate initial params from distribution moments
    first_moment = np.sum(r_idxs * data)/np.sum(data)
    second_moment = np.sum(r_idxs**2 * data)/np.sum(data)
    sigma = np.sqrt(second_moment - first_moment**2)
    waist_guess = sigma
    ishift_guess = np.max(data) - np.mean(data)
    offset_guess = np.mean(data)

    if offset:
        fit_ip = [waist_guess,
                              ishift_guess,
                              first_moment,
                              offset_guess]
    else:
        fit_ip = [waist_guess,
                              ishift_guess,
                              first_moment]

    fit_pass = False
    try:
        if offset:
            # Fit temp_profile with guassian intensity distribution
            popt, pcov = curve_fit(gaussian_intensity,
                                   r_idxs,
                                   data,
                                   p0=fit_ip,
                                   maxfev=3000)
        else:
            # Fit temp_profile with guassian intensity distribution
            popt, pcov = curve_fit(gaussian_intensity_no_offset,
                                   r_idxs,
                                   data,
                                   p0=fit_ip,
                                   maxfev=3000)
        fit_pass = True
    except Exception as e:
        print(f'Fit failed, using initial guess: {fit_ip}')
        print(e)
        popt = fit_ip
        fit_pass = False
        DEBUG_fit=True

    if DEBUG_fit:
        # Generate title and strings
        fit_str = ''
        for label, value in zip(['w', 'Io', 'mu', 'bg'], popt):
            fit_str+=f"{label}={np.round(value, 3)}, "

        po_str = ''
        for label, value in zip(['w', 'Io', 'mu', 'bg'], fit_ip):
            fit_str+=f"{label}={np.round(value, 3)}, "

        # Create plot
        fig, ax = plt.subplots(1,1, figsize=(15,10))
        fig.suptitle(f"Debugging 1d Gaussian fit")
        ax.set(ylabel='I(r)',
               xlabel='r (idx)',
               title=f"fit: {fit_str}\ninital_params:{po_str}")

        # Plot fit and data
        ax.plot(r_idxs, data, 'r.', markersize=2.5, label='data')

        if offset:
            ax.plot(r_idxs,
                    gaussian_intensity(r_idxs, *popt),
                    'b',
                    label='fit')
        else:
            ax.plot(r_idxs,
                    gaussian_intensity_no_offset(r_idxs, *popt),
                    'b',
                    label='fit')

        ax.legend()

        if savedir:
            fig.savefig(savedir / Path(f"{label}_debug_fit_result.png"))

        if showfig:
            plt.show()
        else:
            plt.close('all')

    return {'fit_params':popt,
            'fit_keys':['w', 'Io', 'mu', 'bg'],
            'initial_params':fit_ip,
            'fit_pass':fit_pass}


def fit_1d_gaussian_zstack(data: np.ndarray,
                           r_idxs: np.ndarray,
                           DEBUG_fits: bool = False,
                           savedir: Path = None,
                           showfig: bool = False,
                           label: str = ''):
    """
    fit a z stack of 1d data for gaussian dist. with Dask multi-processing return results in pixel units
    """
    # Define pixel coord for fitting
    z_idxs = np.arange(0, data.shape[0])

    with ProgressBar():
        tasks = [dask.delayed(fit_gaussian_1d)(data[z_idx], r_idxs)
                 for z_idx in z_idxs]
        r = dask.compute(*tasks)

    results= {'z_idxs':z_idxs,
                   'r_idxs':r_idxs,
                   'fit_results':np.stack(r, axis=0)}

    if DEBUG_fits:
        # Plots in idx units
        plot_fit_zstack(results,
                        showfig=showfig,
                        savedir=savedir,
                        label=label)

    return results


def plot_fit_zstack(results: np.ndarray,
                    dz: float = 1.0,
                    dx: float = 1.0,
                    showfig: bool = False,
                    savedir: Path = None,
                    label: str = 'zstack_results') -> None:

    """
    Plotting function do summarize z stack fitting results

    :param np.ndarray results: Dictionary returned from fit_gaussian...
    :param float dz: dz spacing.
    :param float dx: transverse sampling
    :param bool showfig: Optionally show figure.
    :param Path savedir: Optionally save figure by passing Path to dir.
    :param str label: If saving, used to make file name.
    """
    n_fits = len(results['fit_results'][0]['fit_params'])
    # Show distribution of fit parameters
    fig = plt.figure(figsize=(n_fits*12, 8))
    grid = fig.add_gridspec(nrows=1,
                            ncols=n_fits,
                            wspace=0.3)
    fig.suptitle('Z-stack fit summary')

    for ii in range(n_fits):
        ax = fig.add_subplot(grid[ii])
        ax.set_title(results['fit_results'][0]['fit_keys'][ii])
        ax.set_xlabel(f"z (dz={dz})")

        z_vals = results['z_idxs']*dz
        fits = np.abs([fit['fit_params'][ii]
                       for fit in results['fit_results']])*dx
        fit_mins = np.min(np.abs(
                [f['fit_params'][ii] for f in results['fit_results']]
                )*dx)

        if ii==0:
            min_radius = np.min(np.abs(
                [f['fit_params'][ii] for f in results['fit_results']]
                )*dx)
            ax.set_ylabel(f"x (dx={dx})")
            ax.plot(z_vals, fits, '.', label='Fits')
            ax.axhline(y=fit_mins,
                       c='r',
                       label=f"min. radius={min_radius:.2f}"
                       )
            ax.legend()
        elif ii==2:
            ax.set_ylabel('y (px)')
            ax.plot(z_vals, fits, '.')
        else:
            ax.set_ylabel('I(r)')
            ax.plot(z_vals, fits, '.')

    if savedir:
        fig.savefig(savedir / Path(f"{label}_z_stack_fits.png"))
    if showfig:
        plt.show()
    else:
        plt.close(fig)


def fit_gaussianmixture_1d(data: np.ndarray,
                           r_idxs: np.ndarray,
                           peak_distance: int = 15,
                           peak_widths: list[int, int] = [1,100],
                           plot_fit_results: bool = False,
                           savedir: Path = None,
                           showfig: bool = False,
                           label: str = 'fit_analysis') -> dict:
    """
    Fit a 3 Gaussian intensity model.

    assumes data has been normalized 0->1

    The point being, to make it possible to characterize multiple lobes
        in a 1d slice of an aberrated laser focus.

    Based on the central lobe intensity, it either returns only the central lobe width or
        a sum over the 3 largest peak's widths. 'max_peak_ratio' controls this rate.

    Initializes peaks using scipy.signal.find_peaks(), but only keeps the three largest to fit.
        As a first filter, peaks must be larger than *peak_height_ratio* times the maximum of data,
        and peaks must be separated by *peak_distance.

    By default 'max_peak_ratio' is set ignore side lobes which are less than 50% of the main lobe peak intensity.

    :param np.ndarray data: 1d array of data to fit.
    :param np.ndarray r_idxs: 1d array for radial positions.
    :param int peak_distance: Number of pnts between peaks in initial guess..
    :param list peak_widths: The range of widths in initial guess.
    :param bool plot_fit_results: Optionally plot fit summary.
    :param Path savedir: Optionally save fit summary.
    :param bool showfig: Optionally display figure.
    :param str label: If saving figure, used to make filename.
    """
    #-------------------------------------------------------------------------#
    # setup for peak finding and crop data about first moment to speed up fitting.
    bg_guess = np.mean(data[:3])

    #-------------------------------------------------------------------------#
    # find peaks in data
    peak_idxs, peaks = find_peaks(data,
                                  height=[bg_guess+0.020, np.inf],
                                  distance=peak_distance,
                                  width=peak_widths,
                                  prominence=0.040,
                                  wlen=150,
                                  rel_height=1 - 1/np.e**2
                                  )

    if len(peak_idxs) == 0:
        # Use data moments for an initial guess
        first_moment = np.sum(r_idxs * (data))/np.sum(data)
        second_moment = np.sum(r_idxs**2 * data)/np.sum(data)
        sigma = np.sqrt(second_moment - first_moment**2)
        peak_guess = np.max(data) - bg_guess
        waist_guess =  sigma * 0.5

        # makeup the expected output of find_peaks
        peaks = {'widths':np.array([waist_guess]),
                 'peak_heights':np.array([peak_guess]),
                 'prominences':np.array([peak_guess])}
        peak_idxs = [np.argmax(data)]

        # Turn on debug flag for review
        # DEBUG_fit = True
        label += '_no_peaks'

    #-------------------------------------------------------------------------#
    # Define initail parameters and bounds for fitting
    # sort the args of peak from greatest to smallest
    peaks_sorted = np.argsort(peaks['peak_heights'])[::-1]

    # Changed the above from "prominences" -> "peak_heights"
    # assign initial params and bounds based on the number of peaks
    n_peaks = len(peak_idxs)
    fit_ip = [1e-12] * 10
    bounds = ([0, 0, 0] * 3 + [0],
              [1e-12, 1e-12, 1e-12] * 3 + [np.inf])

    if n_peaks==1:
        fit_ip[:3] = [peaks['widths'][peaks_sorted][0]/2,
                      peaks['peak_heights'][peaks_sorted][0] - bg_guess,
                      r_idxs[peak_idxs[peaks_sorted[0]]]
                      ]
        fit_ip[-1] = bg_guess
        bounds[1][:3] = [np.inf, np.inf, np.max(r_idxs)+2]

    elif n_peaks==2:
        fit_ip[:6] = [peaks['widths'][peaks_sorted][0]/2,
                      peaks['peak_heights'][peaks_sorted][0] - bg_guess,
                      r_idxs[peak_idxs[peaks_sorted[0]]],
                      peaks['widths'][peaks_sorted][1]/2,
                      peaks['peak_heights'][peaks_sorted][1] - bg_guess,
                      r_idxs[peak_idxs[peaks_sorted[1]]]
                      ]
        fit_ip[-1] = bg_guess
        bounds[1][:6] = 2 * [np.inf, np.inf, np.max(r_idxs)+2]

    elif n_peaks>2:
        fit_ip = [peaks['widths'][peaks_sorted][0]/2,
                  peaks['peak_heights'][peaks_sorted][0] - bg_guess,
                  r_idxs[peak_idxs[peaks_sorted[0]]],
                  peaks['widths'][peaks_sorted][1]/2,
                  peaks['peak_heights'][peaks_sorted][1] - bg_guess,
                  r_idxs[peak_idxs[peaks_sorted[1]]],
                  peaks['widths'][peaks_sorted][2]/2,
                  peaks['peak_heights'][peaks_sorted][2] - bg_guess,
                  r_idxs[peak_idxs[peaks_sorted[2]]],
                  bg_guess]
        bounds[1][:9] = 3 * [np.inf, np.inf, np.max(r_idxs)+2]

    #-------------------------------------------------------------------------#
    # fit data using 3 gaussian mixture model
    fit_pass = False

    try:
        # Fit temp_profile with guassian intensity distribution
        popt, pcov = curve_fit(gaussian_mixture,
                               r_idxs,
                               data,
                               p0=fit_ip,
                               bounds=bounds,
                               maxfev=7500)
        fit_pass = True
        fail_code = ""

    except Exception as e:
        # set popt to initial params
        popt = fit_ip

        # initiate debugging and id as failed fit
        fit_pass = False
        # DEBUG_fit = True
        label += "_failed"
        fail_code = e

    if plot_fit_results or not fit_pass:
        # Compile initial param string to print in title
        po_str = ''
        for val in fit_ip:
            po_str+=f" {np.round(val, 3)}, "
        popt_str = ''
        for val in popt:
            popt_str+=f" {np.round(val, 3)}, "

        # Create figure
        title_str = "".join([f"Debugging 3 Gaussian model fit,",
                             f" Npeaks:{n_peaks}\n {fit_pass}: {fail_code}"])

        fig, ax = plt.subplots(1,1, figsize=(18,10))
        fig.suptitle(title_str)
        ax.set(ylabel='I(r)',
               xlabel='r (idx)',
               title=f"fit results:{popt_str} \n inital_params:{po_str}")

        # Plot fit and data
        r_sampling = np.linspace(r_idxs[0], r_idxs[-1], 2000)
        ax.plot(r_sampling,
                gaussian_mixture(r_sampling, *popt),
                'b.',
                markersize=5,
                label='3gauss fit')
        ax.plot(r_idxs,
                gaussian_mixture(r_idxs, *fit_ip),
                'm.',
                markersize=5,
                label='initial params')
        ax.plot(r_idxs, data, 'r+', markersize=8, label='data')

        for idx in peak_idxs:
            if idx == peak_idxs[-1]:
                ax.axvline(x=r_idxs[idx], c='k', lw=0.5, ls='--', label='peaks')
            else:
                ax.axvline(x=r_idxs[idx], c='k', lw=0.5, ls='--')
        ax.legend()

        if savedir:
            fig.savefig(savedir / Path(f"{label}_debug_fit.png"))

        if showfig:
            plt.show()
        else:
            plt.close('all')

    return {'multi_fit_params':popt,
            'fit_keys':['w', 'Io', 'mu', 'w', 'Io', 'mu', 'w', 'Io', 'mu','bg'],
            'initial_params':fit_ip,
            'fit_pass':fit_pass}


def fit_gaussianmixture_zstack(data: np.ndarray,
                               peak_distance: int=10,
                               plot_zstack_summary: bool=False,
                               plot_fit_results: bool=False,
                               savedir: Path=None,
                               showfig: bool=False,
                               label: str='zstack_results') -> dict:
    """
    Fit multiple planes of 1d data using a Gaussian mixture model.

    :param np.ndarray data: 1d array of data to fit.
    :param int peak_distance: Number of pnts between peaks in initial guess..
    :param list peak_widths: The range of widths in initial guess.
    :param bool plot_zstack_summary: Optionally plot fit summary.
    :param bool plot_fit_results: Optionally plot individual fit summaries.
    :param Path savedir: Optionally save fit summary.
    :param bool showfig: Optionally display figure.
    :param str label: If saving figure, used to make filename.
    """
    # Define pixel coord for fitting
    nz, ny = data.shape
    z_idxs = np.arange(0, nz)
    r_idxs = np.arange(0, ny)

    # Change matplotlib backend during mulitprocessing
    matplotlib.use('agg')

    with ProgressBar():
        tasks = [dask.delayed(fit_gaussianmixture_1d)(
            data=data[z_idx],
            r_idxs=r_idxs,
            peak_distance=peak_distance,
            plot_fit_results=plot_fit_results,
            savedir=savedir,
            showfig=False,
            label=label + f"_{z_idx}"
            )
                    for z_idx in z_idxs]
        r = dask.compute(*tasks)

    results= {'z_idxs':z_idxs,
              'r_idxs':r_idxs,
              'fit_results':np.stack(r, axis=0)}

    # Change backend back to interactive
    matplotlib.use('QtAgg')

    if plot_zstack_summary:
        # Plots in idx units
        plot_fit_zstack(results,
                         dz=1.0,
                         dx=1.0,
                         showfig=showfig,
                         savedir=savedir,
                         label=label)

    return results


def light_sheet_analysis(results: dict,
                         width_factor: float = 0.5,
                         width_scale: float = 1.0,
                         w_window: int = 1,
                         i_window: int = 1,
                         w_filter: list[float , float] = [0.5, 20],
                         plot_waist_fit: bool = False,
                         plot_width_calcs: bool = False,
                         plot_acq_results: bool = True,
                         label: str = "",
                         savedir: Path = None,
                         showfig: bool = False) -> dict:
    """
    Given a a dictionary containing light sheet results,
    generated by either the model or load_acquisition,
    calculate the light sheet length and width parameters.

    - Width and length units are UM.
    - Fitting is done in pixel units.
    - results.keys() = ["light_sheet_slice", "dx", "dz"]

    :param dict results: containing light_sheet_slice and coords.
    :param float width_factor: Relative intenisity cutoff for defining width.
    :param float width_scale: Factor to scale the width after calculation.
    :param int w_window: Smoothing window width for width distribution.
    :param int i_window: Smoothing window width for intensity distribution.
    :param list w_filter: Range of accepted widths (um).
    :param bool plot_waist_fit: Optionally plot the focus Gaussian mixture fit.
    :param bool plot_width_calcs: Optionally plot each z-plane width.
    :param bool plot_acq_results: Optionally plot summary of analysis.
    :param str label: To be used in creating filename if plots are saved.
    :param Path savedir: Path to saving directory.
    :param bool showfig: Optionally show any figures.
    """
    w = 1/np.e**2
    fwhm = 0.5
    fwhm_factor = np.sqrt(2*np.log(2))

    # Setup coordinates
    # Define light_sheet_slice by with normalization
    light_sheet_slice = (results["light_sheet_slice"]
                         / np.max(results["light_sheet_slice"]))

    # Define pixel coordinates used in calculating the width and length
    dx = results["dx"]
    dz = results["dz"]

    nz, ny = light_sheet_slice.shape
    # Data coords.
    z_idxs = np.arange(0, nz)
    r_idxs = np.arange(0, ny)

    # Used in fitting / interpolation
    r_interp = np.linspace(r_idxs[0], r_idxs[-1], int(1e4))
    z_interp = np.linspace(z_idxs[0], z_idxs[-1], int(1e4))

    # Initialize guess
    # Define the center of the light sheet using maximums
    cx = np.argmax(np.max(light_sheet_slice, axis=0))
    cx_i = np.argmin(np.abs(r_interp - cx))
    cz = np.argmax(light_sheet_slice[:, cx])

    # Numerical width calculation
    # estimate light sheet width plane by plane
    wz = np.zeros(nz)

    if plot_width_calcs:
        # make a new directory to hold all the width plots
        width_plots_dir = savedir / Path(f"width_calculations")
        width_plots_dir.mkdir(exist_ok=True)

    for z_idx in z_idxs:
        # interpolate zplane
        zx_interp_fn = interp1d(r_idxs,
                                light_sheet_slice[z_idx],
                                bounds_error=False,
                                fill_value="extrapolate",
                                kind="linear")
        zx_interp = zx_interp_fn(r_interp)
        bg_estimate = np.mean([zx_interp[0], zx_interp[-1]])
        i_max = width_factor*(zx_interp.max() - bg_estimate)

        # left edge is defined as the maximum idx where the intensity
        ## is less than or equal to the max intensity * width factor, up to the peak index
        try:
            wz_left = r_interp[np.min(np.where(zx_interp[:cx_i] >= i_max))]
        except Exception as e:
            wz_left = r_interp[0]

        # right edge is defined as the minimum idx where the intensity
        ## is greater or equal to the max width, from the diffraction focal plane on
        try:
            wz_right = r_interp[
                cx_i + np.max(np.where(zx_interp[cx_i:] >= i_max))
                ]
        except Exception as e:
            wz_right = r_interp[-1]

        # Define width accounting for any scaling factor
        wz[z_idx] = np.abs(wz_right - wz_left) * width_scale

        if plot_width_calcs:
            fig, ax = plt.subplots(1,1, figsize=(10,8))
            t_str = "".join([f"Intensity cutoff = {width_factor:.3f}, ",
                             f"I_max=:{zx_interp.max():.2f}, ",
                             f"I_fwhm={i_max:.2f}, ",
                             f"BG={bg_estimate:.2f}, ",
                             f"w(z)={dx * wz[z_idx]:.3f}"])
            ax.set_title(t_str)

            # Plot z plane intensities
            ax.plot(r_idxs,
                    light_sheet_slice[z_idx],
                    "rx",
                    label="data",
                    linestyle="--")

            # Plot final integration bounds
            ax.axvline(x=wz_right, c="g", label="int. bounds")
            ax.axvline(x=wz_left, c="g")
            ax.axvline(x=cx, label="center idx")
            ax.axhline(y=zx_interp.max(), label="Imax", c="m", ls="--")
            ax.axhline(y=i_max, label="Icut", c="r", ls="--")
            ax.legend()

            # save and close figure
            fig.savefig(width_plots_dir / Path(f"{label}_width_{z_idx}.png"))
            plt.close(fig)

    # Smooth distributions
    # Convert widths from idx -> real units (um)
    wz_um = wz * dx
    z_intensity = light_sheet_slice[:, cx]

    # Define filter based on reasonable width values
    to_keep = np.where((wz_um > w_filter[0])
                        & (wz_um < w_filter[1]))[0]

    #-------------------------------------------------------------------------#
    # Optionally smooth data using mean convolution kernel
    if w_window > 1:
        wz_tokeep = mean_convolution(wz_um[to_keep], w_window)
    else:
        wz_tokeep = wz_um[to_keep]
    if i_window > 1:
        iz_tokeep = mean_convolution(z_intensity[to_keep], i_window)
    else:
        iz_tokeep = z_intensity[to_keep]

    #-------------------------------------------------------------------------#
    # interpolate intensity and widths distributions
    intensity_interp = interp1d(to_keep,
                                iz_tokeep,
                                bounds_error=False,
                                fill_value="extrapolate",
                                kind="linear")
    iz_interp = intensity_interp(z_interp)

    width_interp = interp1d(to_keep,
                            wz_tokeep,
                            bounds_error=False,
                            fill_value="extrapolate",
                            kind="linear")
    wz_interp = width_interp(z_interp)

    #-------------------------------------------------------------------------#
    # Calculate w0
    # Define diffraction focal plane as plane of maximum intensity
    diff_fp_i = np.argmax(iz_interp)
    diff_fp_interp = z_interp[diff_fp_i]
    diff_fp_data = np.argmin(np.abs(z_idxs-diff_fp_interp))

    #-------------------------------------------------------------------------#
    # fit focal plane to estimate the light sheet waist in pixels
    fit_dict = fit_gaussianmixture_1d(data=light_sheet_slice[diff_fp_data],
                                      r_idxs=r_idxs,
                                      peak_distance=5,
                                      peak_widths=[3, 250],
                                      plot_fit_results=False,
                                      savedir=savedir,
                                      showfig=False,
                                      label=f"{label}_waist_fit")

    # Define the waist Gaussian width and FWHM
    wo_gauss = fit_dict["multi_fit_params"][0]*dx
    wo_fit_fwhm = wo_gauss*fwhm_factor
    # wo_fwhm = wz_interp[diff_fp_i]
    wo_fwhm = wo_fit_fwhm

    if plot_waist_fit:
        # Define the intensity at the focal plane and fit results
        zx_intensity = light_sheet_slice[cz]
        wz_fit = gaussian_mixture(r_interp,
                                  *fit_dict["multi_fit_params"])

        # Create figure to plot results
        title_str = "".join([f"{label} light sheet",
                             f" w0={wo_gauss:.2f}um,"
                             f" fwhm={wo_fwhm:.2f}um"]
                            )
        fig, ax = plt.subplots(1,1, figsize=(8,5))
        ax.set_title(title_str)
        ax.set_xlabel("r (um)")
        ax.set_ylabel("I (a.u)")

        # Plot fit results and fit input
        ax.plot((r_idxs - cx)*dx,
                zx_intensity,
                marker="x",
                markersize=10,
                c="r",
                linestyle="--",
                label="data")
        ax.plot((r_interp - r_interp[np.argmin(np.abs(r_interp - cx))])*dx,
                wz_fit,
                c="b",
                label="fit")
        ax.legend()

        # save and close figure
        fig.savefig(savedir / Path(f"waist_fit_results_{label}.png"))
        plt.close(fig)

    #-------------------------------------------------------------------------#
    # Numerically estimate the length
    # estimate light sheet propagation length / confocal parameter / optical sectioning
    max_width = np.sqrt(2) * wo_fwhm

    # left edge is defined as the maximum idx where the the width is is greater than the cutoff
    ## is less than or equal to the max intensity * width factor, up to the peak index
    try:
        lz_left = z_interp[np.max(np.where(wz_interp[:diff_fp_i] >= max_width))]
    except Exception as e:
        lz_left = z_interp[0]
    # right edge is defined as the minimum idx where the intensity
    ## is greater or equal to the max width, from the diffraction focal plane on
    try:
        lz_right = z_interp[diff_fp_i + np.min(np.where(wz_interp[diff_fp_i:] >= max_width))]
    except Exception as e:
        lz_right = z_interp[-1]

    pl = np.abs(lz_right - lz_left)
    pl_um = pl * dz
    left_edge = np.argmin(np.abs(z_idxs-lz_left))
    right_edge = np.argmin(np.abs(z_idxs-lz_right))

    # append to results dict.
    results |= {"width":wo_fwhm,
                "length":pl_um,
                "diffraction_focal_plane":results['stage_positions'][diff_fp_data],
                "left_edge":results['stage_positions'][left_edge],
                "right_edge":results['stage_positions'][ right_edge]
                }

    # Plot Results
    if plot_acq_results:
        fig = plt.figure(figsize=(26, 9))
        fig.suptitle( f"{label} analysis", fontsize=25)

        # Setup grid such that the colorbar is shared for each type of result...
        grid = fig.add_gridspec(nrows=2,
                                ncols=6,
                                width_ratios=[1, 0.05, 1, 0.05, 1.1, 0.075],
                                height_ratios=[1,1],
                                hspace=0.2,
                                wspace=0.1)

        #---------------------------------------------------------------------#
        # on the left plot the intensity distribution
        ax = fig.add_subplot(grid[:,0])
        ax.set(title="Intensity dist.",
                ylabel="I",
                xlabel="z (um)")
        ax.plot(z_idxs[to_keep] * dz,
                z_intensity[to_keep],
                c="r", marker="x", markersize=8, label="data", linestyle="none")
        ax.plot(z_idxs[to_keep] * dz,
                iz_tokeep,
                c="b", marker=".", markersize=4, label="I(z)", linestyle="none")

        # Plot vertical planes
        ax.axvline(x=diff_fp_interp * dz, c="m", ls="--", label="diff. fp")
        ax.axvline(x=lz_left * dz, c="orange", label="left")  # left edge
        ax.axvline(x=lz_right* dz, c="g", label="right") # right edge
        ax.legend()

        #---------------------------------------------------------------------#
        # In the middle plot the widths distribution
        ax = fig.add_subplot(grid[:,2])
        title_str = "".join([f"Width dist., Gauss w0={wo_gauss:.2f}",
                             f" fwhm={wo_fit_fwhm:.2f}, ",
                             f"FWHM={wo_fwhm:.2f}, 2*zR={pl_um:.2f}"])
        ax.set(title=title_str,
                ylabel="w (um)",
                xlabel="z (um)",
                ylim=(0, 15))

        ax.plot(z_idxs[to_keep] * dz,
                wz_um[to_keep],
                c="r", marker="x", markersize=8, linestyle="none",
                label="filtered")
        ax.plot(z_idxs[to_keep] * dz,
                wz_tokeep, c="b", marker="x", markersize=5, linestyle="none",
                label="convolved")
        ax.plot(z_interp * dz,
                wz_interp,
                c="tomato", marker=".", markersize=5, linestyle="none",
                label="w(z)")

        # Plot vertical planes
        ax.axvline(x=diff_fp_interp * dz, c="m", ls="--", label="fp", ymax=0.5)
        ax.axvline(x=lz_left * dz, c="orange", label="left")
        ax.axvline(x=lz_right* dz, c="g", label="right")

        # Plot horizontal planes
        ax.axhline(y=wo_fwhm, c="dodgerblue", ls="--", label="waist FWHM")
        ax.axhline(y=max_width, c="b", label="sqrt(2) * w0")

        ax.legend()

        #---------------------------------------------------------------------#
        # Plot the lightsheet with planes highlighted
        ax = fig.add_subplot(grid[0,4])
        title_str = "Light sheet intensity, analysis coordinates"

        # Plot heatmap
        im = ax.imshow(np.rot90(light_sheet_slice, 1),
                        cmap="hot",
                        extent=[z_idxs[0] * dz,
                                z_idxs[-1] * dz,
                                -r_idxs[-1]//2 * dx,
                                r_idxs[-1]//2 * dx],
                        aspect=1,
                        origin="lower",
                        interpolation=None)

        # Plot OS edges
        ax.axvline(x=diff_fp_interp * dz, c="m", ls="--", label="diff. fp")
        ax.axvline(x=lz_left * dz, c="orange", label="left")  # left edge
        ax.axvline(x=lz_right* dz, c="g", label="right") # right edge


        ax.set(title=title_str,
               ylabel="r (um)",
               xlabel="z (um)")

        ax.legend()

        ax = fig.add_subplot(grid[1,4])
        title_str = "real coordinates"

        # Plot heatmap
        im = ax.imshow(np.rot90(light_sheet_slice, 1),
                        cmap="hot",
                        extent=[results['stage_positions'][0],
                                results['stage_positions'][-1],
                                -r_idxs[-1]//2 * dx,
                                r_idxs[-1]//2 * dx],
                        aspect=1,
                        origin="lower",
                        interpolation=None)

        # Plot OS edges
        ax.axvline(x=results["diffraction_focal_plane"],
                   c="m", ls="--", label="diffraction fp")

        if "model" in label:
            ax.axvline(x=results["midpoint_focal_plane"],
                       c="g", ls="--", label="midpoint fp"
                       )
            ax.axvline(x=results["paraxial_focal_plane"],
                       c="limegreen", ls="--", label="paraxial fp"
                       )
            ax.axvline(x=results["marginal_focal_plane"],
                       c="b", ls="--", label="marginal fp"
                       )

        ax.set(title=title_str,
               ylabel="r (um)",
               xlabel="stage positions (um)")

        ax.legend()

        # show colorbar
        cbax = fig.add_subplot(grid[:,-1])
        plt.colorbar(im, cax=cbax)

        if savedir:
            fig.savefig(savedir / Path(f"summary_{label}.png"))

        if showfig:
            plt.show()
        else:
            plt.close("all")

    return None


def load_light_sheet_acquisition(acq_dir: Path,
                                 acq_amps: float,
                                 acq_apt: float,
                                 plot_light_sheet: bool = False,
                                 label:str = "",
                                 savedir: Path = None) -> dict:
    """
    Given dataset directory, loads light sheet data to extract "slice"
    1. Crop and center
    2. Rotate
    3. Return central slice

    Args:
    :param Path acq_dir: Path containing Dataset.
    :param float acq_amps: Experimental ETL amps.
    :param float acq_apt: Experimentalacquisition aperture opening.
    :param bool plot_light_sheet: Plot light sheet loaded, defaults to False.
    :param str label: Label to use in plotting and results dict.
    :param Path savedir: Directory to save figure. Defaults to None.

    Returns:
        dict: Dictionary with light sheet and other parameters.
    """
    #-------------------------------------------------------------------------#
    # Check the imaging media
    if "water" in label:
        z_scale = 1.333
    else:
        z_scale = 1

    #-------------------------------------------------------------------------#
    # Open dataset
    with Dataset(str(acq_dir)) as dataset:

        # Use z indexes to load raw data
        z_idxs = dataset.axes["z"]

        # Create empty arrays for raw data and stage positions
        raw_data = np.zeros((len(z_idxs),
                             dataset.image_height,
                             dataset.image_width))
        stg_positions = np.zeros((len(z_idxs)))

        # Loading raw data and stage positions
        for z_idx in dataset.axes["z"]:
            # grab raw data
            raw_data[z_idx] = dataset.read_image(channel=0,
                                                 time=0,
                                                 position=0,
                                                 z=z_idx)

            # grab metadata for current image
            img_md = dataset.read_metadata(channel=0,
                                           time=0,
                                           position=0,
                                           z=z_idx)

            # grab current stage position
            stg_positions[z_idx] = img_md["ZPositionUm"]

        # Define grid spacing from dataset metadata
        dz = dataset.summary_metadata["z-step_um"] * z_scale
        dx = img_md["PixelSizeUm"]

    # modify offset stg_positions so that it reflects imaging space dz
    mod_stg_positions = (stg_positions[0]
                         + (stg_positions-stg_positions[0])*dz)

    #-------------------------------------------------------------------------#
    # Rotate raw data
    # linear function to extract slope
    def linear(x, a, b):
        return a * x + b

    # Define coordinates
    nz, ny, nx = raw_data.shape
    x_idxs = np.arange(nx)

    # guess at focal plane using maximum intensity along propagation axis
    cz = np.argmax(np.array([np.max(z_plane) for z_plane in raw_data]))

    # guess light sheet center XY
    cy = np.argmax(np.array([np.mean(raw_data[:, yy, :]) for yy in range(ny)]))
    cx = np.argmax(raw_data[cz, cy, :])

    # select light sheet "slice" using guesses
    # and crop camera FOV down for analysis
    x_pad = int(40 // dx)
    if cx * dx < 20:
        cx = nx//2
    if cx < x_pad+5:
        x_pad = cx - 15

    # enforce odd grid numbers
    if x_pad%2 == 0:
        x_pad += 1

    x_idxs = x_idxs[int(cx)-x_pad:int(cx)+x_pad+1]
    ls_slice = raw_data[:, cy, int(cx)-x_pad:int(cx)+x_pad+1]

    # recalculate x moments on light sheet slice for more accurate cx and rot.
    mx = np.array(
        [np.sum(x_idxs*ls_slice[zz,:]) / np.sum(ls_slice[zz,:])
         for zz in z_idxs]
        )

    # fit using linear function to extract the propagation tilt in zx plane
    [slope, offset], cov = curve_fit(linear, np.arange(nz), mx)
    tilt_angle = -np.arctan(slope)

    if np.abs(tilt_angle)>0.001:
        # rotate 2d lightsheet plane
        light_sheet_slice = transform.rotate(image=ls_slice,
                                             angle=tilt_angle * 180/np.pi,
                                             resize=False,
                                             center=None,
                                             preserve_range=True,
                                             mode="symmetric")
    else:
        light_sheet_slice = ls_slice

    #-------------------------------------------------------------------------#
    # Plot results
    if plot_light_sheet:
        fig = plt.figure(figsize=(10,5))

        grid = fig.add_gridspec(nrows=1, ncols=2,
                                width_ratios=[1, 0.1],
                                hspace=0.2, wspace=0.2)

        ax = fig.add_subplot(grid[0])
        ax.set_title((f"{label}" + "\n" + acq_dir.name + "\n"
                      + f"dx:{dx:.3f}um, dz:{dz:.2f}um" + "\n"
                      + f"ETL:{float(acq_amps):.2f}amps, "
                      + f"apt:{acq_apt:.1f}mm, tilt angle={tilt_angle:.3f}")
                     )
        ax.set_xlabel("stage position (um)")
        ax.set_ylabel("r (um)")

        im = ax.imshow(np.rot90(light_sheet_slice/np.max(light_sheet_slice), 1),
                       cmap="hot",
                       extent= [mod_stg_positions[0],
                                mod_stg_positions[-1],
                                x_idxs[0]*dx,
                                x_idxs[-1]*dx],
                       aspect=1,
                       origin="lower",
                       interpolation=None)

        cbax = fig.add_subplot(grid[1])
        plt.colorbar(im, cax=cbax)

        if savedir:
            f_str = "".join([f"ls_slice_{label}_",
                             f"{acq_apt:.2f}mmapt_{acq_amps:.2f}etlamps.png"])
            fig.savefig(savedir / Path(f_str))

    return {"light_sheet_slice":light_sheet_slice,
            "dx":dx, # Um
            "dz":dz, # Um
            "stage_positions":mod_stg_positions, # Um
            "etl_amps":acq_amps,
            "aperture":acq_apt,
            "label":label,
            "acq_dir_path":acq_dir}


def plot_light_sheet(results,
                     show_focal_planes = False,
                     show_edges = False,
                     show_legend=False,
                     ax = None,
                     ax_title = None,
                     show_cbar = False,
                     z_range=50,
                     x_range=25,
                     return_im=False):
    ls_slice = results["light_sheet_slice"]
    dx = results["dx"]
    dz = results["dz"]
    fp = np.argmin(np.abs(results["stage_positions"]
                           - results["diffraction_focal_plane"])
                   )
    if z_range or x_range:
        # Crop light sheet array down before plotting
        n_z = ls_slice.shape[0]
        n_xy = ls_slice.shape[1]

        if z_range and x_range:
            z_trim = int((z_range/dz)/2)
            x_trim = int((x_range/dx)/2)
            ls_slice = ls_slice[fp-z_trim:fp+z_trim,
                                n_xy//2-x_trim:n_xy//2+x_trim
                                ]

        elif z_range and x_range is None:
            z_trim = int((z_range/dz)/2)
            ls_slice = ls_slice[fp-z_trim:fp+z_trim,
                                :]
        elif x_range and z_range is None:
            x_trim = int((x_range/dx)/2)
            ls_slice = ls_slice[:,
                                n_xy//2-x_trim:n_xy//2+x_trim
                                ]
    else:
        fp = np.argmin(np.abs(results["stage_positions"]
                              - results["diffraction_focal_plane"]))

    n_zx, n_xy, = ls_slice.shape
    # Enforce odd numbers
    if n_xy%2 == 0:
        n_xy += 1
    if n_zx%2 == 0:
        n_zx += 1
    grid_params = pt.field_grid(num_xy=n_xy,
                                num_zx=n_zx,
                                dx=dx,
                                dz=dz,
                                return_field=False
                                )
    try:
        z_start = results["stage_positions"][fp-n_zx//2]
        grid_params[-1][0] += 0
        grid_params[-1][1] += results["stage_positions"][fp+n_zx//2] - z_start
    except:
        zmax = max([len(results["stage_positions"]) - fp, fp])
        z_start = results["stage_positions"][fp-zmax//2]
        grid_params[-1][0] += 0
        grid_params[-1][1] += results["stage_positions"][fp+zmax//2] - z_start
        print(f"z_range is too large, zmax={zmax}")


    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(7,3))
        ax.set(ylabel="r ($\me m$)", xlabel="z (\mu m$")
        show_cbar=True

    if ax_title is not None:
        ax.set_title(ax_title)

    im = ax.imshow(np.rot90(ls_slice/ls_slice.max(),1),
                    cmap='hot',
                    extent=grid_params[-1],
                    aspect='auto',
                    origin='lower',
                    interpolation=None,
                    vmin=0, vmax=1
                    )
    # add results to top corner of plot
    ax.text(0.05, 0.95,
            f"L: {results['length']:.0f}$\mu m$\nW: {results['width']:.1f}$\mu m$",
            color="white",
            fontsize=9,
            ha="left", va="top",
            transform=ax.transAxes)

    if show_edges:
        ax.axvline(x=results["diffraction_focal_plane"] - z_start,
                    label="focal plane",
                    linestyle="-",
                    color="white"
                    )
        try:
            ax.axvline(x=results["left_edge"] - z_start,
                        label="edge",
                        linestyle="--",
                        color="white"
                        )
            ax.axvline(x=results["right_edge"] - z_start,
                        linestyle="--",
                        color="white"
                        )
        except:
            pass
    if show_focal_planes:
        ax.axvline(x=results["midpoint_focal_plane"] - z_start,
                    label="mipoint",
                    linestyle="--",
                    color="g"
                    )
        ax.axvline(x=results["paraxial_focal_plane"] - z_start,
                    label="paraxial",
                    linestyle="--",
                    color="limegreen"
                    )
        ax.axvline(x=results["marginal_focal_plane"] - z_start,
                    label="marginal",
                    linestyle="--",
                    color="red"
                    )
    if show_legend:
        ax.legend(fontsize=7,
                  framealpha=0.1,
                  labelcolor="white"
                  )

    if show_cbar:
        plt.colorbar(im)
    if return_im:
        return im
    else:
        return None
