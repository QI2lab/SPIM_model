'''
Field propagation tools for simulating optical trains.

Main function:
- propagate: Propagate electric field give the Field transform

Helper functions:
- field_grid
- get_3d_field
- model_psf
- diffraction_focal_plane

Plotting functions:
- plot fields
- plot zx projections

2024/04/24
Steven Sheppard
'''
import model_tools.raytrace as rt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
from pathlib import Path
import gc

# multi processing with dask
import dask
from dask.diagnostics import ProgressBar

# Check for cupy
_cupy_available = True
try:
   import cupy as cp
except ImportError:
    cp = np
    _cupy_available = False

#------------------------------------------------------------------------------#
# Propagation

def propagate(Ek: np.ndarray,
              dz: float,
              wl: float,
              dx: float,
              ri: float):
    '''
    Field propagation function propagates electric fields distance dz.
    Reference Goodman exact transfer function method, pg 140.

    kmax == ko * ri

    :param array Ek: Electric Field distribution in k-space(mask).
    :param float dz: Distance to propagate in model units.
    :param float wl: Field wavelength.
    :param float dx: Grid period, in model units.
    :param float ri: Propagation media refractive index.
    :returns array Ef: Electric field at dz from initial field.
    '''

    # Todo: we could "vectorize" this function to work for array of z-planes
    # Then could use this with dask.array

    if isinstance(Ek, cp.ndarray):
        xp = cp
    else:
        xp = np

    if dz == 0:
        return Ek

    else:
        n_xy = Ek.shape[-1]
        ko = 2*np.pi / wl

        # k-space BFP grid
        kx = 2 * np.pi * xp.fft.fftshift(xp.fft.fftfreq(n_xy, d=dx))
        ky = kx
        kxx, kyy = xp.meshgrid(kx, ky)

        # kz component
        kz = xp.sqrt((ko*ri)**2 - kxx**2 - kyy**2 + 0j)

        # # filter based on physical extent of propagating field
        # kz[(kxx**2 + kyy**2)>(ko*ri)**2] = 0

        # Define exactr transfer function kernel
        kernel = xp.exp(1j * kz * dz)

        # filter based on imaginary part of kernel
        kernel[kz.imag > 1e-12] = 0

        # Propagate with exact transfer function kernel
        Ef = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(Ek * kernel)))

        return Ef


#------------------------------------------------------------------------------#
# Create field arrays

def field_grid(num_xy: int = 1001,
               num_zx: int = 100,
               dx: float = 0.001,
               dz: float = 0.001,
               return_field=True):
    '''
    Define a 3-d grid and parameters to be used in field propagation.
    if num_zx==1 return 2-d grid and parameters.

    Note: Number of grid points should be odd

    :param int num_xy: Number of xy grid points
    :param int num_zx: Number of z grid points
    :param float dx: Field sampling period along xy
    :param float dz: Field sampling period along z
    :return array grid: Field like zeros array for given parameters.
    :return dict grid_params: Field linear array, extent and radius.
    '''
    if (num_xy % 2) == 0:
        raise ValueError("number of XY points must be odd")

    # Dk = 2 * np.pi / (dx * num_xy)

    # Create k-space grid for given spatial period.
    # kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(num_xy, d=dx))
    # ky = kx
    # kxx, kyy = np.meshgrid(kx, ky)
    # extent_kxky = [kx[0] - dk/2, kx[-1] + dk/2, ky[0] - dk/2, ky[-1] + dk/2]

    # Create real-space grid.
    x = dx * np.arange(-(num_xy - 1)/2, (num_xy - 1)/2 + 1)
    y = x
    xx, yy = np.meshgrid(x, y)
    extent_xy = [x[0] - dx/2, x[-1] + dx/2, y[0] - dx/2, y[-1] + dx/2]

    z = dz * np.arange(-(num_zx - 1)/2, (num_zx - 1)/2 + 1)
    extent_zx = [z[0] - dz/2, z[-1] + dz/2, x[0] - dx/2, x[-1] + dx/2]

    # Create real radius
    radius_xy = np.sqrt(xx**2 + yy**2)

    if num_zx != 1:
        if return_field:
            grid = np.ones((num_zx, num_xy, num_xy), dtype=np.complex64)
        grid_params = [x, radius_xy, extent_xy, z, extent_zx]

    else:
        if return_field:
            grid = np.ones((num_xy, num_xy), dtype=np.complex64)
        grid_params = [x, radius_xy, extent_xy]


    if return_field:
        return grid, grid_params
    else:
        return grid_params


def grid_phase_mask(Er: np.ndarray,
                    dx: float,
                    kmax: float):
    '''
    Create circular k-space mask, can be used for filtering for propagating through limiting NA.

    # TODO: Make take NA..

    :param array Er: real space electric field
    :param float wl: wavelength
    :param float dx: Grid sampling period in xy
    :param float kmax: Maximum extent in k-space
    :return array new_field:
    '''
    if isinstance(Er, cp.ndarray):
        xp = cp
    else:
        xp = np

    n_xy = xp.shape(Er)[-1]

    # k-space grid
    kx = 2 * xp.pi * xp.fft.fftshift(xp.fft.fftfreq(n_xy, d=dx))
    ky = kx
    kxx, kyy = xp.meshgrid(kx, ky)
    k_radius = xp.sqrt(kxx**2 + kyy**2)

    mask = k_radius >= kmax

    # FFT field
    k_field = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(Er)))

    # Cropped field
    new_field = k_field * mask
    new_field = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(new_field)))

    return new_field


#------------------------------------------------------------------------------#
# helper functions for propagation and electric fields/intensity

def get_3d_field(initial_field: np.ndarray,
                 z: np.ndarray,
                 wl: float,
                 dx: float,
                 ri: float,
                 DEBUG: bool = False,
                 save_path: Path = None,
                 showfig: bool = False):
    '''
    Generate 3d Field using propagation function and parallelize with Dask.

    :param array intial_field: real space complex electric field
    :param array z: propagation distances
    :param float wl: wavelength
    :param float dx: real space grid spacing
    :return array field: real space 3d field
    '''

    # Run propagation
    # Check for GPU
    if _cupy_available:
        # Send psf norm to GPU
        initial_field_gpu = cp.array(initial_field)

        # FFT(E_o)
        field_fft_gpu = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(initial_field_gpu)))
        # helper function to call propagate, then bring back from GPU
        def propagate_gpu(*args): return propagate(*args).get()

        # Dask + GPU
        with ProgressBar():
            tasks = [dask.delayed(propagate_gpu)(field_fft_gpu,
                                                 zpos,
                                                 wl,
                                                 dx,
                                                 ri)
                     for zpos in z]
            results = dask.compute(*tasks)
            field = np.stack(results, axis=0)

    # If no GPU, run on cpu with dask
    else:
        # FFT initial field
        field_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(initial_field)))

        # Propagate field, parallelize with dask
        with ProgressBar():
            tasks = [dask.delayed(propagate)(field_fft,
                                             zpos,
                                             wl,
                                             dx,
                                             ri)
                     for zpos in z]
            results = dask.compute(*tasks)
            field = np.stack(results, axis=0)

    if DEBUG:
        fig = plt.figure(figsize=(30, 30))
        grid = fig.add_gridspec(nrows=5,
                                ncols=4,
                                width_ratios=[1, 0.1] * 2,
                                height_ratios=[1] * 5,
                                wspace=0.1,
                                hspace=0.1)

        fig.suptitle("3d field propagation")

        titles = [['abs(Initial Field)**2', 'Re(FFT(Initial Field))'],
                  ['abs(field[0])**2', 'Re(FFT(field[0]))'],
                  ['abs(field[nz//2])**2', 'Re(FFT(field[n2//2]))'],
                  ['abs(field[-1])**2', 'Re(FFT(field[-1]))']
                  ]

        to_plot = [[np.abs(initial_field)**2,
                    np.real(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(initial_field))))],
                   [np.abs(field[0])**2,
                    np.real(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field[0]))))],
                   [np.abs(field[len(z)//2])**2,
                    np.real(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field[len(z)//2]))))],
                   [np.abs(field[-1])**2,
                    np.real(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field[-1]))))]
                   ]

        for ii in range(len(titles)):
            # Plot the initial Field intensity
            ax = fig.add_subplot(grid[ii, 0])
            ax.set_title(titles[ii][0])
            ax.set_xlabel('x (px)')
            ax.set_ylabel('y (px)')

            im = ax.imshow(to_plot[ii][0],
                           norm=PowerNorm(gamma=1),
                           cmap='hot',
                           origin="lower",
                           aspect='equal',
                           interpolation=None)

            cax = fig.add_subplot(grid[ii, 1])
            cbar = plt.colorbar(im, cax=cax)

            # Plot the first propagation plane intensity
            ax = fig.add_subplot(grid[ii, 2])
            ax.set_title(titles[ii][1])
            ax.set_xlabel('kx (px)')
            ax.set_ylabel('ky (px)')

            im = ax.imshow(np.abs(to_plot[ii][1])**2,
                           norm=PowerNorm(gamma=1),
                           cmap='hot',
                           origin="lower",
                           aspect='equal',
                           interpolation=None)

            cax = fig.add_subplot(grid[ii, 3])
            cbar = plt.colorbar(im, cax=cax)

        # Plot the zx projection of resulting 3d field
        ax = fig.add_subplot(grid[-1, :-1])
        ax.set_title('ZX Intensity projection')
        ax.set_xlabel('x (px)')
        ax.set_ylabel('z (px)')

        im = ax.imshow(xz_projection(field),
                       norm=PowerNorm(gamma=1),
                       cmap='hot',
                       origin="lower",
                       aspect='equal',
                       interpolation=None)
        # Cbar axes
        cax = fig.add_subplot(grid[-1, -1])
        cbar = plt.colorbar(im, cax=cax)

        if save_path:
            plt.savefig(save_path)
        if showfig:
            plt.show()
        else:
            plt.close('all')

    return field


def model_psf(na: float,
              f: float,
              wl: float,
              grid_params: list,
              beam_type: str,
              power: float = 1.0):
    '''
    Generate 3d PSF for perfect lens with NA and focal length f.

    :param float na: lens numerical aperture
    :param float f: lens focal length
    :param float ko: field wave number
    :param list grid_params: [x, radius_xy, extent_xy, z, extent_zx]
    :param float power: Integrated field power
    :return array psf_field: 3d real space E_psf
    '''
    x, radius_xy, extent_xy, z, extent_zx = grid_params
    n_xy = len(x)
    n_zx = len(z)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    ko =  (2*np.pi)/wl

    pupil_radius = f*na

    # Create perfect lens to generate psf
    exc_obj = rt.Perfect_lens(z1=f,
                              f=f,
                              na=na,
                              ri_in=1.0,
                              ri_out=1.0)

    # Raytracing
    n_rays = 1e6

    # Create rays
    rays = rt.create_rays(type=beam_type,
                          source='infinity',
                          n_rays=n_rays,
                          diameter=pupil_radius*2)

    # raytrace lens
    rays = exc_obj.raytrace(rays, final_plane='pp')

    # offset z grid to lens front focal plane
    z = z + exc_obj.ffp

    # raytrace to propagation plane
    rays_to_field_z = rt.rays_to_field_plane(rays, np.max(x))
    rays = rt.intersect_plane(rays, rays_to_field_z, ri_in=1.0, refract=False)

    ### Create initial field normalized to the beam power ###
    initial_field = rt.rays_to_field(mask_radius=radius_xy,
                                     rays=rays,
                                     ko=ko,
                                     binning='doane',
                                     amp_type='power',
                                     phase_type='opl',
                                     power=power)

    # generate psf
    psf_field = get_3d_field(initial_field,
                             z=z-rays_to_field_z,
                             wl=wl,
                             dx=dx,
                             ri=1.0)

    return psf_field


def xz_projection(field: np.ndarray):
    '''
    Calculate zx intensity projection (sum)

    :params array field: 3-d field array
    :return array zx_projection: 2-d Intensity array summed over axes=1
    '''
    if isinstance(field, cp.ndarray):
        xp = cp
    else:
        xp = np

    zx_projection = xp.sum(xp.abs(field)**2, axis=1).T

    return zx_projection


def diffraction_focal_plane(initial_field: np.ndarray,
                            grid_params: list,
                            wl: float,
                            ri: float,
                            focal_plane_guess: float,
                            rays_to_field_z: float,
                            interp_dz: float = 1e-4,
                            DEBUG: bool = False,
                            showfig: bool = False,
                            title: str = '',
                            savedir: Path = None):

    '''
    Units in mm

    :param array initial_field: focusing real space electric field
    :param list grid_params: grid params associated with initial field
    :param float wl: wavelength
    :param flaot ri: refractive index
    :param float focal_plane_guess: paraxial focal plane
    :param float rays_to_field_z: z position where electric field was created from rays
    :param float interp_dz: sampling period in supersampling
    :param boolean DEBUG: Show plot of zx projection and interpolation with focal planes drawn
    :param boolean showfig: optional have figure show
    :param string title: Figure title, passed to suptitle
    :param Path savedir: complete path, including filename for saving figure.
    '''
    #--------------------------------------------------------------------------#
    # simulate electric field to realize intensity distribution
    # Unpack grid params to search
    x, radius_xy, extent_xy, z, extent_zx = grid_params

    n_xy = len(x)
    dx = x[1] - x[0]
    dz = z[1] - z[0]

    # Define z planes
    z_real = z + focal_plane_guess
    z_prop = z_real - rays_to_field_z

    # offset extent around focal plane
    extent_zx[0] += focal_plane_guess
    extent_zx[1] += focal_plane_guess

    # generate 3d field
    field = get_3d_field(initial_field,
                         z=z_prop,
                         wl=wl,
                         dx=dx,
                         ri=ri,
                         DEBUG=False,
                         showfig=False)

    #--------------------------------------------------------------------------#
    # Interpolate O.A. intensity and supersample to find best focal plane
    # extract optical axis intensity
    oa_intensity = np.abs(field[:, int(n_xy//2), int(n_xy//2)])**2

    # interpolate axial intensity distribution
    oa_intensity_interp = interp1d(z_real,
                                   oa_intensity,
                                   kind='quadratic',
                                   bounds_error=False,
                                   fill_value="extrapolate")

    # Define finely sampled grid to pass into interpolation
    search_z = np.arange(z_real[0], z_real[-1], interp_dz)
    search_oa_intensity = oa_intensity_interp(search_z)

    # The global peak is the diffraction focus
    fp_diffraction = search_z[np.argmax(search_oa_intensity)]

    if DEBUG:
        # Define the diffraction focal plane with no interpolation
        fp_diffraction_v2 = z_real[np.argmax(oa_intensity)]

        # Create figure to summarize results
        fig, axs = plt.subplots(1, 2, figsize=(20,8), sharex=True)
        fig.suptitle(title)

        # Plot interpolated intensity
        ax= axs[0]
        ax.set_title('I(0,z)')

        ax.plot(z_real,
                oa_intensity,
                c='b',
                marker='x',
                markersize=5,
                label='prop.')
        ax.plot(search_z,
                search_oa_intensity,
                c='r',
                marker='.',
                markersize=2,
                linestyle='none',
                label='interp.')

        # Plot vertical planes
        ax.axvline(x=fp_diffraction,
                   c='g',
                   label='interp. max int.')
        ax.axvline(x=focal_plane_guess,
                   c='b',
                   linestyle='--',
                   label='fp guess')
        ax.axvline(x=fp_diffraction_v2,
                   c='m',
                   linestyle='--',
                   label='prop. max int.')

        # set axis labels
        ax.set_xlabel('distance from guess fp (mm)')
        ax.set_ylabel('I(0,z)')
        ax.legend()

        # Plot zx projection
        ax= axs[1]
        ax.set_title(f'Search ZX projection')

        # Display projection
        im = ax.imshow(xz_projection(field),
                       cmap='hot',
                       extent=extent_zx,
                       origin='lower',
                       aspect=dx/dz,
                       interpolation=None)

        # Plot vertical planes
        ax.axvline(x=fp_diffraction,
                   c='g',
                   label='interp. max int.')
        ax.axvline(x=focal_plane_guess,
                   c='b',
                   linestyle='--',
                   label='fp guess')
        ax.axvline(x=fp_diffraction_v2,
                   c='m',
                   linestyle='--',
                   label='prop. max int.')

        # set axis labels
        ax.set_xlabel(r'z ($mm$)', fontsize=13)
        ax.set_ylabel(r'x ($mm$)', fontsize=13)
        ax.legend()

        # add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.83, 0.1, 0.03, 0.8])
        plt.colorbar(im, cax=cbar_ax)

        if savedir:
            plt.savefig(savedir / Path(f"{title}_diffraction_focal_plane.png"))
        if showfig:
            plt.show()
        else:
            plt.close(fig)

    return fp_diffraction


def normalize_field(field: np.ndarray,
                    power: float,
                    dx: float,
                    return_scale: bool = False):
    '''
    Return field array normalized to given power

    :param array field: realspace 2d electric field
    :param float power: total field power
    :param float dx: field sampling period in xy
    :param boolean return_scale: optionally return the scale factor with normalized field
    :return array new_field: normalized field
    '''
    # integrate over field to calculate total power
    scale_factor = simps(simps(np.abs(field)**2, dx=dx), dx=dx)

    # scale by total power
    new_field = field * np.sqrt(power/scale_factor)

    if return_scale:
        return [new_field, scale_factor]
    else:
        return new_field


def calc_field_power(field: np.ndarray,
                    dx: float):
    '''
    Integate electric field to calculate field Power

    :param array field: 2d Complex Electric field
    :param float dx: Field grid dx
    :return float power: Integrated field power
    '''
    power = simps(simps(np.abs(field)**2, dx=dx), dx=dx)
    return power


def phase_rms(field_phase: np.ndarray,
              dx: float):
    '''
    calculate the RMS over phase

    Calculate it based on the integral form:
    sqrt( 1/pi int dphi int dr r * (w(r) - np.mean(w(r))**2 )

    '''
    # get mean and make sure it is zero or subtract
    rms = simps(simps((field_phase - np.mean(field_phase))**2, dx=dx), dx=dx)
    return np.sqrt(rms)


#------------------------------------------------------------------------------## Plotting functions

def plot_field(fields: list = [],
               field_labels: list = [],
               grid_params: list = [],
               fig_title: str = 'Field $I(r)$ and complex angle',
               gamma_scale: float = 0.5,
               save_path: Path = None,
               showfig: bool = False):
    '''
    Plot field intensity and phase angle. Similar to plot_rays_to_field

    x, radius_xy, extent_xy, z, extent_zx = field_params

    :param list fields: List of 3-d field arrays.
    :param list field_labels: List of labels for corresponding field subplot titles.
    :param list grid_params: grid params for given field
    :param str fig_title: Figure title.
    :param float gamma_scale: gamma scale passed into PowerNorm
    :param str save_path: Save directory path.
    :param boolean showfig: Optionally choose to display figure.
    '''
    n_plots = len(fields)

    fig = plt.figure(figsize=(15, int(n_plots*5)))
    grid = fig.add_gridspec(nrows=n_plots*2,
                            ncols=6,
                            width_ratios=[1, 0.075, 0.1]*2,
                            height_ratios=[1, 0.1]*n_plots,
                            wspace=0.2,
                            hspace=0.1)
    fig.suptitle(fig_title)

    idx = 0
    for (field,
         field_title,
         field_params) in zip(fields,
                              field_labels,
                              grid_params):
        x, radius_xy, extent_xy, z, extent_zx = field_params

        # Convert from mm to um
        extent_xy = [1e3 * xt for xt in extent_xy]

        ax = fig.add_subplot(grid[idx, 0])
        ax.set_title(field_title+r' $I(r)$')
        ax.set_ylabel(r"y ($\mu m$)")

        if idx == (n_plots-1)*2:
            ax.set_xlabel(r"x ($\mu m$)")

        im = ax.imshow(np.abs(field)**2,
                       norm=PowerNorm(gamma=gamma_scale),
                       cmap='hot',
                       extent=extent_xy,
                       origin="lower",
                       aspect='equal',
                       interpolation=None)

        # Cbar axes
        cax = fig.add_subplot(grid[idx, 1])
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('A.U.',
                           rotation='horizontal',
                           labelpad=18)
        cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

        ax = fig.add_subplot(grid[idx, 3])
        ax.set_title(field_title+r' $\Phi(r)$')

        if idx == (n_plots-1)*2:
            ax.set_xlabel(r"x ($\mu m$)")

        im = ax.imshow(np.angle(field),
                        vmin=-np.pi, vmax=np.pi,
                        cmap='hot',
                        extent=extent_xy,
                        origin="lower",
                        aspect='equal',
                        interpolation=None)
        # Cbar axes
        cax = fig.add_subplot(grid[idx, 4])
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('A.U.', rotation='horizontal', labelpad=18)
        cbar.ax.set_yticks([-np.pi, 0, np.pi])
        cbar.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])

        idx += 2

    if save_path:
        plt.savefig(save_path)
    if showfig:
        plt.show()
    else:
        plt.close('all')


def plot_xz_projection(fields: list = [],
                       field_labels: list = [],
                       grid_params: list = [],
                       fig_title: str = 'Intensity xz max projection',
                       gamma_scale: float = 0.7,
                       norm: bool = False,
                       x_max: float= None,
                       z_max: float = None,
                       save_path: Path =None, showfig=False):
    '''
    Plot the zx projection (sum) for any number of 3-d (propagated) fields.
    Units==um

    x, radius_xy, extent_xy, z, extent_zx = field_params

    :param list fields: List of 3-d field arrays.
    :param list field_labels: List of labels for corresponding field subplot titles.
    :param list grid_params: grid params for given field
    :param str fig_title: Figure title.
    :param float gamma_scale: gamma scale passed into PowerNorm,
    :param float x_max: optionally pass to crop projection
    :param float z_max: optionally pass to crop projection
    :param str save_path: Save directory path.
    :param boolean showfig: Optionally choose to display figure.
    '''
    n_plots = len(fields)

    if norm:
        for ii, data in enumerate(fields):
            fields[ii] = data/np.linalg.norm(data)


    fig = plt.figure(figsize=(12, int(n_plots*6.5)))
    grid = fig.add_gridspec(nrows=n_plots*2,
                            ncols=2,
                            width_ratios=[1, 0.05],
                            height_ratios=[1, 0.05]*n_plots,
                            wspace=0.1,
                            hspace=0.1)
    fig.suptitle(fig_title)

    idx = 0
    for (field,
         field_title,
         field_params) in zip(fields,
                              field_labels,
                              grid_params):
        # generate projection
        projection = xz_projection(field)

        x, radius_xy, extent_xy, z, extent_zx = field_params


        # Convert from mm to um
        x = [1e3 * xx for xx in x]
        z = [1e3 * zz for zz in z]
        extent_zx = [1e3 * xt for xt in extent_zx]

        x = np.asarray(x)
        z = np.asarray(z)
        dx = x[1] - x[0]
        dz = z[1] - z[0]

        # Optional, crop projection
        if x_max:
            idx_x = [(np.abs(x + x_max)).argmin(), (np.abs(x - x_max)).argmin()]
            projection = projection[idx_x[0]:idx_x[1], :]
            extent_zx[2:] =  [x[idx_x[0]] - dx/2, x[idx_x[1]] + dx/2]

        if z_max:
            z_temp = z - np.mean(z)
            idx_z = [(np.abs(z_temp + z_max)).argmin(), (np.abs(z_temp - z_max)).argmin()]
            projection = projection[:, idx_z[0]:idx_z[1]]
            extent_zx[:2] =  [z[idx_z[0]] - dz/2, z[idx_z[1]] + dz/2]

        ax = fig.add_subplot(grid[idx, 0])
        ax.set_title(field_title+r' $I(r)$')
        ax.set_ylabel(r"x ($\mu m$)")

        if idx == (n_plots-1)*2:
            ax.set_xlabel(r"z ($\mu m$)")

        im = ax.imshow(projection,
                    #    norm=PowerNorm(gamma=gamma_scale),
                       cmap='hot',
                       extent=extent_zx,
                       origin="lower",
                    #    aspect='equal',
                       interpolation='none')

        # Cbar axes
        cax = fig.add_subplot(grid[idx, 1])
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('A.U.', rotation='horizontal', labelpad=16)
        cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

        idx += 2

    if save_path:
        plt.savefig(save_path)
    if showfig:
        plt.show()
