# -*- coding: utf-8 -*-
"""
Steven Sheppard
04/24/2024
"""
import model_tools.raytrace as rt
import numpy as np
from numpy.typing import NDArray
from scipy.special import j1 as bessel1
from scipy.special import j0 as bessel0
from scipy.integrate import quad

# multi processing with dask
import dask
from dask.diagnostics import ProgressBar

#------------------------------------------------------------------------------#
# Gaussian optics functions
#------------------------------------------------------------------------------#

def fwhm(w: float):
    '''
    Calculate the FWHM from gaussian standard deviation.

    :param float w: Gaussian beam radiu.
    :return float fwhm: Beam radius converted to FWHM.
    '''
    fwhm = np.sqrt(2*np.log(2))*w

    return fwhm


def gaussian_field(r: NDArray,
                   z: float,
                   wl: float,
                   wo: float,
                   ri: float,
                   Io: float) -> NDArray:
    '''
    Gaussian Beam defined in Goodman, pg 109
    wz is defined as the FWHM, 1/e.

    :param arrray r: radial distance of beam, from meshgrid (transverse)
    :param float z: Field distance from waist
    :param float wl: Laser wavelength
    :param float wo: Gaussian beam waist
    :param float ri: Refractive index of propagation medium
    :param float Io:
    :returns array: Gaussian Field * mask
    '''
    if z == 0:
        field = np.sqrt(Io)*np.exp(-(r / wo)**2)

    else:
        k = 2 * np.pi*ri / wl
        zr = np.pi*ri*wo**2 / wl
        guoy_ph = np.arctan(z/zr)
        rz = z * (1 + (zr/z)**2)
        wz = wo * np.sqrt(1 + (z/zr)**2)

        field = np.sqrt(Io)*(wo / wz)*(np.exp(-(r/wz)**2)
                                       * np.exp(1j*(k*z + k*r**2/(2*rz) - guoy_ph)))

    return field


def gaussian_intensity(r: NDArray,
                       w: float,
                       Io: float,
                       mu: float,
                       offset: float = 0):
    '''
    Calculates Intensity of gaussian distribution with intensity offset

    :param float r: Radius
    :param float mu: Mean
    :param float w: gaussian width, intensity values fall to 1/e2
    :param float Io: Peak intensity at this plane
    :param float offset: Intensity offset, background

    :returns float Ir: Intensity at radius r
    '''
    I_r = Io*np.exp(-2*(r - mu)**2/w**2) + offset

    return I_r


def gaussian_intensity_no_offset(r: NDArray,
                                 w: float,
                                 Io: float,
                                 mu: float):
    '''
    Calculates Intensity of gaussian distribution with intensity offset

    :param float r: Radius
    :param float mu: Mean
    :param float w: gaussian width, intensity values fall to 1/e2
    :param float Io: Peak intensity at this plane
    :param float offset: Intensity offset, background

    :returns float Ir: Intensity at radius r
    '''
    I_r = Io*np.exp(-2*(r - mu)**2/w**2)

    return I_r


def gaussian_mixture(r: NDArray,
                     w0: float,
                     I0: float,
                     mu0: float,
                     w1: float,
                     I1: float,
                     mu1: float,
                     w2: float,
                     I2: float,
                     mu2: float,
                     bg: float):
    '''
    3 Peak Gaussian intensity model
    '''
    I = (I0*np.exp(-2*(r - mu0)**2/w0**2)
        + I1*np.exp(-2*(r - mu1)**2/w1**2)
        + I2*np.exp(-2*(r - mu2)**2/w2**2)
        + bg)

    return I


def gaussian_mixture_no_bg(r: NDArray,
                           w0: float,
                           I0: float,
                           mu0: float,
                           w1: float,
                           I1: float,
                           mu1: float,
                           w2: float,
                           I2: float,
                           mu2: float):
    '''
    3 Peak Gaussian intensity model
    '''
    I = (I0 * np.exp(-2 * (r - mu0)**2 / w0**2)
         + I1 * np.exp(-2 * (r - mu1)**2 / w1**2)
         + I2 * np.exp(-2 * (r - mu2)**2 / w2**2))

    return I


def gauss_waist(wl: float,
                ri:float,
                na: float = None,
                theta: float = None):
    '''
    Returns the Gaussian beam waist, the minimum Gaussian beam width, (intensity falls off by 1/e**2).
    Can Calculate using  NA or divergence angle, (half angle!)
     defined using

    Parameters
    ----------
    :param float wl: Wavelength
    :param float ri: Refractive index
    :param float na: Optional, focusing lens objective
    :param float theta: Convergence angle of focus

    :return float wo: Gaussian waist, std
    '''
    if na:
        wo = wl/(np.pi * na)*ri
    elif theta:
        wo = wl/(np.pi*theta)*ri
    else:
        raise ValueError("Must pass na or div angle, theta")
    return wo


def gauss_rayleigh(wl: float,
                   na: float = None,
                   wo: float = None,
                   ri: float = 1.0):
    '''
    Returns the Rayliegh length of Gaussian focus

    :param float wl: Wavelength
    :param float wo: Gaussian waist parameter (minimum Gaussian width)

    :return float zR: Rayleigh Range
    '''
    if na:
        zR = ri*wl/(np.pi*na**2)
    elif wo:
        zR = np.pi*ri*wo**2/wl
    else:
        raise ValueError("Must pass na or wo")

    return zR


def gauss_width(z: NDArray,
                wl: float,
                wo: float = None,
                na: float = None,
                ri: float = 1.0,
                offset: float = 0.0):
    '''
    Calculates the beam radius at distance z from the waist

    :param float z: Distance from focus
    :param float zR: Rayliegh Range
    :param float wo: Gaussian waist
    :param float a: z offset
    :return float w(z): Beam radius at z, defined by 1/e^2
    '''
    if wo:
        zR = gauss_rayleigh(wl=wl, wo=wo, ri=ri)
    elif na:
        zR = gauss_rayleigh(wl=wl, na=na, ri=ri)
        wo = gauss_waist(wl=wl, na=na, ri=ri)

    w_z = wo * np.sqrt(1 + ((z - offset)/zR)**2)

    return w_z


def gauss_curvature(z: NDArray,
                    wl: float,
                    wo: float,
                    ri: float = 1.0):
    '''
    Returns the wavefront radius of curvature

    :param float z: Distance from waist.
    :param float wl: Laser wavelength.
    :return float curvature: Field curvature at z.
    '''
    zR = gauss_rayleigh(wl=wl, wo=wo, ri=ri)
    R = z*(1 + (zR/z)**2)

    return R


def gauss_invwaist(z: float,
                   wl: float,
                   w: float):
    '''
    Solve for the beam waist given the radius some distance z from focus

    It can be solved either by using the roots function,
    or an analytical solution can be used, need to see which is faster...
    But before considering that, I need to decide which solution our is \pm?

    :params float z: distance from Gaussian focus.
    :params float wl: Laser wavelength
    :params float w: Gaussian beam width, std.
    '''
    coeff = [1, -w**2, ( wl*z/np.pi)**2]
    wo = np.sqrt(np.roots(coeff))

    return wo


def gauss_matrix_prop(q1: NDArray,
                      abcd: NDArray):
    '''
    1/q = 1/R - i wl / pi * w
    :param complex q1: Initial Gaussian complex parameter
    :param list abcd: vector of ABCD matrix components
    '''
    a, b, c, d = abcd
    q_final = (a*q1 + b)/(c*q1 + d)
    return q_final


#------------------------------------------------------------------------------#
# PSF functions
#------------------------------------------------------------------------------#

def psf_born_wolf(radius_xy: NDArray,
                  z_planes: NDArray,
                  na: float,
                  ri: float,
                  ko: float):
    '''
    radius_xy: 2d array of radii generated by meshgrid
    z_planes: 1d array of zplanes to calculate
    na: spot size parameter
    ri: propagation refractive index
    ko: wave number
    '''

    def integral(r):
        return quad(lambda rho: rho * bessel0(ko * na / ri * r * rho) \
                                    * np.exp(-1j * ko * rho**2 * z * (na / ri)**2 ), 0, 1)[0]

    field = np.zeros((len(z_planes), radius_xy.shape[0], radius_xy.shape[1]))

    for ii, z in enumerate(z_planes):
        field_flattened = np.ones((radius_xy.flatten().shape))

        with ProgressBar(minimum=5):
            tasks = [dask.delayed(integral)(r) for r in radius_xy.flatten()]
            results = dask.compute(*tasks)
            field_flattened = np.stack(results, axis=0)

        field[ii] = field_flattened.reshape(radius_xy.shape)

    return field


def airydisk(r: NDArray,
             na: float,
             wl: float):
    '''
    calculate 2d airy disk pattern

    I(r,0) = Io * (J_1(k*pupil_radius*r*NA))/(k*pupil_radius*r*NA)

    :param float r: NxN meshgrid arrray of radii
    :param float na: Focusing lens numerical aperture
    :param float wl: Wavelength
    '''
    bessel_arg = ((2 * np.pi  * na) / wl) * r
    airy_disk = (2 * bessel1(bessel_arg) / bessel_arg)**2

    # replace center pixel with 1
    airy_disk[np.logical_or(np.isinf(airy_disk), np.isnan(airy_disk))] = 1

    return airy_disk


def airydisk_value(r: float,
                   na: float = 0.3,
                   wl: float = 0.5):
    '''
    calculate airy disk at single radius

    I(r,0) = Io * (J_1(k*pupil_radius*r*NA))/(k*pupil_radius*r*NA)

    :param float radius: Radius to evaluate airy disk
    :param float na: Focusing lens numerical aperture
    :param float wl: Wavelength
    '''
    if r == 0:
        airy_disk = 1
    try:
        bessel_arg = ((2 * np.pi  * na) / wl) * r
        airy_disk = (2 * bessel1(bessel_arg) / bessel_arg)**2

    except RuntimeWarning:
        # Catch r=0 at airy center
        airy_disk = 1

    return airy_disk


def airydisk_2d(mask_radius: NDArray,
                x: NDArray,
                na: float = 0.5,
                wl: float = 0.5,
                sf: int = 2):

    '''
    calculate 2d airy disk pattern

    I(r,0) = Io * (J_1(k*r*r*NA))/(k*r*r*NA)

    :param float mask_radius: NxN meshgrid arrray of radii
    :param float na: Focusing lens numerical aperture
    :param float wl: Wavelength
    '''
    dx = x[1] - x[0]

    # sample airy disk over large grid
    airy_disk = airydisk_2d(mask_radius, na, wl)

    # super sampling
    # Create grid to calculate airy disk over.
    x_sample_max = 20
    x_max_idx = np.argmin(x - x_sample_max)
    x_min_idx = np.argmin(-x - x_sample_max)
    sample_idxs = np.arange(x_min_idx, x_max_idx, 1)

    for ii in sample_idxs:
        for jj in sample_idxs:
            # Create super sampled grid
            sample_grid_x = np.linspace(x[ii] - dx/2, x[ii] + dx/2, sf)
            sample_grid_y = np.linspace(x[jj] - dx/2, x[jj] + dx/2, sf)
            grid_xx, grid_yy = np.meshgrid(sample_grid_x, sample_grid_y)
            grid_radius = np.sqrt(grid_xx**2 + grid_yy**2)

            # replace current value with mean over super sampled grid
            airy_disk[ii, jj] = np.mean(airydisk_2d(grid_radius, na, wl))

    return airy_disk


def airy_waist(wl: float = 0.5,
               ri: float = 1.0,
               na: float = 0.5):

    return 0.61*wl*ri / na


#------------------------------------------------------------------------------#
# Optical path length functions
#------------------------------------------------------------------------------#

def botcherby_herschel_phase(rho: NDArray,
                             phi: float,
                             pnt_source_xyz: list = [0,0,0],
                             na: float = 0.5,
                             wl: float = 0.5,
                             ri: float = 1.0):
    """
    Calculate the phase in the pupil plane of an objective for a point source.

    :param NDArray rho: Pupil radial coordinate
    :param float phi: Azimuthal angle in pupil and real space
    :param list pnt_source_xyz: Point source coords in image plane.
    :param float na: Objective numerical aperture.
    :param float wl: wavelength.
    :param float ri: propagation refractive index.
    """
    x, y, z = pnt_source_xyz
    k = 2 * np.pi / wl
    alpha = np.arcsin(na)
    psi = 2*ri*k*np.sin(alpha/2) * ((x * rho * np.cos(phi))
                                    + (y * rho * np.sin(phi)*np.sqrt(1/np.sin(alpha/2)**2 - rho**2))
                                    + (z*(1/(2*np.sin(alpha/2)**2) - rho**2)))

    return psi


def defocused_opl(r: NDArray,
                  dz: float,
                  ri: float):
    """
    Analytic optical path length difference for ray with radius r, dz from a perfect focus.

    :param float r: Ray radius or height
    :param float dz: Distance from perfect focus
    :param float ri: Propagation media refractive index
    """
    return  ri*(np.sqrt(r**2 + dz**2) - dz)


def ray_opl_quad_curvature(rays,
                           pupil_radius):
    """
    Grab quadradic defocus radius of curvature from C_2 term of opld polynomial fit.

    R = 1/(2C_2)

    :param array rays: Rays like array
    :param float pupil_radius: Radius normalization factor for wf fit
    :param str method: Whether to fit OPL or OPLD, passed into wf fit
    :return float quad_curvature: Radius of curvature derived from C2 polynomial coeffiecient
    """
    wf_fit = rt.ray_opl_polynomial(rays=rays,
                                   pupil_radius=pupil_radius,
                                   method="opl")
    radius = 1/(2*wf_fit[2])

    return radius


def defocus_rays(rays,
                 dz: float,
                 na: float,
                 ret="rays"):
    """
    Adds the additional opl needed in BFP for some defocus in the BFP of an
    objective with NA and pupil radius equal to the radial extent of the rays.

    :param array rays: Rays like array
    :param float dz: Defocus distance, mm
    :param float na: Numerical aperture of rays
    :param str ret: Return rays array or phase
    :return rays or phase
    """
    # normalized radius
    rho = rays[-1, :, 0]/np.nanmax(np.abs(rays[-1, :, 0]))

    # add defocus to opl
    rays[-1, :, 3] += dz*np.sqrt(1 - (rho * na)**2)

    if ret=="rays":
        return rays

    elif ret=="opl":
        return rays[-1, :, 3]


def get_reference_sphere_opl(h, R):
    """
    Calculate the reference sphere path length as a function of pupil hieght (radius not normalized)

    h^6 order taylor expansion for spherical equation.

    :param float h: height from optical axis
    :param float R: radius of reference sphere
    :return float opl: path length as function of height for spherical rays focusing R away
    """
    return R * (1
                + h**2/(2*R**2)
                - h**4/(8*R**4)
                + h**6/(16*R**6))


def get_zernike_polynomial(r,
                           coeff=[]):
    """
    First three symetric zernike polynomial terms, piston (z0), focus(z3) and
    spherical & defocus (z8)

    Ref: Basic Wavefront Aberration Theory for Optical Metrology, JAMES C. WYANT,
         Optical Sciences Center, University of Arizona

    TODO: WIP
    params float r: Normalized wavefront radius.
    params float z0: Piston term
    params float z3: Defocus term.
    params float z8: Spherical terms.
    return float zernike: Return the wavefront characterized with Zernike coefficients.
    """
    z0, z3, z8 = coeff
    zernike = (z0
               + z3*(2*r**2 -1)
               + z8*(6*r**4 - 6*r**2 + 1))

    return zernike


#------------------------------------------------------------------------------#
# Misc. analytic forms
#------------------------------------------------------------------------------#

def quadratic(
    r: NDArray,
    a: float,
    b: float,
    c: float    
) -> NDArray:
    """_summary_

    Parameters
    ----------
    r : NDArray
        f(x), x to evaluate quadratic
    a : float
        squared term
    b : float
        linear term
    c : float
        constant term

    Returns
    -------
    NDArray
        quadratic function
    """
    return a*r**2 + b*r + c