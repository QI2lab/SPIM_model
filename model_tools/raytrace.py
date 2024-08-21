# -*- coding: utf-8 -*-
"""
Ray Tracing Functions

Steven Sheppard
2024/04/24
"""
import model_tools.propagation as pt

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.stats import truncnorm
from scipy.linalg import lstsq
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import sympy as sy
import datetime

DEBUG =False
if not DEBUG:
    import warnings
    warnings.filterwarnings("ignore")

#%% Ray creation

def create_rays(type: str = "gaussian",
                source: str = "infinity",
                n_rays: int = 1e3,
                diameter: str = 10,
                sampling_limit: float = 10,
                radial_offset: float = 0,
                axial_offset: float = None,
                na: float = 0.3):
    """
    Create an array of rays with an initial radial density to represent
    the initial electric field amplitude.

    Indexing rays:
            rays[Surface, ray_idx, parameter]
    Indexing parameters:
            rays[0, :][radius, angle, z position, OPL]

    If type=="gaussian", diameter refers to the Gaussian E(r,z) width, w_0.

    - Use Gaussian beam definition of width = w, where w is
      the radius at which the field amplitude falls off
      by 1/e or intensity falls off by 1/e^2.
        - https://en.wikipedia.org/wiki/Gaussian_beam

    Ray types assign the radial sampling distribution:
        "gaussian" : Sampled from truncated Gaussian PDF
                     D = w = sqrt(2) * standard deviation (field amplitude)
        "uniform" : Sampled from uniform PDF
        "flat_top" : Evenly sampled set of rays using np.linspace

    Ray sources assign the angular distribution:
        "infinity" : Collimated beam
        "point" : Ray fan defined by na, does not accurately represent amplitude distribution

    :param str type: radial distribution, "guassian" "uniform" "flat_top"
    :param str source: angular distribution, "infinity" or "point"
    :param int n_rays: number of rays
    :param float diameter: beam diameter, Gaussian intensity 1/e width
    :param float radial_offset: radial offset distance from optical axis, mm
    :param float sampling_limit: Scipy truncation distance in terms of sigmas
    :para float na: initial rays numerical aperture, required for point source
    :return array rays:
    """
    # initiate rays
    rays = np.zeros((int(n_rays), 4))

    # Check for point source input
    if source == "point":
        # Define maximum angle, assumes rays start in air
        theta_max = np.arcsin(na)

        # Define ray fan
        if type=="gaussian":
            theta = np.arctan(np.sort(truncnorm.rvs(-sampling_limit,
                                                    sampling_limit,
                                                    loc=0,
                                                    scale=np.tan(theta_max),
                                                    size=int(n_rays))))
        elif type == "uniform":
            theta = np.arctan(np.sort(np.random.uniform(low=-np.tan(theta_max),
                                                        high=np.tan(theta_max),
                                                        size=int(n_rays))))
        elif type == "flat_top":
            theta = np.arctan(np.linspace(-np.tan(theta_max),
                                          np.tan(theta_max),
                                          num=int(n_rays)))

        rays[:, 0] = radial_offset
        rays[:, 1] = theta

    # Check for flat wavefront input
    elif source == "infinity":
        if type == "gaussian":
            rays[:, 0] = truncnorm.rvs(-sampling_limit/(np.sqrt(2)),
                                       sampling_limit/(np.sqrt(2)),
                                       loc=radial_offset,
                                       scale=(diameter/(np.sqrt(2))),
                                       size=int(n_rays))
        elif type == "uniform":
            rays[:, 0] = (np.random.uniform(low=-(diameter/2 + radial_offset),
                                            high=(diameter/2 + radial_offset),
                                            size=int(n_rays)))
        elif type == "flat_top":
            rays[:, 0] = np.linspace(-diameter/2 + radial_offset,
                                     diameter/2 + radial_offset,
                                     num=int(n_rays))

    # Paraxial ray, if there is none, make one
    is_on_axis = np.abs(rays[:, 0]) < 1e-6
    is_paralell = np.abs(rays[:, 1]) < 1e-6
    paraxial_ray_filter = np.logical_and(is_on_axis, is_paralell)

    if not np.any(paraxial_ray_filter):
        # add ray at zero (paraxial ray)
        rays[-1] = np.array([1e-6, 0, 0, 0])

    if source=="infinity":
        # sort rays based on initial height
        sort_index = np.argsort(rays[:, 0])
        rays = rays[sort_index, :]

    elif source=="point":
        # sort based on initial angle
        sort_index = np.argsort(rays[:, 1])
        rays = rays[sort_index, :]

    if axial_offset:
        rays[:,2] = axial_offset

    # Expand array dims, the first plane is index 0
    rays = np.expand_dims(rays, axis=0)

    return rays


#%% Optical classes


class Thick_lens:
    """

    """
    def __init__(self,
                 z1: float,
                 r1: float,
                 t: float,
                 ri: float,
                 r2: float,
                 aperture_radius: float = 12.5,
                 ri_in: float = 1.0,
                 ri_out: float = 1.0,
                 type: str = None):
        """
        Create a thick lens defined by two surfaces and maximum thickness, at vertex.

        Radius sign convention: convex R1>0, R2<0

        Reference for principle planes and focal length:

        :param float z1: First lens vertex
        :param float r1: Radius of spherical lens surface 1
        :param float t: Lens thickness at vertex
        :param float ri: Refractive index of lens material
        :param float r2: Radius of spherical lens surface 2
        :param float aperture_radius: Lens aperture radius at front vertex
        :param float ri_in: Left hand side refractive index
        :param float ri_out: Right hand side refractive index
        :return class thicklens:
        """
        self.type = "Thick_lens"
        self.z1  = z1
        self.ri = ri
        self.r1 = r1
        self.r2 = r2
        self.t = t
        self.z2 = self.z1 + self.t
        self.ri_in = ri_in
        self.ri_out = ri_out
        self.aperture = aperture_radius

        # assign lens type using surface curvature
        if type is not None:
            self.type = type
        else:
            if (self.r1==np.inf
                    and self.r2==np.inf):
                self.type="plano-plano"
            elif self.r1==np.inf:
                self.type="plano-sphr"
            elif self.r2==np.inf:
                self.type="sphr-plano"
            else:
                self.type="spherical"

        # Use ABCD matrix methods to calculate cardinal points
        lens_abcd = [abcd_refract_spher(R=self.r2,
                                        n1=self.ri,
                                        n2=self.ri_out),
                     abcd_freespace(d=self.t,
                                    n=self.ri),
                     abcd_refract_spher(R=self.r1,
                                        n1=self.ri_in,
                                        n2=self.ri)]
        abcd = np.array([[1, 0],
                         [0, 1]])
        for m in lens_abcd:
            abcd = abcd.dot(m)

        f1, f2, h1, h2, f1_pp, f2_pp = abcd_cardinal_points(abcd,
                                                            self.ri_in, self.ri_out)
        # assign lens parameters
        self.f1 = f1
        self.f2 = f2
        self.h1 = h1
        self.h2 = h2
        self.f1_pp = f1_pp
        self.f2_pp = f2_pp
        self.abcd = abcd
        self.bfp = self.z1 + self.f1
        self.ffp = self.z2 + self.f2
        self.na = ri_in * np.sin(np.arctan(self.aperture / self.f1_pp))
        self.pupil_radius = self.na * self.f1_pp * ri_in


    def params(self):
        params = self.__dict__
        return params


    def draw(self, ax,
             label: str = "Thick_lens",
             color: str = "k",
             show_focal_planes: bool = False):
        """
        Draw lens on matplotlib axes

        :param class self: Thick lens
        :param class ax: Matplotlib figure axes
        :param str label: Label associated with lens
        :param str color: Lens surface color
        :param boolean show_focal_planes: Optionally display lens BFP and FFP
        """
        # Draw first lens surface
        if self.r1==np.inf:
            ax.plot([self.z1]*2, [-self.aperture, self.aperture], c=color)
            t1 = 0
        else:
            # Calculate thickness of curvature section, t1
            t1 = self.r1 * (1 - np.sqrt(1 - (self.aperture/self.r1)**2))
            x1 = np.linspace(self.z1, self.z1 + t1, 5001)
            if self.r1>0:
               t1 = -1*t1
            elif self.r1<0:
                t1 = t1
            x_offset = self.z1 + self.r1
            c1 = np.sqrt(self.r1**2 - (x1 - x_offset)**2)

            # Plot surface
            ax.plot(x1, c1, c=color)
            ax.plot(x1, -c1, c=color)

        # Draw second lens surface
        if self.r2==np.inf:
            ax.plot([self.z2]*2, [-self.aperture, self.aperture], c=color)
            t2 = 0
        else:
            # Calculate thickness of curvature section, t2
            t2 = self.r2 * (1 - np.sqrt(1 - (self.aperture / self.r2)**2))
            x2 = np.linspace(self.z2, self.z2 + t2, 5001)
            x_offset = self.z2 + self.r2
            c2 = np.sqrt(self.r2**2 - (x2 - x_offset)**2)

            # Plot surface
            ax.plot(x2, c2, c=color, label=label)
            ax.plot(x2, -c2, c=color)

        # Plot top/bottom of lens
        ax.hlines(y=self.aperture,
                  xmin=(self.z1 + t1),
                  xmax=(self.z2 + t2),
                  colors=[color]
                  )
        ax.hlines(y=-self.aperture,
                  xmin=(self.z1 + t1),
                  xmax=(self.z2 + t2),
                  colors=[color]
                  )

        if show_focal_planes:
            if np.abs(self.f1) < 200:
                ax.axvline(x=self.bfp, ls="--", c=color, label=label+"bfp")
            if np.abs(self.f2) < 200:
                ax.axvline(x=self.ffp, ls="--", c=color, label=label+"ffp")


    def raytrace(self, rays):
        """
        Raytrace through thick lens.
        Intersect and refract through each lens surface.

        :param array rays: [r, theta, z, opl]
        :param float ri_in: Refractive index of material to left of lens.
        :param float ri_out: Refractive index of material to right of lens.

        :return array rays: Rays at each surface concatenated with input rays.
        """
        # intersect first surface, check for plano surface
        if self.r1 == np.inf:
            rays = intersect_plane(rays,
                                   zf=self.z1,
                                   ri_in=self.ri_in,
                                   ri_out=self.ri,
                                   refract=True
                                   )
        else:
            rays = intersect_sphere(rays,
                                    [self.r1, self.z1, self.ri_in, self.ri], refract=True
                                    )

        # Crop rays after aperture
        filter_idx_aperture = np.where(np.abs(rays[-1,:,0]) >= self.aperture)
        rays[-1, :, :][filter_idx_aperture] = np.nan

        # interesect second surface, check for plano surface
        if self.r2 == np.inf:
            rays = intersect_plane(rays,
                                   zf=self.z2,
                                   ri_in=self.ri,
                                   ri_out=self.ri_out,
                                   refract=True
                                   )
        else:
            rays = intersect_sphere(rays,
                                    [self.r2, self.z2, self.ri, self.ri_out],
                                    refract=True
                                    )

        # Filter at second lens aperture
        filter_idx_aperture = np.where(np.abs(rays[-1,:,0]) >= self.aperture)
        rays[-1, :, :][filter_idx_aperture] = np.nan

        return rays


class Doublet_lens:
    """

    """
    def __init__(self,
                 z1: float,
                 r1: float,
                 t1: float,
                 ri1: float,
                 r2: float,
                 t2: float,
                 ri2: float,
                 r3: float,
                 aperture_radius=12.5,
                 ri_in=1.0,
                 ri_out=1.0,
                 type: str = None):
        """
        Create a doublet lens.

        Curvature sign convention:  convex -> R<0

        :param float z1: lens first vertex location
        :param float r1: first surface radius of curvature
        :param float t2: first lens thickness at midpoint
        :param float ri1: first lens refractive index
        :param float r2: second surface radius of curvature
        :param float t2: second lens thickness at midpoint
        :param float ri2: second lens refractive index
        :param float r3: final surface radius of curvature
        :return class doublet_lens:
        """
        self.type = "Doublet_lens"
        self.z1 = z1
        self.t = t1 + t2
        self.z2 = self.z1 + t1
        self.z3 = z1 + self.t
        self.r1 = r1
        self.t1 = t1
        self.ri1 = ri1
        self.r2 = r2
        self.t2 = t2
        self.ri2 = ri2
        self.r3 = r3
        self.aperture = aperture_radius
        self.ri_in = ri_in
        self.ri_out = ri_out

        if type!=None:
            self.type = type

        lens_abcd = [abcd_refract_spher(R=self.r3,
                                        n1=self.ri2,
                                        n2=self.ri_out),
                     abcd_freespace(d=self.t2,
                                    n=self.ri2),
                     abcd_refract_spher(R=self.r2,
                                        n1=self.ri1,
                                        n2=self.ri2),
                     abcd_freespace(d=self.t1,
                                    n=self.ri1),
                     abcd_refract_spher(R=self.r1,
                                        n1=self.ri_in,
                                        n2=self.ri1)
                     ]

        abcd = np.array([[1, 0],
                         [0, 1]])

        for m in lens_abcd:
            abcd = abcd.dot(m)

        f1, f2, h1, h2, f1_pp, f2_pp = abcd_cardinal_points(abcd,
                                                            self.ri_in, self.ri_out)
        # assign lens parameters
        self.f1 = f1
        self.f2 = f2
        self.h1 = h1
        self.h2 = h2
        self.f1_pp = f1_pp
        self.f2_pp = f2_pp
        self.abcd = abcd
        self.bfp = self.z1 - self.f1
        self.ffp = self.z3 + self.f2
        self.na = ri_in * np.sin(np.arctan(self.aperture / self.f1_pp))
        self.pupil_radius = self.na * self.f1_pp * ri_in


    def params(self):
        return self.__dict__


    def draw(self,
             ax,
             label: str = "Thick lens",
             color: str = "k",
             show_focal_planes: bool = False):
        """
        Draw lens on matplotlib axes

        :param class self: Doublet lens
        :param class ax: Matplotlib figure axes
        :param str label: Label associated with lens
        :param str color: Lens surface color
        :param boolean show_focal_planes: Optionally display lens BFP and FFP
        """
        # Draw first lens surface
        if self.r1==np.inf:
            ax.plot([self.z1]*2, [-self.aperture, self.aperture], c=color)
            t1 = 0
        else:
            # Calculate thickness of curvature section, t1
            t1 = self.r1 * (1 - np.sqrt(1 - (self.aperture/self.r1)**2))
            x1 = np.linspace(self.z1-2, self.z1 + t1, 5001)

            x_offset = self.z1 + self.r1
            c1 = np.sqrt(self.r1**2 - (x1 - x_offset)**2)

            # Plot surface
            ax.plot(x1, c1, c=color)
            ax.plot(x1, -c1, c=color)

        # Draw second lens surface
        if self.r2==np.inf:
            ax.plot([self.z2]*2, [-self.aperture, self.aperture], c=color)
            t2 = 0
        else:
            # Calculate thickness of curvature section, t2
            t2 = self.r2 * (1 - np.sqrt(1 - (self.aperture/self.r2)**2))
            x2 = np.linspace(self.z2, self.z2 + t2, 5001)

            x_offset = self.z2 + self.r2
            c2 = np.sqrt(self.r2**2 - (x2 - x_offset)**2)

            # Plot surface
            ax.plot(x2, c2, c=color, ls="--")
            ax.plot(x2, -c2, c=color, ls="--")


        # Draw third lens surface
        if self.r3==np.inf:
            ax.plot([self.z3]*2, [-self.aperture, self.aperture], c=color)
            t3 = 0
        else:
            # Calculate thickness of curvature section, t3
            t3 = self.r3 * (1 - np.sqrt(1 - (self.aperture/ self.r3)**2))
            x3 = np.linspace(self.z3, self.z3 + t3, 5001)

            x_offset = self.z3 + self.r3
            c3 = np.sqrt(self.r3**2 - (x3 - x_offset)**2)

            # Plot surface
            ax.plot(x3, c3, c=color)
            ax.plot(x3, -c3, c=color, label=label)

        # Plot top/bottom of lens
        ax.hlines(y=self.aperture,
                  xmin=(self.z1 + t1),
                  xmax=(self.z3 + t3),
                  colors=[color])
        ax.hlines(y=-self.aperture,
                  xmin=(self.z1 + t1),
                  xmax=(self.z3 + t3),
                  colors=[color])

        if show_focal_planes:
            if np.abs(self.f1) < 1000:
                ax.axvline(x=self.bfp, ls="--", c=color, label=label+"bfp")
            if np.abs(self.f2) < 1000:
                ax.axvline(x=self.ffp, ls="--", c=color, label=label+"ffp")


    def raytrace(self, rays):
        """
        Raytrace through each thick lens.

        :param array rays: [r, theta, y, opl]
        :param float ri_in: Refractive index of material to left of lens1
        :param float ri_out: Refractive index of material to right of lens2
        :return array rays: Rays at each surface concatenated with input rays.
        """
        # intersect first surface, first lens
        if self.r1 == np.inf:
            rays = intersect_plane(rays,
                                   zf=self.z1,
                                   ri_in=self.ri_in,
                                   ri_out=self.ri1,
                                   refract=True)
        else:
            rays = intersect_sphere(rays,
                                    [self.r1, self.z1, self.ri_in, self.ri1], refract=True)

        # Filter at lens aperture
        filter_idx_aperture = np.where(np.abs(rays[-1,:,0]) >= self.aperture)
        rays[-1, :, :][filter_idx_aperture] = np.nan

        # intersect second surface and refract into second lens material
        if self.r2 == np.inf:
            rays = intersect_plane(rays,
                                   zf=self.z2,
                                   ri_in=self.ri1,
                                   ri_out=self.ri2,
                                   refract=True
                                   )
        else:
            rays = intersect_sphere(rays,
                                    [self.r2, self.z2, self.ri1, self.ri2], refract=True
                                    )

        # intersect final surface and refract into output medium
        if self.r3 == np.inf:
            rays = intersect_plane(rays,
                                   zf=self.z3,
                                   ri_in=self.ri2,
                                   ri_out=self.ri_out,
                                   refract=True
                                   )
        else:
            rays = intersect_sphere(rays,
                                    [self.r3, self.z3, self.ri2, self.ri_out], refract=True
                                    )

        #  filter at lens aperture
        filter_idx_aperture = np.where(np.abs(rays[-1,:,0]) >= self.aperture)
        rays[-1, :, :][filter_idx_aperture] = np.nan
        rays[-2, :, :][filter_idx_aperture] = np.nan

        return rays


class Perfect_lens:
    """

    """
    def __init__(self,
                 z1: float,
                 f: float,
                 na: float,
                 wd: float = None,
                 fov: float = None,
                 mag: float = None,
                 ri_in: float = 1.0,
                 ri_out: float = 1.0,
                 type: str = None):
        """
        Perfect lens meeting Abbe sine condition.

        :param float z1: Lens principle plane
        :param float f: lens focal length
        :param float na: lens numerical aperture
        :param float ri_in: Left hand side refractive index
        :param float ri_out: Right hand side refractive index
        :return class Perfect_lens:
        """
        # If no working distance is give, use the focal length
        if not wd:
            wd = f
        self.z1 = z1
        self.f = f
        self.na = na
        self.wd = wd
        self.fov = fov
        self.mag = mag
        self.pupil_radius = np.abs(f * na)
        self.f1 = f*ri_in
        self.f2 = f*ri_out
        self.bfp = z1 - self.f1
        self.ffp = z1 + self.f2
        self.ri_in = ri_in
        self.ri_out = ri_out
        self.aperture = np.abs(f * na)

        if type is not None:
            self.type = type
        elif not fov:
            self.type="perfect tel. lens"
        elif fov and mag:
            self.type="objective"
            self.mag=mag
            self.fov=fov

        # abcd matrix propagates from bfp to lens center and then transforms using thin lens
        lens_abcd = [abcd_thinlens(self.f),
                     abcd_freespace(self.f1, 1.0)]

        abcd = np.array([[1, 0],
                         [0, 1]])

        for m in lens_abcd:
            abcd = abcd.dot(m)

        self.abcd = abcd


    def params(self):
        return self.__dict__


    def draw(self,
             ax,
             label: str = "Perfect_lens",
             color: str = "k",
             show_focal_planes: bool = False):
        """
        Draw lens on matplotlib axes.
        Highlight lens z1 position with thick line and optionally show front and back focal planes.

        :param class self: Thick lens
        :param class ax: Matplotlib figure axes
        :param str label: Label associated with lens
        :param str color: Lens surface color
        :param boolean show_focal_planes: Optionally display lens BFP and FFP
        """
        # Plot line at lens vertex, z1
        ax.axvline(x=self.z1, c=color, lw=3, label=label)

        if show_focal_planes:
            if np.abs(self.f1)<1000:
                ax.axvline(x=self.bfp, c="r", ls="--", label=label + "bfp")
            if np.abs(self.f2)<1000:
                ax.axvline(x=self.ffp, c="r", ls="--", label=label + "ffp")


    def raytrace(self, rays, final_plane="pp"):
        """
        Raytrace through perfect lens via transform.
        No output refractive index is needed, uses the design RI.

        :param array rays: [r, theta, y, opl]
        :param float ri_in: Refractive index to the left of lens
        :param str final_plane: pp~PrinciplePlane, bfp~front focal plane
        :return array rays: Rays at back specified plane, concatenated with input rays
        """
        # Check to see if rays is single or multi-d stack
        if rays.ndim == 2:
            rays = np.expand_dims(rays, axis=0)

        # Propagate rays to FFP and principle plane
        rays1 = intersect_plane(rays,
                                self.bfp,
                                ri_in=self.ri_in,
                                refract=False) # BFP
        rays2 = intersect_plane(rays1,
                                self.z1,
                                ri_in=self.ri_in,
                                refract=False) # lens center

        # initial ray params at FFP
        r_i = rays1[-1, :, 0]
        theta_i = rays1[-1, :, 1]
        opl_i = rays1[-1, :, 3]

        # Rays at BFP obey transform and abbe sign conditions
        rays4 = np.zeros(np.shape(rays1[0]))
        rays4[:, 0] = self.ri_in * self.f * np.sin(theta_i)
        rays4[:, 1] = -np.arcsin(r_i / (self.f * self.ri_out))
        rays4[:, 2] = self.z1 + self.f * self.ri_out
        rays4[:, 3] = (opl_i
                       + self.ri_in*self.f
                       + self.ri_out*self.f
                       - r_i * self.ri_in*np.sin(theta_i))

        rays4 = np.expand_dims(rays4, axis=0)

        # filter at the bfp aperture
        filter_idx_bfp = np.abs(rays1[-1,:,0]) > self.pupil_radius

        if self.fov and self.mag:
            # filter at ffp FOV
            filter_idx_ffp = np.abs(rays4[-1,:,0]) > self.fov/self.mag
            filter_ang_ffp = np.abs(rays4[-1,:,1]) > np.arcsin(self.na/self.ri_out)

            filter = np.logical_or.reduce((filter_idx_bfp,
                                           filter_ang_ffp,
                                           filter_idx_ffp))

            # Print("Perfect lens Filtering Results:\n",
            #       f"FFP hieght filter: {np.sum(filter_idx_bfp)} \n",
            #       f"BFP hieght filter: {np.sum(filter_idx_ffp)} \n",
            #       f"BFP angle filter: {np.sum(filter_ang_ffp)} \n",
            #       f"Total filtered: {np.sum(filter)} \n")

        else:
            filter = filter_idx_bfp

        rays4[-1, filter, :] = np.nan

        # Back propagate to lens principle plane
        rays3 = intersect_plane(rays4,
                                self.z1,
                                ri_in=self.ri_out,
                                refract=False)

        if final_plane == "pp":
            # Return rays to lens principle plane before and after lens
            rays = np.concatenate((rays,
                                   rays1[-1:],
                                   rays2[-1:],
                                   rays3[-1:]))

        elif final_plane=="ffp":
            # return rays at each plane to ffp
            rays = np.concatenate((rays,
                                   rays1[-1:],
                                   rays2[-1:],
                                   rays3[-1:],
                                   rays4[-1:]))

        return rays


def create_etl(z1: float,
               dpt: float,
               d: float,
               t0: float,
               ri: float = 1.3,
               ri_in: float = 1.0,
               ri_out: float = 1.0):
    """
    Create an ETL type lens using thick lens class.
    Calculates the radius of curvature using the lens power in diopters.
    Default arguments correspond to Optotune EL-16-40.

    units: mm

    etl.t0 = Flat etl thickness, minimum thickness

    TODO: add option to pass in voltage or focal length instead of dpt

    :param float dpt: Lens power in diopters, units: m**-1
    :param float z1: z1 of lens vertex (plano surface.)
    :param float diameter: ETL diameter
    :param float ri: ETL refractive index
    :param float t0: Minimum thickness when lens is flat

    :return class etl: Lens class for ETL
    """
    if dpt==0.0:
        etl = Thick_lens(z1=z1,
                         r1=np.inf,
                         t=t0,
                         ri=ri,
                         r2=np.inf,
                         aperture_radius=d/2,
                         ri_in=ri_in, ri_out=ri_out,
                         type="etl")

        # assign back principle plane at the back vertex for plano/curved lens
        etl.t0 = t0
        etl.dpt = 0.0
        etl.type="etl"
        etl.dt=0

    else:
        # Convert to dpt**-1 to mm with factor of 10**3
        # Relate the focal length to the Radis of curv. from lens maker"s eqn
        r2 = -(ri - 1) / dpt * 10**3

         # Calculate curvature thickness from geometry
        curvature_thickness = r2*(1 - np.sqrt(1 - (d/(2*r2))**2))
        t = t0 - curvature_thickness

        etl = Thick_lens(z1=z1,
                         r1=np.inf,
                         t=t,
                         ri=ri,
                         r2=r2,
                         aperture_radius=d/2,
                         ri_in=ri_in,
                         ri_out=ri_out,
                         type="etl")

        etl.dpt = dpt
        etl.t0 = t0
        etl.dt = curvature_thickness

    return etl


def etl_dz_to_dpt(rf_dz,
                  obj_f,
                  relay_mag,
                  n_image):
    """
    takes remote focus distances in unit of mm, returns diopter [m^-1]

    params rf_ds [mm]
    returns dpt [m^-1]
    """
    return - rf_dz/n_image * (relay_mag/obj_f)**2 * 1e3


def etl_dpt_to_dz(dpt,
                  obj_f,
                  relay_mag,
                  n_image):
    """
    returns defocus in mm

    params dpt [m^-1]
    params obj_f [mm]
    returns rf_dzs [mm]
    """
    rf_dzs = -n_image*dpt*(obj_f/relay_mag)**2 * 1e-3
    return rf_dzs


#%% Ratracing functions


def raytrace_ot(optical_train: list = [],
                rays: np.ndarray = None,
                fit_plane: str = None,
                fit_method: str="opld",
                wl: float = 0.0005,
                rays_type: str = "gaussian",
                rays_source: str = "infinity",
                rays_diameter: float = None,
                rays_na: float = None,
                n_rays: int = 3e6,
                grid_params: list = [],
                xy_grid_padding: float = 0.100,
                return_rays: str="all",
                plot_raytrace_results: bool = False,
                showfig: bool = False,
                save_path: Path = None):
    """
    Ray trace optical train that produces a focus and locate ray focal planes.
    Optionally choose to fit using a matching psuedo objective to resample the rays from the focal plane.

    fit plane: "paraxial", "midpoint", "diffraction",


    :param list optical_train:
    :param str ray_type: type arg passed to create_rays()
    :param float rays_diameter: diameter arg passed to create_rays()
    :param str fit_plane: Optional, define the to fit using "paraxial", "midpoint" or "diffraction"
    :param float wl: wl used in wavefront fitting
    :param str return_rays: Option to return "all" or "first and last" plane of rays

    """
    # Create rays if none are provided
    if rays is None:
        rays = create_rays(type=rays_type,
                           source=rays_source,
                           n_rays=n_rays,
                           diameter=rays_diameter,
                           na = rays_na,
                           sampling_limit=10)

    # raytrace through lenses
    for lens in optical_train:
        rays = lens.raytrace(rays)

    # find the focal plane using ray tracing paraxial and marginal ray focii.
    fp_paraxial, fp_midpoint, fp_marginal = ray_focal_plane(rays=rays,
                                                                 ri=1.0,
                                                                 method="all")

    if fit_plane:
        # find the focusing objective
        if len(optical_train)==1:
            obj=optical_train[0]
        else:
            try:
                ii = 1
                obj_type = ""
                while ("obj" not in obj_type
                       or "erfect" in obj_type
                       or "tel" in obj_type):
                    obj_type = optical_train[-ii].type
                    ii += 1
                obj = optical_train[-(ii-1)]
            except Exception as e:
                print(f"Objective not identified, {ii}, \n Error:{e}")

    if fit_plane==None:
        if plot_raytrace_results:
            plot_rays(intersect_optical_axis(rays, ri=lens.ri_out),
                      n_rays_to_plot=15,
                      optical_train=optical_train,
                      show_focal_planes=True,
                      planes_of_interest={"paraxial f.p.":fp_paraxial,
                                          "midpoint f.p.":fp_midpoint,
                                          "marginal f.p.":fp_marginal},
                      title=f"Raytracing OT",
                      figsize=(40,10),
                      save_path=save_path,
                      showfig=showfig)

        if return_rays=="first and last":
            return_ray = np.stack([rays[0], rays[-1]])
        elif return_rays=="all":
            return_ray = rays
        else:
            return_ray = None

        return {"rays":return_ray,
                "paraxial_focal_plane":fp_paraxial,
                "midpoint_focal_plane":fp_midpoint,
                "marginal_focal_plane":fp_marginal,
                "axial_extent":fp_marginal - fp_paraxial,
                "optical_train":optical_train,
                "fit_plane":None}

    elif fit_plane=="diffraction":
        # Prepare electric field grid
        x, radius_xy, extent_xy, z, extent_zx = grid_params
        n_xy = x.shape[0]

        # raytrace to propagation plane
        rays_to_field_z = rays_to_field_plane(rays, np.max(x),xy_grid_padding)
        rays = intersect_plane(rays,
                               rays_to_field_z,
                               ri_in=lens.ri_out,
                               refract=False
                               )

        # Create initial field to propagate
        initial_field = rays_to_field(mask_radius=radius_xy,
                                      rays=rays,
                                      ko=2 * np.pi / wl,
                                      amp_binning=n_xy//5,
                                      amp_type="power",
                                      phase_type="opld",
                                      results="field",
                                      power=0.1,
                                      plot_field=DEBUG,
                                      showfig=showfig,
                                      save_path=None
                                      )

        # Estimate focal plane using plane of maximum intensity
        fp_diffraction = pt.diffraction_focal_plane(initial_field=initial_field,
                                                  grid_params=grid_params,
                                                  wl=wl,
                                                  ri=lens.ri_out,
                                                  focal_plane_guess=fp_midpoint,
                                                  rays_to_field_z=rays_to_field_z,
                                                  interp_dz=wl/10,
                                                  DEBUG=False
                                                  )

        focal_plane = fp_diffraction

        # neative to raytrace backwards from focal plane
        fit_obj_f = (focal_plane-obj.z1) / lens.ri_out
        fit_obj_z = focal_plane + fit_obj_f*lens.ri_out

        fit_obj =  Perfect_lens(z1=fit_obj_z,
                                f=fit_obj_f,
                                na=obj.na,
                                ri_in=lens.ri_out,
                                ri_out=lens.ri_out
                                )

        fit_rays = fit_obj.raytrace(rays[-1], final_plane="ffp")

        wavefront_results = ray_opl_analysis(pupil_rays=fit_rays,
                                             pupil_radius=fit_obj.pupil_radius,
                                             fit_method=fit_method,
                                             wl=wl
                                             )


        if plot_raytrace_results:
            optical_train_to_plot = optical_train + [fit_obj]
            plot_rays(np.concatenate([rays, fit_rays]),
                      n_rays_to_plot=15,
                      optical_train=optical_train_to_plot,
                      show_focal_planes=True,
                      show_legend=True,
                      planes_of_interest={"paraxial f.p.":fp_paraxial,
                                          "midpoint f.p.":fp_midpoint,
                                          "marginal f.p.":fp_marginal,
                                          "diffraction f.p.":fp_diffraction},
                      title=f"focal planes: {[fp_paraxial, fp_midpoint, fp_diffraction]}",
                      figsize=(40,10),
                      save_path=save_path,
                      showfig=showfig
                      )


        if return_rays=="first and last":
            return_ray = np.stack([rays[0], rays[-1]])
        elif return_rays=="all":
            return_ray = rays
        else:
            return_ray = None

        return {"rays":return_ray,
                "paraxial_focal_plane":fp_paraxial,
                "midpoint_focal_plane":fp_midpoint,
                "marginal_focal_plane":fp_marginal,
                "axial_extent":fp_marginal - fp_paraxial,
                "diffraction_focal_plane":fp_diffraction,
                "opl_fit":wavefront_results["wavefront fit"],
                "strehl":wavefront_results["strehl"],
                "rms":wavefront_results["rms"],
                "optical_train":optical_train,
                "fit_plane":"diffraction"
                }

    else:
        if fit_plane=="midpoint":
            focal_plane = fp_midpoint

        elif fit_plane=="paraxial":
            focal_plane = fp_paraxial

        elif fit_plane=="marginal":
            focal_plane = fp_marginal

        # neative to raytrace backwards from focal plane
        fit_obj_f = (focal_plane-obj.z1) / lens.ri_out
        fit_obj_z = focal_plane + fit_obj_f*lens.ri_out

        fit_obj =  Perfect_lens(z1=fit_obj_z,
                                f=fit_obj_f,
                                na=obj.na,
                                ri_in=lens.ri_out,
                                ri_out=lens.ri_out
                                )

        fit_rays = fit_obj.raytrace(rays[-1], final_plane="ffp")

        wavefront_results = ray_opl_analysis(pupil_rays=fit_rays,
                                             pupil_radius=fit_obj.pupil_radius,
                                             fit_method=fit_method,
                                             wl=wl
                                             )

        if plot_raytrace_results:

            optical_train_to_plot = optical_train + [fit_obj]

            plot_rays(np.concatenate([rays, fit_rays]),
                      n_rays_to_plot=15,
                      optical_train=optical_train_to_plot,
                      planes_of_interest={"paraxial f.p.":fp_paraxial,
                                          "midpoint f.p.":fp_midpoint,
                                          "marginal f.p.":fp_marginal},
                      show_focal_planes=True,
                      show_legend=True,
                      title=(f"focal planes: paraxial={fp_paraxial:.2f}, ",
                             f"midpoint={fp_midpoint:.2f}, ",
                             f"marginal={fp_marginal:.2f}"),
                      figsize=(40,10),
                      ax=None,
                      save_path=save_path,
                      showfig=showfig
                      )

        if return_rays=="first and last":
            return_ray = np.stack([rays[0], rays[-1]])
        elif return_rays=="all":
            return_ray = rays
        else:
            return_ray = None

        return {"rays":return_ray,
                "paraxial_focal_plane":fp_paraxial,
                "midpoint_focal_plane":fp_midpoint,
                "marginal_focal_plane":fp_marginal,
                "axial_extent":fp_marginal - fp_paraxial,
                "diffraction_focal_plane": None,
                "opl_fit":wavefront_results["wavefront fit"],
                "strehl":wavefront_results["strehl"],
                "rms":wavefront_results["rms"],
                "optical_train":optical_train,
                "fit_plane":fit_plane
                }


def calc_opl(dz: float,
             theta: float,
             ri: float):
    """
    Calculate optical path length for a ray segment.

    :param float dz: Ray segment projection along optical axis
    :param float theta: Ray angle, measured off optical axis
    :param float ri: refractive index of propagation media

    :return float opl: Optical path length
    """
    return  ri * (dz / np.cos(theta))


def refract_angle(theta_in:float,
                   ri_in: float,
                   ri_out: float):
    """
    Refract according to Snell"s law.

    :param float theta_in: Incident angle, measured off normal
    :param float ri_in: Incident refractive index
    :param float ri_out Output refractive index
    :returns float theta_out: Refracted angle, measured off surface normal
    """
    return np.arcsin(ri_in * np.sin(theta_in) / ri_out)


def intersect_plane(rays: np.ndarray,
                    zf: float,
                    ri_in: float = 1.0,
                    refract: bool = False,
                    ri_out: float = None):
    """
    Trace rays to zf. if refract is True then the rays are refracted.
    if refract, must also pass relevant refractive indices.

    :param array rays: rays type array
    :param float zf: y coordinate to trace rays to
    :param float ri_in: Propagation media refractive index
    :param boolean refract: Optional, refract at plane
    :param float ri_out: Right hand side refractive index used in refraction
    :returns array rays: old rays with new rays traced to zf appended
    """
    # Check to see if rays is single or multi-d stack
    # Single stack of rays has ndim==2
    if rays.ndim == 2:
        rays = np.expand_dims(rays, axis=0)

    if refract:
        if not ri_out:
            raise ValueError("Intersect plane with refraction requires an output refractive index")

    rays_new = np.zeros(np.shape(rays[0]))

    dz = zf - rays[-1, :, 2]
    dr = dz * np.tan(rays[-1, :, 1])
    dopl = calc_opl(dz, rays[-1, :, 1], ri_in)

    # Calculate new ray params
    rays_new[:, 0] = rays[-1, :, 0] + dr

    if refract and ri_in!=ri_out:
        rays_new[:, 1] = refract_angle(rays[-1, :, 1], ri_in, ri_out)
    else:
        rays_new[:, 1] = rays[-1, :, 1]

    rays_new[:, 2] = zf
    rays_new[:, 3] = rays[-1, :, 3] + dopl

    # Combine old and new rays
    rays_new = np.expand_dims(rays_new, axis=0)
    rays_new = np.concatenate((rays, rays_new))

    return rays_new


def intersect_sphere(rays: np.ndarray,
                     surface_param: list,
                     refract: bool = True):
    """
    Trace rays to their intersection with spherical geometry. This is typically used to ray trace
    through lens so refraction is True by default. Surface parameters are radius of sphere,
    vertex location, in/output RI.

    :param array rays: rays type array
    :param list surface_param: [Sphere Radius, Sphere OA-intersect, ri_in, ri_out]
    :param boolean refract: whether to refract at the surface
    :returns array rays: old rays with new rays traced to the surface intersect
    """
    rS, yS, ri_in, ri_out = surface_param

    # rays must have ndim=3, check for single stack of rays to expand dims.
    if rays.ndim == 2:
        rays = np.expand_dims(rays, axis=0)

    # Spherical center
    yl = yS + rS

    rays_new = np.zeros(np.shape(rays[0]))

    # Calculate z intersect with spherical surface
    # Quadratic Coefficients
    A = np.tan(rays[-1, :, 1])**2 + 1
    B = 2 * (rays[-1, :, 0] * np.tan(rays[-1, :, 1])
             - rays[-1, :, 2] * np.tan(rays[-1, :, 1])**2
             - yl)
    C = (yl**2
         - rS**2
         + rays[-1, :, 0]**2
         + (rays[-1, :, 2]*np.tan(rays[-1, :, 1]))**2
         - 2*rays[-1, :, 0]*rays[-1, :, 2]*np.tan(rays[-1, :, 1]))

    # Solve quadratic
    # Check for concave vs convex lens configs
    if rS >= 0:
        rays_new[:, 2] = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    else:
        rays_new[:, 2] = (-B + np.sqrt(B**2 - 4 *A*C)) / (2*A)

    # Calculate new ray heights
    dz = (rays_new[:, 2] - rays[-1, :, 2])
    rays_new[:, 0] = dz*np.tan(rays[-1, :, 1]) + rays[-1, :, 0]

    # New ray angle
    if refract and ri_in!=ri_out:
        theta_normal = np.arcsin(rays_new[:, 0] / rS)
        theta_in_relative = rays[-1, :, 1] + theta_normal
        theta_out_relative = refract_angle(theta_in_relative, ri_in, ri_out)
        rays_new[:, 1] = theta_out_relative - theta_normal
    else:
        rays_new[:, 1] = rays[-1, :, 1]

    # New ray optical path length
    rays_new[:, 3] = rays[-1, :, 3] + calc_opl(dz, rays[-1, :, 1], ri_in)

    # Combine rays
    rays_new = np.expand_dims(rays_new, axis=0)
    new_rays = np.concatenate((rays, rays_new))

    return new_rays


def intersect_optical_axis(rays: np.ndarray,
                           ri: float = 1.0):
    """
    Trace rays to their intersection with the optical axis, OA
    If rays are not converging, then rays are traced back to OA

    :param array rays: rays type array
    :param float ri: Propagation media refractive index

    :returns array rays: old rays with new rays traced to the surface intersect
    """
    # Check to see if rays is single or multi-d stack
    # Single stack of rays has ndim==2
    if rays.ndim == 2:
        rays = np.expand_dims(rays, axis=0)

    rays_new = np.zeros(np.shape(rays[0]))

    # Calculate new ray params
    rays_new[:, 0] = 0
    rays_new[:, 1] = rays[-1, :, 1]
    rays_new[:, 2] = (rays[-1, :, 2]
                      + np.abs(rays[-1, :, 0]/np.tan(rays[-1, :, 1])))
    rays_new[:, 3] = (rays[-1, :, 3]
                      + calc_opl((rays_new[:, 2]-rays[-1, :, 2]),
                                rays[-1, :, 1], ri))

    rays_new = np.expand_dims(rays_new, axis=0)
    new_rays = np.concatenate((rays, rays_new))

    nan_mask = np.isnan(rays[-1, :, 2])
    new_rays[-1, nan_mask, :] = np.nan
    return new_rays


def intersect_rays(rays: np.ndarray,
                   ri: float = 1.0):
    """
    TODO: Testing
    Intersect a set of rays.
    Returns rays at their z -intersects, which can be used to calculate caustic profiles.

    param array rays: Rays like array
    param float ri: Propagating media refractive index
    return array rays_new: Rays like array
    """
    # Check to see if rays is single or multi-d stack
    # Single stack of rays has ndim==2
    if rays.ndim == 2:
        rays = np.expand_dims(rays, axis=0)

    rays_new = np.zeros(np.shape(rays[0]))

    # angle doesn"t change..
    rays_new[:, 1] = rays[-1, :, 1]

    # Calculate new z planes
    rays_new[1:, 2] =(np.diff(rays[-1, :, 0])/np.tan(rays[-1, :, 1] - np.tan(rays[-1,:-1])))
    rays_new[0, 2] = intersect_optical_axis(rays, ri)[-1, 0, 2]

    # Calculate new distance from OA
    rays_new[:, 0] = (rays_new[:, 2]
                      - rays[-1, :, 2])*np.tan(rays[-1, :, 1]) + rays[-1, :, 0]

    # new OPL
    rays_new[:, 3] = (rays[-1, :, 3]
                      + calc_opl((rays_new[:, 2]-rays[-1, :, 2]),
                                rays_new[:, 1],
                                ri))

    rays_new = np.expand_dims(rays_new, axis=0)
    new_rays = np.concatenate((rays, rays_new))

    return new_rays


def ray_focal_plane(rays: np.ndarray,
                    ri: float = 1.0,
                    method: str = "paraxial",
                    return_rays: bool = False):
    """
    Calculate best focal plane.

    Methods:
    "paraxial": Use paraxial ray OA intersection
    "midpoint": Valid for small aberrations, returns the midway point between paraxial and marginal ray OA intersects.


    :param array rays: Rays like array
    :param float ri: refractive index of media
    :param str method: Optionally choose from "paraxial" rays or "midpoint" between paraxial and marginal ray OA intersections
    :return float focal_plane: Midpoint between paraxial and marginal ray OA intersects
    """
    if method not in ["paraxial","midpoint","marginal","both","all"]:
        raise ValueError("method must be given for ray tracing focal plane")

    # intersect the optical axis
    rays = intersect_optical_axis(rays=rays, ri=ri)

    # Create nan mask
    nan_mask = np.logical_not(np.isnan(rays[-1, :, 2]))

    # Paraxial ray, use on axis if it exist
    is_on_axis = np.abs(rays[0, :, 0][nan_mask]) < 1e-4
    is_paralell = np.abs(rays[0, :, 1][nan_mask]) < 1e-4
    paraxial_ray_filter = np.logical_and(is_on_axis, is_paralell)

    if np.any(paraxial_ray_filter):
        paraxial_focus = np.min(rays[-1, :, 2][nan_mask][paraxial_ray_filter])
    else:
        # Use the ray that originates nearest the optical axis
        paraxial_idx = (np.nanargmin(np.abs(rays[0, :, 0][nan_mask])
                                     + np.abs(rays[0, :, 1][nan_mask])))
        paraxial_focus = rays[-1, :, 2][nan_mask][paraxial_idx]

    # find the marginal ray, which may be cropped.
    marginal_ray_idx = np.argmax(np.abs(rays[0, :, 0][nan_mask]))
    marginal_focus = rays[-1, :, 2][nan_mask][marginal_ray_idx]

    # Take midpoint between marginal and paraxial ray intersects
    midpoint_focus = paraxial_focus*(1 - (1 - marginal_focus/paraxial_focus)/2)

    if method=="paraxial":
        if return_rays:
            return paraxial_focus, rays
        else:
            return paraxial_focus
    elif method=="midpoint":
        if return_rays:
            return midpoint_focus, rays
        else:
            return midpoint_focus
    elif method=="marginal":
        if return_rays:
            return marginal_focus, rays
        else:
            return marginal_focus
    elif method=="both":
        if return_rays:
            return [paraxial_focus, midpoint_focus], rays
        else:
            return [paraxial_focus, midpoint_focus]
    elif method=="all":
        if return_rays:
            return [paraxial_focus, midpoint_focus, marginal_focus], rays
        else:
            return [paraxial_focus, midpoint_focus, marginal_focus]


def ot_focal_plane(ot: list,
                   rays: np.ndarray = None,
                   aperture_radius: float=1.0,
                   method: str = "paraxial"):
    """
    Calculate best focal plane. Assumes the optical train takes plane wave as source

    Methods:
    "paraxial": Use paraxial ray OA intersection
    "midpoint": Valid for small aberrations, returns the midway point between paraxial and marginal ray OA intersects.


    :param array rays: Rays like array
    :param float ri: refractive index of media
    :param str method: Optionally choose from paraxial rays or halfway between paraxial and marginal ray OA intersections (Born & Wolf)
    :return float focal_plane: Midpoint between paraxial and marginal ray OA intersects
    """
    if rays is None:
        n_rays = int(1e3)
        rays = create_rays(type="flat_top",
                           source="infinity",
                           n_rays=n_rays,
                           diameter=aperture_radius*2
                           )
    for comp in ot:
        rays = comp.raytrace(rays)

    focal_planes = ray_focal_plane(rays,
                                        ri=comp.ri_out,
                                        method=method)
    return focal_planes


def rays_to_field_plane(rays: np.ndarray,
                        x_max: float,
                        padding: float = 0.050):
    """
    Calculate the zplane where the marginal ray"s radius is eqaul to the given max value.

    dz = (r_i - x_max) / abs(tan(theta))

    :param array rays: Rays array
    :param float x_max: Maximum extent of field array
    :param float padding: Expected field array padding in factors of maximum radius, x_max = padding*max_radius
    :retrun float z_final: z coordinate to propagate rays.
    """
    marginal_ray_idx = np.nanargmax(np.abs(rays[-1, :, 0]))
    dx = np.abs(rays[-1, marginal_ray_idx, 0]) - (x_max - padding)
    dz = dx / np.abs(np.tan(rays[-1, marginal_ray_idx, 1]))
    zf = rays[-1, marginal_ray_idx, 2] + dz

    return zf


def get_ray_oa_intersects(rays: np.ndarray):
    """
    Given a 'rays' array, find the number of optical axis intersections.

    return np.ndarray crossings: the surface idx before the intersection.
    """
    ray_to_use = [
        i for i in range(rays.shape[1]) if not np.isnan(rays[:, i, 0]).any()
        ]

    if not ray_to_use:
        raise ValueError("No valid rays found.")

    ray_idx = max(ray_to_use, key=lambda idx: abs(rays[0, idx, 0]))
    crossings = np.where(np.diff(np.sign(rays[:, ray_idx, 0])))[0]
    return crossings


#%% wavefront analysis


def get_ray_wf(rays: np.ndarray,
               pupil_radius: float = None,
               method: str = "opld"):
    # Check for shape of rays
    if rays.ndim == 2:
        rays = np.expand_dims(rays, axis=0)

    # Check for given pupil radius, if not use maximum radius of rays
    if pupil_radius is None:
        pupil_radius = np.nanmax(np.abs(rays[-1, :, 0]))

    # Drop any nan rays
    nan_mask = np.logical_not(np.isnan(rays[-1, :, 0]))

    # Normalized radius and enforce positive
    rho, keep_idx = np.unique(
        np.abs(rays[-1, :, 0][nan_mask]/pupil_radius),
                              return_index=True)
    opl = np.take(rays[-1, :, 3][nan_mask], keep_idx)

    # sort using rho
    order = np.argsort(rho)
    rho = rho[order]
    opl = opl[order]

    if method == "opld":
        wf = opl - opl[0]
    elif method == "opl":
        wf = opl

    return rho, wf


def ray_opl_polynomial(rays: np.ndarray,
                       pupil_radius: float = None,
                       method: str = "opld"):
    """
    Fit OPL from ray tracing results.

    :param array rays: Rays like array
    :param float pupil_radius: Radius normalization factor, pupil radius for optical systems
    :param str method: fit the "opld" (-paraxial) or "opl"
    :param boolean DEBUG: Optionally plot all the fits against the wf data
    :return dict fit_results: Fit results labelled by their coeff order.
    """
    rho, wf = get_ray_wf(rays, pupil_radius, method)

    # Fit wavefront using lstsq
    A = np.stack((np.ones(rho.size),
                  rho, rho**2, rho**3, rho**4,
                  rho**5, rho**6, rho**7, rho**8), axis=1)
    B = wf
    fit_params, res, rank, s = lstsq(A, B)

    # filter out numerical zeros
    for ii, fit in enumerate(fit_params):
        if np.abs(fit) <= 1e-12:
            fit_params[ii] = 0

    return fit_params


def opl_polynomial(r: np.ndarray,
                   coefficients):
    """
    Returns the wavefront polynomial given first 6 coefficients.

    :param float r: Array of radii
    :param float coefficients: List of polynomial fit coefficients
    :return float wf: OPL calculated from radii and coefficients
    """
    wf = 0
    for n, c in enumerate(coefficients):
        wf += r**n * c

    return wf


def ray_opl_strehl(pupil_rays: np.ndarray,
                   wl: float,
                   pupil_radius: float,
                   dist_to_focus=None):
    """
    Calculate the Strehl ratio by integrating over aberrations in pupil.
    *** assumes the reference sphere opl has already
        been removed using psuedo pupil lens.
    ****
    The wavefront aberration is the


    Born & Wolf pg463

    :param array pupil_rays: Rays in the pupil plane of pupil_obj
    :param float wl: wavelength
    :param float pupil_radius: lens pupil radius in mm.
    :return float strehl: Numerical Strehl ratio
    """
    ko = 2*np.pi / wl
    rho, wf = get_ray_wf(pupil_rays, pupil_radius, "opld")

    if dist_to_focus:
        # Subtract the Reference sphere for focusing rays
        def defocused_opl(r, dz, ri):
            return  ri*(np.sqrt(r**2 + dz**2) - dz)

        dopl = defocused_opl(rho, dz=dist_to_focus, ri=1.0)
        wf_aberration = wf - dopl

    else:
        # Fit opl to subtract piston C0 term.
        wf_fit = ray_opl_polynomial(pupil_rays, pupil_radius, "opld")
        wf_aberration = wf - wf_fit[0]

    # interpolate wavefront aberration
    wf_aberration_interp = interp1d(rho,
                                    wf_aberration,
                                    kind="linear",
                                    bounds_error=False,
                                    fill_value=0
                                    )

    def strehl_integrand(rho, ko):
        return rho * np.exp(1j * ko * (wf_aberration_interp(rho)))

    def real_strehl(x, ko):
        return np.real(strehl_integrand(x, ko))

    def imag_strehl(x, ko):
        return np.imag(strehl_integrand(x, ko))

    try:
        # integrate real and imag parts seperately
        real_integral = quad(real_strehl,
                             a=0, b=1,
                             args=ko,
                             limit=2000,
                             full_output=1
                             )
        imag_integral = quad(imag_strehl,
                             a=0,
                             b=1,
                             args=ko,
                             limit=2000,
                             full_output=1
                             )
        strehl = 4*np.abs(real_integral[0] + 1j*imag_integral[0])**2
    except:
        strehl = np.nan

    return strehl


def ray_opl_rms(pupil_rays: np.ndarray,
                pupil_radius: float,
                wl: float = 0.000561,
                units: str = ""):
    """
    Calculate the wavefront RMS from 1d ray tracing slice
    """
    rho, wf = get_ray_wf(pupil_rays, pupil_radius, "opld")
    wf_fit = ray_opl_polynomial(pupil_rays, pupil_radius, "opld")
    wf_aberration = wf - wf_fit[0]

    if units == "wl":
        wf_aberration = wf_aberration / wl

    wf_aberration_mean = wf_aberration.mean()
    wf_aberration_interp = interp1d(rho,
                                    wf_aberration,
                                    kind="linear",
                                    bounds_error=False, fill_value="extrapolate")

    # Define integrand for quad()
    def rms_integral(rho):
        return rho * (wf_aberration_interp(rho) - wf_aberration_mean)**2

    try:
        mean_square_error = quad(rms_integral,
                                 a=0,
                                 b=1,
                                 limit=2000,
                                 full_output=1)
        rms = np.sqrt(2*mean_square_error[0])
    except:
        print("RMS Failed!")
        rms = np.nan

    return rms


def ot_opl_rms(optical_train: list,
               rays: np.ndarray):
    """
    Calculate the RMS at the ray tracing "midpoint" focal plane.
    """

    # raytrace through lenses
    for lens in optical_train:
        rays = lens.raytrace(rays)

    # find the focal plane using ray tracing paraxial and marginal ray focii.
    fp_midpoint = ray_focal_plane(rays=rays,
                                       ri=lens.ri_out,
                                       method="midpoint")

    rays = intersect_plane(rays=rays,
                           zf = fp_midpoint,
                           ri_in=lens.ri_out,
                           refract=False)

    pupil_radius = np.nanmax(np.abs(rays[-1, :, 0]))
    if pupil_radius==0:
        pupil_radius=1e-10

    focal_plane_rms = ray_opl_rms(pupil_rays=rays,
                                  pupil_radius=pupil_radius,
                                  units="mm")

    return focal_plane_rms


def ray_opl_analysis(pupil_rays: np.ndarray,
                     pupil_radius: float,
                     fit_method: str = "opld",
                     units: str = "",
                     wl: float = 0.0005,
                     dist_to_focus = None):
    """
    Return wavefront polynomial fit, calculate Strehl ratio and RMS over normalized radial grid.
    Helper function for calling wavefront analysis tools

    :param array pupil_rays: Rays in the pupil plane of pupil_obj
    :param float pupil_radius: factor to normalize radius, use 1 if none. (mm)
    :param str fit_method: Whether to fit the OPL or OPL difference.
    :param str units: Optionally pass "wl" to normalize RMS with WL
    :param float wl: wavelength
    """
    wf_fit = ray_opl_polynomial(rays=pupil_rays,
                                pupil_radius=pupil_radius,
                                method=fit_method)
    strehl = ray_opl_strehl(pupil_rays=pupil_rays,
                            pupil_radius=pupil_radius,
                            dist_to_focus=dist_to_focus,
                            wl=wl)
    rms = ray_opl_rms(pupil_rays=pupil_rays,
                      pupil_radius=pupil_radius,
                      units=units,
                      wl=wl)
    results_dict = {"wavefront fit":wf_fit,
                    "strehl":strehl,
                    "rms":rms}

    return results_dict


def ray_opl_strehl_with_amp(pupil_rays,
                            ko,
                            pupil_radius,
                            binning="doane",
                            DEBUG=False):
    """
    TODO: WIP
    TODO: Update conversion to field amplitude

    Calculate the Strehl ratio by integrating over aberrations in pupil.
    Born & Wolf pg463.


    :param array pupil_rays: Rays in the pupil plane of pupil_obj
    :param float pupil_radius: lens pupil radius in mm.
    :param float k: wavenumber, 2pi/wl
    """
    rho, wf = get_ray_wf(pupil_rays, pupil_radius, "opld")
    wf_fit = ray_opl_polynomial(pupil_rays, pupil_radius, "opld")
    wf_aberration = wf - wf_fit[0]

    # Get amplitude interpolation
    # Compute ray density using binning, doane should have uniform bin widths
    hist, bin_edges = np.histogram(rho,
                                   bins="doane",
                                   density=True,
                                   range=(0,1))
    bin_centers = bin_edges - (bin_edges[1] - bin_edges[0])/2

    # interpolate field amplitude
    amp_interp = interp1d(bin_centers[:-1],
                          hist,
                          kind="linear",
                          bounds_error=False,
                          fill_value=0)

    # interpolate wf aberration
    wf_aberration_interp = interp1d(rho,
                                    wf_aberration,
                                    kind="linear",
                                    bounds_error=False,
                                    fill_value=0)

    def analytic_pupil_amplitude(r):
        """
        :param float rho: normalized radius
        """
        sigma = 1/4
        amplitude = amp_interp(0)*np.exp(-r**2/sigma**2)
        return amplitude

    def strehl_integrand(r, ko):
        return r * ((amp_interp(r)/analytic_pupil_amplitude(r))
                    * np.exp(1j*ko*(wf_aberration_interp(r)))
                    )

    def real_strehl(x, ko):
        return np.real(strehl_integrand(x, ko))

    def imag_strehl(x, ko):
        return np.imag(strehl_integrand(x, ko))

    try:
        # integrate real and imag parts seperately
        real_integral = quad(real_strehl,
                             a=0,
                             b=1,
                             args=ko,
                             limit=2000,
                             full_output=1)
        imag_integral = quad(imag_strehl,
                             a=0,
                             b=1,
                             args=ko,
                             limit=2000,
                             full_output=1)
    except:
        strehl = np.nan
    else:
        strehl = 4*np.abs(real_integral[0] + 1j*imag_integral[0])**2

    return strehl


#%% Rays to field
def rays_to_field(mask_radius: np.ndarray,
                  rays: np.ndarray,
                  ko: float,
                  amp_binning = 1000,
                  amp_type: str = "flux",
                  phase_type: str = "opld",
                  power: float = 1.0,
                  results:str = "field",
                  title: str = "",
                  plot_field: bool = False,
                  save_path: Path = None,
                  showfig: bool = False) -> np.ndarray:
    """
    Create complex field array using the last ray tracing plane.
    Combination of rays_to_amp and create_phase_mask

    amp_type:
    - "pdf" -> np.histogram probability density, sum(hist*bins)=1
    - "power" -> Normalize field to input power, default P=1

    :param array mask_radius: Field array radius using meshgrid.
    :param array rays: Ray like array.
    :param float ko: Wave vector magnitude.
    :param str binning: Amplitude histogram bin arguement, use "doane" for gaussian or n_xy for generality
    :param str amp_type: Choose amplitude normalization, "pdf" or "flux"
    :param str phase_type: choose OPL or OPLD
    :param float power: Normalize field to carry total power
    :param str results: Specify whether to return the total "field", "amplitude" or "phase"
    :return array field: Complex electric field
    """
    # rays must have ndim=3, check for single stack of rays to expand dims.
    if rays.ndim == 2:
        rays = np.expand_dims(rays, axis=0)

    # Drop any nan rays
    nan_mask = np.logical_not(np.isnan(rays[-1, :, 0]))
    last_rays = rays[-1, :, :][nan_mask]

    # Only keep the unique radii to avoid interpolation errors
    radius, keep_idx = np.unique(last_rays[:, 0], return_index=True)

    # Generate optical phase distribution
    if results=="field" or results=="phase":
        if phase_type=="opl":
            phase_interp = interp1d(last_rays[:, 0][keep_idx],
                                    last_rays[:, 3][keep_idx],
                                    kind="linear",
                                    bounds_error=False,
                                    fill_value=0)
        elif phase_type=="opld":
            # Difference with respect to mean path travelled.
            phase_interp = interp1d(last_rays[:, 0][keep_idx],
                                    last_rays[:, 3][keep_idx] - np.nanmean(last_rays[:, 3]),
                                    kind="linear",
                                    bounds_error=False,
                                    fill_value=0
                                    )
        phase = phase_interp(mask_radius) * ko

    # Generate the electric field amplitude
    if results=="field" or results=="amplitude":
        # Define the angular interpolation function to apply to the amplitude
        angle_interp = interp1d(last_rays[:, 0][keep_idx],
                                last_rays[:, 1][keep_idx],
                                kind="linear",
                                bounds_error=False,
                                fill_value=0)
        angles = angle_interp(mask_radius)

        if amp_type=="pdf":
            density=True
        elif amp_type=="flux":
            density=False

        # Use binning to calculate ray flux
        ray_density, bin_edges = np.histogram(radius,
                                              bins=amp_binning,
                                              range=(radius.min(),radius.max()),
                                              density=density)

        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

        # Interpolate field amplitude
        flux_interp = interp1d(bin_centers,
                               ray_density,
                               kind="linear",
                               bounds_error=False,
                               fill_value=0)

        amp = np.sqrt(flux_interp(mask_radius) / np.cos(angles))

    if results=="field":
        # Calculate field
        field = amp * np.exp(1j * phase)

    elif results=="amplitude":
        field = amp

    # Optional, normalize to given power
    if amp_type=="power" and (results=="field" or results=="amplitude"):
        n_grid = mask_radius.shape[0]
        dx = mask_radius[int(n_grid//2),int(n_grid//2 + 1)] - mask_radius[int(n_grid//2), int(n_grid//2)]
        field = pt.normalize_field(field, power, dx)

    if plot_field:
        # Define custom colormap symmetric about black
        cdict = {'red':[(0.0, 0.0, 0.0),
                        (0.5, 0.0, 0.0),
                        (1.0, 1.0, 1.0)],
                 'green':[(0.0, 0.0, 0.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 0.0, 0.0)],
                 'blue':  [(0.0, 1.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 0.0, 0.0)]
                 }

        black_centered_cmap = LinearSegmentedColormap('BlackCentered',
                                                      segmentdata=cdict)

        # Plotting
        fig = plt.figure(figsize=(8.0, 5.0))
        fig.suptitle(title, fontsize=16)
        grid = fig.add_gridspec(nrows=2,
                                ncols=5,
                                width_ratios=[0.65, 0.65, 0.001, 1, 0.075],
                                height_ratios=[1, 1],
                                wspace=0.35,
                                hspace=0.40)
        label_size = 12
        ticklbl_size = 10
        title_size = 12
        label_pad = 2

        # Plot amplitude histogram
        ax = fig.add_subplot(grid[0, 0])
        ax.set_title("$\phi(r)$", fontsize=title_size)
        ax.set_ylabel(r"# of rays", fontsize=label_size, labelpad=label_pad)
        ax.plot(bin_centers, ray_density, ".m", ms=1, c="r")
        ax.tick_params(axis="both", pad=5,
                       labelsize=ticklbl_size, labelbottom=False)

        # Plot amplitude
        ax = fig.add_subplot(grid[0, 1])
        ax.set_title("$\phi_{int}(r)$", fontsize=title_size)
        ax.plot(radius, flux_interp(radius), ".m", ms=1)
        ax.tick_params(axis="both", labelsize=ticklbl_size,
                       labelbottom=False, labelleft=False)

        # Plot field intensity
        extent_xy = [-mask_radius[0,0],
                     mask_radius[0,0],
                     -mask_radius[0,0],
                     mask_radius[0,0]]

        ax_i = fig.add_subplot(grid[0, 3])
        ax_i.set_title("$I(r)$", fontsize=title_size)
        ax_i.set_ylabel(r"y (mm)", fontsize=label_size, labelpad=label_pad)
        ax_i.yaxis.set_major_locator(MaxNLocator(3))
        ax_i.xaxis.set_major_locator(MaxNLocator(3))
        ax_i.set_xticks([-0.2, 0.0, 0.2])
        ax_i.tick_params(axis="both", labelsize=label_size)
        im = ax_i.imshow(np.abs(field)**2/np.max(np.abs(field)**2),
                         cmap="hot",
                         vmin=0,
                         origin="lower",
                         extent=extent_xy,
                         interpolation=None)

        # Cbar axes
        cax = fig.add_subplot(grid[0, 4])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.set_ylabel("A.U.", rotation="horizontal", labelpad=17, fontsize=12)

        # Plot opl
        ax = fig.add_subplot(grid[1, 0])
        t_str = "$\Delta OPL(r)$"
        ax.set_title(t_str, fontsize=title_size)
        ax.set_ylabel(r"OPL (mm)",fontsize=label_size, labelpad=label_pad)
        ax.set_xlabel(r"Radius (mm)", fontsize=label_size, labelpad=label_pad)
        ax.tick_params(axis="both", labelsize=ticklbl_size)
        ax.plot(last_rays[:, 0][keep_idx],
                last_rays[:, 3][keep_idx] - np.nanmean(last_rays[:, 3][keep_idx]),
                "-m")

        # Plot phase
        ax = fig.add_subplot(grid[1, 1])
        t_str = "$\Delta OPL_{int}(r)$"
        ax.set_title(t_str, fontsize=title_size)
        # ax.set_ylabel(f"{t_str} (mm)",fontsize=label_size)
        ax.set_xlabel(r"Radius (mm)", fontsize=label_size, labelpad=label_pad)
        ax.plot(radius[::5], phase_interp(radius[::5]), ".m", ms=1)
        ax.tick_params(axis="both",
                       labelsize=ticklbl_size, labelleft=False)

        # Plot 2d phase
        abs_max = np.max(np.abs(phase))
        ax = fig.add_subplot(grid[1, 3], sharex=ax_i, sharey=ax_i)
        ax.set_title("$\Phi(r)$", fontsize=title_size)
        ax.set_ylabel(r"y (mm)", fontsize=label_size, labelpad=label_pad)
        ax.set_xlabel(r"x (mm)", fontsize=label_size, labelpad=label_pad)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="both", labelsize=label_size)

        im = ax.imshow(phase,
                       extent=extent_xy,
                       cmap=black_centered_cmap,
                       vmin=-abs_max,
                       vmax=abs_max,
                       origin="lower",
                       aspect="equal",
                       interpolation=None)

        # Cbar axes
        cax = fig.add_subplot(grid[1, 4])
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel("A.U.", rotation="horizontal",
                           labelpad=15, fontsize=12)

        if save_path:
            plt.savefig(save_path)

        if showfig:
            plt.show()
        else:
            plt.close("all")

    if results=="field":
        return field
    elif results=="phase":
        return phase
    elif results=="amplitude":
        return amp


def raytrace_to_field(results: dict,
                      grid_params: list,
                      wl: float,
                      rays_to_field_z: float = None,
                      grid_padding: float = 0.100,
                      amp_binning = 1000,
                      amp_type: str = "flux",
                      focal_plane: str = "midpoint",
                      power: float = 1,
                      plot_rays_to_field: bool = False,
                      plot_raytrace_results: bool = False,
                      label: str = "",
                      savedir: Path = None,
                      showfig: bool = False):
    """
    Helper function for creating initial fields using Dask from
    raytrace_ot results. Only requires x and radius_xy parameters from grid_params.
    Returns both the 2d electric field but also the rays to field plane, useful for offsetting in propagation.

    Optionally can plot the raytracing results up to the rays-to-field plane.

    """
    # grab the propagation media RI from results
    prop_ri = results["optical_train"][-1].ri_out

    # grab the maximum extent of field grid
    x, radius_xy = grid_params[0:2]

    # ray trace to propagation plane
    rays = np.copy(results["rays"])

    if rays_to_field_z == None:
        rays_to_field_z = rays_to_field_plane(rays=rays,
                                              x_max=x.max(),
                                              padding=grid_padding)
    # calculate the field plane distance to the focus
    if focal_plane=="midpoint":
        focal_plane = "midpoint_focal_plane"
    elif focal_plane=="paraxial":
        focal_plane=="paraxial_focal_plane"
    elif focal_plane=="marginal":
        focal_plane=="marginal_focal_plane"
    dist_to_focus = rays_to_field_z - results[focal_plane]

    rays = intersect_plane(rays, rays_to_field_z, ri_in=prop_ri, refract=False)

    # Create initial field from rays
    initial_field = rays_to_field(mask_radius=radius_xy,
                                  rays=rays.copy(),
                                  ko=2 * np.pi / wl,
                                  amp_binning=amp_binning,
                                  amp_type=amp_type,
                                  phase_type="opld",
                                  power=power,
                                  results="field",
                                  plot_field=plot_rays_to_field,
                                  title=(f"Rays to Field, ",
                                         f"dz to focus:{dist_to_focus:.3f}"),
                                  save_path=savedir / Path(
                                      f"initial_field_{label}.png"),
                                  showfig=showfig)

    if plot_raytrace_results:
        plot_rays(rays=intersect_optical_axis(rays=rays, ri=prop_ri),
                  n_rays_to_plot=15,
                  optical_train=results["optical_train"],
                  planes_of_interest={"rays -> field":rays_to_field_z,
                                      "paraxial fp":results["paraxial_focal_plane"],
                                      "midpoint fp":results["midpoint_focal_plane"],
                                      "marginal fp":results["marginal_focal_plane"]},
                  show_focal_planes=True,
                  show_legend=True,
                  title=f"Raytracing {label:s}",
                  figsize=(40,10),
                  save_path= savedir / Path(f"ray_tracing_{label:s}.png"),
                  showfig=showfig)

    return [initial_field, rays_to_field_z]


#%% Matrix Raytracing
# ABCD matrices follow (r, n theta) formulation

def abcd_freespace(d: float,
                   n: float,
                   symbolic: bool = False):
    """
    :param float d: distance to propagate
    :param float n: Refractive index
    """
    if symbolic:
        return sy.Matrix([[1, d/n],
                          [0, 1]])
    else:
        return np.array([[1, d/n],
                         [0, 1]])


def abcd_thinlens(f: float,
                  symbolic: bool = False):
    """
    Applies action of thin lens of focal length f.

    :param float f: Thin lens focal length
    """
    if symbolic:
        return sy.Matrix([[1, 0],
                          [-1/f, 1]])
    else:
        return np.array([[1, 0],
                         [-1/f, 1]])


def abcd_refract_plano(symbolic: bool = False):
    """
    :param None:
    """
    if symbolic:
        return sy.Matrix([[1, 0],
                          [0, 1]])
    else:
        return np.array([[1, 0],
                         [0, 1]])


def abcd_refract_spher(R: float,
                       n1: float,
                       n2: float,
                       symbolic: bool = False):
    """
    R<0 Center of curvature to the left.

    :param float R: Spherical radius of curvature
    :param float n1: Left hand side refractive index
    :param float n2: Right hand side refractive index
    """
    if symbolic:
        return sy.Matrix([[1, 0],
                          [-(n2 - n1) / R, 1]])
    else:
        return np.array([[1, 0],
                         [-(n2 - n1) / R, 1]])


def abcd_ft(f: float,
            symbolic: bool = False):
    """
    Inverts position and angle by f

    :param float f: Fourrier transform focal length
    """
    if symbolic:
     return sy.Matrix([[0, f],
                       [1/f, 0]])

    else:
        return np.array([[0, f],
                         [1/f, 0]])


def abcd_relay(symbolic: bool = False):
    """
    Flips angle and radius

    :param None:
    """
    if symbolic:
        return sy.Matrix([[-1, 0],
                          [0, -1]])
    else:
        return np.array([[-1, 0],
                         [0, -1]])


def abcd_cardinal_points(optical_train: list,
                         n1: float,
                         n2:float,
                         DEBUG: bool = False):
    """
    Calculate principle plane and focal planes for a given optical train using ABCD matrices.

    ref: validated f1_pp & f2_pp analytically
    ref: https://www.montana.edu/ddickensheets/courses/eele482/handouts/abcdCardinal.pdf

    :params list optical_train: List of component class
    :params float n1: Left hand side refractive index
    :params float n2: Right hand side refractive index
    """
    a, b, c, d = optical_train.ravel()
    if DEBUG:
        print(a, b, c, d)

    # focal plane with respect to input/output surfaces
    f1 = -d / c
    f2 = -a / c

    # Principle planes
    h1 = (n1 - n2 * d) / c
    h2 = (1-a) / c

    # Measured from principle planes h1, h2
    f1_pp = -(n1 / n2) / c
    f2_pp = -1 / c

    return [f1, f2, h1, h2, f1_pp, f2_pp]


def abcd_focal_plane(optical_train: list,
                     return_df: bool = False):
    """
    Calculate the focal plane for given optical train

    :param list optical_train: List of lens like objects
    :param boolean return_df: Optionally, return the distance from last lens surface to focal plane
    """
    # Compile len"s abcd in list
    if len(optical_train)==1:
        ot_abcd = [optical_train[0].abcd]

    else:
        # reverse optical train order
        ot_abcd = [lens.abcd for lens in optical_train[::-1]]

    # Compute abcd matrix
    abcd = np.array([[1, 0],
                     [0, 1]])

    for m in ot_abcd:
        abcd = abcd.dot(m)

    # grab last vertex based on type of lens
    if( optical_train[-1].type=="Thick_lens"
       or optical_train[-1].type=="ETL"):
        vertex = optical_train[-1].z2
    elif optical_train[-1].type=="Doublet_lens":
        vertex = optical_train[-1].z3
    elif optical_train[-1].type=="Perfect_lens":
        vertex = optical_train[-1].z1

    df = -optical_train[-1].ri_out*abcd[0,0] / abcd[1,0]
    focal_plane = vertex + df

    if return_df:
        return [focal_plane, df]
    else:
        return focal_plane


#-----------------------------------------------------------------------------#
# %%Plotting functions

def plot_rays(rays: np.ndarray,
              n_rays_to_plot: int=31,
              optical_train = None,
              planes_of_interest: dict = None,
              title: str="Raytracing",
              show_focal_planes: bool = False,
              show_legend:bool = False,
              save_path: Path=None,
              showfig: bool=False,
              ax=None,
              figsize=(30,10)):
    """
    Plot arbitrary ray tracing.

    :param array rays: Rays like array
    :param int n_rays_to_plot: Number of rays to display in plot
    :param str title: Axes.title()
    :param Path savedir: Optional, choose to save fig by passing save path
    :param boolean showfig: Optional, choose whether to display figure by calling show()
    """

    if not n_rays_to_plot % 2 == 1:
        raise ValueError("n_rays_to_plot must be odd")

    # Choose rays to plot
    num_all_rays = rays[0,:,0].shape[0]
    inds_to_plot = (np.arange(0, n_rays_to_plot)*((num_all_rays)
                                                  // (n_rays_to_plot-1)))
    inds_to_plot = np.concatenate(([0], inds_to_plot))

    # ensure we have ray on optical axis if present
    is_on_axis = np.abs(rays[0, :, 0]) < 1e-12
    inds = np.arange(rays.shape[-2])
    if np.any(is_on_axis):
        inds_to_plot[0] = inds[is_on_axis][0]

    # enforce marginal rays
    inds_to_plot[-1] = inds[-1]

    # Plot ray tracing of OT and psuedo lens
    make_new_figure = ax is None

    # if no axes was provided, generate a new figure
    if make_new_figure:
        fig, ax = plt.subplots(1,1, figsize=(figsize))
        fig.suptitle("Raytracing Results")
        ax.set_aspect(1.0)

    # set ax params
    ax.set_title(title)
    ax.set_ylabel("radius (mm)")
    ax.set_xlabel("Optical axis (mm)")

    ax.plot(rays[:, inds_to_plot, 2],
            rays[:, inds_to_plot, 0],
            c="r",
            linewidth=1)

    # Plot optical train
    if optical_train:
        colors = plt.cm.cool(np.linspace(0, 1, len(optical_train)))

        for ii, comp in enumerate(optical_train):
            comp.draw(ax,
                      label=comp.type,
                      color="k", #colors[ii],
                      show_focal_planes=show_focal_planes)

        ax.set_ylim(bottom=-2*np.max([_c.aperture for _c in optical_train]),
                    top=2*np.max([_c.aperture for _c in optical_train]))

    if planes_of_interest is not None:
        colors = plt.cm.rainbow(np.linspace(0,
                                            1,
                                            len(planes_of_interest.keys()))
                                )

        for ii, (k, v) in enumerate(planes_of_interest.items()):
            ax.axvline(x=v, label=k, c=colors[ii])

    if show_legend:
        try:
            ax.legend()
        except Exception:
            pass
        else:
            pass

    if make_new_figure:
        if save_path:
            fig.savefig(save_path)

        if showfig:
            fig.show()
        else:
            plt.close(fig)


def plot_optical_train(optical_trains: list = [],
                       axes_titles: list[str] = [],
                       rays: list[np.ndarray] = [],
                       planes_of_interest: list[dict] = None,
                       n_rays_to_plot: int = 31,
                       show_focal_planes: bool = False,
                       fig_title: str = "Ray Tracing Results",
                       savedir = None,
                       showfig: bool = False):
    """
    Plot ray tracing results for given set of lenses and initial rays.
    The rays type should be consider for display, the number of rays and distribution.
    Lenses are passed in a list where each list holds the lens object and in/output refractive indeces.

    :param list optical_trains: Nested list containing optical train components
    :param str axes_title: axis.title() for each OT
    :param rays:
    :param planes_of_interest:
    :param n_rays_to_plot:
    :param str fig_title: fig.suptitle()
    :param Path savedir: Optional, choose to save fig by passing save path
    :param boolean showfig: Optional, choose whether to display figure by calling show()
    :returns figure: Figure with each raytraced optical train
    """


    ntrains = len(optical_trains)
    if planes_of_interest is None:
        planes_of_interest = [None for ii in range(ntrains)]

    # Create figure and grid
    fig = plt.figure(figsize=(20, 6.5*ntrains))
    fig.suptitle(fig_title)
    grid = fig.add_gridspec(nrows=ntrains,
                            ncols=1,
                            height_ratios=[1]*ntrains,
                            hspace=0.3,
                            wspace=0.3)

    for ii, ot in enumerate(optical_trains):
        ax = fig.add_subplot(grid[ii, 0])
        ax.set_title(axes_titles[ii])
        ax.set_ylabel("Height (mm)")

        plot_rays(rays[ii],
                  n_rays_to_plot,
                  optical_train=ot,
                  planes_of_interest=planes_of_interest[ii],
                  title=axes_titles[ii],
                  show_focal_planes=show_focal_planes,
                  savedir=None,
                  showfig=False,
                  ax=ax)

    # xlabel for last OT only
    ax.set_xlabel("radius (mm)")

    if savedir:
        plt.savefig(savedir)

    if showfig:
        plt.show()
    else: plt.close(fig)


    return fig


def plot_opld(rays: list = [],
              rays_labels: list  = [],
              axes_title: str = "Optical length difference",
              num_to_plot: float = 101,
              units="mm",
              save_path: Path = None,
              showfig: bool = False):
    """
    Plot the OPL accross beam

    :param list rays: List of rays type arrays
    :param str rays_labels: plot label for each OT
    :param str axes_title: ax.set_title()
    :param Path savedir: Optional, choose to save fig by passing save path
    :param boolean showfig: Optional, choose whether to display figure by calling show()
    :returns figure: Optical path length plots.
    """
    scale=1
    if units=="um":
        scale=1e3

    # Create figure to plot OPLd
    fig = plt.figure(figsize=(8,5))

    ax = fig.add_subplot(111)

    # Plot OPLD
    for (ray, label) in zip(rays, rays_labels):

        # Choose rays to plot
        num_all_rays = ray.shape[1]
        inds_to_plot = np.arange(0, num_to_plot)*(num_all_rays // num_to_plot)

        # Plot rays
        ax.plot(ray[0, inds_to_plot, 0],
                scale*(ray[-1, inds_to_plot, 3]-np.nanmean(ray[-1, :, 3])),
                ".",
                label=label)

    ax.set_title(axes_title)
    ax.set_ylabel(r"$d\Delta L$" + f"({units})" )
    ax.set_xlabel("Initial ray height (mm)")
    ax.legend()

    if save_path:
        plt.savefig(save_path)

    if showfig:
        fig.show()
    else: plt.close(fig)


def plot_ot_aberration(optical_trains: list,
                       fig_title: str="Aberrations",
                       rays_diameter: float=20.0,
                       save_path: Path=None,
                       showfig: bool=False):
    """
    Representation of spherical aberrations, plot ray optical axis intersect vs initial ray height.
    Does not account for RI mismatches between lens pupil and focal plane!

    :param list optical_trains: Nested list containing optical train components
    :param list ot_labels: Legend labels for each OT
    :param str axes_title: axis.title() for each OT
    :param float rays_diameter: Diameter of initial rays
    :param str aberration: Choose to display "transverse", "spherical"
    :param Path savedir: Optional, choose to save fig by passing save path
    :param boolean showfig: Optional, choose whether to display figure by calling show()
    :returns figure: Figure with each raytraced optical train
    """
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(fig_title)

    grid = fig.add_gridspec(nrows=1,
                            ncols=3,
                            width_ratios=[1,1,0.075],
                            wspace=0.3,
                            hspace=0.3)

    ax1 = fig.add_subplot(grid[0])
    ax1.set_title("Spherical Aberrations")
    ax1.set_xlabel("Initial ray height (mm)")
    ax1.set_ylabel("Optical axis intersect - mean() (mm)")

    ax2 = fig.add_subplot(grid[1])
    ax2.set_title("Transverse Aberrations")
    ax2.set_xlabel("Initial ray height (mm)")
    ax2.set_ylabel("Ray height in F.F.P. (mm)")

    cmaps = plt.cm.coolwarm(np.linspace(0, 1, len(optical_trains)))
    import matplotlib as mpl

    for ii, lenses in enumerate(optical_trains):
        # Create rays
        rays = create_rays(type="flat_top",
                           source="infinity",
                           n_rays=1001,
                           diameter=rays_diameter)

        # Raytrace OT
        for lens in lenses:
            rays = lens.raytrace(rays)

        focal_plane = ray_focal_plane(rays,
                                          ri=lens.ri_out,
                                          method="paraxial")

        # intersect O.A.
        rays_sph = intersect_optical_axis(rays.copy(), ri=lens.ri_out)

        sph_aber = rays_sph[-1,:,2] - np.nanmean(rays_sph[-1,:,2])

        rays_tran = intersect_plane(rays.copy(),
                                    zf=focal_plane,
                                    ri_in=lens.ri_out,
                                    refract=False)

        tran_aber = rays_tran[-1,:,0]
        ax1.plot(rays[0, :, 0], sph_aber, c=cmaps[ii])
        ax2.plot(rays[0, :, 0], tran_aber, c=cmaps[ii])

    cax = fig.add_subplot(grid[2])
    mpl.colorbar.ColorbarBase(cax,
                              cmap=mpl.cm.coolwarm,
                              orientation="vertical")
    if save_path:
        plt.savefig(save_path, dpi=150)

    if showfig:
        plt.show()
    else: plt.close(fig)


def plot_rays_to_field(rays: np.ndarray,
                       mask_radius: np.ndarray,
                       extent_xy: list,
                       ko: float,
                       binning: str="doane",
                       phase_type: str="opld",
                       amp_type: str="power",
                       power: float=1.0,
                       fig_title: str="Rays to field",
                       savedir: Path=None,
                       showfig: bool=False):
    """
    Plot the sequence from initial ray / final amplitude distributions, wave aberration, and phase angle.

    :param array rays: Rays like array
    :param array mask_radius: n_xy X n_xy array of field radius
    :param list extent_xy: imshow() extent for field plot
    :param float ko: Field wave number
    :param float ko: Wave vector magnitude.
    :param str binning: Amplitude histogram bin arguement, use "doane" for gaussian or n_xy for generality
    :param str amp_type: Choose amplitude normalization
    :param float scale: Required arg if amp_type=="scale"
    :param str fig_title: fig.suptitle()
    :param Path savedir: Optional, choose to save fig by passing save path
    :param boolean showfig: Optional, choose whether to display figure by calling show()
    """
    # rays must have ndim=3, check for single stack of rays to expand dims.
    if rays.ndim == 2:
        rays = np.expand_dims(rays, axis=0)

    # Drop any nan rays
    nan_mask = np.logical_not(np.isnan(rays[-1, :, 0]))
    last_rays = rays[-1, :, :][nan_mask]

    # Only keep the unique radii and opls to avoid interpolation errors
    radius, keep_idx = np.unique(last_rays[:, 0], return_index=True)

    # Calculate electric field phase
    if phase_type=="opl":
        phase_interp = interp1d(last_rays[:, 0][keep_idx],
                                last_rays[:, 3][keep_idx],
                                kind="linear",
                                bounds_error=False,
                                fill_value=0)
    elif phase_type=="opld":
        phase_interp = interp1d(last_rays[:, 0][keep_idx],
                                last_rays[:, 3][keep_idx]-last_rays[len(last_rays)//2, 3],
                                kind="linear",
                                bounds_error=False,
                                fill_value=0)

    phase = phase_interp(mask_radius)*ko

    # Calculate electric field amplitude
    # Use binning to calculate PDF, oversample range
    hist, bin_edges = np.histogram(radius,
                                   bins=binning,
                                   range=(-1.2*mask_radius.max(),
                                          1.2*mask_radius.max()),
                                   density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    # Interpolate field amplitude
    flux_interp = interp1d(bin_centers,
                          hist,
                          kind="linear",
                          bounds_error=False,
                          fill_value=0)

    amp = flux_interp(mask_radius)

    # Calculate field
    field = amp*np.exp(1j*phase)

    # Optional, normalize to given power
    if amp_type=="power":
        n_grid = mask_radius.shape[0]
        dx =( mask_radius[int(n_grid//2), int(n_grid//2 + 1)]
             - mask_radius[int(n_grid//2), int(n_grid//2)])
        field = pt.normalize_field(field, power, dx)

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    grid = fig.add_gridspec(nrows=3,
                            ncols=3,
                            width_ratios=[1, 1, 0.075],
                            height_ratios=[1, 0.05, 1],
                            wspace=0.1,
                            hspace=0.2)
    fig.suptitle(fig_title)

    # Plot amplitude histogram
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title("Radial PDF", fontsize=12)
    ax.set_ylabel(r"A.U.", fontsize=10)
    # ax.set_xlim(0, np.max(radius))
    # ax.plot(bin_centers, hist, ".m", ms=1)
    ax.hist(bin_centers, hist)

    # Plot field amplitude
    ax = fig.add_subplot(grid[0, 1])
    ax.set_title("Field Intensity", fontsize=12)
    ax.set_ylabel(r"y (mm)", fontsize=10)
    # ax.set_xlabel(r"x ($\mu m$)", fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    im = ax.imshow(np.abs(field)**2,
                   cmap="hot",
                   norm=PowerNorm(gamma=0.7),
                   extent=extent_xy,
                   origin="lower",
                   interpolation=None)

    # Cbar axes
    cax = fig.add_subplot(grid[0, 2])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("A.U.", rotation="horizontal", labelpad=18)

    # Plot wavefront
    ax = fig.add_subplot(grid[2, 0])
    ax.set_title("Field wavefront aberration", fontsize=12)
    ax.set_ylabel(r"$\Delta \phi$ $(mm)$",fontsize=10)
    ax.set_xlabel(r"Radius $(mm)$", fontsize=10)
    # ax.set_xlim(0, np.max(radius))
    ax.plot(last_rays[:, 0][keep_idx], last_rays[:, 3][keep_idx], "-m")

    # Plot phase angle
    ax = fig.add_subplot(grid[2, 1])
    ax.set_title("Field phase angle", fontsize=12)
    ax.set_ylabel(r"y ($mm$)", fontsize=10)
    ax.set_xlabel(r"x ($mm$)", fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    im = ax.imshow(np.angle(field),
                   cmap="hot",
                   vmin=-np.pi,
                   vmax=np.pi,
                   extent=extent_xy,
                   origin="lower",
                   aspect="equal",
                   interpolation=None)

    # Cbar axes
    cax = fig.add_subplot(grid[2, 2])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("A.U.", rotation="horizontal", labelpad=12)

    if savedir:
        plt.savefig(savedir, dpi=150)

    if showfig:
        plt.show()
    else: plt.close("all")

    return None


def plot_radial_distribution(rays: np.ndarray,
                             binning: str = "doane",
                             title: str = "Ray Distribution",
                             save_path: Path = None,
                             showfig: bool = False):
    """
    Plot the initial and final ray radial distributions.

    :param      rays:
    :param str or float binning: bin arguement of numpy.histogram
    """
    # Drop any nan rays
    nan_mask = np.logical_not(np.isnan(rays[-1, :, 0]))
    rays_f = rays[-1, :, :][nan_mask]
    rays_i = rays[0, :, :][nan_mask]
    radius_i = rays_i[:, 0]
    radius_f = rays_f[:, 0]

    hist_i, bins_i = np.histogram(radius_i, bins=binning, density=False)
    hist_f, bins_f = np.histogram(radius_f, bins=int((radius_f.max()/radius_i.max())*binning), density=False)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    ax.set_title("Initial ray distribution", fontsize=15)
    ax.set_xlabel("Radius off OA, mm", fontsize=14)
    ax.set_ylabel("PDF", fontsize=14)

    ax.plot(bins_i[:-1], hist_i, ".", c="r", label="Initial Rays")
    ax.plot(bins_f[:-1], hist_f, ".", c="b", label="Final Rays")
    ax.legend(loc=0, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150)

    if showfig:
        plt.show()
    else: plt.close(fig)


def plot_fit_summary(fit: np.ndarray,
                     axes_title: str="Fit Results",
                     wl: float=None,
                     save_path: Path=None,
                     showfig: bool=False):
    """
    plot fit results in a summary
    TODO: add option to display zernike terms instead of polynomial coeff.

    :param list fit: List of fit coefficients
    :param str axes_title: ax.set_title()
    :param float wl: Optional, include wavelength to scale fit coeffients by the wavelength
    :param Path savedir: Optional, choose to save fig by passing save path
    :param boolean showfig: Optional, choose whether to display figure by calling show()
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
    fig.suptitle("Fit Summary")
    fit_labels = [f"$C_{ii}$" for ii in range(len(fit))]

    if wl:
        scaled_fit = fit / wl
        ax.bar((np.arange(len(fit))), scaled_fit, tick_label=fit_labels)
        ax.set_ylabel(r"$\frac{|C_i|}{\lambda}$",
                      labelpad=30,
                      rotation="horizontal")
        ax.set_xlabel("Fit Coeff", labelpad=10, rotation="horizontal")

    else:
        ax.set_ylabel(r"$|C_i|$")
        ax.bar((np.arange(len(fit))), fit, tick_label=fit_labels)
        ax.set_ylabel(r"$|C_i]$", labelpad=10, rotation="horizontal")
        ax.set_xlabel("Fit Coeff", labelpad=10, rotation="horizontal")

    ax.tick_params("both")
    ax.set_title(axes_title)
    if fit[0]/10 > np.sum(fit[1:]):
        ax.set_yscale("log")
    if save_path:
        plt.savefig(save_path, dpi=150)

    if showfig:
        fig.show()
    else: fig.close()


#-----------------------------------------------------------------------------#
#%% misc. helpful functions

def get_unique_dir(parent_dir: Path,
                   dir_name: str):
    """
    Create unique directory using timestamp and directory name

    :param Path dir_name: directory name
    :return str savedir: Unique directory
    """
    # Save results to unique directory
    now = datetime.datetime.now()
    time_stamp = "%04d%02d%02d_%02d%02d%02d" % \
                    (now.year,
                     now.month,
                     now.day,
                     now.hour,
                     now.minute,
                     now.second)

    # Todo: fix
    new_dir = Path("%s_%s" % (time_stamp, dir_name))

    savedir = parent_dir/new_dir

    savedir.mkdir(exist_ok=True)

    return  savedir