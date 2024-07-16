# -*- coding: utf-8 -*-
'''
Perfect lens validation.

Validate perfect lens model by focusing a uniform flat top beam and Gaussian beam.
Change the number of field grid points to avoid memory errors.

units: mm

Steven Sheppard
04/05/2023
'''
import model_tools.propagation as pt
import model_tools.raytrace as rt
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

DEBUG = False

root_dir = Path('/home/steven/Documents/qi2lab/github/raytracing_sjs/data')
save_dir = Path('Doublet lens test')
savedir = rt.get_unique_dir(root_dir, save_dir)

# Unit conversion, um -> mm: * , mm -> um: /
mm = 1e-3

# Ray tracing density is given in the number of rays per unit mm of gaussian diameter (2*w(z))
ray_density = 10

# Field model parameters
wl = 0.5 * mm
ko = 2 * np.pi / wl

clear_aperture = 45.72

# Create thorlabs f150
ac508_150_a = rt.Doublet_lens(z1=155,
                              r1=83.2,
                              t1=12.0,
                              ri1=1.5214,
                              r2=-72.1,
                              t2=3.0,
                              ri2=1.6848,
                              r3=-247.7,
                              aperture_radius=(clear_aperture/2),
                              ri_in=1.0, ri_out=1.0)
ac508_150_a.label="ac508-150-A"

ac508_180_a = rt.Doublet_lens(z1=0,
                              r1=109.7,
                              t1=12.0,
                              ri1= 1.5180,
                              r2=-80.7,
                              t2=2.0,
                              ri2= 1.6757,
                              r3=-238.5,
                              aperture_radius=(clear_aperture/2),
                              ri_in=1.0, ri_out=1.0)
ac508_180_a.label="ac508-180-A"


# relay 2 is backwards with infinity towards the objective
ac508_100_a_b = rt.Doublet_lens(z1=0,
                              r1=363.1,
                              t1=4.0,
                              ri1=1.7320,
                              r2=44.2,
                              t2=16.0,
                              ri2=1.6721,
                              r3=-71.1,
                              aperture_radius=(clear_aperture/2),
                              ri_in=1.0, ri_out=1.0)
ac508_100_a_b.label="ac508-100-A bckwrd"

ac508_100_a = rt.Doublet_lens(z1=0,
                              r1=71.1,
                              t1=16.0,
                              ri1=1.6721,
                              r2=-44.2,
                              t2=4.0,
                              ri2=1.7320,
                              r3=-363.1,
                              aperture_radius=(clear_aperture/2),
                              ri_in=1.0, ri_out=1.0)
ac508_100_a.label="ac508-100-A"

lenses_to_plot = [ac508_180_a, ac508_150_a, ac508_100_a, ac508_100_a_b]
rays_to_plot = []
labels_to_plot = []
for lens_to_test in lenses_to_plot:
      if lens_to_test.label[-6:] != 'bckwrd':
            rays = rt.create_rays(type='flat_top', source='infinity', n_rays=2e5 + 1, diameter=lens_to_test.aperture*2.0)
      else:
            rays = rt.create_rays(type='flat_top', source='point', n_rays=1e5, na=np.sin(np.arctan(lens_to_test.aperture / lens_to_test.f1)))
            rays[-1, :, 2] = lens_to_test.bfp

      rays = rt.intersect_plane(rays=rays, zf=lens_to_test.bfp, ri_in=1.0, refract=False)
      rays = lens_to_test.raytrace(rays)
      rays_to_plot.append(rt.intersect_plane(rays=rays, zf=lens_to_test.ffp, ri_in=1.0))
      labels_to_plot.append(lens_to_test.label)

      focal_planes = rt.ray_focal_plane(rays=rays, ri=1.0, method='both')

      rays_spot_diag = rays=rt.intersect_plane(rays=rays, zf=lens_to_test.ffp, ri_in=1.0)

      # Plot histogram of
      fig, ax = plt.subplots(1,1, figsize=(10,10))
      ax.set_title('Histogram(radius)')
      ax.hist(rays_spot_diag[-1, :, 0], 101)
      ax.set_ylabel('count')
      ax.set_xlabel('ffp hieght (mm)')

      rt.plot_rays(rays=rt.intersect_plane(rays=rays, zf=lens_to_test.ffp, ri_in=1.0),
                  n_rays_to_plot=51,
                  title=lens_to_test.label,
                  optical_train=[lens_to_test],
                  planes_of_interest={"paraxial f.p.": focal_planes[0],
                                      "midpoint f.p.": focal_planes[1]},
                  show_focal_planes=True,
                  save_path=savedir / Path("testing_doublet_raytrace.png"),
                  showfig=True
                  )

      rms = rt.ray_opl_rms(rays_spot_diag,
                                 wl = 0.0005,
                                 pupil_radius = 1.0)

      # Plot spot diagram
      # Color coding for initial ray height
      height_cmap = plt.cm.rainbow(np.linspace(0, 1, rays.shape[1]))

      skp = 3
      fig, ax = plt.subplots(1,1, figsize=(10,10))
      ax.set_title(f"spot diagram, path length RMS = {rms*1e3:.3f}um")
      ax.plot(rays[0, :, 0], np.abs(rays[-1, :, 0]), 'r.')
      ax.set_xlabel("initial ray height (mm)")
      ax.set_ylabel("ray height F.F.P (mm)")

      plt.show()

# Use the final height for point source
rays_to_plot[-1][0, :, 0] = rays_to_plot[-1][-1, :, 0]

rt.plot_opld(rays=rays_to_plot,
             rays_labels=labels_to_plot,
             axes_title="Doublet path length difference",
             save_path=savedir / Path('doublet_path_length.png'),
             num_to_plot=101,
             showfig=True)
