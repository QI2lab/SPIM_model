'''
Test perfect lens.
1. Focus collimated rays, check strehl ratio, wavefront fit and axial spread of rays.
2. Propagate field and compare to theoretical beam parameters

Steven Sheppard
05/20/2022
'''
# model imports
import model_tools.raytrace as rt
import model_tools.propagation as pt

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

run_perfect = True
run_rimismatch = True
#------------------------------------------------------------------------------#
# model setup
#------------------------------------------------------------------------------#
root_dir = Path('/mnt/tilingspim/extFOV_results/debugging')
save_dir = Path('mismatched_objective_debugging')
savedir = rt.get_unique_dir(root_dir, save_dir)

showfig=True

# Unit conversion: mm to um /, um to mm *
mm = 1e-3

# Field parameters
beam_power = 1
wl = 0.5 * mm
ko = 2 * np.pi / wl
initial_propagation_diameter = 500 * mm
propagation_length = 100 * mm
dx = wl / 2
dz = 1 * mm
n_xy = int(initial_propagation_diameter // dx)
n_zx = int(propagation_length // dz)

# Check if n_grid is odd
if n_xy%2 == 0:
    n_xy += 1
if n_zx%2 == 0:
    n_zx += 1

# number of rays
n_rays = int(1e6)

# Create perfect objective
obj_z = 1
obj_f = 30
obj_na = 0.4
obj = rt.Perfect_lens(z1=obj_z, f=obj_f, na=obj_na, ri_in=1.0, ri_out=1.0)

# Create RI mismatch using coverslip and PROTOs for imaging media
coverslip_dz = obj.f / 2
cs_thickness = 1.0
ri_protos = 1.47
ri_glass = 1.556
coverslip = rt.Thick_lens(z1=(obj.ffp - coverslip_dz),
                          ri=ri_glass,
                          r1=np.inf,
                          r2=np.inf,
                          t=cs_thickness,
                          aperture_radius=8,
                          ri_in=1.0,
                          ri_out=ri_protos)

#------------------------------------------------------------------------------#
# PSF without index mismatch
if run_perfect:
    n_rays =  n_rays
    rays = rt.create_rays(type='gaussian', source='infinity', n_rays=n_rays, diameter=obj.pupil_radius, sampling_limit=5)

    # raytrace
    rays = obj.raytrace(rays)

    # Calculate focal plane
    focal_plane = rt.ray_focal_plane(rays, ri=obj.ri_out, method='paraxial')

    # Create a psuede perfect lens to obtain the bfp of our new wf
    pupil_obj =  rt.Perfect_lens(z1=focal_plane + obj.f, f=focal_plane - obj.z1, na=obj.na, ri_in=obj.ri_out, ri_out=obj.ri_out)

    pupil_rays = pupil_obj.raytrace(rays, final_plane='ffp')
    ray_opl_analysis = rt.ray_opl_analysis(pupil_rays, pupil_obj.pupil_radius, wl)
    strehl = ray_opl_analysis['strehl ratio']
    fit = ray_opl_analysis['wavefront fit']

    print(f'wavefront polynomial fit: {fit} \nStrehl ratio: {strehl}')

    # Create field grid
    field, grid_params = pt.field_grid(num_xy=n_xy, num_zx=n_zx, dx=dx, dz=dz)
    x, radius_xy, extent_xy, z, extent_zx = grid_params

    # raytrace to field plane
    rays_to_field_z = rt.rays_to_field_plane(rays, np.max(x))
    rays = rt.intersect_plane(rays, rays_to_field_z, ri_in=1.0, refract=False)

    # rays to field
    field_i = rt.rays_to_field(mask_radius=radius_xy,
                               rays=rays,
                               ko=ko,
                               binning='doane',
                               amp_type='power',
                               power=1)

    # offest field's z coords to lab frame best focus
    focal_plane = rt.ray_focal_plane(rays, ri=obj.ri_out, method='midpoint')
    z_prop = z - rays_to_field_z + focal_plane

    field = pt.get_3d_field(initial_field=field_i, z=z_prop, wl=wl, dx=dx, ri=1.0)

    pt.plot_xz_projection(fields=[field],
                          grid_params=[grid_params],
                          field_labels=['Perfect lens'],
                          fig_title='Perfect lens focus',
                          save_path=savedir,
                          showfig=showfig)

#------------------------------------------------------------------------------#
# PSF for mismatched objective model
if run_rimismatch:
    n_rays =  n_rays
    rays = rt.create_rays(type='gaussian', source='infinity', n_rays=n_rays, diameter=obj.pupil_radius, sampling_limit=5)

    # raytrace
    rays = obj.raytrace(rays, final_plane='pp')
    rays = coverslip.raytrace(rays)

    # Calculate focal plane
    focal_plane = rt.ray_focal_plane(rays, ri=coverslip.ri_out, method='midpoint')

    # Create a psuede perfect lens to obtain the bfp of our new wf
    pupil_obj =  rt.Perfect_lens(z1=focal_plane + obj.f,
                                 f=focal_plane - obj.z1,
                                 na=obj.na,
                                 ri_in=coverslip.ri_out,
                                 ri_out=coverslip.ri_out)

    pupil_rays = pupil_obj.raytrace(rays, final_plane='ffp')

    fit = rt.ray_opl_polynomial(pupil_rays, pupil_obj.pupil_radius)
    strehl = rt.ray_opl_strehl(pupil_rays, wl, pupil_obj.pupil_radius)
    print(f'wavefront polynomial fit: {fit} \nStrehl ratio: {strehl}')

    # Create field grid
    field, grid_params = pt.field_grid(num_xy=n_xy, num_zx=n_zx, dx=dx, dz=dz)
    x, radius_xy, extent_xy, z, extent_zx = grid_params

    # raytrace to field plane
    rays_to_field_z = rt.rays_to_field_plane(rays, np.max(x))
    rays = rt.intersect_plane(rays,
                              rays_to_field_z,
                              ri_in=coverslip.ri_out,
                              refract=False)

    # rays to field
    field_i = rt.rays_to_field(mask_radius=radius_xy,
                               rays=rays,
                               ko=ko,
                               binning='doane',
                               amp_type='power',
                               power=1)

    # offest field's z coords to lab frame best focus
    z_prop = z  + (focal_plane - rays_to_field_z)

    field = pt.get_3d_field(initial_field=field_i,
                            z=z_prop,
                            wl=wl,
                            dx=dx,
                            ri=coverslip.ri_out)


    extent_real = extent_zx.copy()
    extent_real[0] = z_prop[0]+ rays_to_field_z
    extent_real[1] = z_prop[-1]+ rays_to_field_z
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    im = ax.imshow(pt.xz_projection(field),
                    extent=extent_real)

    ax.axvline(x=z_prop[n_zx//2] + rays_to_field_z)


    pt.plot_xz_projection(fields=[field],
                          grid_params=[grid_params],
                          field_labels=['Perfect lens'],
                          fig_title='Perfect lens focus',
                          save_path=savedir,
                          showfig=showfig)
