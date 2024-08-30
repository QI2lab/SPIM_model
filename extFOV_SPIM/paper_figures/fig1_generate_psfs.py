
'''
Generate light sheet microscope PSF for multiple defocused light sheet positions

Used in figure 1 to show the results of considering a point source outside,
the light sheet focus.

! uses modelpsf package !
2024/08/29
'''
from psfmodels._core import tot_psf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import model_tools.raytrace as rt
from tqdm import tqdm

root_dir = Path(r"C:\Users\Steven\Documents\qi2lab\github\SPIM_model\data")
save_dir = rt.get_unique_dir(root_dir, "fig1_psfs")

air_lens = {
    'ni0': 1.0, # immersion medium RI design value
    'ni': 1.0,  # immersion medium RI experimental value
    'ns': 1.0,  # specimen refractive index
    'tg': 0, # microns, coverslip thickness
    'tg0': 0 # microns, coverslip thickness design value
}
ex_lens = {**air_lens, 'NA': 0.14}
ex_wvl = 0.488
em_lens = {**air_lens, 'NA': 0.5}
em_wvl = 0.525

dxy = 0.2
dz = dxy
nx = 251
nz = 501

x = dxy * np.arange(-(nx - 1)/2, (nx - 1)/2 + 1)
y = x
z = dz * np.arange(-(nz - 1)/2, (nz - 1)/2 + 1)
extent_xy = [x[0] - dxy/2, x[-1] + dxy/2, y[0] - dxy/2, y[-1] + dxy/2]
extent_z = [x[0] - dxy/2, x[-1] + dxy/2, z[0] - dz/2, z[-1] + dz/2]

func = 'scalar'
for offset in tqdm([0, 10, 25, 50, 75]):
    # The main function
    ex_psf, em_psf, total_psf = tot_psf(nx=nx, nz=nz, dxy=dxy, dz=dz, pz=0,
                                        x_offset=offset, z_offset=0,
                                        ex_wvl = ex_wvl, em_wvl = em_wvl,
                                        ex_params=ex_lens, em_params=em_lens,
                                        psf_func=func)

    fig, (a1,a2,a3) = plt.subplots(1,3, figsize=(15,7))
    a1.imshow(ex_psf, norm=PowerNorm(gamma=0.6), extent=extent_xy, cmap='hot')
    a2.imshow(em_psf[:, nx//2], norm=PowerNorm(gamma=0.6), extent=extent_z, cmap='hot')
    a3.imshow(total_psf[:, nx//2], norm=PowerNorm(gamma=0.6), extent=extent_z, cmap='hot')
    a2.set_ylim(-20, 20)
    a2.set_xlim(-20,20)
    a3.set_ylim(-10, 10)
    a3.set_xlim(-10,10)
    a1.set_title('excitation psf')
    a2.set_title('detection psf')
    a3.set_title('total psf')
    fig.savefig(save_dir / Path(f'psf1_{offset}.pdf'), dpi=300)
