#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 10:31:23 2025

@author: aeferretti
"""



from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
import matplotlib.colors as colors
from itertools import product

import chromatix.functional as cx
from chromatix.ops import init_plane_resample
from chromatix.field import crop, pad

import jax
import jax.numpy as jnp
from jax import Array
# jax.config.update('jax_enable_x64', True)  # needed for 1e-9, 1e-6 order-of-magnitude beta/delta?
PI = jnp.pi

import optax
import optax.tree
from optax import contrib

from jaxopt.projection import projection_non_negative
from flax import linen as nn
from flax.core import unfreeze, freeze

from xpc.psf import apply_psf
from xpc.transformations import rotate_volume
from xpc.xscatter import Material, get_wavenum, get_wavelen
%matplotlib qt5

tissue = Material('tissue', 'H(10.2)C(14.3)N(3.4)O(70.8)Na(0.2)P(0.3)S(0.3)Cl(0.2)K(0.3)', 1.06)
adipose = Material('adipose', 'H(11.4)C(59.8)N(0.7)O(27.8)Na(0.1)S(0.1)Cl(0.1)', 0.95) 
bone = Material('bone', 'H(3.4)C(15.5)N(4.2)O(43.5)Na(0.1)Mg(0.2)P(10.3)S(0.3)Ca(22.5)', 1.92)


def make_raw_phantom(N, p0=1.0, p1=0.45, p2=0.35, c1=1, c2=1, id0=1, id1=2, id2=3, DTYPE=np.float64):
    # Make a phantom with 3 different spheres
    # p and c are scaling factors, location and size factors 
    # the ids are the fill values: 0=cylinder=1, 1=sphere=2, 2=sphere=3
    assert (p1 <= p0/2) and (p2 <= p0/2) and (p1 > 0) and (p2 > 0)
    coords = np.linspace(-N/2, N/2, N)
    Z, Y, X = np.meshgrid(coords, coords, coords)
    r1 = p1*N/2
    r2 = p2*N/2
    x1, y1, z1 = -c1*r1/np.sqrt(2), -c1*r1/np.sqrt(2), -r1/2
    x2, y2, z2 = c2*r2/np.sqrt(2), c2*r2/np.sqrt(2), r2/2
    obj = np.zeros([N,N,N], dtype=DTYPE)
    obj[np.where(X**2 + Y**2 < (p0*N/2)**2)] = id0
    obj[np.where((X-x1)**2 + (Y-y1)**2 + (Z-z1)**2 < r1**2)] = id1
    obj[np.where((X-x2)**2 + (Y-y2)**2 + (Z-z2)**2 < r2**2)] = id2
    return obj
    




class ProjPBI(nn.Module): 

    # Simulation settings
    propdist = 10e-3
    energy = 10
    material_basis = {1:tissue, 2:bone}
             
    # Phantom
    phantom_Nx: int = 64#64
    phantom_Ny: int = 30#30
    phantom_dx: float = 0.25e-6
    phantom_fov = phantom_dx * phantom_Nx
    up_samp_fac: int = 2
    # Detector
    det_Nx: int = 64#64  # 32 -- TODO: should have det_N < phantom_N, but need to account for this in phantom init during recon!    
    det_Ny: int = 30#30  # 10
    det_fwhm: float = 1e-6
    det_psf: str = 'lorentzian'  # code for the PSF is in fun.py
    resampling_method: str = 'linear'
    I0: int = 1e8 # very low noise to start
    det_fov: float = phantom_fov
    det_dx: float = det_fov / det_Nx
    
    # Misc.
    wavelen = get_wavelen(energy)
    N_pad: int = 16   # note -- this is probably pushing the lower end of acceptable. Need to check?
    n_medium: float = 1
    cval = 1 + 0j

    def setup(self):
        
        self.volume = self.param(
            'volume',    # -- make this float64?
            lambda key, shape: jnp.stack((jnp.full((self.phantom_Nx, self.phantom_Ny, self.phantom_Nx), 1e-10),
                                          jnp.full((self.phantom_Nx, self.phantom_Ny, self.phantom_Nx), 1e-7)), axis=-1),  
            (self.phantom_Nx, self.phantom_Ny, self.phantom_Nx, 2),
        )

        # function to resample source field to detectory geometry
        self.det_resample_func = init_plane_resample(
            (self.det_Nx, self.det_Ny), 
            (self.det_dx, self.det_dx), 
            resampling_method=self.resampling_method
        )
    
    def __call__(self, angle: float) -> Array:

        up_samp_fac = self.up_samp_fac # do upsampling. Look to do a linear upsampling later
        volume = jnp.repeat(jnp.repeat(jnp.repeat(self.volume,up_samp_fac,axis=0),up_samp_fac,axis=1),up_samp_fac,axis=2)
        N_pad = self.N_pad 
        total_pad = N_pad*up_samp_fac
    
        # TODO (for AD recon)
        ## -- the initial phantom volume will match detector geometry
        ## -- then, upsample the volume from detector res to phantom res for accurate forward project.
        ## -- currently, this takes an already upsampled phantom (not compatible with good recon)
        
        # incident wave
        field = cx.plane_wave(
            shape = (self.phantom_Nx*up_samp_fac, self.phantom_Ny*up_samp_fac),
            dx = self.phantom_dx/up_samp_fac,
            spectrum = self.wavelen,
            spectral_density = 1
        ) 
        field = field / field.intensity.max()**0.5 / (self.phantom_Nx / self.det_Nx) / (self.phantom_Ny / self.det_Ny) 
        cval = self.cval
        field = pad(field, up_samp_fac*N_pad, cval=cval)
        
        # thru object
        rotated_vol = jax.vmap(
            rotate_volume,
            in_axes=(-1, None)
        )(volume, -1*angle)
        rotated_vol = jnp.swapaxes(jnp.stack([rotated_vol[0], rotated_vol[1]], axis=-1), 1, 2)

        beta_proj = jnp.pad(jnp.sum(rotated_vol[:,:,:,0],axis=1)[None,:,:,None,None],((0,0),(total_pad,total_pad),(total_pad,total_pad),(0,0),(0,0)),'constant', constant_values=0)
        dn_proj = jnp.pad(jnp.sum(rotated_vol[:,:,:,1],axis=1)[None,:,:,None,None],((0,0),(total_pad,total_pad),(total_pad,total_pad),(0,0),(0,0)),'constant', constant_values=0)

        #dn_proj = dn_proj[newaxis,:,jnp.newaxis]
        #beta_proj = beta_proj[jnp.newaxis,:,jnp.newaxis]

        exit_field = cx.thin_sample(field, beta_proj, dn_proj, self.phantom_dx/up_samp_fac)

        # to detector
        det_field = cx.transfer_propagate(exit_field, self.propdist, self.n_medium, 0, cval=model.cval, mode='same')
        img = self.det_resample_func(det_field.intensity.squeeze()[...,None,None], field.dx.ravel()[:1])[...,0,0]
        img = img / (self.det_dx/(self.phantom_dx/up_samp_fac))**2  # normalize counts to new pixel size
        img = img.swapaxes(0,1)      

        # TODO - consider cropping the top/bottom few rows (interference at cylinder bounds?)
        return img



# Set up model
key = jax.random.PRNGKey(3)  # pick any number
model = ProjPBI()
params = model.init(key, 0)



# make phantom after setting the model, since it depends on source energy.

Nx, Ny = model.phantom_Nx, model.phantom_Ny
vol_raw = make_raw_phantom(Nx, p0=0.7, p1=0.2, p2=0.1, id2=2, c1=1, c2=3)
subvol_raw = vol_raw[:,(Nx-Ny)//2:(Nx+Ny)//2,:]

delta_beta_phantom = np.zeros([Nx, Ny, Nx, 2])
for i, item in enumerate(model.material_basis.items()):
    idx, mat = item
    delta, beta = mat.delta_beta(model.energy)
    delta_beta_phantom[:,:,:,0][subvol_raw==idx] = beta # beta, 0
    delta_beta_phantom[:,:,:,1][subvol_raw==idx] = delta  # delta, 1
  
# View the phantom
fig, ax = plt.subplots(1, 6, figsize=[9,2], sharey=True, layout='constrained')
for i in range(len(ax)):
    yslice = i*Ny//len(ax)
    ax[i].set_title(f'$i$ = {yslice}')
    ax[i].imshow(delta_beta_phantom[:,yslice,:,1], vmin=0, vmax=delta_beta_phantom[:,:,:,1].max())
    ax[i].set_xticks([]); ax[i].set_yticks([])
plt.show()





# For simulation, replace the params with the phantom data (instead of the zeros).
params = unfreeze(params)
params['params']['volume'] = delta_beta_phantom
params = freeze(params)

# Define the CT rotation angles
tot_theta = jnp.pi  
N_theta = 100
thetas = jnp.linspace(0, tot_theta*(1-1/N_theta), N_theta)

# Simulate
forward = jax.jit(jax.vmap(model.apply, in_axes=(None, 0)))
data = forward(params, thetas)
data = jax.random.poisson(key, model.I0*data, data.shape) / model.I0  # add noise

# Show
fig, ax = plt.subplots(1, 5, figsize=[10,3], sharey=True, layout='constrained')
fig.suptitle(f'energy = {model.energy} keV')
for i in range(len(ax)):
    i_theta = i*N_theta//len(ax)
    m = ax[i].imshow(data[i_theta,:,:], vmin=0.5, vmax=1.5)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')
    ax[i].set_title(f'$\\theta$ = {180*thetas[i_theta]/PI:.1f} deg')
fig.colorbar(m)
plt.show()

#small_pixel_low_noise_projection_data_proj_approx
np.save('small_pixel_low_noise_projection_data_proj_approx', data)
np.save('projection_angles', thetas)





