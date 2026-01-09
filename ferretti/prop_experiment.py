#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 23:33:23 2025

@author: aeferretti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 19:09:50 2025

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
    



# Simulation settings
propdist = 10e-3
energy = 10
material_basis = {1:tissue, 2:bone}
         
# Phantom
#up_samp_fac: int = 2
phantom_Nx: int = 64 
phantom_Ny: int = 30 
phantom_dx: float = 0.5e-6
phantom_fov = phantom_dx * phantom_Nx

# Detector
det_Nx: int = 64  # 32 -- TODO: should have det_N < phantom_N, but need to account for this in phantom init during recon!    
det_Ny: int = 30  # 10
det_fwhm: float = 1e-6
det_psf: str = 'lorentzian'  # code for the PSF is in fun.py
resampling_method: str = 'linear'
I0: int = 1e8  # very low noise to start
det_fov: float = phantom_fov
det_dx: float = det_fov / det_Nx
n = 1
kernel_size = 16
# Misc.
wavelen = get_wavelen(energy)
N_pad: int = 16  # note 8-- this is probably pushing the lower end of acceptable. Need to check?
n_medium: float = 1
cval = 1 + 0j
det_resample_func = init_plane_resample(
    (det_Nx + kernel_size, det_Ny + kernel_size), 
    (det_dx, det_dx ), 
    resampling_method=resampling_method
)

#make volume for testing
Nx = phantom_Nx + 0
Ny = phantom_Ny + 0
vol_raw = make_raw_phantom(Nx, p0=0.7, p1=0.2, p2=0.1, id2=2, c1=1, c2=3)
subvol_raw = vol_raw[:,(Nx-Ny)//2:(Nx+Ny)//2,:]

delta_beta_phantom = np.zeros([Nx, Ny, Nx, 2])
for i, item in enumerate(material_basis.items()):
    idx, mat = item
    delta, beta = mat.delta_beta(energy)
    delta_beta_phantom[:,:,:,0][subvol_raw==idx] = beta # beta, 0
    delta_beta_phantom[:,:,:,1][subvol_raw==idx] = delta  # delta, 1







up_samp_facs = [1,2]#[1,2,4,8]
fig, axs = plt.subplots(1,len(up_samp_facs))

for i in range(len(up_samp_facs)):
    t1 = time()
    up_samp_fac = up_samp_facs[i]
    #volume = delta_beta_phantom
    volume = jnp.repeat(jnp.repeat(jnp.repeat(delta_beta_phantom,up_samp_fac,axis=0),up_samp_fac,axis=1),up_samp_fac,axis=2)
    angle = 0 #np.pi/6
    #volume_up = jnp.kron(volume,jnp.ones((up_samp_fac,up_samp_fac)))
    # TODO (for AD recon)
    ## -- the initial phantom volume will match detector geometry
    ## -- then, upsample the volume from detector res to phantom res for accurate forward project.
    ## -- currently, this takes an already upsampled phantom (not compatible with good recon)
    
    # incident wave
    field = cx.plane_wave(
        shape = (phantom_Nx*up_samp_fac, phantom_Ny*up_samp_fac),
        dx = phantom_dx/up_samp_fac,
        spectrum = wavelen,
        spectral_density = 1
    ) 
    field = field / field.intensity.max()**0.5 / (phantom_Nx / det_Nx) / (phantom_Ny / det_Ny) 
    field = pad(field, up_samp_fac*N_pad, cval=cval)
    
    
    # thru object
    rotated_vol = jax.vmap(
        rotate_volume, 
        in_axes=(-1, None)
    )(volume, angle)
    rotated_vol = jnp.swapaxes(jnp.stack([rotated_vol[0], rotated_vol[1]], axis=-1), 1, 2)
        
    # This time, multislice:
    obj_beta = jnp.pad(rotated_vol[:,:,:,0], up_samp_fac*N_pad, mode='constant', constant_values=0.0) #[:,:,up_samp_fac*N_pad:-1*up_samp_fac*N_pad]  # beta values 3D
    obj_delta = jnp.pad(rotated_vol[:,:,:,1], up_samp_fac*N_pad, mode='constant', constant_values=0.0) #[:,:,up_samp_fac*N_pad:-1*up_samp_fac*N_pad] # delta values 3D
    #n = jnp.mean(1-obj_delta - 1j*obj_beta)
    propagator_transfer = cx.compute_transfer_propagator(field, phantom_dx/up_samp_fac, n_medium)
    exit_field = cx.multislice_thick_sample(field, obj_beta, obj_delta, n, phantom_dx/up_samp_fac, 0, \
        propagator=propagator_transfer, reverse_propagate_distance=None)  
    
    # to detector
    det_field = cx.transfer_propagate(exit_field, propdist, n_medium, 0, cval=cval, mode='same')
    img = det_resample_func(det_field.intensity.squeeze()[...,None,None], field.dx.ravel()[:1])[...,0,0]
    img = img / (det_dx/(phantom_dx/up_samp_fac))**2  # normalize counts to new pixel size
    img = apply_psf(img, det_fov, det_dx, psf=det_psf, fwhm=det_fwhm, kernel_width=0.09)
    img = img[8:img.shape[0]-8,8:img.shape[1]-8]
    
    img = img.swapaxes(0,1)
    
    print(f'exicution time was {time()-t1}')
    
    axs[i].imshow(img,aspect='auto')
    axs[i].set_xlabel('$x$')
    axs[i].set_ylabel('$y$')
    axs[i].set_title(f'Upsample = {up_samp_fac}')
    


plt.figure()
plt.imshow(det_field.intensity.squeeze().T)
plt.xlabel('Detector X')
plt.ylabel('Detector Y')
plt.title(f'Pre-Downsample at x {up_samp_facs[i]} upsampling')


plt.figure()
plt.imshow(exit_field.intensity.squeeze().T)
plt.xlabel('Sample Edge X')
plt.ylabel('Sample Edge Y')
plt.title(f'Signal at object edge')



test_1 = cx.transfer_propagate(exit_field, propdist, n_medium, N_pad, cval=cval, mode='same')
test_2 = cx.transfer_propagate(exit_field, propdist, n_medium, N_pad, cval=cval, mode='full')
test_3 = cx.transfer_propagate(exit_field, 10**-6, n_medium, N_pad, cval=cval, mode='full')

plt.figure()
plt.imshow(test_2.intensity.squeeze().T)
plt.xlabel('Det X')
plt.ylabel('Det Y')
plt.title(f'Signal at detector edge with padding')


plt.figure()
plt.imshow(test_3.intensity.squeeze().T)
plt.xlabel('Det X')
plt.ylabel('Det Y')
plt.title(f'Signal at small propdist edge with padding')


