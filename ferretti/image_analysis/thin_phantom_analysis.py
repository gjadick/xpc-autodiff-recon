#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 14:21:24 2025

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




#First generate the phantom again using the model 
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
    


class MultiSlicePBI(nn.Module): 

    # Simulation settings
    propdist = 10e-3
    energy = 10
    material_basis = {1:tissue, 2:bone}
             
    # Phantom
    phantom_Nx: int = 16
    phantom_Ny: int = 30 
    phantom_dx: float = 0.5e-6
    phantom_fov = phantom_dx * phantom_Nx
    up_samp_fac: int = 2 
    # Detector
    det_Nx: int = 16  # 32 -- TODO: should have det_N < phantom_N, but need to account for this in phantom init during recon!    
    det_Ny: int = 30  # 10
    det_fwhm: float = 1e-6
    det_psf: str = 'lorentzian'  # code for the PSF is in fun.py
    resampling_method: str = 'linear'
    I0: int = 1e8  # very low noise to start
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

        up_samp_fac = self.up_samp_fac  # do upsampling. Look to do a linear upsampling later
        volume = jnp.repeat(jnp.repeat(jnp.repeat(self.volume,up_samp_fac,axis=0),up_samp_fac,axis=1),up_samp_fac,axis=2)
        N_pad = self.N_pad 
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
        field = pad(field, up_samp_fac*N_pad, cval=model.cval)  #pad field manually 

        # thru object
        rotated_vol = jax.vmap(
            rotate_volume, 
            in_axes=(-1, None)
        )(volume, angle)
        rotated_vol = jnp.swapaxes(jnp.stack([rotated_vol[0], rotated_vol[1]], axis=-1), 1, 2)
        
        # This time, multislice:
        obj_beta = jnp.pad(rotated_vol[:,:,:,0], up_samp_fac*N_pad, mode='constant', constant_values=0.0) # beta values 3D + pad
        obj_delta = jnp.pad(rotated_vol[:,:,:,1], up_samp_fac*N_pad, mode='constant', constant_values=0.0) ## delta values 3D + pad

        propagator_transfer = cx.compute_transfer_propagator(field, self.phantom_dx/up_samp_fac, self.n_medium)
        exit_field = cx.multislice_thick_sample(field, obj_beta, obj_delta, self.n_medium, self.phantom_dx/up_samp_fac, 0, \
            propagator=propagator_transfer, reverse_propagate_distance=None)  

        # to detector
        det_field = cx.transfer_propagate(exit_field, self.propdist, self.n_medium, 0, cval=model.cval, mode='same')
        img = self.det_resample_func(det_field.intensity.squeeze()[...,None,None], field.dx.ravel()[:1])[...,0,0]
        img = img / (self.det_dx/(self.phantom_dx/up_samp_fac))**2  # normalize counts to new pixel size
        img = img.swapaxes(0,1)      

        # TODO - consider cropping the top/bottom few rows (interference at cylinder bounds?)
        return img


# Set up model
key = jax.random.PRNGKey(3)  # pick any number
model = MultiSlicePBI()
params = model.init(key, 0)



# make phantom after setting the model, since it depends on source energy.

Nx, Ny = model.phantom_Nx, model.phantom_Ny
vol_raw = make_raw_phantom(Nx, p0=0.7, p1=0.2, p2=0.1, id2=2, c1=1, c2=3)
pad_for_vol = np.permute_dims(np.dstack([vol_raw[:,0,:]]*7), (0,2,1))
vol_raw = np.concatenate((pad_for_vol, vol_raw, pad_for_vol),axis=1)

#np.pad(vol_raw,((0,0),(7,7),(0,0)),mode='edge')
subvol_raw = vol_raw + 0 # I think to make volume smaller in y. Ignoting for now 

delta_beta_phantom = np.zeros([Nx, Ny, Nx, 2])
for i, item in enumerate(model.material_basis.items()):
    idx, mat = item
    delta, beta = mat.delta_beta(model.energy)
    delta_beta_phantom[:,:,:,0][vol_raw==idx] = beta # beta, 0
    delta_beta_phantom[:,:,:,1][vol_raw==idx] = delta  # delta, 1
  
# View the phantom
fig, ax = plt.subplots(1, 6, figsize=[9,2], sharey=True, layout='constrained')
for i in range(len(ax)):
    yslice = i*Ny//len(ax)
    ax[i].set_title(f'$i$ = {yslice}')
    ax[i].imshow(delta_beta_phantom[:,yslice,:,1], vmin=0, vmax=delta_beta_phantom[:,:,:,1].max())
    ax[i].set_xticks([]); ax[i].set_yticks([])
plt.show()




##Phantom generated. Start immage analysis 
mult_im = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/multislice_image_thin_sample.npy')
proj_im = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/proj_approx_image_thin.npy')

#Multi diff plots 
#deltas
slice_no = 15

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11,6.65))
true_delta_plot = axes[0,0].imshow(delta_beta_phantom[:,slice_no,:,1])
axes[0,0].set_title('Multislice Delta Reconstuction, \nCenter Slice')
axes[0,0].set_xlabel('X')
axes[0,0].set_ylabel('Y')
fig.colorbar(true_delta_plot,ax=axes[0,0])
recon_delta_plot = axes[0,1].imshow(mult_im[:,slice_no,:,1])
axes[0,1].set_title('Multislice Delta Reconstuction, \nCenter Slice')
axes[0,1].set_xlabel('X')
axes[0,1].set_ylabel('Y')
fig.colorbar(recon_delta_plot,ax=axes[0,1])
diff_plot = axes[0,2].imshow(delta_beta_phantom[:,slice_no,:,1] - mult_im[:,slice_no,:,1])
axes[0,2].set_title('Delta Differnce Image (True - Recon), \nCenter Slice')
axes[0,2].set_xlabel('X')
axes[0,2].set_ylabel('Y')
fig.colorbar(diff_plot,ax=axes[0,2])
#beta
true_delta_plot = axes[1,0].imshow(delta_beta_phantom[:,slice_no,:,0])
axes[1,0].set_title('Multislice Beta Reconstuction, \nCenter Slice')
axes[1,0].set_xlabel('X')
axes[1,0].set_ylabel('Y')
fig.colorbar(true_delta_plot,ax=axes[1,0])
recon_delta_plot = axes[1,1].imshow(mult_im[:,slice_no,:,0])
axes[1,1].set_title('Multislice Beta Reconstuction, \nCenter Slice')
axes[1,1].set_xlabel('X')
axes[1,1].set_ylabel('Y')
fig.colorbar(recon_delta_plot,ax=axes[1,1])
diff_plot = axes[1,2].imshow(delta_beta_phantom[:,slice_no,:,0] - mult_im[:,slice_no,:,0])
axes[1,2].set_title('Beta Differnce Image (True - Recon), \nCenter Slice')
axes[1,2].set_xlabel('X')
axes[1,2].set_ylabel('Y')
fig.colorbar(diff_plot,ax=axes[1,2])
plt.tight_layout()



#Proj diff plots 
#deltas
fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(11,6.75))
true_delta_plot = axes[0,0].imshow(delta_beta_phantom[:,slice_no,:,1])
axes[0,0].set_title('Proj. Approx. Delta Reconstuction,\nCenter Slice')
axes[0,0].set_xlabel('X')
axes[0,0].set_ylabel('Y')
fig.colorbar(true_delta_plot,ax=axes[0,0])
recon_delta_plot = axes[0,1].imshow(proj_im[:,slice_no,:,1])
axes[0,1].set_title('Proj. Approx. Delta Reconstuction, \nCenter Slice')
axes[0,1].set_xlabel('X')
axes[0,1].set_ylabel('Y')
fig.colorbar(recon_delta_plot,ax=axes[0,1])
diff_plot = axes[0,2].imshow(delta_beta_phantom[:,slice_no,:,1] - proj_im[:,slice_no,:,1])
axes[0,2].set_title('Delta Differnce Image (True - Recon), \nCenter Slice')
axes[0,2].set_xlabel('X')
axes[0,2].set_ylabel('Y')
fig.colorbar(diff_plot,ax=axes[0,2])
#beta
true_delta_plot = axes[1,0].imshow(delta_beta_phantom[:,slice_no,:,0])
axes[1,0].set_title('Proj. Approx. Beta Reconstuction, \nCenter Slice')
axes[1,0].set_xlabel('X')
axes[1,0].set_ylabel('Y')
fig.colorbar(true_delta_plot,ax=axes[1,0])
recon_delta_plot = axes[1,1].imshow(proj_im[:,slice_no,:,0])
axes[1,1].set_title('Proj. Approx. Beta Reconstuction, \nCenter Slice')
axes[1,1].set_xlabel('X')
axes[1,1].set_ylabel('Y')
fig.colorbar(recon_delta_plot,ax=axes[1,1])
diff_plot = axes[1,2].imshow(delta_beta_phantom[:,slice_no,:,0] - proj_im[:,slice_no,:,0])
axes[1,2].set_title('Beta Differnce Image (True - Recon), \nCenter Slice')
axes[1,2].set_xlabel('X')
axes[1,2].set_ylabel('Y')
fig.colorbar(diff_plot,ax=axes[1,2])
plt.tight_layout()


#Do some quantitative calculations
def calculate_rmse(vol,delta_beta_phantom):  
    rmse = np.sqrt(np.mean((vol - delta_beta_phantom) ** 2))
    return rmse

def beta_rmse(vol,delta_beta_phantom):
    vol_beta = vol[:,:,:,0]
    beta_phantom = delta_beta_phantom[:,:,:,0]
    return np.sqrt(np.mean((vol_beta - beta_phantom) ** 2))

def delta_rmse(vol,delta_beta_phantom):
    vol_delta = vol[:,:,:,1]
    delta_phantom = delta_beta_phantom[:,:,:,1]
    return np.sqrt(np.mean((vol_delta - delta_phantom) ** 2))


mult_total_rmse = calculate_rmse(mult_im,delta_beta_phantom)
mult_delta_rmse = delta_rmse(mult_im,delta_beta_phantom)
mult_beta_rmse = beta_rmse(mult_im,delta_beta_phantom)
proj_total_rmse = calculate_rmse(proj_im,delta_beta_phantom)
proj_delta_rmse = delta_rmse(proj_im,delta_beta_phantom)
proj_beta_rmse = beta_rmse(proj_im,delta_beta_phantom)
print(f'Mutlislice RMSE: \n  Total RMSE: {mult_total_rmse:.4E} \n  Delta RMSE: {mult_delta_rmse:.4E} \n  Beta RSME: {mult_beta_rmse:.4E}')
print(f'Projection Approx RMSE: \n  Total RMSE: {proj_total_rmse:.4E} \n  Delta RMSE: {proj_delta_rmse:.4E} \n  Beta RSME: {proj_beta_rmse:.4E}')

#Mean values of materials 
true_bone_d, true_bone_b = model.material_basis[2].delta_beta(model.energy)
true_tissue_d, true_tissue_b = model.material_basis[1].delta_beta(model.energy)

bone_log = subvol_raw == 2 
tissue_log = subvol_raw == 1 
#multi
print('Material Value Results for Multislice')
mult_tissue_d_mean = np.mean(mult_im[tissue_log,1])
mult_tissue_d_std = np.std(mult_im[tissue_log,1])
mult_bone_d_mean = np.mean(mult_im[bone_log,1])
mult_bone_d_std = np.std(mult_im[bone_log,1])
print(f'  Delta true tissue mean: {true_tissue_d:.4E}')
print(f'  Delta tissue mean+/-std: {mult_tissue_d_mean:.4E}+/-{mult_tissue_d_std:.4E}')
print(f'  Delta true bone mean: {true_bone_d:.4E}')
print(f'  Delta bone mean+/-std: {mult_bone_d_mean:.4E}+/-{mult_bone_d_std:.4E}')

mult_tissue_b_mean = np.mean(mult_im[tissue_log,0])
mult_tissue_b_std = np.std(mult_im[tissue_log,0])
mult_bone_b_mean = np.mean(mult_im[bone_log,0])
mult_bone_b_std = np.std(mult_im[bone_log,0])
print(f'  Beta true tissue mean: {true_tissue_b:.4E}')
print(f'  Beta tissue mean+/-std: {mult_tissue_b_mean:.4E}+/-{mult_tissue_b_std:.4E}')
print(f'  Beta true bone mean: {true_bone_b:.4E}')
print(f'  Beta bone mean+/-std: {mult_bone_b_mean:.4E}+/-{mult_bone_b_std:.4E}')

#proj approx
print('Material Value Results for Projection Approximation')
proj_tissue_d_mean = np.mean(proj_im[tissue_log,1])
proj_tissue_d_std = np.std(proj_im[tissue_log,1])
proj_bone_d_mean = np.mean(proj_im[bone_log,1])
proj_bone_d_std = np.std(proj_im[bone_log,1])
print(f'  Delta true tissue mean: {true_tissue_d:.4E}')
print(f'  Delta tissue mean+/-std: {proj_tissue_d_mean:.4E}+/-{proj_tissue_d_std:.4E}')
print(f'  Delta true bone mean: {true_bone_d:.4E}')
print(f'  Delta bone mean+/-std: {proj_bone_d_mean:.4E}+/-{proj_bone_d_std:.4E}')

proj_tissue_b_mean = np.mean(proj_im[tissue_log,0])
proj_tissue_b_std = np.std(proj_im[tissue_log,0])
proj_bone_b_mean = np.mean(proj_im[bone_log,0])
proj_bone_b_std = np.std(proj_im[bone_log,0])
print(f'  Beta true tissue mean: {true_tissue_b:.4E}')
print(f'  Beta tissue mean+/-std: {proj_tissue_b_mean:.4E}+/-{proj_tissue_b_std:.4E}')
print(f'  Beta true bone mean: {true_bone_b:.4E}')
print(f'  Beta bone mean+/-std: {proj_bone_b_mean:.4E}+/-{proj_bone_b_std:.4E}')


#SNR calculation 1
# Idea 1: Use exact volumes for signal and background 
#For delta images for multi and proj
#signal_pixels = mult_im[signal_log,1]
#back_pixels = mult_im[background_log,1]


snr_mult_d = (mult_bone_d_mean - mult_tissue_d_mean) / np.sqrt(0.5 * (mult_bone_d_std)**2 + 0.5 * (mult_tissue_d_std)**2)
snr_mult_b = (mult_bone_b_mean - mult_tissue_b_mean) / np.sqrt(0.5 * (mult_bone_b_std)**2 + 0.5 * (mult_tissue_b_std)**2)
print(f'SNR for Multi:')
print(f'  Delta: {snr_mult_d}')
print(f'  Beta: {snr_mult_b}')

snr_mult_d = (proj_bone_d_mean - proj_tissue_d_mean) / np.sqrt(0.5 * (proj_bone_d_std)**2 + 0.5 * (proj_tissue_d_std)**2)
snr_mult_b = (proj_bone_b_mean - proj_tissue_b_mean) / np.sqrt(0.5 * (proj_bone_b_std)**2 + 0.5 * (proj_tissue_b_std)**2)
print(f'SNR for Proj:')
print(f'  Delta: {snr_mult_d}')
print(f'  Beta: {snr_mult_b}')




