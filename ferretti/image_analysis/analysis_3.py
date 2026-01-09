#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 19:43:02 2026

@author: aeferretti
"""


from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
import matplotlib.colors as colors
from itertools import product
import pandas as pd

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



#Do some quantitative calculations
def calculate_rmse(vol,delta_beta_phantom):  
    rmse = np.sqrt(np.mean((vol - delta_beta_phantom) ** 2))
    return rmse

def beta_rmse_fn(vol,delta_beta_phantom):
    vol_beta = vol[:,:,:,0]
    beta_phantom = delta_beta_phantom[:,:,:,0]
    return np.sqrt(np.mean((vol_beta - beta_phantom) ** 2))

def delta_rmse_fn(vol,delta_beta_phantom):
    vol_delta = vol[:,:,:,1]
    delta_phantom = delta_beta_phantom[:,:,:,1]
    return np.sqrt(np.mean((vol_delta - delta_phantom) ** 2))

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
    phantom_Nx: int = 64 
    phantom_Ny: int = 30 
    phantom_dx: float = 0.5e-6
    phantom_fov = phantom_dx * phantom_Nx
    up_samp_fac: int = 2 
    # Detector
    det_Nx: int = 64  # 32 -- TODO: should have det_N < phantom_N, but need to account for this in phantom init during recon!    
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





##Phantom generated. Start immage analysis 

mult_im_l_50 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/low_noise_multislice_image.npy')
mult_im_h_50 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/high_noise_multislice_image.npy')
proj_im_l_50 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/low_noise_proj_approx_image_2.npy')
proj_im_h_50 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/high_noise_proj_approx_image_2.npy')

mult_im_l_25 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/small_pixel_low_noise_multislice_image.npy')
mult_im_h_25 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/small_pixel_high_noise_multislice_image.npy')
proj_im_l_25 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/small_pixel_low_noise_proj_approx_image_2.npy')
proj_im_h_25 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/small_pixel_high_noise_proj_approx_image.npy')

mult_im_l_10 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/very_small_pixel_low_noise_multislice_image.npy')
mult_im_h_10 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/very_small_pixel_high_noise_multislice_image.npy')
proj_im_l_10 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/very_small_pixel_low_noise_proj_approx_image_2.npy')
proj_im_h_10 = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/very_small_pixel_high_noise_proj_approx_image_2.npy')


ims = [mult_im_l_50,proj_im_l_50,mult_im_h_50,proj_im_h_50,mult_im_l_25,proj_im_l_25,mult_im_h_25,proj_im_h_25,mult_im_l_10,proj_im_l_10,mult_im_h_10,proj_im_h_10]
names = ['Multislice Low Noise 50', 'Projection Low Noise 50', 'Multislice High Noise 50', 'Projection High Noise 50', 'Multislice Low Noise 25', 'Projection Low Noise 25', 'Multislice High Noise 25', 'Projection High Noise 25', 'Multislice Low Noise 10', 'Projection Low Noise 10', 'Multislice High Noise 10', 'Projection High Noise 10']
rmse_list = []
delta_tissue_mean_list = []
delta_bone_mean_list = []
delta_tissue_std_list = []
delta_bone_std_list = []
delta_snr_list = []
delta_rmse_list = []
beta_tissue_mean_list = []
beta_bone_mean_list = []
beta_tissue_std_list = []
beta_bone_std_list = []
beta_snr_list = []
beta_rmse_list = []



for i in range(len(ims)):
    print('########################## ' + f'{names[i]}' ' #########################')
    total_rmse = calculate_rmse(ims[i],delta_beta_phantom)
    delta_rmse = delta_rmse_fn(ims[i],delta_beta_phantom)
    beta_rmse = beta_rmse_fn(ims[i],delta_beta_phantom)
    rmse_list.append(total_rmse)
    delta_rmse_list.append(delta_rmse)
    beta_rmse_list.append(beta_rmse)

    print(f'RMSE: \n  Total RMSE: {total_rmse:.4E} \n  Delta RMSE: {delta_rmse:.4E} \n  Beta RSME: {beta_rmse:.4E}')
    
    #Mean values of materials 
    true_bone_d, true_bone_b = model.material_basis[2].delta_beta(model.energy)
    true_tissue_d, true_tissue_b = model.material_basis[1].delta_beta(model.energy)
    
    bone_log = subvol_raw == 2 
    tissue_log = subvol_raw == 1 
    #multi
    print('Material Value Results')
    tissue_d_mean = np.mean(ims[i][tissue_log,1])
    tissue_d_std = np.std(ims[i][tissue_log,1])
    bone_d_mean = np.mean(ims[i][bone_log,1])
    bone_d_std = np.std(ims[i][bone_log,1])
    print(f'  Delta true tissue mean: {true_tissue_d:.4E}')
    print(f'  Delta tissue mean+/-std: {tissue_d_mean:.4E}+/-{tissue_d_std:.4E}')
    print(f'  Delta true bone mean: {true_bone_d:.4E}')
    print(f'  Delta bone mean+/-std: {bone_d_mean:.4E}+/-{bone_d_std:.4E}')
    
    tissue_b_mean = np.mean(ims[i][tissue_log,0])
    tissue_b_std = np.std(ims[i][tissue_log,0])
    bone_b_mean = np.mean(ims[i][bone_log,0])
    bone_b_std = np.std(ims[i][bone_log,0])
    print(f'  Beta true tissue mean: {true_tissue_b:.4E}')
    print(f'  Beta tissue mean+/-std: {tissue_b_mean:.4E}+/-{tissue_b_std:.4E}')
    print(f'  Beta true bone mean: {true_bone_b:.4E}')
    print(f'  Beta bone mean+/-std: {bone_b_mean:.4E}+/-{bone_b_std:.4E}')
    
    
    #SNR calculation 1
    # Idea 1: Use exact volumes for signal and background 
    #For delta images for multi and proj
    #signal_pixels = mult_im[signal_log,1]
    #back_pixels = mult_im[background_log,1]
    
    
    snr_d = (bone_d_mean - tissue_d_mean) / np.sqrt(0.5 * (bone_d_std)**2 + 0.5 * (tissue_d_std)**2)
    snr_b = (bone_b_mean - tissue_b_mean) / np.sqrt(0.5 * (bone_b_std)**2 + 0.5 * (tissue_b_std)**2)
    print(f'SNR:')
    print(f'  Delta: {snr_d}')
    print(f'  Beta: {snr_b}')
    
    
    
    delta_tissue_mean_list.append(tissue_d_mean)
    delta_bone_mean_list.append(bone_d_mean)
    delta_tissue_std_list.append(tissue_d_std)
    delta_bone_std_list.append(bone_d_std)
    delta_snr_list.append(snr_d)
    beta_tissue_mean_list.append(tissue_b_mean)
    beta_bone_mean_list.append(bone_b_mean)
    beta_tissue_std_list.append(tissue_b_std)
    beta_bone_std_list.append(bone_b_std)
    beta_snr_list.append(snr_b)
    
    
    
results_dict = {
    'Recon':names,
    'rmse_list':rmse_list,
    'delta_rmse_list':delta_rmse_list,
    'delta_tissue_mean':delta_tissue_mean_list,
    'delta_tissue_std':delta_tissue_std_list,
    'delta_bone_mean':delta_bone_mean_list,
    'delta_bone_std':delta_bone_std_list,
    'delta_snr':delta_snr_list,
    'beta_rmse_list':beta_rmse_list,
    'beta_tissue_mean':beta_tissue_mean_list,
    'beta_tissue_std':beta_tissue_std_list,
    'beta_bone_mean':beta_bone_mean_list,
    'beta_bone_std':beta_bone_std_list,
    'beta_snr':beta_snr_list,
    }



results_df = pd.DataFrame(results_dict)


pm_array = np.array(['±']* len(names),dtype=str)
scale = 10**6
results_dict_delta = {
    'Recon':names,
    'delta_tissue_mean+std': np.array(scale*np.array(delta_tissue_mean_list),dtype = str) + pm_array + np.array(scale*np.array(delta_tissue_std_list),dtype=str),
    'delta_bone_mean+std': np.array(scale*np.array(delta_bone_mean_list),dtype = str) + pm_array + np.array(scale*np.array(delta_bone_std_list),dtype=str),
    'delta_snr':delta_snr_list,
    }

results_df_delta = pd.DataFrame(results_dict_delta)

rmse_dict = {
    'Recon':names,
    'rmse_list':rmse_list,
    'delta_rmse_list':delta_rmse_list,
    'beta_rmse_list':beta_rmse_list,
    }
rmse_df = pd.DataFrame(rmse_dict)




#delta comparision plots

slice_no = 15

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.5,7.5))
mult_l_delta_plot = axes[0,0].imshow(ims[0][:,slice_no,:,1])
axes[0,0].set_title('Multislice Low Noise Recon., ')
axes[0,0].set_xlabel('X')
axes[0,0].set_ylabel('Y')
fig.colorbar(mult_l_delta_plot,ax=axes[0,0])
mult_h_delta_plot = axes[0,1].imshow(ims[2][:,slice_no,:,1])
axes[0,1].set_title('Multislice High Noise Recon., ')
axes[0,1].set_xlabel('X')
axes[0,1].set_ylabel('Y')
fig.colorbar(mult_h_delta_plot,ax=axes[0,1])
proj_l_delta_plot = axes[1,0].imshow(ims[1][:,slice_no,:,1])
axes[1,0].set_title('Projection Low Noise Recon, ')
axes[1,0].set_xlabel('X')
axes[1,0].set_ylabel('Y')
fig.colorbar(proj_l_delta_plot,ax=axes[1,0])
proj_h_delta_plot = axes[1,1].imshow(ims[3][:,slice_no,:,1])
axes[1,1].set_title('Projection High Noise Recon, ')
axes[1,1].set_xlabel('X')
axes[1,1].set_ylabel('Y')
fig.colorbar(proj_h_delta_plot,ax=axes[1,1])


plt.tight_layout()


for i in [0,1]:
    
    slice_no = 15
    title_pad = 15
    frac = 0.05
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,7.0))
    mult_l_beta_plot = axes[0,0].imshow(ims[0][:,slice_no,:,i])
    axes[0,0].set_title('M.S. Low Noise 0.50pix',pad=title_pad)
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('Y')
    fig.colorbar(mult_l_beta_plot,ax=axes[0,0],fraction=frac)
    mult_h_beta_plot = axes[0,1].imshow(ims[2][:,slice_no,:,i])
    axes[0,1].set_title('M.S High Noise 0.50pix',pad=title_pad)
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('Y')
    fig.colorbar(mult_h_beta_plot,ax=axes[0,1],fraction=frac)
    proj_l_beta_plot = axes[0,2].imshow(ims[1][:,slice_no,:,i])
    axes[0,2].set_title('P.A. Low Noise 0.50pix',pad=title_pad)
    axes[0,2].set_xlabel('X')
    axes[0,2].set_ylabel('Y')
    fig.colorbar(proj_l_beta_plot,ax=axes[0,2],fraction=frac)
    proj_h_beta_plot = axes[0,3].imshow(ims[3][:,slice_no,:,i])
    axes[0,3].set_title('P.A. High Noise 0.50pix',pad=title_pad)
    axes[0,3].set_xlabel('X')
    axes[0,3].set_ylabel('Y')
    fig.colorbar(proj_h_beta_plot,ax=axes[0,3],fraction=frac)
    
    
    mult_l_beta_plot = axes[1,0].imshow(ims[4][:,slice_no,:,i])
    axes[1,0].set_title('M.S. Low Noise 0.25pix',pad=title_pad)
    axes[1,0].set_xlabel('X')
    axes[1,0].set_ylabel('Y')
    fig.colorbar(mult_l_beta_plot,ax=axes[1,0],fraction=frac)
    mult_h_beta_plot = axes[1,1].imshow(ims[6][:,slice_no,:,i])
    axes[1,1].set_title('M.S. High Noise 0.25pix',pad=title_pad)
    axes[1,1].set_xlabel('X')
    axes[1,1].set_ylabel('Y')
    fig.colorbar(mult_h_beta_plot,ax=axes[1,1],fraction=frac)
    proj_l_beta_plot = axes[1,2].imshow(ims[5][:,slice_no,:,i])
    axes[1,2].set_title('P.A. Low Noise 0.25pix',pad=title_pad)
    axes[1,2].set_xlabel('X')
    axes[1,2].set_ylabel('Y')
    fig.colorbar(proj_l_beta_plot,ax=axes[1,2],fraction=frac)
    proj_h_beta_plot = axes[1,3].imshow(ims[7][:,slice_no,:,i])
    axes[1,3].set_title('P.A. High Noise 0.25pix',pad=title_pad)
    axes[1,3].set_xlabel('X')
    axes[1,3].set_ylabel('Y')
    fig.colorbar(proj_h_beta_plot,ax=axes[1,3],fraction=frac)
    
    
    mult_l_beta_plot = axes[2,0].imshow(ims[8][:,slice_no,:,i])
    axes[2,0].set_title('M.S. Low Noise 0.10pix',pad=title_pad)
    axes[2,0].set_xlabel('X')
    axes[2,0].set_ylabel('Y')
    fig.colorbar(mult_l_beta_plot,ax=axes[2,0],fraction=frac)
    mult_h_beta_plot = axes[2,1].imshow(ims[10][:,slice_no,:,i])
    axes[2,1].set_title('M.S. High Noise 0.10pix',pad=title_pad)
    axes[2,1].set_xlabel('X')
    axes[2,1].set_ylabel('Y')
    fig.colorbar(mult_h_beta_plot,ax=axes[2,1],fraction=frac)
    proj_l_beta_plot = axes[2,2].imshow(ims[9][:,slice_no,:,i])
    axes[2,2].set_title('P.A. Low Noise 0.10pix',pad=title_pad)
    axes[2,2].set_xlabel('X')
    axes[2,2].set_ylabel('Y')
    fig.colorbar(proj_l_beta_plot,ax=axes[2,2],fraction=frac)
    proj_h_beta_plot = axes[2,3].imshow(ims[11][:,slice_no,:,i])
    axes[2,3].set_title('P.A. High Noise 0.10pix',pad=title_pad)
    axes[2,3].set_xlabel('X')
    axes[2,3].set_ylabel('Y')
    fig.colorbar(proj_h_beta_plot,ax=axes[2,3],fraction=frac)
    
    plt.tight_layout()
    
    
