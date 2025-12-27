#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 11:55:19 2025

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



# Regularization functions for the loss function:
def TV(img, axes=[0,1,2]):  
    tot_grad = 0
    for axi in axes:
        tot_grad += jnp.sum(jnp.abs(jnp.diff(img, axis=axi)))  
    return tot_grad
    
def L1(img):
    return jnp.abs(img).sum()

# Convenience function for showing the optimization progress
def show_compare(params, loss, kw={}):
    vol1 = params['params']['volume']
    y_index = vol1.shape[1]//2
    
    fig, ax = plt.subplots(1,3,figsize=[10/2,3/1.5], dpi=300, layout='constrained')
    
    ax[0].plot(loss)
    ax[0].set_title('loss')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('iteration #')
    
    for i in range(2):
        axi = ax[i+1]
        m = axi.imshow(vol1[:,y_index,:,i], **kw) 
        axi.set_title(['beta', 'delta'][i])
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_xlabel('$x$')
        axi.set_ylabel('$z$')
        fig.colorbar(m, ax=axi)
    plt.show()
    
    
    
    
#Read in data and geometyr 
data = np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/noisy_projection_data.npy') 
thetas= np.load(r'/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/tuned_recons/projection_angles.npy')

# Number of epochs with no improvement after which learning rate will be reduced:
PATIENCE = 2  # @param{type:"integer"} # maybe try 10
# Number of epochs to wait before resuming normal operation after the learning rate reduction:
COOLDOWN = 50  # @param{type:"integer"}
# Factor by which to reduce the learning rate:
FACTOR = 0.8 # @param{type:"number"}
# Relative tolerance for measuring the new optimum:
RTOL = 1e-4  # @param{type:"number"}
atol = 0
# Number of iterations to accumulate an average value:
ACCUMULATION_SIZE = 1

max_iter = 2000

LRATE = 1e-6
EPS = 1e-8


#noisy_sgd did not work well with the given parameters 
optimizer = optax.chain(
    optax.adam(learning_rate=LRATE),
    contrib.reduce_on_plateau(
        patience=PATIENCE,
        cooldown=COOLDOWN,
        factor=FACTOR,
        rtol=RTOL,
        atol=atol,
        accumulation_size=ACCUMULATION_SIZE,
    ),
)


# TODO -- tune the regularization weights. For now, all regularization is "off"
w_tv_beta = 2500 #000#10
w_tv_delta = 500 #1614.2858 #10#5
w_l1_beta = 0 #-400 #10#00#10
w_l1_delta = 0#-200 #10#1000 #5
w_edge = 0.000 #0.001
def loss_fn(params, data):
    vol = params['params']['volume']
    vol_beta, vol_delta = vol[:,:,:,0], vol[:,:,:,1]   
    y_k = forward(params, thetas)   
    L2_norm = jnp.sqrt(jnp.sum((y_k - data)**2)) 
    L1_delta_term = w_l1_delta*L1(vol_delta)            
    L1_beta_term = w_l1_beta*L1(vol_beta)
    TV_delta_term = w_tv_delta*TV(vol_delta)
    TV_beta_term = w_tv_beta*TV(vol_beta)
    egde_penalty = w_edge *(jnp.abs(vol_delta/np.max(vol_delta) - vol_beta/np.max(vol_beta)).sum())
    loss = L2_norm + L1_delta_term + L1_beta_term + TV_delta_term + TV_beta_term + egde_penalty
    return loss


@jax.jit  
def update(params, opt_state, *args):
    loss, grads = jax.value_and_grad(loss_fn)(params, *args)
    updates, opt_state = optimizer.update(grads, opt_state, params,value=loss, grad=grads, value_fn=loss_fn)
    params = projection_non_negative(optax.apply_updates(params, updates))
    return params, opt_state, loss

    
###################################

 
# # Init the model
key = jax.random.PRNGKey(3)
model = MultiSlicePBI()
params = model.init(key, 0)
in_rand_array = np.zeros((64,30,64,2))
in_rand_array[:,:,:,0] = np.ones([64,30,64])*10**-8 #Starting intial guess set to correct order of magnitude
in_rand_array[:,:,:,1] = np.ones([64,30,64])*10**-6 #Starting intial guess set to correct order of magnitude
params['params']['volume'] = jnp.array(in_rand_array)
forward = jax.vmap(model.apply, in_axes=(None, 0))  
opt_state = optimizer.init(params)

# # Run
vols = np.zeros((max_iter,) + in_rand_array.shape)
loss = []
t0 = time()
lr_list = []
iter_k = 0
lr_scale = 1
while lr_scale > 0.001 or iter_k > max_iter:
    params, opt_state, loss_k = update(params, opt_state, data)
    loss.append(loss_k)
    vols[1 + iter_k,:,:,:,:] = params['params']['volume']

    lr_scale = optax.tree.get(opt_state, "scale")
    lr_list.append(lr_scale)
    iter_k = iter_k +1 
    
    print(f'iter {iter_k} (t = {time() - t0:.1f} s)')
    print(lr_scale)



np.save('multislice_image',params['params']['volume'])
show_compare(params, loss)  



