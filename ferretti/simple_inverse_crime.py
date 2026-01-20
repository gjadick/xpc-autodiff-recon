#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:23:58 2025

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
    



class MultiSlicePBI(nn.Module): 

    # Simulation settings
    propdist = 10e-3
    energy = 10
    material_basis = {1:tissue, 2:bone}
             
    # Phantom
    phantom_Nx: int = 64#64
    phantom_Ny: int = 30#30
    phantom_dx: float = 0.5e-6
    phantom_fov = phantom_dx * phantom_Nx
    up_samp_fac: int = 2 
    # Detector
    det_Nx: int = 64#64  # 32 -- TODO: should have det_N < phantom_N, but need to account for this in phantom init during recon!    
    det_Ny: int = 30#30  # 10
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

        up_samp_fac = 1  # do upsampling. Look to do a linear upsampling later
        #volume = jnp.repeat(jnp.repeat(jnp.repeat(self.volume,up_samp_fac,axis=0),up_samp_fac,axis=1),up_samp_fac,axis=2)
        N_pad = self.N_pad 
        volume = self.volume 
        # TODO (for AD recon)
        ## -- the initial phantom volume will match detector geometry
        ## -- then, upsample the volume from detector res to phantom res for accurate forward project.
        ## -- currently, this takes an already upsampled phantom (not compatible with good recon)
        
        # incident wave
        field = cx.plane_wave(
            shape = (self.phantom_Nx, self.phantom_Ny),
            dx = self.phantom_dx,
            spectrum = self.wavelen,
            spectral_density = 1
        ) 
        field = field / field.intensity.max()**0.5 / (self.phantom_Nx / self.det_Nx) / (self.phantom_Ny / self.det_Ny) 
        cval = field.intensity.max()
        
        # thru object
        rotated_vol = jax.vmap(
            rotate_volume,
            in_axes=(-1, None)
        )(volume, angle)
        rotated_vol = jnp.swapaxes(jnp.stack([rotated_vol[0], rotated_vol[1]], axis=-1), 1, 2)

        beta_proj = jnp.sum(rotated_vol[:,:,:,0],axis=1)[None,:,:,None,None]
        dn_proj = jnp.sum(rotated_vol[:,:,:,1],axis=1)[None,:,:,None,None]

        #dn_proj = dn_proj[newaxis,:,jnp.newaxis]
        #beta_proj = beta_proj[jnp.newaxis,:,jnp.newaxis]

        exit_field = cx.thin_sample(field, beta_proj, dn_proj, self.phantom_dx)

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


# For simulation, replace the params with the phantom data (instead of the zeros).
params = unfreeze(params)
params['params']['volume'] = delta_beta_phantom
params = freeze(params)

# Define the CT rotation angles
tot_theta = jnp.pi  
N_theta = 100
thetas = jnp.linspace(0, tot_theta*(1-1/N_theta), N_theta)

# Simulate
forward_1 = jax.vmap(model.apply, in_axes=(None, 0))
data = forward_1(params, thetas)
#data = jax.random.poisson(key, model.I0*data, data.shape) / model.I0  # add noise

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
    
    
    
    

max_iter = 50
show_iter = 10

LRATE = 1e-6
EPS = 1e-8
optimizer = optax.adam(learning_rate=LRATE, eps=EPS)

# TODO -- tune the regularization weights. For now, all regularization is "off"
w_tv_beta = 0 #000#10
w_tv_delta = 0 #10#5
w_l1_beta = 0 #10#00#10
w_l1_delta = 0 #10#1000 #5
def loss_fn(params, data):
    vol = params['params']['volume']
    vol_beta, vol_delta = vol[:,:,:,0], vol[:,:,:,1]   
    y_k = forward(params, thetas)   
    #if np.all(y_k == data):
    #    print('Equal')
    L2_norm = jnp.sqrt(jnp.sum((y_k - data)**2)) 
    #L1_delta_term = w_l1_delta*L1(vol_delta)            
    #L1_beta_term = w_l1_beta*L1(vol_beta)
    #TV_delta_term = w_tv_delta*TV(vol_delta)
    #TV_beta_term = w_tv_beta*TV(vol_beta)
    #loss = L2_norm + L1_delta_term + L1_beta_term + TV_delta_term + TV_beta_term 
    loss = L2_norm
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
#in_rand_array = np.zeros((16,15,16,2))
in_rand_array = np.zeros((64,30,64,2))
#in_rand_array[:,:,:,0] = np.random.rand(64,30,64)*10**-8 #delta_beta_phantom[:,:,:,0] #np.random.rand(64,30,64)*10**-6
#in_rand_array[:,:,:,1] = np.random.rand(64,30,64)*10**-8 #delta_beta_phantom[:,:,:,1]
in_rand_array[:,:,:,0] = delta_beta_phantom[:,:,:,0] #np.random.rand(64,30,64)*10**-6
in_rand_array[:,:,:,1] = delta_beta_phantom[:,:,:,1]
params['params']['volume'] = jnp.array(in_rand_array)
forward = jax.vmap(model.apply, in_axes=(None, 0)) #jax.jit(  #jax.vmap(model.apply, in_axes=(None, 0))  
opt_state = optimizer.init(params)  
data_crime = forward(params,thetas)



# # Run
loss = []
t0 = time()
for iter_k in range(max_iter):
    params, opt_state, loss_k = update(params, opt_state, data_crime)
    loss.append(loss_k)
    print(f'iter {iter_k} (t = {time() - t0:.1f} s)')
    
    if (iter_k%show_iter == 0):
        show_compare(params, loss)  
    

test_view = 50
test_proj = forward(params, thetas) 
plt.figure()
plt.imshow(test_proj[test_view,:,:],aspect='auto')
plt.title('Projection From Reconstructed Data')
plt.colorbar()




#Another silly test:
in_rand_array = np.zeros((64,30,64,2))
params['params']['volume'] = delta_beta_phantom

loss_fn(params, data)
print(np.sum(loss_fn(params, data_crime)))



grad_fn = jax.grad(loss_fn)
grads = grad_fn(params, data_test)



#trying for a function whose grad should be zero at 0
def easy_loss_fn(params):
    vol = params['params']['volume']
    return jnp.sum(vol**2)
key = jax.random.PRNGKey(3)
model = MultiSlicePBI()
params = model.init(key, 0)
#in_rand_array = np.zeros((16,15,16,2))
in_rand_array = np.zeros((64,30,64,2))
#in_rand_array[:,:,:,0] = np.random.rand(64,30,64)*10**-8 #delta_beta_phantom[:,:,:,0] #np.random.rand(64,30,64)*10**-6
#in_rand_array[:,:,:,1] = np.random.rand(64,30,64)*10**-8 #delta_beta_phantom[:,:,:,1]
in_rand_array[:,:,:,0] = delta_beta_phantom[:,:,:,0] #np.random.rand(64,30,64)*10**-6
in_rand_array[:,:,:,1] = delta_beta_phantom[:,:,:,1]
params['params']['volume'] = jnp.array(in_rand_array)*0

params['params']['volume'] = params['params']['volume']*0
grad_fn_easy = jax.grad(easy_loss_fn)
grads = grad_fn_easy(params)
#Indeed grad is zero everywhere!
