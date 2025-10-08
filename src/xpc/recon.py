#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28

@author: gjadick

For parallel-beam filtered-back projection.

"""

import numpy as np
import jax.numpy as jnp
from jax import vmap
from einops import rearrange

DTYPE = jnp.float32
PI = jnp.pi


def make_test_phantom(N, p1, p2, DTYPE=np.uint8):
    assert (p1 <= 0.5) and (p2 <= 0.5) and (p1 > 0) and (p2 > 0)
    
    coords = np.linspace(-N/2, N/2, N)
    Y, Z, X = np.meshgrid(coords, coords, coords)
    r1 = p1*N/2
    r2 = p2*N/2
    x1, y1, z1 = -r1/np.sqrt(2), -r1/np.sqrt(2), -r1/3
    x2, y2, z2 = r2/np.sqrt(2), r2/np.sqrt(2), r2/3
    
    obj = np.zeros([N,N,N], dtype=DTYPE)
    obj[np.where(X**2 + Y**2 < (N/2)**2)] = 1   
    obj[np.where((X-x1)**2 + (Y-y1)**2 + (Z-z1)**2 < r1**2)] = 2
    obj[np.where((X-x2)**2 + (Y-y2)**2 + (Z-z2)**2 < r2**2)] = 3

    return obj


def pre_process(sino_log, s):
    """
    Pre-process the raw projections for FFBP.
    Uses the parallel beam ramp filter `h` (Hsieh CT, 4th ed, Eq. 3.28)

    Parameters
    ----------
    sino_log : 2D array
        Log'd sinogram, i.e. ln(I0/I) ~ mu*L.
    s : float
        Channel sampling distance [m].

    Returns
    -------
    sino_filtered : 2D array
        The pre-processed sinogram (same shape as input sino_log).
    """    
    _, N_channels = sino_log.shape
    n = jnp.arange(-N_channels//2, N_channels//2, 1, dtype=DTYPE)
    # h = 1 / jnp.where(n%2==1, -(n*PI*s)**2, (2*s)**2)
    h = -1 / (n*PI*s)**2
    h = h.at[n%2==0].set(0)
    h = h.at[N_channels//2].set(1 / (2*s)**2)

    # sino_filtered = s * jnp.array([jnp.convolve(row, h, mode='same') for row in sino_log], dtype=DTYPE)  
    sino_filtered = vmap(
        lambda row: s * jnp.convolve(row, h, mode='same'),
        in_axes=0,
        out_axes=0
    )(sino_log)
    
    return sino_filtered

        

def get_recon_coords(N_matrix, FOV):
    """
    Compute the polar coordinates corresponding to each pixel in the final 
    reconstruction matrix (common to all recons with same matrix dimensions.)

    Parameters
    ----------
    N_matrix : int
        Number of pixels in the recon matrix so that its shape = [N_matrix, N_matrix].
    FOV : float
        Field of view of the recon matrix [m].

    Returns
    -------
    r_matrix : 2D array ~ [N_matrix, N_matrix]
        Radial coordinate [m] of each pixel.
    theta_matrix : 2D array ~ [N_matrix, N_matrix]
        Angle coordinate [radians] of each pixel.

    """
    x = (FOV/N_matrix) * jnp.arange((1-N_matrix)/2, N_matrix/2, 1, dtype=DTYPE)
    X, Y = jnp.meshgrid(x, -x)
    r_matrix = jnp.sqrt(X**2 + Y**2)
    theta_matrix = jnp.arctan2(X, Y) + PI
    return r_matrix, theta_matrix


def do_fbp(sino, r_matrix, theta_matrix, dchannel, dtheta):
    """
    Reconstruct a log'd sinogram using parallel-beam filtered back-projection.

    Parameters
    ----------
    sino_log : 2D array ~ [N_proj, N_col]
        The log'd sinogram data, ln(-I/I0). Should already be pre-processed.
        Shape is the number of projection views (N_proj) by number of detector
        channels (N_col).
    r_matrix : 2D array ~ [N_matrix, N_matrix]
        Radial coordinate [m] of each pixel.
    theta_matrix : 2D array ~ [N_matrix, N_matrix]
        Angle coordinate [radians] of each pixel.
    dchannel : float
        Channel sampling [m], i.e. width of each detector channel.
    dtheta : float
        Angular sampling [radians], i.e. angle between projection views.
        NOT the same as `theta_matrix`, though view angle is sometimes called theta elsewhere.

    Returns
    -------
    matrix : 2D array ~ [N_matrix, N_matrix]
        The reconstructed CT image.
    """
    N_proj, N_channels = sino.shape
    N_matrix, _ = r_matrix.shape
    beam_width = dchannel * N_channels
    
    matrix = jnp.zeros([N_matrix, N_matrix], dtype=DTYPE)
    for i_proj in range(N_proj):  
        beta = i_proj * dtheta + PI
        channel_targets = r_matrix * jnp.cos(theta_matrix - beta)
        i_channel_matrix = ((channel_targets + beam_width/2) / dchannel).astype(jnp.int32)  # convert channel targets to indices
        
        fbp_i = jnp.choose(i_channel_matrix, sino[i_proj], mode='clip')
        jnp.nan_to_num(fbp_i, copy=False)  # just in case, check for NaN
        matrix += fbp_i 

    # def fbp_1proj():   # TODO : write vmap 

    matrix = matrix * (PI / N_proj)  
    return matrix


def get_recon(sino_log, N_matrix, FOV, s, dbeta):
    '''
    Reconstruct a CT sinogram into a cross-sectional image.

    Parameters
    ----------
    sino_log : 2D array ~ [N_proj, N_channels]
        The input sinogram. For normal CT recon, this should be the log data.
        For a basis material sinogram, this should be the density line integrals.
    N_matrix : int
        Number of pixels in the reconstructed matrix, shape [N_matrix, N_matrix]
    FOV : float
        Size of field-of-view to reconstruct [m].
    s : float
        Channel sampling [m], i.e. width of each detector channel.
    dbeta : float
        Angular sampling [radians], i.e. angle between projection views. Sometimes called dtheta.
    Returns
    -------
    recon : 2D array, shape [N_matrix, N_matrix].
        The reconstructed image.
    '''    
    sino_filtered = pre_process(sino_log, s)
    r_matrix, theta_matrix = get_recon_coords(N_matrix, FOV)
    recon = do_fbp(sino_filtered, r_matrix, theta_matrix, s, dbeta)
    return recon



        

        
