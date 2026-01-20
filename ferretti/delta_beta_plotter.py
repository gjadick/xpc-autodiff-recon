#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 18:55:55 2025

@author: aeferretti
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#vol_grid = np.load('beta_tv_search_vols.npy')
#vol_grid = np.load('/home/aeferretti/variation_lr_scheduling_beta_search_2500_10000_20_vols.npy')
vol_grid = np.load('/home/aeferretti/rotations/fall/xpc-autodiff-recon/ferretti/proj_variation_lr_scheduling_improv_guess_tv_5_vols.npy')

I = 5 #6
J = 4 #1
K = 0
L = 0
slice_no = 15
fig, ax = plt.subplots(1,2)
beta_im = ax[0].imshow(vol_grid[I,J,K,L,:,slice_no,:,0])
ax[0].set_title('Beta Center Slice')
fig.colorbar(beta_im,ax=ax[0])
delta_im = ax[1].imshow(vol_grid[I,J,K,L,:,slice_no,:,1])
ax[1].set_title('Delta Center Slice')
fig.colorbar(delta_im,ax=ax[1])

for i in range(15):
    plt.figure()
    plt.imshow(vol_grid[I,J,K,L,:,i*2,:,0])
    #plt.imshow(vol_grid[i,0,0,0,:,i*2,:,1])
    
losses = np.load('variation_lr_scheduling_search_edge_pen_losses.npy')
plt.figure()
plt.plot(losses[I,J,K,L,:])



I = 1
J = 1
K = 0
L = 0
M = 0
slice_no = 15
fig, ax = plt.subplots(1,2)
beta_im = ax[0].imshow(vol_grid[I,J,K,L,M,:,slice_no,:,0])
ax[0].set_title('Beta Center Slice')
fig.colorbar(beta_im,ax=ax[0])
delta_im = ax[1].imshow(vol_grid[I,J,K,L,M,:,slice_no,:,1])
ax[1].set_title('Delta Center Slice')
fig.colorbar(delta_im,ax=ax[1])

for i in range(15):
    plt.figure()
    plt.imshow(vol_grid[I,J,K,L,M,:,i*2,:,0])
    #plt.imshow(vol_grid[i,0,0,0,:,i*2,:,1])
    
losses = np.load('big_search_lr07_seach_2_losses.npy')
plt.figure()
plt.plot(losses[I,J,K,L,:])




rsmes = np.load('variation_lr_scheduling_search_edge_pen_search_rsmes.npy')
np.where(rsmes==np.min(rsmes))