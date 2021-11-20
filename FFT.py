"""
  EoR_research/FFT.py

  Author : Hugo Baraer
  Affiliation : McGill University
  Date of creation : 2021-09-21
  
  This module computes the Fourrier transform of the over-density and the over-redshift and plots them
  
"""

import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import z_re_field as zre
from tqdm import tqdm

def compute_fft(field,delta,box_dim):
    '''
    This function computes the discrete Fourier transform and it's frequencies
    :param field: the Gaussian field to perform fft on
    :type field: 3D array
    :param delta: the delta t use for frequency calculations
    :type delta: float
    :param box_dim: the dimension of the box
    :type box_dim: int
    :return: X,Y : the mesh grid of the frequencies
    :return: fft_field: the Fourrier transformed field
    :rtype: 2D arrays, 3D array
    '''

    freqs = np.fft.fftshift(np.fft.fftfreq(box_dim, d=delta))
    fft_shifted = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(field)))
    X, Y = np.meshgrid(freqs, freqs)

    return X, Y, fft_shifted

def plot_ftt_field(field,slice,X,Y, title = ''):
    '''
    This functions plots a slice of the fast Discrete transform of the 3D  the Gaussian field at z=slice
    :param field: the Gaussian field to be plot
    :type field: 3d array
    :param sliced: the z at which the slice must be performed
    :type sliced: int
    :param title: the desired title of the graph (default an empty string)
    :type title: 'string'
    :params X,Y : the mesh grid of the frequencies
    :type X,Y: 2D arrays
    :param mu, std: the mean  and standard deviation of the distribution respectively
    :type mu, std: in
    :return: a 2d contour plot of the field
    :rtype: 2D array
    '''
    fig, ax = plt.subplots()
    plt.contourf(X, Y, field[:,:,slice])
    plt.colorbar()
    ax.set_xlabel(r'$k_x [Mpc^{-1}]$')
    ax.set_ylabel(r'$k_y [Mpc^{-1}]$')
    plt.title(title)
    plt.show()

#gif the field through the z dimension
# filename = []
# levels = np.linspace(0, 16000, 51)
# for i in tqdm(range(box_dim)):
#     fig, ax = plt.subplots()
#     plt.contourf(abs(gaussian_shifted[i]),levels=levels)
#     plt.colorbar()
#     plt.savefig(f'F()_{i}.png')
#     plt.close()
#     filename.append(f'F()_{i}.png')
# images = []
# for filename in filename:
#     images.append(imageio.imread(filename))
# imageio.mimsave('fft_gaussian_through_.gif', images)