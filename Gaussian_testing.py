"""
  EoR_research/Gaussian_testing.py
 
  Author : Hugo Baraer
  Affiliation : McGill University
  Date of creation : 2021-10-26
  
  This module creates a 3D Gaussian field and Fourier transforms to get a similar gaussian in momentum space
  for testing purposes
  
"""
import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import z_re_field as zre
from tqdm import tqdm
from scipy import signal



def generate_gaussian_field(dim):
    '''
    This function creates a 3d Gaussian field to test np.FFT. This functions is inspired by :
    https://stackoverflow.com/questions/25720600/generating-3d-gaussian-distribution-in-python
    :param dim: the dimension of the desired Gaussian field
    :type dim: int
    :return: the 3d Gaussian field
    :rtype: 3D array
    '''
    x, y, z = np.mgrid[-1.0:1.0:int(dim)*1j, -1.0:1.0:int(dim)*1j, -1.0:1.0:int(dim)*1j]
    # Need an (N, 2) array of (x, y) pairs.
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([.050, .050, .050])
    covariance = np.diag(sigma ** 2)
    zi = multivariate_normal.pdf(xyz, mean=mu, cov=covariance)
    zi2 = np.reshape(zi, (x.shape))
    print(zi2)
    # Reshape back to a (30, 30) grid.
    return zi2, mu[0], sigma[0]

def plot_field(field,slice, mu, std):
    '''
    This functions plots a slice of the 3D  the Gaussian field at z=slice
    :param field: the Gaussian field to be plot
    :type field: 3d array
    :param sliced: the z at which the slice must be performed
    :type sliced: int
    :param mu, std: the mean  and standard deviation of the distribution respectively
    :type mu, std: in
    :return: a 2d contour plot of the field
    :rtype:
    '''
    fig, ax = plt.subplots()
    plt.contourf(field[slice])
    plt.colorbar()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title(r'slice of a 3D Gaussian centered in z with the mean at {}, and a standard deviation of {}'.format(mu, std))
    plt.show()

def gaussian_fft(field,delta,box_dim):
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


def plot_ftt_field(field,slice, mu, std, X,Y):
    '''
    This functions plots a slice of the fast Discrete transform of the 3D  the Gaussian field at z=slice
    :param field: the Gaussian field to be plot
    :type field: 3d array
    :param sliced: the z at which the slice must be performed
    :type sliced: int
    :params X,Y : the mesh grid of the frequencies
    :type X,Y: 2D arrays
    :param mu, std: the mean  and standard deviation of the distribution respectively
    :type mu, std: in
    :return: a 2d contour plot of the field
    :rtype: 2D array
    '''
    fig, ax = plt.subplots()
    plt.contourf(X, Y, field[slice])
    plt.colorbar()
    ax.set_xlabel(r'$k_x [Mpc^{-1}]$')
    ax.set_ylabel(r'$k_y [Mpc^{-1}]$')
    plt.title(r'F($Gaussian$) centered in z with the mean at {}, and a standard deviation of {}'.format(mu, std))
    plt.show()

#test the fft module with a perfect Gaussian field. This will later be moved in it's own module

#test fft with 1d gaussian

# window = signal.gaussian(51, std=7)
#
# fig, ax = plt.subplots()
# plt.plot(window)
# plt.title(r'1D Gaussian centered in z with the mean at {}, and a standard deviation of {}'.format(51,7))
# plt.show()
#
# gaussian1d_FFT = np.fft.fft(window)
# gaussian1d_shifted = np.fft.fftshift(np.fft.fft(window))
#
# fig, ax = plt.subplots()
# plt.plot(gaussian1d_FFT)
# plt.title(r'1D Gaussian centered in z with the mean at {}, and a standard deviation of {}'.format(51,7))
# plt.show()