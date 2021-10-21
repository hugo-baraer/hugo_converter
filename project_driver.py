"""
  EoR_research/project_driver.py

  Author : Hugo Baraer
  Affiliation : McGill University
  Date of creation : 2021-09-21

  This module is the driver and interacts between 21cmFast and the modules computing the require fields and parameters.
"""

import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import z_re_field as zre
from tqdm import tqdm
from scipy import signal
import imageio
#adjustable parameters to look out before running the driver

box_dim = 50 #the desired spatial resolution of the box
z_mean = 8.0 #the redshift z_bar at which the over-density is computed


#intialize a coeval cube at red shift z = z\bar
coeval = p21c.run_coeval(redshift=z_mean,user_params={'HII_DIM': box_dim, "USE_INTERPOLATION_TABLES": False})

#generate a sliced plot of the over-density the same way as we did for over-redshift
# fig, ax = plt.subplots()
# plt.contourf(coeval.density[:,:,0], cmap = 'jet')
# plt.colorbar()
# plt.title('slice of dark matter over-density at a redshfit of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
# plt.show()

#Take the Fourrier transform of the over density
# overdensity_FFT = np.fft.fft(coeval.density)
# fig, ax = plt.subplots()
# plt.contourf(overdensity_FFT[:,:,0])
# #plt.contourf(np.absolute(overdensity_FFT[:,:,0])**2)
# plt.colorbar()
# plt.title(r'F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
# plt.show()


# #plot dark_matter density for testing purposes
# plotting.coeval_sliceplot(coeval, kind = 'density')
# plt.tight_layout()
# plt.title('slice of dark matter over-density at a redshfit of {} and a pixel dimension of {}³'.format(coeval.redshift,150)) #coeval.user_params(HII_DIM)
# plt.show()

#plot the reionization redshift (test pursposes)
# plotting.coeval_sliceplot(coeval, kind = 'z_re_box', cmap = 'jet')
# plt.tight_layout()
# plt.title('reionization redshift ')
# plt.show()

"""
it appears coeval has a Z_re component, which shows if yes or not, the pixel was ionized at that reshift. This means that the pixel value is either
the redshift parameter entred in coeval, or -1 if it wasn't ionized at that redshift. 
With these information, I could plot z_re as function of time, by looking at a bunch of redshifts.
"""

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


"""Test the FFT function for a 3D Gaussian field"""

#start by generating a 3d gaussian field and plotting a slice of it
gaussian_field, mu, std = zre.generate_gaussian_field(box_dim)
fig, ax = plt.subplots()
plt.contourf(gaussian_field[int(box_dim//2.0)])
plt.colorbar()
plt.title(r'slice of a 3D Gaussian centered in z with the mean at {}, and a standard deviation of {}'.format(mu,std))
plt.show()

#Gaussian_FFT for the 3D field, shift the field and plots with frquencies
gaussian_FFT = np.fft.fftn(gaussian_field)
delta = gaussian_field[0,0,1]-gaussian_field[0,0,0]
freqs = np.fft.fftfreq(len(gaussian_field[0,0]), delta)
gaussian_shifted = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(gaussian_field)))

fig, ax = plt.subplots()
plt.contourf(abs(gaussian_shifted[int(box_dim//2.0)]))
plt.colorbar()
plt.title(r'F($Gaussian$) centered in z with the mean at {}, and a standard deviation of {}'.format(mu,std))
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
#
# images = []
# for filename in filename:
#     images.append(imageio.imread(filename))
# imageio.mimsave('fft_gaussian_through_.gif', images)


#Compute the reionization redshift from the module z_re
coeval.z_re_box = zre.generate_zre_field(16, 1, 1, coeval.z_re_box.shape[0])
overzre, zre_mean = zre.over_zre_field(coeval.z_re_box)

#Take and plot the Fourrier transform of the over-redshift
overzre_shifted_fft = abs(np.fft.fftshift(np.fft.fftn(np.fft.fftshift(overzre))))
fig, ax = plt.subplots()
plt.contourf(overzre_shifted_fft[:,:,25])
plt.colorbar()
plt.title(r'F($\delta_z$ (x)) at a pixel dimension of {}³'.format(coeval.redshift,box_dim))
plt.show()

"""
#plot a slice of this new redshift field, saved as the new z_re_box
plotting.coeval_sliceplot(coeval, kind = 'z_re_box', cmap = 'jet')
plt.tight_layout()
plt.title('reionization redshift field ')
plt.show()

#plot a slice of the over redshift
fig, ax = plt.subplots()
plt.contourf(overzre[:,:,0], cmap = 'jet')
plt.colorbar()
plt.title('over-redshift of reionization')
plt.show()


#Take the Fourrier transform of the over-redshift

coeval = p21c.run_coeval(redshift=z_mean,user_params={'HII_DIM': box_dim, "USE_INTERPOLATION_TABLES": False})

overdensity_FFT = np.fft.fft(coeval.density)
# fig, ax = plt.subplots()
# plt.contourf(overdensity_FFT[:,:,0])
# #plt.contourf(np.absolute(overdensity_FFT[:,:,0])**2)
# plt.colorbar()
# plt.title(r'F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
# plt.show()

division = np.divide(overzre_FFT,overdensity_FFT)
fig, ax = plt.subplots()
plt.contourf(division[:,:,0])
plt.colorbar()
plt.title(r'F($\delta_zre$ (x))/F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
plt.show()
a =1
"""