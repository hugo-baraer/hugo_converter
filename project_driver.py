"""
  EoR_research/project_driver.py

  Author : Hugo Baraer
  Supervision by : Prof. Adrian Liu
  Affiliation : Cosmid dawn group at McGill University
  Date of creation : 2021-09-21

  This module is the driver and interacts between 21cmFast and the modules computing the require fields and parameters.
"""
#import classic python librairies
import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal
import imageio
from mpl_toolkits.mplot3d import Axes3D

#import this project's modules
import z_re_field as zre
import Gaussian_testing as gauss
import FFT
import statistical_analysis as sa

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
gaussian_field, mu, std = gauss.generate_gaussian_field(box_dim)
gauss.plot_field(gaussian_field,int(box_dim//2), mu, std)

#Gaussian_FFT for the 3D field, shift the field and plots with frquencies
delta = 0.1 #an arbitrary desired time interval for the Gaussian
X, Y, fft_gaussian_shifted = gauss.gaussian_fft(gaussian_field,delta,box_dim)
gauss.plot_ftt_field(fft_gaussian_shifted,int(box_dim//2), mu, std, X,Y)


#Axes3D.contourf(gaussian_shifted[0],gaussian_shifted[1], gaussian_shifted[2])

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

#Take and plot the Fourrier transform of the over-redshift along with it's frequnecy
delta = coeval.user_params.BOX_LEN / coeval.user_params.DIM
Xz, Yz, overzre_fft= FFT.compute_fft(overzre, delta, box_dim)

#plot this F(over_zre(x))
FFT.plot_ftt_field(overzre_fft, int(box_dim//2), Xz, Yz, title = r'F($\delta_z$ (x)) at a pixel dimension of {}³'.format(box_dim))


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

"""

coeval = p21c.run_coeval(redshift=z_mean,user_params={'HII_DIM': box_dim, "USE_INTERPOLATION_TABLES": False})

Xd, Yd, overd_fft= FFT.compute_fft(coeval.density, delta, box_dim)
FFT.plot_ftt_field(overd_fft, int(box_dim//2), Xd, Yd, title = r'F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))

freqs = np.fft.fftshift(np.fft.fftfreq(box_dim, d=delta))

#test the division process
division = np.divide(overzre_fft[:,25,25], overd_fft[:,25,25])
fig, ax = plt.subplots()
plt.scatter(freqs, division)
#plt.contourf(division[:,:,25])
#plt.colorbar()
plt.title(r'F($\delta_zre$ (x))/F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
plt.show()
print(freqs[25:], division[25:])
a, b = sa.get_param_value(freqs[25:], division[25:])
a,b0,k0 = a[0:]
print(a, b0, k0)
y_plot_fit = sa.lin_bias(freqs[25:], a,b0,k0)
fig, ax = plt.subplots()
plt.plot(freqs[25:], y_plot_fit)
plt.scatter(freqs[25:], division[25:])
#plt.contourf(division[:,:,25])
#plt.colorbar()
plt.title(r'F($\delta_zre$ (x))/F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
plt.show()