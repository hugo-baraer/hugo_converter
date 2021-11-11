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

box_dim = 80 #the desired spatial resolution of the box #the redshift z_bar at which the over-density is computed
radius_thick = 1. # the radii thickness (will affect the number of bins



#intialize a coeval cube at red shift z = z\bar
coeval = p21c.run_coeval(redshift=8.0,user_params={'HII_DIM': box_dim, "USE_INTERPOLATION_TABLES": False})

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

#start by generating a 3d gaussian field and plotting a slice of it (uncomment second line for plotting)
gaussian_field, mu, std = gauss.generate_gaussian_field(box_dim)
#gauss.plot_field(gaussian_field,int(box_dim//2), mu, std)

#Gaussian_FFT for the 3D field, shift the field and plots with frquencies
delta = 0.1 #an arbitrary desired time interval for the Gaussian
X, Y, fft_gaussian_shifted = gauss.gaussian_fft(gaussian_field,delta,box_dim)
#gauss.plot_ftt_field(fft_gaussian_shifted,int(box_dim//2), mu, std, X,Y)


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

coeval = p21c.run_coeval(redshift=zre_mean,user_params={'HII_DIM': box_dim, "USE_INTERPOLATION_TABLES": False})

Xd, Yd, overd_fft= FFT.compute_fft(coeval.density, delta, box_dim)
#FFT.plot_ftt_field(overd_fft, int(box_dim//2), Xd, Yd, title = r'F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
freqs = np.fft.fftshift(np.fft.fftfreq(box_dim, d=delta))


#division = np.divide(overzre_fft[int(box_dim//2)], overd_fft[int(box_dim//2)])

#polar_div = np.ones((box_dim,box_dim,box_dim))
#polar_div = sa.cart2sphA(division)
# fig, ax = plt.subplots()
# plt.contourf(Xd, Yd, division)
# plt.colorbar()
# plt.show()

values = np.arange(0, box_dim)
count = np.arange(0, box_dim)
cx = int(box_dim//2)
cy = int(box_dim//2)

#wanted radius for plotting


# The two lines below could be merged, but I stored the mask
# for code clarity.
#mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 == r**2
#oneD_div = []

cx = int(box_dim//2)
cy = int(box_dim//2)
radii = np.linspace(0,np.sqrt(2*(cx)**2),num = int(np.sqrt(2*(cx)**2)/int(radius_thick)))
values = np.zeros(len(radii))
count = np.zeros(len(radii))
values_overd = np.zeros(len(radii))
count_overd= np.zeros(len(radii))

for i in tqdm(range(box_dim), 'transfering fields into k 1D array'):
     for j in range(box_dim):
        k_radius = np.sqrt((i-cx)**2 + (j-cy)**2)
        print(k_radius)
        for step, radius in enumerate(radii):
            if k_radius < radius:
                count[(step)]+=1
                count_overd[(step)]+=1
                values[(step)] += overzre_fft[i,j,int(box_dim//2)]
                values_overd[(step)] += overd_fft[i, j, int(box_dim // 2)]
                break
                #print(step)

            # if step == len(radii)-1:
            #     values[step] += overzre_fft[i, j, 25]
            #     count[step] += 1



print(values)
print(count)

overzre_fft_k = np.divide(values,count)
overd_fft_k = np.divide(values_overd,count_overd)

# This plot shows that only within the circle the value is set to 123.
# fig, ax = plt.subplots()
# plt.contourf(Xd, Yd, division)
# plt.colorbar()
# plt.show()

# #test the division process
# for i in division[:,0,0]:
#     for j in division[0,:,0]:
#         for z in division[0,0,:]:
#             polar_div[i,j,z] = sa.cart2pol(float(i),j,z)

#print(a[6,2], overzre_fft[6,2,int(box_dim//2)], overd_fft[6,2,int(box_dim//2)],int(box_dim//2))
#xx=np.arange(0,len(oneD_div))
#division = np.divide(overzre_fft[:,int(box_dim//2),int(box_dim//2)], overd_fft[:,int(box_dim//2),int(box_dim//2)])
fig, ax = plt.subplots()
plt.scatter(radii, overd_fft_k)

#plt.scatter(freqs, a[25], label = '25')
# plt.scatter(freqs, a[50], label = '50')
# plt.scatter(freqs, a[95], label = '95')
#plt.contourf(Xd,Yd,a)
#plt.colorbar()
plt.legend()
ax.set_xlabel(r'$k [Mpc^{-1}]$')
ax.set_ylabel(r'$\delta_m$ (k)')
plt.title(r'$\delta_m$ (k)) as a function of k '.format(coeval.redshift,box_dim))
plt.show()


fig, ax = plt.subplots()
plt.scatter(radii, overzre_fft_k)

#plt.scatter(freqs, a[25], label = '25')
# plt.scatter(freqs, a[50], label = '50')
# plt.scatter(freqs, a[95], label = '95')
#plt.contourf(Xd,Yd,a)
#plt.colorbar()
plt.legend()
ax.set_xlabel(r'$k [Mpc^{-1}]$')
ax.set_ylabel(r'$\delta_zre$ (k)')
plt.title(r'$\delta_zre$ (k) as a function of k ')
plt.show()





# print(freqs[25:], division[25:])
a1, b = sa.get_param_value(overd_fft_k[1:], overzre_fft_k[1:])
a0,b0,k0 = a1[0:]
a0 = 0.0
print(a0, b0, k0)
y_plot_fit = sa.lin_bias(overd_fft_k[1:], a0,b0,k0)

fig, ax = plt.subplots()
plt.plot(overd_fft_k[1:], y_plot_fit)
plt.scatter(overd_fft_k[1:], overzre_fft_k[1:], label = 'data fitting for')
# #plt.contourf(division[:,:,25])
# #plt.colorbar()
plt.title(r'best curve fitting for the linear bias')
plt.show()