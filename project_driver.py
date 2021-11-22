"""
  EoR_research/project_driver.py

  Author : Hugo Baraer
  Supervision by : Prof. Adrian Liu
  Affiliation : Cosmic dawn group at McGill University
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
import emcee
from scipy import signal
import imageio
from mpl_toolkits.mplot3d import Axes3D
import corner
#import this project's modules
import z_re_field as zre
import Gaussian_testing as gauss
import FFT
import statistical_analysis as sa
#import pymks

#adjustable parameters to look out before running the driver
box_dim = 50 #the desired spatial resolution of the box #the redshift z_bar at which the over-density is computed
radius_thick = 3. #the radii thickness (will affect the number of bins
box_len = 300 #default value of 300


#intialize a coeval cube at red shift z = z\bar
coeval = p21c.run_coeval(redshift=8.0,user_params={'HII_DIM': box_dim, 'BOX_LEN': box_len, "USE_INTERPOLATION_TABLES": False})



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



"""Test the FFT function for a 3D Gaussian field"""

#start by generating a 3d gaussian field and plotting a slice of it (uncomment second line for plotting)
gaussian_field, mu, std = gauss.generate_gaussian_field(box_dim)
#gauss.plot_field(gaussian_field,int(box_dim//2), mu, std)

#Gaussian_FFT for the 3D field, shift the field and plots with frquencies
delta = 0.1 #an arbitrary desired time interval for the Gaussian
X, Y, fft_gaussian_shifted = gauss.gaussian_fft(gaussian_field,delta,box_dim)
#gauss.plot_ftt_field(fft_gaussian_shifted,int(box_dim//2), mu, std, X,Y)


#Compute the reionization redshift from the module z_re
coeval.z_re_box = zre.generate_zre_field(16, 1, 1, coeval.z_re_box.shape[0],box_len)
overzre, zre_mean = zre.over_zre_field(coeval.z_re_box)

#Take and plot the Fourrier transform of the over-redshift along with it's frequnecy
delta = coeval.user_params.BOX_LEN / coeval.user_params.DIM
Xz, Yz, overzre_fft, freqzre= FFT.compute_fft(overzre, delta, box_dim)

#plot this F(over_zre(x))
FFT.plot_ftt_field(overzre_fft, int(box_dim//2), Xz, Yz, title = r'F($\delta_z$ (x)) at a pixel dimension of {}³'.format(box_dim))


coeval = p21c.run_coeval(redshift=zre_mean,user_params={'HII_DIM': box_dim, 'BOX_LEN': box_len, "USE_INTERPOLATION_TABLES": False})

Xd, Yd, overd_fft, freqd = FFT.compute_fft(coeval.density, delta, box_dim)
#FFT.plot_ftt_field(overd_fft, int(box_dim//2), Xd, Yd, title = r'F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
freqs = np.fft.fftshift(np.fft.fftfreq(box_dim, d=delta))

#polar_div = np.ones((box_dim,box_dim,box_dim))
#polar_div = sa.cart2sphA(division)
overd_fft = np.square(abs(overd_fft))
overzre_fft = np.square(abs(overzre_fft))


#plot the power of the field
FFT.plot_ftt_field(overzre_fft, int(box_dim//2), Xz, Yz, title = r'$|F(\delta_z (x))|^2$ at a pixel dimension of {}³'.format(box_dim))
#wanted radius for plotting

#mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 == r**2


cx = int(box_dim//2)
cy = int(box_dim//2)
radii = np.linspace(0,np.sqrt(3*(cx)**2),num = int(np.sqrt(3*(cx)**2)/int(radius_thick)))
kvalues = np.linspace(0,np.sqrt(3*(freqd)**2), num = int(np.sqrt(3*(cx)**2)/int(radius_thick)))
kvalues = kvalues[1:]
radii = radii[1:]
values = np.zeros(len(radii))
count = np.zeros(len(radii))
values_overd = np.zeros(len(radii))
count_overd= np.zeros(len(radii))

#loop through each point and seperate them in
for i in tqdm(range(box_dim), 'transfering fields into k 1D array'):
     for j in range(box_dim):
         for z in range(box_dim):
            k_radius = np.sqrt((i-cx)**2 + (j-cy)**2 +(z-cy)**2)
            for step, radius in enumerate(radii):
                if k_radius < radius:
                    count[(step)]+=1
                    count_overd[(step)]+=1
                    values[(step)] += overzre_fft[i,j,z]
                    values_overd[(step)] += overd_fft[i, j, z]
                    break
                    #print(step)

sigmad = np.zeros(len(values))
sigmazre = np.zeros(len(values))
#compute the error (standard deviation of each point)
for i in tqdm(range(box_dim), 'co'):
    for j in range(box_dim):
        for z in range(box_dim):
            k_radius = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (z - cy) ** 2)
            for step, radius in enumerate(radii):
                if k_radius < radius:
                    average = values_overd[step]/count_overd[step]
                    sigmad[step] += (overd_fft[i, j, z] - average)**2
                    sigmazre[step] += (overzre_fft[i, j, z] - (values[step]/count[step]))**2
                    break

            # if step == len(radii)-1:
            #     values[step] += overzre_fft[i, j, 25]
            #     count[step] += 1

sigmad = np.sqrt(np.divide(sigmad,count_overd))
sigmazre = np.sqrt(np.divide(sigmazre,count))

#print(sigmad, sigmazre)

print(values)
print(count)
print(sigmazre)
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
plt.errorbar(kvalues, overd_fft_k, yerr = sigmad, linestyle = 'None',capsize=4, marker ='o')
#plt.legend()
ax.set_xlabel(r'$k [Mpc^{-1}]$')
ax.set_ylabel(r'$\delta_m$ (k)')
plt.title(r'$\delta_m$ (k)) as a function of k '.format(coeval.redshift,box_dim))
plt.show()


fig, ax = plt.subplots()
plt.errorbar(kvalues, overzre_fft_k, yerr = sigmazre, linestyle = 'None',capsize=4, marker ='o')
#plt.legend()
ax.set_xlabel(r'$k [Mpc^{-1}]$')
ax.set_ylabel(r'$\delta_zre$ (k)')
plt.title(r'$\delta_zre$ (k) as a function of k ')
plt.show()


#prim_basis = pymks.PrimitiveBasis(n_states=2)
#X_ = prim_basis.discretize(overzre_fft_k)

b_mz = np.sqrt(np.divide(overzre_fft_k,overd_fft_k))
bmz_errors = sa.compute_bmz_error(b_mz,overzre_fft_k,overd_fft_k,sigmad,sigmazre)
#bmz = sa.compute_bmz(overzre_fft_k[1:],overd_fft_k[1:])
#radii = radii[1:]
print(b_mz)

#take out nan values in radii

#nan_array = np.isnan(b_mz)
#not_nan_array = ~ nan_array
#bmz = b_mz[not_nan_array]
#radii = radii[not_nan_array]

fig, ax = plt.subplots()
#plt.plot(overd_fft_k[1:], y_plot_fit)
plt.errorbar(kvalues, b_mz, label = 'data fitting for', linestyle = 'None',capsize=4, marker ='o') #xerr = sigmad[2:], yerr = sigmazre[2:], yerr = np.sqrt(bmz_errors)
plt.title(r'$b_{zm}$ as a function of k ')
ax.set_ylabel(r'$b_{mz}$ ')
ax.set_xlabel(r'k')
plt.show()

errs = np.ones_like(b_mz)*0.05
#initialize the MCMC
num_iter = 5000
ndim = 4 # number of parameters to fit for
nwalkers = 32
initial_pos = np.array((6.4, 1.7, 0.4, 0.4) + 0.1 * np.random.randn(nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, sa.log_post_bmz, args=(kvalues, b_mz, errs))
sampler.run_mcmc(initial_pos, num_iter, progress=True);

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
fig = corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84])
inds = np.random.randint(len(flat_samples), size=100)

x0 = np.linspace(0, 1, 13)
f, ax = plt.subplots(figsize=(6,4))
for ind in inds:
    sample = flat_samples[ind]
    ax.plot(x0, (sample[1]/(1+(x0/sample[2]))**sample[0])+sample[3], alpha=0.05, color='red')
ax.scatter(kvalues, b_mz, marker ='o')
#ax.set_xlim(0, 10.)
ax.set_ylabel(r'$\delta_zre$ (k) ')
ax.set_xlabel(r'$\delta_m$ (k)')
plt.show()

#these lines represents curve fitting with scipy's weird algorithm

# print(freqs[25:], division[25:])
a1, b = sa.get_param_value(kvalues, b_mz)
a0,b0,k0 = a1[0:]
print(a0, b0, k0)
y_plot_fit = sa.lin_bias(b_mz, a0,b0,k0)

fig, ax = plt.subplots()
plt.plot(kvalues, y_plot_fit)
plt.errorbar(kvalues, b_mz, label = 'data fitting for',linestyle = 'None',capsize=4, marker ='o')
plt.title(r'best curve fitting for the linear bias')
plt.show()
