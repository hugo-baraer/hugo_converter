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
box_dim = 143 #the desired spatial resolution of the box #the redshift z_bar at which the over-density is computed
radius_thick = 3. #the radii thickness (will affect the number of bins
box_len = 143 #int(143) #default value of 300
user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM":box_len}
cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27,
OMb= 0.045)

initial_conditions = p21c.initial_conditions(
user_params = user_params,
cosmo_params = cosmo_params
)


#xHI = p21c.ionize_box(redshift=7.0, zprime_step_factor=2.0, z_heat_max=15.0)

#zre.generate_quick_zre_field(16, 1, 1, initial_conditions)

#intialize a coeval cube at red shift z = z\bar
#coeval = p21c.run_coeval(redshift=8.0,user_params={'HII_DIM': box_dim, 'BOX_LEN': box_len, "USE_INTERPOLATION_TABLES": False})



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




"""Compute the over-redshfit or reionization and overdensity"""
#Compute the reionization redshift from the module z_re
z_re_box= zre.generate_zre_field(16, 1, 1, initial_conditions,box_dim)
overzre, zre_mean = zre.over_zre_field(z_re_box)
print(zre_mean)
position_vec = np.linspace(-49,50,143)
X, Y = np.meshgrid(position_vec, position_vec)
fig, ax = plt.subplots()
plt.contourf(X,Y,overzre[int(box_dim//2)])
plt.colorbar()
ax.set_xlabel(r'[Mpc h⁻¹]')
ax.set_ylabel(r'[Mpc h⁻¹]')
plt.title(r'slice of a the over-redshift of reionization at the center with a pixel resolution of {} Mpc h⁻¹'.format('1'))
plt.show()


#Take and plot the Fourrier transform of the over-redshift along with it's frequnecy
delta = 1 / (box_len / box_dim)
Xz, Yz, overzre_fft, freqzre= FFT.compute_fft(overzre, delta, box_dim)

#plot this F(over_zre(x))
FFT.plot_ftt_field(overzre_fft, int(box_dim//2), Xz, Yz, title = r'F($\delta_z$ (x)) at a pixel dimension of {}³'.format(box_dim))

perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes = initial_conditions)
#coeval = p21c.run_coeval(redshift=zre_mean,user_params={'HII_DIM': box_dim, 'BOX_LEN': box_len, "USE_INTERPOLATION_TABLES": False})

position_vec = np.linspace(-49,50,143)
X, Y = np.meshgrid(position_vec, position_vec)
fig, ax = plt.subplots()
plt.contourf(X,Y,perturbed_field.density[int(box_dim//2)])
plt.colorbar()
ax.set_xlabel(r'[Mpc h⁻¹]')
ax.set_ylabel(r'[Mpc h⁻¹]')
plt.title(r'slice of a the over-redshift of reionization at the center with a pixel resolution of {} Mpc h⁻¹'.format('1'))
plt.show()

Xd, Yd, overd_fft, freqd = FFT.compute_fft(perturbed_field.density, delta, box_dim)
#FFT.plot_ftt_field(overd_fft, int(box_dim//2), Xd, Yd, title = r'F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
freqs = np.fft.fftshift(np.fft.fftfreq(box_dim, d=delta))

#polar_div = np.ones((box_dim,box_dim,box_dim))
#polar_div = sa.cart2sphA(division)
overd_fft = np.square(abs(overd_fft))
overzre_fft = np.square(abs(overzre_fft))


#plot the power of the field
FFT.plot_ftt_field(overzre_fft, int(box_dim//2), Xz, Yz, title = r'$|F(\delta_z (x))|^2$ at a pixel dimension of {}³'.format(box_dim))
#wanted radius for plotting

cx = int(box_dim // 2)
#mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 == r**2

radii1 = np.linspace(0, np.sqrt((freqd/2) ** 2), num=int(3*np.sqrt((cx) ** 2) / int(radius_thick)))
radii2 = np.linspace(np.sqrt((freqd/2) ** 2), np.sqrt(3 * (freqd) ** 2), num=int(0.5*np.sqrt((cx) ** 2) / int(radius_thick)))
kvalues = np.concatenate((radii1[1:-1],radii2))


#kvalues = np.linspace(0,np.sqrt(3*(freqd)**2), num = int(np.sqrt(3*(cx)**2)/int(radius_thick)))
#kvalues = kvalues[1:]

#xompute the average for the field
values_zre, values_d, count_zre, count_d = sa.average_overk(box_dim,overzre_fft,overd_fft,radius_thick)

overzre_fft_k = np.divide(values_zre,count_zre)
overd_fft_k = np.divide(values_d,count_d)
#xcompute the standard deviation of that average
sigmad, sigmazre = sa.average_std(box_dim,overzre_fft, overd_fft, radius_thick, overzre_fft_k, overd_fft_k, count_zre, count_d)


#print(sigmad, sigmazre)
#print(a[6,2], overzre_fft[6,2,int(box_dim//2)], overd_fft[6,2,int(box_dim//2)],int(box_dim//2))
#xx=np.arange(0,len(oneD_div))
#division = np.divide(overzre_fft[:,int(box_dim//2),int(box_dim//2)], overd_fft[:,int(box_dim//2),int(box_dim//2)])


fig, ax = plt.subplots()
plt.errorbar(kvalues, overd_fft_k, yerr = sigmad, linestyle = 'None',capsize=4, marker ='o')
#plt.legend()
ax.set_xlabel(r'$k [Mpc^{-1} h]$')
ax.set_ylabel(r'$\delta_m$ (k)')
plt.title(r'$\delta_m$ (k)) as a function of k ')
plt.show()


fig, ax = plt.subplots()
plt.errorbar(kvalues, overzre_fft_k, yerr = sigmazre, linestyle = 'None',capsize=4, marker ='o')
#plt.legend()
ax.set_xlabel(r'$k [Mpc^{-1}h]$')
ax.set_ylabel(r'$\delta_zre$ (k)')
plt.title(r'$\delta_zre$ (k) as a function of k ')
plt.show()


#prim_basis = pymks.PrimitiveBasis(n_states=2)
#X_ = prim_basis.discretize(overzre_fft_k)

b_mz = np.sqrt(np.divide(overzre_fft_k,overd_fft_k))
bmz_errors = sa.compute_bmz_error(b_mz,overzre_fft_k,overd_fft_k,sigmad,sigmazre)
#bmz = sa.compute_bmz(overzre_fft_k[1:],overd_fft_k[1:])
#radii = radii[1:]
b_mz = b_mz[1:]
bmz_errors=bmz_errors[1:]
print(b_mz)
kvalues=kvalues[1:]
#take out nan values in radii

#nan_array = np.isnan(b_mz)
#not_nan_array = ~ nan_array
#bmz = b_mz[not_nan_array]
#radii = radii[not_nan_array]

fig, ax = plt.subplots()
#plt.plot(overd_fft_k[1:], y_plot_fit)
plt.errorbar(kvalues, b_mz, label = 'data fitting for',yerr = bmz_errors, linestyle = 'None',capsize=4, marker ='o') #xerr = sigmad[2:], yerr = sigmazre[2:], yerr = np.sqrt(bmz_errors)
#plt.title(r'$b_{zm}$ as a function of k ')
ax.set_ylabel(r'$b_{mz}$ ')
ax.set_xlabel(r'k [$Mpc^{-1} h$]')
plt.show()

#plot the log version of this graph
fig, ax = plt.subplots()
#plt.plot(overd_fft_k[1:], y_plot_fit)
plt.errorbar(kvalues, b_mz, label = 'data fitting for',yerr = bmz_errors, linestyle = 'None',capsize=4, marker ='o') #xerr = sigmad[2:], yerr = sigmazre[2:], yerr = np.sqrt(bmz_errors)
plt.title(r'$b_{zm}$ as a function of k ')
ax.set_ylabel(r'$b_{mz}$')
plt.xscale('log')
plt.yscale('log')
#ax.set_ylim(bottom=0,top=1)
ax.set_xlabel(r'k')
plt.show()



errs = np.ones_like(b_mz)*0.05
#initialize the MCMC
num_iter = 5000

ndim = 2
#ndim = 3 # number of parameters to fit for
nwalkers = 32
initial_pos = np.array((0.55, 0.025) + 0.02 * np.random.randn(nwalkers, ndim))
#initial_pos = np.zeros((ndim,nwalkers))
#initial_pos[0] = np.array((0.5) + 0.1 * np.random.randn(nwalkers))
#initial_pos[1] = np.array((0.1) + 0.02 * np.random.randn(nwalkers))
#initial_pos = np.array((0.5, 1.7, 0.15) + 0.1 * np.random.randn(nwalkers, ndim))
bmz_errors = np.ones_like(b_mz)*0.03


sampler = emcee.EnsembleSampler(nwalkers, ndim, sa.log_post_bmz_nob, args=(kvalues, b_mz, bmz_errors))
sampler.run_mcmc(initial_pos, num_iter, progress=True);

"""
f, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = [r"$\alpha$", r"$b_0$", r"$k_0$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number");
"""
f, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = [r"$\alpha$", r"$k_0$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number");


flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
inds = np.random.randint(len(flat_samples), size=100)

x0 = np.linspace(0, 1, 100)
y_plot_fit = sa.lin_bias(kvalues, 0.564,0.593,0.185)

"""
f, ax = plt.subplots(figsize=(6,4))
for ind in inds:
    sample = flat_samples[ind]
    ax.plot(x0, (sample[1]/(1+(x0/sample[2]))**sample[0]), alpha=0.05, color='red')
ax.errorbar(kvalues, b_mz, yerr = bmz_errors, linestyle = 'None',capsize=4, marker ='o')
ax.plot(kvalues,y_plot_fit)
#ax.set_xlim(0, 10.)
ax.set_ylabel(r'$b_{mz}$ ')
ax.set_xlabel(r'$k[Mpc⁻1 h]$')
plt.title(r'$b_{zm}$ as a function of k ')
plt.show()
"""
f, ax = plt.subplots(figsize=(6,4))
for ind in inds:
    sample = flat_samples[ind]
    ax.plot(x0, (1./(1+(x0/sample[1]))**sample[0]), alpha=0.05, color='red')
ax.errorbar(kvalues, b_mz, yerr = bmz_errors, linestyle = 'None',capsize=4, marker ='o')
#ax.plot(kvalues,y_plot_fit, color = 'b', label = 'values obtain by Battaglia et al. model')
#ax.set_xlim(0, 10.)
ax.set_ylabel(r'$b_{mz}$ ')
ax.set_xlabel(r'$k[Mpc^{⁻1} h]$')
plt.title(r'$b_{zm}$ as a function of k ')
plt.legend()
plt.show()

sample = flat_samples[50]
plt.scatter(kvalues, (b_mz-(1./(1+(x0/sample[1]))**sample[0])), alpha=0.05, color='red')
plt.show()
#these lines represents curve fitting with scipy's weird algorithm
"""
# print(freqs[25:], division[25:])
a1, b = sa.get_param_value(kvalues, b_mz)
a0,b0,k0 = a1[0:]
print(a0, b0, k0)
y_plot_fit = sa.lin_bias(kvalues, a0,b0,k0)

fig, ax = plt.subplots()
plt.plot(kvalues, y_plot_fit)
plt.errorbar(kvalues, b_mz, yerr = errs, label = 'data fitting for',linestyle = 'None',capsize=4, marker ='o')
plt.title(r'best curve fitting for the linear bias')
plt.show()
"""