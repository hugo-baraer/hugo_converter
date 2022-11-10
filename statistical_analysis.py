"""
  EoR_research/statistical_analysis.py
 
  Author : Hugo Baraer
  Affiliation : McGill University
  Supervision by : Prof. Adrian Liu
  Date of creation : 2021-10-26
  
  This module contains the necessary statistical analysis function required for the analysis
  of the free parameters alpha, beta, and k_0
  
"""
import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import z_re_field as zre
from tqdm import tqdm
from scipy.optimize import curve_fit
import math as m
import emcee
import corner
import plot_params as pp
import FFT
import powerbox as pbox
#import pymks
def average_overk(box_dim,field, nb_bins, logbins = False):
    '''
    this modules compute the average of 3d fields over theta and phi to make it only k dependant
    :param box_dim: the number of pixels in one side of the box
    :type box_dim: int
    :param field: the field to average over
    :type field: 3D array
    :param nb_bins: the numbver of avergaging shells (nb of data points)
    :type nb_bins: float (or int)
    :param logbins: if the repartition of the bin points is logarithmic (if True)
    :return: the average over theta and phi of the 2 selected fields
    :rtype: 1D array
    '''
    cx = int(box_dim // 2)

    #uncomment these lines for an uneven distribution of points
    # radii1 = np.linspace(0, np.sqrt((cx/2) ** 2), num=int(3*np.sqrt((cx) ** 2) / int(radius_thick)))
    # radii2 = np.linspace(np.sqrt((cx/2) ** 2), np.sqrt(3 * (cx) ** 2), num=int(0.5*np.sqrt((cx) ** 2) / int(radius_thick)))
    # radii = np.concatenate((radii1[1:-1],radii2))

    #radii = np.linspace(0, np.sqrt(3 * (cx) ** 2), num=int(np.sqrt(3 * (cx) ** 2) / int(radius_thick)))
    radii = np.linspace(0, np.sqrt(3 * (cx) ** 2), nb_bins)
    radii = radii[1:]  # exlude the radii 0 to avoid divison by 0

    if logbins :
        radii =  np.logspace(0, np.log10(np.sqrt(3 * (cx) ** 2)), nb_bins)


    values = np.zeros(len(radii))
    count = np.zeros(len(radii))

    for i in tqdm(range(box_dim), 'Computing the cross correlation', position=0, leave=True):
        for j in range(box_dim):
            for z in range(box_dim):
                k_radius = np.sqrt((i - cx) ** 2 + (j - cx) ** 2 + (z - cx) ** 2)
                for step, radius in enumerate(radii):
                    if k_radius < radius:
                        count[(step)] += 1
                        values[(step)] += field[i, j, z]
                        break
    return values, count

def average_overk_2fields(box_dim,overzre_fft, overd_fft, radius_thick):
    '''
    this modules compute the average of 3d fields over theta and phi to make it only k dependant, but with 2 fields(goes quicker than two loops)
    :param box_dim: the number of pixels in the box
    :type box_dim: int
    :param fields (overzre and overd): the averaged fields
    :type field: 3D array
    :param radius_thick: the thickness of the averaged shells
    :type radius_thick: float (or int)
    :return: the average over theta and phi of the 2 selected fields
    :rtype: 1D array
    '''
    cx = int(box_dim // 2)
    cy = int(box_dim // 2)

    #uncomment these lines for an uneven distribution of points
    # radii1 = np.linspace(0, np.sqrt((cx/2) ** 2), num=int(3*np.sqrt((cx) ** 2) / int(radius_thick)))
    # radii2 = np.linspace(np.sqrt((cx/2) ** 2), np.sqrt(3 * (cx) ** 2), num=int(0.5*np.sqrt((cx) ** 2) / int(radius_thick)))
    # radii = np.concatenate((radii1[1:-1],radii2))

    radii = np.linspace(0, np.sqrt(3 * (cx) ** 2), num=int(np.sqrt(3 * (cx) ** 2) / int(radius_thick)))
    radii = radii[1:]  # exlude the radii 0 to avoid divison by 0

    values = np.zeros(len(radii))
    count = np.zeros(len(radii))
    values_overd = np.zeros(len(radii))
    count_overd = np.zeros(len(radii))


    for i in tqdm(range(box_dim), 'transfering fields into k 1D array'):
        for j in range(box_dim):
            for z in range(box_dim):
                k_radius = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (z - cy) ** 2)
                for step, radius in enumerate(radii):
                    if k_radius < radius:
                        count[(step)] += 1
                        count_overd[(step)] += 1
                        values[(step)] += overzre_fft[i, j, z]
                        values_overd[(step)] += overd_fft[i, j, z]
                        break
    return values, values_overd, count, count_overd


def average_std(box_dim,overzre_fft, overd_fft, radius_thick, averagezre, averaged, countzre, countd):
    '''
    This function computes the standard devitation of the averaged field
    :param box_dim: the number of pixels in the array (size)
    :type box_dim: float or int
    :param overzre_fft: the averaged on over redshift fields
    :type overzre_fft: 3d array
    :param overd_fft: the averaged on over density fields
    :type overd_fft: 3d array
    :param radius_thick: the size of the radius range of the averaged shells
    :type radius_thick: float or int
    :param averagezre: the average of the values for each rings
    :type averagezre: 1D array
    :param averaged: the average of the values for the each ring (density)
    :type averaged: 1d array
    :param countzre: the number of points per ring (over-redshift)
    :type countzre: 1d array
    :param countd: the number of points per ring(dnesity)
    :type countd: 1d array
    :return: the standard deviation of the ring (werves as statistical error)
    :rtype:  1d array
    '''
    cx = int(box_dim // 2)
    cy = int(box_dim // 2)

    # radii1 = np.linspace(0, np.sqrt((cx/2) ** 2), num=int(3*np.sqrt((cx) ** 2) / int(radius_thick)))
    # radii2 = np.linspace(np.sqrt((cx/2) ** 2), np.sqrt(3 * (cx) ** 2), num=int(0.5*np.sqrt((cx) ** 2) / int(radius_thick)))
    # radii = np.concatenate((radii1[1:-1],radii2))

    radii = np.linspace(0, np.sqrt(3 * (cx) ** 2), num=int(np.sqrt(3 * (cx) ** 2) / int(radius_thick)))
    radii = radii[1:] #exlude the radii 0 to avoid divison by 0

    sigmad = np.zeros(len(countd))
    sigmazre = np.zeros(len(countzre))
    # compute the error (standard deviation of each point)
    for i in tqdm(range(box_dim), 'computing the standard deviation for the averaged field'):
        for j in range(box_dim):
            for z in range(box_dim):
                k_radius = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (z - cy) ** 2) #the radius of the point
                for step, radius in enumerate(radii):
                    if k_radius < radius:
                        sigmad[step] += (overd_fft[i, j, z] - averaged[step]) ** 2
                        sigmazre[step] += (overzre_fft[i, j, z] - averagezre[step]) ** 2
                        break
    std_d = np.sqrt(np.divide(sigmad,countd)) #get the sqrt of each ring for the overdensity
    std_zre = np.sqrt(np.divide(sigmazre,countzre)) #get the sqrt of each ring for the over-redshift of reionization
    return np.divide(std_d,np.sqrt(countd)), np.divide(std_zre,np.sqrt(countzre)) #divide the sqrt by their number of points to get the confidence in the mean.

def compute_bmz_error(b_mz, overzre_fft_k,overd_fft_k, sigmad, sigmazre):
    '''
    This module computes the errors in the bm_z error with error propagation from the average std
    :param b_mz: the linear bias factor
    :type b_mz: 1d array
    :param overzre_fft_k: the average over phi and theta of the over_density field
    :type overzre_fft_k: 1d array
    :param overd_fft_k: the average over phi and theta of the over_redshift of reionization field
    :type overd_fft_k: 1d array
    :param sigmad: the std of the averaged over-density values
    :type sigmad: 1d array
    :param sigmazre: the std of the averaged over-redshift values
    :type sigmazre: 1d array
    :return: the errors bars for the bmz factor
    :rtype: 1d array
    '''
    term1 = np.divide((0.5*sigmad),overd_fft_k)**2
    term2 = np.divide((0.5*sigmazre),overzre_fft_k)**2
    return np.multiply(b_mz,(np.sqrt(term1 + term2)))


def lin_bias(x, a,b0,k0):
    '''
    This function represents the linear bias equation that will be fitted.
    :param k: the k_space that will be iterated over
    :type k: array
    :param alpha: free parameter fitting for
    :type alpha: float
    :param b0:
    :type b0: float
    :param ko:
    :type ko: float
    :return: the linear bias equation
    :rtype: array
    '''
    return b0/(1+x/k0)**a

def get_param_value(x,y):
    '''
    This function computes the best-fit value for
    :param x:
    :type x:
    :param y:
    :type y:
    :return:
    :rtype:
    '''
    return curve_fit(lin_bias, np.asarray(x), np.asarray(y))

def cart2pol(x, y, z):
    '''
    convert cartesian values to polar coordinates.
    inspired by https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    :param x: values in x of the polar coordinates
    :type x: arr
    :param y: values in y of the polar coordinates
    :type y: arr
    :param z: values in z of the polar coordinates
    :type z: arr
    :return: the polar coordinates values of the array
    :rtype:
    '''
    XsqPlusYsq = x ** 2 + y ** 2
    r = m.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(XsqPlusYsq))  # theta
    az = m.atan2(y, x)  # phi
    return r, elev, az

def cart2sphA(pts):
    return np.array([cart2pol(x,y,z) for x,y,z in pts])

def plot_bestfit(x,y):
    '''
    This function computes the best-fit value for
    :param x:
    :type x:
    :param y:
    :type y:
    :return:
    :rtype:
    '''

def log_prior_lin(theta):
    a, b = theta
    if -20 < a < 20. and -50< b < 50:
        return 0.0
    return -np.inf

def log_likelihood_lin(theta, x, y, yerr):
    '''
    this functions evaluates the likelihood, to test on the data we have
    :param theta:
    :type theta:
    :param x:
    :type x:
    :param y:
    :type y:
    :param yerr:
    :type yerr:
    :return:
    :rtype:
    '''
    a, b = theta
    sigma2 = yerr ** 2
    model = a*x +b
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))  # the 2pi factor doesn't affect the shape


def log_post_lin(theta, x, y, yerr):
    lp = log_prior_lin(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lin(theta, x, y, yerr)


def log_prior_bmz(theta):
    a, b0, k0 = theta
    if 0 < a < 3. and 0 < b0 < 15 and 0 < k0 < 0.3:
        return 0.0
    return -np.inf

def log_likelihood_bmz(theta, x, y, yerr):
    '''
    this functions evaluates the likelihood, to test on the data we have
    :param theta:
    :type theta:
    :param x:
    :type x:
    :param y:
    :type y:
    :param yerr:
    :type yerr:
    :return:
    :rtype:
    '''
    a, b0, k0= theta
    sigma2 = yerr ** 2
    try:
        model = (b0/(1+(x/k0)))**a
        likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        return likelihood
    except:
        return -np.inf



def log_post_bmz(theta, x, y, yerr):
    lp = log_prior_bmz(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_bmz(theta, x, y, yerr)


def log_prior_bmz_nob(theta):

    a, k0 = theta
    if 0.4 < a < 2.5 and 0. < k0 < 0.3:
        return 0.0
    return -np.inf

def log_likelihood_bmz_nob(theta, x, y, yerr):
    '''
    this functions evaluates the likelihood of the b_mz parameter, to test on the data we have
    :param theta: the a and k0 parameter we are fitting for
    :type theta: 2d array
    :param x: the x values fitting for
    :type x:
    :param y:
    :type y:
    :param yerr:
    :type yerr:
    :return:
    :rtype:
    '''
    #sigma2 = -np.exp(1.2*x-1)+1.35
    a, k0= theta
    #sigma2 = (yerr ) ** 2
    sigma2 = (yerr
              /(-np.exp((0.7*x)+0.1)+2.2)) ** 2

    try:
        model = 0.93/((1+((x)/k0))**(a))
        likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        return likelihood
    except:
        return -np.inf



def log_post_bmz_nob(theta, x, y, yerr):
    lp = log_prior_bmz_nob(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_bmz_nob(theta, x, y, yerr)


'''
The next functions is when fitting the MCMC with error weighting
'''

def log_prior_bmz_errs(theta):

    a, k0, p = theta
    if 0.1 < a < 20 and 0. < k0 < 20 and 0<p<0.5:
        return 0.0
    return -np.inf

def log_likelihood_bmz_errs(theta, x, y, yerr):
    '''
    this functions evaluates the likelihood of the b_mz parameter, to test on the data we have
    :param theta: the a and k0 and p parameter we are fitting for
    :type theta: 3d array
    :param x: the x values fitting for
    :type x:
    :param y:
    :type y:
    :param yerr:
    :type yerr:
    :return:
    :rtype:
    '''

    #sigma2 = -np.exp(1.2*x-1)+1.35
    a, k0, p = theta
    #sigma2 = (yerr ) ** 2
    sigma2 = ((p)
              /(yerr)) ** 2

    try:
        model = 0.593/((1+((x)/k0))**(a))
        likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        return likelihood
    except:
        return -np.inf



def log_post_bmz_errs(theta, x, y, yerr):
    lp = log_prior_bmz_errs(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_bmz_errs(theta, x, y, yerr)


def compute_bmz(overzre_k,overd_k):
    '''
    this module computes the bmz factor as it should be according to the equation in Battaglia et al.
    :param overzre_k: the over-redshift of reionization in momentum space
    :type overzre_k: 1d array
    :param overd_k: the over density in momentum space
    :type overd_k: 1d array
    :return: the linear bias factor
    :rtype: 1d array
    '''
    #prim_basis = pymks.PrimitiveBasis(n_states =2)
    #X_ = prim_basis.discretize(X)
    inner_product_dzre = np.correlate(overzre_k,overzre_k, "full")
    inner_product_dzre = inner_product_dzre[inner_product_dzre.size//2:]
    inner_product_dm = np.correlate(overd_k,overd_k, "full")
    inner_product_dm = inner_product_dm[inner_product_dm.size // 2:]
    return np.sqrt(np.divide(inner_product_dzre,inner_product_dm))


"""
One last test of probability function MCMC, with b_mz and erros scaling
"""

def log_likelihood_bmz_b_errs(theta, x, y, yerr):
    '''
    this functions evaluates the likelihood, to test on the data we have
    :param theta:
    :type theta:
    :param x:
    :type x:
    :param y:
    :type y:
    :param yerr:
    :type yerr:
    :return:
    :rtype:
    '''
    a, b0, k0, p = theta
    sigma2 = ((p)
              / (yerr)) ** 2
    try:
        model = (b0/(1+(x/k0)))**a
        likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        return likelihood
    except:
        return -np.inf

def log_prior_bmz_b_errs(theta):

    a, b0, k0, p = theta
    if 0.3 < a < 3.5 and 0. < k0 < 3.5 and 0.5 < b0 < 1.6 and 0 < p < 0.15:
        return 0.0
    return -np.inf

def log_post_bmz_b_errs(theta, x, y, yerr):
    lp = log_prior_bmz_b_errs(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_bmz_b_errs(theta, x, y, yerr)

def run_MCMC_free_params(x,y,errs, zre_mean, num_iter = 5000, nwalkers = 32, plot_walkers = False, plot_corners = False, plot_best_fit_sample = False, data_dict= None, varying_input = 'None', varying_in_value = 0 ):
    '''
    This function runs the MCMC
    :param x: the kbins fitting for
    :param y: the b_mz scatter point fitting for
    :param errs: the error (associated with the cross corelation error weighting)
    :param num_iter: the number of iteration to run the Markov Chain Monte Carlo (default 5000)
    :param nwalkers: the number of walker completing MCMC (default 32)
    :param plot_walkers: plot the walker behaviour plot
    :param plot_corners: plot the corner plots for the posterior distributions,
    :param plot_best_fit_sample: plot the b_mz and the associated
    :param dict_storing: to store the best-fitted value and confidence interval in the provided dictionnary (default none)
    :param varying_input: the name of the varying input the value are computed over (default is Heff)
    :param varying_in_value: the value of the varying parameter
    :return: the values of the posterior distribution for the free parameters (mean value)
    '''

    ndim = 4
    #initial parameters for 4 dim
    initial_pos = np.ones((32,4))
    initial_pos[:,0] *= 0.7 +0.06 * np.random.randn(nwalkers)
    initial_pos[:,1] *= 0.8 + 0.1 * np.random.randn(nwalkers)
    initial_pos[:,2] *= 0.2 + 0.01 * np.random.randn(nwalkers)
    initial_pos[:,3] *= 0.01 + 0.005 * np.random.randn(nwalkers)



    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_bmz_b_errs, args=(x, y, errs))
    sampler.run_mcmc(initial_pos, num_iter, progress=True)
    samples = sampler.get_chain()

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    inds = np.random.randint(len(flat_samples), size=100)

    #the folloing line computes the highest probable result (peak of the posterior distribution) instead of computing the mean of the distribution
    best_walker = np.argmax(np.max(sampler.lnprobability,axis=1))
    best_params = samples[-1][np.argmax(sampler.lnprobability.T[best_walker])]
    if plot_walkers:
        f, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = [r"$\alpha$", r"$b_0$", r"$k_0$", r"$p$"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            # ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step number");
        plt.show()
    if plot_corners:
        labels = [r"$\alpha$", r"$b_0$", r"$k_0$", r"$p$"]
        fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
        plt.show()

    if plot_best_fit_sample:
        f, ax = plt.subplots(figsize=(6, 4))
        for ind in inds:
            sample = flat_samples[ind]
            ax.plot(x, (sample[1] / (1 + (x / sample[2])) ** sample[0]), alpha=0.05, color='red')
        ax.errorbar(x, y, yerr=(sample[3] / errs), linestyle='None', capsize=4,
                    marker='o')  # yerr = bmz_errors*sample[2]
        # plt.plot(kvalues,y_plot_fit, label = 'Battaglia fit') #this lines plots the Battaglia best fit value
        ax.set_xlim(0.05, 10)
        ax.set_ylabel(r'$b_{mz}$ ')
        ax.set_xlabel(r'$k[Mpc⁻1 h]$')
        plt.title(r'$b_{zm}$ as a function of k ')
        plt.legend()
        plt.loglog()
        plt.show()
    if data_dict != None:
        if data_dict == {}:
            if varying_input != 'None': data_dict = {'Z_re': [], '{}'.format(varying_input): [], "medians": [], "a16":[], "a50":[], "a84":[], "b16":[], "b50":[], "b84":[], "k16":[], "k50":[], "k84":[], "p16":[], "p50":[], "p84":[], "width50":[],"width90":[]}
            else: data_dict = {'Z_re': [], "medians": [], "a16":[], "a50":[], "a84":[], "b16":[], "b50":[], "b84":[], "k16":[], "k50":[], "k84":[], "p16":[], "p50":[], "p84":[], "width50":[],"width90":[]}


        data_dict['medians'].append(best_params)
        if varying_input != 'None': data_dict['{}'.format(varying_input)].append(varying_in_value)
        data_dict['a16'].append(corner.quantile(flat_samples[:, 0], [0.16]))
        data_dict['a50'].append(corner.quantile(flat_samples[:, 0], [0.5]))
        data_dict['a84'].append(corner.quantile(flat_samples[:, 0], [0.84]))
        data_dict['b16'].append(corner.quantile(flat_samples[:, 1], [0.16]))
        data_dict['b50'].append(corner.quantile(flat_samples[:, 1], [0.5]))
        data_dict['b84'].append(corner.quantile(flat_samples[:, 1], [0.84]))
        data_dict['k16'].append(corner.quantile(flat_samples[:, 2], [0.16]))
        data_dict['k50'].append(corner.quantile(flat_samples[:, 2], [0.5]))
        data_dict['k84'].append(corner.quantile(flat_samples[:, 2], [0.84]))
        data_dict['p16'].append(corner.quantile(flat_samples[:, 3], [0.16]))
        data_dict['p50'].append(corner.quantile(flat_samples[:, 3], [0.5]))
        data_dict['p84'].append(corner.quantile(flat_samples[:, 3], [0.84]))
        data_dict['Z_re'].append(zre_mean)
        return data_dict
    return best_params


def generate_bias(zre_range, initial_conditions, box_dim, box_len, astro_params, flag_options, density_field = None, z_re_box = None,varying_input = 'None', varying_in_value = 0, data_dict ={}, nb_bins = 20, logbins = True, comp_width = True, comp_ion_hist = False, comp_zre_PP = False, comp_bt = False, return_zre = False, return_density = False, plot_best_fit = False, plot_corner = False):
    '''
    This function generates the linear bias from the power spectrum from a set of data.
    :param zre_range: [1D arr] the range of redshift on which to compute the redshfit of reionization field  over
    :param initial_conditions: [obj] the 21cmFAST initial conditions objects
    :param astro_params: [obj] the astro input parameter of 21cmFASt
    :param flag_options: [obj] the flag options input parameter of 21cmFASt
    :param box_dim: [int] the spatial dimension of the generated fields The units of this box side will define the units of the returned parameters (can be for example Mpc or Mpc/h)
    :param density_field: [3D array] The density field used for the bias computation. None computes and uses 21cmFAST density field (default None)
    :param z_re_box: [3D array] The redshift of reionization field used for the bias computation. None computes and uses 21cmFAST density field (default None)
    :param varying_input: [string] the varied 21cmFAST input (or inputs)
    :param varying_in_value: [float] the varying 21cmFAST input value
    :param data_dict: [dict] the dictionnary in which to stroe the value of the free parameters
    :param nb_bins: [int] the number of data points for the power spectrums and the bias
    :param logbins: [bool] using logbins if true
    :param comp_width: [bool] return the width of reionization for 21cmFAST if True
    :param comp_ion_hist: [bool] computes and return 21cmFAST ionization history of True
    :param comp_zre_PP: [boo] computes and return the power spectrum of the redshift of reionization field
    :param return_zre : [bool] return the redshift of reionization field if True False
    :param return_density: [bool] return the density field if True
    :return: b_mz: the computed linear bias (and the ionization history if applicable)
    :param plot_best_fit: [bool] Will plot the best fitted paramters over the computed bias if True (default True)
    :param plot_corner: [bool] Will plot the posterior distribution of the best fitted parameters if True (default True)
    '''

    #if not 21cmFASt, import redshift of reionization field

    #generate the redshift of reionization field in 21cmFAST
    if z_re_box is None:
        z_re_box = zre.generate_zre_field(zre_range, initial_conditions, box_dim, astro_params, flag_options,
                                      comP_ionization_rate=False, comp_brightness_temp=False)

    overzre, zre_mean = zre.over_zre_field(z_re_box)

    #zre.plot_zre_slice(overzre)
    """This section uses computes the ionization history from the redhsift of reionization field if applicable"""
    if comp_ion_hist:
        redshifts = zre_range
        if comp_width: cmFast_hist, width_50_21, width_90_21 = pp.reionization_history(redshifts, z_re_box, comp_width=comp_width)
        else: cmFast_hist = pp.reionization_history(redshifts, z_re_box, comp_width=comp_width)
    # pp.plot_21zreion_ionhist([reion_hist_zreion,cmFast_hist, reion_hist_zreion_0593])
    # ionization_rates.append(cmFast_hist)

    nb_bins = int(nb_bins)
    if density_field is None:
        perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes=initial_conditions)
        density_field = perturbed_field.density


    #[1:] is because the first result given by powerbox is always Nan
    #compute the power spectrums of the respective fields
    zre_pp = pbox.get_power(overzre,box_len, bins = nb_bins,log_bins=True)[0][1:]
    den_pp = pbox.get_power(density_field, box_len, bins = nb_bins,log_bins=True)[0][1:]
    #compute the linear bias as the sqrt of the power spectrum ratio
    b_mz = np.sqrt(np.divide(zre_pp, den_pp))

    #This section computes the cross correlation for error weighting in the MCMC
    cross_cor_pbox = pbox.get_power(overzre, box_len, bins=nb_bins, log_bins=True, deltax2=density_field)# equivalent to: cross_cor_pbox2 = pbox.get_power(density_field, box_dim, bins = nb_bins, log_bins = True, deltax2 = overzre)

    #compute the cross correlation ration r_mz that is used to weight the errors as presented in the Battaglia model
    r_mz= np.divide(np.array(cross_cor_pbox[0][1:]),
                          np.sqrt((np.array(zre_pp) * np.array(den_pp))))

    # This computes the values of k for which the power spectrums are computed and a function of (one again discarding first Nan
    kbins_zre = pbox.get_power(density_field, box_len, bins=nb_bins, log_bins=True)[1][1:]


    '''
    MCMC analysis and posterior distribution on the b_mz 
    '''


    # no b_mz fitting
    k_values = kbins_zre

    data_dict = run_MCMC_free_params(kbins_zre, b_mz, r_mz, zre_mean, data_dict=data_dict,
                                        varying_input=varying_input, varying_in_value=varying_in_value, plot_corners=plot_corner, plot_best_fit_sample=plot_best_fit)
    # if comp_width:
    #     data_dict['width50'].append(width_50_21)
    #     data_dict['width90'].append(width_90_21)

    if comp_zre_PP:
        return z_re_box, b_mz, k_values, data_dict, density_field, cmFast_hist, zre_pp, den_pp
    elif not return_density and not return_zre:
        return data_dict
    elif not return_density and return_zre:
        return data_dict, z_re_box
    elif not return_zre and return_density:
        return data_dict, density_field
    elif return_density and return_zre:
        return data_dict, density_field, z_re_box

    # elif comp_ion_hist and not comp_zre_PP:
    #     return b_mz, k_values, data_dict, density_field, cmFast_hist
    # elif comp_ion_hist and comp_zre_PP and comp_bt and not return_zre:
    #     return b_mz, k_values, data_dict, density_field, cmFast_hist, zre_pp, den_pp, b_temp_ps, z_4_bt
    # elif comp_ion_hist and comp_zre_PP and comp_bt and return_zre:
    #     return z_re_box, b_mz, k_values, data_dict, density_field, cmFast_hist, zre_pp, den_pp, b_temp_ps, z_4_bt
    # elif comp_ion_hist and comp_zre_PP and not comp_bt and return_zre:
    #     return z_re_box, b_mz, k_values, data_dict, density_field, cmFast_hist, zre_pp, den_pp
    #
    else:
        return data_dict


    # print(ionization_rates)

def run_MCMC_free_params_nob(x,y,errs, zre_mean, num_iter = 5000, nwalkers = 32, plot_walkers = False, plot_corners = False, plot_best_fit_sample = False, data_dict= None, varying_input = 'Heff', varying_in_value = 0 ):
    '''
    This function runs the MCMC
    :param x: the kbins fitting for
    :param y: the b_mz scatter point fitting for
    :param errs: the error (associated with the cross corelation error weighting)
    :param num_iter: the number of iteration to run the Markov Chain Monte Carlo (default 5000)
    :param nwalkers: the number of walker completing MCMC (default 32)
    :param plot_walkers: plot the walker behaviour plot
    :param plot_corners: plot the corner plots for the posterior distributions,
    :param plot_best_fit_sample: plot the b_mz and the associated
    :param dict_storing: to store the best-fitted value and confidence interval in the provided dictionnary (default none)
    :param varying_input: the name of the varying input the value are computed over (default is Heff)
    :param varying_in_value: the value of the varying parameter
    :return: the values of the posterior distribution for the free parameters (mean value)
    '''

    ndim = 3
    #initial parameters for 4 dim
    initial_pos = np.ones((32,3))
    initial_pos[:,0] *= 0.8 +0.06 * np.random.randn(nwalkers)
    initial_pos[:,1] *= 0.4 + 0.1 * np.random.randn(nwalkers)
    initial_pos[:,2] *= 0.01 + 0.01 * np.random.randn(nwalkers)



    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_bmz_errs, args=(x, y, errs))
    sampler.run_mcmc(initial_pos, num_iter)#, progress=True
    samples = sampler.get_chain()

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    inds = np.random.randint(len(flat_samples), size=100)

    #the folloing line computes the highest probable result (peak of the posterior distribution) instead of computing the mean of the distribution
    best_walker = np.argmax(np.max(sampler.lnprobability,axis=1))
    best_params = samples[-1][np.argmax(sampler.lnprobability.T[best_walker])]
    if plot_walkers:
        f, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = [r"$\alpha$", r"$k_0$", r"$p$"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            # ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step number");
        plt.show()
    if plot_corners:
        labels = [r"$\alpha$", r"$k_0$", r"$p$"]
        fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
        plt.show()

    if plot_best_fit_sample:
        f, ax = plt.subplots(figsize=(6, 4))
        for ind in inds:
            sample = flat_samples[ind]
            ax.plot(x0, (sample[1] / (1 + (x0 / sample[2])) ** sample[0]), alpha=0.05, color='red')
        ax.errorbar(kbins_zre, b_mz, yerr=(sample[3] / (cross_cor)), linestyle='None', capsize=4,
                    marker='o')  # yerr = bmz_errors*sample[2]
        # plt.plot(kvalues,y_plot_fit, label = 'Battaglia fit') #this lines plots the Battaglia best fit value
        ax.set_xlim(0.01, 1.2)
        ax.set_ylabel(r'$b_{mz}$ ')
        ax.set_xlabel(r'$k[Mpc⁻1 h]$')
        plt.title(r'$b_{zm}$ as a function of k ')
        plt.legend()
        plt.loglog()
        plt.show()
    if data_dict != None:
        if data_dict == {}: data_dict = {'Z_re': [], '{}'.format(varying_input): [], "medians": [], "a16":[], "a50":[], "a84":[], "b16":[], "b50":[], "b84":[], "k16":[], "k50":[], "k84":[], "p16":[], "p50":[], "p84":[], "width50":[],"width90":[]}
        data_dict['medians'].append(best_params)
        data_dict['{}'.format(varying_input)].append(varying_in_value)
        data_dict['a16'].append(corner.quantile(flat_samples[:, 0], [0.16]))
        data_dict['a50'].append(corner.quantile(flat_samples[:, 0], [0.5]))
        data_dict['a84'].append(corner.quantile(flat_samples[:, 0], [0.84]))
        data_dict['k16'].append(corner.quantile(flat_samples[:, 1], [0.16]))
        data_dict['k50'].append(corner.quantile(flat_samples[:, 1], [0.5]))
        data_dict['k84'].append(corner.quantile(flat_samples[:, 1], [0.84]))
        data_dict['p16'].append(corner.quantile(flat_samples[:, 2], [0.16]))
        data_dict['p50'].append(corner.quantile(flat_samples[:, 2], [0.5]))
        data_dict['p84'].append(corner.quantile(flat_samples[:, 2], [0.84]))
        data_dict['Z_re'].append(zre_mean)
        return data_dict
    return best_params