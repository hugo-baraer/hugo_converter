"""
  /path/to/my/file/%(dir_id)s/is/here/%(file_id)s
 
  Author : Hugo Baraer
  Affiliation : McGill University
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
#import pymks

def average_overk(box_dim,field, radius_thick):
    '''

    :param box_dim:
    :type box_dim:
    :param field:
    :type field:
    :param radius_thick:
    :type radius_thick:
    :return:
    :rtype:
    '''

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
    a, b0, k0, h = theta
    if 1 < a < 25. and 0.1 < b0 < 20 and 0.01 < k0 < 2 and h>0.2:
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
    a, b0, k0, h = theta
    sigma2 = yerr ** 2
    try:
        model = (b0/(1+(x/k0))**a)+h
        likelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        return likelihood
    except:
        return -np.inf



def log_post_bmz(theta, x, y, yerr):
    lp = log_prior_bmz(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_bmz(theta, x, y, yerr)


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