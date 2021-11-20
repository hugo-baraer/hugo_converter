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
    if -10 < a < 10. and -20 < b0 < 20 and -40 < k0 < 20 :
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
    a, b0, k0 = theta
    sigma2 = yerr ** 2
    try:
        model = b0/(1+x/k0)**a
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