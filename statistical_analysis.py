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
    a=1
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