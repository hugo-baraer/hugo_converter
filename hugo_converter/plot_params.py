"""
plot_params.py

Author: Hugo Baraer
  Supervision by : Prof. Adrian Liu
Affilitation : Cosmic dawwn group at Mcgill University

This module is used for the comparison between z-reion and 21cmFAST, and deals with redshift of reionization fields.
It also includes the general plotting functions for the z-reion parameters variability range over 21cmFAST inputs (with dict or Json data)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from numpy import array
from statistical_analysis import *
from z_re_field import *
import statistical_analysis as sa
import scipy

#For testing purposes, these two modules can be installed:
# import Gaussian_testing as gauss
# from FFT import *


# {'Z_re': [8.01], 'Heff': [30.0], 'medians': [array([2.08873889, 0.97190616, 2.00851018, 0.02540628])], 'a16': [array([0.99442916])], 'a50': [array([1.41932357])], 'a84': [array([2.16887687])], 'b16': [array([0.97635207])], 'b50': [array([0.99655587])], 'b84': [array([1.03807445])], 'k16': [array([0.66037472])], 'k50': [array([1.15658818])], 'k84': [array([2.09772472])], 'p16': [array([0.02413837])], 'p50': [array([0.02966204])], 'p84': [array([0.03743525])]}

def reionization_history(redshifts, field, resolution=143, plot=False, comp_width=False):
    '''
    This function computes the reionization history of a given redshift of reionization field.
    :param redshifts: the redshift range to look the history over (1D array)
    :param field: the 3d redshift of reionization field
    :param resolution: the resolution of the observed field
    :param plot: to plot or not the histories (True default)
    :return: the ionization history at each given redshfit (1D list)
    '''
    ionization_rate = []
    dummy = 0
    if comp_width:
        for i in tqdm(redshifts, 'computing the reionization history', position=0, leave=True):
            new_box = np.zeros((resolution, resolution, resolution))
            new_box[field <= i] = 1
            ionization_rate.append((new_box == 0).sum() / (resolution) ** 3)
            # if (new_box == 0).sum() / (resolution) ** 3 < 0.5 and dummy == 0:
            #     dummy += 1
            #     print(i)
            if ((new_box == 0).sum() / (resolution) ** 3) < 0.95 and dummy == 0:
                dummy += 1
                lower_bound5 = i
            if (new_box == 0).sum() / (resolution) ** 3 < 0.75 and dummy == 1:
                dummy += 1
                lower_bound25 = i
            if (new_box == 0).sum() / (resolution) ** 3 < 0.25 and dummy == 2:
                dummy += 1
                upper_bound25 = i
            if (new_box == 0).sum() / (resolution) ** 3 < 0.05 and dummy == 3:
                dummy += 1
                upper_bound5 = i
        width50 = upper_bound25 - lower_bound25
        width90 = upper_bound5 - lower_bound5

    else:
        for i in tqdm(redshifts, 'computing the reionization history'):
            new_box = np.zeros((resolution, resolution, resolution))
            new_box[field <= i] = 1
            ionization_rate.append((new_box == 0).sum() / (resolution) ** 3)

    if plot:
        fig, ax = plt.subplots()
        # plt.scatter(redshifts,ionization_rate)
        plt.plot(redshifts, ionization_rate)
        ax.set_xlabel(r'z ')
        ax.set_ylabel(r'$x_i(z)$')
        plt.legend()
        plt.title(r'ionization fraction as function of redshift')
        plt.show()
    if comp_width:
        return ionization_rate, width50, width90

    else:
        return ionization_rate


def ionization_map_gen(redshift, resolution, field, plot=False):
    '''
    This function computes the ionization maps at a given redshift from a given redshfit of reionization field
    :param redshift: the redshift at which to  compute the ionizaiton maps
    :param resolution: the resolution of the fields
    :param field: the redshift of reionization map to substract from
    :param plot: plot if True
    :return:  the 3D ionization map
    '''
    new_box1 = np.zeros((resolution, resolution, resolution))
    new_box1[field <= redshift] = 1
    if plot:
        fig, ax = plt.subplots()
        if resolution % 2:
            position_vec = np.linspace(-int((resolution * (100 / 143)) // 2) - 1, int(resolution * (100 / 143) // 2),
                                       resolution)
        else:
            position_vec = np.linspace(-int((resolution * (100 / 143)) // 2), int(resolution * (100 / 143) // 2),
                                       resolution)
        X, Y = np.meshgrid(position_vec, position_vec)
        plt.contourf(X, Y, new_box1[int(resolution // 2)], cmap='Blues')
        ax.set_xlabel(r'$[\frac{Mpc}{\hbar}]$')
        ax.set_ylabel(r'$[\frac{Mpc}{\hbar}]$')
        plt.show()
    return new_box1


def compute_tau(ion_hist, redshifts=np.linspace(5, 18, 60)):
    '''
    This function computes the TAU parameter from the ionization history by performing a simple simpson intergral
    :param ion_hist: the ionization history to comute the integral over
    :param redshifts: [arr] 1D the redshift range [default is 5-15]
    :return: the TAU value
    '''

    return (8 * np.pi / 3) * (
            ((np.e ** 2) / (scipy.constants.m_e * (scipy.constants.c ** 2))) ** 2) * scipy.integrate.simps(ion_hist,
                                                                                                           x=redshifts)


def ionization_map_diff(redshift, resolution, field1, field2, plot=True):
    '''
    This functions computes the ionization map differences and plots it
    :param redshift: the redshift at which to  compute the ionizaiton maps
    :param resolution: the resolution of the fields
    :param field1: the redshift of reionization map to substract from
    :param field2: the redshift of reionization to be substracted
    :param plot: plot if True
    :return:  the 3D field difference
    '''
    # for the first field
    new_box1 = np.zeros((resolution, resolution, resolution))
    new_box1[field1 <= redshift] = 1
    # for the second field
    new_box2 = np.zeros((resolution, resolution, resolution))
    new_box2[field2 <= redshift] = 1
    diff_box = new_box1 - new_box2
    if plot:
        fig, ax = plt.subplots()
        if resolution % 2:
            position_vec = np.linspace(-int((resolution * (100 / 143)) // 2) - 1, int(resolution * (100 / 143) // 2),
                                       resolution)
        else:
            position_vec = np.linspace(-int((resolution * (100 / 143)) // 2), int(resolution * (100 / 143) // 2),
                                       resolution)
        X, Y = np.meshgrid(position_vec, position_vec)
        plt.contourf(X, Y, diff_box[int(resolution // 2)], cmap='RdBu', vmin=-1.0, vmax=1.0)
        ax.set_xlabel(r'$[\frac{Mpc}{\hbar}]$')
        ax.set_ylabel(r'$[\frac{Mpc}{\hbar}]$')
        plt.show()
    return diff_box


def ps_ion_map(map, nb_bins, resolution, delta=1, plot=False, logbins=False):
    '''
    This module computes the power spectrum of an ionization maps
    :param map: the ionization map to compute the power spectrum of
    :param resolution: the resolution of the fields
    :param nb_bins: the thickness of the shells to average the field over (see average k in statistuical tools for more)
    :param delta: (see fft description)
    :return: the power spewctrum as a function of k (1d array)
    '''
    cx = int(resolution // 2)
    Xr, Yr, field_fft, freq_field = compute_fft(map, delta, resolution)
    field_fft = np.square(abs(field_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(143, d=1))
    # kvalues = np.linspace(0, np.sqrt(3 * (freq_field) ** 2), num=int(np.sqrt(3 * (cx) ** 2) / int(radius_thick)))
    # kvalues = kvalues[1:]
    # compute the  average k values in Fourier space to compute the power spectrum
    # values, count= average_overk(resolution, field_fft, radius_thick)
    values, count = sa.average_overk(resolution, field_fft, nb_bins, logbins=logbins)
    field_fft_k = np.divide(values, count)
    if plot:
        fig, ax = plt.subplots()
        plt.scatter(kvalues, field_fft_k)
        ax.set_xlabel(r'$ k [\frac{\hbar}{Mpc}]$')
        ax.set_ylabel(r'Power spectrum')
        plt.show()
    return field_fft_k


def ionization_movie(redshifts, field, resolution, movie_name):
    '''
    This function creates and saves in the current folder a gif showing the evolution of the reionization process
    :param redshifts: the redshift range in which to compute the movie
    :type redshifts: 1D array
    :param field: the redshift of reionization field to compute the movie on
    :type field: 3D array
    :param resolution: the size of the field
    :param movie_name: the name and path (relative or absolute) of the movie
    :type movie_name: string
    :return:
    :rtype:
    '''
    os.mkdir('./ionization_map')
    if resolution % 2:
        position_vec = np.linspace(-int((resolution * (100 / 143)) // 2) - 1, int(resolution * (100 / 143) // 2),
                                   resolution)
    else:
        position_vec = np.linspace(-int((resolution * (100 / 143)) // 2), int(resolution * (100 / 143) // 2),
                                   resolution)
    filenames = []
    Xd, Yd = np.meshgrid(position_vec, position_vec)
    for i in tqdm(redshifts, 'creating the reionization movie'):
        new_box = np.zeros((resolution, resolution, resolution))
        new_box[field <= i] = 1
    fig, ax = plt.subplots()
    plt.contourf(Xd, Yd, new_box[int(resolution // 2)], cmap='Blues', vmin=0, vmax=1.0)
    plt.title(r'slice of the ionization field at a redshift of {} '.format(i))
    plt.colorbar()
    ax.set_xlabel(r'$[\frac{Mpc}{\hbar}]$')
    ax.set_ylabel(r'$[\frac{Mpc}{\hbar}]$')
    plt.savefig('./ionization_map/ionization_field{}.png'.format(i))
    filenames.append('./ionization_map/ionization_field{}.png'.format(i))

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(movie_name, images)


# b = np.load('zreion_for_Hugo.npy')


def plot_ionmap_diff_movie(zre1, zre2, resolution=143, movie_name='ion_map_diff_james.gif'):
    position_vec = np.linspace(-49, 50, resolution)
    Xd, Yd = np.meshgrid(position_vec, position_vec)
    filenames = []
    redshifts = np.linspace(5, 15, 50)
    for i in tqdm(redshifts, 'Making the ionization map movie diff'):
        fig, ax = plt.subplots()
        plt.contourf(Xd, Yd, pp.ionization_map_diff(i, 143, zre1, zre2, plot=False)[int(143 // 2)], cmap='RdBu',
                     vmin=-1.0, vmax=1.0)
        plt.title(r'slice of the ionization map differences at a redshift of {} '.format(i))
        plt.colorbar()
        ax.set_xlabel(r'$[\frac{Mpc}{\hbar}]$')
        ax.set_ylabel(r'$[\frac{Mpc}{\hbar}]$')
        plt.savefig('./ionization_map/ionization_field{}.png'.format(i))
        filenames.append('./ionization_map/ionization_field{}.png'.format(i))
        plt.close()
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(movie_name, images)
    ionization_map_diff = pp.ionization_map_diff(8.265, 143, z_re, z_reion_zre_9)


# over_zre, zre_mean = over_zre_field(b[0])


# redshifts = np.linspace(6.85,7.95,200)
# redshifts = np.linspace(6,9,4)
# redshifts = [3,4,5,5.5,6,6.2,6.4,6.6,6.8,7.0,7.05, 7.1,7.15,7.2,7.25, 7.3,7.35,7.4,7.45,7.5,7.55,7.6,7.65,7.7,7.75,7.8,7.85, 7.9,7.95,8.0,8.05,8.1,8.15,8.2,8.25,8.3,8.35,8.4,8.45,8.5,8.55,8.6,8.65,8.7,8.75,8.8,8.85,8.9,8.95,9.0,9.05,9.1,9.15,9.2,9.25,9.3,9.35,9.4,9.45,9.5,9.55,9.6,9.65,9.7,9.75,9.8,9.85,9.9,9.95,10.0,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.9,12.0,12.2,12.4,12.6,12.8,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.5,16.5,17,17.5,18,18.5,19,19.5,20,20.5]
# redshifts = np.linspace(3.8,16,1000)

def plot_variational_range_1dict(dict1, varying_name='Heff'):
    '''
    This function generates the plot of the variational range for 3 parameters with 1 different input
    :param dict1: the dictionnary of the first input
    :type dict1: dict
    :return:
    :rtype:
    '''
    a = dict1
    fig3, ax3 = plt.subplots(3, 1, sharex='col', sharey='row')
    cont300 = ax3[0].errorbar(a['{}'.format(varying_name)], np.concatenate(a['a50']), yerr=(
        np.concatenate(a['a50']) - np.concatenate(a['a16']), np.concatenate(a['a84']) - np.concatenate(a['a50'])),
                              ls='')
    cont300 = ax3[0].scatter(a['{}'.format(varying_name)], np.concatenate(a['a50']), color='r')

    ax3[1].errorbar(a['{}'.format(varying_name)], np.concatenate(a['b50']), yerr=(
        np.concatenate(a['b50']) - np.concatenate(a['b16']), np.concatenate(a['b84']) - np.concatenate(a['b50'])),
                    ls='')
    ax3[1].scatter(a['{}'.format(varying_name)], np.concatenate(a['b50']), color='r')

    ax3[2].errorbar(a['{}'.format(varying_name)], np.concatenate(a['k50']), yerr=(
        np.concatenate(a['k50']) - np.concatenate(a['k16']), np.concatenate(a['k84']) - np.concatenate(a['k50'])),
                    ls='')
    ax3[2].scatter(a['{}'.format(varying_name)], np.concatenate(a['k50']), color='r')

    plt.setp(ax3[2], xlabel='{}'.format(varying_name))
    plt.setp(ax3[0], ylabel=r'$\alpha$')
    plt.setp(ax3[1], ylabel=r'$b_0$')
    plt.setp(ax3[2], ylabel=r'$k_0$')
    plt.show()


def plot_variational_range(dict1, dict2):
    '''
    This function generates the plot of the variational range for 3 parameters with 2 different inputs
    :param dict1: the dictionnary of the first input
    :type dict1: dict
    :param dict2:the dictionnary of the first input
    :type dict2: dict
    :return:
    :rtype:
    '''
    fig3, ax3 = plt.subplots(3, 2, sharex='col', sharey='row')

    cont300 = ax3[0, 0].errorbar(a['Heff'], np.concatenate(a['a50']), yerr=(
        np.concatenate(a['a50']) - np.concatenate(a['a16']), np.concatenate(a['a84']) - np.concatenate(a['a50'])),
                                 ls='')
    cont300 = ax3[0, 0].scatter(a['Heff'], np.concatenate(a['a50']), color='r')
    ax3[0, 1].errorbar(b['Heff'], np.concatenate(b['a50']), yerr=(
        np.concatenate(b['a50']) - np.concatenate(b['a16']), np.concatenate(b['a84']) - np.concatenate(b['a50'])),
                       ls='')
    ax3[0, 1].scatter(b['Heff'], np.concatenate(b['a50']), color='r')
    ax3[1, 0].errorbar(a['Heff'], np.concatenate(a['b50']), yerr=(
        np.concatenate(a['b50']) - np.concatenate(a['b16']), np.concatenate(a['b84']) - np.concatenate(a['b50'])),
                       ls='')
    ax3[1, 0].scatter(a['Heff'], np.concatenate(a['b50']), color='r')
    ax3[1, 1].errorbar(b['Heff'][1:], np.concatenate(b['b50'][1:]), yerr=(
        np.concatenate(b['b50'][1:]) - np.concatenate(b['b16'][1:]),
        np.concatenate(b['b84'][1:]) - np.concatenate(b['b50'][1:])), ls='')
    ax3[1, 1].scatter(b['Heff'][1:], np.concatenate(b['b50'][1:]), color='r')
    ax3[2, 0].errorbar(a['Heff'], np.concatenate(a['k50']), yerr=(
        np.concatenate(a['k50']) - np.concatenate(a['k16']), np.concatenate(a['k84']) - np.concatenate(a['k50'])),
                       ls='')
    ax3[2, 0].scatter(a['Heff'], np.concatenate(a['k50']), color='r')
    ax3[2, 1].errorbar(b['Heff'][1:], np.concatenate(b['k50'][1:]), yerr=(
        np.concatenate(b['k50'][1:]) - np.concatenate(b['k16'][1:]),
        np.concatenate(b['k84'][1:]) - np.concatenate(b['k50'][1:])), ls='')
    ax3[2, 1].scatter(b['Heff'][1:], np.concatenate(b['k50'][1:]), color='r')

    plt.setp(ax3[2, 1], xlabel='Ionization efficiency')
    plt.setp(ax3[2, 0], xlabel=r'turnover mass $[log_{10}(M_\odot)]$')
    plt.setp(ax3[0, 0], ylabel=r'$\alpha$')
    plt.setp(ax3[1, 0], ylabel=r'$b_0$')
    plt.setp(ax3[2, 0], ylabel=r'$k_0$')


# a = {'Z_re': [9.396701054337125, 9.283932019860426, 9.144286981051614, 8.972942749948961, 8.826463379644464, 8.671627213805316, 8.516525676875816, 8.312715207917908, 8.141544015180868, 7.9694956615588435, 7.79832344290264, 7.588524683786066, 7.403577790491576, 7.2403290875098785, 7.058309483562552], 'Heff': [7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9], 'medians': [array([0.91816496, 1.12202245, 0.04815797, 0.01374211]), array([0.96192428, 1.04661435, 0.05935293, 0.01247621]), array([0.89825585, 1.0765973 , 0.05143589, 0.01438904]), array([1.03012121, 1.02405853, 0.06881685, 0.015596  ]), array([0.88629944, 1.13438821, 0.04543851, 0.01305808]), array([0.95900476, 1.11530534, 0.0534367 , 0.016161  ]), array([0.90162736, 1.08473575, 0.05031027, 0.01306996]), array([0.88414362, 1.15229063, 0.04379291, 0.01467261]), array([0.88213479, 1.14561677, 0.04411855, 0.01329872]), array([0.82354619, 1.21576482, 0.03460948, 0.01180405]), array([0.83382018, 1.20642103, 0.03633014, 0.01411022]), array([0.90223026, 1.15273858, 0.04442337, 0.01663044]), array([0.84768529, 1.21038519, 0.03609344, 0.01420787]), array([0.80365195, 1.24236022, 0.03150117, 0.01444418]), array([0.74846812, 1.35424301, 0.02439833, 0.01742645])], 'a16': [array([0.91919148]), array([0.89086697]), array([0.88277788]), array([0.88939004]), array([0.86152989]), array([0.8569324]), array([0.82876063]), array([0.83994386]), array([0.81199034]), array([0.81207846]), array([0.79190236]), array([0.7939233]), array([0.79852566]), array([0.7708031]), array([0.7537857])], 'a50': [array([0.96659641]), array([0.9369144]), array([0.9295798]), array([0.94097457]), array([0.91115179]), array([0.89929847]), array([0.86900772]), array([0.87923843]), array([0.85046268]), array([0.85020293]), array([0.83164505]), array([0.83376142]), array([0.83589763]), array([0.80801086]), array([0.79048197])], 'a84': [array([1.01805373]), array([0.98676431]), array([0.97927361]), array([0.99389579]), array([0.96332351]), array([0.94512214]), array([0.91197209]), array([0.92094286]), array([0.89156996]), array([0.89206759]), array([0.87489933]), array([0.87766681]), array([0.87701201]), array([0.84734869]), array([0.8306958])], 'b16': [array([1.03991607]), array([1.04495614]), array([1.05272738]), array([1.05604692]), array([1.06221267]), array([1.07511596]), array([1.09533229]), array([1.11195655]), array([1.12533558]), array([1.12087095]), array([1.13191103]), array([1.15798943]), array([1.17517767]), array([1.19813574]), array([1.23811022])], 'b50': [array([1.06657393]), array([1.07313059]), array([1.08224489]), array([1.0865506]), array([1.09547679]), array([1.10852238]), array([1.13177861]), array([1.14905534]), array([1.16645824]), array([1.16344063]), array([1.17921667]), array([1.20946734]), array([1.22794956]), array([1.25895816]), array([1.30834604])], 'b84': [array([1.0963547]), array([1.10635709]), array([1.1174497]), array([1.12454827]), array([1.13743487]), array([1.14694078]), array([1.17380573]), array([1.18916762]), array([1.2128002]), array([1.21239595]), array([1.23508594]), array([1.26979179]), array([1.28977907]), array([1.33054567]), array([1.39319373])], 'k16': [array([0.05103315]), array([0.0481762]), array([0.04703823]), array([0.0471818]), array([0.04391929]), array([0.04251908]), array([0.03917322]), array([0.03900833]), array([0.03566063]), array([0.03472404]), array([0.03235533]), array([0.03092383]), array([0.02957774]), array([0.026396]), array([0.02380858])], 'k50': [array([0.0575195]), array([0.0544741]), array([0.05358098]), array([0.05418379]), array([0.0507811]), array([0.04819353]), array([0.04451395]), array([0.04409987]), array([0.04042883]), array([0.03959684]), array([0.03740933]), array([0.03579055]), array([0.03405229]), array([0.03070715]), array([0.02795813])], 'k84': [array([0.06481614]), array([0.06158762]), array([0.0607405]), array([0.06152289]), array([0.05813161]), array([0.05461512]), array([0.05042011]), array([0.04968332]), array([0.04596644]), array([0.04517002]), array([0.04290505]), array([0.04129801]), array([0.03907767]), array([0.03543515]), array([0.03258164])], 'p16': [array([0.01234072]), array([0.01242572]), array([0.01264166]), array([0.01301108]), array([0.01326862]), array([0.01270141]), array([0.01260699]), array([0.01236695]), array([0.01274681]), array([0.01296933]), array([0.01356192]), array([0.0140807]), array([0.01361917]), array([0.01368417]), array([0.01412747])], 'p50': [array([0.01352539]), array([0.01359935]), array([0.01389093]), array([0.01427796]), array([0.01456819]), array([0.01391331]), array([0.01381404]), array([0.01353384]), array([0.0139517]), array([0.01421298]), array([0.01487084]), array([0.01543753]), array([0.01494865]), array([0.0150288]), array([0.01546528])], 'p84': [array([0.01487546]), array([0.0149929]), array([0.01527847]), array([0.01595142]), array([0.01615175]), array([0.01529999]), array([0.0151997]), array([0.01493231]), array([0.01535871]), array([0.0156907]), array([0.01636462]), array([0.01698239]), array([0.01645685]), array([0.01658366]), array([0.01703918])]}
# b = {'Z_re': [2.4447038120078366, 4.189303630009777, 5.002679358882596, 5.5369602767519535, 5.974575329311502, 6.314296833295318, 6.585703406085821, 6.80900462928924, 7.030162365386582, 7.2209361375579775, 7.372540316058337, 7.524227935984012, 7.648473244199197, 7.767343419942569, 7.8799575406255435, 7.991388434539689, 8.09683719381015, 8.195645862279928, 8.281522135744837, 8.360399930647864, 8.446752572577797, 8.51515436492697, 8.580785149614922, 8.643948598714113, 8.704875886009438, 8.765059039938008, 8.822138788396307, 8.877833887956632, 8.932503068353233, 8.986066307891337, 9.038525316436218, 9.092207220624259], 'Heff': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96], 'medians': [array([0.67820421, 2.37819193, 0.02012686, 0.0305391 ]), array([0.58069955, 2.28707347, 0.01294444, 0.02443398]), array([0.7089804 , 1.68648271, 0.02335719, 0.02060211]), array([0.71296127, 1.75789146, 0.02120249, 0.01832955]), array([0.74899997, 1.53267205, 0.02775316, 0.02120867]), array([0.83401622, 1.50688849, 0.03152588, 0.01659543]), array([0.76795947, 1.68456615, 0.02306855, 0.02179632]), array([0.89880495, 1.38409356, 0.0371032 , 0.02339746]), array([0.82578002, 1.59057517, 0.02597772, 0.02132738]), array([0.85549434, 1.52185894, 0.02988826, 0.02030037]), array([0.82328785, 1.62216188, 0.02499712, 0.02229914]), array([0.87259237, 1.48032297, 0.03108958, 0.01854296]), array([0.94069335, 1.45764487, 0.03387513, 0.02101927]), array([0.99588922, 1.33268293, 0.04124865, 0.02399893]), array([0.83872133, 1.63173393, 0.02473375, 0.02271535]), array([0.87470033, 1.65745566, 0.0254436 , 0.0221859 ]), array([0.94303178, 1.50470651, 0.03107342, 0.01801905]), array([0.91223206, 1.56925813, 0.02757401, 0.02272781]), array([0.77361904, 1.83765661, 0.01712159, 0.02227504]), array([0.95217855, 1.52761705, 0.03108174, 0.02340588]), array([0.96617067, 1.35456332, 0.03910781, 0.0246104 ]), array([0.93595512, 1.44903951, 0.03242727, 0.02427982]), array([0.85085426, 1.68032536, 0.02229842, 0.02465398]), array([0.94408139, 1.4835386 , 0.03037015, 0.02437015]), array([1.04913846, 1.31080257, 0.04185847, 0.02147918]), array([0.99888562, 1.33270872, 0.03782382, 0.02611196]), array([0.92563991, 1.44408361, 0.03109314, 0.02604089]), array([1.03405673, 1.35013779, 0.03906027, 0.02422579]), array([1.03948884, 1.42329563, 0.03488679, 0.02351002]), array([0.86478725, 1.53721433, 0.02420433, 0.02502088]), array([0.95885833, 1.47910244, 0.02980264, 0.0243835 ]), array([0.97282858, 1.61009335, 0.02586306, 0.02295738])], 'a16': [array([0.66456685]), array([0.65217863]), array([0.67854491]), array([0.71924908]), array([0.71957893]), array([0.73841017]), array([0.74471917]), array([0.77706338]), array([0.78240452]), array([0.78546619]), array([0.80966418]), array([0.79988536]), array([0.82445022]), array([0.84033427]), array([0.85021226]), array([0.85311799]), array([0.86139726]), array([0.86263545]), array([0.87088038]), array([0.88156714]), array([0.8651304]), array([0.87719979]), array([0.8881233]), array([0.90308319]), array([0.91438787]), array([0.91935462]), array([0.92905955]), array([0.93812791]), array([0.93952633]), array([0.94465057]), array([0.95075546]), array([0.94568216])], 'a50': [array([0.68604164]), array([0.68251861]), array([0.7111957]), array([0.75227322]), array([0.75606103]), array([0.77389378]), array([0.78308992]), array([0.81877228]), array([0.82464559]), array([0.82967838]), array([0.85550275]), array([0.84457222]), array([0.87331754]), array([0.88955689]), array([0.90182873]), array([0.90595195]), array([0.91383359]), array([0.91594424]), array([0.92465106]), array([0.93584446]), array([0.92166764]), array([0.93216544]), array([0.94589976]), array([0.96494596]), array([0.97545805]), array([0.98149364]), array([0.99202282]), array([1.00311339]), array([1.00137395]), array([1.01424743]), array([1.01720518]), array([1.01152353])], 'a84': [array([0.71115326]), array([0.71470369]), array([0.74624116]), array([0.78887306]), array([0.79579962]), array([0.81208631]), array([0.82490567]), array([0.86420134]), array([0.87069962]), array([0.87549417]), array([0.9060375]), array([0.8944769]), array([0.92651992]), array([0.94444407]), array([0.96182893]), array([0.96240352]), array([0.97264259]), array([0.97884204]), array([0.9857719]), array([0.99602891]), array([0.98417591]), array([0.99439594]), array([1.01369222]), array([1.03232973]), array([1.0422521]), array([1.05186571]), array([1.06349683]), array([1.07527168]), array([1.07424538]), array([1.09156667]), array([1.08898148]), array([1.0850625])], 'b16': [array([2.251937]), array([1.51136294]), array([1.51655134]), array([1.4942101]), array([1.49323313]), array([1.49262669]), array([1.49446428]), array([1.45785069]), array([1.45422551]), array([1.45729317]), array([1.43435563]), array([1.4510488]), array([1.41906328]), array([1.40329616]), array([1.39489088]), array([1.4052801]), array([1.39686411]), array([1.38779146]), array([1.38559639]), array([1.38505979]), array([1.39266576]), array([1.38562533]), array([1.36772176]), array([1.35050161]), array([1.34825746]), array([1.34365805]), array([1.3374548]), array([1.33737173]), array([1.34696735]), array([1.33539594]), array([1.34203414]), array([1.34442955])], 'b50': [array([2.39212889]), array([1.6219032]), array([1.62646174]), array([1.59425206]), array([1.59954522]), array([1.59372848]), array([1.60393091]), array([1.55933281]), array([1.55713965]), array([1.55749721]), array([1.53176744]), array([1.55539013]), array([1.51653562]), array([1.49828506]), array([1.4941061]), array([1.500155]), array([1.49170209]), array([1.48871588]), array([1.48088367]), array([1.47526997]), array([1.48840028]), array([1.47876501]), array([1.46332957]), array([1.44198567]), array([1.43559926]), array([1.43065616]), array([1.42785534]), array([1.42387726]), array([1.43469826]), array([1.42464794]), array([1.42631556]), array([1.43630206])], 'b84': [array([2.47072812]), array([1.75205421]), array([1.76291059]), array([1.71461701]), array([1.73375291]), array([1.71922766]), array([1.74041174]), array([1.68139007]), array([1.68046532]), array([1.68802655]), array([1.65406403]), array([1.68009527]), array([1.64161795]), array([1.61271215]), array([1.606767]), array([1.61837556]), array([1.60694809]), array([1.60953675]), array([1.59839816]), array([1.58887716]), array([1.60840825]), array([1.59626395]), array([1.58281501]), array([1.55435924]), array([1.54185372]), array([1.53818041]), array([1.53254678]), array([1.52740377]), array([1.53756295]), array([1.53531466]), array([1.53377329]), array([1.54551817])], 'k16': [array([0.01858722]), array([0.02123097]), array([0.02116376]), array([0.02249961]), array([0.02179124]), array([0.02191744]), array([0.02169343]), array([0.02343203]), array([0.02317008]), array([0.02270396]), array([0.02381445]), array([0.02307182]), array([0.02438148]), array([0.02529267]), array([0.02554101]), array([0.02512719]), array([0.02536482]), array([0.02513464]), array([0.02539681]), array([0.02588435]), array([0.02502347]), array([0.0255083]), array([0.02600641]), array([0.02712867]), array([0.02764913]), array([0.02778849]), array([0.02806067]), array([0.02845896]), array([0.02805715]), array([0.02818482]), array([0.0282588]), array([0.02771777])], 'k50': [array([0.01982712]), array([0.02507559]), array([0.02513814]), array([0.02637011]), array([0.02592932]), array([0.02585692]), array([0.02586292]), array([0.02779916]), array([0.02758706]), array([0.02726481]), array([0.02854061]), array([0.02753327]), array([0.02931464]), array([0.03022707]), array([0.0304883]), array([0.03021144]), array([0.03055058]), array([0.03035443]), array([0.03063239]), array([0.03115998]), array([0.03029305]), array([0.03077314]), array([0.03150902]), array([0.03284666]), array([0.03328086]), array([0.03361055]), array([0.03392153]), array([0.03434739]), array([0.03368032]), array([0.03440212]), array([0.03427589]), array([0.03356978])], 'k84': [array([0.02210726]), array([0.02933071]), array([0.02956638]), array([0.0307745]), array([0.0305573]), array([0.03019742]), array([0.03063087]), array([0.03283014]), array([0.0326061]), array([0.0320595]), array([0.03376005]), array([0.03282218]), array([0.03493163]), array([0.03600226]), array([0.03673264]), array([0.03600242]), array([0.03631743]), array([0.03676325]), array([0.03681262]), array([0.03704161]), array([0.03647049]), array([0.03679229]), array([0.03828141]), array([0.03954995]), array([0.03989391]), array([0.04029167]), array([0.04102358]), array([0.04117345]), array([0.04050079]), array([0.04184082]), array([0.04105286]), array([0.04046513])], 'p16': [array([0.02828658]), array([0.01638152]), array([0.0169032]), array([0.01645835]), array([0.01728926]), array([0.01684764]), array([0.01802863]), array([0.01815319]), array([0.01810147]), array([0.0180891]), array([0.0183444]), array([0.01918629]), array([0.01939671]), array([0.01937719]), array([0.0193702]), array([0.01930242]), array([0.01927463]), array([0.0194303]), array([0.01967277]), array([0.01985863]), array([0.02060795]), array([0.02069283]), array([0.020702]), array([0.02078111]), array([0.02074345]), array([0.02081746]), array([0.02090737]), array([0.02083131]), array([0.02088194]), array([0.02098735]), array([0.02099244]), array([0.02115601])], 'p50': [array([0.03102903]), array([0.01792838]), array([0.01846805]), array([0.01805508]), array([0.0189315]), array([0.01849991]), array([0.01975651]), array([0.01990947]), array([0.01982792]), array([0.01985266]), array([0.02007009]), array([0.02102076]), array([0.02119819]), array([0.02116727]), array([0.02122989]), array([0.02114211]), array([0.02111985]), array([0.02129444]), array([0.02149811]), array([0.0217764]), array([0.02253839]), array([0.02267288]), array([0.02279089]), array([0.02277871]), array([0.02272454]), array([0.02281119]), array([0.02290772]), array([0.02288628]), array([0.02292312]), array([0.02303319]), array([0.02301076]), array([0.0230957])], 'p84': [array([0.03433467]), array([0.01973708]), array([0.02034006]), array([0.01992919]), array([0.02087723]), array([0.02035499]), array([0.0217374]), array([0.02195341]), array([0.0218862]), array([0.02182405]), array([0.02207152]), array([0.02317768]), array([0.02334819]), array([0.02333737]), array([0.02350985]), array([0.02327419]), array([0.02324945]), array([0.02369203]), array([0.0236243]), array([0.02394874]), array([0.0247922]), array([0.02493012]), array([0.02512827]), array([0.02515452]), array([0.02497634]), array([0.02514934]), array([0.02524154]), array([0.02523403]), array([0.02524184]), array([0.02545687]), array([0.025385]), array([0.02542237])]}


"""
The following commented code is to plot the vairational range of a single free parameter (alpha k_0 or b_0) as a function of  

fig, ax = plt.subplots()
plt.errorbar(b['Heff'][1:], np.concatenate(b['b50'][1:]), yerr=(
    np.concatenate(b['b50'][1:]) - np.concatenate(b['b16'][1:]),
    np.concatenate(b['b84'][1:]) - np.concatenate(b['b50'][1:])), ls='')
plt.scatter(b['Heff'][1:], np.concatenate(b['b50'][1:]), color='r')
# plt.errorbar(a['Heff'], np.concatenate(a['a50']), yerr=(np.concatenate(a['a50'])-np.concatenate(a['a16']),np.concatenate(a['a84'])-np.concatenate(a['a50'])), ls = '')
#plt.scatter(a['Heff'], np.concatenate(a['a50']), color = 'r')
plt.xlabel('Ionization efficiency')
plt.xlabel(r'turnover mass $[log_{10}(M_\odot)]$')
plt.ylabel(r'$\alpha$ best fitted value')
plt.show()

fig, ax = plt.subplots()
plt.errorbar(a['Heff'], np.concatenate(a['b50']), yerr=(np.concatenate(a['b50'])-np.concatenate(a['b16']),np.concatenate(a['b84'])-np.concatenate(a['b50'])), ls = '')
plt.scatter(a['Heff'], np.concatenate(a['b50']), color = 'r')
#plt.xlabel('Ionization efficiency')
plt.xlabel(r'turnover mass $[log_{10}(M_\odot)]$')
plt.ylabel(r'$b_0$ best fitted value')
plt.show()

fig, ax = plt.subplots()
plt.errorbar(a['Heff'], np.concatenate(a['k50']), yerr=(np.concatenate(a['k50'])-np.concatenate(a['k16']),np.concatenate(a['k84'])-np.concatenate(a['k50'])), ls = '')
plt.scatter(a['Heff'], np.concatenate(a['k50']), color = 'r')
#plt.xlabel('Ionization efficiency')
plt.xlabel(r'turnover mass $[log_{10}(M_\odot)]$')
plt.ylabel(r'$k_0$ best fitted value')
plt.show()

fig, ax = plt.subplots()
plt.scatter(a['Heff'], a['Z_re'], color = 'r')
#plt.xlabel('Ionization efficiency')
plt.xlabel(r'turnover mass $[log_{10}(M_\odot)]$')
plt.ylabel(r'mean redshift of reionization')
plt.show()

fig, ax = plt.subplots()
plt.errorbar(a['Heff'], np.concatenate(a['a50']), yerr=(np.concatenate(a['a50'])-np.concatenate(a['a16']),np.concatenate(a['a84'])-np.concatenate(a['a50'])), ls = '', label = r'$\alpha$', color = 'r')
plt.errorbar(a['Heff'], np.concatenate(a['b50']), yerr=(np.concatenate(a['b50'])-np.concatenate(a['b16']),np.concatenate(a['b84'])-np.concatenate(a['b50'])), ls = '', label = r'$b_0$', color = 'g')
plt.errorbar(a['Heff'], np.concatenate(a['k50']), yerr=(np.concatenate(a['k50'])-np.concatenate(a['k16']),np.concatenate(a['k84'])-np.concatenate(a['k50'])), ls = '', label = r'$k_0$', color = 'b')
plt.scatter(a['Heff'], np.concatenate(a['k50']), color = 'b')
plt.scatter(a['Heff'], np.concatenate(a['b50']), color = 'g')
plt.scatter(a['Heff'], np.concatenate(a['a50']), color = 'r')
plt.xlabel('Ionization efficiency')
plt.ylabel(r'$\alpha$ best fitted value')
plt.legend()
plt.show()
"""


def plot_multiple_ionhist(ion_rates, data_dict, varying_name, zreion=None, zreion2=None):
    fig, ax = plt.subplots()
    for count, ion_rate in enumerate(ion_rates):
        nplot, = plt.plot(np.linspace(5, 18, 100), ion_rate,
                          label='{} = {}'.format(varying_name, data_dict['{}'.format(varying_name)][count]))
    if zreion != None: plt.plot(np.linspace(5, 18, 100), zreion, label='zreion')
    # label = 'M_turn {}'.format(H_eff_zre50['Heff'][count])
    if zreion2 != None: plt.plot(np.linspace(5, 18, 100), zreion2, label='zreion b_0 = 0.593')
    plt.scatter([6.8, 8.0], [0.83, 0.48])
    plt.legend(fontsize='x-small')
    plt.xlabel('redshift')
    plt.ylabel('ionization fraction')
    plt.show()


def plot_21zreion_ionhist(ion_rates, saveforgif=False, filenames=[], imnb=0, title=''):
    fig, ax = plt.subplots()
    # for count, ion_rate in enumerate(ion_rates):
    #     nplot, = plt.plot(np.linspace(5, 15, 100), ion_rate, labels ='{}'.format(labels[count]))
    #     #label = 'M_turn {}'.format(H_eff_zre50['Heff'][count])
    plt.plot(np.linspace(5, 18, 100), ion_rates[0], label='z_reion with me', linewidth=2)
    plt.plot(np.linspace(5, 18, 100), ion_rates[1], linewidth=2, label='21cmFAST')
    if len(ion_rates) == 3: plt.plot(np.linspace(5, 15, 100), ion_rates[2], label='z-reion with James')
    plt.legend(fontsize='small')
    plt.xlabel('redshift')
    plt.ylabel('ionization fraction')
    plt.title(title)
    if saveforgif:
        plt.savefig('./ionization_map/ionization_hist{}.png'.format(imnb))
        filenames.append('./ionization_map/ionization_hist{}.png'.format(imnb))
        plt.close()
        images = []
        return filenames
    else:
        plt.show()
