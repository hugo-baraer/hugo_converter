"""
  FFT_overdensity.py

  Author : Hugo Baraer
  Affiliation : McGill University
  Date of creation : 2021-09-21

  This module is the driver and interacts between 21cmFast and the modules
"""

import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#intialize the coeval cube at several redshift
coeval = p21c.run_coeval(redshift=5.0,user_params={'HII_DIM': 100, "USE_INTERPOLATION_TABLES": False})

#plot dark_matter density for testing purposes
plotting.coeval_sliceplot(coeval, kind = 'density')
plt.tight_layout()
plt.title('over-density at a redshfit of {}'.format(coeval.redshift))
plt.show()

#plot the reionization redshift
plotting.coeval_sliceplot(coeval, kind = 'z_re_box', cmap = 'jet')
plt.tight_layout()
plt.title('reionization redshift ')
plt.show()

"""
it appears coeval has a Z_re component, which shows either the redshift entered as a parameter of the
coeval, or -1 if it's a smaller redshift. With these information, I could plot z_re as function of time
"""



def generate_zre_field(max_z, min_z, z_shift, HII_Dim):
    """
    This function generate a z_re field with coeval at different reionization redshift
    :param max_z: [float] the maximum redshift of the plot
    :param min_z: [float] the minimum redshift of the plot
    :param z_shift [float] the desired amount different redshift computed for the plot (redshift resolution)
    :param HII_Dim: [float] the minimum redshift of the plot
    :return: a 3D array of the reionization field
    """
    z_re = np.full(50, -1)
    redshift_range = np.linspace(max_z, min_z, z_shift)
    for redshift in redshift_range:
        coeval = p21c.run_coeval(redshift=redshift,user_params={'HII_DIM': HII_Dim, "USE_INTERPOLATION_TABLES": False})



