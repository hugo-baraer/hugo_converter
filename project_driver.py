"""
  EoR_research/project_driver.py

  Author : Hugo Baraer
  Affiliation : McGill University
  Date of creation : 2021-09-21

  This module is the driver and interacts between 21cmFast and the modules computing the require fields and parameters.
"""

import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import z_re_field as zre
from tqdm import tqdm

#intialize a coeval cube at red shift z = z\bar
coeval = p21c.run_coeval(redshift=8.0,user_params={'HII_DIM': 50, "USE_INTERPOLATION_TABLES": False})


#plot dark_matter density for testing purposes
plotting.coeval_sliceplot(coeval, kind = 'density')
plt.tight_layout()
plt.title('slice of dark matter over-density at a redshfit of {} and a pixel dimension of {}³'.format(coeval.redshift,150)) #coeval.user_params(HII_DIM)
plt.show()

#plot the reionization redshift (test pursposes)
plotting.coeval_sliceplot(coeval, kind = 'z_re_box', cmap = 'jet')
plt.tight_layout()
plt.title('reionization redshift ')
plt.show()

"""
it appears coeval has a Z_re component, which shows if yes or not, the pixel was ionized at that reshift. This means that the pixel value is either
the redshift parameter entred in coeval, or -1 if it wasn't ionized at that redshift. 
With these information, I could plot z_re as function of time, by looking at a bunch of redshifts.
"""

#Compute the reionization redshift from the module z_re
coeval.z_re_box = zre.generate_zre_field(16, 1, 1, coeval.z_re_box.shape[0])
overzre = zre.over_zre_field(coeval.z_re_box)

#plot a slice of this new redshift field, saved as the new z_re_box
plotting.coeval_sliceplot(coeval, kind = 'z_re_box', cmap = 'jet')
plt.tight_layout()
plt.title('reionization redshift field ')
plt.show()

#plot a slice of the over redshift field, saved as the new z_re_box


