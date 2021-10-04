"""
  EoR_research/FFT.py

  Author : Hugo Baraer
  Affiliation : McGill University
  Date of creation : 2021-09-21
  
  This module computes the Fourrier transform of the over-density and the over-redshift
  
"""

import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import z_re_field as zre
from tqdm import tqdm

