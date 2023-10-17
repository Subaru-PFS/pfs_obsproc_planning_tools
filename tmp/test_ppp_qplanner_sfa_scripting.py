from __future__ import print_function

import collections
import glob
import multiprocessing
import os
import random
import sys
import time
from collections import defaultdict
from itertools import chain

import colorcet as cc
import ets_fiber_assigner.netflow as nf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from ics.cobraOps import plotUtils
from ics.cobraOps.Bench import Bench
from ics.cobraOps.cobraConstants import NULL_TARGET_ID, NULL_TARGET_POSITION
from ics.cobraOps.CobrasCalibrationProduct import CobrasCalibrationProduct
from ics.cobraOps.CollisionSimulator import CollisionSimulator
from ics.cobraOps.TargetGroup import TargetGroup
from IPython.display import clear_output
from logzero import logger
from matplotlib import gridspec, rcParams, ticker
from matplotlib.path import Path
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity

np.random.seed(1)

"""Change the directory name if you like.
If you make a new directory, copy /work/kiyoyabe/obsproc/e2e/test/20230615/sample_inuse to the directory
"""

# workDir='/home/kiyoyabe/obsproc/e2e/notebooks/test_ppp+qplan+sfa'
workDir = "/work/monodera/tmp/e2e/test_ppp_qplan_sfa/"
# repoDir='/work/kiyoyabe/obsproc/e2e/src/repo'
repoDir = None

# sys.path.append(os.path.join(workDir, 'utils'))
# sys.path.append(os.path.join(workDir, 'planning'))
from pfs_obsproc_planning import generatePfsDesign

# FIXME: config file must be located in the workDir
config = "config.toml"

#
# Run PPP
#
gpd = generatePfsDesign.GeneratePfsDesign(config, workDir, repoDir)

# n_pccs_l = 40
# n_nccs_m = 10
n_pccs_l = 5
n_nccs_m = 5
# gpd.runPPP(n_pccs_l, n_nccs_m, show_plots=False)

#
# Run qplan
#
obs_dates = ["2023-12-20", "2023-12-21", "2023-12-22"]
gpd.runQPlan(obs_dates)

#
# Run SFA
#
gpd.runSFA(clearOutput=True)
