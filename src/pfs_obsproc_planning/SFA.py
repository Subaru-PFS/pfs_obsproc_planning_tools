#!/usr/bin/env python3
# SFA.py : Subaru Fiber Allocation software

import os
import sys
import warnings

import numpy as np
import pandas as pd
from astropy.time import Time
from logzero import logger

warnings.filterwarnings('ignore')

def run(conf, workDir='.', repoDir='.', clearOutput=False):
    # sys.path.append(os.path.join(repoDir, 'ets_pointing/pfs_design_tool'))
    # from pfs_design_tool.pointing_utils import dbutils, designutils, nfutils
    from pfs_design_tool import reconfigure_fibers_ppp as sfa

    # import pointing_utils.dbutils as dbutils
    # import pointing_utils.nfutils as nfutils
    # import pointing_utils.designutils as designutils

    infile = os.path.join(workDir, conf['ppp']['outputDir'], 'ppp+qplan_outout.csv')
    list_pointings, dict_pointing, design_ids, observation_dates_in_hst = sfa.reconfigure(conf, workDir, infile=infile, clearOutput=clearOutput)

    return list_pointings, dict_pointing, design_ids, observation_dates_in_hst
