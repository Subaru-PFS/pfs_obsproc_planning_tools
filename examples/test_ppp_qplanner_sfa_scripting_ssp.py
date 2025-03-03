import numpy as np

#from pfs_obsproc_planning import GUI
from pfs_obsproc_planning import generatePfsDesign
import time

np.random.seed(1)

workDir = "/home/wanqqq/ssp_design/spt_ssp_observation/runs/2025-03/"
config = "config.toml"

SFA = True
validation = True

gpd = generatePfsDesign.GeneratePfsDesign(config, workDir=workDir)

# Run SFA
if SFA:
    gpd.runSFA_ssp()

# Run validation
if validation:
    gpd.runValidation()