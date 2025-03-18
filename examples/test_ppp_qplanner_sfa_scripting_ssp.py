import numpy as np

#from pfs_obsproc_planning import GUI
from pfs_obsproc_planning import generatePfsDesign_ssp
import time

np.random.seed(1)

workDir = "/home/wanqqq/ssp_design/spt_ssp_observation/runs/2025-03/"
config = "config.toml"

validation_input = False
make_design = True
update_ope = False
validation_output = True

gpd = generatePfsDesign_ssp.GeneratePfsDesign_ssp(config, workDir=workDir)

# Validate input target&ppc lists
if validation_input:
    gpd.ssp_input_validation()

# Run everything -- output design and ope
if make_design:
    gpd.runSFA_ssp()

# Update scheduling and ope files
if update_ope:
    gpd.ssp_obsplan_update()

# Validate output designs
if validation_output:
    gpd.runValidation()