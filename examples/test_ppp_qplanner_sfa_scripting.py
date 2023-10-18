
import numpy as np
from pfs_obsproc_planning import generatePfsDesign


np.random.seed(1)


workDir = "workdir_example"
config = "config.toml"

n_pccs_l = 5
n_pccs_m = 0
obs_dates = ["2023-10-08", "2023-10-09", "2023-10-10"]

# initialize a GeneratePfsDesign instance
gpd = generatePfsDesign.GeneratePfsDesign(config, workDir=workDir)

# Run PPP
gpd.runPPP(n_pccs_l, n_pccs_m, show_plots=False)

# Run qplan
gpd.runQPlan(obs_dates)

# Run SFA
gpd.runSFA(clearOutput=True)
