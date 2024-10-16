import numpy as np
from pfs_obsproc_planning import GUI

np.random.seed(1)

gui = GUI.GeneratePfsDesignGUI(
    repoDir="~/pfs_obsproc_planning_tools/src/pfs_obsproc_planning/"
)
gui.run()
