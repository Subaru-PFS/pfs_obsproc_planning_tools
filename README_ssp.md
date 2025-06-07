# PFS Obsproc Planning Tools

### For SSP use:

### Work directory
The structure of the work directory looks like the following.

```
<workdir>/
    └── config.toml
    └── template_pfs_v2024-03-05_singleExptime.ope
    └── targets
        └── CO
            ├── ppcList.ecsv
            └── science
                └── ppc_code1.ecsv
                └── ppc_code2.ecsv
                └── ...
            └── sky
                └── ppc_code1.ecsv
                └── ppc_code2.ecsv
                └── ...
            └── fluxstd
                └── ppc_code1.ecsv
                └── ppc_code2.ecsv
                └── ...
        └── GE
            └── ...
        └── GA
            └── ..._

    └── ope_files
        └── YYYY-MM-DD.ope
        └── ..._    

    └── pfs_designs
        └── CO_summary_reconfigure.csv
        └── GE_summary_reconfigure.csv
        └── GA_summary_reconfigure.csv
        └── CO
            ├── pfsDesign-xxx.fits
            └── ..._  
        └── GE
            ├── pfsDesign-xxx.fits
            └── ..._  
        └── GA
            └── pfsDesign-xxx.fits
            └── ..._  
    └── validations
        └── CO
            ├── check-pfsDesign-xxx.pdf
            └── ..._  
        └── GE
            ├── check-pfsDesign-xxx.pdf
            └── ..._  
        └── GA
            └── check-pfsDesign-xxx.pdf
            └── ..._  
```

Now `workdir = runs/2025-03/`.

### Config file
```
[targetdb.db]
host = "example.com"
port = 5432
dbname = "dbname"
user = "username"
password = "password"
dialect = "postgresql"
```

```
[gaiadb]
host = "example.com"
port = 5432
dbname = "dbname"
user = "username"
password = "password"
```

```
[targetdb.sky]
version = ["20230209"]

[targetdb.fluxstd]
version = ["3.3"]
```

```
[sfa]
pfs_utils_dir = ""
pfs_instdata_dir = "/home/wanqqq/ssp_design/pfs_obsproc_planning_tools/src/pfs_instdata"
pfs_utils_ver = "w.2025.06"
pfs_instdata_ver = "1.8.14"
cobra_coach_dir = "/home/wanqqq/cobracoach"
cobra_coach_module_version = "None"
sm = [1, 2, 3, 4]
dot_margin = 1.65
dot_penalty = "None"
guidestar_mag_min = 12.0
guidestar_mag_max = 22.0
guidestar_neighbor_mag_min = 21.0
guidestar_minsep_deg = 0.0002778
multiprocessing = true

[ope]
template = "template_pfs_v2024-03-05_singleExptime.ope"
outfilePath = "ope_files/"
designPath = "pfs_designs/"
runName = "2025-03"
n_split_frame=1

[ssp]
ssp = true
WG = ["CO", "GE"] # WGs set to be processed
```

`pfs_utils_dir`, `pfs_instdata_dir`, `pfs_utils_ver`, `pfs_instdata_ver`, `cobra_coach_dir`, `WG` need to be modified.

### Run

```python
import numpy as np
from pfs_obsproc_planning import generatePfsDesign

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
```

You can try `example/test_ppp_qplanner_sfa_scripting_ssp.py`.

