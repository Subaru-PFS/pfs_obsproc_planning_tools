# PFS Obsproc Planning Tools

Tools for planning of PFS observations using PPP, queue planner, and pfsDesign generation script.

## Requirements

- Python >= 3.9 (MO uses Python 3.9 for development)
- Need to access the target database and Gaia database in Hilo.
- A valid Gurobi license is required and `GRB_LICENSE_FILE` environment valiable must be set correctly.
- A number of Python pakcages are required, and will be installed by using `pyproject.toml` or `requirements.txt`.

## Installation

```shell
# clone the repository
git clone https://github.com/Subaru-PFS/pfs_obsproc_planning_tools.git

# move to the directory
cd pfs_obsproc_planning_tools

# switch to the branch (will be merged into main at some point)
git switch u/monodera/pip_compatible

# create a python virtual environment (recommended)
python3 -m venv .venv

# activate the venv
source .venv/bin/activate

# install dependencies
python3 -m pip install -r requirements.txt

# install this package itself as editable
python3 -m pip install -e .
```

## Configuration

### Work directory
A work directory has to be set. A template directory is provided as `examples/workdir_template`. Let's copy them.

```shell
cp -a examples/workdir_template examples/workdir_example
```

### Config file
Then you need to create a file as a `toml` file. You can find an example in `examples/workdir_example/config.toml.

The following sections correspond to the database configuration. Please ask Yabe-san or myself if you need to access the databases.

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

Other parameters will be documentend later. Please let us know if you want to know the details of the parameters.

### Directory paths
- Note that most of directory paths are in relative to the working directory
- If you have `pfs_instdata_dir` directory elsewhere in the system, please provide the absolute path for it. If it is not found, it will be cloned from GitHub under the work directory.
- If you have EUPS installed and set `PFS_UTILS_DIR` correctly, data for fiber IDs are looked up there.


## Run

A simple example is the following.

```python
from pfs_obsproc_planning import generatePfsDesign


workDir = "<path_to_workdir>/workdir_example"
config = "config.toml"

# set number of pointings in the low- and medium-resolution modes
n_pccs_l = 5
n_pccs_m = 0

# set observation dates
obs_dates = ["2023-10-08", "2023-10-09", "2023-10-10"]

# initialize a GeneratePfsDesign instance
gpd = generatePfsDesign.GeneratePfsDesign(config, workDir=workDir)

# Run PPP
gpd.runPPP(n_pccs_l, n_pccs_m, show_plots=False)

# Run qplan
gpd.runQPlan(obs_dates)

# Run SFA
gpd.runSFA(clearOutput=True)
```


## Notes on some dependencies

### `numpy`
The `numpy` version has to be `<1.24.0`. `ets_fiberalloc` uses `np.complex` which was deprecated and removed.

### `pyyaml`
The `pyyaml` version has to be `<=5.3.1`. The more recent versions require to use `safe_load()` to read the `yaml` file (https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation)

### `ets_fiberalloc`
We use `u/kiyoyabe/e2e_test_2023oct`.


### `pfs_utils`
`pfs_utils` asuumes that EUPS is installed, but it's not easy to install only the EUPS package in non EUPS environment. As workaround `FiberIds()` can accept `path` keyword, but it is hard-coded in several places. The `u/monodera/add_path_gfm` fixes it for our purpose without disturbing others. This can also be merged into master, but needs some discussion with others.


### `ics_cobraCharmer`
We use the commit `1af21d85a0af309cc57c939b78d625e884ece404` to be consistent with other packages. We hope that the other ics packages become consistent at some point. Also, it appears that the package needs to be installed as editable.

### `ics_fpsActor`
We use the branch `u/monodera/tweak_sdss3tools_dep` to avoid `sdss3tools` dependency. It shouldn't be difficult to be merged into master. We should discuss with the developer of `ics_fpsActor`. It appears that the package needs to be installed as editable.

### `ets_pointing`

`ets_pointing` is installed as `pfs_design_tool` (name TBD) and the `u/monodera/reconfigure-20230620-pyproject` branch is used as a installable repository.
