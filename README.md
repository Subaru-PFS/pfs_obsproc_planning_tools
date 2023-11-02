# PFS Obsproc Planning Tools

Tools for planning of PFS observations using PPP, queue planner, and pfsDesign generation script.

## Requirements

- Python >= 3.9,\<3.12 (MO uses Python 3.9 for development)
- Need to access the target database and Gaia database in Hilo.
- A valid Gurobi license is required and `GRB_LICENSE_FILE` environment valiable must be set correctly.
- A number of Python pakcages are required, and will be installed by using `pyproject.toml` or `requirements.txt`.

## Installation

```shell
# clone the repository
git clone https://github.com/Subaru-PFS/pfs_obsproc_planning_tools.git

# move to the directory
cd pfs_obsproc_planning_tools

# update submodule
git submodule update --init

# create a python virtual environment (recommended)
python3 -m venv .venv

# activate the venv
source .venv/bin/activate

# install dependencies
python3 -m pip install -r requirements.txt

# install this package itself as editable
python3 -m pip install -e .

# # it seems that you need to force reinstall when one of packages installed directly via GitHub repositories
# python3 -m pip install --upgrade --force-reinstall pfs_design_tool
```

## Configuration

### Work directory
A work directory has to be set. A template directory is provided as `examples/workdir_template`. Let's copy them.

```shell
cp -a examples/workdir_template examples/workdir_example
```

The structure of the work directory should look like the following.

```
workdir_template/
├── config.toml
└── templates
    └── template_pfs_v2023-09-12.ope
```

### Config file
Then you need to create a file as a `toml` file. You can find an example in `examples/workdir_example/config.toml`.

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

Before running the program, the following environment parameters are recommended to be unset
by using `unset` shell command if you intentionally want to use `EUPS` and pre-installed `pfs_utils`
and other specially installed packages with the `pfs_obsproc_planning`.

```shell
unset EUPS_DIR
unset EUPS_PATH
unset RUBIN_EUPS_PATH
unset EUPS_PKGROOT

unset PFS_UTILS_DIR
unset PFS_INSTDATA_DIR

unset PYTHONPATH
```

A simple example (`example.py`) is the following.

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

You can also try `notebooks/example_notebook.ipynb` in Jupyter Notebook or JupyterLab.


## Notes on some dependencies

Some of the following constraints are also applied to `ets_pointing`.

### `numpy`
The `numpy` version has to be `<1.24.0`. `ets_fiberalloc` uses `np.complex` which was deprecated and removed.

### `ics_cobraCharmer`
We use the commit `1af21d85a0af309cc57c939b78d625e884ece404` to be consistent with other packages. We hope that the other ics packages become consistent at some point. Also, it appears that the package needs to be installed as editable. Special treatment can be found in `__init__.py`.

### `ics_fpsActor`
We use the branch `u/monodera/tweak_sdss3tools_dep` to avoid `sdss3tools` dependency. It shouldn't be difficult to be merged into master. We should discuss with the developer of `ics_fpsActor`. It appears that the package needs to be installed as editable.

### `ets_pointing`

`ets_pointing` is installed as `pfs_design_tool` (name TBD).

### `ets_shuffle`
`ets_shuffle` is a repository without `setup.py` or `pyproject.toml`. Therefore, it cannot be installed via `pip`.  In the meantime, it is included as a submodule under `src` directory.
