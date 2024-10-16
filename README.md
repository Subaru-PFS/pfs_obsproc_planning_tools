# PFS Obsproc Planning Tools

Tools for planning of PFS observations using PPP, queue planner, and pfsDesign generation script.

## Requirements

- Python >= 3.9,\<3.12 (MO uses Python 3.9 for development)
- Need to access the target database, queue database and Gaia database in Hilo.
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

Set the path to your Gurobi license:

```shell
export GRB_LICENSE_FILE = <path_to_gurobi_license>/gurobi.lic
```

If you have not set `GUROBI_HOME`, please also do so:

```shell
export GUROBI_HOME="YOUR_GUROBI_HOME/linux64"
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
```

## Configuration

### Work directory
A work directory has to be set. A template directory is provided as `examples/workdir_template`. Let's copy them.

```shell
cp -a examples/workdir_template examples/workdir_example
```

The structure of the work directory looks like the following.

```
workdir_template/
├── config.toml
└── input
└── output
    └── design
    └── ope
    └── ppp
    └── qplan
└── templates
    └── template_pfs_xxx.ope
```

If you have local target lists or pointings lists, please put them under `input/`, otherwise please keep the folder blank. The outputs will be stored under `output/`. The template ope files should be stored under `templates/`.

### Config file
Then you need to create a file as a `toml` file. It should contain all necessary parameters to run the integrated codes. You can find an example in `examples/workdir_example/config.toml`.

- **Connection with database**
The following sections correspond to the database configuration. Please ask Yabe-san, Onodera-san or Eric-san if you need to access the databases.

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
    [queuedb]
    filepath = "<path_to_queue_access_file>/queue_access_file.yml"
    ```

    Please note that you need to create a `yml` file to connect to queue database. It should include the following information:

    ```
    db_host: "example.com"
    db_port: port
    db_user: "username"
    db_pass: "password"
    ```

    For the calibrators, please use the latest version of sky and Fstar catalogs in target database:

    ```
    [targetdb.sky]
    version = ["20221031"]

    [targetdb.fluxstd]
    version = ["3.3"]
    ```

- **PPP**
The following parameters correspond to the PPP configuration. There are two modes, "Queue" and "Classic". The parameters in the two modes are different. 
    - In the "Queue" mode, a typical set of parameters looks like:

        ```
        [ppp]
        mode = 'queue'
        include_classicalPPC = False
        localPath_design_classic = ''
        sql_query = 'query'
        inputDir = 'input'
        outputDir = 'output/ppp'
        reserveFibers = false
        fiberNonAllocationCost = 1.0e+05
        ```

        The default way to query targets in "Queue" mode is through databases. The sql query is input by `sql_query` parameter. An example query command is listed below:

        ```
        SELECT ob_code,obj_id,c.input_catalog_id,ra,dec,epoch,priority,pmra,pmdec,parallax,effective_exptime,single_exptime,is_medium_resolution,proposal.proposal_id,rank,grade,allocated_time_lr+allocated_time_mr as \"allocated_time\",allocated_time_lr,allocated_time_mr,filter_g,filter_r,filter_i,filter_z,filter_y,psf_flux_g,psf_flux_r,psf_flux_i,psf_flux_z,psf_flux_y,psf_flux_error_g,psf_flux_error_r,psf_flux_error_i,psf_flux_error_z,psf_flux_error_y  
        FROM target JOIN proposal ON target.proposal_id=proposal.proposal_id JOIN input_catalog AS c ON target.input_catalog_id = c.input_catalog_id 
        WHERE proposal.proposal_id LIKE 'S24B%' AND proposal.grade in ('A','B')  AND c.active;"
        ```

        If you want to assign blank fibers on classical pointings to queue targets, you can set `include_classicalPPC = True` and set `localPath_design_classic` with the local path to the design files of classical pointings.

    - In the "Classic" mode, a typical set of parameters looks like:

        ```
        [ppp]
        mode = 'classic'
        localPath_tgt = '<path_to_target_file>/target.csv' 
        localPath_ppc = '<path_to_ppc_file>/ppc.csv'
        sql_query = 'query'
        inputDir = 'input'
        outputDir = 'output/ppp'
        reserveFibers = false
        fiberNonAllocationCost = 1.0e+05
        ```

        The default way to query targets in "Classic" mode is through databases as well. The sql query is input by `sql_query` parameter. An example query command is listed below:

        ```
        SELECT ob_code,obj_id,c.input_catalog_id,ra,dec,epoch,priority,pmra,pmdec,parallax,effective_exptime,single_exptime,is_medium_resolution,proposal.proposal_id,rank,grade,allocated_time_lr+allocated_time_mr as \"allocated_time\",allocated_time_lr,allocated_time_mr,filter_g,filter_r,filter_i,filter_z,filter_y,psf_flux_g,psf_flux_r,psf_flux_i,psf_flux_z,psf_flux_y,psf_flux_error_g,psf_flux_error_r,psf_flux_error_i,psf_flux_error_z,psf_flux_error_y  
        FROM target JOIN proposal ON target.proposal_id=proposal.proposal_id JOIN input_catalog AS c ON target.input_catalog_id = c.input_catalog_id 
        WHERE proposal.proposal_id = 'S24B-920' AND c.active;"
        ```

        It also accepts local target lists and pointing lists. Please set `localPath_tgt` and `localPath_ppc` with the path to the local target lists and pointing lists, respectively. Please note the files should be in csv format, and it should contain necessary information listed [here](https://pfs-etc.naoj.hawaii.edu/uploader/doc/inputs.html).

- **QPlan**
The following parameters correspond to the QPlan configuration. You can set overhead, start and stop time of the observation. The default start/stop observation time is `twilight_18`. Please note the overhead should be given in units of minutes, and start/stop time should be given in the format of `%Y-%m-%d %H:%M:%S` (UTC). There are several weighting parameters. The recommended values are listed below. If you do not have specific requirements, please do not change them. 

    ```
    [qplan]
    outputDir = 'output/qplan'
    overhead = "7" # minute
    start_time = "" # UTC
    stop_time = "" # UTC

    [qplan.weight]
    slew = 0.2
    delay = 3.0
    filter = 0.0
    rank = 0.85
    priority = 0.1
    ```

- **Generation of design files**
The following parameters correspond to the configuration to generate design and ope files. 

    ```
    [sfa]
    n_sky = 400 #number of fibers assigned to skys per pointing
    sky_random = false 
    n_sky_random = 30000 
    reduce_sky_targets = true 
    pfs_instdata_dir = "<path_to_pfs_instdata>/pfs_instdata"
    cobra_coach_dir = "<path_to_cobracoach>/cobracoach"
    cobra_coach_module_version = "None"
    sm = [1, 2, 3, 4]
    dot_margin = 1.65
    dot_penalty = "None"
    guidestar_mag_min = 12.0 #minimal magnitude of guide stars
    guidestar_mag_max = 22.0 #maximal magnitude of guide stars
    guidestar_neighbor_mag_min = 21.0
    guidestar_minsep_deg = 0.0002778
    n_fluxstd = 200 #number of fibers assigned to flux calibrators per pointing
    fluxstd_mag_max = 19.0 #maximal magnitude of flux calibrator stars
    fluxstd_mag_min = 17.0 #minimal magnitude of flux calibrator stars
    fluxstd_mag_filter = "g"
    good_fluxstd = false
    fluxstd_min_prob_f_star = 0.5
    fluxstd_min_teff = 5000
    fluxstd_max_teff = 8000
    fluxstd_flags_dist = false
    fluxstd_flags_ebv = false
    filler = true #whether to include fillers
    filler_mag_min = 18.0 #minimal magnitude of fillers
    filler_mag_max = 22.5 #maximal magnitude of fillers
    n_fillers_random = 15000
    reduce_fillers = true
    multiprocessing = true #whether to allow parallel running of generating design files

    [ope]
    template = "templates/template_pfs_xxx.ope" #path to template ope file
    outfilePath = "output/ope"
    designPath = "output/design"
    runName = "test"
    ```
    The recommended values of parameters are listed above (some parameters TBD). If you do not have specific requirements, please do not modify them. Please note we use Gurobi to solve the problem of fiber assignment. Please also include the following parameters in your config file: 
    ```
    [netflow]
    use_gurobi = true
    two_stage = false #whether to allow fibers assigned to science targets and calibrators in the 1st stage, and to fillers in the 2nd stage
    cobra_location_group_n = 50 #(Related to uniform distribution of calibrators) number of sub-regions per pointing
    min_sky_targets_per_location = 8 #(Related to uniform distribution of calibrators) minimal number of calibrators per sub-region
    location_group_penalty = 5e11

    [gurobi]
    seed = 0
    presolve = 1
    method = 0
    degenmoves = 0
    heuristics = 0.6
    mipfocus = 0
    mipgap = 5.0e-3
    PreSOS2Encoding = 0
    PreSOS1Encoding = 0
    threads = 4
    ```


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
