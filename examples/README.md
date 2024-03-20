# Notes on setting up design tools for commissioning

## Prerequisite
### access PFSA servers
You need to login so-called PFSA servers. If you don't have access to those servers, ask somebody who knows about the environment.

### get Gurobi lisence and set `GUROBI_HOME`

Go the Gurobi website (https://www.gurobi.com/features/academic-named-user-license/) and get an academic `Named-User Academic` license. Please download `gurobi10.0.3_linux64.tar.gz` in `Older Versions` section. Just tar and gunzip the file in the location you like. Then, please set the following environment variable in e.g. ~/.bashrc. `GUROBI_HOME` is the location of `linux64` directory. Note that the license expires in 1 year, so if you already have the license, please confirm that it is still valid.

```console
export GUROBI_HOME="YOUR_GUROBI_HOME"
export PATH="${PATH}:${GUROBI_HOME}/linux64/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/linux64/lib"
export GRB_LICENSE_FILE="${GUROBI_HOME}/gurobi.lic"
```

Run `grbgetkey` to create license file following the gurobi instruction and set the location of license file properly as you specified above.

## setup environment

After you login to one of the PFSA servers (pfsa-usr01 or pfsa-usr02), just copy `deploy.sh` to your directory for this test.

```console
cp /work/pfs/obsproc/examples/deploy.sh <<<YOUR_DIRECTORY_PATH>>
```

And run the script.

```console
source deploy.sh
```

It probably takes a few minutes. Then, modify`export GUROBI_HOME="YOUR_GUROBI_HOME"
` in `kernel.sh` for the Gurobi home location as you specified above.

## if you have already setup the environement

Go to the directory and run `start.sh`.

```console
source start.sh
```

## test design tools for commissioning

Go to `commissioning` directory and run `scripts/example.sh`. You need to change the password for database in `configs/config.toml` (ask somebody if you don't know).


```console
cd commissioning
source ./scripts/example.sh
```

## test design tools for e2e test

Go to PFSA JupyterHub (http://pfsa-usr01.subaru.nao.ac.jp:8200/ or http://pfsa-usr02.subaru.nao.ac.jp:8200/) and copy `/work/pfs/obsproc/examples/notebooks/ppp+qplan+sfa_example.ipynb` to your location (under /home/your_name/somewhere). Loggin account and password is the same as your STN account. Choose kernel `run15` , which should have been installed, and run the notebook. Again, you need to change the password for database in `e2e/configs/config.toml` (ask somebody if you don't know).


## updates on repositories before/during the run

Repositories (`pfs_obsproc_planning_tools` and `ets_pointing` mostly) are living so they should be updated before and during the run. Go to the repository and `git pull`. 

## reference
- https://github.com/Subaru-PFS/pfs_obsproc_planning_tools
