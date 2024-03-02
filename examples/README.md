# Notes on setting up design tools for commissioning

## setup environment

After you login to one of the PFSA servers (pfsa-usr01 or pfsa-usr02), just copy `deploy.sh` to your directory for this test.

```console
cp /work/pfs/obsproc/examples/deploy.sh <<<YOUR_DIRECTORY_PATH>>
```

And run the script.

```console
source deploy.sh
```

It probably takes a few minutes.

## get Gurobi lisence and set `GUROBI_HOME`

Go the Gurobi website (https://www.gurobi.com/features/academic-named-user-license/) and get an academic license.
After you get your lisence, edit the following line of `kernel.sh` to specify the location of `gurobi.lic`.

```
export GUROBI_HOME="GUROBI_HOME"
```

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

Go to PFSA JupyterHub (http://pfsa-usr01.subaru.nao.ac.jp:8200/ or http://pfsa-usr02.subaru.nao.ac.jp:8200/) and copy `/work/pfs/obsproc/examples/notebooks/ppp+qplan+sfa_example.ipynb` to your location. Loggin account and password is the same as your STN account. Choose kernel `run15` , which should have been installed, and run the notebook. Again, you need to change the password for database in `e2e/configs/config.toml` (ask somebody if you don't know).


## updates on repositories before/during the run

Repositories (`pfs_obsproc_planning_tools` and `ets_pointing` mostly) are living so they should be updated before and during the run. Go to the repository and `git pull`. 

## reference
- https://github.com/Subaru-PFS/pfs_obsproc_planning_tools
