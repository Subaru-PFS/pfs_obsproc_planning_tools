# Tutorial on using the GUI of integrated codes

## Pre-requirements

- Please refer to [README](../README.md) for installation 
    - Please clone the branch `u/wanqqq/update_2024aug` instead of `main` for now (Oct. 2024)
        ```shell
        git clone -b u/wanqqq/update_2024aug https://github.com/Subaru-PFS/pfs_obsproc_planning_tools.git
        ```
- Key points:
    1. Please ensure the Gurobi environment and license are correctly set;
    2. Please ensure a work directory has been correctly set;
        ```shell
        workdir_example/
        ├── config.toml
        └── input
        └── output
        └── templates
            └── template_pfs_xxx.ope...
        ```
    3. Please ensure you log in the PFSA server to connect with multiple databases;
    4. Running the codes under a python virtual environment is recommended.

## Run

- **Step1**: start GUI
    - please open the script `test_ppp_qplanner_sfa_scripting_gui.py` under `examples/` folder, and set `repoDir` to the folder you install the package:
        ```shell
        repoDir="<path_to_pfs_obsproc_planning_tools>/pfs_obsproc_planning_tools/src/pfs_obsproc_planning"
        ```
    - run the script
        ```shell
        python3 examples/test_ppp_qplanner_sfa_scripting_gui.py 
        ```
        - if you connect to PFSA server remotely, please firstly log in the server using
            ```shell
            ssh -X username@pfsa-usr01.subaru.nao.ac.jp
            ```
    - the GUI should pop up like this:
    <figure markdown>
        ![Status indicators](tutorial_fig/gui_window.png){ width="1000" }
    </figure>



