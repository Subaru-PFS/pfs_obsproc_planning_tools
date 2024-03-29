#!/usr/bin/env python3
# SFA.py : Subaru Fiber Allocation software

import warnings

from pfs_design_tool import reconfigure_fibers_ppp as sfa

warnings.filterwarnings("ignore")


def run(
    conf, workDir=".", repoDir=".", infile="ppp+qplan_output.csv", clearOutput=False
):
    (
        list_pointings,
        dict_pointing,
        design_ids,
        observation_dates_in_hst,
    ) = sfa.reconfigure(conf, workDir, infile=infile, clearOutput=clearOutput)

    return list_pointings, dict_pointing, design_ids, observation_dates_in_hst
