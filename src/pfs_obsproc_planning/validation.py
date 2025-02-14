#!/usr/bin/env python3
# validation.py : Subaru Fiber Allocation software

import numpy as np
import pandas as pd
from dataclasses import make_dataclass
import os

# from pfs.drp.stella.readLineList import ReadLineListTask,  ReadLineListConfig
# from pfs.drp.stella import DetectorMap
# import lsst.daf.persistence as dafPersist
from pfs.datamodel.pfsConfig import *
from pfs.utils.coordinates.CoordTransp import ag_pixel_to_pfimm
from pfs.utils.coordinates.DistortionCoefficients import radec_to_subaru
from pfs.utils.fiberids import FiberIds

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# The script to make a figure to check pfsDesign
import sys

#sys.path.append("/work/moritani/codes/obstools/")
from . import plotPfsDesign as pldes

import warnings

warnings.filterwarnings("ignore")


def njy_mag(j):
    return 23.9 - 2.5 * np.log10(j / 1e3)


def calc_inr(df):
    az, el, inr = radec_to_subaru(
        df["ra_center"],
        df["dec_center"],
        df["pa_center"],
        df["observation_time"],
        2016.0,
        0.0,
        0.0,
        0.0,
    )
    return inr


def validation(parentPath, figpath, save, show):
    pfsDesignDir = f"{parentPath}/design"
    df_design = pd.read_csv(
        f"{parentPath}/summary_reconfigure_ppp-ppp+qplan_output.csv"
    )

    # Calcurate InR at observing time
    # df_design['pa']=0.
    df_design["inr"] = df_design.apply(calc_inr, axis=1)

    # Make pdf files and store the statistical data
    pfsDesignIds = (
        df_design["design_filename"]
        .str.split(r"-|\.", regex=True, expand=True)[1]
        .map(lambda x: int(x, 16))
    )

    if not os.path.exists(figpath):
        os.makedirs(figpath)

    # This routine just combines a few cells in "trial" section
    df_ch = pldes.init_check_design()
    for designId in pfsDesignIds:
        pfsDesign0 = PfsDesign.read(designId, dirName=pfsDesignDir)
        print(f"{pfsDesign0.designName}, {pfsDesign0.arms}")
        pfsflux = np.array([a[0] if len(a) > 0 else np.nan for a in pfsDesign0.psfFlux])
        # print(len(pfsDesign0[pfsDesign0.fiberStatus==3]))
        df_fib = pd.DataFrame(
            data=np.column_stack(
                (
                    pfsDesign0.fiberId, 
                    pfsDesign0.targetType,
                    pfsDesign0.pfiNominal,
                    pfsDesign0.spectrograph,
                    pfsDesign0.fiberHole,
                    pfsflux,
                )
            ),
            columns=["fiberId", "targetType", "pfi_x", "pfi_y", "spec", "fh", "pfsFlux"],
        )
        df_fib["proposalId"] = pfsDesign0.proposalId
        df_fib["psfMag"] = njy_mag(df_fib["pfsFlux"])
        df_fib["sector"] = df_fib.apply(pldes.get_field_sector, axis=1)
        df_ag = pd.DataFrame(
            data=np.column_stack(
                (
                    pfsDesign0.guideStars.agX,
                    pfsDesign0.guideStars.agY,
                    pfsDesign0.guideStars.magnitude,
                    pfsDesign0.guideStars.agId,
                )
            ),
            columns=["agx", "agy", "agMag", "camId"],
        )
        df_ag["camId"] = df_ag["camId"].astype(int)
        arr = df_ag[["camId", "agx", "agy"]].values
        pfix = []
        pfiy = []
        for a in arr:
            x, y = ag_pixel_to_pfimm(int(a[0]), a[1], a[2])
            pfix.append(x)
            pfiy.append(y)
        df_ag["ag_pfi_x"] = np.array(pfix)
        df_ag["ag_pfi_y"] = np.array(pfiy)

        df = pldes.check_design(designId, df_fib, df_ag)
        df_ch = pd.concat([df_ch, df], ignore_index=True)
        title = f"designId=0x{designId:016x} ({pfsDesign0.raBoresight:.2f},{pfsDesign0.decBoresight:.2f},PA={pfsDesign0.posAng:.1f})\n{pfsDesign0.designName}"
        fname = f"{figpath}/check_0x{designId:016x}"
        pldes.plot_FoV(
            df_fib, df_ag, alpha=1.0, title=title, fname=fname, save=save, show=show
        )
    df_ch["inr"] = df_design["inr"]

    """
    df_ch.style.applymap(pldes.colour_background_warning_sky_min, subset=['sky_min'])\
         .applymap(pldes.colour_background_warning_std_min, subset=['std_min'])\
         .applymap(pldes.colour_background_warning_sky_tot, subset=['sky_sum'])\
         .applymap(pldes.colour_background_warning_std_tot, subset=['std_sum'])\
         .applymap(pldes.colour_background_warning_ag_min, subset=['ag1', 'ag2', 'ag3', 'ag4', 'ag5', 'ag6'])\
         .applymap(pldes.colour_background_warning_ag_tot, subset=['ag_sum'])\
         .applymap(pldes.colour_background_warning_inr, subset=['inr']).format(precision=1)
    #"""

    df_ch.to_csv(f"{figpath}/validation_report.csv")
