#!/usr/bin/env python3
# validation.py : Subaru Fiber Allocation software

import os

# The script to make a figure to check pfsDesign
import sys
import warnings
from dataclasses import make_dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logzero import logger
from mpl_toolkits.mplot3d import Axes3D

# from pfs.drp.stella.readLineList import ReadLineListTask,  ReadLineListConfig
# from pfs.drp.stella import DetectorMap
# import lsst.daf.persistence as dafPersist
from pfs.datamodel.pfsConfig import *
from pfs.utils.coordinates.CoordTransp import ag_pixel_to_pfimm
from pfs.utils.coordinates.DistortionCoefficients import radec_to_subaru
from pfs.utils.fiberids import FiberIds

# sys.path.append("/work/moritani/codes/obstools/")
from . import plotPfsDesign as pldes

warnings.filterwarnings("ignore")


def njy_mag(j):
    if j > 0:
        return 23.9 - 2.5 * np.log10(j / 1e3)
    else:
        return np.nan


def calc_inr(df, obstime):
    try:
        az, el, inr = radec_to_subaru(
            df["ppc_ra"],
            df["ppc_dec"],
            df["ppc_pa"],
            obstime,
            2016.0,
            0.0,
            0.0,
            1.0e-7,
        )
    except KeyError:
        az, el, inr = radec_to_subaru(
            df["ra_center"],
            df["dec_center"],
            df["pa_center"],
            obstime,
            2016.0,
            0.0,
            0.0,
            1.0e-7,
        )
    except ValueError as e:
        logger.warning(f"Error in calculating InR: {e}")
        inr = np.nan
    return inr, el 

def validation(parentPath, figpath, save, show, ssp):
    if not ssp:
        pfsDesignDir = f"{parentPath}/design"
        df_design = pd.read_csv(
            f"{parentPath}/summary_reconfigure_ppp-ppp+qplan_output.csv"
        )
        df_design["observation_time"] = pd.to_datetime(df_design["observation_time"], format='%Y-%m-%dT%H:%M:%SZ')
        df_design["observation_time_stop"] = (df_design["observation_time"] + pd.DateOffset(seconds=1260)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    else:
        pfsDesignDir = parentPath
        df_design = pd.read_csv(
            os.path.join(parentPath, "..", f"{parentPath[-2:]}_summary_reconfigure.csv")
        )
        df_design["observation_time"] = pd.to_datetime(df_design["ppc_obstime_utc"], format='%Y-%m-%dT%H:%M:%SZ')
        df_design["observation_time_stop"] = (df_design["observation_time"] + pd.to_timedelta(df_design["ppc_exptime"], unit="s")).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Calcurate InR at observing time
    # df_design['pa']=0.
    df_design[["inr1", "el1"]] = df_design.apply(
        lambda row: pd.Series(calc_inr(row, obstime=row["observation_time"])), axis=1
    )
    df_design[["inr2", "el2"]] = df_design.apply(
        lambda row: pd.Series(calc_inr(row, obstime=row["observation_time_stop"])), axis=1
    )
    
    # Make pdf files and store the statistical data
    pfsDesignIds = (
        df_design["design_filename"]
        .str.split(r"-|\.", regex=True, expand=True)[1]
        .map(lambda x: int(x, 16))
    )

    # This routine just combines a few cells in "trial" section
    df_ch = pldes.init_check_design()
    for designId in pfsDesignIds:
        pfsDesign0 = PfsDesign.read(designId, dirName=pfsDesignDir)
        pfsDesign0.validate()
        print(f"{pfsDesign0.designName}, {pfsDesign0.arms}")

        # check fiber duplicates
        df_t = pd.DataFrame(
            {"fiberId": pfsDesign0.fiberId, "obCode": pfsDesign0.obCode}
        )
        index_dup = df_t.duplicated(subset=["fiberId"], keep=False)
        if sum(index_dup) > 0:
            logger.warning(
                f"[Validation of output] There are duplicated fibers: {df_t[index_dup]}"
            )
        else:
            logger.info("[Validation of output] No duplicated fiber")
        pfsflux = np.array([a[0] if len(a) > 0 else np.nan for a in pfsDesign0.psfFlux])
        totalflux = np.array([a[0] if len(a) > 0 else np.nan for a in pfsDesign0.totalFlux])
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
                    totalflux,
                    pfsDesign0.catId,
                )
            ),
            columns=[
                "fiberId",
                "targetType",
                "pfi_x",
                "pfi_y",
                "spec",
                "fh",
                "pfsFlux",
                "totalFlux",
                "catId",
            ],
        )
        df_fib["proposalId"] = pfsDesign0.proposalId
        df_fib["psfMag"] = df_fib["pfsFlux"].apply(njy_mag)
        df_fib["totalMag"] = df_fib["totalFlux"].apply(njy_mag)

        # Identify rows where either magnitude is < 13
        df_too_bright = df_fib[(df_fib["psfMag"] < 13) | (df_fib["totalMag"] < 13)]
        
        if not df_too_bright.empty:
            logger.warning(f"[Validation of output] Too bright sources with psfMag or totalMag < 13 (0x{designId:016x}, {pfsDesign0.designName}): {df_too_bright}")
    
        df_fib["sector"] = pldes.get_field_sector2(df_fib)
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
            df_fib,
            df_ag,
            alpha=1.0,
            title=title,
            fname=fname,
            save=save,
            show=show,
            pa=pfsDesign0.posAng,
        )
    df_ch["inr1"] = df_design["inr1"]
    df_ch["inr2"] = df_design["inr2"]
    df_ch["el1"] = df_design["el1"]
    df_ch["el2"] = df_design["el2"]

    # """
    styled_html = (
        df_ch.style.applymap(
            pldes.colour_background_warning_sky_min, subset=["sky_min"]
        )
        .applymap(pldes.colour_background_warning_std_min, subset=["std_min"])
        .applymap(pldes.colour_background_warning_sky_tot, subset=["sky_sum"])
        .applymap(pldes.colour_background_warning_std_tot, subset=["std_sum"])
        .applymap(
            pldes.colour_background_warning_ag_min,
            subset=["ag1", "ag2", "ag3", "ag4", "ag5", "ag6"],
        )
        .applymap(pldes.colour_background_warning_ag_tot, subset=["ag_sum"])
        .applymap(pldes.colour_background_warning_inr, subset=["inr1", "inr2"])
        .applymap(pldes.colour_background_warning_el, subset=["el1", "el2"])
        .format(precision=1)
    )
    # """

    # Convert the Styler object to an HTML string
    html_str = styled_html.to_html()

    with open(os.path.join(figpath, "validation_report.html"), "w") as f:
        f.write(html_str)

    # df_ch.to_csv(f"{figpath}/validation_report.csv")
