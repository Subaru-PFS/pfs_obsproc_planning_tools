#!/usr/bin/env python3
"""Validation utilities for PFS designs.

This module verifies PfsDesign outputs, checks for problematic guide stars
and bright sources, computes InR/El at observation times, and generates
an HTML validation report and per-design FoV plots.

Refactored for clarity: helper functions were added to simplify the
flow and make individual checks easier to test.
"""

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
from pfs_design_tool import reconfigure_fibers_ppp as sfa

warnings.filterwarnings("ignore")


from importlib import reload

reload(pldes)


def njy_mag(j):
    """Convert flux `j` (in arbitrary units) to an approximate magnitude.

    The original behaviour uses 23.9 - 2.5*log10(j/1e3) when j>0 and NaN otherwise.
    """
    if j > 0:
        return 23.9 - 2.5 * np.log10(j / 1e3)
    else:
        return np.nan


def calc_inr(df, obstime):
    """Compute instrument rotator angle (InR) and elevation at `obstime`.

    Tries to use 'ppc_ra/dec/pa' columns first (when present), otherwise
    falls back to 'ra_center/dec_center/pa_center'. Returns (inr, el).
    """
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


def _load_design_summary(parentPath: str, ssp: bool):
    """Load design summary CSV and return (pfsDesignDir, df_design).

    Handles both PPP output layout (ssp=False) and SSP layout.
    """
    if not ssp:
        pfsDesignDir = f"{parentPath}/design"
        df_design = pd.read_csv(
            f"{parentPath}/summary_reconfigure_ppp-ppp+qplan_output.csv"
        )
        df_design["observation_time"] = pd.to_datetime(
            df_design["observation_time"], format="%Y-%m-%dT%H:%M:%SZ"
        )
        df_design["observation_time_stop"] = (
            df_design["observation_time"] + pd.DateOffset(seconds=1260)
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        pfsDesignDir = parentPath
        df_design = pd.read_csv(
            os.path.join(parentPath, "..", f"{parentPath[-2:]}_summary_reconfigure.csv")
        )
        df_design["observation_time"] = pd.to_datetime(
            df_design["ppc_obstime_utc"], format="%Y-%m-%dT%H:%M:%SZ"
        )
        df_design["observation_time_stop"] = (
            df_design["observation_time"]
            + pd.to_timedelta(df_design["ppc_exptime"], unit="s")
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return pfsDesignDir, df_design


def _add_inr_columns(df_design: pd.DataFrame) -> pd.DataFrame:
    """Compute inr1/el1 and inr2/el2 for all entries and add as columns.

    Returns the modified DataFrame (in-place-and-return for convenience).
    """
    df_design[["inr1", "el1"]] = df_design.apply(
        lambda row: pd.Series(calc_inr(row, obstime=row["observation_time"])), axis=1
    )
    df_design[["inr2", "el2"]] = df_design.apply(
        lambda row: pd.Series(calc_inr(row, obstime=row["observation_time_stop"])),
        axis=1,
    )
    return df_design


def _warn_duplicate_fibers(pfsDesign0):
    """Log a warning if a PfsDesign contains duplicated fiber IDs."""
    df_t = pd.DataFrame({"fiberId": pfsDesign0.fiberId, "obCode": pfsDesign0.obCode})
    index_dup = df_t.duplicated(subset=["fiberId"], keep=False)
    if sum(index_dup) > 0:
        logger.warning(
            f"[Validation of output] There are duplicated fibers: {df_t[index_dup]}"
        )


def _warn_too_bright_guidestars(ppc_ra, ppc_dec, ppc_pa, ppc_obstime_utc, conf):
    """Query Gaia for very bright guide stars and return a DataFrame of results.

    Returns an empty DataFrame if none found.
    """
    guidestars_toobright = sfa.designutils.generate_guidestars_from_gaiadb(
        ppc_ra,
        ppc_dec,
        ppc_pa,
        ppc_obstime_utc,  # obstime should be in UTC
        telescope_elevation=None,
        conf=conf,
        guidestar_mag_min=0,
        guidestar_mag_max=10,
        guidestar_neighbor_mag_min=21.0,
        guidestar_minsep_deg=0.0002778,
    )
    df_guidestars_toobright = pd.DataFrame(
        {
            "agId": guidestars_toobright.agId,
            "objId": guidestars_toobright.objId,
            "ra": guidestars_toobright.ra,
            "dec": guidestars_toobright.dec,
            "magnitude": guidestars_toobright.magnitude,
            "passband": guidestars_toobright.passband,
        }
    )
    if not df_guidestars_toobright.empty:
        logger.warning(
            f"[Validation of output] There are too bright guide stars: {df_guidestars_toobright}"
        )
    return df_guidestars_toobright


def _find_unassigned_bright_nearby(pfsDesign0, bench, fibId, ppc_ra, ppc_dec, ppc_pa, conf):
    """Search for bright Gaia sources around unassigned fibers.

    Returns a DataFrame (possibly empty) with columns matching the original script.
    """
    unassigened_fibers = pfsDesign0[pfsDesign0.targetType == 4].fiberId
    df_unassigned_toobright = pd.DataFrame()
    for unfib in unassigened_fibers:
        if (
            pfsDesign0[pfsDesign0.fiberId == unfib].fiberStatus
            != FiberStatus.BROKENFIBER
        ):
            cidx = fibId.fiberIdToCobraId(unfib) - 1
            ccenter = bench.cobras.centers[cidx]
            un_ra, un_dec = sfa.designutils.get_skypos_cobra(
                ccenter, pfsDesign0.obstime, ppc_ra, ppc_dec, ppc_pa
            )

            df_gaia_toobright = sfa.dbutils.generate_targets_from_gaiadb(
                un_ra,
                un_dec,
                conf=conf,
                search_radius=conf["sfa"]["fill_unassign_radius_check"],
                band_select="phot_g_mean_mag",
                mag_min=-2.0,
                mag_max=12.0,
                good_astrometry=False,
                write_csv=False,
            )

            if not df_gaia_toobright.empty:
                df_tmp = pd.DataFrame(
                    {
                        "fiber_id": [unfib] * len(df_gaia_toobright),
                        "fiber_ra": [un_ra] * len(df_gaia_toobright),
                        "fiber_dec": [un_dec] * len(df_gaia_toobright),
                        "source_id": df_gaia_toobright.source_id,
                        "ra": df_gaia_toobright.ra,
                        "dec": df_gaia_toobright.dec,
                        "magnitude": df_gaia_toobright.phot_g_mean_mag,
                    }
                )
                if len(df_unassigned_toobright) == 0:
                    df_unassigned_toobright = df_tmp.copy()
                else:
                    df_unassigned_toobright = pd.concat(
                        [df_unassigned_toobright, df_tmp]
                    )

                logger.warning(
                    f"[Validation of output] There are {len(df_gaia_toobright)} bright stars nearby fiber {unfib}: {df_tmp}"
                )

    return df_unassigned_toobright


def _build_df_fib(pfsDesign0):
    """Construct the per-fiber DataFrame used for plotting and checks.

    This preserves the original logic and column names.
    """
    pfsflux = np.array([a[0] if len(a) > 0 else np.nan for a in pfsDesign0.psfFlux])
    totalflux = np.array([a[0] if len(a) > 0 else np.nan for a in pfsDesign0.totalFlux])

    # Combine the pfs/total Flux for validation plot
    pfsflux_l = []
    pfsflux_f = []
    try:
        filt = pfsDesign0[pfsDesign0.targetType == 3].filterNames[0][0]
    except Exception:
        filt = None

    for t, fl, a, b in zip(
        pfsDesign0.targetType, pfsDesign0.filterNames, pfsDesign0.totalFlux, pfsDesign0.psfFlux
    ):
        if t == 3: #TargetType.FLUXSTD
            pfsflux_l.append(b[0])
            pfsflux_f.append(fl[0])
        elif t == 1: #TargetType.SCIENCE
            if filt is not None and filt in fl:
                # Preserve the original branching behaviour (note: original used `if not np.nan`)
                pfsflux_l.append(a[fl.index(filt)] if not np.isnan(a[fl.index(filt)]) else b[fl.index(filt)])
                pfsflux_f.append(filt)
            else:
                indices = [i for i, item in enumerate(fl) if item != "none"]
                pfsflux_l.append(a[indices[0]] if not np.isnan(a[indices[0]]) else b[indices[0]])
                pfsflux_f.append(fl[indices[0]])
        else:
            pfsflux_l.append(np.nan)
            pfsflux_f.append("none")

    pfsflux_plot = np.array(pfsflux_l)
    pfsflux_plot_filter = np.array(pfsflux_f)

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
                pfsflux_plot,
                pfsflux_plot_filter,
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
            "pfsFlux_plot",
            "pfsflux_plot_filter",
        ],
    )

    # Set column types and derived columns
    df_fib["proposalId"] = pfsDesign0.proposalId
    df_fib["pfsFlux"] = df_fib.pfsFlux.astype(float)
    df_fib["totalFlux"] = df_fib.totalFlux.astype(float)
    df_fib["pfsFlux_plot"] = df_fib.pfsFlux_plot.astype(float)
    df_fib["pfi_x"] = df_fib.pfi_x.astype(float)
    df_fib["pfi_y"] = df_fib.pfi_y.astype(float)
    df_fib["cadId"] = df_fib.catId.astype(int)
    df_fib["targetType"] = df_fib.targetType.astype(int)
    df_fib["fiberId"] = df_fib.fiberId.astype(int)

    df_fib["psfMag"] = df_fib["pfsFlux"].apply(njy_mag)
    df_fib["totalMag"] = df_fib["totalFlux"].apply(njy_mag)
    df_fib["obCode"] = pfsDesign0.obCode
    df_fib["pfsMag_plot"] = df_fib["pfsFlux_plot"].apply(njy_mag)

    return df_fib


def _build_df_ag(pfsDesign0):
    """Return a DataFrame of AG positions in PFI coordinates for plotting."""
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
    return df_ag


def validation(parentPath, figpath, save, show, ssp, conf):
    """Run validation for all PfsDesigns found in the summary at `parentPath`.

    Parameters
    - parentPath: top-level directory containing PPP or SSP outputs
    - figpath: directory to write figures and the HTML report
    - save, show: booleans controlling plotting behaviour
    - ssp: if True, use SSP summary layout; otherwise use PPP layout
    - conf: configuration dict used for external queries and data paths
    """
    # Load design summary and normalize observation times
    pfsDesignDir, df_design = _load_design_summary(parentPath, ssp)

    # Compute InR/El at start and stop times
    df_design = _add_inr_columns(df_design)


    # InR/El columns were added by _add_inr_columns above.
    # (Kept for backward compatibility and readability.)

    # Make pdf files and store the statistical data
    pfsDesignIds = (
        df_design["design_filename"]
        .str.split(r"-|\.", regex=True, expand=True)[1]
        .map(lambda x: int(x, 16))
    )

    # This routine just combines a few cells in "trial" section
    df_ch = pldes.init_check_design()

    # Bench and fiber mappings are shared across designs; set them up once.
    bench = sfa.nfutils.getBench(
        conf["packages"]["pfs_instdata_dir"],
        conf["sfa"]["cobra_coach_dir"],
        conf["sfa"]["cobra_coach_module_version"],
        conf["sfa"]["sm"],
        conf["sfa"]["dot_margin"],
    )

    fibId = FiberIds(
        path=os.path.join(conf["packages"]["pfs_utils_dir"], "data", "fiberids")
    )

    # Accumulate bright sources near unassigned fibers across all designs
    df_all_unassigned_toobright = pd.DataFrame()

    count = 0
    for designId in pfsDesignIds:
        pfsDesign0 = PfsDesign.read(designId, dirName=pfsDesignDir)
        pfsDesign0.validate()
        print(f"{pfsDesign0.designName}, {pfsDesign0.arms}")

        # check fiber duplicates
        _warn_duplicate_fibers(pfsDesign0)

        # check bright stars in the guiding field
        ppc_ra = pfsDesign0.raBoresight
        ppc_dec = pfsDesign0.decBoresight
        ppc_pa = pfsDesign0.posAng
        if not ssp:
            ppc_obstime_utc = df_design["observation_time"][count]
        else:
            ppc_obstime_utc = df_design["ppc_obstime_utc"][count]

        count += 1

        df_guidestars_toobright = _warn_too_bright_guidestars(ppc_ra, ppc_dec, ppc_pa, ppc_obstime_utc, conf)

        # check bright stars nearby unassigned fibers including disabled fibers
        # Use pre-initialized bench and fiber-id mapping created outside the loop.

        # Find bright Gaia sources near unassigned fibers
        df_unassigned_toobright = _find_unassigned_bright_nearby(
            pfsDesign0, bench, fibId, ppc_ra, ppc_dec, ppc_pa, conf
        )

        # Accumulate unassigned-bright objects to write once after all designs processed
        if not df_unassigned_toobright.empty:
            if len(df_all_unassigned_toobright) == 0:
                df_all_unassigned_toobright = df_unassigned_toobright.copy()
            else:
                df_all_unassigned_toobright = pd.concat(
                    [df_all_unassigned_toobright, df_unassigned_toobright],
                    ignore_index=True,
                )

        # Build per-fiber DataFrame and check magnitudes
        df_fib = _build_df_fib(pfsDesign0)

        df_too_bright = df_fib[(df_fib["psfMag"] < 13) | (df_fib["totalMag"] < 13)]
        if not df_too_bright.empty:
            logger.warning(
                f"[Validation of output] Too bright sources with psfMag or totalMag < 13 (0x{designId:016x}, {pfsDesign0.designName}): {df_too_bright}"
            )

        df_fib["sector"] = pldes.get_field_sector2(df_fib)

        df_ag = _build_df_ag(pfsDesign0)

        if not df_unassigned_toobright.empty:
            unfib_bright = df_unassigned_toobright["fiber_id"].unique().tolist()
        else:
            unfib_bright = []

        ppc_code_ = pfsDesign0.designName
        df = pldes.check_design(designId, ppc_code_, df_fib, df_ag, df_guidestars_toobright, n_unfib_bright=len(unfib_bright))
        df_ch = pd.concat([df_ch, df], ignore_index=True)
        title = f"designId=0x{designId:016x} ({pfsDesign0.raBoresight:.2f},{pfsDesign0.decBoresight:.2f},PA={pfsDesign0.posAng:.1f})\n{ppc_code_}"
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
            conf=conf,
            unfib_bright=unfib_bright,
        )

    # After processing all designs, optionally save a single CSV containing
    # all bright sources found near unassigned fibers.
    if conf["validation"]["save_unassign_toobright"] and not df_all_unassigned_toobright.empty:
        out_path = os.path.join(figpath, "df_unassign_bright_nearby.csv")
        df_all_unassigned_toobright.to_csv(out_path, index=False)

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
        .applymap(pldes.colour_background_warning_unfib, subset=["unfib_bright"])
        .format(precision=1)
    )
    # """

    # Convert the Styler object to an HTML string
    html_str = styled_html.to_html()

    with open(os.path.join(figpath, "validation_report.html"), "w") as f:
        f.write(html_str)
        f.close()
