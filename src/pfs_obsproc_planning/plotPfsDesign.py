"""Plot utilities for PFS designs.

This module provides plotting helpers to visualize a PFS design (fiber
positions, guide stars, flux standards, sky fibers, histograms, and AG
counts). The focus is readability: add docstrings, small helpers, and
clear inline comments while preserving original behavior.
"""

# import os,sys,re
# import math as mt
import numpy as np
import re

# from scipy import ndimage
# from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from logzero import logger

# event
# from eventPlot import PointBrowser
import matplotlib.patches as patches

import pandas as pd
import itertools as it

# from scipy import interpolate as ipol
# from PIL import Image
from pfs.utils.fiberids import FiberIds

"""
    TargetType.BLACKSPOT: 10
    TargetType.FLUXSTD: 3
    TargetType.SCIENCE: 1
    TargetType.SKY: 2
    TargetType.UNASSIGNED: 4
 """


def get_ag_counts(df_ag, threshold=19, radius=245.0):
    """Compute guide-star counts per camera and positions for annotation.

    This helper returns a tuple (agnum_table, ag_positions) where:
      - agnum_table is [[total_cam1, total_cam2, ...], [bright_cam1, bright_cam2, ...]]
      - ag_positions is list of tuples (count, (xpos, ypos)) for annotation in plotting order

    Parameters
    ----------
    df_ag : pandas.DataFrame
        Guide-star table with 'camId' and 'agMag' and 'ag_pfi_x', 'ag_pfi_y' columns.
    threshold : float
        Magnitude threshold for considering a guide star 'bright'.
    radius : float
        Radial distance from the center where camera labels are placed.

    Returns
    -------
    (agnum_table, ag_positions)
    """
    agnum_table = []
    ag_positions = []

    # iterate cameras 0..5 and compute counts and simple label positions
    for i in range(1, 7):
        cam_idx = i - 1
        ang = np.deg2rad(-60 * cam_idx - 9)
        total = int((df_ag.camId == cam_idx).sum())
        bright = int(((df_ag.camId == cam_idx) & (df_ag.agMag <= threshold)).sum())
        agnum_table.append([total, bright])
        # label positions at circle radius with angle same as original code
        xpos = radius * np.cos(ang)
        ypos = radius * np.sin(ang)
        ag_positions.append((total, (xpos, ypos)))

    return agnum_table, ag_positions


# ---------------------- Drawing helpers ----------------------
def _draw_fov_points(ax, df_fib, df_ag, unfib_bright, c, alpha, s):
    """Draw scatter points on the FoV axes (unassigned, sky, std, sci, AGs).

    This consolidates the multiple consecutive scatter() calls into a single
   , well-named helper for readability.
    """
    # unassigned (targetType == 4)
    ax.scatter(
        df_fib[df_fib.targetType == 4].pfi_x,
        df_fib[df_fib.targetType == 4].pfi_y,
        c=c["un"],
        marker="x",
        s=s * 1.3,
        alpha=alpha,
        lw=1,
        label=f"UNASSIGNED ({len(df_fib[df_fib.targetType==4].pfi_y)})",
    )

    # highlight unassigned near bright stars
    if len(unfib_bright) > 0:
        ax.scatter(
            df_fib[df_fib.fiberId.isin(unfib_bright)].pfi_x,
            df_fib[df_fib.fiberId.isin(unfib_bright)].pfi_y,
            facecolor="none",
            edgecolor=c["un_brt"],
            marker="s",
            s=s * 2.6,
            alpha=alpha,
            lw=1,
            label=f"nearby bright star ({len(unfib_bright)})",
        )

    # black spots
    ax.scatter(
        df_fib[df_fib.targetType == 10].pfi_x,
        df_fib[df_fib.targetType == 10].pfi_y,
        c=c["dot"],
        marker="o",
        s=s,
        alpha=alpha,
        lw=0,
        label=f"BLACKSPOT ({len(df_fib[df_fib.targetType==10].pfi_y)})",
    )

    # sky, fluxstd, science
    ax.scatter(
        df_fib[df_fib.targetType == 2].pfi_x,
        df_fib[df_fib.targetType == 2].pfi_y,
        c=c["sky"],
        marker="o",
        s=s,
        alpha=alpha,
        lw=0,
        label=f"SKY ({len(df_fib[df_fib.targetType==2].pfi_y)})",
    )

    ax.scatter(
        df_fib[df_fib.targetType == 3].pfi_x,
        df_fib[df_fib.targetType == 3].pfi_y,
        c=c["fstar"],
        marker="o",
        s=s * 2.0,
        alpha=alpha,
        lw=0,
        label=f"FLUXSTD ({len(df_fib[df_fib.targetType==3].pfi_y)})",
    )

    ax.scatter(
        df_fib[df_fib.targetType == 1].pfi_x,
        df_fib[df_fib.targetType == 1].pfi_y,
        c=c["sci"],
        marker="o",
        s=s,
        alpha=alpha,
        lw=0,
        label=f"SCIENCE ({len(df_fib[df_fib.targetType==1].pfi_y)})",
    )

    # special highlight (kept for backward compatibility)
    mask_special = df_fib.obCode == "M31_30410_R31_02730_v2"
    if mask_special.any():
        ax.scatter(
            df_fib[mask_special].pfi_x,
            df_fib[mask_special].pfi_y,
            c="red",
            marker="o",
            s=s,
            alpha=alpha,
            lw=0,
        )

    # guide stars
    ax.scatter(
        df_ag.ag_pfi_x,
        df_ag.ag_pfi_y,
        c=c["ag"],
        marker="s",
        s=s,
        alpha=alpha,
        lw=0,
        label="guidestar",
    )


def _draw_sector_lines(ax):
    """Draw circular section borders and radial sector lines on the FoV plot."""
    r = 235
    rs1 = 50
    rs2 = 132.5

    for rs in [rs1, rs2]:
        circle = patches.Circle((0, 0), radius=rs, color="navy", fill=False, lw=0.5)
        ax.add_patch(circle)

    phis = np.radians(np.linspace(0, 360, 6, endpoint=False) + 15 + 90)
    phis = phis - np.radians(360 / 60.0 / 2.0)
    for phi in phis:
        xx1, yy1 = rs1 * np.cos(phi), rs1 * np.sin(phi)
        xx2, yy2 = rs2 * np.cos(phi), rs2 * np.sin(phi)
        line = patches.FancyArrow(
            xx1, yy1, (xx2 - xx1), (yy2 - yy1), head_width=0, color="navy", lw=0.5
        )
        ax.add_patch(line)

    phis = np.radians(np.linspace(0, 360, 13, endpoint=False) - 20 + 90)
    phis = phis - np.radians(360 / 13.0 / 2.0)
    for phi in phis:
        xx1, yy1 = rs2 * np.cos(phi), rs2 * np.sin(phi)
        xx2, yy2 = r * np.cos(phi), r * np.sin(phi)
        line = patches.FancyArrow(
            xx1, yy1, (xx2 - xx1), (yy2 - yy1), head_width=0, color="navy", lw=0.5
        )
        ax.add_patch(line)


def _draw_ne_arrows(ax, pa):
    """Draw North and East arrows and annotate them on the FoV plot."""
    de, dn = calc_nedirection(pa)
    dl = 20.0
    posne = np.array([200, -200])
    arr_e = patches.FancyArrow(
        posne[0], posne[1], dl * de[0], dl * de[1], color="dimgrey", width=3.0
    )
    arr_n = patches.FancyArrow(
        posne[0], posne[1], dl * dn[0], dl * dn[1], color="dimgrey", width=3.0
    )
    ax.add_patch(arr_e)
    ax.add_patch(arr_n)
    ax.text(
        posne[0] + dl * 2.5 * de[0],
        posne[1] + dl * 2.5 * de[1],
        f"E",
        ha="center",
        va="center",
        fontsize=10,
        color="dimgrey",
    )
    ax.text(
        posne[0] + dl * 2.5 * dn[0],
        posne[1] + dl * 2.5 * dn[1],
        f"N",
        ha="center",
        va="center",
        fontsize=10,
        color="dimgrey",
    )


def _draw_histograms(ax2, ax3, df_fib, df_ag, conf):
    """Plot histograms for std stars and AG magnitudes.

    Maintains behavior of the original code but is more readable.
    """
    # mag bins
    bins = 10
    mmin = 13
    mmax = 25
    bins = int((mmax - mmin) * 2)

    c = {
        "fstar": "green",
        "ag": "blueviolet",
    }

    # Flux standard histogram
    filtername_std = df_fib[df_fib.targetType == 3].pfsflux_plot_filter.values[0]
    ax2.hist(
        df_fib[df_fib.targetType == 3]["pfsMag_plot"],
        bins=bins,
        range=(mmin, mmax),
        color=c["fstar"],
        lw=0,
        alpha=0.4,
        label=f"Std star ({filtername_std}, {len(df_fib[df_fib.targetType==3]['pfsMag_plot'])})",
    )

    # stacked histogram per proposal/filter
    df_mags = df_fib[["proposalId", "pfsMag_plot", "pfsflux_plot_filter"]].groupby(by=["proposalId", "pfsflux_plot_filter"], as_index=False)
    c_list = it.cycle(["salmon", "royalblue", "orange", "limegreen", "violet"])
    c_sci = []
    mag_per_prog = np.zeros(df_fib.shape[0])
    label_per_prog = []

    for k, v in df_mags.groups.items():
        if k[0] == "N/A":
            continue
        mags = df_fib.pfsMag_plot[v].values
        n_mags = len(mags)
        n_too_bright = sum(mags < 13)
        for j in range((df_fib.shape[0] - n_mags)):
            mags = np.append(mags, np.nan)
        mag_per_prog = np.vstack((mag_per_prog, mags))
        label_per_prog.append(f"{p} ({n_mags}; {n_too_bright}<13mag)")
    """
    for k, v in df_mags.groups.items():
        if k[0] == 'N/A': continue
        mags = df_fib.pfsMag_plot[v].values
        n_mags = len(mags)
        n_too_bright = sum(mags<13)
        for j in range((df_fib.shape[0] - n_mags)):
            mags = np.append(mags, np.nan)
        #print(mag_per_prog.shape, mags.shape)
        mag_per_prog = np.vstack((mag_per_prog, mags))
        if conf["ppp"]["mode"] == "classic":
            if k[0] in conf["sfa"]["proposalIds_obsFiller"]:
                label_per_prog.append(f"obs. filler ({n_mags}; {n_too_bright}<13mag)")
            elif k[0] in conf["ppp"]["proposalIds"]:
                label_per_prog.append(f"{k[0]} ({n_mags}; {n_too_bright}<13mag)")
            else:
                label_per_prog.append(f"usr filler ({n_mags}; {n_too_bright}<13mag)")
        else:
            label_per_prog.append(f"{k[0]} ({k[1]}, {n_mags}; {n_too_bright}<13mag)")
        c_sci.append(next(c_list))

    mag_per_prog = mag_per_prog[1:]
    ax2.hist(
        mag_per_prog.T,
        bins=bins,
        range=(mmin, mmax),
        histtype="barstacked",
        color=c_sci,
        lw=0,
        alpha=0.4,
        label=label_per_prog,
        stacked=True,
    )
    ax2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.2, 1.2), ncol=2)
    ax2.set_xlabel("mag", fontsize=12)
    ax2.set_ylabel("N (target or STD)", fontsize=12)

    # AG histogram
    ax3.hist(df_ag["agMag"], bins=bins, range=(mmin, mmax), color=c["ag"], lw=0, alpha=0.4)
    ax3.set_ylabel("N (AG)", fontsize=12)
    ax3.set_xlabel("mag", fontsize=12)


def _draw_ag_table(ax4, agnum_table, agnum_threshold):
    """Draw the AG count table (2 rows: total and bright<=threshold)"""
    ax4.table(
        agnum_table,
        loc='center',
        colLabels=['AG1', 'AG2', 'AG3', 'AG4', 'AG5', 'AG6'],
        rowLabels=['Total', f"<={agnum_threshold} mag"],
        colWidths=[0.1] * 6,
    )
    ax4.tick_params(
        axis='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
        labelsize=7,
    )
    ax4.set_frame_on(False)

def get_pfs_utils_path():
    try:
        import eups

        logger.info(
            "eups was found. "
            "No attempt to find a pfs_utils directory is made. "
            "Please set an appropriate PFS_UTILS_DIR"
        )

        return None

    except ModuleNotFoundError:
        try:
            from pathlib import Path

            import pfs.utils

            p = Path(pfs.utils.__path__[0])
            p_fiberdata = p.parent.parent.parent / "data" / "fiberids"
            if p_fiberdata.exists():
                logger.info(
                    f"pfs.utils's fiber data directory {p_fiberdata} was found and will be used."
                )
                return p_fiberdata
            else:
                raise FileNotFoundError
        except ModuleNotFoundError as e:
            logger.exception(e)
            return None
        except FileNotFoundError as e:
            logger.exception(e)
            # print("pfs_utils/data/fiberids cannot be found automatically")
            return None


def plot_FoV(
    df_fib,
    df_ag,
    alpha=1.0,
    title="",
    fname="",
    save=True,
    show=True,
    pa=0.0,
    conf=None,
    unfib_bright=None,
):
    """Create a multi-panel figure summarizing a PFS design FoV.

    Parameters
    ----------
    df_fib : pandas.DataFrame
        Fiber table with fields like 'targetType', 'pfi_x', 'pfi_y', 'spec', 'fh', 'pfsFlux', 'proposalId'
    df_ag : pandas.DataFrame
        Guide-star table with fields like 'agx', 'agy', 'agMag', 'camId', 'ag_pfi_x', 'ag_pfi_y'
    unfib_bright : list, optional
        FiberIds of unassigned fibers near bright stars; used to highlight them in the plot.

    Returns
    -------
    fig.tight_layout() result
    """
    # Avoid mutable default
    if unfib_bright is None:
        unfib_bright = []

    # Ensure common expected columns exist (light defensive typing/formatting)
    df_fib = df_fib.copy()
    df_ag = df_ag.copy()
    """
    df_fib: pandas dataframe with
           'targetType', 'pfi_x', 'pfi_y', 'spec', 'fh', 'pfsFlux', 'proposalId'
    df_ag: pandas dataframe with
           'agx', 'agy', 'agMag', 'camId', 'ag_pfi_x', 'ag_pfi_y'
    """

    fiberIds = FiberIds(get_pfs_utils_path())
    df_fibId = pd.DataFrame(fiberIds.data)

    """
    ## it seems ideal to handle these before calling this function.
    df_fib["cadId"] = df_fib.catId.astype(int)
    df_fib["pfsFlux"] = df_fib.pfsFlux.astype(float)
    df_fib["pfi_x"] = df_fib.pfi_x.astype(float)
    df_fib["pfi_y"] = df_fib.pfi_y.astype(float)
    df_fib["targetType"] = df_fib.targetType.astype(int)
    df_fib["fiberId"] = df_fib.fiberId.astype(int)
    """
    df_fib["obCode"] = df_fib.obCode.astype(str)

    # set the font sizes for labels

    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)

    # a quick kludge to filter out bad quality points in poorly focussed images
    # ind=np.arange(len(val))

    # Get x,y position from grand fiber map for unassigned fiber.
    uafib = df_fib[df_fib.targetType == 4].fiberId
    for f in uafib:
        df_fib.loc[df_fib.fiberId == f, "pfi_x"] = df_fibId.loc[
            df_fibId.fiberId == f, "x"
        ].values
        df_fib.loc[df_fib.fiberId == f, "pfi_y"] = df_fibId.loc[
            df_fibId.fiberId == f, "y"
        ].values

    # plt.clf()
    fig = plt.figure(
        num="design_FoV",
        figsize=(12, 9),
        dpi=80,
        clear=True,
        facecolor="w",
        edgecolor="k",
    )
    gs = fig.add_gridspec(5, 2)
    ax1 = fig.add_subplot(gs[:4,0], aspect='equal')
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax3 = fig.add_subplot(gs[2:4, 1], sharex=ax2)
    ax4 = fig.add_subplot(gs[4,:])

    c = {
        "un": "slategrey",
        "un_brt": "brown",
        "dot": "black",
        "sky": "deepskyblue",
        "fstar": "green",
        "sci": "salmon",
        "ag": "blueviolet",
    }
    c_list = it.cycle(["salmon", "royalblue", "orange", "limegreen", "violet"])

    # point marker size
    s = 10.0

    # draw scatter points and guide stars on the main FoV axes (delegated)
    _draw_fov_points(ax1, df_fib, df_ag, unfib_bright, c, alpha, s)

    # store AG counts and annotation positions
    agnum_threshold = 19
    agnum_table, ag_positions = get_ag_counts(df_ag, threshold=agnum_threshold)

    # Annotate camera positions with counts
    for cam_index, (count, (xpos, ypos)) in enumerate(ag_positions, start=1):
        ax1.text(xpos, ypos, f"{cam_index} ({count})", ha="center", fontsize=10)

    # transpose table for the Axes.table layout (rows: total, bright<=threshold)
    agnum_table = [list(i) for i in zip(*agnum_table)]

    ax1.legend(fontsize="x-small", loc='upper left', bbox_to_anchor=(0.2, 1.2), ncol=3)
    ax1.set_xlim(xmin=-255, xmax=255)
    ax1.set_ylim(ymin=-255, ymax=255)

    # axis labels for clarity
    ax1.set_xlabel("PFI X [mm]", fontsize=12)
    ax1.set_ylabel("PFI Y [mm]", fontsize=12)

    # border sectors
    _draw_sector_lines(ax1)

    # north/east markers
    _draw_ne_arrows(ax1, pa)

    # plot histograms for flux standards, stacked proposals, and AG magnitudes
    _draw_histograms(ax2, ax3, df_fib, df_ag, conf)

    # draw AG summary table in the reserved axis
    _draw_ag_table(ax4, agnum_table, agnum_threshold)

    # AG magnitude table
    ax4.table(agnum_table, loc='center',
              colLabels=['AG1', 'AG2', 'AG3', 'AG4', 'AG5', 'AG6'],
              rowLabels=['Total', f"<={agnum_threshold} mag"],
              colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ax4.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False,
                     labelsize=7)
    ax4.set_frame_on(False)
    
    fig.suptitle(title, fontsize=12)

    if save == True:
        plt.savefig(fname + ".pdf")
    if show == True:
        plt.show()

    return fig.tight_layout()


warning = "hotpink"  # colour for warning


def colour_background_warning_sky_tot(val):
    """Return table cell background CSS when total sky is too low."""
    colour = warning if float(val) < 400 else ""

    return f"background-color: {colour}"


def colour_background_warning_std_tot(val):
    """Return table cell background CSS when total standards are too low."""
    colour = warning if float(val) < 40 else ""

    return f"background-color: {colour}"


def colour_background_warning_sky_min(val):
    """Return table cell background CSS when minimum per-sector sky is too low."""
    colour = warning if float(val) < 12 else ""

    return f"background-color: {colour}"


def colour_background_warning_std_min(val):
    """Return table cell background CSS when minimum per-sector std is too low."""
    colour = warning if float(val) < 3 else ""

    return f"background-color: {colour}"


def colour_background_warning_ag_tot(val):
    # extract numbers: e.g. '3 (1)' -> ['3', '1']
    nums = list(map(int, re.findall(r"\d+", str(val))))
    i = nums[0] if nums else 0
    j = nums[1] if len(nums) > 1 else 0
    colour = warning if (i < 10 or j > 0) else ""

    return f"background-color: {colour}"


def colour_background_warning_ag_min(val):
    # extract numbers: e.g. '3 (1)' -> ['3', '1']
    nums = list(map(int, re.findall(r"\d+", str(val))))
    i = nums[0] if nums else 0
    j = nums[1] if len(nums) > 1 else 0
    colour = warning if (i < 2 or j > 0) else ""

    return f"background-color: {colour}"

def colour_background_warning_inr(val):
    colour = warning if (float(val) < -174) or (float(val) > 174) else ""

    return f"background-color: {colour}"


def colour_background_warning_el(val):
    colour = warning if (float(val) < 32) or (float(val) > 75) else ""

    return f"background-color: {colour}"


def colour_background_warning_unfib(val):
    colour = warning if float(val) > 0 else ""

    return f"background-color: {colour}"


def set_style(df_ch):
    styles = df_ch.copy()
    # styles.style.applymap(colour_background_warning_sky_min, subset=['sky_min']).applymap(colour_background_warning_std_min, subset=['std_min']).applymap(colour_background_warning_sky_tot, subset=['sky_sum']).applymap(colour_background_warning_std_tot, subset=['std_sum']).applymap(colour_background_warning_ag_min, subset=['ag1', 'ag2', 'ag3', 'ag4', 'ag5', 'ag6']).applymap(colour_background_warning_ag_min, subset=['ag_sum']).format(precision=1)
    styles.style.applymap(colour_background_warning_sky_min, subset=["sky_min"])

    return styles


def init_check_design():
    df = pd.DataFrame(
        data=None,
        columns=[
            "sky_mean",
            "sky_std",
            "sky_min",
            "sky_max",
            "sky_sum",
            "std_mean",
            "std_std",
            "std_min",
            "std_max",
            "std_sum",
            "ag1",
            "ag2",
            "ag3",
            "ag4",
            "ag5",
            "ag6",
            "ag_sum",
            "designId",
            "ppc_code",
            "unfib_bright",
        ],
    )
    return df


def check_design(designId, df_fib, df_ag, df_guidestars_toobright):
    a, a_ = check_ags(df_ag, df_guidestars_toobright)
    ag_cols = [
        f"{ai} ({aj})" if aj > 0 else f"{ai}"
        for ai, aj in zip(a, a_)
    ]
    
    vals = check_fibers(df_fib)

    df_ch = pd.DataFrame(
        data=np.append(vals, ag_cols).reshape(1, len(vals) + len(ag_cols)),
        columns=[
                "sky_mean", "sky_std", "sky_min", "sky_max", "sky_sum",
                "std_mean", "std_std", "std_min", "std_max", "std_sum",
                "ag1", "ag2", "ag3", "ag4", "ag5", "ag6", "ag_sum",
            ],
    )

    df_ch["designId"] = f"0x{designId:016x}"
    # df_ch.style.applymap(colour_background_warning_sky_min, subset=['sky_min'])

    return df_ch


def check_fibers(df_fib):
    """
    df_fib: pandas dataframe with
           'targetType', 'pfi_x', 'pfi_y', 'spec', 'fh', 'pfsFlux', 'pfsFlux', 'sector'
    """

    targets = df_fib["targetType"]
    f = []
    # sky
    try:
        arr = (
            df_fib[df_fib["targetType"] == 2]
            .groupby(by="sector")["targetType"]
            .count()
            .values
        )
        f.extend((np.mean(arr), np.std(arr), np.min(arr), np.max(arr), np.sum(arr)))
        if arr.size < 12:
            f[2] = 0.0
    except ValueError:
        f.extend((np.nan, np.nan, np.nan, np.nan, np.nan))
    # std
    try:
        arr = (
            df_fib[df_fib["targetType"] == 3]
            .groupby(by="sector")["targetType"]
            .count()
            .values
        )
        f.extend((np.mean(arr), np.std(arr), np.min(arr), np.max(arr), np.sum(arr)))
        if arr.size < 12:
            f[7] = 0.0
    except ValueError:
        f.extend((np.nan, np.nan, np.nan, np.nan, np.nan))

    return np.array(f)


def check_ags(df_ag, df_guidestars_toobright):
    """
    df_ag: pandas dataframe with
           'agx', 'agy', 'agMag', 'camId', 'ag_pfi_x', 'ag_pfi_y'
    """

    cam = df_ag["camId"].values
    cam_ = df_guidestars_toobright["agId"].values
    a = []
    a_ = []
    for i in range(0, 6):
        a.append(np.sum(cam == i))
        a_.append(np.sum(cam_ == i))

    a.append(np.sum(a))
    a_.append(np.sum(a_))
    return np.array(a), np.array(a_)


def get_field_sector(df_fib):
    """
    df_fib: pandas dataframe with
           'targetType', 'pfi_x', 'pfi_y', 'spec', 'fh', 'pfsFlux', 'pfsFlux'
    divede sectors to 12 regions / innner and outer of 60-deg pi-shape
    """

    r = np.sqrt(df_fib["pfi_x"] * df_fib["pfi_x"] + df_fib["pfi_y"] * df_fib["pfi_y"])
    t = np.rad2deg(np.arctan2(df_fib["pfi_y"], df_fib["pfi_x"])) + 180.0

    r0 = 220 / np.sqrt(2)

    if r <= r0:
        s = 1
    else:
        s = 2
    if 0 <= t < 60:
        s += 10
    elif 60 <= t < 120:
        s += 20
    elif 120 <= t < 180:
        s += 30
    elif 180 <= t < 240:
        s += 40
    elif 240 <= t < 300:
        s += 50
    else:
        s += 60

    return s


def get_field_sector2(df_fib):
    """
    df_fib: pandas dataframe with
           'targetType', 'pfi_x', 'pfi_y', 'spec', 'fh', 'pfsFlux', 'pfsFlux'
    divede sectors to 20 (1 + 6 + 13) regions, proposed by Laszlo
    """

    points = []
    # centre points
    points.append([[0, 0]])

    # inner points
    phi = np.radians(np.linspace(0, 360, 6, endpoint=False) + 15 + 90)
    points.append(100 * np.stack([np.cos(phi), np.sin(phi)], axis=-1))

    # outer points
    phi = np.radians(np.linspace(0, 360, 13, endpoint=False) - 20 + 90)
    points.append(175 * np.stack([np.cos(phi), np.sin(phi)], axis=-1))

    xy = np.concatenate(points, axis=0)

    # Cobra centres
    uv = np.stack([df_fib.pfi_x, df_fib.pfi_y], axis=-1)

    # Find the closest point to each cobra centre
    d = distance_matrix(xy, uv)

    tag = np.argmin(d, axis=0)
    # tag.shape

    return tag


def calc_nedirection(pa):
    """
    pa: position angle [deg]
    """

    ne = np.array([[0.0, 1.0], [-1.0, 0.0]])
    a = np.deg2rad(pa)
    rot = np.array([[np.cos(a), -1.0 * np.sin(a)], [np.sin(a), np.cos(a)]])

    dne = np.matmul(ne, rot)

    return dne[0], dne[1]
