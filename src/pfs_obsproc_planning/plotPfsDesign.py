# import os,sys,re
# import math as mt
import numpy as np

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
    df_fib, df_ag, alpha=1.0, title="", fname="", save=True, show=True, pa=0.0
):
    """
    df_fib: pandas dataframe with
           'targetType', 'pfi_x', 'pfi_y', 'spec', 'fh', 'pfsFlux', 'proposalId'
    df_ag: pandas dataframe with
           'agx', 'agy', 'agMag', 'camId', 'ag_pfi_x', 'ag_pfi_y'
    """

    fiberIds = FiberIds(get_pfs_utils_path())
    df_fibId = pd.DataFrame(fiberIds.data)

    ## it seems ideal to handle these before calling this function.
    df_fib["cadId"] = df_fib.catId.astype(int)
    df_fib["pfsFlux"] = df_fib.pfsFlux.astype(float)
    df_fib["pfi_x"] = df_fib.pfi_x.astype(float)
    df_fib["pfi_y"] = df_fib.pfi_y.astype(float)
    df_fib["targetType"] = df_fib.targetType.astype(int)
    df_fib["fiberId"] = df_fib.fiberId.astype(int)

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
        figsize=(10, 6),
        dpi=80,
        clear=True,
        facecolor="w",
        edgecolor="k",
    )
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0], aspect="equal")
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

    c = {
        "un": "slategrey",
        "dot": "black",
        "sky": "deepskyblue",
        "fstar": "green",
        "sci": "salmon",
        "ag": "blueviolet",
    }
    c_list = it.cycle(["salmon", "royalblue", "orange", "limegreen", "violet"])

    s = 10.0
    ax1.scatter(
        df_fib[df_fib.targetType == 4].pfi_x,
        df_fib[df_fib.targetType == 4].pfi_y,
        c=c["un"],
        marker="x",
        s=s * 1.3,
        alpha=alpha,
        lw=1,
        label=f"UNASSIGNED ({len(df_fib[df_fib.targetType==4].pfi_y)})",
    )
    ax1.scatter(
        df_fib[df_fib.targetType == 10].pfi_x,
        df_fib[df_fib.targetType == 10].pfi_y,
        c=c["dot"],
        marker="o",
        s=s,
        alpha=alpha,
        lw=0,
        label=f"BLACKSPOT ({len(df_fib[df_fib.targetType==10].pfi_y)})",
    )
    ax1.scatter(
        df_fib[df_fib.targetType == 2].pfi_x,
        df_fib[df_fib.targetType == 2].pfi_y,
        c=c["sky"],
        marker="o",
        s=s,
        alpha=alpha,
        lw=0,
        label=f"SKY ({len(df_fib[df_fib.targetType==2].pfi_y)})",
    )
    ax1.scatter(
        df_fib[df_fib.targetType == 3].pfi_x,
        df_fib[df_fib.targetType == 3].pfi_y,
        c=c["fstar"],
        marker="o",
        s=s * 2.0,
        alpha=alpha,
        lw=0,
        label=f"FLUXSTD ({len(df_fib[df_fib.targetType==3].pfi_y)})",
    )
    ax1.scatter(
        df_fib[df_fib.targetType == 1].pfi_x,
        df_fib[df_fib.targetType == 1].pfi_y,
        c=c["sci"],
        marker="o",
        s=s,
        alpha=alpha,
        lw=0,
        label=f"SCIENCE ({len(df_fib[df_fib.targetType==1].pfi_y)})",
    )
    ax1.scatter(
        df_ag.ag_pfi_x,
        df_ag.ag_pfi_y,
        c=c["ag"],
        marker="s",
        s=s,
        alpha=alpha,
        lw=0,
        label="guidestar",
    )
    for i in range(1, 7):
        ang = np.deg2rad(-60 * (i - 1) - 9)
        agn = len(df_ag[df_ag.camId == (i - 1)].ag_pfi_x)
        ax1.text(
            245 * np.cos(ang),
            245 * np.sin(ang),
            f"{i} ({agn})",
            ha="center",
            fontsize=10,
        )

    ax1.legend(fontsize="x-small", bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax1.set_xlim(xmin=-255, xmax=255)
    ax1.set_ylim(ymin=-255, ymax=255)

    xlname = "PFI X [mm]"
    ylname = "PFI Y [mm]"

    ax1.set_xlabel(xlname, fontsize=12)
    ax1.set_ylabel(ylname, fontsize=12)

    # Border of sections
    r = 235
    rs1 = 50
    rs2 = 132.5
    for rs in [rs1, rs2]:
        circle = patches.Circle((0, 0), radius=rs, 
                                color='navy', fill=False, lw=0.5)
        ax1.add_patch(circle)
    phis = np.radians(np.linspace(0, 360, 6, endpoint=False) + 15 + 90)
    phis = phis - np.radians(360/60./2.)
    for phi in phis:
        xx1, yy1 = rs1*np.cos(phi), rs1*np.sin(phi)
        xx2, yy2 = rs2*np.cos(phi), rs2*np.sin(phi)
        line = patches.FancyArrow(xx1, yy1, (xx2-xx1), (yy2-yy1),
                                  head_width=0, color='navy', lw=0.5)
        ax1.add_patch(line)
    phis = np.radians(np.linspace(0, 360, 13, endpoint=False) -20 + 90)
    phis = phis - np.radians(360/13./2.)
    for phi in phis:
        xx1, yy1 = rs2*np.cos(phi), rs2*np.sin(phi)
        xx2, yy2 = r*np.cos(phi), r*np.sin(phi)
        line = patches.FancyArrow(xx1, yy1, (xx2-xx1), (yy2-yy1), head_width=0, color='navy', lw=0.5)
        ax1.add_patch(line)

    # show North/East
    de, dn = calc_nedirection(pa)
    dl = 20.0
    posne = np.array([200, -200])
    arr_e = patches.FancyArrow(
        posne[0], posne[1], dl * de[0], dl * de[1], color="dimgrey", width=3.0
    )
    arr_n = patches.FancyArrow(
        posne[0], posne[1], dl * dn[0], dl * dn[1], color="dimgrey", width=3.0
    )

    ax1.add_patch(arr_e)
    ax1.add_patch(arr_n)
    ax1.text(
        posne[0] + dl * 2.5 * de[0],
        posne[1] + dl * 2.5 * de[1],
        f"E",
        ha="center",
        va="center",
        fontsize=10,
        color="dimgrey",
    )
    ax1.text(
        posne[0] + dl * 2.5 * dn[0],
        posne[1] + dl * 2.5 * dn[1],
        f"N",
        ha="center",
        va="center",
        fontsize=10,
        color="dimgrey",
    )

    proposals = np.unique(df_fib.proposalId.values)
    # c_sci = np.full(df_fib.shape[0], '')
    c_sci = []
    for i in range(len(proposals)):
        c_sci.append(next(c_list))
        # cc = np.full(df_fib.shape[0], next(c_list))
        # c_sci = np.vstack((c_sci, cc))
    # c_sci = c_sci[1:]
    # print(c_sci, proposals, end='')

    bins = 10
    mmin = 13
    mmax = 25
    bins = int((mmax - mmin) * 2)
    mag_per_prog = np.zeros(df_fib.shape[0])
    label_per_prog = []
    ax2.hist(
        df_fib[df_fib.targetType == 3]["psfMag"],
        bins=bins,
        range=(mmin, mmax),
        color=c["fstar"],
        lw=0,
        alpha=0.4,
        label="Std star",
    )
    for i, p in enumerate(proposals):
        mags = df_fib[(df_fib.targetType == 1) & (df_fib.proposalId == p)][
            "psfMag"
        ].values
        n_mags = len(mags)
        for j in range((df_fib.shape[0] - n_mags)):
            mags = np.append(mags, np.nan)
        # print(mag_per_prog.shape, mags.shape)
        mag_per_prog = np.vstack((mag_per_prog, mags))
        label_per_prog.append(f"{p} ({n_mags})")

    mag_per_prog = mag_per_prog[1:]
    # print(mag_per_prog.shape, mag_per_prog)
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
    # ax2.hist(df_fib[df_fib['targetType']==1]['psfMag'], bins=bins, range=(mmin, mmax),
    #         color=c['sci'], lw=0, alpha=0.4, label='Target')
    ax2.legend()
    ax2.set_xlabel("mag", fontsize=12)
    ax2.set_ylabel("N (target or STD)", fontsize=12)

    ax3.hist(
        df_ag["agMag"], bins=bins, range=(mmin, mmax), color=c["ag"], lw=0, alpha=0.4
    )
    ax3.set_ylabel("N (AG)", fontsize=12)
    ax3.set_xlabel("mag", fontsize=12)

    fig.suptitle(title, fontsize=12)

    if save == True:
        plt.savefig(fname + ".pdf")
    if show == True:
        plt.show()

    return fig.tight_layout()


warning = "hotpink"  # colour for warning


def colour_background_warning_sky_tot(val):
    colour = warning if val < 400 else ""

    return f"background-color: {colour}"


def colour_background_warning_std_tot(val):
    colour = warning if val < 40 else ""

    return f"background-color: {colour}"


def colour_background_warning_sky_min(val):
    colour = warning if val < 12 else ""

    return f"background-color: {colour}"


def colour_background_warning_std_min(val):
    colour = warning if val < 3 else ""

    return f"background-color: {colour}"


def colour_background_warning_ag_tot(val):
    colour = warning if val < 10 else ""

    return f"background-color: {colour}"


def colour_background_warning_ag_min(val):
    colour = warning if val < 2 else ""

    return f"background-color: {colour}"


def colour_background_warning_inr(val):
    colour = warning if (val < -174) or (val > 174) else ""

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
        ],
    )
    return df


def check_design(designId, df_fib, df_ag):
    df_ch = pd.DataFrame(
        data=np.append(check_fibers(df_fib), check_ags(df_ag)).reshape(1, 17),
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


def check_ags(df_ag):
    """
    df_ag: pandas dataframe with
           'agx', 'agy', 'agMag', 'camId', 'ag_pfi_x', 'ag_pfi_y'
    """

    cam = df_ag["camId"].values
    a = []
    for i in range(0, 6):
        a.append(np.sum(cam == i))

    a.append(np.sum(a))
    return np.array(a)


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
    phi =  np.radians(np.linspace(0, 360, 13, endpoint=False) - 20 + 90)
    points.append(175 * np.stack([np.cos(phi), np.sin(phi)], axis=-1))

    xy = np.concatenate(points, axis=0)

    # Cobra centres
    uv = np.stack([df_fib.pfi_x, df_fib.pfi_y], axis=-1)

    # Find the closest point to each cobra centre
    d = distance_matrix(xy, uv)

    tag = np.argmin(d, axis=0)
    #tag.shape
    
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
