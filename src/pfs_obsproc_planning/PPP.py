#!/usr/bin/env python3
# PPP.py : PPP full version

import multiprocessing
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from functools import partial
from itertools import chain
from loguru import logger
from matplotlib.path import Path
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KernelDensity

warnings.filterwarnings("ignore")

# below for netflow
import ets_fiber_assigner.netflow as nf
from ics.cobraOps.Bench import Bench
from ics.cobraOps.cobraConstants import NULL_TARGET_ID, NULL_TARGET_POSITION
from ics.cobraOps.CollisionSimulator import CollisionSimulator
from ics.cobraOps.TargetGroup import TargetGroup

# netflow configuration (FIXME; should be load from config file)
cobra_location_group = None
min_sky_targets_per_location = None
location_group_penalty = None
cobra_instrument_region = None
min_sky_targets_per_instrument_region = None
instrument_region_penalty = None
black_dot_penalty_cost = None


def DBinfo(para_db):
    # the link of DB to connect
    dialect, user, pwd, host, port, dbname = para_db
    return "{0}://{1}:{2}@{3}:{4}/{5}".format(dialect, user, pwd, host, port, dbname)


def count_N_overlap(_tb_tgt_psl, _tb_tgt):
    # calculate local count of targets (bin_width is 1 deg in ra&dec)
    # lower limit of dec is -40
    count_bin = [[0 for i in np.arange(0, 361, 1)] for j in np.arange(-40, 91, 1)]

    n_tgt = len(_tb_tgt)
    for ii in range(n_tgt):
        m = int(_tb_tgt["ra"][ii])
        n = int(_tb_tgt["dec"][ii] + 40)  # dec>-40
        count_bin[n][m] += 1
    den_local = [
        count_bin[int(_tb_tgt_psl["dec"][ii] + 40)][int(_tb_tgt_psl["ra"][ii])]
        for ii in range(len(_tb_tgt_psl))
    ]

    _tb_tgt_psl["local_count"] = den_local

    return _tb_tgt_psl

def removeObjIdDuplication(df):
    num1 = len(df)
    df = df.drop_duplicates(subset=['proposal_id', 'obj_id', 'input_catalog_id', 'resolution'], inplace=False, ignore_index=True)
    num2 = len(df)
    logger.info(f'Duplication removed: {num1} --> {num2}')
    return df    

def readTarget(mode, para):
    """Read target list including:
       'ob_code' 'ra' 'dec' 'priority' 'exptime' 'exptime_tac' 'resolution' 'proposal_id' 'rank' 'grade' 'allocated_time'

    Parameters
    ==========
    para : dict
        mode:
            'local' -- read target list from local machine
            'DB' -- read target list from Database

        localPath:
            (if mode == local) the path of the target list
        DBPath_tgt(dialect,user,pwd,host,port,dbname):
            (if mode == DB) used to create the link to connect DB
        sql_query:
            (if mode == DB) used to query necessary information of targets from DB

    Returns
    =======
    target sample (all), target sample (low-resolution mode), target sample (medium-resolution mode)
    """
    time_start = time.time()
    logger.info(f"[S1] Read targets started (PPP)")

    if mode == "local":
        tb_tgt = Table.read(para["localPath_tgt"])

    elif mode == "DB":
        import pandas as pd
        import psycopg2
        import sqlalchemy as sa

        DBads = DBinfo(para["DBPath_tgt"])
        tgtDB = sa.create_engine(DBads)

        sql = para["sql_query"]

        conn = tgtDB.connect()
        query = conn.execute(sa.sql.text(sql))

        df_tgt = pd.DataFrame(
            query.fetchall(),
            columns=[
                "ob_code",
                "obj_id",
                "input_catalog_id",
                "ra",
                "dec",
                "epoch",
                "priority",
                "pmra",
                "pmdec",
                "parallax",
                "effective_exptime",
                "is_medium_resolution",
                "proposal_id",
                "rank",
                "grade",
                "allocated_time",
                "allocated_time_lr",
                "allocated_time_mr",
                "filter_g",
                "filter_r",
                "filter_i",
                "filter_z",
                "filter_y",
                "psf_flux_g",
                "psf_flux_r",
                "psf_flux_i",
                "psf_flux_z",
                "psf_flux_y",
                "psf_flux_error_g",
                "psf_flux_error_r",
                "psf_flux_error_i",
                "psf_flux_error_z",
                "psf_flux_error_y",
            ],
        )
        # convert column names
        df_tgt = df_tgt.rename(columns={"epoch": "equinox"})
        df_tgt = df_tgt.rename(columns={"effective_exptime": "exptime"})
        df_tgt = df_tgt.rename(columns={"is_medium_resolution": "resolution"})

        # convert Boolean to String
        df_tgt["resolution"] = ["M" if v == True else "L" for v in df_tgt["resolution"]]
        df_tgt["allocated_time"] = [
            df_tgt["allocated_time_lr"][ii]
            if df_tgt["resolution"][ii] == "L"
            else df_tgt["allocated_time_mr"][ii]
            for ii in range(len(df_tgt))
        ]
        df_tgt = df_tgt.drop(columns=["allocated_time_lr", "allocated_time_mr"])

        df_tgt = removeObjIdDuplication(df_tgt)

        df_tgt["psf_flux_g"][np.isnan(df_tgt["psf_flux_g"])]=9e-5
        df_tgt["psf_flux_r"][np.isnan(df_tgt["psf_flux_r"])]=9e-5
        df_tgt["psf_flux_i"][np.isnan(df_tgt["psf_flux_i"])]=9e-5
        df_tgt["psf_flux_z"][np.isnan(df_tgt["psf_flux_z"])]=9e-5
        df_tgt["psf_flux_y"][np.isnan(df_tgt["psf_flux_y"])]=9e-5

        df_tgt["psf_flux_error_g"][np.isnan(df_tgt["psf_flux_error_g"])]=9e-5
        df_tgt["psf_flux_error_r"][np.isnan(df_tgt["psf_flux_error_r"])]=9e-5
        df_tgt["psf_flux_error_i"][np.isnan(df_tgt["psf_flux_error_i"])]=9e-5
        df_tgt["psf_flux_error_z"][np.isnan(df_tgt["psf_flux_error_z"])]=9e-5
        df_tgt["psf_flux_error_y"][np.isnan(df_tgt["psf_flux_error_y"])]=9e-5

        tb_tgt = Table.from_pandas(df_tgt)

        conn.close()

    tb_tgt["ra"] = tb_tgt["ra"].astype(float)
    tb_tgt["dec"] = tb_tgt["dec"].astype(float)
    tb_tgt["ob_code"] = tb_tgt["ob_code"].astype(str)
    tb_tgt["identify_code"] = [tt["proposal_id"] + "_" + tt["ob_code"] for tt in tb_tgt]
    tb_tgt["exptime_assign"] = 0

    # exptime needs to be multiples of 900s so netflow can be successfully executed
    tb_tgt["exptime_PPP"] = np.ceil(tb_tgt["exptime"] / 900) * 900

    # separete the sample by 'resolution' (L/M)
    tb_tgt_l = tb_tgt[tb_tgt["resolution"] == "L"]
    tb_tgt_m = tb_tgt[tb_tgt["resolution"] == "M"]

    # select targets based on the allocated FH (for determining PPC)
    psl_id = sorted(set(tb_tgt["proposal_id"]))
    _tgt_select_l = []
    _tgt_select_m = []

    for psl_id_ in psl_id:
        tb_tgt_tem_l = tb_tgt[
            (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "L")
        ]
        tb_tgt_tem_m = tb_tgt[
            (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "M")
        ]

        size_step = 50
        size_ = 50

        if len(tb_tgt_tem_l) > 0:
            FH_select = 0
            FH_tac = tb_tgt_tem_l["allocated_time"][0] * 3600.0

            while FH_select < FH_tac:
                tb_tgt_tem_l = count_N_overlap(tb_tgt_tem_l, tb_tgt_l)
                pri_ = (
                    (10 - tb_tgt_tem_l["priority"]) 
                    + 2
                    * (tb_tgt_tem_l["local_count"] / max(tb_tgt_tem_l["local_count"]))
                    + 2 * (1 - tb_tgt_tem_l["exptime"] / max(tb_tgt_tem_l["exptime"]))
                )
                psl_pri_l = pri_ / sum(pri_)
                if len(tb_tgt_tem_l) < size_step:
                    size_ = len(tb_tgt_tem_l)
                index_t = np.random.choice(
                    np.arange(0, len(tb_tgt_tem_l), 1),
                    size=size_,
                    replace=False,
                    p=psl_pri_l,
                )

                _tb_tem = Table.copy(tb_tgt_tem_l)
                [_tgt_select_l.append(_tb_tem_) for _tb_tem_ in _tb_tem[index_t]]
                FH_select += sum(tb_tgt_tem_l[index_t]["exptime"])
                tb_tgt_tem_l.remove_rows(index_t)

                if len(tb_tgt_tem_l) == 0:
                    break

        if len(tb_tgt_tem_m) > 0:
            FH_select = 0
            FH_tac = tb_tgt_tem_m["allocated_time"][0] * 3600.0

            while FH_select < FH_tac:
                tb_tgt_tem_m = count_N_overlap(tb_tgt_tem_m, tb_tgt_m)
                pri_ = (
                    (10 - tb_tgt_tem_m["priority"])
                    + 2
                    * (tb_tgt_tem_m["local_count"] / max(tb_tgt_tem_m["local_count"]))
                    + 2 * (1 - tb_tgt_tem_m["exptime"] / max(tb_tgt_tem_m["exptime"]))
                )
                psl_pri_m = pri_ / sum(pri_)
                if len(tb_tgt_tem_m) < size_step:
                    size_ = len(tb_tgt_tem_m)
                index_t = np.random.choice(
                    np.arange(0, len(tb_tgt_tem_m), 1),
                    size=size_,
                    replace=False,
                    p=psl_pri_m,
                )

                _tb_tem = Table.copy(tb_tgt_tem_m)
                [_tgt_select_m.append(_tb_tem_) for _tb_tem_ in _tb_tem[index_t]]
                FH_select += sum(tb_tgt_tem_m[index_t]["exptime"])
                tb_tgt_tem_m.remove_rows(index_t)

                if len(tb_tgt_tem_m) == 0:
                    break

    if len(_tgt_select_l) > 0:
        tgt_select_l = vstack(_tgt_select_l)
    else:
        tgt_select_l = Table()

    if len(_tgt_select_m) > 0:
        tgt_select_m = vstack(_tgt_select_m)
    else:
        tgt_select_m = Table()
    # """

    logger.info(
        f"[S1] Read targets done (takes {round(time.time()-time_start,3):.2f} sec)."
    )
    logger.info(f"[S1] There are {len(set(tb_tgt['proposal_id'])):.0f} proposals.")
    logger.info(
        f"[S1] n_tgt_low = {len(tb_tgt_l):.0f} ({len(tgt_select_l):.0f}), n_tgt_medium = {len(tb_tgt_m):.0f} ({len(tgt_select_m):.0f})"
    )

    return tb_tgt, tgt_select_l, tgt_select_m, tb_tgt_l, tb_tgt_m


def count_N(_tb_tgt):
    # calculate local count of targets (bin_width is 1 deg in ra&dec)
    # lower limit of dec is -40
    if len(_tb_tgt) == 0:
        return _tb_tgt

    count_bin = [[0 for i in np.arange(0, 361, 1)] for j in np.arange(-40, 91, 1)]

    n_tgt = len(_tb_tgt)
    for ii in range(n_tgt):
        m = int(_tb_tgt["ra"][ii])
        n = int(_tb_tgt["dec"][ii] + 40)  # dec>-40
        count_bin[n][m] += 1
    den_local = [
        count_bin[int(_tb_tgt["dec"][ii] + 40)][int(_tb_tgt["ra"][ii])]
        for ii in range(n_tgt)
    ]

    _tb_tgt["local_count"] = den_local

    return _tb_tgt


def sciRank_pri(_tb_tgt):
    # calculate rank+priority of targets (higher value means more important)
    # re-order the rank (starting from 0)
    if len(_tb_tgt) == 0:
        return _tb_tgt

    SciRank = [0.0] + sorted(list(set(_tb_tgt["rank"])))

    # give each user priority a rank in the interval of the two ranks
    # (0-9, with 0=rank_i, 9=0.5*(rank_[i-1]+rank_i))
    SciRank_usrPri = [
        np.arange(
            0.55 * SciRank[i1] + 0.45 * SciRank[i1 - 1],
            1.05 * SciRank[i1] - 0.05 * SciRank[i1 - 1],
            0.05 * (SciRank[i1] - SciRank[i1 - 1]),
        )
        for i1 in range(1, len(SciRank))
    ]

    SciUsr_Ranktot = np.array(
        [
            SciRank_usrPri[i2 - 1][9 - j2]
            for s_tem in _tb_tgt
            for i2 in range(1, len(SciRank))
            for j2 in range(0, 10, 1)
            if s_tem["rank"] == SciRank[i2] and s_tem["priority"] == j2
        ]
    )

    _tb_tgt["rank_fin"] = SciUsr_Ranktot

    return _tb_tgt


def weight(_tb_tgt, para_sci, para_exp, para_n):
    # calculate weights of targets (higher weights mean more important)
    if len(_tb_tgt) == 0:
        return _tb_tgt

    weight_t = (
        pow(para_sci, _tb_tgt["rank_fin"])
        * pow(_tb_tgt["exptime_PPP"] / 900.0, para_exp)
        * pow(_tb_tgt["local_count"], para_n)
    )

    _tb_tgt["weight"] = weight_t

    return _tb_tgt


def target_DBSCAN(_tb_tgt, sep=1.38):
    # separate targets into different groups
    # haversine uses (dec,ra) in radian;
    tgt_cluster = DBSCAN(eps=np.radians(sep), min_samples=1, metric="haversine").fit(
        np.radians([_tb_tgt["dec"], _tb_tgt["ra"]]).T
    )

    labels = tgt_cluster.labels_
    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    tgt_group = []
    tgt_pri_ord = []

    for ii in range(n_clusters):
        tgt_t_pri_tot = sum(_tb_tgt[labels == ii]["weight"])
        tgt_pri_ord.append([ii, tgt_t_pri_tot])

    tgt_pri_ord.sort(key=lambda x: x[1], reverse=True)

    for ii in np.array(tgt_pri_ord)[:, 0]:
        tgt_t = _tb_tgt[labels == ii]
        tgt_group.append(tgt_t)

    return tgt_group


def PFS_FoV(ppc_ra, ppc_dec, PA, _tb_tgt):
    # pick up targets in the ppcs
    tgt_lst = np.vstack((_tb_tgt["ra"], _tb_tgt["dec"])).T
    ppc_lst = SkyCoord(ppc_ra * u.deg, ppc_dec * u.deg)

    # PA=0 along y-axis, PA=90 along x-axis, PA=180 along -y-axis...
    hexagon = ppc_lst.directional_offset_by(
        [30 + PA, 90 + PA, 150 + PA, 210 + PA, 270 + PA, 330 + PA, 30 + PA] * u.deg,
        1.38 / 2.0 * u.deg,
    )
    ra_h = hexagon.ra.deg
    dec_h = hexagon.dec.deg

    # for pointings around RA~0 or 360, parts of it will move to the opposite side (e.g., [[1,0],[-1,0]] -->[[1,0],[359,0]])
    # correct for it
    ra_h_in = np.where(np.fabs(ra_h - ppc_ra) > 180)
    if len(ra_h_in[0]) > 0:
        if ra_h[ra_h_in[0][0]] > 180:
            ra_h[ra_h_in[0]] -= 360
        elif ra_h[ra_h_in[0][0]] < 180:
            ra_h[ra_h_in[0]] += 360

    polygon = Path([(ra_h[t], dec_h[t]) for t in range(len(ra_h))])
    index_ = np.where(polygon.contains_points(tgt_lst))[0]

    return index_


def KDE_xy(_tb_tgt, X, Y):
    # calculate a single KDE
    tgt_values = np.vstack((np.deg2rad(_tb_tgt["dec"]), np.deg2rad(_tb_tgt["ra"])))
    kde = KernelDensity(
        bandwidth=np.deg2rad(1.38 / 2.0),
        kernel="linear",
        algorithm="ball_tree",
        metric="haversine",
    )
    kde.fit(tgt_values.T, sample_weight=_tb_tgt["weight"])

    X1 = np.deg2rad(X)
    Y1 = np.deg2rad(Y)
    positions = np.vstack([Y1.ravel(), X1.ravel()])
    Z = np.reshape(np.exp(kde.score_samples(positions.T)), Y.shape)

    return Z


def KDE(_tb_tgt, multiProcesing):
    # define binning and calculate KDE
    if len(_tb_tgt) == 1:
        # if only one target, set it as the peak
        return (
            _tb_tgt["ra"].data[0],
            _tb_tgt["dec"].data[0],
            np.nan,
            _tb_tgt["ra"].data[0],
            _tb_tgt["dec"].data[0],
        )
    else:
        # determine the binning for the KDE cal.
        # set a bin width of 0.5 deg in ra&dec if the sample spans over a wide area (>50 degree)
        # give some blank spaces in binning, otherwide KDE will be wrongly calculated
        ra_low = min(min(_tb_tgt["ra"]) * 0.9, min(_tb_tgt["ra"]) - 1)
        ra_up = max(max(_tb_tgt["ra"]) * 1.1, max(_tb_tgt["ra"]) + 1)
        dec_up = max(max(_tb_tgt["dec"]) * 1.1, max(_tb_tgt["dec"]) + 1)
        dec_low = min(min(_tb_tgt["dec"]) * 0.9, min(_tb_tgt["dec"]) - 1)

        if (max(_tb_tgt["ra"]) - min(_tb_tgt["ra"])) / 100 < 0.5 and (
            max(_tb_tgt["dec"]) - min(_tb_tgt["dec"])
        ) / 100 < 0.5:
            X_, Y_ = np.mgrid[ra_low:ra_up:101j, dec_low:dec_up:101j]
        elif (max(_tb_tgt["dec"]) - min(_tb_tgt["dec"])) / 100 < 0.5:
            X_, Y_ = np.mgrid[0:360:721j, dec_low:dec_up:101j]
        elif (max(_tb_tgt["ra"]) - min(_tb_tgt["ra"])) / 100 < 0.5:
            X_, Y_ = np.mgrid[ra_low:ra_up:101j, -40:90:261j]
        else:
            X_, Y_ = np.mgrid[0:360:721j, -40:90:261j]
        positions1 = np.vstack([Y_.ravel(), X_.ravel()])

        if multiProcesing:
            threads_count = round(multiprocessing.cpu_count() / 2)
            thread_n = min(
                threads_count, round(len(_tb_tgt) * 0.5)
            )  # threads_count=10 in this machine

            with multiprocessing.Pool(thread_n) as p:
                dMap_ = p.map(
                    partial(KDE_xy, X=X_, Y=Y_), np.array_split(_tb_tgt, thread_n)
                )

            Z = sum(dMap_)

        else:
            Z = KDE_xy(_tb_tgt, X_, Y_)

        # calculate significance level of KDE
        obj_dis_sig_ = (Z - np.mean(Z)) / np.std(Z)
        peak_pos = np.where(obj_dis_sig_ == obj_dis_sig_.max())

        peak_y = positions1[0, peak_pos[1][round(len(peak_pos[1]) * 0.5)]]
        peak_x = sorted(set(positions1[1, :]))[
            peak_pos[0][round(len(peak_pos[0]) * 0.5)]
        ]

        return X_, Y_, obj_dis_sig_, peak_x, peak_y


def PPP_centers(_tb_tgt, nPPC, weight_para, randomseed=0, mutiPro=True):
    # determine pointing centers
    time_start = time.time()
    logger.info(f"[S2] Determine pointing centers started")

    para_sci, para_exp, para_n = weight_para

    ppc_lst = []

    if "PPC_origin" in _tb_tgt.meta.keys():
        logger.warning(
            f"[S2] PPCs from uploader adopted (takes {round(time.time()-time_start,3):.2f} sec)."
        )
        ppc_lst = _tb_tgt.meta["PPC"]
        return ppc_lst

    if len(_tb_tgt) == 0:
        logger.warning(f"[S2] no targets")
        return np.array(ppc_lst)

    _tb_tgt = sciRank_pri(_tb_tgt)
    _tb_tgt = count_N(_tb_tgt)
    _tb_tgt = weight(_tb_tgt, para_sci, para_exp, para_n)

    ppc_totPri = []

    for _tb_tgt_t in target_DBSCAN(_tb_tgt, 1.38):
        _tb_tgt_t_ = _tb_tgt_t[_tb_tgt_t["exptime_PPP"] > 0]  # targets not finished

        ppc_totPri_sub = []
        iter_n = 0
        while any(_tb_tgt_t_["exptime_PPP"] > 0):
            # peak_xy from KDE peak with weights -------------------------------
            X_, Y_, obj_dis_sig_, peak_x, peak_y = KDE(_tb_tgt_t_, mutiPro)

            # select targets falling in the PPC-------------------------------
            index_ = PFS_FoV(
                peak_x, peak_y, 0, _tb_tgt_t_
            )  # all PA set to be 0 for simplicity

            if len(index_) > 0:
                ppc_lst.append(
                    [len(ppc_lst), peak_x, peak_y, 0]
                )  # ppc_id,ppc_ra,ppc_dec,ppc_PA=0

            else:
                # add a small random shift so that it will not repeat over a blank position
                while len(index_) == 0:
                    peak_x_t = peak_x + np.random.uniform(-0.15, 0.15, 1)[0]
                    peak_y_t = peak_y + np.random.uniform(-0.15, 0.15, 1)[0]
                    index_ = PFS_FoV(peak_x_t, peak_y_t, 0, _tb_tgt_t_)

                ppc_lst.append(
                    [len(ppc_lst), peak_x_t, peak_y_t, 0]
                )  # ppc_id,ppc_ra,ppc_dec,ppc_PA=0

            # run netflow to assign fibers for targets falling in the PPC-------------------------------
            lst_tgtID_assign = netflowRun4PPC(
                _tb_tgt_t_[list(index_)], ppc_lst[-1][1], ppc_lst[-1][2]
            )

            index_assign = np.in1d(_tb_tgt_t_["identify_code"], lst_tgtID_assign)
            _tb_tgt_t_["exptime_PPP"][
                index_assign
            ] -= 900  # targets in the PPC observed with 900 sec

            # add a small random so that PPCs determined would not have totally same weights
            weight_random = np.random.uniform(-0.05, 0.05, 1)[0]
            ppc_totPri.append(sum(_tb_tgt_t_["weight"][index_assign]) + weight_random)
            ppc_totPri_sub.append(
                sum(_tb_tgt_t_["weight"][index_assign]) + weight_random
            )

            if len(lst_tgtID_assign) == 0:
                # quit if no targets assigned
                break

            if iter_n>25 and ppc_totPri_sub[-1] < ppc_totPri_sub[0] * 0.15:
                # quit if ppc contains too limited targets
                break

            # -------------------------------
            _tb_tgt_t_ = _tb_tgt_t_[
                _tb_tgt_t_["exptime_PPP"] > 0
            ]  # targets not finished
            _tb_tgt_t_ = count_N(_tb_tgt_t_)
            _tb_tgt_t_ = weight(_tb_tgt_t_, para_sci, para_exp, para_n)
            iter_n += 1

            print(
                f"PPC_{len(ppc_lst):03d}: {len(_tb_tgt_t)-len(_tb_tgt_t_):5d}/{len(_tb_tgt_t):10d} targets are finished (w={ppc_totPri[-1]:.2f})."
            )

    if len(ppc_lst) > nPPC:
        ppc_totPri_limit = sorted(ppc_totPri, reverse=True)[nPPC - 1]
        ppc_lst_fin = [
            ppc_lst[iii]
            for iii in range(len(ppc_lst))
            if ppc_totPri[iii] >= ppc_totPri_limit
        ]

    else:
        ppc_lst_fin = ppc_lst[:]

    logger.info(
        f"[S2] Determine pointing centers done ( nppc = {len(ppc_lst_fin):.0f}; takes {round(time.time()-time_start,3)} sec)"
    )

    return np.array(ppc_lst_fin)


def ppc_DBSCAN(_tb_tgt):
    # separate pointings into different group (skip due to FH upper limit -24-02-07; NEED TO FIX)
    ppc_xy = _tb_tgt.meta["PPC"]
    ppc_group = []
    """
    # haversine uses (dec,ra) in radian;
    ppc_cluster = DBSCAN(eps=np.radians(1.38), min_samples=1, metric="haversine").fit(
        np.fliplr(np.radians(ppc_xy[:, [1, 2]]))
    )

    labels = ppc_cluster.labels_
    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    logger.info(f"[S3] There are {len(ppc_xy):5d} pointings, they are grouped into {n_clusters:5d} clusters.")

    for ii in range(n_clusters):
        ppc_t = ppc_xy[labels == ii]
        ppc_group.append(ppc_t)

    ppc_group.sort(key=lambda x: len(x), reverse=True)
    #"""

    ppc_group.append(ppc_xy)

    return ppc_group


def sam2netflow(_tb_tgt, for_ppc=False):
    # put targets to the format which can be read by netflow
    tgt_lst_netflow = []
    _tgt_lst_psl_id = []

    int_ = 0
    for tt in _tb_tgt:
        if for_ppc:
            # set exptime = 900s if running netflow to determine PPC
            tgt_id_, tgt_ra_, tgt_dec_, tgt_exptime_, tgt_proposal_id_ = (
                tt["identify_code"],
                tt["ra"],
                tt["dec"],
                900,
                tt["proposal_id"],
            )
        else:
            tgt_id_, tgt_ra_, tgt_dec_, tgt_exptime_, tgt_proposal_id_ = (
                tt["identify_code"],
                tt["ra"],
                tt["dec"],
                tt["exptime_PPP"],
                tt["proposal_id"],
            )
        tgt_lst_netflow.append(
            nf.ScienceTarget(
                tgt_id_,
                tgt_ra_,
                tgt_dec_,
                tgt_exptime_,
                int_,
                "sci_" + tgt_proposal_id_,
            )
        )
        _tgt_lst_psl_id.append("sci_" + tgt_proposal_id_ + "_P" + str(int(int_)))
        int_ += 1

    # set FH limit bundle
    tgt_psl_FH_tac_ = {}

    if for_ppc == False:
        psl_id = sorted(set(_tb_tgt["proposal_id"]))

        for psl_id_ in psl_id:
            tt_ = tuple([tt for tt in _tgt_lst_psl_id if psl_id_ in tt])
            tgt_psl_FH_tac_[tt_] = _tb_tgt[_tb_tgt["proposal_id"] == psl_id_][
                "allocated_time"
            ][0]

    return tgt_lst_netflow, tgt_psl_FH_tac_


def NetflowPreparation(_tb_tgt):
    # assign cost to each target
    classdict = {}

    int_ = 0
    for tt in _tb_tgt:
        classdict["sci_" + tt["proposal_id"] + "_P" + str(int_)] = {
            "nonObservationCost": tt["weight"],
            "partialObservationCost": tt["weight"] * 1.5,
            "calib": False,
        }
        int_ += 1

    return classdict


def cobraMoveCost(dist):
    # optional: penalize assignments where the cobra has to move far out
    return 0.1 * dist


def netflowRun_single(
    ppc_lst,
    _tb_tgt,
    TraCollision=False,
    numReservedFibers=0,
    fiberNonAllocationCost=0.0,
    otime="2024-05-20T08:00:00Z",
    for_ppc=False,
):
    # run netflow (without iteration)
    Telra = ppc_lst[:, 1]
    Teldec = ppc_lst[:, 2]
    Telpa = ppc_lst[:, 3]

    tgt_lst_netflow, tgt_psl_FH_tac = sam2netflow(_tb_tgt, for_ppc)
    classdict = NetflowPreparation(_tb_tgt)

    telescopes = []

    nvisit = len(Telra)
    for ii in range(nvisit):
        telescopes.append(nf.Telescope(Telra[ii], Teldec[ii], Telpa[ii], otime))
    tpos = [tele.get_fp_positions(tgt_lst_netflow) for tele in telescopes]

    # optional: slightly increase the cost for later observations,
    # to observe as early as possible
    vis_cost = [0 for i in range(nvisit)]

    gurobiOptions = dict(
        seed=0,
        presolve=1,
        method=4,
        degenmoves=0,
        heuristics=0.8,
        mipfocus=0,
        mipgap=5.0e-2,
        LogToConsole=0,
    )

    forbiddenPairs = [[] for i in range(nvisit)]
    alreadyObserved = {}

    if TraCollision:
        done = False
        while not done:
            # compute observation strategy
            prob = nf.buildProblem(
                bench,
                tgt_lst_netflow,
                tpos,
                classdict,
                900,
                vis_cost,
                cobraMoveCost=cobraMoveCost,
                collision_distance=2.0,
                elbow_collisions=True,
                gurobi=True,
                gurobiOptions=gurobiOptions,
                alreadyObserved=alreadyObserved,
                forbiddenPairs=forbiddenPairs,
                cobraLocationGroup=cobra_location_group,
                minSkyTargetsPerLocation=min_sky_targets_per_location,
                locationGroupPenalty=location_group_penalty,
                cobraInstrumentRegion=cobra_instrument_region,
                minSkyTargetsPerInstrumentRegion=min_sky_targets_per_instrument_region,
                instrumentRegionPenalty=instrument_region_penalty,
                blackDotPenalty=black_dot_penalty_cost,
                numReservedFibers=numReservedFibers,
                fiberNonAllocationCost=fiberNonAllocationCost,
                obsprog_time_budget=tgt_psl_FH_tac,
            )

            prob.solve()

            res = [{} for _ in range(min(nvisit, len(Telra)))]
            for k1, v1 in prob._vardict.items():
                if k1.startswith("Tv_Cv_"):
                    visited = prob.value(v1) > 0
                    if visited:
                        _, _, tidx, cidx, ivis = k1.split("_")
                        res[int(ivis)][int(tidx)] = int(cidx)

            ncoll = 0
            for ivis, (vis, tp) in enumerate(zip(res, tpos)):
                selectedTargets = np.full(
                    len(bench.cobras.centers), NULL_TARGET_POSITION
                )
                ids = np.full(len(bench.cobras.centers), NULL_TARGET_ID)
                for tidx, cidx in vis.items():
                    selectedTargets[cidx] = tp[tidx]
                    ids[cidx] = ""
                for i in range(selectedTargets.size):
                    if selectedTargets[i] != NULL_TARGET_POSITION:
                        dist = np.abs(selectedTargets[i] - bench.cobras.centers[i])

                simulator = CollisionSimulator(bench, TargetGroup(selectedTargets, ids))
                simulator.run()
                if np.any(simulator.endPointCollisions):
                    logger.error(
                        "ERROR: detected end point collision, which should be impossible"
                    )
                coll_tidx = []
                for tidx, cidx in vis.items():
                    if simulator.collisions[cidx]:
                        coll_tidx.append(tidx)
                ncoll += len(coll_tidx)
                for i1 in range(0, len(coll_tidx)):
                    for i2 in range(i1 + 1, len(coll_tidx)):
                        if np.abs(tp[coll_tidx[i1]] - tp[coll_tidx[i2]]) < 10:
                            forbiddenPairs[ivis].append((coll_tidx[i1], coll_tidx[i2]))

        done = ncoll == 0

    else:
        # compute observation strategy
        prob = nf.buildProblem(
            bench,
            tgt_lst_netflow,
            tpos,
            classdict,
            900,
            vis_cost,
            cobraMoveCost=cobraMoveCost,
            collision_distance=2.0,
            elbow_collisions=True,
            gurobi=True,
            gurobiOptions=gurobiOptions,
            alreadyObserved=alreadyObserved,
            forbiddenPairs=forbiddenPairs,
            cobraLocationGroup=cobra_location_group,
            minSkyTargetsPerLocation=min_sky_targets_per_location,
            locationGroupPenalty=location_group_penalty,
            cobraInstrumentRegion=cobra_instrument_region,
            minSkyTargetsPerInstrumentRegion=min_sky_targets_per_instrument_region,
            instrumentRegionPenalty=instrument_region_penalty,
            blackDotPenalty=black_dot_penalty_cost,
            numReservedFibers=numReservedFibers,
            fiberNonAllocationCost=fiberNonAllocationCost,
            obsprog_time_budget=tgt_psl_FH_tac,
        )

        prob.solve()

        res = [{} for _ in range(min(nvisit, len(Telra)))]
        for k1, v1 in prob._vardict.items():
            if k1.startswith("Tv_Cv_"):
                visited = prob.value(v1) > 0
                if visited:
                    _, _, tidx, cidx, ivis = k1.split("_")
                    res[int(ivis)][int(tidx)] = int(cidx)

    return res, telescopes, tgt_lst_netflow


def netflowRun_nofibAssign(
    ppc_lst,
    _tb_tgt,
    for_ppc=False,
    randomseed=0,
    TraCollision=False,
    numReservedFibers=0,
    fiberNonAllocationCost=0.0,
):
    # run netflow (with iteration)
    #    if no fiber assignment in some PPCs, shift these PPCs with 0.15 deg
    # (skip due to FH upper limit -24-02-07; NEED TO FIX)

    otime_ = "2024-05-20T08:00:00Z"

    res, telescope, tgt_lst_netflow = netflowRun_single(
        ppc_lst,
        _tb_tgt,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
        otime_,
        for_ppc,
    )
    return res, telescope, tgt_lst_netflow

    """
    if sum(np.array([len(tt) for tt in res]) == 0) == 0:
        # All PPCs have fiber assignment
        return res, telescope, tgt_lst_netflow

    else:
        # if there are PPCs with no fiber assignment
        index = np.where(np.array([len(tt) for tt in res]) == 0)[0]

        ppc_lst = np.array(ppc_lst)
        ppc_lst_t = np.copy(ppc_lst)
        
        iter_1 = 0

        while len(index) > 0 and iter_1 < 8:
            # shift PPCs with 0.2 deg, but only run 8 iterations to save computational time
            # typically one iteration is enough

            logger.info(f"[S3] Re-assign fibers to PPCs without fiber assignment (iter {iter_1+1:.0f}/8)")

            shift_ra = np.random.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], 1)[0]
            shift_dec = np.random.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], 1)[0]

            ppc_lst_t[index,1] = ppc_lst[index,1] + shift_ra
            ppc_lst_t[index,2] = ppc_lst[index,2] + shift_dec

            res, telescope, tgt_lst_netflow = netflowRun_single(
                ppc_lst_t, 
                _tb_tgt, 
                TraCollision, 
                numReservedFibers, 
                fiberNonAllocationCost, 
                otime_,
                for_ppc
            )

            index = np.where(np.array([len(tt) for tt in res]) == 0)[0]

            iter_1 += 1

            if iter_1 >= 4:
                otime_ = "2024-04-20T08:00:00Z"#"""


def netflowRun(
    _tb_tgt,
    randomseed=0,
    TraCollision=False,
    numReservedFibers=0,
    fiberNonAllocationCost=0.0,
):
    # run netflow (with iteration and DBSCAN)

    time_start = time.time()
    logger.info("[S3] Run netflow started")

    if ("PPC" not in _tb_tgt.meta.keys()) or (len(_tb_tgt.meta["PPC"]) == 0):
        logger.warning("[S3] No PPC has been determined")
        ppc_lst = []
        return ppc_lst

    if len(_tb_tgt) == 0:
        logger.warning("[S3] No targets")
        ppc_lst = []
        return ppc_lst

    ppc_g = ppc_DBSCAN(_tb_tgt)  # separate ppc into different groups

    ppc_lst = []

    for uu in range(len(ppc_g)):  # run netflow for each ppc group
        # only consider sample in the group
        _index = list(
            chain.from_iterable(
                [
                    list(
                        PFS_FoV(
                            ppc_g[uu][iii, 1],
                            ppc_g[uu][iii, 2],
                            ppc_g[uu][iii, 3],
                            _tb_tgt,
                        )
                    )
                    for iii in range(len(ppc_g[uu]))
                ]
            )
        )

        if len(_index) == 0:
            continue
        _tb_tgt_inuse = _tb_tgt[list(set(_index))]

        logger.info(
            f"[S3] Group {uu + 1:3d}: nppc = {len(ppc_g[uu]):5d}, n_tgt = {len(_tb_tgt_inuse):6d}"
        )

        res, telescope, tgt_lst_netflow = netflowRun_nofibAssign(
            ppc_g[uu],
            _tb_tgt_inuse,
            False,
            randomseed,
            TraCollision,
            numReservedFibers,
            fiberNonAllocationCost,
        )

        for i, (vis, tel) in enumerate(zip(res, telescope)):
            ppc_fib_eff = len(vis) / 2394.0 * 100

            logger.info(f"PPC {i:4d}: {ppc_fib_eff:.2f}% assigned Cobras")

            # assigned targets in each ppc
            tgt_assign_id_lst = []
            for tidx, cidx in vis.items():
                tgt_assign_id_lst.append(tgt_lst_netflow[tidx].ID)

            # calculate the total weights in each ppc (smaller value means more important)
            if len(vis) == 0:
                ppc_tot_weight = np.nan

            else:
                ppc_tot_weight = 1 / sum(
                    _tb_tgt[np.in1d(_tb_tgt["identify_code"], tgt_assign_id_lst)][
                        "weight"
                    ]
                )

            ppc_lst.append(
                [
                    "PPC_"
                    + _tb_tgt["resolution"][0]
                    + "_"
                    + str(int(time.time() * 1e7))[-8:],
                    "Group_" + str(uu + 1),
                    tel._ra,
                    tel._dec,
                    tel._posang,
                    ppc_tot_weight,
                    ppc_fib_eff,
                    tgt_assign_id_lst,
                    _tb_tgt["resolution"][0],
                ]
            )

    tb_ppc_netflow = Table(
        np.array(ppc_lst, dtype=object),
        names=[
            "ppc_code",
            "group_id",
            "ppc_ra",
            "ppc_dec",
            "ppc_pa",
            "ppc_priority",
            "ppc_fiber_usage_frac",
            "ppc_allocated_targets",
            "ppc_resolution",
        ],
        dtype=[
            np.str_,
            np.str_,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            object,
            np.str_,
        ],
    )

    tb_ppc_netflow["ppc_priority"] = (
        tb_ppc_netflow["ppc_priority"] / max(tb_ppc_netflow["ppc_priority"]) * 1e3
    )

    logger.info(
        f"[S3] Run netflow done (takes {round(time.time() - time_start, 3)} sec)"
    )

    return tb_ppc_netflow


def netflowRun4PPC(
    _tb_tgt_inuse,
    ppc_x,
    ppc_y,
):
    # run netflow (for PPP_centers)
    ppc_lst = np.array([[0, ppc_x, ppc_y, 0]])

    res, telescope, tgt_lst_netflow = netflowRun_nofibAssign(
        ppc_lst, _tb_tgt_inuse, True
    )

    for i, (vis, tel) in enumerate(zip(res, telescope)):
        # assigned targets in each ppc
        tgt_assign_id_lst = []
        for tidx, cidx in vis.items():
            tgt_assign_id_lst.append(tgt_lst_netflow[tidx].ID)

    return tgt_assign_id_lst


def netflowAssign(_tb_tgt, _tb_ppc):
    # check fiber assignment of targets
    if len(_tb_ppc) == 0:
        # no ppc
        return _tb_tgt

    _tb_tgt["exptime_assign"] = 0

    # sort ppc by its total priority == sum(weights of the assigned targets in ppc)
    _tb_ppc_pri = _tb_ppc[_tb_ppc.argsort(keys="ppc_priority")]

    # targets with allocated fiber
    for ppc_t in _tb_ppc_pri:
        lst = np.where(
            np.in1d(_tb_tgt["identify_code"], ppc_t["ppc_allocated_targets"]) == True
        )[0]
        _tb_tgt["exptime_assign"].data[lst] += 900

    return _tb_tgt


def netflow_iter(
    _tb_tgt,
    _tb_ppc_netflow,
    weight_para,
    nPPC,
    randomseed=0,
    TraCollision=False,
    numReservedFibers=0,
    fiberNonAllocationCost=0.0,
):
    # iterate the total procedure to re-assign fibers to targets which have not been assigned in the previous/first iteration
    # note that some targets in the dense region may need very long time to be assigned with fibers
    # if targets can not be successfully assigned with fibers in >10 iterations, then directly stop
    # if total number of ppc > nPPC, then directly stop

    if len(_tb_tgt) == 0 or len(_tb_ppc_netflow) == 0:
        return _tb_ppc_netflow

    if (
        sum(_tb_tgt["exptime_assign"] == _tb_tgt["exptime_PPP"]) == len(_tb_tgt)
        or len(_tb_ppc_netflow) >= nPPC
    ):
        # remove ppc with fiber assignment < 0.1%
        _tb_ppc_netflow.remove_rows(
            np.where(_tb_ppc_netflow["ppc_fiber_usage_frac"] < 0.1)[0]
        )
        return _tb_ppc_netflow

    else:
        _tb_ppc_netflow.remove_rows(
            np.where(_tb_ppc_netflow["ppc_fiber_usage_frac"] == 0)[0]
        )
        return _tb_ppc_netflow
        """ skip due to FH upper limit -24-02-07 NEED TO FIX
        #  select non-assigned targets --> PPC determination --> netflow --> if no fibre assigned: shift PPC
        iter_m2 = 0

        while any(_tb_tgt["exptime_assign"] < _tb_tgt["exptime_PPP"]):
            _tb_tgt_t1 = _tb_tgt[_tb_tgt["exptime_assign"] < _tb_tgt["exptime_PPP"]]
            _tb_tgt_t1["exptime_PPP"] = (
                _tb_tgt_t1["exptime_PPP"] - _tb_tgt_t1["exptime_assign"]
            )  # remained exposure time

            _tb_ppc_netflow.remove_rows(np.where(_tb_ppc_netflow["ppc_fiber_usage_frac"] == 0)[0])
            _tb_tgt_t2 = PPP_centers(
                _tb_tgt_t1,
                nPPC - len(_tb_ppc_netflow),
                weight_para,
                randomseed,
            )

            _tb_ppc_netflow_t = netflowRun(
                _tb_tgt_t2, 
                randomseed, 
                TraCollision, 
                numReservedFibers, 
                fiberNonAllocationCost, 
            )

            if len(_tb_ppc_netflow) >= nPPC or iter_m2 >= 10:
                # stop if n_ppc exceeds the requirment
                return _tb_ppc_netflow

            else:
                _tb_ppc_netflow = vstack([_tb_ppc_netflow, _tb_ppc_netflow_t])
                _tb_ppc_netflow.remove_rows(np.where(_tb_ppc_netflow["ppc_fiber_usage_frac"] == 0)[0])
                _tb_tgt = netflowAssign(_tb_tgt, _tb_ppc_netflow)
            
                iter_m2 += 1
                
        return _tb_ppc_netflow 
        #"""


def complete_ppc(_tb_tgt, mode):
    """check completion rate

    Parameters
    ==========
    _tb_tgt : sample

    mode :
        "compOFtgt_weighted" -- completion = (weight(finished) + 0.5 * weight(partial)) / weight(tgt_all)

        "compOFtgt_n"          -- completion = (N(finished) + 0.5 * N(partial)) / N(tgt_all)

        "compOFpsl_n"       -- completion in count, completion in ratio, list of (psl_id, rank) ordered by rank

    Returns
    =======
    completion rates
    """

    if mode == "compOFtgt_weighted":
        # finished
        index_allo = np.where(_tb_tgt["exptime_PPP"] == _tb_tgt["exptime_assign"])[0]

        if len(index_allo) == 0:
            weight_allo = 0

        else:
            weight_allo = sum(_tb_tgt[index_allo]["weight"])

        # patrly observed
        index_part = np.where(
            (_tb_tgt["exptime_PPP"] > _tb_tgt["exptime_assign"])
            & (_tb_tgt["exptime_assign"] > 0)
        )[0]

        if len(index_part) > 0:
            weight_allo += 0.5 * sum(_tb_tgt[index_part]["weight"])

        weight_tot = sum(_tb_tgt["weight"])

        comp = weight_allo / weight_tot

        return comp

    elif mode == "compOFtgt_n":
        # finished
        index_allo = np.where(_tb_tgt["exptime_PPP"] == _tb_tgt["exptime_assign"])[0]
        weight_allo = len(index_allo)

        # patrly observed
        index_part = np.where(
            (_tb_tgt["exptime_PPP"] > _tb_tgt["exptime_assign"])
            & (_tb_tgt["exptime_assign"] > 0)
        )[0]
        weight_allo += 0.5 * len(index_part)

        comp = weight_allo / len(_tb_tgt)

        return comp

    elif mode == "compOFpsl_n":
        # proposal list
        listPsl_ = list(set(_tb_tgt["proposal_id"]))

        PslRank_ = [_tb_tgt[_tb_tgt["proposal_id"] == kk]["rank"][0] for kk in listPsl_]
        rank_index = reversed(np.argsort(PslRank_))

        listPsl = [
            [listPsl_[ll], PslRank_[ll]] for ll in rank_index
        ]  # proposal list ordered by rank

        n_psl = len(listPsl)

        # user priority
        sub_l = np.arange(0, 9.1, 1)

        # completion rate in each proposal (each user-defined priority, each proposal, all input targets)
        comN_sub_psl = []
        comRatio_sub_psl = []

        comp_tot = 0
        for jj in range(n_psl):
            _tb_tgt_t = _tb_tgt[_tb_tgt["proposal_id"] == listPsl[jj][0]]

            count_sub = (
                [sum(_tb_tgt_t["priority"] == ll) for ll in sub_l]
                + [len(_tb_tgt_t)]
                + [len(_tb_tgt)]
            )

            comp_psl = np.where(
                _tb_tgt_t["exptime_PPP"] == _tb_tgt_t["exptime_assign"]
            )[0]
            comp_tot += len(comp_psl)
            comT_t = (
                [sum(_tb_tgt_t["priority"][comp_psl] == ll) for ll in sub_l]
                + [len(comp_psl)]
                + [comp_tot]
            )
            comN_sub_psl.append(comT_t)

            comRatio_sub_psl.append(
                [comT_t[oo] / count_sub[oo] for oo in range(len(count_sub))]
            )

        return np.array(comN_sub_psl), np.array(comRatio_sub_psl), np.array(listPsl)


def PPC_efficiency(tab_ppc_netflow):
    # calculate fiber allocation efficiency

    fib_eff = tab_ppc_netflow["ppc_fiber_usage_frac"].data  # unit --> %

    if max(fib_eff) == 0:
        return fib_eff, 0, 0

    else:
        fib_eff_mean1 = np.mean(fib_eff / max(fib_eff))
        fib_eff_mean2 = np.mean(fib_eff) * 0.01  # unit --> fraction without %
        return fib_eff, fib_eff_mean1, fib_eff_mean2


def fun2opt(para, info):
    """function to be optimized

    Parameters
    ==========
    para: float
        conta,b,c -- weighting scheme

    info:
        samp -- input sample (all, low-mode, medium-mode)

        nPPC_L -- number of PPC for low-resolution mode
        nPPC_M -- number of PPC for medium-resolution mode

        randomSeed -- random seed for np.random

        crMode -- the same with complete_ppc

        checkTraCollision -- boolean; whether or not to allow netflow to check collision of trajectory

    Returns
    =======
    (2 - average_fibEfficiency_L - average_completion_L) + (2 - average_fibEfficiency_M - average_completion_M)
    """
    para_sci, para_exp, para_n = para

    _tb_tgt, _tb_tgt_l, _tb_tgt_m = info["tb_tgt"]

    nppc_l = info["nPPC_l"]
    nppc_m = info["nPPC_m"]

    index_op1 = info["iter"]
    randomseed = info["randomSeed"]

    TraCollision = info["checkTraCollision"]

    completeMode = info["crMode"]

    # --------------------
    tem1 = 0
    tem2 = 0

    mfibEff1 = 0
    CR_fin1 = 0
    mfibEff2 = 0
    CR_fin2 = 0

    if len(_tb_tgt_l) > 0:
        _tb_tgt_l1 = PPP_centers(_tb_tgt_l, nppc_l, para, randomseed, True)

        _tb_ppc_netflow_l = netflowRun(_tb_tgt_l1, randomseed)

        _tb_tgt_l1 = netflowAssign(_tb_tgt_l1, _tb_ppc_netflow_l)

        CR_l = complete_ppc(_tb_tgt_l1, completeMode)

        CR_fin1 = sum(CR_l[-2][:, -2]) / len(CR_l[-1])

        mfibEff1 = PPC_efficiency(_tb_ppc_netflow_l)[2]  # mean of fib --/max(fib)--

        tem1 = 200 - (mfibEff1 + CR_fin1) * 100

    if len(_tb_tgt_m) > 0:
        _tb_tgt_m1 = PPP_centers(_tb_tgt_m, nppc_m, para, randomseed, True)

        _tb_ppc_netflow_m = netflowRun(_tb_tgt_m1, randomseed)

        _tb_tgt_m1 = netflowAssign(_tb_tgt_m1, _tb_ppc_netflow_m)

        CR_m = complete_ppc(_tb_tgt_m1, completeMode)

        CR_fin2 = sum(CR_m[-2][:, -2]) / len(CR_m[-1])

        mfibEff2 = PPC_efficiency(_tb_ppc_netflow_m)[2]  # mean of fib --/max(fib)--

        tem2 = 200 - (mfibEff2 + CR_fin2) * 100

    logger.info(
        f"[S4] Iter {info['iter']+1:.0f}, w_para is [{para_sci:.3f}, {para_exp:.3f}, {para_n:.3f}]; \
                objV is {mfibEff1:.2f} {CR_fin1:.2f} (low-mode) {mfibEff2:.2f} {CR_fin2:.2f} (medium-mode) {tem1+tem2:.2f} (total)."
    )

    info["iter"] += 1

    return tem1 + tem2


def iter_weight(
    _tb_tgt, weight_initialGuess, nppc_l, nppc_m, crMode, randomSeed, TraCollision
):
    """optimize the weighting scheme

    Parameters
    ==========
    samp: table

    weight_initialGuess: [conta, b, c]

    nppc_l -- number of PPC for low-resolution mode
    nppc_m -- number of PPC for medium-resolution mode

    randomSeed -- random seed for np.random

    crmode -- the same with complete_ppc

    TraCollision -- boolean; whether or not to allow netflow to check collision of trajectory

    printTF -- boolean; print results or not

    Returns
    =======
    the optimal weighting scheme [conta, b, c]
    """
    time_s = time.time()
    logger.info("[S4] Optimization started")

    best_weight = opt.fmin(
        fun2opt,
        weight_initialGuess,
        xtol=0.1,
        ftol=0.1,
        args=(
            {
                "tb_tgt": _tb_tgt,
                "nPPC_l": nppc_l,
                "nPPC_m": nppc_m,
                "crMode": crMode,
                "iter": 0,
                "randomSeed": randomSeed,
                "checkTraCollision": TraCollision,
            },
        ),
        disp=True,
        retall=False,
        full_output=False,
        maxiter=200,
        maxfun=200,
    )

    logger.info(f"[S4] Optimization done (takes {time.time()-time_s:.3f} sec)")

    return best_weight


def output(_tb_ppc_tot, _tb_tgt_tot, dirName="output/"):
    """write outputs into ecsv files

    Parameters
    ==========
    _tb_ppc_tot: table of ppc information
    _tb_tgt_tot: table of targets

    Returns
    =======
    ppcList & obList in output/ folder
    """
    ppc_code = _tb_ppc_tot["ppc_code"].data
    ppc_ra = _tb_ppc_tot["ppc_ra"].data
    ppc_dec = _tb_ppc_tot["ppc_dec"].data
    ppc_pa = _tb_ppc_tot["ppc_pa"].data
    ppc_equinox = ["J2000"] * len(_tb_ppc_tot)
    ppc_priority = _tb_ppc_tot["ppc_priority"].data
    ppc_exptime = [900] * len(_tb_ppc_tot)
    ppc_totaltime = [900 + 240] * len(_tb_ppc_tot)
    ppc_resolution = _tb_ppc_tot["ppc_resolution"].data
    ppc_fibAlloFrac = _tb_ppc_tot["ppc_fiber_usage_frac"].data
    ppc_tgtAllo = _tb_ppc_tot["ppc_allocated_targets"].data
    ppc_comment = [" "] * len(_tb_ppc_tot)

    ppcList = Table(
        [
            ppc_code,
            ppc_ra,
            ppc_dec,
            ppc_pa,
            ppc_equinox,
            ppc_priority,
            ppc_exptime,
            ppc_totaltime,
            ppc_resolution,
            ppc_fibAlloFrac,
            ppc_tgtAllo,
            ppc_comment,
        ],
        names=[
            "ppc_code",
            "ppc_ra",
            "ppc_dec",
            "ppc_pa",
            "ppc_equinox",
            "ppc_priority",
            "ppc_exptime",
            "ppc_totaltime",
            "ppc_resolution",
            "ppc_fiber_usage_frac",
            "ppc_allocated_targets",
            "ppc_comment",
        ],
    )

    ppcList.write(
        os.path.join(dirName, "ppcList.ecsv"), format="ascii.ecsv", overwrite=True
    )

    ob_code = _tb_tgt_tot["ob_code"].data
    ob_obj_id = _tb_tgt_tot["obj_id"].data
    ob_cat_id = _tb_tgt_tot["input_catalog_id"].data
    ob_ra = _tb_tgt_tot["ra"].data
    ob_dec = _tb_tgt_tot["dec"].data
    ob_equinox = ["J2000"] * len(_tb_tgt_tot)
    ob_pmras = _tb_tgt_tot["pmra"].data
    ob_pmdecs = _tb_tgt_tot["pmdec"].data
    ob_parallaxs = _tb_tgt_tot["parallax"].data
    ob_priority = _tb_tgt_tot["priority"].data
    ob_exptime = _tb_tgt_tot["exptime"].data
    ob_resolution = _tb_tgt_tot["resolution"].data
    proposal_id = _tb_tgt_tot["proposal_id"].data
    proposal_rank = _tb_tgt_tot["rank"].data
    ob_weight_best = _tb_tgt_tot["weight"].data
    ob_allocate_time_netflow = _tb_tgt_tot["exptime_assign"].data
    ob_filter_g = _tb_tgt_tot["filter_g"].data
    ob_filter_r = _tb_tgt_tot["filter_r"].data
    ob_filter_i = _tb_tgt_tot["filter_i"].data
    ob_filter_z = _tb_tgt_tot["filter_z"].data
    ob_filter_y = _tb_tgt_tot["filter_y"].data
    ob_psf_flux_g = _tb_tgt_tot["psf_flux_g"].data
    ob_psf_flux_r = _tb_tgt_tot["psf_flux_r"].data
    ob_psf_flux_i = _tb_tgt_tot["psf_flux_i"].data
    ob_psf_flux_z = _tb_tgt_tot["psf_flux_z"].data
    ob_psf_flux_y = _tb_tgt_tot["psf_flux_y"].data
    ob_psf_flux_error_g = _tb_tgt_tot["psf_flux_error_g"].data
    ob_psf_flux_error_r = _tb_tgt_tot["psf_flux_error_r"].data
    ob_psf_flux_error_i = _tb_tgt_tot["psf_flux_error_i"].data
    ob_psf_flux_error_z = _tb_tgt_tot["psf_flux_error_z"].data
    ob_psf_flux_error_y = _tb_tgt_tot["psf_flux_error_y"].data
    ob_identify_code = _tb_tgt_tot["identify_code"].data

    obList = Table(
        [
            ob_code,
            ob_obj_id,
            ob_cat_id,
            ob_ra,
            ob_dec,
            ob_equinox,
            ob_pmras,
            ob_pmdecs,
            ob_parallaxs,
            ob_priority,
            ob_exptime,
            ob_resolution,
            proposal_id,
            proposal_rank,
            ob_weight_best,
            ob_allocate_time_netflow,
            ob_filter_g,
            ob_filter_r,
            ob_filter_i,
            ob_filter_z,
            ob_filter_y,
            ob_psf_flux_g,
            ob_psf_flux_r,
            ob_psf_flux_i,
            ob_psf_flux_z,
            ob_psf_flux_y,
            ob_psf_flux_error_g,
            ob_psf_flux_error_r,
            ob_psf_flux_error_i,
            ob_psf_flux_error_z,
            ob_psf_flux_error_y,
            ob_identify_code,
        ],
        names=[
            "ob_code",
            "ob_obj_id",
            "ob_cat_id",
            "ob_ra",
            "ob_dec",
            "ob_equinox",
            "ob_pmra",
            "ob_pmdec",
            "ob_parallax",
            "ob_priority",
            "ob_exptime",
            "ob_resolution",
            "proposal_id",
            "proposal_rank",
            "ob_weight_best",
            "ob_exptime_assign",
            "ob_filter_g",
            "ob_filter_r",
            "ob_filter_i",
            "ob_filter_z",
            "ob_filter_y",
            "ob_psf_flux_g",
            "ob_psf_flux_r",
            "ob_psf_flux_i",
            "ob_psf_flux_z",
            "ob_psf_flux_y",
            "ob_psf_flux_error_g",
            "ob_psf_flux_error_r",
            "ob_psf_flux_error_i",
            "ob_psf_flux_error_z",
            "ob_psf_flux_error_y",
            "ob_identify_code",
        ],
    )

    obList.write(
        os.path.join(dirName, "obList.ecsv"), format="ascii.ecsv", overwrite=True
    )

    np.save(os.path.join(dirName, "obj_allo_tot.npy"), _tb_ppc_tot)


def plotCR(CR, sub_lst, _tb_ppc_tot, dirName="output/", show_plots=False):
    # plot completion rate and fiber allocation efficiency

    plt.figure(figsize=(13, 5))

    plt.subplot(121)

    plt.bar(
        np.arange(1, len(CR) + 1, 1),
        100 * CR[:, -2],
        width=0.8,
        fc="tomato",
        ec="none",
        alpha=0.6,
        zorder=10,
    )

    plt.plot([0, len(CR) + 1], [80, 80], "k--", lw=2, zorder=11)
    plt.plot(
        [0, len(CR) + 1],
        [100 * np.mean(CR[:, -2]), 100 * np.mean(CR[:, -2])],
        "--",
        color="tomato",
        lw=2,
        zorder=11,
    )
    plt.text(
        (len(CR) + 1) * 0.85,
        100 * np.mean(CR[:, -2]),
        "{:2.2f}%".format(100 * np.mean(CR[:, -2])),
        color="r",
        fontsize=12,
    )

    plt.xlim(0, len(CR) + 1)
    plt.ylim(0, 100 * CR[:, -2].max() + 5)
    plt.ylabel("completeness (%)", fontsize=18)
    plt.xticks(
        np.arange(1, len(sub_lst) + 1, 1),
        [str(kk[0])[5:] + "_" + str(kk[1]) for kk in sub_lst],
        fontsize=12,
        rotation=90,
    )
    plt.yticks(fontsize=16)
    plt.grid()

    plt.subplot(122)

    _tb_ppc_tot = _tb_ppc_tot[_tb_ppc_tot.argsort(keys="ppc_priority")]
    fib_eff = _tb_ppc_tot["ppc_fiber_usage_frac"].data

    plt.bar(
        np.arange(0, len(fib_eff), 1),
        fib_eff,
        width=0.8,
        fc="tomato",
        ec="none",
        alpha=0.6,
        zorder=10,
    )
    plt.plot([0, len(fib_eff) + 1], [80, 80], "k--", lw=2, zorder=11)
    plt.plot(
        [0, len(fib_eff) + 1],
        [np.mean(fib_eff), np.mean(fib_eff)],
        "--",
        color="tomato",
        lw=2,
        zorder=11,
    )
    plt.text(
        len(fib_eff) * 0.85,
        np.mean(fib_eff),
        "{:2.2f}%".format(np.mean(fib_eff)),
        color="r",
        fontsize=12,
    )

    plt.xlim(0, len(fib_eff) + 1)
    plt.ylim(0, max(fib_eff) * 1.1)
    plt.xlabel("PPC", fontsize=18)
    plt.ylabel("fiber alloc fraction (%)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(dirName, "ppp_result.jpg"), dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()


def run(
    bench_info,
    readtgt_con,
    TimeOnSource_l,
    TimeOnSource_m,
    dirName="output/",
    numReservedFibers=0,
    fiberNonAllocationCost=0.0,
    show_plots=False,
):
    global bench
    bench = bench_info

    tb_tgt, tb_sel_l, tb_sel_m, tb_tgt_l, tb_tgt_m = readTarget(
        readtgt_con["mode_readtgt"], readtgt_con["para_readtgt"]
    )

    nppc_l = int(np.ceil(TimeOnSource_l / 900.0))
    nppc_m = int(np.ceil(TimeOnSource_m / 900.0))

    randomseed = 2

    TraCollision = False
    multiProcess = True

    crMode = "compOFpsl_n"

    """
    weight_guess = [2, -0.1, 0.05]
    para_sci,para_exp,para_n = iter_weight(
        [tb_tgt,tb_tgt_l,tb_tgt_m],
        weight_guess,
        nppc_l,
        nppc_m,
        crMode,
        randomseed,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
    )
    #"""
    para_sci, para_exp, para_n = [1.4, 0.1, 0.1]

    # LR--------------------------------------------
    ppc_lst_l = PPP_centers(
        tb_sel_l, nppc_l, [para_sci, para_exp, para_n], randomseed, multiProcess
    )

    tb_tgt_l1 = Table.copy(tb_tgt_l)
    tb_tgt_l1.meta["PPC"] = ppc_lst_l

    tb_tgt_l1 = sciRank_pri(tb_tgt_l1)
    tb_tgt_l1 = count_N(tb_tgt_l1)
    tb_tgt_l1 = weight(tb_tgt_l1, para_sci, para_exp, para_n)

    tb_ppc_l = netflowRun(
        tb_tgt_l1,
        randomseed,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
    )

    tb_tgt_l1 = netflowAssign(tb_tgt_l1, tb_ppc_l)

    tb_ppc_l_fin = netflow_iter(
        tb_tgt_l1,
        tb_ppc_l,
        [para_sci, para_exp, para_n],
        nppc_l,
        randomseed,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
    )
    tb_tgt_l_fin = netflowAssign(tb_tgt_l1, tb_ppc_l_fin)

    # MR--------------------------------------------
    ppc_lst_m = PPP_centers(
        tb_sel_m, nppc_m, [para_sci, para_exp, para_n], randomseed, multiProcess
    )

    tb_tgt_m1 = Table.copy(tb_tgt_m)
    tb_tgt_m1.meta["PPC"] = ppc_lst_m

    tb_tgt_m1 = sciRank_pri(tb_tgt_m1)
    tb_tgt_m1 = count_N(tb_tgt_m1)
    tb_tgt_m1 = weight(tb_tgt_m1, para_sci, para_exp, para_n)

    tb_ppc_m = netflowRun(
        tb_tgt_m1,
        randomseed,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
    )

    tb_tgt_m1 = netflowAssign(tb_tgt_m1, tb_ppc_m)

    tb_ppc_m_fin = netflow_iter(
        tb_tgt_m1,
        tb_ppc_m,
        [para_sci, para_exp, para_n],
        nppc_m,
        randomseed,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
    )
    tb_tgt_m_fin = netflowAssign(tb_tgt_m1, tb_ppc_m_fin)

    if nppc_l > 0:
        if nppc_m > 0:
            tb_ppc_tot = vstack([tb_ppc_l_fin, tb_ppc_m_fin])
            tb_tgt_tot = vstack([tb_tgt_l_fin, tb_tgt_m_fin])
        else:
            tb_ppc_tot = tb_ppc_l_fin.copy()
            tb_tgt_tot = tb_tgt_l_fin.copy()
            if len(tb_tgt_m) > 0:
                logger.warning("no allocated time for MR")
    else:
        if nppc_m > 0:
            tb_ppc_tot = tb_ppc_m_fin.copy()
            tb_tgt_tot = tb_tgt_m_fin.copy()
            if len(tb_tgt_l) > 0:
                logger.warning("no allocated time for LR")
        else:
            logger.error("Please specify n_pcc_l or n_pcc_m")

    output(tb_ppc_tot, tb_tgt_tot, dirName=dirName)

    CR_tot, CR_tot_, sub_tot = complete_ppc(tb_tgt_tot, "compOFpsl_n")

    plotCR(CR_tot_, sub_tot, tb_ppc_tot, dirName=dirName, show_plots=show_plots)
