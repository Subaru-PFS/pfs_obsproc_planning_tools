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
from scipy.optimize import minimize

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, join
from dateutil import parser, tz
from functools import partial
from itertools import chain
from loguru import logger
from matplotlib.path import Path
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.neighbors import KernelDensity
from qplan import q_db, q_query, entity
from qplan.util.site import site_subaru as observer
from qplan.util.eph_cache import EphemerisCache
from datetime import datetime, timedelta, timezone, date
from ginga.misc.log import get_logger

logger_qplan = get_logger("qplan_test", null=True)
eph_cache = EphemerisCache(logger_qplan, precision_minutes=5)

warnings.filterwarnings("ignore")

# below for netflow
import ets_fiber_assigner.netflow as nf
from ics.cobraOps.Bench import Bench
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
    df = df.drop_duplicates(
        subset=["proposal_id", "obj_id", "input_catalog_id", "resolution"],
        inplace=False,
        ignore_index=True,
    )
    num2 = len(df)
    logger.info(f"Duplication removed: {num1} --> {num2}")
    return df


def visibility_checker(tb_tgt, obstimes, start_time_list, stop_time_list):
    tz_HST = tz.gettz("US/Hawaii")

    min_el = 30.0
    max_el = 85.0

    tb_tgt["is_visible"] = False

    for i in range(len(tb_tgt)):
        target = entity.StaticTarget(
            name=tb_tgt["ob_code"][i],
            ra=tb_tgt["ra"][i],
            dec=tb_tgt["dec"][i],
            equinox=2000.0,
        )
        total_time = np.ceil(tb_tgt["exptime_usr"][i] / tb_tgt["single_exptime"][i]) * (
            tb_tgt["single_exptime"][i] + 300.0
        )  # SEC

        t_obs_ok = 0

        for date in obstimes:
            date_t = parser.parse(f"{date} 12:00 HST")
            observer.set_date(date_t)
            default_start_time = observer.evening_twilight_18()
            default_stop_time = observer.morning_twilight_18()

            start_override = None
            stop_override = None

            for item in start_time_list:
                next_date = (
                    datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if (date in item) and parser.parse(f"{item} HST") > default_start_time:
                    start_override = parser.parse(f"{item} HST")
                elif (next_date in item) and parser.parse(
                    f"{item} HST"
                ) < default_stop_time:
                    start_override = parser.parse(f"{item} HST")

            for item in stop_time_list:
                next_date = (
                    datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if (date in item) and parser.parse(f"{item} HST") > default_start_time:
                    stop_override = parser.parse(f"{item} HST")
                elif (next_date in item) and parser.parse(
                    f"{item} HST"
                ) < default_stop_time:
                    stop_override = parser.parse(f"{item} HST")

            if start_override is not None:
                start_time = start_override
            else:
                start_time = default_start_time

            if stop_override is not None:
                stop_time = stop_override
            else:
                stop_time = default_stop_time

            if i == 0:
                logger.info(f"date: {date}, start={start_time}, stop={stop_time}")

            key = target
            obs_ok, t_start, t_stop = eph_cache.observable(
                key,
                target,
                observer,
                start_time,
                stop_time,
                min_el,
                max_el,
                total_time,
            )

            if t_start is None or t_stop is None:
                t_obs_ok += 0
                continue

            if t_stop > t_start:
                t_obs_ok += (t_stop - t_start).seconds  # SEC
            else:
                t_obs_ok += 0

        if t_obs_ok >= total_time:
            tb_tgt["is_visible"][i] = True

    logger.info(
        f'{sum(tb_tgt["is_visible"])}/{len(tb_tgt)} are visible during the given obstimes.'
    )

    tb_tgt = tb_tgt[tb_tgt["is_visible"] == True]

    psl_id = sorted(set(tb_tgt["proposal_id"]))

    for psl_id_ in psl_id:
        tb_tgt_ = tb_tgt[tb_tgt["proposal_id"] == psl_id_]

        if sum(tb_tgt_["exptime_usr"]) / 3600.0 < tb_tgt_["allocated_time_tac"][0]:
            logger.error(
                f"{psl_id_}: visible targets too limited to achieve the allocated FHs. Please change obstime."
            )

    return tb_tgt

def queryQueue(psl_id_list, DBPath_qDB, tb_queuedb_filename):
    """
    Query queue database for executed observations and exposure times per proposal.

    Parameters
    ----------
    psl_id_list : list[str]
        List of proposal IDs to query (e.g., ["S25B-024QN", "S25A-043QF"]).
    DBPath_qDB : str
        Path to the QueueDB configuration file.
    tb_queuedb_filename : str
        Path to save/load the cached results table (ECSV).

    Returns
    -------
    tb_queuedb : astropy.table.Table
        Table containing exposure time information per observation.
    """
    # --- try reading cached result ---
    if os.path.exists(tb_queuedb_filename):
        try:
            tb_queuedb = Table.read(tb_queuedb_filename)
            logger.info(f"Loaded cached queue table: {tb_queuedb_filename}")
            return tb_queuedb
        except Exception as e:
            logger.info("[S1] Querying the qdb (no cache found)")

    qdb = q_db.QueueDatabase(logger_qplan) 
    qdb.read_config(DBPath_qDB) 
    qdb.connect() 
    qa = q_db.QueueAdapter(qdb) 
    qq = q_query.QueueQuery(qa, use_cache=False)

    # --- collect results ---
    results = []
    counter = 0

    for psl_id in psl_id_list:
        logger.info(f"Querying qDB for {psl_id}")
        ex_obs_list = qq.get_executed_obs_by_proposal(psl_id)
        if not ex_obs_list:
            continue

        for ex_ob in ex_obs_list:
            exps = qq.get_exposures(ex_ob)
            ob = qq.get_ob(ex_ob.ob_key)
            arm = ob.inscfg.qa_reference_arm

            exptime_b = sum(exp.effective_exptime_b or 0 for exp in exps)
            exptime_r = sum(exp.effective_exptime_r or 0 for exp in exps)
            exptime_m = sum(exp.effective_exptime_m or 0 for exp in exps)
            exptime_n = sum(exp.effective_exptime_n or 0 for exp in exps)

            # select arm-specific exposure time
            arm_map = {"b": exptime_b, "r": exptime_r, "m": exptime_m, "n": exptime_n}
            exptime_selected = arm_map.get(arm, 0)

            if exptime_selected >= 0:
                counter += 1
                results.append([
                    counter,
                    psl_id,
                    ex_ob.ob_key[1],
                    arm,
                    exptime_selected,
                    exptime_b,
                    exptime_r,
                    exptime_m,
                    exptime_n,
                    len(exps) * 450.0,  # nominal exposure time per OB
                ])

    if not results:
        logger.warning("No executed observations found in any proposal.")
        return Table()

    # --- create and save table ---
    tb_queuedb = Table(
        np.array(results),
        names=[
            "N",
            "psl_id",
            "ob_code",
            "ref_arm",
            "eff_exptime_done_real",
            "eff_exptime_done_real_b",
            "eff_exptime_done_real_r",
            "eff_exptime_done_real_m",
            "eff_exptime_done_real_n",
            "exptime_done_real",
        ],
    )

    tb_queuedb.write(tb_queuedb_filename, overwrite=True)

    return tb_queuedb


def readTarget(mode, para, tb_queuedb):
    """Read target list including:
       'ob_code' 'ra' 'dec' 'priority' 'exptime' 'exptime_tac' 'resolution' 'proposal_id' 'rank' 'grade' 'allocated_time'

    Parameters
    ==========
    para : dict
        mode:
            'classic' , 'queue'

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

    if len(para["localPath_tgt"]) > 0:
        tb_tgt = Table.read(para["localPath_tgt"])
        logger.info(f"[S1] Target list is read from {para['localPath_tgt']}.")

        if len(tb_tgt) == 0:
            logger.warning("[S1] No input targets.")
            return Table(), Table(), Table(), Table(), Table()

    elif None in para["DBPath_tgt"]:
        logger.error("[S1] Incorrect connection info to database is provided.")
        return Table(), Table(), Table(), Table(), Table()

    else:
        import pandas as pd
        import psycopg2
        import sqlalchemy as sa

        DBads = DBinfo(para["DBPath_tgt"])
        tgtDB = sa.create_engine(DBads)

        def query_target_from_db(proposalId):
            sql = f"SELECT ob_code,obj_id,c.input_catalog_id,ra,dec,epoch,priority,pmra,pmdec,parallax,effective_exptime,single_exptime,qa_reference_arm,is_medium_resolution,proposal.proposal_id,rank,grade,allocated_time_lr+allocated_time_mr as \"allocated_time\",allocated_time_lr,allocated_time_mr,filter_g,filter_r,filter_i,filter_z,filter_y,psf_flux_g,psf_flux_r,psf_flux_i,psf_flux_z,psf_flux_y,psf_flux_error_g,psf_flux_error_r,psf_flux_error_i,psf_flux_error_z,psf_flux_error_y,total_flux_g,total_flux_r,total_flux_i,total_flux_z,total_flux_y,total_flux_error_g,total_flux_error_r,total_flux_error_i,total_flux_error_z,total_flux_error_y FROM target JOIN proposal ON target.proposal_id=proposal.proposal_id JOIN input_catalog AS c ON target.input_catalog_id = c.input_catalog_id WHERE proposal.proposal_id in ('{proposalId}') AND c.active;"

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
                    "single_exptime",
                    "qa_reference_arm",
                    "is_medium_resolution",
                    "proposal_id",
                    "rank",
                    "grade",
                    "allocated_time_tac",
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
                    "total_flux_g",
                    "total_flux_r",
                    "total_flux_i",
                    "total_flux_z",
                    "total_flux_y",
                    "total_flux_error_g",
                    "total_flux_error_r",
                    "total_flux_error_i",
                    "total_flux_error_z",
                    "total_flux_error_y",
                ],
            )
            # convert column names
            df_tgt = df_tgt.rename(columns={"epoch": "equinox"})
            df_tgt = df_tgt.rename(columns={"effective_exptime": "exptime_usr"})
            df_tgt = df_tgt.rename(columns={"is_medium_resolution": "resolution"})

            # convert Boolean to String
            df_tgt["resolution"] = [
                "M" if v == True else "L" for v in df_tgt["resolution"]
            ]
            df_tgt["allocated_time_tac"] = [
                df_tgt["allocated_time_lr"][ii]
                if df_tgt["resolution"][ii] == "L"
                else df_tgt["allocated_time_mr"][ii]
                for ii in range(len(df_tgt))
            ]
            df_tgt = df_tgt.drop(columns=["allocated_time_lr", "allocated_time_mr"])

            df_tgt = removeObjIdDuplication(df_tgt)

            df_tgt["psf_flux_g"][df_tgt["psf_flux_g"].isnull()] = np.nan
            df_tgt["psf_flux_r"][df_tgt["psf_flux_r"].isnull()] = np.nan
            df_tgt["psf_flux_i"][df_tgt["psf_flux_i"].isnull()] = np.nan
            df_tgt["psf_flux_z"][df_tgt["psf_flux_z"].isnull()] = np.nan
            df_tgt["psf_flux_y"][df_tgt["psf_flux_y"].isnull()] = np.nan

            df_tgt["psf_flux_error_g"][df_tgt["psf_flux_error_g"].isnull()] = np.nan
            df_tgt["psf_flux_error_r"][df_tgt["psf_flux_error_r"].isnull()] = np.nan
            df_tgt["psf_flux_error_i"][df_tgt["psf_flux_error_i"].isnull()] = np.nan
            df_tgt["psf_flux_error_z"][df_tgt["psf_flux_error_z"].isnull()] = np.nan
            df_tgt["psf_flux_error_y"][df_tgt["psf_flux_error_y"].isnull()] = np.nan

            df_tgt["total_flux_g"][df_tgt["total_flux_g"].isnull()] = np.nan
            df_tgt["total_flux_r"][df_tgt["total_flux_r"].isnull()] = np.nan
            df_tgt["total_flux_i"][df_tgt["total_flux_i"].isnull()] = np.nan
            df_tgt["total_flux_z"][df_tgt["total_flux_z"].isnull()] = np.nan
            df_tgt["total_flux_y"][df_tgt["total_flux_y"].isnull()] = np.nan

            df_tgt["total_flux_error_g"][df_tgt["total_flux_error_g"].isnull()] = np.nan
            df_tgt["total_flux_error_r"][df_tgt["total_flux_error_r"].isnull()] = np.nan
            df_tgt["total_flux_error_i"][df_tgt["total_flux_error_i"].isnull()] = np.nan
            df_tgt["total_flux_error_z"][df_tgt["total_flux_error_z"].isnull()] = np.nan
            df_tgt["total_flux_error_y"][df_tgt["total_flux_error_y"].isnull()] = np.nan

            cols = [
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
                "total_flux_g",
                "total_flux_r",
                "total_flux_i",
                "total_flux_z",
                "total_flux_y",
                "total_flux_error_g",
                "total_flux_error_r",
                "total_flux_error_i",
                "total_flux_error_z",
                "total_flux_error_y",
            ]

            for col in cols:
                # Convert the column to numeric; non-convertible values (e.g., "N/A") become np.nan.
                df_tgt[col] = pd.to_numeric(df_tgt[col], errors="coerce")

            tb_tgt = Table.from_pandas(df_tgt)

            for col in ["filter_g", "filter_r", "filter_i", "filter_z", "filter_y"]:
                tb_tgt[col] = tb_tgt[col].astype("str")

            if proposalId == "S25B-126QN":
                tb_tgt = tb_tgt[tb_tgt["priority"] <= 3]

            """
            for col in ["psf_flux_g","psf_flux_r", "psf_flux_i", "psf_flux_z", "psf_flux_y"]:
                tb_tgt[col] = tb_tgt[col].astype(float)

            for col in ["psf_flux_error_g","psf_flux_error_r", "psf_flux_error_i", "psf_flux_error_z", "psf_flux_error_y"]:
                tb_tgt[col] = tb_tgt[col].astype(float)
            #"""

            conn.close()

            return tb_tgt

    proposalid = para["proposalIds"]

    tb_tgt_lst = []
    for proposalid_ in proposalid:
        tb_tgt_lst.append(query_target_from_db(proposalid_))
    tb_tgt = vstack(tb_tgt_lst)

    tb_tgt["ra"] = tb_tgt["ra"].astype(float)
    tb_tgt["dec"] = tb_tgt["dec"].astype(float)
    tb_tgt["ob_code"] = tb_tgt["ob_code"].astype(str)
    tb_tgt["identify_code"] = [
        tt["proposal_id"] + "_" + tt["ob_code"]
        if "proposal_id" in tb_tgt.columns
        else tt["ob_code"]
        for tt in tb_tgt
    ]
    tb_tgt["exptime_assign"] = 0.0
    tb_tgt["exptime_done"] = 0.0  # observed exptime

    ## --only for 020QN, need to confirm with Pyo-san-- updated FH (250530): no need as tgt DB has updated allocated_time_tac
    # """
    # tb_tgt["allocated_time_tac"][tb_tgt["proposal_id"] == 'S25A-058QN'] = 19848.75
    # tb_tgt["allocated_time_tac"][tb_tgt["proposal_id"] == 'S25A-020QN'] = 3237.25
    # tb_tgt["allocated_time_tac"][tb_tgt["proposal_id"] == 'S25A-099QN'] = 6803.00
    # tb_tgt["allocated_time_tac"][tb_tgt["proposal_id"] == 'S25A-096QN'] = 2661.00
    # tb_tgt["allocated_time_tac"][tb_tgt["proposal_id"] == 'S25A-042QN'] = 18758.75
    # tb_tgt["allocated_time_tac"][tb_tgt["proposal_id"] == 'S25A-101QN'] = 4363.50
    # tb_tgt["allocated_time_tac"][tb_tgt["proposal_id"] == 'S25A-064QN'] = 10000.0
    # """

    # for grade c programs, set completion rate as 70% as upper limit: no need as tgt DB has updated allocated_time_tac
    """
    proposalid = ["S25B-086QN", "S25B-056QN", "S25B-092QN", "S25B-134QN", "S25B-120QN", "S25B-047QN", "S25B-126QN", "S25B-125QN", "S25B-136QN", "S25B-048QN"]
    mask = np.isin(tb_tgt["proposal_id"], proposalid)
    tb_tgt["allocated_time_tac"][mask] = tb_tgt["allocated_time_tac"][mask] * 0.7
    #"""

    if len(set(tb_tgt["single_exptime"])) > 1:
        logger.error(
            "[S1] Multiple single-exptime are given. Not accepted now (240709)."
        )
        return Table(), Table(), Table(), Table(), Table()

    tb_tgt.meta["single_exptime"] = list(set(tb_tgt["single_exptime"]))[0]
    logger.info(
        f"[S1] The single exptime is set to {tb_tgt.meta['single_exptime']:.2f} sec."
    )

    tb_tgt.meta["PPC"] = np.array([])
    tb_tgt.meta["PPC_origin"] = "auto"

    if mode == "queue":
        tb_tgt["allocated_time_done"] = 0
        tb_tgt["allocated_time"] = 0

        # list of all psl_id
        psl_id = sorted(set(tb_tgt["proposal_id"]))

        # """
        # connect with queueDB
        # join on the key columns
        tb_tgt = join(tb_tgt, tb_queuedb,
                        keys_left=["proposal_id", "ob_code"],
                        keys_right=["psl_id", "ob_code"],
                        join_type="left")
        
        exptime_usr = np.ma.filled(tb_tgt["exptime_usr"], 0.0)
        exptime_done_real = np.ma.filled(tb_tgt["eff_exptime_done_real"], 0.0)

        exptime_usr = exptime_usr.astype(float)
        exptime_done_real = exptime_done_real.astype(float)
        
        tb_tgt["exptime_done"] = np.minimum(exptime_usr, exptime_done_real)

        tb_tgt.rename_column("ob_code_1", "ob_code")
        cols_to_remove = [c for c in tb_tgt.colnames if c in tb_queuedb.colnames and "ob_code" not in c]
        tb_tgt.remove_columns(cols_to_remove)
        # """

        tb_tgt["exptime"] = tb_tgt["exptime_usr"] - tb_tgt["exptime_done"]

        # update allocated time
        for psl_id_ in psl_id:
            tb_tgt_tem_l = tb_tgt[
                (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "L")
            ]
            tb_tgt_tem_m = tb_tgt[
                (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "M")
            ]

            if len(tb_tgt_tem_l) > 0:
                # total observed FH for the proposal (LR)
                FH_psl_done = sum(tb_tgt_tem_l["exptime_done"]) / 3600.0
                tb_tgt["allocated_time_done"][
                    (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "L")
                ] = FH_psl_done
                FH_psl = tb_tgt["allocated_time_tac"][
                    (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "L")
                ].data[0]
                FH_comp = tb_tgt["allocated_time_done"][
                    (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "L")
                ].data[0]
                logger.info(
                    f"{psl_id_} (LR): allocated FH = {FH_psl:.2f}, achieved FH = {FH_comp:.2f}, CR = {FH_comp/FH_psl*100.0:.2f}%"
                )

            if len(tb_tgt_tem_m) > 0:
                # total observed FH for the proposal (MR)
                FH_psl_done = sum(tb_tgt_tem_m["exptime_done"]) / 3600.0
                tb_tgt["allocated_time_done"][
                    (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "M")
                ] = FH_psl_done
                FH_psl = tb_tgt["allocated_time_tac"][
                    (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "M")
                ].data[0]
                FH_comp = tb_tgt["allocated_time_done"][
                    (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "M")
                ].data[0]
                logger.info(
                    f"{psl_id_} (MR): allocated FH = {FH_psl:.2f}, achieved FH = {FH_comp:.2f}, CR = {FH_comp/FH_psl*100.0:.2f}%"
                )

        tb_tgt["allocated_time"] = (
            tb_tgt["allocated_time_tac"] - tb_tgt["allocated_time_done"]
        )
        # tb_tgt["allocated_time"][tb_tgt["allocated_time"] < 0] = 0
        tb_tgt = tb_tgt[tb_tgt["allocated_time"] > 0]

        # remove complete targets
        n_tgt1 = len(tb_tgt)
        tb_tgt = tb_tgt[tb_tgt["exptime"] > 0]
        tb_tgt["exptime_PPP"] = (
            np.ceil(tb_tgt["exptime"] / 900) * 900
        )  # exptime needs to be multiples of 900 so netflow can be successfully executed
        n_tgt2 = len(tb_tgt)
        logger.info(
            f"There are {n_tgt2:.0f} (partial-obs: {sum(tb_tgt['exptime_done'] > 0):.0f}) / {n_tgt1:.0f} targets not completed"
        )

        # only for 064 since too huge list, FIX needed
        #msk = np.in1d(tb_tgt["proposal_id"], ["S25B-049QN"]) * (tb_tgt["ra"] > 100)
        #tb_tgt = tb_tgt[~msk]

        """
        if proposalId == 'S25B-126QN':
            tb_tgt = tb_tgt[tb_tgt["priority"]<=3]

        if proposalId in ["S25A-043QF", "S25A-119QF", "S25A-111QF", "S25A-116QF", "S25A-126QF", "S25A-017QF", "S25A-019QF", "S25A-112QF", "S25A-030QF", "S25A-034QF"]:
            n_tgt1 = len(tb_tgt)
            tb_tgt = tb_tgt[(tb_tgt["ra"]>280) * (tb_tgt["dec"]<70) * (tb_tgt["dec"]>-20)]
            n_tgt2 = len(tb_tgt)
            logger.info(f"Visibility limits: {proposalId} {n_tgt1:.0f} -> {n_tgt2:.0f}")
    
        msk = (tb_tgt["priority"] > 3) & (tb_tgt["proposal_id"] == 'S25A-064QN')
        tb_tgt = tb_tgt[~msk]

        for pslid_ in ["S25A-043QF", "S25A-119QF", "S25A-111QF", "S25A-116QF", "S25A-126QF", "S25A-017QF", "S25A-019QF", "S25A-112QF", "S25A-030QF", "S25A-034QF"]:
            n_tgt1 = len(tb_tgt)
            msk = ((tb_tgt["ra"] <= 280) | (tb_tgt["dec"] > 5)) & (tb_tgt["proposal_id"] == pslid_)
            tb_tgt = tb_tgt[~msk]
            n_tgt2 = len(tb_tgt)
            logger.info(f"Visibility limits: {pslid_} {n_tgt1:.0f} -> {n_tgt2:.0f}")
        #"""

        if para["visibility_check"]:
            tb_tgt = visibility_checker(
                tb_tgt, para["obstimes"], para["starttimes"], para["stoptimes"]
            )

        # separete the sample by 'resolution' (L/M)
        tb_tgt_l = tb_tgt[tb_tgt["resolution"] == "L"]
        tb_tgt_m = tb_tgt[tb_tgt["resolution"] == "M"]

        logger.info(f"[S1] observation mode = {mode}")
        logger.info(
            f"[S1] Read targets done (takes {round(time.time()-time_start,3):.2f} sec)."
        )
        logger.info(f"[S1] There are {len(set(tb_tgt['proposal_id'])):.0f} proposals.")
        logger.info(
            f"[S1] n_tgt_low = {len(tb_tgt_l):.0f}, n_tgt_medium = {len(tb_tgt_m):.0f}"
        )
        return tb_tgt, tb_tgt_l, tb_tgt_m, tb_queuedb, tb_queuedb


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

    _tb_tgt["rank_fin"] = np.exp(SciUsr_Ranktot)

    weight_max = max(_tb_tgt["rank_fin"])
    _tb_tgt["rank_fin"][
        (_tb_tgt["exptime_done"] > 0) | (_tb_tgt["exptime_PPP"] < _tb_tgt["exptime"])
    ] += weight_max

    return _tb_tgt


"""
def weight(_tb_tgt, para_sci, para_exp, para_n):
    # calculate weights of targets (higher weights mean more important)
    if len(_tb_tgt) == 0:
        return _tb_tgt

    weight_t = (
        pow(para_sci, _tb_tgt["rank_fin"])
        * pow(_tb_tgt["exptime_PPP"] / _tb_tgt.meta["single_exptime"], para_exp)
        * pow(_tb_tgt["local_count"], para_n)
    )

    _tb_tgt["weight"] = weight_t
    _tb_tgt["weight"][np.isnan(_tb_tgt["weight"])] = 0

    weight_max = max(_tb_tgt["weight"])
    _tb_tgt["weight"][
        (_tb_tgt["exptime_done"] > 0) | (_tb_tgt["exptime_PPP"] < _tb_tgt["exptime"])
    ] += weight_max

    return _tb_tgt
#"""


def target_DBSCAN(_tb_tgt, sep=1.38):
    # separate targets into different groups
    # haversine uses (dec,ra) in radian;
    """
    if len(_tb_tgt) < 2:
        db = DBSCAN(eps=np.radians(sep), min_samples=1, metric="haversine").fit(
            np.radians([_tb_tgt["dec"], _tb_tgt["ra"]]).T
        )
        labels = db.labels_
    else:
        db = HDBSCAN(min_cluster_size=2, metric="haversine").fit(
                    np.radians([_tb_tgt["dec"], _tb_tgt["ra"]]).T
                )
        labels = db.dbscan_clustering(np.radians(sep), min_cluster_size=1)
    #"""

    db = DBSCAN(eps=np.radians(sep), min_samples=1, metric="haversine").fit(
        np.radians([_tb_tgt["dec"], _tb_tgt["ra"]]).T
    )
    labels = db.labels_

    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    tgt_group = []
    tgt_pri_ord = []

    for ii in range(n_clusters):
        tgt_t_pri_tot = sum(_tb_tgt[labels == ii]["rank_fin"])
        tgt_pri_ord.append([ii, tgt_t_pri_tot])

    tgt_pri_ord.sort(key=lambda x: x[1], reverse=True)

    for ii in np.array(tgt_pri_ord)[:, 0]:
        tgt_t = _tb_tgt[labels == ii]
        tgt_group.append(tgt_t)
        print(
            f'({tgt_t["ra"][0]}, {tgt_t["dec"][0]}): {set(tgt_t["proposal_id"])}, {sum(tgt_t["rank_fin"])}'
        )

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


def objective1(params, _tb_tgt):
    """
    Objective function to optimize the PPC parameters.

    Parameters:
      params: list or array-like of [ppc_ra, ppc_dec, ppc_pa]
      tb: pandas DataFrame containing target information, including 'i2_mag'

    Returns:
      Negative of the weighted sum of bright and faint counts (since we minimize).
    """
    ra, dec = params
    pa = 0.0
    # lst_tgtID_assign = netflowRun4PPC(_tb_tgt, ra, dec, pa)
    index_ = PFS_FoV(ra, dec, pa, _tb_tgt)
    lst_tgtID_assign = _tb_tgt["identify_code"][index_]

    index_assign = np.in1d(_tb_tgt["identify_code"], lst_tgtID_assign)

    N_P0 = sum(index_assign * (_tb_tgt["priority"] == 0))
    N_P1 = sum(index_assign * (_tb_tgt["priority"] == 1))
    N_P2 = sum(index_assign * (_tb_tgt["priority"] == 2))
    N_P3 = sum(index_assign * (_tb_tgt["priority"] == 3))
    N_P4 = sum(index_assign * (_tb_tgt["priority"] == 4))
    N_P5 = sum(index_assign * (_tb_tgt["priority"] == 5))
    N_P6 = sum(index_assign * (_tb_tgt["priority"] == 6))
    N_P7 = sum(index_assign * (_tb_tgt["priority"] == 7))
    N_P8 = sum(index_assign * (_tb_tgt["priority"] == 8))
    N_P9 = sum(index_assign * (_tb_tgt["priority"] == 9))
    N_P999 = sum(index_assign * (_tb_tgt["priority"] == 999))

    N_P0_ = sum((_tb_tgt["priority"] == 0))
    N_P1_ = sum((_tb_tgt["priority"] == 1))
    N_P2_ = sum((_tb_tgt["priority"] == 2))
    N_P3_ = sum((_tb_tgt["priority"] == 3))
    N_P4_ = sum((_tb_tgt["priority"] == 4))
    N_P5_ = sum((_tb_tgt["priority"] == 5))
    N_P6_ = sum((_tb_tgt["priority"] == 6))
    N_P7_ = sum((_tb_tgt["priority"] == 7))
    N_P8_ = sum((_tb_tgt["priority"] == 8))
    N_P9_ = sum((_tb_tgt["priority"] == 9))
    N_P999_ = sum((_tb_tgt["priority"] == 999))

    N_Pall = len(lst_tgtID_assign)

    print(
        f"{ra}, {dec}, {pa}, {N_Pall}/{len(_tb_tgt)}, {N_P0}/{N_P0_}, {N_P1}/{N_P1_}, {N_P2}/{N_P2_}, {N_P3}/{N_P3_}, {N_P4}/{N_P4_}, {N_P5}/{N_P5_}, {N_P6}/{N_P6_}, {N_P7}/{N_P7_}, {N_P8}/{N_P8_}, {N_P9}/{N_P9_}, {N_P999}/{N_P999_}"
    )

    # Define weights: you can adjust these based on your priorities.
    weight_pall = 1.0
    weight_p0 = 0.5
    weight_p999 = 1.10

    # We want to maximize the weighted sum; since the optimizer minimizes,
    # we return the negative of the weighted sum.
    score = weight_pall * N_Pall + weight_p0 * N_P0 + weight_p999 * N_P999
    return -score


def PPP_centers(
    _tb_tgt, nPPC, weight_para=[1.5, 0, 0], randomseed=0, mutiPro=True, backup=False
):
    # determine pointing centers
    time_start = time.time()
    logger.info(f"[S2] Determine pointing centers started")

    ppc_lst = []

    if len(_tb_tgt) == 0:
        logger.warning(f"[S2] no targets")
        return np.array(ppc_lst), Table()

    _tb_tgt = sciRank_pri(_tb_tgt)

    single_exptime_ = _tb_tgt.meta["single_exptime"]

    _tb_tgt_ = _tb_tgt[_tb_tgt["exptime_PPP"] > 0]

    pslID_ = sorted(set(_tb_tgt_["proposal_id"]))

    FH_goal = [
        _tb_tgt_["allocated_time"][_tb_tgt_["proposal_id"] == tt][0] for tt in pslID_
    ]

    tb_fh = Table([pslID_, FH_goal], names=["proposal_id", "FH_goal"])
    tb_fh["FH_done"] = 0.0
    tb_fh["N_done"] = 0.0
    tb_fh["N_obs"] = 0.0
    tb_fh["N_psl"] = 0.0

    while (
        (
            sum((tb_fh["FH_done"] >= tb_fh["FH_goal"]) * (tb_fh["N_done"] > 0.0))
            < len(tb_fh)
        )
        and len(_tb_tgt_) > 0
        and len(ppc_lst) < nPPC
    ):
        psl_id_undone = list(
            set(
                tb_fh["proposal_id"][
                    (tb_fh["FH_done"] < tb_fh["FH_goal"]) | (tb_fh["N_done"] == 0.0)
                ]
            )
        )
        print(f"The non-complete proposals: {psl_id_undone}")

        _tb_tgt_ = _tb_tgt_[
            (_tb_tgt_["exptime_PPP"] > 0)
            * np.in1d(_tb_tgt_["proposal_id"], psl_id_undone)
        ]  # targets not finished
        _tb_tgt_["priority"][_tb_tgt_["exptime_done"] > 0] = 999

        tb_tgt_t_group = target_DBSCAN(_tb_tgt_, 1.38)

        _tb_tgt_t_ = tb_tgt_t_group[0]

        """
        _df_tgt_t = Table.to_pandas(_tb_tgt_t_)
        n_tgt = min(200, len(_tb_tgt_t_))
        _df_tgt_t = _df_tgt_t.sample(n_tgt, ignore_index=True, random_state=1)
        _tb_tgt_t_1 = Table.from_pandas(_df_tgt_t)  
        #"""

        initial_guess = [_tb_tgt_t_["ra"][0], _tb_tgt_t_["dec"][0]]
        result = minimize(
            objective1,
            initial_guess,
            args=(_tb_tgt_t_,),
            method="Nelder-Mead",
            options={"xatol": 0.01, "fatol": 0.001},
        )
        print(result.x)
        ppc_ra_, ppc_dec_ = result.x[0], result.x[1]
        ppc_pa_ = 0.0

        index_ = PFS_FoV(ppc_ra_, ppc_dec_, ppc_pa_, _tb_tgt_)

        lst_tgtID_assign = netflowRun4PPC(
            _tb_tgt_[list(index_)], ppc_ra_, ppc_dec_, ppc_pa_
        )

        iter_tem = 0
        while len(lst_tgtID_assign) == 0 and iter_tem < 2:
            ppc_ra_ += np.random.uniform(-0.15, 0.15, 1)[0]
            ppc_dec_ += np.random.uniform(-0.15, 0.15, 1)[0]
            lst_tgtID_assign = netflowRun4PPC(
                _tb_tgt_[list(index_)],
                ppc_ra_,
                ppc_dec_,
                ppc_pa_,
                otime="2025-04-10T08:00:00Z",
            )
            iter_tem += 1

        index_assign = np.in1d(_tb_tgt_["identify_code"], lst_tgtID_assign)
        weight_tem_tot = sum(_tb_tgt_["rank_fin"][index_assign])

        lst_pslID_assign = [ii.split("_")[0] for ii in lst_tgtID_assign]
        pslID_ = sorted(set(lst_pslID_assign))
        pslID_n = {
            tt: lst_pslID_assign.count(tt) * single_exptime_ / 3600.0 for tt in pslID_
        }

        print(f"{ppc_ra_}, {ppc_dec_}, {len(lst_pslID_assign)}")

        ppc_lst.append(
            np.array(
                [
                    len(ppc_lst),
                    ppc_ra_,
                    ppc_dec_,
                    ppc_pa_,
                    weight_tem_tot,
                    sum(pslID_n.values()) / sum(FH_goal),
                    len(lst_pslID_assign) / 2394.0,
                    lst_tgtID_assign,
                ],
                dtype=object,
            ),
        )
        # ppc_id, ppc_ra, ppc_dec, ppc_pa, ppc_weight, ppc_fh, ppc_FE

        _tb_tgt_["exptime_PPP"][
            index_assign
        ] -= single_exptime_  # targets in the PPC observed with single_exptime sec

        _tb_tgt_["exptime_done"][index_assign] += single_exptime_

        _tb_tgt_["priority"][_tb_tgt_["exptime_done"] > 0] = 999

        for tt in list(pslID_n.keys()):
            tb_fh["N_psl"].data[tb_fh["proposal_id"] == tt] = sum(
                _tb_tgt["proposal_id"] == tt
            )
            tb_fh["FH_done"].data[tb_fh["proposal_id"] == tt] += pslID_n[tt]
            tb_fh["N_done"].data[tb_fh["proposal_id"] == tt] += sum(
                (_tb_tgt_["exptime_PPP"] <= 0) * (_tb_tgt_["proposal_id"] == tt)
            )
            tb_fh["N_obs"].data[tb_fh["proposal_id"] == tt] = sum(
                (_tb_tgt_["exptime_PPP"] < _tb_tgt_["exptime"])
                * (_tb_tgt_["proposal_id"] == tt)
            )

        n_uncom1 = sum(_tb_tgt_["exptime_done"] > 0)
        _tb_tgt_ = _tb_tgt_[_tb_tgt_["exptime_PPP"] > 0]  # targets not finished
        n_uncom2 = sum(_tb_tgt_["exptime_done"] > 0)

        print(
            f"PPC_{len(ppc_lst):3d}: {len(_tb_tgt)-len(_tb_tgt_):5d}/{len(_tb_tgt):10d} targets are finished (w={weight_tem_tot:.2f}). (partial = {n_uncom1}, {n_uncom2})"
        )
        Table.pprint_all(tb_fh)

    if len(ppc_lst) > nPPC:
        ppc_lst_fin = sorted(ppc_lst, key=lambda x: x[4], reverse=True)[:nPPC]

    else:
        ppc_lst_fin = ppc_lst[:]

    ppc_lst_fin = np.array(ppc_lst_fin)
    epsilon = 1e-3  # small number to avoid divide-by-zero
    col = np.where(ppc_lst_fin[:, 4] == 0, epsilon, ppc_lst_fin[:, 4])
    recip = 1 / col
    weight_for_qplan = (recip / recip.max()) * 1000.0

    # write
    nPPC = len(ppc_lst_fin)
    resol = _tb_tgt["resolution"][0]
    if backup:
        ppc_code = [
            f"que_{resol}_{datetime.now().strftime('%y%m%d')}_{int(nn + 1)}_backup"
            for nn in range(nPPC)
        ]
    else:
        ppc_code = [
            f"que_{resol}_{datetime.now().strftime('%y%m%d')}_{int(nn + 1)}"
            for nn in range(nPPC)
        ]
    ppc_ra = ppc_lst_fin[:, 1]
    ppc_dec = ppc_lst_fin[:, 2]
    ppc_pa = ppc_lst_fin[:, 3]
    ppc_equinox = ["J2000"] * nPPC
    ppc_priority = weight_for_qplan
    ppc_priority_usr = weight_for_qplan
    ppc_exptime = [900.0] * nPPC
    ppc_totaltime = [1200.0] * nPPC
    ppc_resolution = [resol] * nPPC
    ppc_fibAlloFrac = ppc_lst_fin[:, -2]
    ppc_tgtAllo = ppc_lst_fin[:, -1]
    ppc_comment = [""] * nPPC

    ppcList = Table(
        [
            ppc_code,
            ppc_ra,
            ppc_dec,
            ppc_pa,
            ppc_equinox,
            ppc_priority,
            ppc_priority_usr,
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
            "ppc_priority_usr",
            "ppc_exptime",
            "ppc_totaltime",
            "ppc_resolution",
            "ppc_fiber_usage_frac",
            "ppc_allocated_targets",
            "ppc_comment",
        ],
        dtype=[
            np.str_,
            np.float64,
            np.float64,
            np.float64,
            np.str_,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.str_,
            np.float64,
            object,
            np.str_,
        ],
    )

    logger.info(
        f"[S2] Determine pointing centers done ( nppc = {len(ppc_lst_fin):.0f}; takes {round(time.time()-time_start,3)} sec)"
    )

    return ppc_lst_fin, ppcList


def ppc_DBSCAN(_tb_tgt):
    # separate pointings into different group (skip due to FH upper limit -24-02-07; NEED TO FIX)
    ppc_xy = _tb_tgt.meta["PPC"]
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

    return np.array([ppc_xy])


def sam2netflow(_tb_tgt, for_ppc=False):
    # put targets to the format which can be read by netflow
    tgt_lst_netflow = []
    _tgt_lst_psl_id = []
    single_exptime_ = _tb_tgt.meta["single_exptime"]

    int_ = 0
    for tt in _tb_tgt:
        if for_ppc:
            # set exptime = single_exptime_ if running netflow to determine PPC
            tgt_id_, tgt_ra_, tgt_dec_, tgt_exptime_, tgt_proposal_id_, tgt_pri_ = (
                tt["identify_code"],
                tt["ra"],
                tt["dec"],
                single_exptime_,
                tt["proposal_id"],
                int(tt["priority"]),
            )
        else:
            tgt_id_, tgt_ra_, tgt_dec_, tgt_exptime_, tgt_proposal_id_, tgt_pri_ = (
                tt["identify_code"],
                tt["ra"],
                tt["dec"],
                tt["exptime_PPP"],
                tt["proposal_id"],
                int(tt["priority"]),
            )
        """  
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
        #"""

        tgt_lst_netflow.append(
            nf.ScienceTarget(
                tgt_id_,
                tgt_ra_,
                tgt_dec_,
                tgt_exptime_,
                tgt_pri_,
                "sci",
            )
        )
        _tgt_lst_psl_id.append("sci_P" + str(int(tgt_pri_)))

    # set FH limit bundle
    tgt_psl_FH_tac_ = {}

    #'''
    if for_ppc == False:
        psl_id = sorted(set(_tb_tgt["proposal_id"]))

        for psl_id_ in psl_id:
            tt_ = tuple([tt for tt in _tgt_lst_psl_id if psl_id_ in tt])
            fh_ = _tb_tgt[_tb_tgt["proposal_id"] == psl_id_]["allocated_time"][0]
            tgt_psl_FH_tac_[tt_] = fh_

            print(f"{psl_id_}: FH_limit = {tgt_psl_FH_tac_[tt_]:.2f}")
    #'''

    return tgt_lst_netflow, tgt_psl_FH_tac_


def NetflowPreparation(_tb_tgt):
    """
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
    #"""
    classdict = {
        # Priorities correspond to the magnitudes of bright stars (in most case for the 2022 June Engineering)
        "sci_P999": {
            "nonObservationCost": 200,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P0": {
            "nonObservationCost": 100,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P1": {
            "nonObservationCost": 90,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P2": {
            "nonObservationCost": 80,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P3": {
            "nonObservationCost": 70,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P4": {
            "nonObservationCost": 60,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P5": {
            "nonObservationCost": 50,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P6": {
            "nonObservationCost": 40,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P7": {
            "nonObservationCost": 30,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P8": {
            "nonObservationCost": 20,
            "partialObservationCost": 200,
            "calib": False,
        },
        "sci_P9": {
            "nonObservationCost": 10,
            "partialObservationCost": 200,
            "calib": False,
        },
        "cal": {
            "numRequired": 200,
            "nonObservationCost": 200,
            "calib": True,
        },
        "sky": {
            "numRequired": 400,
            "nonObservationCost": 200,
            "calib": True,
        },
    }

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
    otime="2025-03-22T08:00:00Z",
    for_ppc=False,
):
    # run netflow (without iteration)
    Telra = ppc_lst[:, 1]
    Teldec = ppc_lst[:, 2]
    Telpa = ppc_lst[:, 3]
    Telweight = ppc_lst[:, 4]

    tgt_lst_netflow, tgt_psl_FH_tac = sam2netflow(_tb_tgt, for_ppc)
    classdict = NetflowPreparation(_tb_tgt)

    telescopes = []

    nvisit = len(Telra)
    for ii in range(nvisit):
        telescopes.append(nf.Telescope(Telra[ii], Teldec[ii], Telpa[ii], otime))
    tpos = [tele.get_fp_positions(tgt_lst_netflow) for tele in telescopes]

    single_exptime_ = _tb_tgt.meta["single_exptime"]

    # optional: slightly increase the cost for later observations,
    # to observe as early as possible
    vis_cost = [0] * nvisit
    # positions = {e: i for i, e in enumerate(sorted(Telweight, reverse = True), 0)}
    # vis_cost = [positions[e] for e in Telweight]
    # vis_cost_upper = min(_tb_tgt["weight"])*0.3
    # if len(vis_cost)>1:
    #    vis_cost = np.array(vis_cost)/max(vis_cost)*vis_cost_upper
    # else:
    #    vis_cost = [0]*nvisit
    # print(vis_cost_upper,vis_cost)

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
                single_exptime_,
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

            status = prob._prob.status  # or prob.getStatus() / prob.solverStatus, etc.
            print("Model status:", status)

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
                    len(bench.cobras.centers), TargetGroup.NULL_TARGET_POSITION
                )
                ids = np.full(len(bench.cobras.centers), TargetGroup.NULL_TARGET_ID)
                for tidx, cidx in vis.items():
                    selectedTargets[cidx] = tp[tidx]
                    ids[cidx] = ""
                for i in range(selectedTargets.size):
                    if selectedTargets[i] != TargetGroup.NULL_TARGET_POSITION:
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
            single_exptime_,
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

        # status = prob._prob.status  # or prob.getStatus() / prob.solverStatus, etc.
        # print("Model status:", status)

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
    otime="2025-04-20T08:00:00Z",
):
    # run netflow (with iteration)
    #    if no fiber assignment in some PPCs, shift these PPCs with 0.15 deg
    # (skip due to FH upper limit -24-02-07; NEED TO FIX)

    res, telescope, tgt_lst_netflow = netflowRun_single(
        ppc_lst,
        _tb_tgt,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
        otime,
        for_ppc,
    )
    return res, telescope, tgt_lst_netflow


def netflowRun(
    _tb_tgt,
    randomseed=0,
    TraCollision=False,
    numReservedFibers=0,
    fiberNonAllocationCost=0.0,
    backup=False,
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

            logger.info(
                f"PPC {i:4d}: {len(vis):.0f}/2394={ppc_fib_eff:.2f}% assigned Cobras"
            )

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
                        "rank_fin"
                    ]
                )

            if backup:
                ppc_code_ = f"que_{_tb_tgt['resolution'][0]}_{datetime.now().strftime('%y%m%d')}_{int(i + 1)}_backup"
            else:
                ppc_code_ = f"que_{_tb_tgt['resolution'][0]}_{datetime.now().strftime('%y%m%d')}_{int(i + 1)}"

            ppc_lst.append(
                [
                    ppc_code_,
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

    if _tb_tgt.meta["PPC_origin"] == "usr":
        lst_ppc_usr = _tb_tgt.meta["PPC"]
        tb_ppc_usr = Table(
            lst_ppc_usr[:, 1:],
            names=["ppc_ra", "ppc_dec", "ppc_pa", "ppc_priority_usr"],
        )
        df_ppc_usr = Table.to_pandas(tb_ppc_usr)
        df_ppc_usr = df_ppc_usr.drop_duplicates(
            subset=["ppc_ra", "ppc_dec", "ppc_pa"],
            inplace=False,
            ignore_index=True,
        )
        tb_ppc_usr = Table.from_pandas(df_ppc_usr)
        tb_ppc_netflow = join(
            tb_ppc_netflow, tb_ppc_usr, keys=["ppc_ra", "ppc_dec", "ppc_pa"]
        )
    else:
        tb_ppc_netflow["ppc_priority_usr"] = tb_ppc_netflow["ppc_priority"]

    logger.info(
        f"[S3] Run netflow done (takes {round(time.time() - time_start, 3)} sec)"
    )

    return tb_ppc_netflow


def netflowRun4PPC(
    _tb_tgt_inuse,
    ppc_x,
    ppc_y,
    ppc_pa,
    otime="2025-05-20T08:00:00Z",
):
    # run netflow (for PPP_centers)
    ppc_lst = np.array([[0, ppc_x, ppc_y, ppc_pa, 0]])

    res, telescope, tgt_lst_netflow = netflowRun_nofibAssign(
        ppc_lst,
        _tb_tgt_inuse,
        True,
        otime=otime,
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
        _tb_tgt["exptime_assign"].data[lst] += int(_tb_tgt.meta["single_exptime"])

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
    mode="queue",
):
    # iterate the total procedure to re-assign fibers to targets which have not been assigned in the previous/first iteration
    # note that some targets in the dense region may need very long time to be assigned with fibers
    # if targets can not be successfully assigned with fibers in >10 iterations, then directly stop
    # if total number of ppc > nPPC, then directly stop

    if len(_tb_tgt) == 0 or len(_tb_ppc_netflow) == 0:
        return _tb_ppc_netflow

    if _tb_tgt.meta["PPC_origin"] == "usr":
        return _tb_ppc_netflow

    # """
    FH_goal = _tb_tgt["allocated_time"].data[0]
    FH_done = sum(_tb_tgt["exptime_assign"]) / 3600.0
    print(f"FH_goal = {FH_goal}, FH_done = {FH_done}")
    while FH_done > FH_goal and mode == "queue":  # reduce Nppc only for queue modes
        ppc_lst = [
            [0, tt["ppc_ra"], tt["ppc_dec"], tt["ppc_pa"], tt["ppc_fiber_usage_frac"]]
            for tt in _tb_ppc_netflow
        ]  # ppc_id, ppc_ra, ppc_dec, ppc_pa, ppc_FE

        ppc_lst_new = sorted(ppc_lst, key=lambda x: x[-1])[1:]

        _tb_tgt.meta["PPC"] = ppc_lst_new

        _tb_ppc_netflow_ = netflowRun(
            _tb_tgt,
            randomseed,
            TraCollision,
            numReservedFibers,
            fiberNonAllocationCost,
        )

        _tb_tgt = netflowAssign(_tb_tgt, _tb_ppc_netflow_)

        FH_done = sum(_tb_tgt["exptime_assign"]) / 3600.0

        print(FH_done, np.mean(_tb_ppc_netflow_["ppc_fiber_usage_frac"]))

        if FH_done > FH_goal * 0.95:
            _tb_ppc_netflow = _tb_ppc_netflow_
    # """

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
    return _tb_ppc_netflow


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
            weight_allo = sum(_tb_tgt[index_allo]["rank_fin"])

        # patrly observed
        index_part = np.where(
            (_tb_tgt["exptime_PPP"] > _tb_tgt["exptime_assign"])
            & (_tb_tgt["exptime_assign"] > 0)
        )[0]

        if len(index_part) > 0:
            weight_allo += 0.5 * sum(_tb_tgt[index_part]["rank_fin"])

        weight_tot = sum(_tb_tgt["rank_fin"])

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


def PPC_efficiency(tb_ppc_netflow):
    # calculate fiber allocation efficiency

    fib_eff = tb_ppc_netflow["ppc_fiber_usage_frac"].data  # unit --> %

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
    # para_sci, para_exp, para_n = para

    _tb_tgt = info["tb_tgt"]

    nppc_ = info["nPPC"]

    index_op1 = info["iter"]
    randomseed = info["randomSeed"]

    TraCollision = info["checkTraCollision"]

    completeMode = info["crMode"]

    # --------------------
    tem1 = 0

    mfibEff1 = 0
    CR_fin1 = 0

    n_exptime = len(set(_tb_tgt["exptime"]))
    n_rank = len(set(_tb_tgt["priority"]))
    if n_rank <= 1 and n_exptime > 1:
        para_exp, para_n = para
        para_sci = 1.5
        lst_ppc = PPP_centers(
            _tb_tgt, nppc_, [para_sci, para_exp, para_n], randomseed, True
        )[0]
    if n_rank > 1 and n_exptime <= 1:
        para_sci, para_n = para
        para_exp = 0
        lst_ppc = PPP_centers(
            _tb_tgt, nppc_, [para_sci, para_exp, para_n], randomseed, True
        )[0]
    if n_rank > 1 and n_exptime > 1:
        para_sci, para_exp, para_n = para
        lst_ppc = PPP_centers(
            _tb_tgt, nppc_, [para_sci, para_exp, para_n], randomseed, True
        )[0]
    if n_rank <= 1 and n_exptime <= 1:
        para_n = para[0]
        para_sci = 1.5
        para_exp = 0
        lst_ppc = PPP_centers(
            _tb_tgt, nppc_, [para_sci, para_exp, para_n], randomseed, True
        )[0]
        # ppc_id, ppc_ra, ppc_dec, ppc_pa, ppc_weight, ppc_fh, ppc_FE

    tem1 = (
        len(lst_ppc) / nppc_ + 1.5 - sum(lst_ppc[:, -2]) + 1 - np.mean(lst_ppc[:, -1])
    )
    print(len(lst_ppc) / nppc_, 1.5 - sum(lst_ppc[:, -2]), 1 - np.mean(lst_ppc[:, -1]))

    logger.info(
        f"[S4] Iter {info['iter']+1:.0f}, w_para is [{para_sci:.3f}, {para_exp:.3f}, {para_n:.3f}]; objV is {tem1:.2f}."
    )

    info["iter"] += 1

    return tem1


def iter_weight(_tb_tgt, weight_initialGuess, nppc_, crMode, randomSeed, TraCollision):
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

    best_weight = opt.least_squares(
        fun2opt,
        weight_initialGuess,
        xtol=0.05,
        ftol=0.005,
        args=(
            {
                "tb_tgt": _tb_tgt,
                "nPPC": nppc_,
                "crMode": crMode,
                "iter": 0,
                "randomSeed": randomSeed,
                "checkTraCollision": TraCollision,
            },
        ),
        diff_step=4,
        gtol=0.005,
        max_nfev=50,
        # disp=True,
        # retall=False,
        # full_output=False,
        # maxiter=200,
        # maxfun=200,
    )

    logger.info(f"[S4] Optimization done (takes {time.time()-time_s:.3f} sec)")

    return best_weight["x"]


def optimize(_tb_tgt, nppc_, crMode, randomSeed, TraCollision):
    n_exptime = len(set(_tb_tgt["exptime"]))
    n_rank = len(set(_tb_tgt["priority"]))
    if n_rank <= 1 and n_exptime > 1:
        weight_guess = [0.1, 0.1]
        para_sci = 1.5
        para_exp, para_n = iter_weight(
            _tb_tgt,
            weight_guess,
            nppc_,
            crMode,
            randomSeed,
            TraCollision,
            # numReservedFibers,
            # fiberNonAllocationCost,
        )
    if n_rank > 1 and n_exptime <= 1:
        weight_guess = [1.5, 0.1]
        para_exp = 0
        para_sci, para_n = iter_weight(
            _tb_tgt,
            weight_guess,
            nppc_,
            crMode,
            randomSeed,
            TraCollision,
            # numReservedFibers,
            # fiberNonAllocationCost,
        )
    if n_rank > 1 and n_exptime > 1:
        weight_guess = [1.5, 0.1, 0.1]
        para_sci, para_exp, para_n = iter_weight(
            _tb_tgt,
            weight_guess,
            nppc_,
            crMode,
            randomSeed,
            TraCollision,
            # numReservedFibers,
            # fiberNonAllocationCost,
        )
    if n_rank <= 1 and n_exptime <= 1:
        weight_guess = [-0.1]
        para_sci = 1.5
        para_exp = 0
        para_n = iter_weight(
            _tb_tgt,
            weight_guess,
            nppc_,
            crMode,
            randomSeed,
            TraCollision,
            # numReservedFibers,
            # fiberNonAllocationCost,
        )
    return para_sci, para_exp, para_n


def output(_tb_ppc_tot, _tb_tgt_tot, dirName="output/", backup=False):
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
    ppc_priority_usr = _tb_ppc_tot["ppc_priority_usr"].data
    ppc_exptime = [_tb_tgt_tot.meta["single_exptime"]] * len(_tb_ppc_tot)
    ppc_totaltime = [_tb_tgt_tot.meta["single_exptime"] + 300] * len(_tb_ppc_tot)
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
            ppc_priority_usr,
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
            "ppc_priority_usr",
            "ppc_exptime",
            "ppc_totaltime",
            "ppc_resolution",
            "ppc_fiber_usage_frac",
            "ppc_allocated_targets",
            "ppc_comment",
        ],
    )

    # ppcList.write(
    #    os.path.join(dirName, "ppcList.ecsv"), format="ascii.ecsv", overwrite=True
    # )

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
    ob_exptime_usr = _tb_tgt_tot["exptime_usr"].data
    ob_single_exptime = _tb_tgt_tot["single_exptime"].data
    ob_resolution = _tb_tgt_tot["resolution"].data
    proposal_id = _tb_tgt_tot["proposal_id"].data
    proposal_rank = _tb_tgt_tot["rank"].data
    proposal_FH = _tb_tgt_tot["allocated_time_tac"].data
    ob_weight_best = _tb_tgt_tot["rank_fin"].data
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
    ob_total_flux_g = _tb_tgt_tot["total_flux_g"].data
    ob_total_flux_r = _tb_tgt_tot["total_flux_r"].data
    ob_total_flux_i = _tb_tgt_tot["total_flux_i"].data
    ob_total_flux_z = _tb_tgt_tot["total_flux_z"].data
    ob_total_flux_y = _tb_tgt_tot["total_flux_y"].data
    ob_total_flux_error_g = _tb_tgt_tot["total_flux_error_g"].data
    ob_total_flux_error_r = _tb_tgt_tot["total_flux_error_r"].data
    ob_total_flux_error_i = _tb_tgt_tot["total_flux_error_i"].data
    ob_total_flux_error_z = _tb_tgt_tot["total_flux_error_z"].data
    ob_total_flux_error_y = _tb_tgt_tot["total_flux_error_y"].data
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
            ob_exptime_usr,
            ob_single_exptime,
            ob_resolution,
            proposal_id,
            proposal_rank,
            proposal_FH,
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
            ob_total_flux_g,
            ob_total_flux_r,
            ob_total_flux_i,
            ob_total_flux_z,
            ob_total_flux_y,
            ob_total_flux_error_g,
            ob_total_flux_error_r,
            ob_total_flux_error_i,
            ob_total_flux_error_z,
            ob_total_flux_error_y,
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
            "ob_exptime_usr",
            "ob_single_exptime",
            "ob_resolution",
            "proposal_id",
            "proposal_rank",
            "allocated_time_tac",
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
            "ob_total_flux_g",
            "ob_total_flux_r",
            "ob_total_flux_i",
            "ob_total_flux_z",
            "ob_total_flux_y",
            "ob_total_flux_error_g",
            "ob_total_flux_error_r",
            "ob_total_flux_error_i",
            "ob_total_flux_error_z",
            "ob_total_flux_error_y",
            "ob_identify_code",
        ],
    )

    if not backup:
        obList.write(
            os.path.join(dirName, "obList.ecsv"), format="ascii.ecsv", overwrite=True
        )

        np.save(os.path.join(dirName, "obj_allo_tot.npy"), _tb_ppc_tot)
    else:
        obList.write(
            os.path.join(dirName, "obList_backup.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )

        np.save(os.path.join(dirName, "obj_allo_tot_backup.npy"), _tb_ppc_tot)


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
    nppc_l,
    nppc_m,
    dirName="output/",
    numReservedFibers=0,
    fiberNonAllocationCost=0.0,
    show_plots=False,
    backup=False,
    conf=None,
):
    global bench
    bench = bench_info

    today = date.today().strftime("%Y%m%d")
    tb_queuedb_filename = os.path.join(dirName, f"tgt_queueDB_{today}.csv")
    psl_id = conf["ppp"]["proposalIds"] + conf["ppp"]["proposalIds_backup"]
    tb_queuedb = queryQueue(psl_id, conf["queuedb"]["filepath"], tb_queuedb_filename)

    tb_tgt, tb_tgt_l, tb_tgt_m, tb_queuedb, tb_queuedb = readTarget(
        readtgt_con["mode_readtgt"], readtgt_con["para_readtgt"], tb_queuedb
    )

    randomseed = 2

    TraCollision = False
    multiProcess = True

    # """
    # LR--------------------------------------------
    # ppc_lst_l = PPP_centers(
    #    tb_tgt_l, nppc_l, [para_sci_l, para_exp_l, para_n_l], randomseed, multiProcess
    # )

    ppc_lst_l, tb_ppcList_l = PPP_centers(tb_tgt_l, nppc_l, backup=backup)

    tb_tgt_l1 = Table.copy(tb_tgt_l)
    tb_tgt_l1.meta["PPC"] = ppc_lst_l

    tb_tgt_l1 = sciRank_pri(tb_tgt_l1)

    tb_ppc_l = netflowRun(
        tb_tgt_l1,
        randomseed,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
        backup=backup,
    )

    tb_tgt_l_fin = netflowAssign(tb_tgt_l1, tb_ppc_l)

    # MR--------------------------------------------
    # ppc_lst_m = PPP_centers(
    #    tb_tgt_m, nppc_m, [para_sci_m, para_exp_m, para_n_m], randomseed, multiProcess
    # )
    ppc_lst_m, tb_ppcList_m = PPP_centers(tb_tgt_m, nppc_m, backup=backup)

    tb_tgt_m1 = Table.copy(tb_tgt_m)
    tb_tgt_m1.meta["PPC"] = ppc_lst_m

    tb_tgt_m1 = sciRank_pri(tb_tgt_m1)

    tb_ppc_m = netflowRun(
        tb_tgt_m1,
        randomseed,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
        backup=backup,
    )

    tb_tgt_m_fin = netflowAssign(tb_tgt_m1, tb_ppc_m)

    if nppc_l > 0:
        if nppc_m > 0:
            tb_ppc_tot = vstack([tb_ppcList_l, tb_ppcList_m])
            tb_tgt_tot = vstack([tb_tgt_l_fin, tb_tgt_m_fin])
        else:
            tb_ppc_tot = tb_ppcList_l.copy()
            tb_tgt_tot = tb_tgt_l_fin.copy()
            if len(tb_tgt_m) > 0:
                logger.warning("no allocated time for MR")
    else:
        if nppc_m > 0:
            tb_ppc_tot = tb_ppcList_m.copy()
            tb_tgt_tot = tb_tgt_m_fin.copy()
            if len(tb_tgt_l) > 0:
                logger.warning("no allocated time for LR")
        else:
            logger.error("Please specify n_pcc_l or n_pcc_m")
    # """

    if not backup:
        tb_ppc_tot.write(
            os.path.join(dirName, f"ppcList.ecsv"), format="ascii.ecsv", overwrite=True
        )
    else:
        tb_ppc_tot.write(
            os.path.join(dirName, f"ppcList_backup.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )

    output(tb_ppc_tot, tb_tgt_tot, dirName=dirName, backup=backup)
