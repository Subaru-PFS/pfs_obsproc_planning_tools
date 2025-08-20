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
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KernelDensity
from qplan import q_db, q_query, entity
from qplan.util.site import site_subaru as observer
from ginga.misc.log import get_logger

logger_qplan = get_logger("qplan_test", null=True)

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


def visibility_checker(tb_tgt, obstimes):
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

            obs_ok, t_start, t_stop = observer.observable(
                target,
                default_start_time,
                default_stop_time,
                min_el,
                max_el,
                total_time,
                airmass=None,
                moon_sep=None,
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


def readTarget(mode, para):
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

            if proposalId == "S25A-UH022-A":
                sql = sql[:-1] + " AND c.input_catalog_id = 10154;"
    
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
            df_tgt["resolution"] = ["M" if v == True else "L" for v in df_tgt["resolution"]]
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

            cols = ["psf_flux_g", "psf_flux_r", "psf_flux_i", "psf_flux_z", "psf_flux_y",
                    "psf_flux_error_g", "psf_flux_error_r", "psf_flux_error_i", "psf_flux_error_z", "psf_flux_error_y",
                    "total_flux_g", "total_flux_r", "total_flux_i", "total_flux_z", "total_flux_y",
                    "total_flux_error_g", "total_flux_error_r", "total_flux_error_i", "total_flux_error_z", "total_flux_error_y"]
            for col in cols:
                # Convert the column to numeric; non-convertible values (e.g., "N/A") become np.nan.
                df_tgt[col] = pd.to_numeric(df_tgt[col], errors='coerce')
    
            tb_tgt = Table.from_pandas(df_tgt)

            for col in ["filter_g","filter_r", "filter_i", "filter_z", "filter_y"]:
                tb_tgt[col] = tb_tgt[col].astype("str")
    
            conn.close()

            return tb_tgt

    # only for S25A march run
    proposalid = ['S25A-UH022-A']

    tb_tgt_lst = []
    for proposalid_ in proposalid:
        tb_tgt_lst.append(query_target_from_db(proposalid_))
    tb_tgt=vstack(tb_tgt_lst)

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

    #tb_tgt.write("/home/wanqqq/workDir_pfs/run_2505/S25A-UH041-A/output/ppp/obList.ecsv", format="ascii.ecsv", overwrite=True) 

    if len(set(tb_tgt["single_exptime"])) > 1:
        logger.error(
            "[S1] Multiple single-exptime are given. Not accepted now (240709)."
        )
        return Table(), Table(), Table(), Table(), Table()

    # fix needed
    tb_tgt["single_exptime"][tb_tgt["proposal_id"] == 'S25A-UH022-A'] = 3000.0
    tb_tgt["exptime_usr"][(tb_tgt["proposal_id"] == 'S25A-UH022-A') * (tb_tgt["priority"] == 0)] = 12000.0

    tb_tgt.meta["single_exptime"] = list(set(tb_tgt["single_exptime"]))[0]
    logger.info(
        f"[S1] The single exptime is set to {tb_tgt.meta['single_exptime']:.2f} sec."
    )

    tb_tgt.meta["PPC"] = np.array([])
    tb_tgt.meta["PPC_origin"] = "auto"

    if para["visibility_check"]:
        tb_tgt = visibility_checker(tb_tgt, para["obstimes"])

    if mode == "queue":
        # FIX!! just for this test
        tb_tgt["allocated_time_tac"][tb_tgt["proposal_id"] == "S24B-QT917"] = 2000.0
        tb_tgt["allocated_time_tac"][tb_tgt["proposal_id"] == "S24B-QT915"] = 750.0

        tb_tgt["allocated_time_done"] = 0
        tb_tgt["allocated_time"] = 0

        # list of all psl_id
        psl_id = sorted(set(tb_tgt["proposal_id"]))

        # connect with queueDB
        qdb = q_db.QueueDatabase(logger_qplan)
        qdb.read_config(para["DBPath_qDB"])
        qdb.connect()
        qa = q_db.QueueAdapter(qdb)
        qq = q_query.QueueQuery(qa, use_cache=False)

        # determine observed exptime
        nn = 0
        tt = []
        for psl_id_ in psl_id:
            # if psl_id_ not in ['S24B-QT908','S24B-QT910','S24B-920']:continue
            # print(psl_id_)
            ex_obs = qq.get_executed_obs_by_proposal(psl_id_)
            # if len(ex_obs)==0:
            #    continue
            for ex_ob in ex_obs:
                exps = qq.get_exposures(ex_ob)
                exptime_exe = sum(exp.effective_exptime or 0 for exp in exps)
                exptime_usr = tb_tgt[
                    (tb_tgt["proposal_id"] == ex_ob.ob_key[0])
                    * (tb_tgt["ob_code"] == ex_ob.ob_key[1])
                ]["exptime_usr"].data[0]
                exptime_exe_fin = min(
                    exptime_usr, exptime_exe
                )  # ignore over-observation
                # if exptime_exe>=0:
                #    nn+=1
                #    tt.append([nn, psl_id_, ex_ob.ob_key[1], exptime_usr, exptime_exe, len(exps)*900])
                tb_tgt["exptime_done"][
                    (tb_tgt["proposal_id"] == ex_ob.ob_key[0])
                    * (tb_tgt["ob_code"] == ex_ob.ob_key[1])
                ] = exptime_exe_fin
        # tt_=Table(np.array(tt),names=["N","psl_id","ob_code","exptime","exptime_done","exptime_done_real"])
        # tt_.write("/home/wanqqq/examples/tgt_queueDB.csv",overwrite=True)

        tb_tgt["exptime"] = tb_tgt["exptime_usr"] - tb_tgt["exptime_done"]
        # print(len(tt_),len(tb_tgt[tb_tgt["exptime"]==0]))

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

        # separete the sample by 'resolution' (L/M)
        tb_tgt_l = tb_tgt[tb_tgt["resolution"] == "L"]
        tb_tgt_m = tb_tgt[tb_tgt["resolution"] == "M"]

        # select targets based on the allocated FH (for determining PPC)
        _tgt_select_l = []
        _tgt_select_m = []

        for psl_id_ in psl_id:
            # if psl_id_ != "S24B-920":continue
            tb_tgt_tem_l = tb_tgt[
                (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "L")
            ]
            tb_tgt_tem_m = tb_tgt[
                (tb_tgt["proposal_id"] == psl_id_) & (tb_tgt["resolution"] == "M")
            ]

            if len(tb_tgt_tem_l) > 0:
                _tgt_select_l.append(tb_tgt_tem_l[tb_tgt_tem_l["exptime_done"] > 0])
                FH_select = sum(
                    tb_tgt_tem_l["exptime_usr"][tb_tgt_tem_l["exptime_done"] > 0]
                )
                FH_tac = tb_tgt_tem_l["allocated_time_tac"][0] * 3600.0
                print(
                    "1L",
                    psl_id_,
                    FH_select / 3600.0,
                    FH_tac / 3600.0,
                    len(_tgt_select_l[-1]),
                )

                #'''
                tb_tgt_tem_l = tb_tgt_tem_l[tb_tgt_tem_l["exptime_done"] == 0]

                size_step = max(int(0.02 * len(tb_tgt_tem_l)), 100)
                size_ = max(int(0.02 * len(tb_tgt_tem_l)), 100)

                while FH_select < FH_tac:
                    tb_tgt_tem_l = count_N_overlap(tb_tgt_tem_l, tb_tgt_l)
                    pri_ = (
                        (10 - tb_tgt_tem_l["priority"])
                        + 2
                        * (
                            tb_tgt_tem_l["local_count"]
                            / max(tb_tgt_tem_l["local_count"])
                        )
                        + 2
                        * (1 - tb_tgt_tem_l["exptime"] / max(tb_tgt_tem_l["exptime"]))
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
                    _tgt_select_l.append(_tb_tem[index_t])
                    FH_select += sum(tb_tgt_tem_l["exptime_usr"][index_t])
                    tb_tgt_tem_l.remove_rows(index_t)

                    if len(tb_tgt_tem_l) == 0:
                        break

                    print(
                        "L",
                        psl_id_,
                        FH_select / 3600.0,
                        FH_tac / 3600.0,
                        len(_tgt_select_l[-1]),
                    )
                    #'''

            if len(tb_tgt_tem_m) > 0:
                _tgt_select_m.append(tb_tgt_tem_m[tb_tgt_tem_m["exptime_done"] > 0])
                FH_select = sum(
                    tb_tgt_tem_m["exptime_usr"][tb_tgt_tem_m["exptime_done"] > 0]
                )
                FH_tac = tb_tgt_tem_m["allocated_time_tac"][0] * 3600.0
                print(
                    "1M",
                    psl_id_,
                    FH_select / 3600.0,
                    FH_tac / 3600.0,
                    len(_tgt_select_m[-1]),
                )

                #'''
                # if psl_id_=="S24B-QT915":continue
                tb_tgt_tem_m = tb_tgt_tem_m[tb_tgt_tem_m["exptime_done"] == 0]

                size_step = max(int(0.02 * len(tb_tgt_tem_m)), 100)
                size_ = max(int(0.02 * len(tb_tgt_tem_m)), 100)

                while FH_select < FH_tac:
                    tb_tgt_tem_m = count_N_overlap(tb_tgt_tem_m, tb_tgt_m)
                    pri_ = (
                        (10 - tb_tgt_tem_m["priority"])
                        + 2
                        * (
                            tb_tgt_tem_m["local_count"]
                            / max(tb_tgt_tem_m["local_count"])
                        )
                        + 2
                        * (1 - tb_tgt_tem_m["exptime"] / max(tb_tgt_tem_m["exptime"]))
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
                    _tgt_select_m.append(_tb_tem[index_t])
                    FH_select += sum(tb_tgt_tem_m["exptime_usr"][index_t])
                    tb_tgt_tem_m.remove_rows(index_t)

                    if len(tb_tgt_tem_m) == 0:
                        break

                    print(
                        "M",
                        psl_id_,
                        FH_select / 3600.0,
                        FH_tac / 3600.0,
                        len(_tgt_select_m[-1]),
                    )
                    #'''

        if len(_tgt_select_l) > 0:
            tgt_select_l = vstack(_tgt_select_l)
        else:
            tgt_select_l = Table()

        if len(_tgt_select_m) > 0:
            tgt_select_m = vstack(_tgt_select_m)
        else:
            tgt_select_m = Table()

        # remove complete targets
        n_tgt1 = len(tb_tgt)
        tb_tgt = tb_tgt[tb_tgt["exptime"] > 0]
        tb_tgt["exptime_PPP"] = (
            np.ceil(tb_tgt["exptime"] / 900) * 900
        )  # exptime needs to be multiples of 900 so netflow can be successfully executed
        n_tgt2 = len(tb_tgt)
        logger.info(f"There are {n_tgt2:.0f} / {n_tgt1:.0f} targets not completed")

        if len(_tgt_select_l) > 0:
            tgt_select_l = tgt_select_l[tgt_select_l["exptime"] > 0]
            tb_tgt_l = tb_tgt_l[tb_tgt_l["exptime"] > 0]
            tgt_select_l["exptime_PPP"] = np.ceil(tgt_select_l["exptime"] / 900) * 900
            tb_tgt_l["exptime_PPP"] = np.ceil(tb_tgt_l["exptime"] / 900) * 900

        if len(_tgt_select_m) > 0:
            tgt_select_m = tgt_select_m[tgt_select_m["exptime"] > 0]
            tb_tgt_m = tb_tgt_m[tb_tgt_m["exptime"] > 0]
            tgt_select_m["exptime_PPP"] = np.ceil(tgt_select_m["exptime"] / 900) * 900
            tb_tgt_m["exptime_PPP"] = np.ceil(tb_tgt_m["exptime"] / 900) * 900

        logger.info(f"[S1] observation mode = {mode}")
        logger.info(
            f"[S1] Read targets done (takes {round(time.time()-time_start,3):.2f} sec)."
        )
        logger.info(f"[S1] There are {len(set(tb_tgt['proposal_id'])):.0f} proposals.")
        logger.info(
            f"[S1] n_tgt_low = {len(tb_tgt_l):.0f} ({len(tgt_select_l):.0f}), n_tgt_medium = {len(tb_tgt_m):.0f} ({len(tgt_select_m):.0f})"
        )

        # return tb_tgt, tgt_select_l, tgt_select_m, tb_tgt_l, tb_tgt_m
        return tb_tgt, tb_tgt_l, tb_tgt_m, tb_tgt_l, tb_tgt_m

    elif mode == "classic":
        if "resolution" not in tb_tgt.colnames:
            tb_tgt["resolution"] = [
                "M" if v == "True" else "L" for v in tb_tgt["is_medium_resolution"]
            ]
        if "exptime" not in tb_tgt.colnames:
            tb_tgt.rename_column("exptime_usr", "exptime")

        if "allocated_time" not in tb_tgt.colnames:
            tb_tgt.rename_column("allocated_time_tac", "allocated_time")

        if np.any(tb_tgt["allocated_time"]<0):
            tb_tgt["allocated_time"] = sum(tb_tgt["exptime"]/3600.0)

        single_exptime_ = tb_tgt.meta["single_exptime"]
        
        tb_tgt["exptime_PPP"] = (
            np.ceil(tb_tgt["exptime"] / single_exptime_) * single_exptime_
        )

        #only for s25a-uh006
        if list(set(tb_tgt["proposal_id"])) == ["S25A-UH006-B"]:
            tb_tgt_add1 = Table.read("/home/wanqqq/examples/run_2503/S25A-UH006-B/input/Sanders_Extra_PFS_Targets_2025A_rev.csv")
            tb_tgt_add2 = Table.read("/home/wanqqq/examples/run_2503/S25A-UH006-B/input/PFS_EDFN_March22_rev.csv")

            tb_tgt_add1["ob_code"] = tb_tgt_add1["ob_code"].astype("str")
            tb_tgt_add2["ob_code"] = tb_tgt_add2["ob_code"].astype("str")

            for col in ["filter_g","filter_r", "filter_i", "filter_z", "filter_y"]:
                tb_tgt[col] = tb_tgt[col].astype("str")
                tb_tgt_add1[col] = tb_tgt_add1[col].astype("str")
                tb_tgt_add2[col] = tb_tgt_add2[col].astype("str")

            for col in ["psf_flux_g","psf_flux_r", "psf_flux_i", "psf_flux_z", "psf_flux_y"]:
                tb_tgt[col] = tb_tgt[col].astype(float)
                tb_tgt_add1[col] = tb_tgt_add1[col].astype(float)
                tb_tgt_add2[col] = tb_tgt_add2[col].astype(float)

            for col in ["psf_flux_error_g","psf_flux_error_r", "psf_flux_error_i", "psf_flux_error_z", "psf_flux_error_y"]:
                tb_tgt[col] = tb_tgt[col].astype(float)
                tb_tgt_add1[col] = tb_tgt_add1[col].astype(float)
                tb_tgt_add2[col] = tb_tgt_add2[col].astype(float)

            tb_tgt = vstack([tb_tgt, tb_tgt_add1, tb_tgt_add2])

            tb_tgt["exptime_PPP"] = (
                np.ceil(tb_tgt["exptime"] / single_exptime_) * single_exptime_
            )
            tb_tgt["identify_code"] = [
                tt["proposal_id"] + "_" + tt["ob_code"]
                if "proposal_id" in tb_tgt.columns
                else tt["ob_code"]
                for tt in tb_tgt
            ]
            tb_tgt["exptime_assign"] = 0
            tb_tgt["exptime_done"] = 0  # observed exptime
            
            import astropy.units as u
            tb_tgt["i2_mag"] = (tb_tgt["psf_flux_i"]*u.nJy).to(u.ABmag)
            tb_tgt["exptime_PPP"][tb_tgt["i2_mag"]<24]=900
            tb_tgt["exptime_PPP"][tb_tgt["i2_mag"]>=24]=1800

            logger.info(f"Input target list: {tb_tgt}")

        if list(set(tb_tgt["proposal_id"])) == ["S25A-UH041-A"]:
            tb_tgt["exptime_PPP"]=900
            
        # separete the sample by 'resolution' (L=false/M=true)
        tb_tgt_l = tb_tgt[tb_tgt["resolution"] == "L"]
        tb_tgt_m = tb_tgt[tb_tgt["resolution"] == "M"]
            
        if len(para["localPath_ppc"]) > 0:
            path_ppc = para["localPath_ppc"]
            if path_ppc.endswith(".ecsv"):
                tb_ppc_tem = Table.read(para["localPath_ppc"])
                
                #only for s25a-039
                if list(set(tb_tgt["proposal_id"])) == ["S25A-039"]:
                    tb_ppc_tem["ppc_priority"][(tb_ppc_tem["ppc_ra"]<120)] = 0
                    tb_ppc_tem["ppc_priority"][(tb_ppc_tem["ppc_ra"]>=120)] = 5
                    
            else:
                from glob import glob 
                
                tables = []
                n = 0               
                for file in glob(path_ppc+"*"):
                    tbl = Table.read(file)
                    tbl["ppc_code"] = [code + "_" + str(n+1) for code in tbl["ppc_code"]]
                    
                    #only for s25a-039
                    if list(set(tb_tgt["proposal_id"])) == ["S25A-039"]:
                        if "55bf3ca22fda5257" in file:
                            tbl["ppc_priority"] = 0
                        else:
                            tbl["ppc_priority"] = 5
                            
                    tables.append(tbl)
                    n += 1
                tb_ppc_tem = vstack(tables, join_type="outer")

            if "ppc_priority" not in tb_ppc_tem.colnames:
                tb_ppc_tem["ppc_priority"] = 0

            logger.info(f"[S1] PPC list from usr:\n{tb_ppc_tem['ppc_code','ppc_ra','ppc_dec','ppc_pa','ppc_priority']}.")

            ppc_lst_tem = [
                [
                    ii,
                    tb_ppc_tem["ppc_ra"][ii],
                    tb_ppc_tem["ppc_dec"][ii],
                    tb_ppc_tem["ppc_pa"][ii],
                    tb_ppc_tem["ppc_priority"][ii],
                ]
                for ii in range(len(tb_ppc_tem))
            ]
            ppc_lst_l = [
                [
                    ii,
                    tb_ppc_tem["ppc_ra"][ii],
                    tb_ppc_tem["ppc_dec"][ii],
                    tb_ppc_tem["ppc_pa"][ii],
                    tb_ppc_tem["ppc_priority"][ii],
                ]
                for ii in range(len(tb_ppc_tem))
                if tb_ppc_tem["ppc_resolution"][ii] == "L"
            ]
            ppc_lst_m = [
                [
                    ii,
                    tb_ppc_tem["ppc_ra"][ii],
                    tb_ppc_tem["ppc_dec"][ii],
                    tb_ppc_tem["ppc_pa"][ii],
                    tb_ppc_tem["ppc_priority"][ii],
                ]
                for ii in range(len(tb_ppc_tem))
                if tb_ppc_tem["ppc_resolution"][ii] == "M"
            ]

            tb_tgt.meta["PPC"] = np.array(ppc_lst_tem)
            tb_tgt_l.meta["PPC"] = np.array(ppc_lst_l)
            tb_tgt_m.meta["PPC"] = np.array(ppc_lst_m)

            tb_tgt.meta["PPC_origin"] = "usr"
            tb_tgt_l.meta["PPC_origin"] = "usr"
            tb_tgt_m.meta["PPC_origin"] = "usr"

            logger.info(f"[S1] PPC list is read from {para['localPath_ppc']}.")

        logger.info(f"[S1] observation mode = {mode}")
        logger.info(
            f"[S1] Read targets done (takes {round(time.time()-time_start,3):.2f} sec)."
        )
        logger.info(f"[S1] There are {len(set(tb_tgt['proposal_id'])):.0f} proposals.")
        logger.info(
            f"[S1] n_tgt_low = {len(tb_tgt_l):.0f}, n_tgt_medium = {len(tb_tgt_m):.0f}"
        )

        return tb_tgt, tb_tgt_l, tb_tgt_m, tb_tgt_l, tb_tgt_m


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
    # print(f"There are {len(tgt_group)} groups.")

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
            threads_count = 4  # round(multiprocessing.cpu_count() / 2)
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

        if len(peak_pos[0]) == 0 or len(peak_pos[1]) == 0:
            peak_x = _tb_tgt["ra"].data[0]
            peak_y = _tb_tgt["dec"].data[0]
        else:
            peak_y = positions1[0, peak_pos[1][round(len(peak_pos[1]) * 0.5)]]
            peak_x = sorted(set(positions1[1, :]))[
                peak_pos[0][round(len(peak_pos[0]) * 0.5)]
            ]

        return X_, Y_, obj_dis_sig_, peak_x, peak_y


def PPP_centers(_tb_tgt, nPPC, weight_para=[1.5, 0, 0], randomseed=0, mutiPro=True):
    # determine pointing centers
    time_start = time.time()
    logger.info(f"[S2] Determine pointing centers started")

    para_sci, para_exp, para_n = weight_para

    ppc_lst = []

    if _tb_tgt.meta["PPC_origin"] == "usr":
        logger.warning(
            f"[S2] PPCs from usr adopted (takes {round(time.time()-time_start,3):.2f} sec)."
        )
        ppc_lst = _tb_tgt.meta["PPC"]
        return ppc_lst

    if len(_tb_tgt) == 0:
        logger.warning(f"[S2] no targets")
        return np.array(ppc_lst)

    _tb_tgt = sciRank_pri(_tb_tgt)
    _tb_tgt = count_N(_tb_tgt)
    _tb_tgt = weight(_tb_tgt, para_sci, para_exp, para_n)

    single_exptime_ = _tb_tgt.meta["single_exptime"]

    _tb_tgt_ = _tb_tgt[_tb_tgt["exptime_PPP"] > 0]

    """
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
            while len(lst_tgtID_assign) == 0:
                ppc_lst[-1][1] += np.random.uniform(-0.15, 0.15, 1)[0]
                ppc_lst[-1][2] += np.random.uniform(-0.15, 0.15, 1)[0]
                lst_tgtID_assign = netflowRun4PPC(
                    _tb_tgt_t_[list(index_)], ppc_lst[-1][1], ppc_lst[-1][2]
                )
                print(len(lst_tgtID_assign))
                
            lst_pslID_assign=[ii[5:10] for ii in lst_tgtID_assign]
            pslID_=sorted(set(lst_pslID_assign))
            pslID_n={tt:lst_pslID_assign.count(tt) for tt in pslID_}

            index_assign = np.in1d(_tb_tgt_t_["identify_code"], lst_tgtID_assign)
            _tb_tgt_t_["exptime_PPP"][
                index_assign
            ] -= single_exptime_  # targets in the PPC observed with single_exptime sec

            # add a small random so that PPCs determined would not have totally same weights
            weight_random = np.random.uniform(-0.05, 0.05, 1)[0]
            ppc_totPri.append(sum(_tb_tgt_t_["weight"][index_assign]) + weight_random)
            ppc_totPri_sub.append(
                sum(_tb_tgt_t_["weight"][index_assign]) + weight_random
            )
            ppc_lst[-1].append(sum(_tb_tgt_t_["weight"][index_assign]) + weight_random)

            #if len(lst_tgtID_assign) == 0:
                # quit if no targets assigned
                #break

            if iter_n>25 and len(lst_tgtID_assign)<2394*0.01: #ppc_totPri_sub[-1] < ppc_totPri_sub[0] * 0.1:
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
                f"PPC_{len(ppc_lst):03d}: {len(_tb_tgt_t)-len(_tb_tgt_t_):5d}/{len(_tb_tgt_t):10d} targets are finished (w={ppc_totPri[-1]:.2f}, fh={len(lst_tgtID_assign)*single_exptime_/3600.0:.2f})."
            )
            print(pslID_n)
            
    #"""
    pslID_ = sorted(set(_tb_tgt_["proposal_id"]))

    FH_goal = [
        _tb_tgt_["allocated_time"][_tb_tgt_["proposal_id"] == tt][0] for tt in pslID_
    ]

    tb_fh = Table([pslID_, FH_goal], names=["proposal_id", "FH_goal"])
    tb_fh["FH_done"] = 0.0
    tb_fh["N_done"] = 0.0

    while (
        (
            sum((tb_fh["FH_done"] >= tb_fh["FH_goal"]) * (tb_fh["N_done"] > 0.0))
            < len(tb_fh)
        )
        and len(_tb_tgt_) > 0
        and len(ppc_lst) <= nPPC
    ):
        weight_peak = []

        psl_id_undone = list(
            set(
                tb_fh["proposal_id"][
                    (tb_fh["FH_done"] < tb_fh["FH_goal"]) | (tb_fh["N_done"] == 0.0)
                ]
            )
        )
        print(f"The non-complete proposals: {psl_id_undone}")

        for _tb_tgt_t in target_DBSCAN(_tb_tgt_, 1.38):  # [_tb_tgt_]: #
            _tb_tgt_t_ = _tb_tgt_t[
                (_tb_tgt_t["exptime_PPP"] > 0)
                & np.in1d(_tb_tgt_t["proposal_id"], psl_id_undone)
            ]  # targets not finished
            if len(_tb_tgt_t_) == 0:
                continue

            _df_tgt_t = Table.to_pandas(_tb_tgt_t_)
            n_tgt = min(200, len(_tb_tgt_t_))
            _df_tgt_t = _df_tgt_t.sample(n_tgt, ignore_index=True, random_state=1)
            _tb_tgt_t_1 = Table.from_pandas(_df_tgt_t)

            X_, Y_, obj_dis_sig_, peak_x, peak_y = KDE(_tb_tgt_t_1, mutiPro)

            index_ = PFS_FoV(
                peak_x, peak_y, 90, _tb_tgt_t_
            )  # all PA set to be 0 for simplicity

            iter_tem = 0
            while len(index_) == 0 and iter_tem < 2:
                peak_x += np.random.uniform(-0.15, 0.15, 1)[0]
                peak_y += np.random.uniform(-0.15, 0.15, 1)[0]
                index_ = PFS_FoV(peak_x, peak_y, 0, _tb_tgt_t_)
                iter_tem += 1

            lst_tgtID_assign = netflowRun4PPC(_tb_tgt_t_[list(index_)], peak_x, peak_y, 0)
            iter_tem = 0
            while len(lst_tgtID_assign) == 0 and iter_tem < 2:
                peak_x += np.random.uniform(-0.15, 0.15, 1)[0]
                peak_y += np.random.uniform(-0.15, 0.15, 1)[0]
                lst_tgtID_assign = netflowRun4PPC(
                    _tb_tgt_t_[list(index_)],
                    peak_x,
                    peak_y,
                    otime="2025-04-10T08:00:00Z",
                )
                iter_tem += 1

            lst_pslID_assign = [ii.split("_")[0] for ii in lst_tgtID_assign]
            pslID_ = sorted(set(lst_pslID_assign))
            pslID_n = {
                tt: lst_pslID_assign.count(tt) * single_exptime_ / 3600.0
                for tt in pslID_
            }

            index_assign = np.in1d(_tb_tgt_t_["identify_code"], lst_tgtID_assign)
            weight_tem_tot = sum(_tb_tgt_t_["weight"][index_assign])
            weight_peak.append(
                [
                    peak_x,
                    peak_y,
                    0,
                    weight_tem_tot,
                    pslID_n,
                    lst_tgtID_assign,
                    index_assign,
                ]
            )

        weight_peak = sorted(weight_peak, key=lambda x: x[3])

        print(f"{weight_peak[-1][0]}, {weight_peak[-1][1]}, {len(weight_peak[-1][5])}")
        ppc_lst.append(
            np.array(
                [
                    len(ppc_lst),
                    weight_peak[-1][0],
                    weight_peak[-1][1],
                    weight_peak[-1][2],
                    weight_peak[-1][3],
                    sum(weight_peak[-1][4].values()) / sum(FH_goal),
                    len(weight_peak[-1][5]) / 2394.0,
                ]
            )
        )
        # ppc_id, ppc_ra, ppc_dec, ppc_pa, ppc_weight, ppc_fh, ppc_FE
        # print(ppc_lst[-1])

        index_assign = np.in1d(_tb_tgt_["identify_code"], weight_peak[-1][5])
        _tb_tgt_["exptime_PPP"][
            index_assign
        ] -= single_exptime_  # targets in the PPC observed with single_exptime sec

        for tt in list(weight_peak[-1][4].keys()):
            tb_fh["FH_done"].data[tb_fh["proposal_id"] == tt] += weight_peak[-1][4][tt]
            tb_fh["N_done"].data[tb_fh["proposal_id"] == tt] += sum(
                (_tb_tgt_["exptime_PPP"] <= 0) * (_tb_tgt_["proposal_id"] == tt)
            )

        _tb_tgt_ = _tb_tgt_[_tb_tgt_["exptime_PPP"] > 0]  # targets not finished
        _tb_tgt_ = count_N(_tb_tgt_)
        _tb_tgt_ = weight(_tb_tgt_, para_sci, para_exp, para_n)

        print(
            f"PPC_{len(ppc_lst):3d}: {len(_tb_tgt)-len(_tb_tgt_):5d}/{len(_tb_tgt):10d} targets are finished (w={weight_peak[-1][3]:.2f})."
        )
        Table.pprint_all(tb_fh)
    #

    if len(ppc_lst) > nPPC:
        ppc_lst_fin = sorted(ppc_lst, key=lambda x: x[4], reverse=True)[:nPPC]

    else:
        ppc_lst_fin = ppc_lst[:]

    ppc_lst_fin = np.array(ppc_lst_fin)

    logger.info(
        f"[S2] Determine pointing centers done ( nppc = {len(ppc_lst_fin):.0f}; takes {round(time.time()-time_start,3)} sec)"
    )

    return ppc_lst_fin

def PPC_centers_single(_tb_tgt, nPPC):
    def objective1(params, _tb_tgt):
        """
        Objective function to optimize the PPC parameters.
        
        Parameters:
          params: list or array-like of [ppc_ra, ppc_dec, ppc_pa]
          tb: pandas DataFrame containing target information, including 'i2_mag'
        
        Returns:
          Negative of the weighted sum of bright and faint counts (since we minimize).
        """
        #ra, dec, pa = params
        pa = params
        lst_tgtID_assign = netflowRun4PPC(_tb_tgt, ra, dec, pa)
        #index_ = PFS_FoV(ra, dec, pa, _tb_tgt)
        #lst_tgtID_assign = _tb_tgt["identify_code"][index_]
        
        index_assign = np.in1d(_tb_tgt["identify_code"], lst_tgtID_assign)

        N_P0 = sum(index_assign * (_tb_tgt["priority"]==0))
        N_P1 = sum(index_assign * (_tb_tgt["priority"]==1))
        N_P2 = sum(index_assign * (_tb_tgt["priority"]==2))
        N_P3 = sum(index_assign * (_tb_tgt["priority"]==3))
        N_P4 = sum(index_assign * (_tb_tgt["priority"]==4))
        N_P5 = sum(index_assign * (_tb_tgt["priority"]==5))
        N_P6 = sum(index_assign * (_tb_tgt["priority"]==6))
        N_P7 = sum(index_assign * (_tb_tgt["priority"]==7))
        N_P8 = sum(index_assign * (_tb_tgt["priority"]==8))
        N_P9 = sum(index_assign * (_tb_tgt["priority"]==9))
        N_P999 = sum(index_assign * (_tb_tgt["priority"]==999))

        N_P0_ = sum((_tb_tgt["priority"]==0))
        N_P1_ = sum((_tb_tgt["priority"]==1))
        N_P2_ = sum((_tb_tgt["priority"]==2))
        N_P3_ = sum((_tb_tgt["priority"]==3))
        N_P4_ = sum((_tb_tgt["priority"]==4))
        N_P5_ = sum((_tb_tgt["priority"]==5))
        N_P6_ = sum((_tb_tgt["priority"]==6))
        N_P7_ = sum((_tb_tgt["priority"]==7))
        N_P8_ = sum((_tb_tgt["priority"]==8))
        N_P9_ = sum((_tb_tgt["priority"]==9))
        N_P999_ = sum((_tb_tgt["priority"]==999))

        N_Pall = len(lst_tgtID_assign)

        print(f"{ra}, {dec}, {pa}, {N_Pall}/{len(_tb_tgt)}, {N_P0}/{N_P0_}, {N_P1}/{N_P1_}, {N_P2}/{N_P2_}, {N_P3}/{N_P3_}, {N_P4}/{N_P4_}, {N_P5}/{N_P5_}, {N_P6}/{N_P6_}, {N_P7}/{N_P7_}, {N_P8}/{N_P8_}, {N_P9}/{N_P9_}, {N_P999}/{N_P999_}")

        if list(set(_tb_tgt["proposal_id"])) == ["S25A-UH006-B"]:
            N_P0 = sum(index_assign * (_tb_tgt["priority"]==20))
            N_P1 = sum(index_assign * (_tb_tgt["priority"]==21))
            N_P2 = sum(index_assign * (_tb_tgt["priority"]==22))
            N_P3 = sum(index_assign * (_tb_tgt["priority"]==23))
            N_P4 = sum(index_assign * (_tb_tgt["priority"]==24))
            N_P5 = sum(index_assign * (_tb_tgt["priority"]==25))
            N_P6 = sum(index_assign * (_tb_tgt["priority"]==26))
            N_P7 = sum(index_assign * (_tb_tgt["priority"]==27))
            N_P8 = sum(index_assign * (_tb_tgt["priority"]==28))
            N_P9 = sum(index_assign * (_tb_tgt["priority"]==29))
            N_P999 = sum(index_assign * (_tb_tgt["priority"]==999))
    
            N_P0_ = sum((_tb_tgt["priority"]==20))
            N_P1_ = sum((_tb_tgt["priority"]==21))
            N_P2_ = sum((_tb_tgt["priority"]==22))
            N_P3_ = sum((_tb_tgt["priority"]==23))
            N_P4_ = sum((_tb_tgt["priority"]==24))
            N_P5_ = sum((_tb_tgt["priority"]==25))
            N_P6_ = sum((_tb_tgt["priority"]==26))
            N_P7_ = sum((_tb_tgt["priority"]==27))
            N_P8_ = sum((_tb_tgt["priority"]==28))
            N_P9_ = sum((_tb_tgt["priority"]==29))
            N_P999_ = sum((_tb_tgt["priority"]==999))
    
            N_Pall = len(lst_tgtID_assign)

            print(f"{ra}, {dec}, {pa}, {N_Pall}/{len(_tb_tgt)}, {N_P0}/{N_P0_}, {N_P1}/{N_P1_}, {N_P2}/{N_P2_}, {N_P3}/{N_P3_}, {N_P4}/{N_P4_}, {N_P5}/{N_P5_}, {N_P6}/{N_P6_}, {N_P7}/{N_P7_}, {N_P8}/{N_P8_}, {N_P9}/{N_P9_}, {N_P999}/{N_P999_}")
        
        # Define weights: you can adjust these based on your priorities.
        weight_pall = 1.0
        weight_p0 = 0.5
        weight_p999 = 1.50
        
        # We want to maximize the weighted sum; since the optimizer minimizes,
        # we return the negative of the weighted sum.
        score = weight_pall * N_Pall + weight_p0 * N_P0 + weight_p999 * N_P999
        return -score

    time_start = time.time()
    logger.info(f"[S2] Determine pointing centers started")

    #_tb_tgt = sciRank_pri(_tb_tgt)
    #_tb_tgt = count_N(_tb_tgt)
    #_tb_tgt = weight(_tb_tgt, 1,1,1)

    ppc_lst = []

    single_exptime_ = _tb_tgt.meta["single_exptime"]

    _tb_tgt_ = _tb_tgt[_tb_tgt["exptime_PPP"] > 0]
    _tb_tgt_["exptime_done"] = 0.0   

    pslID_ = sorted(set(_tb_tgt_["proposal_id"]))

    FH_goal = [
        _tb_tgt_["allocated_time"][_tb_tgt_["proposal_id"] == tt][0] for tt in pslID_
    ]

    tb_fh = Table([pslID_, FH_goal], names=["proposal_id", "FH_goal"])
    tb_fh["FH_done"] = 0.0
    tb_fh["N_done"] = 0.0

    

    while len(ppc_lst) < nPPC:
        if list(set(_tb_tgt["proposal_id"])) == ["S25A-UH006-B"]:
            if len(ppc_lst)==0:
                initial_guess = [150.08189537,   2.18829806,  92.51180584]
            elif len(ppc_lst)==1:
                initial_guess = [150.08220377 ,  2.18805709,  92.677787]
            elif len(ppc_lst)==2:
                initial_guess = [270.0,66.0,90.0]

            if len(ppc_lst) == 1:
                # 'tb_' is your DataFrame with target info (including 'i2_mag').
                result = minimize(objective1, initial_guess, args=(_tb_tgt_,), method='Nelder-Mead')
                print(result.x)
                ra, dec, pa = result.x[0], result.x[1], result.x[2]
            elif len(ppc_lst)==0:
                ra, dec, pa = [150.08220377 ,  2.18805709,  92.677787]
            elif len(ppc_lst)==2:
                ra, dec, pa = [270.29782837 , 65.7456042 ,  94.62414553]

        else:
            initial_guess = [150.0, 1.7, 0]
            result = minimize(objective1, initial_guess, args=(_tb_tgt_,), method='Nelder-Mead')
            print(result.x)
            ra, dec, pa = result.x[0], result.x[1], result.x[2]

        lst_tgtID_assign = netflowRun4PPC(_tb_tgt_, ra, dec, pa)

        ppc_lst.append(
            np.array(
                [
                    len(ppc_lst),
                    ra, 
                    dec, 
                    pa,
                    0,
                    lst_tgtID_assign,
                    len(lst_tgtID_assign) / 2394.0 * 100.0,
                ]
            )
        )

        index_assign = np.in1d(_tb_tgt_["identify_code"], lst_tgtID_assign)

        from collections import Counter
        print(dict(Counter(_tb_tgt_["exptime_PPP"][index_assign])))
        
        _tb_tgt_["exptime_PPP"][
            index_assign
        ] -= single_exptime_  # targets in the PPC observed with single_exptime sec

        _tb_tgt_["exptime_done"][
            index_assign
        ] += single_exptime_
        
        _tb_tgt_["priority"][_tb_tgt_["exptime_done"]>0] = 999
        
        if list(set(_tb_tgt["proposal_id"])) == ["S25A-UH006-B"]:
            _tb_tgt_["priority"][(_tb_tgt_["exptime_done"]==0) & (_tb_tgt_["exptime_PPP"]==900)] += 20

        _tb_tgt_ = _tb_tgt_[(_tb_tgt_["exptime_PPP"] >0)]
        print(sum(_tb_tgt_["priority"] == 999))

    ppc_lst_fin = np.array(ppc_lst)

    resol = _tb_tgt["ob_resolution"][0]
    if list(set(_tb_tgt["proposal_id"])) == ["S25A-UH006-B"]:
        ppc_code = ["PPC_L_uh006_" + str(n+1) for n in np.arange(nPPC)]
    else:
        ppc_code = [f"PPC_{resol}_" + str(n+1) for n in np.arange(nPPC)]
    ppc_ra = ppc_lst_fin[:,1]
    ppc_dec = ppc_lst_fin[:,2]
    ppc_pa = ppc_lst_fin[:,3]
    ppc_equinox = ["J2000"] * nPPC
    ppc_priority = [0] * nPPC
    ppc_priority_usr = [0] * nPPC
    ppc_exptime = [900.0] * nPPC
    ppc_totaltime = [1200.0] * nPPC
    ppc_resolution = [resol] * nPPC
    ppc_fibAlloFrac = ppc_lst_fin[:,-1]
    ppc_tgtAllo = ppc_lst_fin[:,-2]
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

    #ppcList.write("/home/wanqqq/examples/run_2503/S25A-UH006-B/output/ppp/ppcList.ecsv", format="ascii.ecsv", overwrite=True) 
    ppcList.write("/home/wanqqq/workDir_pfs/run_2505/S25A-UH041-A/output/ppp/ppcList.ecsv", format="ascii.ecsv", overwrite=True) 

    logger.info(
        f"[S2] Determine pointing centers done ( nppc = {len(ppc_lst_fin):.0f}; takes {round(time.time()-time_start,3)} sec)"
    )

    return ppc_lst_fin


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
            fh_ = _tb_tgt[_tb_tgt["proposal_id"] == psl_id_][
                    "allocated_time"
                ][0]
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
            "nonObservationCost": 200,
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

        #status = prob._prob.status  # or prob.getStatus() / prob.solverStatus, etc.
        #print("Model status:", status)

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
    #"""
    ppc_lst_ = []
    for i, (vis, tel) in enumerate(zip(res, telescope)):
        ppc_fib_eff = len(vis) / 2394.0 * 100

        logger.info(
            f"PPC {i:4d}: {len(vis):.0f}/2394={ppc_fib_eff:.2f}% assigned Cobras"
        )

        # assigned targets in each ppc
        tgt_assign_id_lst = []
        for tidx, cidx in vis.items():
            tgt_assign_id_lst.append(tgt_lst_netflow[tidx].ID)

        ppc_tot_weight = 0

        ppc_lst_.append(
            [
                "PPC_"
                + _tb_tgt_inuse["resolution"][0]
                + "_"
                + str(int(time.time() * 1e7))[-8:],
                "Group_1",
                tel._ra,
                tel._dec,
                tel._posang,
                ppc_tot_weight,
                ppc_fib_eff,
                tgt_assign_id_lst,
                _tb_tgt_inuse["resolution"][0],
            ]
        )

    tb_ppc_netflow = Table(
        np.array(ppc_lst_, dtype=object),
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
    tb_ppc_netflow["ppc_priority_usr"] = tb_ppc_netflow["ppc_priority"]

    #return tgt_assign_id_lst, tb_ppc_netflow
    return tgt_assign_id_lst, tb_ppc_netflow
    #"""


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

    #"""
    FH_goal = _tb_tgt["allocated_time"].data[0]
    FH_done = sum(_tb_tgt["exptime_assign"]) / 3600.0
    print(f"FH_goal = {FH_goal}, FH_done = {FH_done}")
    while FH_done > FH_goal and mode=="queue": # reduce Nppc only for queue modes
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
    #"""

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
        )
    if n_rank > 1 and n_exptime <= 1:
        para_sci, para_n = para
        para_exp = 0
        lst_ppc = PPP_centers(
            _tb_tgt, nppc_, [para_sci, para_exp, para_n], randomseed, True
        )
    if n_rank > 1 and n_exptime > 1:
        para_sci, para_exp, para_n = para
        lst_ppc = PPP_centers(
            _tb_tgt, nppc_, [para_sci, para_exp, para_n], randomseed, True
        )
    if n_rank <= 1 and n_exptime <= 1:
        para_n = para[0]
        para_sci = 1.5
        para_exp = 0
        lst_ppc = PPP_centers(
            _tb_tgt, nppc_, [para_sci, para_exp, para_n], randomseed, True
        )
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
    ob_single_exptime = _tb_tgt_tot["single_exptime"].data
    ob_resolution = _tb_tgt_tot["resolution"].data
    proposal_id = _tb_tgt_tot["proposal_id"].data
    proposal_rank = _tb_tgt_tot["rank"].data
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
            ob_single_exptime,
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
            "ob_single_exptime",
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
    nppc_l,
    nppc_m,
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

    randomseed = 2

    TraCollision = False
    multiProcess = True

    if list(set(tb_tgt["proposal_id"])) == ["S25A-UH006-B"]:
        # LR--------------------------------------------
        ppc_lst_l = PPC_centers_single(
            tb_sel_l, nppc_l
        )

        tb_tgt_l1 = Table.copy(tb_tgt_l)
        tb_tgt_l1.meta["PPC"] = ppc_lst_l

        tb_tgt_l1 = sciRank_pri(tb_tgt_l1)
        tb_tgt_l1 = count_N(tb_tgt_l1)
        tb_tgt_l1 = weight(tb_tgt_l1, 1, 0, 0)

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
            [1, 0, 0],
            nppc_l,
            randomseed,
            TraCollision,
            numReservedFibers,
            fiberNonAllocationCost,
            readtgt_con["mode_readtgt"],
        )
        tb_tgt_l_fin = netflowAssign(tb_tgt_l1, tb_ppc_l_fin)

        if nppc_l > 0:
            tb_ppc_tot = tb_ppc_l_fin.copy()
            tb_tgt_tot = tb_tgt_l_fin.copy()

        output(tb_ppc_tot, tb_tgt_tot, dirName=dirName)
        return None

    crMode = "compOFpsl_n"

    para_sci_l, para_exp_l, para_n_l = [1.5, 0, 0]
    para_sci_m, para_exp_m, para_n_m = [1.5, 0, 0]

    """ optimize
    if len(readtgt_con["para_readtgt"]["localPath_ppc"]) == 0:
        if len(tb_tgt_l) > 0:
            para_sci_l, para_exp_l, para_n_l = optimize(
                tb_tgt_l, nppc_l, crMode, randomseed, TraCollision
            )

        if len(tb_tgt_m) > 0:
            para_sci_m, para_exp_m, para_n_m = optimize(
                tb_tgt_m, nppc_m, crMode, randomseed, TraCollision
            )
    # """

    # LR--------------------------------------------
    ppc_lst_l = PPP_centers(
        tb_sel_l, nppc_l, [para_sci_l, para_exp_l, para_n_l], randomseed, multiProcess
    )

    #ppc_lst_l = PPC_centers_single(
    #    tb_sel_l, nppc_l
    #)
    
    tb_tgt_l1 = Table.copy(tb_tgt_l)
    tb_tgt_l1.meta["PPC"] = ppc_lst_l

    tb_tgt_l1 = sciRank_pri(tb_tgt_l1)
    tb_tgt_l1 = count_N(tb_tgt_l1)
    tb_tgt_l1 = weight(tb_tgt_l1, para_sci_l, para_exp_l, para_n_l)

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
        [para_sci_l, para_exp_l, para_n_l],
        nppc_l,
        randomseed,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
        readtgt_con["mode_readtgt"],
    )
    tb_tgt_l_fin = netflowAssign(tb_tgt_l1, tb_ppc_l_fin)

    # MR--------------------------------------------
    ppc_lst_m = PPP_centers(
        tb_sel_m, nppc_m, [para_sci_m, para_exp_m, para_n_m], randomseed, multiProcess
    )
    #ppc_lst_m = PPC_centers_single(
    #    tb_sel_m, nppc_m
    #)

    tb_tgt_m1 = Table.copy(tb_tgt_m)
    tb_tgt_m1.meta["PPC"] = ppc_lst_m

    tb_tgt_m1 = sciRank_pri(tb_tgt_m1)
    tb_tgt_m1 = count_N(tb_tgt_m1)
    tb_tgt_m1 = weight(tb_tgt_m1, para_sci_m, para_exp_m, para_n_m)

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
        [para_sci_m, para_exp_m, para_n_m],
        nppc_m,
        randomseed,
        TraCollision,
        numReservedFibers,
        fiberNonAllocationCost,
        readtgt_con["mode_readtgt"],
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

    # CR_tot, CR_tot_, sub_tot = complete_ppc(tb_tgt_tot, "compOFpsl_n")

    # plotCR(CR_tot_, sub_tot, tb_ppc_tot, dirName=dirName, show_plots=show_plots)
