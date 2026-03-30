#!/usr/bin/env python3
# PPP.py : PPP full version
import multiprocessing
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, get_body, solar_system_ephemeris
from astropy.time import Time
from astroplan import Observer
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
import ets_fiber_assigner.netflow as nf
from ics.cobraOps.Bench import Bench

warnings.filterwarnings("ignore")

logger_qplan = get_logger("qplan_test", null=True)
eph_cache = EphemerisCache(logger_qplan, precision_minutes=5)

# netflow configuration (FIXME; should be load from config file)
cobra_location_group = None
min_sky_targets_per_location = None
location_group_penalty = None
cobra_instrument_region = None
min_sky_targets_per_instrument_region = None
instrument_region_penalty = None
black_dot_penalty_cost = None
cobraSafetyMargin = 0.1
_COBRA_FEATURE_FLAGS = None

_NETFLOW_PRIORITY_COSTS = {
    999: 200,
    0: 100,
    1: 90,
    2: 80,
    3: 70,
    4: 60,
    5: 50,
    6: 40,
    7: 30,
    8: 20,
    9: 10,
}

_NETFLOW_CLASSDICT_TEMPLATE = {
    **{
        f"sci_P{priority}": {
            "nonObservationCost": non_observation_cost,
            "partialObservationCost": 200,
            "calib": False,
        }
        for priority, non_observation_cost in _NETFLOW_PRIORITY_COSTS.items()
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

def database_info(para_db):
    # Build the SQLAlchemy-style connection string from the DB config tuple.
    dialect, user, pwd, host, port, dbname = para_db
    return "{0}://{1}:{2}@{3}:{4}/{5}".format(dialect, user, pwd, host, port, dbname)


def remove_tgt_duplicate(df):
    # Remove duplicated targets that share the same proposal/object/catalog/resolution key.
    num1 = len(df)
    df = df.drop_duplicates(
        subset=["proposal_id", "obj_id", "input_catalog_id", "resolution"],
        inplace=False,
        ignore_index=True,
    )
    num2 = len(df)
    logger.info(f"Duplication removed: {num1} --> {num2}")
    return df


def visibility_checker2(
    tb_tgt,
    date="2026-01-25",
    ra_col="ra",
    dec_col="dec",
    time_start_utc=10,
    time_end_utc=17,
    time_step_min=15,
    ra_step_deg=1.0,
    dec_step_deg=1.0,
    min_alt=32 * u.deg,
    max_alt=75 * u.deg,
    min_moon_sep=0 * u.deg,
    max_moon_sep=360 * u.deg,
):
    """
    Select targets visible for ALL sampled times during the night
    using a precomputed RA–Dec visibility envelope.

    Parameters
    ----------
    tb_tgt : table-like
        Must contain RA/Dec columns
    date : str
        Date string, e.g. "2026-01-10"
    ra_col, dec_col : str
        Column names for RA/Dec in degrees
    time_start_utc, time_end_utc : int
        UTC hour range [start, end)
    time_step_min : int
        Time sampling in minutes
    ra_step_deg, dec_step_deg : float
        RA/Dec grid resolution (deg)
    min_alt, max_alt : astropy units
        Altitude limits
    min_moon_sep, max_moon_sep : astropy units
        Moon separation limits

    Returns
    -------
    mask_visible : np.ndarray (bool)
        Boolean mask for tb_tgt
    visible_envelope : 2D bool array
        RA–Dec envelope mask (for debugging / plotting)
    """

    # Observatory location used to evaluate altitude and moon separation.
    subaru = EarthLocation.of_site("Subaru Telescope")

    # Sample the night on a regular UTC grid.
    times = Time(
        [
            f"{date} {h:02d}:{m:02d}:00"
            for h in range(time_start_utc, time_end_utc)
            for m in range(0, 60, time_step_min)
        ],
        scale="utc",
    )

    # Build a coarse RA/Dec grid and evaluate visibility on the grid first.
    ra_grid = np.arange(0, 360 + ra_step_deg, ra_step_deg)
    dec_grid = np.arange(-30, 90 + dec_step_deg, dec_step_deg)
    ra2d, dec2d = np.meshgrid(ra_grid, dec_grid)

    skygrid = SkyCoord(ra2d * u.deg, dec2d * u.deg)

    # Track whether each grid point is visible in any sampled time slot.
    visible_all = np.zeros(ra2d.shape, dtype=bool)

    # Evaluate altitude and moon-separation constraints on the grid.
    for t in times:
        with solar_system_ephemeris.set('builtin'):
            moon_icrs = get_body('moon', t)
    
        altaz_frame = AltAz(obstime=t, location=subaru)
        altaz_grid = skygrid.transform_to(altaz_frame)
        moon_altaz = moon_icrs.transform_to(altaz_frame)
        
        moon_sep = altaz_grid.separation(moon_altaz)

        vis = (
            (altaz_grid.alt > min_alt) &
            (altaz_grid.alt < max_alt) &
            (moon_sep > min_moon_sep) &
            (moon_sep < max_moon_sep)
        )

        visible_all |= vis

    # Map each target onto the precomputed grid and keep only visible ones.
    ra_idx = np.round(tb_tgt[ra_col]).astype(int) % int(360 / ra_step_deg)
    dec_idx = np.round(
        (tb_tgt[dec_col] - dec_grid[0]) / dec_step_deg
    ).astype(int)

    valid_idx = (
        (dec_idx >= 0) &
        (dec_idx < visible_all.shape[0])
    )

    mask_visible = np.zeros(len(tb_tgt), dtype=bool)
    mask_visible[valid_idx] = visible_all[dec_idx[valid_idx], ra_idx[valid_idx]]

    logger.info(f"Visible targets: {len(mask_visible)} -> {sum(mask_visible)}")

    tb_tgt = tb_tgt[mask_visible]

    return tb_tgt



def visibility_checker(tb_tgt, obstimes, start_time_list, stop_time_list):
    # Detailed visibility check against actual observing windows.
    tz_HST = tz.gettz("US/Hawaii")

    min_el = 30.0
    max_el = 85.0

    tb_tgt["is_visible"] = False

    # NOTE: this loop is currently limited to the first target only.
    for i in range(1):#len(tb_tgt)):
        target = entity.StaticTarget(
            name=tb_tgt["ob_code"][i],
            ra=tb_tgt["ra"][i],
            dec=tb_tgt["dec"][i],
            equinox=2000.0,
        )
        # Total required observing time includes overhead between exposures.
        total_time = np.ceil(tb_tgt["exptime_usr"][i] / tb_tgt["single_exptime"][i]) * (
            tb_tgt["single_exptime"][i] + 300.0
        )  # SEC

        t_obs_ok = 0
        today = date.today().strftime("%Y-%m-%d")
        date_today = parser.parse(f"{today} 12:00 HST")

        for date_i in obstimes:
            date_t = parser.parse(f"{date_i} 12:00 HST")
            if date_today > date_t:
                # Skip nights that are already in the past.
                continue
            
            observer.set_date(date_t)
            default_start_time = observer.evening_twilight_18()
            default_stop_time = observer.morning_twilight_18()

            # Allow user-provided start/stop windows to override twilight limits.
            start_override = None
            stop_override = None

            for item in start_time_list:
                next_date = (
                    datetime.strptime(date_i, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if (date_i in item) and parser.parse(f"{item} HST") >= default_start_time:
                    start_override = parser.parse(f"{item} HST")
                    start_time_list.remove(item)
                    break
                elif (
                    (date_i in item)
                    and (parser.parse(f"{item} HST") < default_start_time)
                    and (
                        parser.parse(f"{item} HST")
                        > default_start_time - timedelta(hours=1)
                    )
                ):
                    start_override = default_start_time
                    start_time_list.remove(item)
                    break
                elif (next_date in item) and parser.parse(
                    f"{item} HST"
                ) <= default_stop_time:
                    start_override = parser.parse(f"{item} HST")
                    start_time_list.remove(item)
                    break
    
            for item in stop_time_list:
                next_date = (
                    datetime.strptime(date_i, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if (date_i in item) and parser.parse(f"{item} HST") >= default_start_time:
                    stop_override = parser.parse(f"{item} HST")
                    stop_time_list.remove(item)
                    break
                elif (next_date in item) and parser.parse(
                    f"{item} HST"
                ) <= default_stop_time:
                    stop_override = parser.parse(f"{item} HST")
                    stop_time_list.remove(item)
                    break
                elif (
                    (next_date in item)
                    and (parser.parse(f"{item} HST") > default_stop_time)
                    and (
                        parser.parse(f"{item} HST")
                        <= default_stop_time + timedelta(hours=1)
                    )
                ):
                    stop_override = default_stop_time
                    stop_time_list.remove(item)
                    break
    
            if start_override is not None:
                start_time = start_override
            else:
                start_time = default_start_time
    
            if stop_override is not None:
                stop_time = stop_override
            else:
                stop_time = default_stop_time

            if i == 0:
                logger.info(f"date: {date_i}, start={start_time}, stop={stop_time}")

            # Ask qplan whether the target can be observed long enough on this night.
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

    tb_tgt = tb_tgt[tb_tgt["is_visible"]]

    # Check whether each proposal still has enough visible time left.
    psl_id = sorted(set(tb_tgt["proposal_id"]))

    for psl_id_ in psl_id:
        tb_tgt_ = tb_tgt[tb_tgt["proposal_id"] == psl_id_]

        if sum(tb_tgt_["exptime_usr"]) / 3600.0 < tb_tgt_["allocated_time_tac"][0]:
            logger.error(
                f"{psl_id_}: visible targets too limited to achieve the allocated FHs. Please change obstime."
            )

    return tb_tgt

def query_queueDB(psl_id_list, DBPath_qDB, tb_queuedb_filename):
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
    # Reuse a cached queue table when available to avoid repeated DB access.
    if os.path.exists(tb_queuedb_filename):
        try:
            tb_queuedb = Table.read(tb_queuedb_filename)
            logger.info(f"Loaded cached queue table: {tb_queuedb_filename}")
            return tb_queuedb
        except Exception as e:
            logger.info("[S1] Querying the qdb (no cache found)")

    # Otherwise connect to qDB and query proposal-by-proposal.
    qdb = q_db.QueueDatabase(logger_qplan) 
    qdb.read_config(DBPath_qDB) 
    qdb.connect() 
    qa = q_db.QueueAdapter(qdb) 
    qq = q_query.QueueQuery(qa, use_cache=False)

    # Collect one row per executed OB with the effective exposure summary.
    results = []
    counter = 0

    for psl_id in psl_id_list:
        logger.info(f"Querying qDB for {psl_id}")
        ex_obs_list = qq.get_executed_obs_by_proposal(psl_id)
        if not ex_obs_list:
            continue

        for ex_ob in ex_obs_list:
            ex_ob_stats = qq.get_pfs_executed_ob_stats_by_ob_key(ex_ob.ob_key)
            ob = qq.get_ob(ex_ob.ob_key)
            arm = ob.inscfg.qa_reference_arm

            exptime_b = ex_ob_stats.cum_eff_exp_time_b
            exptime_r = ex_ob_stats.cum_eff_exp_time_r
            exptime_m = ex_ob_stats.cum_eff_exp_time_m
            exptime_n = ex_ob_stats.cum_eff_exp_time_n

            # This is the cumulative effective exposure time for the QA reference arm.
            exptime_selected = ex_ob_stats.cum_eff_exp_time

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
                    len(ex_ob.exp_history) * 450.0,  # nominal exposure time per OB
                ])

    if not results:
        logger.warning("No executed observations found in any proposal.")
        return Table()

    # Save the queried summary so the next run can load it directly.
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


def read_target(mode, para, tb_queuedb):
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
    logger.info("[S1] Read targets started (PPP)")

    # Shared constants for early returns and column normalization.
    empty_result = (Table(), Table(), Table(), Table())
    flux_columns = [
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
    filter_columns = ["filter_g", "filter_r", "filter_i", "filter_z", "filter_y"]

    # Step 1: load the raw target table from a local file or from the target DB.
    if len(para["localPath_tgt"]) > 0:
        tb_tgt = Table.read(para["localPath_tgt"])
        logger.info(f"[S1] Target list is read from {para['localPath_tgt']}.")

        if len(tb_tgt) == 0:
            logger.warning("[S1] No input targets.")
            return empty_result

    elif None in para["DBPath_tgt"]:
        logger.error("[S1] Incorrect connection info to database is provided.")
        return empty_result

    else:
        import pandas as pd
        import sqlalchemy as sa

        db_address = database_info(para["DBPath_tgt"])
        tgtDB = sa.create_engine(db_address)

        def query_target_from_db(proposal_ids):
            # Query all requested proposals at once and convert the DB schema to the internal table schema.
            sql = sa.text(
                "SELECT ob_code, obj_id, c.input_catalog_id AS input_catalog_id, ra, dec, epoch, priority, pmra, pmdec, parallax, effective_exptime, single_exptime, qa_reference_arm, is_medium_resolution, proposal.proposal_id AS proposal_id, rank, grade, allocated_time_lr+allocated_time_mr AS allocated_time_tac, allocated_time_lr, allocated_time_mr, filter_g, filter_r, filter_i, filter_z, filter_y, psf_flux_g, psf_flux_r, psf_flux_i, psf_flux_z, psf_flux_y, psf_flux_error_g, psf_flux_error_r, psf_flux_error_i, psf_flux_error_z, psf_flux_error_y, total_flux_g, total_flux_r, total_flux_i, total_flux_z, total_flux_y, total_flux_error_g, total_flux_error_r, total_flux_error_i, total_flux_error_z, total_flux_error_y FROM target JOIN proposal ON target.proposal_id=proposal.proposal_id JOIN input_catalog AS c ON target.input_catalog_id = c.input_catalog_id WHERE proposal.proposal_id IN :proposal_ids AND c.active;"
            ).bindparams(sa.bindparam("proposal_ids", expanding=True))

            with tgtDB.connect() as conn:
                df_tgt = pd.read_sql_query(
                    sql,
                    conn,
                    params={"proposal_ids": list(proposal_ids)},
                )

            df_tgt = df_tgt.rename(
                columns={
                    "epoch": "equinox",
                    "effective_exptime": "exptime_usr",
                    "is_medium_resolution": "resolution",
                }
            )
            df_tgt["resolution"] = np.where(df_tgt["resolution"], "M", "L")
            df_tgt["allocated_time_tac"] = np.where(
                df_tgt["resolution"] == "L",
                df_tgt["allocated_time_lr"],
                df_tgt["allocated_time_mr"],
            )
            df_tgt = df_tgt.drop(columns=["allocated_time_lr", "allocated_time_mr"])
            df_tgt = remove_tgt_duplicate(df_tgt)

            df_tgt[flux_columns] = df_tgt[flux_columns].apply(
                pd.to_numeric, errors="coerce"
            )

            tb_tgt_from_db = Table.from_pandas(df_tgt)
            for column in filter_columns:
                tb_tgt_from_db[column] = tb_tgt_from_db[column].astype("str")

            return tb_tgt_from_db

        proposal_ids = para["proposalIds"]
        tb_tgt = query_target_from_db(proposal_ids)

    # Step 2: standardize core columns used by the rest of the pipeline.
    tb_tgt["ra"] = tb_tgt["ra"].astype(float)
    tb_tgt["dec"] = tb_tgt["dec"].astype(float)
    tb_tgt["ob_code"] = tb_tgt["ob_code"].astype(str)
    if "proposal_id" in tb_tgt.columns:
        tb_tgt["identify_code"] = np.char.add(
            np.char.add(tb_tgt["proposal_id"].astype(str), "_"),
            tb_tgt["ob_code"].astype(str),
        )
    else:
        tb_tgt["identify_code"] = tb_tgt["ob_code"].astype(str)
    tb_tgt["exptime_assign"] = 0.0
    tb_tgt["exptime_done"] = 0.0  # observed exptime

    """
    proposalid = ["S25A-043QF", "S25A-119QF", "S25A-111QF", "S25A-116QF", "S25A-126QF", "S25A-017QF", "S25A-019QF", "S25A-112QF", "S25A-030QF", "S25A-034QF"]
    mask = np.isin(tb_tgt["proposal_id"], proposalid)
    tb_tgt["allocated_time_tac"][mask] = 10000.0
    #"""

    single_exptime_values = np.unique(tb_tgt["single_exptime"])
    if len(single_exptime_values) > 1:
        logger.error(
            "[S1] Multiple single-exptime are given. Not accepted now (240709)."
        )
        return empty_result

    tb_tgt.meta["single_exptime"] = single_exptime_values[0]
    logger.info(
        f"[S1] The single exptime is set to {tb_tgt.meta['single_exptime']:.2f} sec."
    )

    tb_tgt.meta["PPC"] = np.array([])
    tb_tgt.meta["PPC_origin"] = "auto"

    # Step 3: merge already-observed exposure information from queueDB when available.
    tb_tgt["allocated_time_done"] = 0.0
    tb_tgt["allocated_time"] = 0.0

    if len(tb_queuedb) == 0:
        tb_tgt["exptime"] = tb_tgt["exptime_usr"]
    else:
        tb_tgt = join(
            tb_tgt,
            tb_queuedb,
            keys_left=["proposal_id", "ob_code"],
            keys_right=["psl_id", "ob_code"],
            join_type="left",
        )

        exptime_usr = np.ma.filled(tb_tgt["exptime_usr"], 0.0).astype(float)
        exptime_done_real = np.ma.filled(
            tb_tgt["eff_exptime_done_real"], 0.0
        ).astype(float)
        tb_tgt["exptime_done"] = np.minimum(exptime_usr, exptime_done_real)

        tb_tgt.rename_column("ob_code_1", "ob_code")
        queuedb_columns = set(tb_queuedb.colnames)
        columns_to_remove = [
            column
            for column in tb_tgt.colnames
            if column in queuedb_columns and "ob_code" not in column
        ]
        tb_tgt.remove_columns(columns_to_remove)
        tb_tgt["exptime"] = tb_tgt["exptime_usr"] - tb_tgt["exptime_done"]

        proposal_ids = sorted(set(tb_tgt["proposal_id"]))
        for proposal_id in proposal_ids:
            for resolution in ["L", "M"]:
                mask = (tb_tgt["proposal_id"] == proposal_id) & (
                    tb_tgt["resolution"] == resolution
                )
                tb_tgt_resolution = tb_tgt[mask]
                if len(tb_tgt_resolution) == 0:
                    continue

                fh_done = sum(tb_tgt_resolution["exptime_done"]) / 3600.0
                tb_tgt["allocated_time_done"][mask] = fh_done
                fh_allocated = tb_tgt["allocated_time_tac"][mask].data[0]
                fh_completed = tb_tgt["allocated_time_done"][mask].data[0]
                logger.info(
                    f"{proposal_id} ({'LR' if resolution == 'L' else 'MR'}): allocated FH = {fh_allocated:.2f}, achieved FH = {fh_completed:.2f}, CR = {fh_completed/fh_allocated*100.0:.2f}%"
                )

    # Step 4: remove completed programs and completed targets.
    tb_tgt["allocated_time"] = (
        tb_tgt["allocated_time_tac"] - tb_tgt["allocated_time_done"]
    )
    tb_tgt["allocated_time"][tb_tgt["allocated_time"] < 0] = 0 
    tb_tgt = tb_tgt[tb_tgt["allocated_time"] > 0] # remove completed programs
    """ 
    mask = (
        ((tb_tgt["proposal_id"] == "S25B-049QN") & (tb_tgt["exptime"] <= 900.0) & (tb_tgt["exptime_done"] > 0) & (tb_tgt["ra"] > 100)) |
        (np.isin(tb_tgt["proposal_id"], ["S25B-053QN", "S25B-081QN", "S25B-071QN"]) & (tb_tgt["exptime_done"] > 0)) |
        (np.isin(tb_tgt["proposal_id"], ["S25B-047QN", "S25B-048QN", "S25B-086QN"]))
    )
    tb_tgt = tb_tgt[mask]
    tb_tgt["allocated_time"] = 2000.0
    #"""

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

    # Step 5: optionally apply the visibility envelope filter.
    if para["visibility_check"]:
        tb_tgt = visibility_checker2(
            tb_tgt, #para["obstimes"], para["starttimes"], para["stoptimes"]
        )

    # Step 6: split the final sample by resolution for downstream processing.
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
    return tb_tgt, tb_tgt_l, tb_tgt_m, tb_queuedb


def count_local_number(_tb_tgt):
    # calculate local count of targets (bin_width is 1 deg in ra&dec)
    # lower limit of dec is -40
    if len(_tb_tgt) == 0:
        return _tb_tgt

    ra_index = np.asarray(_tb_tgt["ra"], dtype=float).astype(int)
    dec_index = (np.asarray(_tb_tgt["dec"], dtype=float) + 40).astype(int)

    count_bin = np.zeros((131, 361), dtype=int)
    np.add.at(count_bin, (dec_index, ra_index), 1)

    _tb_tgt["local_count"] = count_bin[dec_index, ra_index]

    return _tb_tgt


def rank_recalculate(_tb_tgt):
    # calculate rank+priority of targets (higher value means more important)
    # Each distinct program rank defines its own priority interval.
    # Example for distinct ranks [6, 8, 10]:
    #   rank=10 -> priority ladder 10.0, 9.9, ..., 9.1
    #   rank=8  -> priority ladder  8.0, 7.9, ..., 7.1
    #   rank=6  -> priority ladder  6.0, 5.7, ..., 3.3
    # Programs sharing the same rank use the same interval.
    if len(_tb_tgt) == 0:
        return _tb_tgt

    rank_values = np.asarray(_tb_tgt["rank"], dtype=float)
    priority_values = np.asarray(_tb_tgt["priority"], dtype=int)

    # Targets from different programs can share the same rank value.
    # Use one interval per distinct rank, then map each target back to that interval.
    unique_ranks, rank_index = np.unique(rank_values, return_inverse=True)
    previous_distinct_ranks = np.concatenate(([0.0], unique_ranks[:-1]))

    interval_lower = 0.55 * unique_ranks + 0.45 * previous_distinct_ranks
    interval_step = 0.05 * (unique_ranks - previous_distinct_ranks)

    sci_usr_ranktot = (
        interval_lower[rank_index]
        + (9 - priority_values) * interval_step[rank_index]
    )

    _tb_tgt["rank_fin"] = np.exp(sci_usr_ranktot)

    return _tb_tgt

def weight(_tb_tgt, para_sci, para_exp, para_n):
    # calculate weights of targets (higher weights mean more important)
    if len(_tb_tgt) == 0:
        return _tb_tgt

    rank_fin = np.asarray(_tb_tgt["rank_fin"], dtype=float)
    exptime_ppp = np.asarray(_tb_tgt["exptime_PPP"], dtype=float)
    exptime = np.asarray(_tb_tgt["exptime"], dtype=float)
    exptime_done = np.asarray(_tb_tgt["exptime_done"], dtype=float)
    local_count = np.asarray(_tb_tgt["local_count"], dtype=float)
    single_exptime = float(_tb_tgt.meta["single_exptime"])

    weight_values = (
        np.power(rank_fin, para_sci)
        * np.power(exptime_ppp / single_exptime, para_exp)
        * np.power(local_count, para_n)
    )
    weight_values = np.nan_to_num(weight_values, nan=0.0)

    observed_mask = (exptime_done > 0) | (exptime_ppp < exptime)

    weight_max = np.max(weight_values)
    if np.any(observed_mask):
        weight_values[observed_mask] += weight_max

    rank_fin_updated = rank_fin.copy()
    rank_fin_max = np.max(rank_fin_updated)
    if np.any(observed_mask):
        rank_fin_updated[observed_mask] += rank_fin_max

    _tb_tgt["weight"] = weight_values
    _tb_tgt["rank_fin"] = rank_fin_updated

    return _tb_tgt


def target_clustering(_tb_tgt, sep=1.38):
    # separate targets into different groups
    # haversine uses (dec,ra) in radian;
    if len(_tb_tgt) == 0:
        return []

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

    target_coordinates = np.radians(
        np.column_stack((np.asarray(_tb_tgt["dec"], dtype=float), np.asarray(_tb_tgt["ra"], dtype=float)))
    )
    db = DBSCAN(eps=np.radians(sep), min_samples=1, metric="haversine").fit(
        target_coordinates
    )
    labels = db.labels_

    unique_labels, inverse_labels = np.unique(labels, return_inverse=True)
    cluster_weights = np.bincount(
        inverse_labels,
        weights=np.asarray(_tb_tgt["rank_fin"], dtype=float),
    )
    ordered_clusters = unique_labels[np.argsort(cluster_weights)[::-1]]

    tgt_group = []

    for cluster_label in ordered_clusters:
        tgt_t = _tb_tgt[labels == cluster_label]
        tgt_group.append(tgt_t)
        psl_ids_str = ", ".join(map(str, set(tgt_t["proposal_id"])))
        print(
            f'(RA = {tgt_t["ra"][0]}, DEC = {tgt_t["dec"][0]}): {psl_ids_str}, {sum(tgt_t["rank_fin"])}'
        )

    return tgt_group


def PFS_FoV(ppc_ra, ppc_dec, PA, _tb_tgt):
    # Pick up targets that fall inside the hexagonal PFS field of view.
    if len(_tb_tgt) == 0:
        return np.array([], dtype=int)

    target_coordinates = np.column_stack(
        (np.asarray(_tb_tgt["ra"], dtype=float), np.asarray(_tb_tgt["dec"], dtype=float))
    )
    ppc_center = SkyCoord(ppc_ra * u.deg, ppc_dec * u.deg)

    # PA=0 along y-axis, PA=90 along x-axis, PA=180 along -y-axis...
    # Build the 6 corners plus the closing point of the hexagon on the sky.
    hexagon = ppc_center.directional_offset_by(
        [30 + PA, 90 + PA, 150 + PA, 210 + PA, 270 + PA, 330 + PA, 30 + PA] * u.deg,
        1.38 / 2.0 * u.deg,
    )
    ra_h = hexagon.ra.deg
    dec_h = hexagon.dec.deg

    # For pointings around RA~0 or 360, shift wrapped vertices to the same
    # side as the PPC center before testing polygon containment.
    wrap_mask = np.fabs(ra_h - ppc_ra) > 180
    if np.any(wrap_mask):
        if ra_h[wrap_mask][0] > 180:
            ra_h[wrap_mask] -= 360
        else:
            ra_h[wrap_mask] += 360

    polygon = Path(np.column_stack((ra_h, dec_h)))
    index_ = np.where(polygon.contains_points(target_coordinates))[0]

    return index_

def KDE_xy(_tb_tgt, X, Y):
    # Calculate KDE on one target subset over the given RA/Dec grid.
    target_values = np.deg2rad(
        np.column_stack(
            (
                np.asarray(_tb_tgt["dec"], dtype=float),
                np.asarray(_tb_tgt["ra"], dtype=float),
            )
        )
    )
    kde = KernelDensity(
        bandwidth=np.deg2rad(1.38 / 2.0),
        kernel="linear",
        algorithm="ball_tree",
        metric="haversine",
    )
    kde.fit(target_values, sample_weight=np.asarray(_tb_tgt["weight"], dtype=float))

    grid_positions = np.deg2rad(np.column_stack((Y.ravel(), X.ravel())))
    Z = np.reshape(np.exp(kde.score_samples(grid_positions)), Y.shape)

    return Z


def KDE(_tb_tgt, multiProcesing):
    # Define the RA/Dec grid and calculate the KDE significance map.
    if len(_tb_tgt) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if len(_tb_tgt) == 1:
        # If only one target is available, use its coordinate as the peak.
        return (
            _tb_tgt["ra"].data[0],
            _tb_tgt["dec"].data[0],
            np.nan,
            _tb_tgt["ra"].data[0],
            _tb_tgt["dec"].data[0],
        )

    # Determine the KDE grid. Keep some margin around the target sample so the
    # KDE is not artificially clipped near the boundaries.
    ra_values = np.asarray(_tb_tgt["ra"], dtype=float)
    dec_values = np.asarray(_tb_tgt["dec"], dtype=float)

    ra_min = np.min(ra_values)
    ra_max = np.max(ra_values)
    dec_min = np.min(dec_values)
    dec_max = np.max(dec_values)

    ra_low = min(ra_min * 0.9, ra_min - 1)
    ra_up = max(ra_max * 1.1, ra_max + 1)
    dec_low = min(dec_min * 0.9, dec_min - 1)
    dec_up = max(dec_max * 1.1, dec_max + 1)

    ra_step = (ra_max - ra_min) / 100
    dec_step = (dec_max - dec_min) / 100

    # Use a local grid when the targets already span a compact region in both
    # RA and Dec, so the KDE stays focused around the sample.
    if ra_step < 0.5 and dec_step < 0.5:
        X_, Y_ = np.mgrid[ra_low:ra_up:101j, dec_low:dec_up:101j]
    # If the Dec range is compact but RA is broad, keep full-sky RA coverage
    # while preserving a fine local Dec grid.
    elif dec_step < 0.5:
        X_, Y_ = np.mgrid[0:360:721j, dec_low:dec_up:101j]
    # If the RA range is compact but Dec is broad, keep a local RA window and
    # fall back to the standard full Dec search range.
    elif ra_step < 0.5:
        X_, Y_ = np.mgrid[ra_low:ra_up:101j, -40:90:261j]
    # Otherwise search the default full survey footprint in both coordinates.
    else:
        X_, Y_ = np.mgrid[0:360:721j, -40:90:261j]

    if multiProcesing:
        threads_count = 4  # round(multiprocessing.cpu_count() / 2)
        thread_n = min(threads_count, round(len(_tb_tgt) * 0.5))

        with multiprocessing.Pool(thread_n) as pool:
            kde_maps = pool.map(
                partial(KDE_xy, X=X_, Y=Y_), np.array_split(_tb_tgt, thread_n)
            )

        Z = np.sum(kde_maps, axis=0)
    else:
        Z = KDE_xy(_tb_tgt, X_, Y_)

    # Convert the KDE map into a significance map and pick the central maximum
    # when several grid cells share the same peak value.
    obj_dis_sig_ = (Z - np.mean(Z)) / np.std(Z)
    peak_pos = np.where(obj_dis_sig_ == obj_dis_sig_.max())

    if len(peak_pos[0]) == 0 or len(peak_pos[1]) == 0:
        peak_x = _tb_tgt["ra"].data[0]
        peak_y = _tb_tgt["dec"].data[0]
    else:
        peak_row = peak_pos[0][round(len(peak_pos[0]) * 0.5)]
        peak_col = peak_pos[1][round(len(peak_pos[1]) * 0.5)]
        peak_x = np.unique(X_)[peak_row]
        peak_y = np.unique(Y_)[peak_col]

    return X_, Y_, obj_dis_sig_, peak_x, peak_y


def objective_ppc_assignment(trial_ppc, _tb_tgt, ppc_pa=0.0):
    """
    Objective function used to optimize the PPC center.

    Parameters
    ----------
    trial_ppc : sequence
        Trial PPC center as ``[ppc_ra, ppc_dec]`` in degrees.
    _tb_tgt : table-like
        Target sample used to evaluate the PPC.
    ppc_pa : float, optional
        Fixed PPC position angle in degrees. This parameter is passed through
        to the evaluator but is not included in the optimization vector.

    Returns
    -------
    float
        Negative utility score for the optimizer to minimize.
    """
    ppc_ra, ppc_dec = trial_ppc
    assigned_target_ids = fiber_allocate(
        _tb_tgt,
        single_ppc_mode=True,
        ppc_candidate=(ppc_ra, ppc_dec, ppc_pa),
    )
    #index_ = PFS_FoV(ppc_ra, ppc_dec, ppc_pa, _tb_tgt)
    #assigned_target_ids = _tb_tgt["identify_code"][index_]

    assigned_mask = np.isin(_tb_tgt["identify_code"], assigned_target_ids)

    priority_values = np.asarray(_tb_tgt["priority"])
    tracked_priorities = list(range(10)) + [999] # track priorities 0-9 and 999 (observed targets)
    assigned_counts = {}
    total_counts = {}
    for priority in tracked_priorities:
        priority_mask = priority_values == priority
        assigned_counts[priority] = int(np.sum(assigned_mask & priority_mask))
        total_counts[priority] = int(np.sum(priority_mask))

    n_assigned_total = len(assigned_target_ids)
    priority_summary = ", ".join(
        f"N{priority} = {assigned_counts[priority]}/{total_counts[priority]}"
        for priority in tracked_priorities
    )

    print(
        f"{ppc_ra}, {ppc_dec}, {ppc_pa}, "
        f"Nall = {n_assigned_total}/{len(_tb_tgt)}, {priority_summary}"
    )

    # Define weights: you can adjust these based on your priorities.
    weight_pall = 1.0
    weight_p0 = 0.5
    weight_p999 = 1.10

    # We want to maximize the weighted sum; since the optimizer minimizes,
    # we return the negative of the weighted sum.
    score = (
        weight_pall * n_assigned_total
        + weight_p0 * assigned_counts[0] # assigned targets with priority 0
        + weight_p999 * assigned_counts[999] # assigned targets with priority 999 (already observed)
    )
    return -score


def _prepare_tb_tgt_for_ppc(_tb_tgt, weight_params):
    # Recompute the target ranking metrics used by clustering and PPC scoring.
    science_weight, exposure_weight, density_weight = weight_params
    _tb_tgt = rank_recalculate(_tb_tgt)
    _tb_tgt = count_local_number(_tb_tgt)
    _tb_tgt = weight(_tb_tgt, science_weight, exposure_weight, density_weight)

    return _tb_tgt


def _select_tb_tgt_remaining(_tb_tgt, proposal_ids=None):
    # Build the current working subset from the single source-of-truth target table.
    tb_tgt_remaining = _tb_tgt[_tb_tgt["exptime_PPP"] > 0]

    if proposal_ids is not None:
        tb_tgt_remaining = tb_tgt_remaining[
            np.isin(tb_tgt_remaining["proposal_id"], proposal_ids)
        ]

    return tb_tgt_remaining


def _initialize_tb_proposal_progress(_tb_tgt):
    # Initialize proposal-level FH accounting for the PPC search loop.
    tb_tgt_remaining = _select_tb_tgt_remaining(_tb_tgt)
    proposal_ids = sorted(set(tb_tgt_remaining["proposal_id"]))
    proposal_fh_goal = [
        tb_tgt_remaining["allocated_time"][tb_tgt_remaining["proposal_id"] == proposal_id][0]
        for proposal_id in proposal_ids
    ]

    tb_proposal_progress = Table(
        [proposal_ids, proposal_fh_goal], names=["proposal_id", "FH_goal"]
    )
    tb_proposal_progress["FH_done"] = 0.0
    tb_proposal_progress["N_done"] = 0.0
    tb_proposal_progress["N_obs"] = 0.0
    tb_proposal_progress["N_psl"] = 0.0

    return tb_proposal_progress, proposal_fh_goal


def _sample_tb_tgt_rows(tb_tgt_input, max_rows, rng):
    # Sample rows directly from the Astropy table to avoid pandas conversion.
    if len(tb_tgt_input) <= max_rows:
        return tb_tgt_input

    sample_indices = rng.choice(len(tb_tgt_input), size=max_rows, replace=False)
    return tb_tgt_input[np.sort(sample_indices)]


def _select_ppc_seed(tb_tgt_remaining, rng, use_multiprocessing):
    # Cluster the sky distribution, then use KDE on the top cluster for the initial PPC guess.
    tb_tgt_groups = target_clustering(tb_tgt_remaining, 1.38)
    tb_tgt_group_primary = tb_tgt_groups[0]
    tb_tgt_group_sampled = _sample_tb_tgt_rows(tb_tgt_group_primary, 200, rng)

    _, _, _, initial_ra, initial_dec = KDE(tb_tgt_group_sampled, use_multiprocessing)

    return tb_tgt_group_primary, initial_ra, initial_dec


def _calculate_tb_tgt_credit_seconds(tb_tgt_assigned):
    # Cap credited FH at the requested exposure once exptime_done exceeds exptime.
    requested_exptime = np.asarray(tb_tgt_assigned["exptime"], dtype=float)
    exptime_done = np.asarray(tb_tgt_assigned["exptime_done"], dtype=float)
    credited_exptime = exptime_done.copy()
    overdone_mask = exptime_done > requested_exptime
    credited_exptime[overdone_mask] = requested_exptime[overdone_mask]

    return credited_exptime


def _calculate_fh_done_by_proposal(_tb_tgt):
    # Recompute credited FH directly from the current exptime_done state.
    credited_exptime = _calculate_tb_tgt_credit_seconds(_tb_tgt)
    proposal_ids = np.asarray(_tb_tgt["proposal_id"], dtype=str)
    unique_proposal_ids, inverse_indices = np.unique(proposal_ids, return_inverse=True)
    credited_fh = np.bincount(inverse_indices, weights=credited_exptime) / 3600.0

    return {
        proposal_id: fh_done
        for proposal_id, fh_done in zip(unique_proposal_ids, credited_fh)
    }


def _summarize_tb_tgt_assignment(_tb_tgt, tb_tgt_assigned_mask):
    # Summarize the newly assigned targets for PPC bookkeeping.
    tb_tgt_assigned = _tb_tgt[tb_tgt_assigned_mask]
    assigned_credit_seconds = _calculate_tb_tgt_credit_seconds(tb_tgt_assigned)
    total_assigned_weight = float(np.sum(np.asarray(tb_tgt_assigned["rank_fin"], dtype=float)))

    proposal_ids = np.asarray(tb_tgt_assigned["proposal_id"], dtype=str)
    unique_proposal_ids, inverse_indices = np.unique(proposal_ids, return_inverse=True)
    credited_fh = np.bincount(inverse_indices, weights=assigned_credit_seconds) / 3600.0
    assigned_fh_by_proposal = {
        proposal_id: fh_done
        for proposal_id, fh_done in zip(unique_proposal_ids, credited_fh)
    }

    return tb_tgt_assigned, assigned_credit_seconds, total_assigned_weight, assigned_fh_by_proposal


def _update_tb_proposal_progress(
    tb_proposal_progress,
    _tb_tgt,
):
    # Update proposal-level completion metrics after one PPC assignment.
    fh_done_by_proposal = _calculate_fh_done_by_proposal(_tb_tgt)

    for proposal_id in tb_proposal_progress["proposal_id"]:
        proposal_progress_mask = tb_proposal_progress["proposal_id"] == proposal_id
        proposal_mask = _tb_tgt["proposal_id"] == proposal_id

        tb_proposal_progress["N_psl"].data[proposal_progress_mask] = np.sum(
            proposal_mask
        )
        tb_proposal_progress["FH_done"].data[proposal_progress_mask] = fh_done_by_proposal.get(
            proposal_id, 0.0
        )
        tb_proposal_progress["N_done"].data[proposal_progress_mask] = np.sum(
            _tb_tgt["exptime_PPP"][proposal_mask] <= 0
        )
        tb_proposal_progress["N_obs"].data[proposal_progress_mask] = np.sum(
            _tb_tgt["exptime_PPP"][proposal_mask] < _tb_tgt["exptime"][proposal_mask]
        )


def _build_ppc_list_table(final_ppc_records, _tb_tgt, backup):
    # Convert the legacy PPC record array into the output table used downstream.
    ppc_weights = final_ppc_records[:, 4]
    weight_for_qplan = np.arange(1, len(final_ppc_records) + 1, dtype=int)

    n_ppc_final = len(final_ppc_records)
    resolution = _tb_tgt["resolution"][0]

    if backup:
        ppc_codes = [
            f"que_{resolution}_{datetime.now().strftime('%y%m%d')}_{int(index + 1)}_backup"
            for index in range(n_ppc_final)
        ]
    else:
        ppc_codes = [
            f"que_{resolution}_{datetime.now().strftime('%y%m%d')}_{int(index + 1)}"
            for index in range(n_ppc_final)
        ]

    return Table(
        [
            ppc_codes,
            final_ppc_records[:, 1],
            final_ppc_records[:, 2],
            final_ppc_records[:, 3],
            ["J2000"] * n_ppc_final,
            weight_for_qplan,
            ppc_weights,
            [900.0] * n_ppc_final,
            [1200.0] * n_ppc_final,
            [resolution] * n_ppc_final,
            final_ppc_records[:, -2],
            final_ppc_records[:, -1],
            [""] * n_ppc_final,
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


def PPP_centers(
    _tb_tgt,
    n_ppc,
    weight_params=[2, 0, 0],
    random_seed=0,
    use_multiprocessing=True,
    backup=False,
    fixed_ppc_pa=0.0,
):
    """Determine PPC centers from the remaining target sample.

    The optimizer only searches in RA/Dec. The PPC position angle is kept fixed
    through ``fixed_ppc_pa`` and passed to the evaluator unchanged.
    """
    start_time = time.time()
    rng = np.random.default_rng(random_seed)
    logger.info("[S2] Determine pointing centers started")

    ppc_records = []

    # If there are no targets or no PPC to be determined, skip the optimization and return empty results.
    if len(_tb_tgt) == 0:
        logger.warning(f"[S2] no targets")
        return np.array(ppc_records), Table()

    # If the number of PPC to be determined is set to zero, skip the optimization and return empty results.
    if n_ppc == 0:
        logger.warning(f"[S2] no PPC to be determined")
        return np.array(ppc_records), Table()

    single_exptime = _tb_tgt.meta["single_exptime"]

    # Recompute the target weights and proposal-level FH goals based on the current state of the source-of-truth table.
    _tb_tgt = _prepare_tb_tgt_for_ppc(_tb_tgt, weight_params) 

    # Set up the proposal progress tracking table to monitor completion status during the PPC search loop
    tb_proposal_progress, proposal_fh_goal = _initialize_tb_proposal_progress(_tb_tgt)
    total_fh_goal = float(np.sum(proposal_fh_goal))

    # Keep only incomplete targets and proposals.
    incomplete_proposal_ids = list(tb_proposal_progress["proposal_id"])
    tb_tgt_remaining = _select_tb_tgt_remaining(_tb_tgt, incomplete_proposal_ids)

    while (
        (
            sum(
                (tb_proposal_progress["FH_done"] >= tb_proposal_progress["FH_goal"])
                * (tb_proposal_progress["N_done"] > 0.0)
            )
            < len(tb_proposal_progress)
        )
        and len(tb_tgt_remaining) > 0
        and len(ppc_records) < n_ppc
    ):
        if incomplete_proposal_ids:
            undone_str = ", ".join(map(str, incomplete_proposal_ids))
            print("-----------------------------------------")
            print(f"The non-complete proposals: {undone_str}")
        else:
            print("All proposals complete.")

        # Comfirm partial observed targets are prioritized in the PPC optimization by setting their priority to 999.
        _tb_tgt["priority"][_tb_tgt["exptime_done"] > 0] = 999

        # Group targets to save running time. The top cluster is used to determine the initial PPC center for optimization.
        tb_tgt_group_primary, initial_ra, initial_dec = _select_ppc_seed(
            tb_tgt_remaining, rng, use_multiprocessing
        )

        # Optimize only the PPC center; keep position angle fixed.
        initial_ppc = [initial_ra, initial_dec]
        optimization_result = minimize(
            objective_ppc_assignment,
            initial_ppc,
            args=(tb_tgt_group_primary, fixed_ppc_pa),
            method="Nelder-Mead",
            options={
                "xatol": 0.1,
                "fatol": 0.1,
                "maxiter": 25, # limit the number of function evaluations to 25
                "maxfev": 25, # set maxfev to 25 to ensure the optimization stops after 25 evaluations even if it hasn't converged yet
            },
        )
        print(f"The optimal PPC center: {optimization_result.x}")
        best_ppc_ra, best_ppc_dec = optimization_result.x[0], optimization_result.x[1]
        best_ppc_pa = fixed_ppc_pa

        assigned_target_ids = fiber_allocate(
            tb_tgt_remaining,
            single_ppc_mode=True,
            ppc_candidate=(best_ppc_ra, best_ppc_dec, best_ppc_pa),
        )

        # Retry a couple of small center shifts if the first netflow attempt allocates no targets.
        retry_count = 0
        while len(assigned_target_ids) == 0 and retry_count < 2:
            best_ppc_ra += rng.uniform(-0.15, 0.15)
            best_ppc_dec += rng.uniform(-0.15, 0.15)
            assigned_target_ids = fiber_allocate(
                tb_tgt_remaining,
                single_ppc_mode=True,
                ppc_candidate=(best_ppc_ra, best_ppc_dec, best_ppc_pa),
                observation_time="2026-01-10T10:00:00Z",
            )
            retry_count += 1

        # Update exptime_PPP and exptime_done and FH in tb_tgt for the assigned targets
        tb_tgt_assigned_mask = np.isin(_tb_tgt["identify_code"], assigned_target_ids)
        # Consume one full frame in assigned time, while FH accounting stays capped by exptime.
        _tb_tgt["exptime_PPP"][tb_tgt_assigned_mask] -= single_exptime
        _tb_tgt["exptime_done"][tb_tgt_assigned_mask] += single_exptime
        _tb_tgt["priority"][_tb_tgt["exptime_done"] > 0] = 999

        (
            tb_tgt_assigned,
            assigned_credit_seconds,
            total_assigned_weight,
            assigned_fh_by_proposal,
        ) = _summarize_tb_tgt_assignment(_tb_tgt, tb_tgt_assigned_mask)

        print(f"{best_ppc_ra}, {best_ppc_dec}, {len(tb_tgt_assigned)}")

        # Store PPC metadata in the legacy array layout expected by downstream code.
        ppc_records.append(
            np.array(
                [
                    len(ppc_records),
                    best_ppc_ra,
                    best_ppc_dec,
                    best_ppc_pa,
                    total_assigned_weight,
                    sum(assigned_fh_by_proposal.values()) / total_fh_goal,
                    len(tb_tgt_assigned) / 2394.0,
                    assigned_target_ids,
                ],
                dtype=object,
            ),
        )

        # Update the proposal progress tracking table and the working target subset for the next iteration of the PPC search loop.
        proposal_mask = np.isin(_tb_tgt["proposal_id"], incomplete_proposal_ids)
        n_partially_observed_before_filter = sum(
            (_tb_tgt["exptime_done"] > 0) * proposal_mask
        )

        _update_tb_proposal_progress(
            tb_proposal_progress,
            _tb_tgt,
        )

        incomplete_proposal_ids = list(
            set(
                tb_proposal_progress["proposal_id"][
                    (tb_proposal_progress["FH_done"] < tb_proposal_progress["FH_goal"])
                    | (tb_proposal_progress["N_done"] == 0.0)
                ]
            )
        )
        tb_tgt_remaining = _select_tb_tgt_remaining(_tb_tgt, incomplete_proposal_ids)
        n_partially_observed_after_filter = sum(tb_tgt_remaining["exptime_done"] > 0)

        print(
            f"PPC_{len(ppc_records):3d}: {len(_tb_tgt)-len(tb_tgt_remaining):5d}/{len(_tb_tgt):10d} targets are finished (w={total_assigned_weight:.2f}). (partial = {n_partially_observed_before_filter}, {n_partially_observed_after_filter})"
        )
        Table.pprint_all(tb_proposal_progress)

    # Reformat the PPC records and build the output table for the final set of PPC centers.
    if len(ppc_records) == 0:
        logger.warning("[S2] No valid PPC centers were determined")
        return np.array(ppc_records), Table()
    elif len(ppc_records) > n_ppc:
        final_ppc_records = sorted(ppc_records, key=lambda x: x[4], reverse=True)[:n_ppc]
    else:
        final_ppc_records = sorted(ppc_records, key=lambda x: x[4], reverse=True)

    final_ppc_records = np.asarray(final_ppc_records, dtype=object)

    if final_ppc_records.ndim == 1:
        # Preserve a 2D shape when only one PPC was generated.
        final_ppc_records = final_ppc_records.reshape(1, -1)
    else:
        sort_order = np.argsort(-np.asarray(final_ppc_records[:, 4], dtype=float))
        final_ppc_records = final_ppc_records[sort_order]

    ppc_list_table = _build_ppc_list_table(final_ppc_records, _tb_tgt, backup)

    logger.info(
        f"[S2] Determine pointing centers done ( nppc = {len(final_ppc_records):.0f}; takes {round(time.time()-start_time,3)} sec)"
    )

    return final_ppc_records, ppc_list_table


def build_netflow_targets(tb_tgt, for_single_ppc=False):
    # Convert the target table into the netflow target list and FH budget bundle.
    #
    # When evaluating a single PPC candidate, every target is passed to netflow
    # with one single exposure.
    netflow_targets = []
    proposal_class_keys = []
    single_exptime = tb_tgt.meta["single_exptime"]
    exposure_column = "exptime_PPP"
    use_cobra_feature_flag = tb_tgt.meta.get("cobra_feature_flag", True)

    for tb_tgt_row in tb_tgt:
        target_exptime = (
            single_exptime if for_single_ppc else tb_tgt_row[exposure_column]
        )
        target_priority = int(tb_tgt_row["priority"])
        qa_reference_arm = tb_tgt_row["qa_reference_arm"]

        # Module-2 cobras cannot provide NIR, so NIR targets must carry the
        # special netflow request flag.
        request_flags = 0
        if use_cobra_feature_flag:
            request_flags = 0 if qa_reference_arm == "n" else 1

        netflow_targets.append(
            nf.ScienceTarget(
                tb_tgt_row["identify_code"],
                tb_tgt_row["ra"],
                tb_tgt_row["dec"],
                target_exptime,
                target_priority,
                "sci",
                req_flags=request_flags,
            )
        )
        proposal_class_keys.append(f"sci_P{target_priority}")

    # Keep the proposal FH budget bundle in the legacy format expected by the
    # downstream netflow setup.
    proposal_fh_limits = {}

    if not for_single_ppc:
        proposal_ids = sorted(set(tb_tgt["proposal_id"]))

        for proposal_id in proposal_ids:
            class_key_bundle = tuple(
                class_key for class_key in proposal_class_keys if proposal_id in class_key
            )
            fh_limit = tb_tgt[tb_tgt["proposal_id"] == proposal_id]["allocated_time"][0]
            proposal_fh_limits[class_key_bundle] = fh_limit

            print(f"{proposal_id}: FH_limit = {proposal_fh_limits[class_key_bundle]:.2f}")

    return netflow_targets, proposal_fh_limits


def build_classdict():
    # Build the netflow class-cost mapping used by the optimizer.
    return {
        class_key: class_config.copy()
        for class_key, class_config in _NETFLOW_CLASSDICT_TEMPLATE.items()
    }


def cobra_move_cost(dist):
    # optional: penalize assignments where the cobra has to move far out
    return 0.1 * dist


def _get_cobra_feature_flags():
    global _COBRA_FEATURE_FLAGS

    if _COBRA_FEATURE_FLAGS is not None:
        return _COBRA_FEATURE_FLAGS

    from pathlib import Path
    from pfs.utils.fiberids import FiberIds
    import pfs.utils

    pfs_utils_path = Path(pfs.utils.__path__[0])
    fiber_data_path = pfs_utils_path.parent.parent.parent / "data" / "fiberids"
    if not fiber_data_path.exists():
        fiber_data_path = pfs_utils_path / "data" / "fiberids"

    cobra_indices_module2 = FiberIds(path=fiber_data_path).cobrasForSpectrograph(
        spectrographId=2
    )
    cobra_indices_module2 = np.asarray(cobra_indices_module2)
    cobra_indices_module2 = cobra_indices_module2[cobra_indices_module2 <= 2394]

    module2_mask = np.zeros(2394, dtype=bool)
    module2_mask[cobra_indices_module2] = True

    cobra_feature_flags = np.zeros(2394, dtype=int)
    cobra_feature_flags[module2_mask] = 1
    cobra_feature_flags.setflags(write=False)
    _COBRA_FEATURE_FLAGS = cobra_feature_flags

    return _COBRA_FEATURE_FLAGS


def run_netflow(
    ppc_list,
    tb_tgt,
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    observation_time="2026-01-10T10:00:00Z",
    for_single_ppc=False,
):
    # Run netflow once for the given PPC list.
    telescope_ra = ppc_list[:, 1]
    telescope_dec = ppc_list[:, 2]
    telescope_pa = ppc_list[:, 3]

    netflow_targets, proposal_fh_limits = build_netflow_targets(
        tb_tgt, for_single_ppc=for_single_ppc
    )
    class_dict = build_classdict()

    telescopes = [
        nf.Telescope(telescope_ra[index], telescope_dec[index], telescope_pa[index], observation_time)
        for index in range(len(telescope_ra))
    ]
    focal_plane_positions = [
        telescope.get_fp_positions(netflow_targets) for telescope in telescopes
    ]

    n_visit = len(telescope_ra)
    single_exptime = tb_tgt.meta["single_exptime"]

    # optional: slightly increase the cost for later observations,
    # to observe as early as possible
    visit_costs = [0] * n_visit

    # Set up Gurobi parameters for netflow optimization. These can be tuned for performance and solution quality.
    gurobi_options = dict(
        seed=0,
        presolve=1,
        method=4,
        degenmoves=0,
        heuristics=0.8,
        mipfocus=0,
        mipgap=5.0e-2,
        LogToConsole=0,
    )

    forbidden_pairs = [[] for _ in range(n_visit)]
    already_observed = {}
    if tb_tgt.meta.get("cobra_feature_flag", True):
        # Build the cobra feature flags lazily once per process, then reuse them
        # across subsequent netflow runs.
        cobra_feature_flags = _get_cobra_feature_flags()
    else:
        cobra_feature_flags = None

    # Set up the netflow problem
    problem = nf.buildProblem(
        bench,
        netflow_targets,
        focal_plane_positions,
        class_dict,
        single_exptime,
        visit_costs,
        cobraMoveCost=cobra_move_cost,
        collision_distance=2.0,
        elbow_collisions=True,
        gurobi=True,
        gurobiOptions=gurobi_options,
        alreadyObserved=already_observed,
        forbiddenPairs=forbidden_pairs,
        cobraLocationGroup=cobra_location_group,
        minSkyTargetsPerLocation=min_sky_targets_per_location,
        locationGroupPenalty=location_group_penalty,
        cobraInstrumentRegion=cobra_instrument_region,
        minSkyTargetsPerInstrumentRegion=min_sky_targets_per_instrument_region,
        instrumentRegionPenalty=instrument_region_penalty,
        blackDotPenalty=black_dot_penalty_cost,
        numReservedFibers=num_reserved_fibers,
        fiberNonAllocationCost=fiber_non_allocation_cost,
        obsprog_time_budget=proposal_fh_limits,
        cobraSafetyMargin=cobraSafetyMargin,
        cobraFeatureFlags=cobra_feature_flags,
    )

    problem.solve()

    # status = prob._prob.status  # or prob.getStatus() / prob.solverStatus, etc.
    # print("Model status:", status)

    # Extract the assigned targets for each pointing from the optimization result.
    res = [{} for _ in range(min(n_visit, len(telescope_ra)))]
    for k1, v1 in problem._vardict.items():
        if k1.startswith("Tv_Cv_"):
            visited = problem.value(v1) > 0
            if visited:
                _, _, tidx, cidx, ivis = k1.split("_")
                res[int(ivis)][int(tidx)] = int(cidx)

    return res, telescopes, netflow_targets


def fiber_allocate(
    tb_tgt,
    single_ppc_mode=False,
    ppc_candidate=None,
    observation_time="2026-01-10T10:00:00Z",
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    backup=False,
):
    # Run fiber allocation either for all stored PPCs or for one PPC candidate.
    time_start = time.time()
    logger.info("[S3] Run netflow started")

    # Run for one PPC 
    if single_ppc_mode:
        if ppc_candidate is None: # no PPC candidate provided 
            raise ValueError(
                "ppc_candidate must be provided when single_ppc_mode=True"
            )

        ppc_ra, ppc_dec, ppc_pa = ppc_candidate
        ppc_list = np.array([[0, ppc_ra, ppc_dec, ppc_pa, 0]], dtype=object)

        # Run netflow
        assignments, telescopes, netflow_targets = run_netflow(
            ppc_list,
            tb_tgt,
            for_single_ppc=True,
            observation_time=observation_time,
        )

        # return the assigned target IDs for the single PPC candidate
        return [
            netflow_targets[target_idx].ID
            for assignment_map in assignments
            for target_idx in assignment_map
        ]

    # Skip if no ppc
    elif ("PPC" not in tb_tgt.meta) or (len(tb_tgt.meta["PPC"]) == 0):
        logger.warning("[S3] No PPC has been determined")
        return []
    
    # Skip if no target
    elif len(tb_tgt) == 0:
        logger.warning("[S3] No targets")
        return []

    # Run for all PPCs in the list
    else:
        ppc_records = []

        ppc_list = tb_tgt.meta["PPC"]
        today_date = datetime.now().strftime("%y%m%d")
        resolution = tb_tgt["resolution"][0]

        target_indices_in_group = set()
        for ppc_row in ppc_list:
            # pick up targets in the FoV of each PPC candidate to form the input target sample for netflow, so that netflow can run faster with a smaller target sample.
            target_indices_in_group.update(
                PFS_FoV(ppc_row[1], ppc_row[2], ppc_row[3], tb_tgt)
            )

        if len(target_indices_in_group) > 0:
            tb_tgt_in_group = tb_tgt[sorted(target_indices_in_group)]

            logger.info(
                f"[S3] Group {1:3d}: nppc = {len(ppc_list):5d}, n_tgt = {len(tb_tgt_in_group):6d}"
            )

            # run netflow
            assignments, telescopes, netflow_targets = run_netflow(
                ppc_list,
                tb_tgt_in_group,
                num_reserved_fibers=num_reserved_fibers,
                fiber_non_allocation_cost=fiber_non_allocation_cost,
            )

            # Extract the assigned targets for each pointing
            for pointing_index, (assignment_map, telescope) in enumerate(
                zip(assignments, telescopes), start=1
            ):
                ppc_fiber_usage_frac = len(assignment_map) / 2394.0 * 100

                logger.info(
                    f"PPC {pointing_index - 1:4d}: {len(assignment_map):.0f}/2394={ppc_fiber_usage_frac:.2f}% assigned Cobras"
                )

                assigned_target_ids = [
                    netflow_targets[target_idx].ID for target_idx in assignment_map
                ]

                if len(assigned_target_ids) == 0:
                    # priority is nan if no target is assigned
                    ppc_priority = np.nan
                else:
                    # priority is the sum of the rank_fin of the assigned targets
                    ppc_priority = float(
                        np.sum(
                            tb_tgt["rank_fin"][
                                np.isin(tb_tgt["identify_code"], assigned_target_ids)
                            ]
                        )
                    )

                ppc_records.append(
                    [
                        telescope._ra,
                        telescope._dec,
                        telescope._posang,
                        ppc_priority,
                        ppc_fiber_usage_frac,
                        assigned_target_ids,
                        resolution,
                    ]
                )

        # Skip if no PPC has any assigned targets
        if len(ppc_records) == 0:
            logger.warning("[S3] Netflow returned no PPC allocations")
            return []

        # Convert the PPC records into the output table format, sorting by ppc_priority
        tb_ppc_netflow = Table(
            np.array(ppc_records, dtype=object),
            names=[
                "ppc_ra",
                "ppc_dec",
                "ppc_pa",
                "ppc_priority_usr",
                "ppc_fiber_usage_frac",
                "ppc_allocated_targets",
                "ppc_resolution",
            ],
            dtype=[
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                object,
                np.str_,
            ],
        )

        # Sort the PPCs by their priority (sum of rank_fin of assigned targets) in descending order, so that the PPC with the highest priority lists first in the output 
        tb_ppc_netflow = tb_ppc_netflow[
            np.argsort(np.asarray(tb_ppc_netflow["ppc_priority_usr"], dtype=float))[::-1]
        ]

        # Set ppc_code
        if backup:
            tb_ppc_netflow["ppc_code"] = [
                f"que_{resolution}_{today_date}_{index + 1}_backup"
                for index in range(len(tb_ppc_netflow))
            ]
        else:
            tb_ppc_netflow["ppc_code"] = [
                f"que_{resolution}_{today_date}_{index + 1}"
                for index in range(len(tb_ppc_netflow))
            ]

        tb_ppc_netflow["ppc_priority"] = np.arange(1, len(tb_ppc_netflow)+1, dtype=int)

        logger.info(
            f"[S3] Run netflow done (takes {round(time.time() - time_start, 3)} sec)"
        )

        return tb_ppc_netflow


def check_netflow_assign_exptime(tb_tgt, tb_ppc_netflow):
    # Update assigned exposure time from the PPC netflow allocation result.

    # Skip if no PPC has any assigned targets
    if len(tb_ppc_netflow) == 0:
        return tb_tgt

    tb_tgt["exptime_assign"] = 0
    single_exptime = int(tb_tgt.meta["single_exptime"])
    target_index_by_id = {
        identify_code: index
        for index, identify_code in enumerate(np.asarray(tb_tgt["identify_code"], dtype=str))
    }

    # For each PPC, add the single_exptime to the assigned targets in tb_tgt based on the netflow allocation result
    for ppc_row in tb_ppc_netflow:
        assigned_indices = [
            target_index_by_id[target_id]
            for target_id in ppc_row["ppc_allocated_targets"]
            if target_id in target_index_by_id
        ]
        if assigned_indices:
            tb_tgt["exptime_assign"].data[assigned_indices] += single_exptime

    return tb_tgt


def export_output_tables(tb_ppc, tb_tgt, output_dir="output/", backup=False):
    """Write PPC and target export tables to disk.

    Parameters
    ==========
    tb_ppc : astropy.table.Table
        Final PPC table to export.
    tb_tgt : astropy.table.Table
        Final target table to export.
    output_dir : str
        Output directory path.
    backup : bool
        Whether to write backup filenames.
    """
    # The PPC table already matches the export schema, so write a shallow copy
    # directly instead of rebuilding the table column-by-column.
    tb_ppc_export = tb_ppc.copy(copy_data=True)
    tb_ppc_export.write(
        os.path.join(output_dir, "ppcList_all.ecsv"),
        format="ascii.ecsv",
        overwrite=True,
    )

    # Build the OB export table from a compact source->destination column map
    # to keep the schema explicit while avoiding repetitive boilerplate.
    target_column_map = [
        ("ob_code", "ob_code"),
        ("obj_id", "ob_obj_id"),
        ("input_catalog_id", "ob_cat_id"),
        ("ra", "ob_ra"),
        ("dec", "ob_dec"),
        (None, "ob_equinox"),
        ("pmra", "ob_pmra"),
        ("pmdec", "ob_pmdec"),
        ("parallax", "ob_parallax"),
        ("priority", "ob_priority"),
        ("exptime", "ob_exptime"),
        ("exptime_usr", "ob_exptime_usr"),
        ("single_exptime", "ob_single_exptime"),
        ("resolution", "ob_resolution"),
        ("proposal_id", "proposal_id"),
        ("rank", "proposal_rank"),
        ("allocated_time_tac", "allocated_time_tac"),
        ("qa_reference_arm", "qa_reference_arm"),
        ("rank_fin", "ob_weight_best"),
        ("exptime_assign", "ob_exptime_assign"),
        ("filter_g", "ob_filter_g"),
        ("filter_r", "ob_filter_r"),
        ("filter_i", "ob_filter_i"),
        ("filter_z", "ob_filter_z"),
        ("filter_y", "ob_filter_y"),
        ("psf_flux_g", "ob_psf_flux_g"),
        ("psf_flux_r", "ob_psf_flux_r"),
        ("psf_flux_i", "ob_psf_flux_i"),
        ("psf_flux_z", "ob_psf_flux_z"),
        ("psf_flux_y", "ob_psf_flux_y"),
        ("psf_flux_error_g", "ob_psf_flux_error_g"),
        ("psf_flux_error_r", "ob_psf_flux_error_r"),
        ("psf_flux_error_i", "ob_psf_flux_error_i"),
        ("psf_flux_error_z", "ob_psf_flux_error_z"),
        ("psf_flux_error_y", "ob_psf_flux_error_y"),
        ("total_flux_g", "ob_total_flux_g"),
        ("total_flux_r", "ob_total_flux_r"),
        ("total_flux_i", "ob_total_flux_i"),
        ("total_flux_z", "ob_total_flux_z"),
        ("total_flux_y", "ob_total_flux_y"),
        ("total_flux_error_g", "ob_total_flux_error_g"),
        ("total_flux_error_r", "ob_total_flux_error_r"),
        ("total_flux_error_i", "ob_total_flux_error_i"),
        ("total_flux_error_z", "ob_total_flux_error_z"),
        ("total_flux_error_y", "ob_total_flux_error_y"),
        ("identify_code", "ob_identify_code"),
    ]

    target_export_columns = []
    target_export_names = []
    for source_name, export_name in target_column_map:
        if source_name is None:
            target_export_columns.append(["J2000"] * len(tb_tgt))
        else:
            target_export_columns.append(tb_tgt[source_name].data)
        target_export_names.append(export_name)

    tb_tgt_export = Table(target_export_columns, names=target_export_names)

    if not backup:
        tb_tgt_export.write(
            os.path.join(output_dir, "obList.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )
        np.save(os.path.join(output_dir, "obj_allo_tot.npy"), tb_ppc)
    else:
        tb_tgt_export.write(
            os.path.join(output_dir, "obList_backup.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )
        np.save(os.path.join(output_dir, "obj_allo_tot_backup.npy"), tb_ppc)


def _run_for_resolution(
    tb_tgt_resolution,
    n_ppc,
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    backup=False,
):
    # Run the PPC-center search and the final netflow allocation for one resolution.
    ppc_records, tb_ppc_list = PPP_centers(tb_tgt_resolution, n_ppc, backup=backup)

    tb_tgt_netflow = Table.copy(tb_tgt_resolution)
    tb_tgt_netflow.meta["PPC"] = ppc_records
    tb_tgt_netflow = rank_recalculate(tb_tgt_netflow)

    tb_ppc_netflow = fiber_allocate(
        tb_tgt_netflow,
        num_reserved_fibers=num_reserved_fibers,
        fiber_non_allocation_cost=fiber_non_allocation_cost,
        backup=backup,
    )
    tb_tgt_final = check_netflow_assign_exptime(tb_tgt_netflow, tb_ppc_netflow)

    return tb_ppc_list, tb_tgt_final


def _combine_resolution_outputs(
    n_ppc_l,
    n_ppc_m,
    tb_ppc_l,
    tb_ppc_m,
    tb_tgt_l,
    tb_tgt_m,
    tb_tgt_l_source,
    tb_tgt_m_source,
):
    # Merge LR and MR outputs while preserving the existing warning behavior.
    if n_ppc_l > 0:
        if n_ppc_m > 0:
            return vstack([tb_ppc_l, tb_ppc_m]), vstack([tb_tgt_l, tb_tgt_m])

        if len(tb_tgt_m_source) > 0:
            logger.warning("no allocated time for MR")
        return tb_ppc_l.copy(), tb_tgt_l.copy()

    if n_ppc_m > 0:
        if len(tb_tgt_l_source) > 0:
            logger.warning("no allocated time for LR")
        return tb_ppc_m.copy(), tb_tgt_m.copy()

    raise ValueError("Please specify n_pcc_l or n_pcc_m")


def run(
    bench_model,
    read_target_config,
    n_ppc_l,
    n_ppc_m,
    output_dir="output/",
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    cobra_feature_flag=True,
    backup=False,
    config=None,
    **legacy_kwargs,
):
    """Run the queue PPC workflow and write the final export tables.

    Legacy keyword aliases are still accepted for external callers:
    ``dirName``, ``numReservedFibers``, ``fiberNonAllocationCost``, and ``conf``.
    """
    if "dirName" in legacy_kwargs:
        output_dir = legacy_kwargs.pop("dirName")
    if "numReservedFibers" in legacy_kwargs:
        num_reserved_fibers = legacy_kwargs.pop("numReservedFibers")
    if "fiberNonAllocationCost" in legacy_kwargs:
        fiber_non_allocation_cost = legacy_kwargs.pop("fiberNonAllocationCost")
    if "conf" in legacy_kwargs:
        config = legacy_kwargs.pop("conf")
    if legacy_kwargs:
        unexpected_keys = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected_keys}")

    global bench
    bench = bench_model
    if config is None:
        raise ValueError("config must not be None")

    # Read queueDB exposure history first so target ingestion can account for
    # already completed exposures and remaining proposal time.
    today = date.today().strftime("%Y%m%d")
    tb_queuedb_filename = os.path.join(output_dir, f"tgt_queueDB_{today}.csv")
    proposal_ids = config["ppp"]["proposalIds"] + config["ppp"]["proposalIds_backup"]
    tb_queuedb = query_queueDB(
        proposal_ids,
        config["queuedb"]["filepath"],
        tb_queuedb_filename,
    )

    # Read and split the full target sample into LR and MR subsets.
    tb_tgt, tb_tgt_l, tb_tgt_m, tb_queuedb = read_target(
        read_target_config["mode_readtgt"],
        read_target_config["para_readtgt"],
        tb_queuedb,
    )

    for tb_tgt_current in (tb_tgt, tb_tgt_l, tb_tgt_m):
        tb_tgt_current.meta["cobra_feature_flag"] = cobra_feature_flag

    # Run the same PPC+netflow pipeline independently for LR and MR.
    tb_ppc_list_l, tb_tgt_l_final = _run_for_resolution(
        tb_tgt_l,
        n_ppc_l,
        num_reserved_fibers=num_reserved_fibers,
        fiber_non_allocation_cost=fiber_non_allocation_cost,
        backup=backup,
    )
    tb_ppc_list_m, tb_tgt_m_final = _run_for_resolution(
        tb_tgt_m,
        n_ppc_m,
        num_reserved_fibers=num_reserved_fibers,
        fiber_non_allocation_cost=fiber_non_allocation_cost,
        backup=backup,
    )

    tb_ppc_tot, tb_tgt_tot = _combine_resolution_outputs(
        n_ppc_l,
        n_ppc_m,
        tb_ppc_list_l,
        tb_ppc_list_m,
        tb_tgt_l_final,
        tb_tgt_m_final,
        tb_tgt_l,
        tb_tgt_m,
    )

    # Keep the legacy standalone PPC export in addition to the consolidated export helper.
    if not backup:
        tb_ppc_tot.write(
            os.path.join(output_dir, "ppcList.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )
    else:
        tb_ppc_tot.write(
            os.path.join(output_dir, "ppcList_backup.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )

    export_output_tables(tb_ppc_tot, tb_tgt_tot, output_dir=output_dir, backup=backup)
