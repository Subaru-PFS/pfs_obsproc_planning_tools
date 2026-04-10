#!/usr/bin/env python3

import time
import warnings
from datetime import date, datetime, timedelta

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, solar_system_ephemeris
from astropy.table import Table, join, vstack
from astropy.time import Time
from astroplan import Observer
from dateutil import parser, tz
from ginga.misc.log import get_logger
from loguru import logger
from qplan import entity
from qplan.util.eph_cache import EphemerisCache
from qplan.util.site import site_subaru as observer

from .classic_for_single_proposal import (
    _get_import_user_ppc_from_db,
    apply_proposal_target_adjustments,
)
from .db_query import database_info, query_target_from_db, query_user_ppc_from_db

warnings.filterwarnings("ignore")

logger_qplan = get_logger("qplan_test", null=True)
eph_cache = EphemerisCache(logger_qplan, precision_minutes=5)


def load_raw_target_table(params):
    if len(params["localPath_tgt"]) > 0:
        tb_tgt_raw = Table.read(params["localPath_tgt"])
        logger.info(f"[S1] Target list is read from {params['localPath_tgt']}.")

        if len(tb_tgt_raw) == 0:
            logger.warning("[S1] No input targets.")
            return None

        return tb_tgt_raw

    if None in params["DBPath_tgt"]:
        logger.error("[S1] Incorrect connection info to database is provided.")
        return None

    import sqlalchemy as sa

    db_address = database_info(params["DBPath_tgt"])
    tgt_db = sa.create_engine(db_address)
    try:
        proposal_ids = params["proposalIds"]
        return query_target_from_db(tgt_db, proposal_ids)
    finally:
        tgt_db.dispose()


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
    subaru = EarthLocation.of_site("Subaru Telescope")

    times = Time(
        [
            f"{date} {h:02d}:{m:02d}:00"
            for h in range(time_start_utc, time_end_utc)
            for m in range(0, 60, time_step_min)
        ],
        scale="utc",
    )

    ra_grid = np.arange(0, 360 + ra_step_deg, ra_step_deg)
    dec_grid = np.arange(-30, 90 + dec_step_deg, dec_step_deg)
    ra2d, dec2d = np.meshgrid(ra_grid, dec_grid)

    skygrid = SkyCoord(ra2d * u.deg, dec2d * u.deg)
    visible_all = np.zeros(ra2d.shape, dtype=bool)

    for t in times:
        with solar_system_ephemeris.set("builtin"):
            moon_icrs = get_body("moon", t)

        altaz_frame = AltAz(obstime=t, location=subaru)
        altaz_grid = skygrid.transform_to(altaz_frame)
        moon_altaz = moon_icrs.transform_to(altaz_frame)
        moon_sep = altaz_grid.separation(moon_altaz)

        vis = (
            (altaz_grid.alt > min_alt)
            & (altaz_grid.alt < max_alt)
            & (moon_sep > min_moon_sep)
            & (moon_sep < max_moon_sep)
        )
        visible_all |= vis

    ra_idx = np.round(tb_tgt[ra_col]).astype(int) % int(360 / ra_step_deg)
    dec_idx = np.round((tb_tgt[dec_col] - dec_grid[0]) / dec_step_deg).astype(int)

    valid_idx = (dec_idx >= 0) & (dec_idx < visible_all.shape[0])

    mask_visible = np.zeros(len(tb_tgt), dtype=bool)
    mask_visible[valid_idx] = visible_all[dec_idx[valid_idx], ra_idx[valid_idx]]

    logger.info(f"Visible targets: {len(mask_visible)} -> {sum(mask_visible)}")
    return tb_tgt[mask_visible]


def visibility_checker(tb_tgt, obstimes, start_time_list, stop_time_list):
    tz.gettz("US/Hawaii")

    min_el = 30.0
    max_el = 85.0

    tb_tgt["is_visible"] = False

    for i in range(1):
        target = entity.StaticTarget(
            name=tb_tgt["ob_code"][i],
            ra=tb_tgt["ra"][i],
            dec=tb_tgt["dec"][i],
            equinox=2000.0,
        )
        total_time = np.ceil(tb_tgt["exptime_usr"][i] / tb_tgt["single_exptime"][i]) * (
            tb_tgt["single_exptime"][i] + 300.0
        )

        t_obs_ok = 0
        today = date.today().strftime("%Y-%m-%d")
        date_today = parser.parse(f"{today} 12:00 HST")

        for date_i in obstimes:
            date_t = parser.parse(f"{date_i} 12:00 HST")
            if date_today > date_t:
                continue

            observer.set_date(date_t)
            default_start_time = observer.evening_twilight_18()
            default_stop_time = observer.morning_twilight_18()

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
                    and (parser.parse(f"{item} HST") > default_start_time - timedelta(hours=1))
                ):
                    start_override = default_start_time
                    start_time_list.remove(item)
                    break
                elif (next_date in item) and parser.parse(f"{item} HST") <= default_stop_time:
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
                elif (next_date in item) and parser.parse(f"{item} HST") <= default_stop_time:
                    stop_override = parser.parse(f"{item} HST")
                    stop_time_list.remove(item)
                    break
                elif (
                    (next_date in item)
                    and (parser.parse(f"{item} HST") > default_stop_time)
                    and (parser.parse(f"{item} HST") <= default_stop_time + timedelta(hours=1))
                ):
                    stop_override = default_stop_time
                    stop_time_list.remove(item)
                    break

            start_time = start_override if start_override is not None else default_start_time
            stop_time = stop_override if stop_override is not None else default_stop_time

            if i == 0:
                logger.info(f"date: {date_i}, start={start_time}, stop={stop_time}")

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
                continue

            if t_stop > t_start:
                t_obs_ok += (t_stop - t_start).seconds

        if t_obs_ok >= total_time:
            tb_tgt["is_visible"][i] = True

    logger.info(
        f'{sum(tb_tgt["is_visible"])}/{len(tb_tgt)} are visible during the given obstimes.'
    )

    tb_tgt = tb_tgt[tb_tgt["is_visible"]]
    psl_id = sorted(set(tb_tgt["proposal_id"]))

    for psl_id_ in psl_id:
        tb_tgt_ = tb_tgt[tb_tgt["proposal_id"] == psl_id_]
        if sum(tb_tgt_["exptime_usr"]) / 3600.0 < tb_tgt_["allocated_time_tac"][0]:
            logger.error(
                f"{psl_id_}: visible targets too limited to achieve the allocated FHs. Please change obstime."
            )

    return tb_tgt


def load_user_ppc_table(path_ppc):
    if path_ppc.endswith(".ecsv"):
        return Table.read(path_ppc)

    from glob import glob

    tables = []
    for index, file in enumerate(glob(path_ppc + "*"), start=1):
        tbl = Table.read(file)
        tbl["ppc_code"] = [f"{code}_{index}" for code in tbl["ppc_code"]]
        tables.append(tbl)

    return vstack(tables, join_type="outer")


def build_ppc_meta_array(tb_ppc_tem, resolution=None):
    ppc_rows = []
    for index, row in enumerate(tb_ppc_tem):
        if resolution is not None and row["ppc_resolution"] != resolution:
            continue
        ppc_rows.append(
            [
                index,
                row["ppc_ra"],
                row["ppc_dec"],
                row["ppc_pa"],
                row["ppc_priority"],
            ]
        )

    return np.array(ppc_rows, dtype=object)


def apply_ppc_metadata(tb_tgt, tb_tgt_l, tb_tgt_m, tb_ppc_tem, origin_label):
    if len(tb_ppc_tem) == 0:
        logger.warning(f"[S1] No PPC metadata found from {origin_label}.")
        return

    if "ppc_priority" not in tb_ppc_tem.colnames:
        tb_ppc_tem["ppc_priority"] = 0
    elif np.ma.isMaskedArray(tb_ppc_tem["ppc_priority"]):
        tb_ppc_tem["ppc_priority"] = tb_ppc_tem["ppc_priority"].filled(0)

    logger.info(
        f"[S1] PPC list from {origin_label}:\n{tb_ppc_tem['ppc_code','ppc_ra','ppc_dec','ppc_pa','ppc_priority']}."
    )

    tb_tgt.meta["PPC"] = build_ppc_meta_array(tb_ppc_tem)
    tb_tgt_l.meta["PPC"] = build_ppc_meta_array(tb_ppc_tem, resolution="L")
    tb_tgt_m.meta["PPC"] = build_ppc_meta_array(tb_ppc_tem, resolution="M")

    tb_tgt.meta["PPC_origin"] = "usr"
    tb_tgt_l.meta["PPC_origin"] = "usr"
    tb_tgt_m.meta["PPC_origin"] = "usr"


def apply_user_ppc_metadata(tb_tgt, tb_tgt_l, tb_tgt_m, path_ppc):
    tb_ppc_tem = load_user_ppc_table(path_ppc)
    apply_ppc_metadata(tb_tgt, tb_tgt_l, tb_tgt_m, tb_ppc_tem, "usr")
    logger.info(f"[S1] PPC list is read from {path_ppc}.")


def apply_db_ppc_metadata(tb_tgt, tb_tgt_l, tb_tgt_m, para_db):
    if None in para_db:
        logger.warning("[S1] No local PPC path is given and DB connection info is incomplete.")
        return False

    if "proposal_id" not in tb_tgt.colnames:
        logger.warning("[S1] No local PPC path is given and proposal_id is not available for DB PPC query.")
        return False

    import sqlalchemy as sa

    db_address = database_info(para_db)
    tgt_db = sa.create_engine(db_address)
    try:
        proposal_ids = sorted(set(tb_tgt["proposal_id"].astype(str)))
        tb_ppc_tem = query_user_ppc_from_db(tgt_db, proposal_ids)
    finally:
        tgt_db.dispose()

    apply_ppc_metadata(tb_tgt, tb_tgt_l, tb_tgt_m, tb_ppc_tem, "target DB")
    return True


def read_target_classic(mode, params):
    time_start = time.time()
    logger.info("[S1] Read targets started (PPP)")

    empty_result = (Table(), Table(), Table())
    tb_tgt = load_raw_target_table(params)
    if tb_tgt is None:
        return empty_result

    tb_tgt["ra"] = tb_tgt["ra"].astype(float)
    tb_tgt["dec"] = tb_tgt["dec"].astype(float)
    tb_tgt["ob_code"] = tb_tgt["ob_code"].astype(str)

    if "proposal_id" in tb_tgt.colnames:
        tb_tgt["identify_code"] = np.char.add(
            np.char.add(tb_tgt["proposal_id"].astype(str), "_"),
            tb_tgt["ob_code"].astype(str),
        )
    else:
        tb_tgt["identify_code"] = tb_tgt["ob_code"].astype(str)

    tb_tgt["exptime_assign"] = 0.0
    tb_tgt["exptime_done"] = 0.0

    if "resolution" not in tb_tgt.colnames:
        tb_tgt["resolution"] = np.where(
            tb_tgt["is_medium_resolution"].astype(str) == "True", "M", "L"
        )
    if "exptime" not in tb_tgt.colnames:
        tb_tgt.rename_column("exptime_usr", "exptime")

    if "allocated_time" not in tb_tgt.colnames:
        tb_tgt.rename_column("allocated_time_tac", "allocated_time")

    if np.any(tb_tgt["allocated_time"] < 0):
        tb_tgt["allocated_time"] = np.sum(tb_tgt["exptime"] / 3600.0)

    tb_tgt = apply_proposal_target_adjustments(tb_tgt)
    if tb_tgt is None:
        return empty_result

    logger.info(
        f"[S1] The single exptime is set to {tb_tgt.meta['single_exptime']:.2f} sec."
    )

    tb_tgt.meta["PPC"] = np.array([])
    tb_tgt.meta["PPC_origin"] = "auto"

    if params["visibility_check"]:
        tb_tgt = visibility_checker(
            tb_tgt,
            params["obstimes"],
            params["starttimes"],
            params["stoptimes"],
        )

    tb_tgt_l = tb_tgt[tb_tgt["resolution"] == "L"]
    tb_tgt_m = tb_tgt[tb_tgt["resolution"] == "M"]

    import_user_ppc_from_db = True
    proposal_ids = sorted(set(tb_tgt["proposal_id"].astype(str)))
    if len(proposal_ids) == 1:
        import_user_ppc_from_db = _get_import_user_ppc_from_db(
            proposal_ids[0],
            default=import_user_ppc_from_db,
        )

    if len(params["localPath_ppc"]) > 0:
        apply_user_ppc_metadata(tb_tgt, tb_tgt_l, tb_tgt_m, params["localPath_ppc"])
    elif import_user_ppc_from_db:
        apply_db_ppc_metadata(tb_tgt, tb_tgt_l, tb_tgt_m, params["DBPath_tgt"])
    else:
        logger.info("[S1] DB user PPC import is disabled; PPCs will be determined automatically unless provided locally.")

    if len(tb_tgt.meta.get("PPC", [])) == 0:
        logger.warning("[S1] No PPC is provided, PPCs would be determined automatically.")

    logger.info(f"[S1] observation mode = {mode}")
    logger.info(
        f"[S1] Read targets done (takes {round(time.time()-time_start,3):.2f} sec)."
    )
    logger.info(f"[S1] There are {len(set(tb_tgt['proposal_id'])):.0f} proposals.")
    logger.info(
        f"[S1] n_tgt_low = {len(tb_tgt_l):.0f}, n_tgt_medium = {len(tb_tgt_m):.0f}"
    )

    return tb_tgt, tb_tgt_l, tb_tgt_m


def read_target_queue(mode, para, tb_queuedb):
    time_start = time.time()
    logger.info("[S1] Read targets started (PPP)")

    empty_result = (Table(), Table(), Table(), Table())
    tb_tgt_raw = load_raw_target_table(para)
    if tb_tgt_raw is None:
        return empty_result

    tb_tgt = tb_tgt_raw
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
    tb_tgt["exptime_done"] = 0.0

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
        exptime_done_real = np.ma.filled(tb_tgt["eff_exptime_done_real"], 0.0).astype(float)
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
                mask = (tb_tgt["proposal_id"] == proposal_id) & (tb_tgt["resolution"] == resolution)
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

    tb_tgt["allocated_time"] = tb_tgt["allocated_time_tac"] - tb_tgt["allocated_time_done"]
    tb_tgt["allocated_time"][tb_tgt["allocated_time"] < 0] = 0
    tb_tgt = tb_tgt[tb_tgt["allocated_time"] > 0]

    n_tgt1 = len(tb_tgt)
    tb_tgt = tb_tgt[tb_tgt["exptime"] > 0]
    tb_tgt["exptime_PPP"] = np.ceil(tb_tgt["exptime"] / 900) * 900
    n_tgt2 = len(tb_tgt)
    logger.info(
        f"There are {n_tgt2:.0f} (partial-obs: {sum(tb_tgt['exptime_done'] > 0):.0f}) / {n_tgt1:.0f} targets not completed"
    )

    if para["visibility_check"]:
        tb_tgt = visibility_checker2(tb_tgt)

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
