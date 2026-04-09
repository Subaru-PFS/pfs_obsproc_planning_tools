#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
from astropy.table import Table
from ginga.misc.log import get_logger
from loguru import logger
from qplan import q_db, q_query

logger_qplan = get_logger("qplan_test", null=True)

_TARGET_DB_FLUX_COLUMNS = [
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

_TARGET_DB_FILTER_COLUMNS = ["filter_g", "filter_r", "filter_i", "filter_z", "filter_y"]


def database_info(para_db):
    dialect, user, pwd, host, port, dbname = para_db
    return "{0}://{1}:{2}@{3}:{4}/{5}".format(dialect, user, pwd, host, port, dbname)


def _remove_tgt_duplicate(df):
    num1 = len(df)
    df = df.drop_duplicates(
        subset=["proposal_id", "obj_id", "input_catalog_id", "resolution"],
        inplace=False,
        ignore_index=True,
    )
    num2 = len(df)
    logger.info(f"Duplication removed: {num1} --> {num2}")
    return df


def query_target_from_db(tgt_db, proposal_ids):
    import sqlalchemy as sa

    if isinstance(proposal_ids, str):
        proposal_ids = [proposal_ids]

    sql = sa.text(
        "SELECT ob_code, obj_id, c.input_catalog_id AS input_catalog_id, ra, dec, epoch, priority, pmra, pmdec, parallax, effective_exptime, single_exptime, qa_reference_arm, is_medium_resolution, proposal.proposal_id AS proposal_id, rank, grade, allocated_time_lr+allocated_time_mr AS allocated_time_tac, allocated_time_lr, allocated_time_mr, filter_g, filter_r, filter_i, filter_z, filter_y, psf_flux_g, psf_flux_r, psf_flux_i, psf_flux_z, psf_flux_y, psf_flux_error_g, psf_flux_error_r, psf_flux_error_i, psf_flux_error_z, psf_flux_error_y, total_flux_g, total_flux_r, total_flux_i, total_flux_z, total_flux_y, total_flux_error_g, total_flux_error_r, total_flux_error_i, total_flux_error_z, total_flux_error_y FROM target JOIN proposal ON target.proposal_id=proposal.proposal_id JOIN input_catalog AS c ON target.input_catalog_id = c.input_catalog_id WHERE proposal.proposal_id = :proposal_id AND c.active;"
    )

    query_rows = []
    with tgt_db.connect() as conn:
        for proposal_id in proposal_ids:
            result = conn.execute(sql, {"proposal_id": proposal_id})
            query_rows.extend(result.mappings().all())

    df_tgt = pd.DataFrame(query_rows)
    if len(df_tgt) == 0:
        return Table()

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
    df_tgt = _remove_tgt_duplicate(df_tgt)
    df_tgt[_TARGET_DB_FLUX_COLUMNS] = df_tgt[_TARGET_DB_FLUX_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    tb_tgt_from_db = Table.from_pandas(df_tgt)
    for column in _TARGET_DB_FILTER_COLUMNS:
        tb_tgt_from_db[column] = tb_tgt_from_db[column].astype("str")

    return tb_tgt_from_db


def query_user_ppc_from_db(tgt_db, proposal_ids):
    import sqlalchemy as sa

    if isinstance(proposal_ids, str):
        proposal_ids = [proposal_ids]

    proposal_ids = [proposal_id for proposal_id in proposal_ids if proposal_id]
    if len(proposal_ids) == 0:
        return Table()

    sql = sa.text(
        """
        SELECT
            up.user_pointing_id,
            up.ppc_code,
            up.ppc_ra,
            up.ppc_dec,
            up.ppc_pa,
            up.ppc_resolution,
            up.ppc_priority,
            up.input_catalog_id
        FROM user_pointing up
        JOIN input_catalog ic ON up.input_catalog_id = ic.input_catalog_id
        JOIN target t ON t.input_catalog_id = ic.input_catalog_id
        WHERE ic.active = TRUE
          AND ic.is_classical = TRUE
          AND ic.is_user_pointing = TRUE
          AND t.proposal_id IN :proposal_ids
        """
    ).bindparams(sa.bindparam("proposal_ids", expanding=True))

    with tgt_db.connect() as conn:
        rows = conn.execute(sql, {"proposal_ids": proposal_ids}).mappings().all()

    if len(rows) == 0:
        return Table()

    unique_rows = []
    seen_keys = set()
    for row in rows:
        key = (
            row["ppc_ra"],
            row["ppc_dec"],
            row["ppc_pa"],
            row["ppc_resolution"],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_rows.append(row)

    return Table(
        rows=[tuple(row.values()) for row in unique_rows],
        names=list(unique_rows[0].keys()),
    )


def query_queueDB(psl_id_list, DBPath_qDB, tb_queuedb_filename):
    if os.path.exists(tb_queuedb_filename):
        try:
            tb_queuedb = Table.read(tb_queuedb_filename)
            logger.info(f"Loaded cached queue table: {tb_queuedb_filename}")
            return tb_queuedb
        except Exception:
            logger.info("[S1] Querying the qdb (no cache found)")

    qdb = q_db.QueueDatabase(logger_qplan)
    qdb.read_config(DBPath_qDB)
    qdb.connect()
    qa = q_db.QueueAdapter(qdb)
    qq = q_query.QueueQuery(qa, use_cache=False)

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
            exptime_selected = ex_ob_stats.cum_eff_exp_time

            if exptime_selected >= 0:
                counter += 1
                results.append(
                    [
                        counter,
                        psl_id,
                        ex_ob.ob_key[1],
                        arm,
                        exptime_selected,
                        exptime_b,
                        exptime_r,
                        exptime_m,
                        exptime_n,
                        len(ex_ob.exp_history) * 450.0,
                    ]
                )

    if not results:
        logger.warning("No executed observations found in any proposal.")
        return Table()

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
