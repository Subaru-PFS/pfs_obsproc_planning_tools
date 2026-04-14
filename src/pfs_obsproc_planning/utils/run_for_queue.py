#!/usr/bin/env python3

import os
from datetime import date

import numpy as np
from astropy.table import Table, vstack
from loguru import logger

from .run_PPP import PPP_centers, rank_recalculate
from .build_target import read_target_queue
from .db_query import query_queueDB
from .run_netflow import (
    check_netflow_assign_exptime,
    fiber_allocate,
    set_bench,
)


_PPC_EXPORT_COLUMNS = [
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
]


def _normalize_ppc_export_table(tb_ppc, tb_tgt=None):
    tb_ppc_export = tb_ppc.copy(copy_data=True)
    n_rows = len(tb_ppc_export)
    is_classic = (
        "ppc_code" in tb_ppc_export.colnames
        and n_rows > 0
        and all(str(code).startswith("cla_") for code in tb_ppc_export["ppc_code"])
    )

    default_columns = {
        "ppc_equinox": np.array(["J2000"] * n_rows, dtype=np.str_),
        "ppc_exptime": np.full(n_rows, 900.0, dtype=float),
        "ppc_totaltime": np.full(n_rows, 1200.0, dtype=float),
        "ppc_comment": np.array([""] * n_rows, dtype=np.str_),
        "ppc_allocated_targets": np.array([[] for _ in range(n_rows)], dtype=object),
    }

    if is_classic and "ppc_exptime" not in tb_ppc_export.colnames:
        default_single_exptime = None
        if tb_tgt is not None and hasattr(tb_tgt, "meta"):
            default_single_exptime = tb_tgt.meta.get("single_exptime")
        if default_single_exptime is not None:
            default_columns["ppc_exptime"] = np.full(
                n_rows, float(default_single_exptime), dtype=float
            )
            default_columns["ppc_totaltime"] = np.full(
                n_rows, float(default_single_exptime) + 300.0, dtype=float
            )

    if "ppc_priority_usr" not in tb_ppc_export.colnames:
        if "ppc_priority" in tb_ppc_export.colnames:
            tb_ppc_export["ppc_priority_usr"] = np.asarray(
                tb_ppc_export["ppc_priority"], dtype=float
            )
        else:
            tb_ppc_export["ppc_priority_usr"] = np.arange(1, n_rows + 1, dtype=float)

    if "ppc_priority" not in tb_ppc_export.colnames:
        tb_ppc_export["ppc_priority"] = np.arange(1, n_rows + 1, dtype=float)

    for column_name, default_value in default_columns.items():
        if column_name not in tb_ppc_export.colnames:
            tb_ppc_export[column_name] = default_value

    return tb_ppc_export[_PPC_EXPORT_COLUMNS]


def export_output_tables(tb_ppc, tb_tgt, output_dir="output/", backup=False):
    tb_ppc_export = _normalize_ppc_export_table(tb_ppc, tb_tgt=tb_tgt)
    tb_ppc_export.write(
        os.path.join(output_dir, "ppcList_all.ecsv"),
        format="ascii.ecsv",
        overwrite=True,
    )
    if not backup:
        tb_ppc_export.write(
            os.path.join(output_dir, "ppcList.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )
    else:
        tb_ppc_export.write(
            os.path.join(output_dir, "ppcList_backup.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )

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

    if config is None:
        raise ValueError("config must not be None")

    set_bench(bench_model)

    today = date.today().strftime("%Y%m%d")
    tb_queuedb_filename = os.path.join(output_dir, f"tgt_queueDB_{today}.csv")
    proposal_ids = config["ppp"]["proposalIds"] + config["ppp"]["proposalIds_backup"]
    tb_queuedb = query_queueDB(
        proposal_ids,
        config["queuedb"]["filepath"],
        tb_queuedb_filename,
    )

    tb_tgt, tb_tgt_l, tb_tgt_m, tb_queuedb = read_target_queue(
        read_target_config["mode_readtgt"],
        read_target_config["para_readtgt"],
        tb_queuedb,
    )

    for tb_tgt_current in (tb_tgt, tb_tgt_l, tb_tgt_m):
        tb_tgt_current.meta["cobra_feature_flag"] = cobra_feature_flag

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

    export_output_tables(tb_ppc_tot, tb_tgt_tot, output_dir=output_dir, backup=backup)
