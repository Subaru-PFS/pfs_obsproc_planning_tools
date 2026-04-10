#!/usr/bin/env python3

import numpy as np
from astropy.table import Table

from .run_PPP import (
    PPP_centers,
    PPP_centers_for_single_program,
    _DEFAULT_PPP_WEIGHT_PARAMS,
    _prepare_tb_tgt_for_ppc,
)
from .build_target import read_target_classic
from .classic_for_single_proposal import (
    _get_proposal_policy,
    _get_single_program_mode,
    _get_single_program_ppc_pa,
)
from .run_for_queue import _combine_resolution_outputs, export_output_tables
from .run_netflow import (
    check_netflow_assign_exptime,
    fiber_allocation_classic,
    optimize_non_observation_costs,
    set_bench,
)

from loguru import logger


def _run_classic_for_resolution(
    tb_tgt_resolution,
    n_ppc,
    *,
    proposal_id=None,
    use_single_centers=False,
    use_multiprocessing=True,
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    optimize_costs=False,
    resolution_label=None,
    output_dir=None,
):
    tb_tgt_resolution = _prepare_tb_tgt_for_ppc(
        tb_tgt_resolution,
        _DEFAULT_PPP_WEIGHT_PARAMS,
    )

    user_ppc_list = tb_tgt_resolution.meta.get("PPC")
    use_user_ppc = (
        tb_tgt_resolution.meta.get("PPC_origin") == "usr"
        and user_ppc_list is not None
        and len(user_ppc_list) > 0
    )

    if use_user_ppc:
        ppc_list = user_ppc_list
        if resolution_label is not None:
            logger.info(f"[S2] {resolution_label} using user-provided PPC list.")
    elif use_single_centers:
        fixed_ppc_pa = _get_single_program_ppc_pa(proposal_id)
        ppc_list = PPP_centers_for_single_program(
            tb_tgt_resolution,
            n_ppc,
            fixed_ppc_pa=fixed_ppc_pa,
            output_dir=output_dir,
        )
    else:
        ppc_list = PPP_centers(
            tb_tgt_resolution,
            n_ppc,
            use_multiprocessing=use_multiprocessing,
        )

    tb_tgt_netflow = Table.copy(tb_tgt_resolution)
    tb_tgt_netflow.meta["PPC"] = ppc_list

    if optimize_costs:
        best_classdict, best_metrics = optimize_non_observation_costs(
            tb_tgt_netflow,
            ppc_list,
        )
        if resolution_label is not None:
            logger.info(f"[S3] {resolution_label} cost optimization: {best_metrics}")
    else:
        best_classdict = None

    tb_ppc_netflow = fiber_allocation_classic(
        tb_tgt_netflow,
        num_reserved_fibers=num_reserved_fibers,
        fiber_non_allocation_cost=fiber_non_allocation_cost,
        classdict_override=best_classdict,
    )
    tb_tgt_final = check_netflow_assign_exptime(tb_tgt_netflow, tb_ppc_netflow)

    return tb_ppc_netflow, tb_tgt_final


def _export_classic_outputs(tb_ppc, tb_tgt, output_dir):
    tb_tgt_export = tb_tgt.copy(copy_data=True)
    if "proposal_id" in tb_tgt_export.colnames:
        for proposal_id in sorted(set(tb_tgt_export["proposal_id"])):
            proposal_policy = _get_proposal_policy(str(proposal_id))
            for input_catalog_id, priority_offset in proposal_policy.get(
                "export_priority_restore", {}
            ).items():
                priority_restore_mask = (
                    (tb_tgt_export["proposal_id"] == proposal_id)
                    & (tb_tgt_export["input_catalog_id"] == input_catalog_id)
                )
                if np.any(priority_restore_mask):
                    tb_tgt_export["priority"][priority_restore_mask] -= priority_offset

    export_output_tables(
        tb_ppc,
        tb_tgt_export,
        output_dir=output_dir,
    )


def run(
    bench_info,
    readtgt_con,
    nppc_l,
    nppc_m,
    output_dir="output/",
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    optimize_costs=False,
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
    legacy_kwargs.pop("cobra_feature_flag", None)
    legacy_kwargs.pop("backup", None)
    if legacy_kwargs:
        unexpected_keys = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected_keys}")

    set_bench(bench_info)

    tb_tgt, tb_tgt_l, tb_tgt_m = read_target_classic(
        readtgt_con["mode_readtgt"], readtgt_con["para_readtgt"]
    )

    multi_process = True

    unique_proposal_ids = set(tb_tgt["proposal_id"])
    single_proposal_id = next(iter(unique_proposal_ids), None)
    single_program_mode = (
        _get_single_program_mode(single_proposal_id)
        if len(unique_proposal_ids) == 1
        else None
    )

    use_single_lr = single_program_mode == "LR"
    use_single_mr = single_program_mode == "MR"
    nppc_l_effective = nppc_l if not use_single_mr else 0
    nppc_m_effective = nppc_m if not use_single_lr else 0

    tb_ppc_l_fin = Table()
    tb_tgt_l_fin = Table()
    if nppc_l_effective > 0:
        tb_ppc_l_fin, tb_tgt_l_fin = _run_classic_for_resolution(
            tb_tgt_l,
            nppc_l_effective,
            proposal_id=single_proposal_id,
            use_single_centers=use_single_lr,
            use_multiprocessing=multi_process,
            num_reserved_fibers=num_reserved_fibers,
            fiber_non_allocation_cost=fiber_non_allocation_cost,
            optimize_costs=optimize_costs,
            resolution_label="LR",
            output_dir=output_dir,
        )

    tb_ppc_m_fin = Table()
    tb_tgt_m_fin = Table()
    if nppc_m_effective > 0:
        tb_ppc_m_fin, tb_tgt_m_fin = _run_classic_for_resolution(
            tb_tgt_m,
            nppc_m_effective,
            proposal_id=single_proposal_id,
            use_single_centers=use_single_mr,
            use_multiprocessing=multi_process,
            num_reserved_fibers=num_reserved_fibers,
            fiber_non_allocation_cost=fiber_non_allocation_cost,
            optimize_costs=optimize_costs,
            resolution_label="MR",
            output_dir=output_dir,
        )

    tb_ppc_tot, tb_tgt_tot = _combine_resolution_outputs(
        nppc_l_effective,
        nppc_m_effective,
        tb_ppc_l_fin,
        tb_ppc_m_fin,
        tb_tgt_l_fin,
        tb_tgt_m_fin,
        tb_tgt_l,
        tb_tgt_m,
    )

    _export_classic_outputs(
        tb_ppc_tot,
        tb_tgt_tot,
        output_dir,
    )
