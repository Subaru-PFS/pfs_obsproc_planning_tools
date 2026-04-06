#!/usr/bin/env python3
# PPP.py : PPP full version

import time
import warnings
from functools import partial

import numpy as np
from astropy import units as u
from astropy.table import Table, join, vstack
from loguru import logger
from scipy.optimize import minimize

try:
    from . import PPP_queue as queue_module
    from .PPP_queue import (
        KDE,
        _calculate_fh_done_by_proposal,
        _prepare_tb_tgt_for_ppc,
        _sample_tb_tgt_rows,
        _select_tb_tgt_remaining,
        PFS_FoV,
        build_classdict as queue_build_classdict,
        check_netflow_assign_exptime,
        count_local_number,
        database_info,
        export_output_tables as queue_export_output_tables,
        fiber_allocate,
        load_raw_target_table,
        rank_recalculate,
        run_netflow,
        target_clustering,
        visibility_checker,
        weight,
    )
except ImportError:
    import PPP_queue as queue_module
    from PPP_queue import (
        KDE,
        _calculate_fh_done_by_proposal,
        _prepare_tb_tgt_for_ppc,
        _sample_tb_tgt_rows,
        _select_tb_tgt_remaining,
        PFS_FoV,
        build_classdict as queue_build_classdict,
        check_netflow_assign_exptime,
        count_local_number,
        database_info,
        export_output_tables as queue_export_output_tables,
        fiber_allocate,
        load_raw_target_table,
        rank_recalculate,
        run_netflow,
        target_clustering,
        visibility_checker,
        weight,
    )

warnings.filterwarnings("ignore")

# netflow configuration (FIXME; should be load from config file)
cobra_location_group = None
min_sky_targets_per_location = None
location_group_penalty = None
cobra_instrument_region = None
min_sky_targets_per_instrument_region = None
instrument_region_penalty = None
black_dot_penalty_cost = None
cobraSafetyMargin = 0.1

_NETFLOW_PRIORITY_COSTS = {
    999: 5e4,
    0: 5e4,
    1: 5e4,
    2: 2e4,
    3: 1e4,
    4: 9e3,
    5: 8e3,
    6: 7e3,
    7: 6e3,
    8: 5e3,
    9: 1e3,
    10: 100,
    11: 90,
    12: 80,
    13: 70,
    14: 60,
    15: 50,
    16: 40,
    17: 30,
    18: 20,
    19: 10,
    20: 5,
    21: 6,
    22: 5,
    23: 4,
    24: 3,
    25: 2.5,
    26: 2,
    27: 1.5,
    28: 1,
    29: 1,
}
_NETFLOW_PARTIAL_OBSERVATION_COST = 5e4
_NETFLOW_CALIBRATION_COST = 2000

np.random.seed(0)


def apply_proposal_target_adjustments(tb_tgt):
    """Apply PPP proposal-specific target adjustments and derived exposure updates."""
    has_proposal_id = "proposal_id" in tb_tgt.colnames
    proposal_ids = sorted(set(tb_tgt["proposal_id"])) if has_proposal_id else []

    is_only_s25a_uh006 = proposal_ids == ["S25A-UH006-B"]
    is_only_s25a_uh041 = proposal_ids == ["S25A-UH041-A"]
    is_only_s25b_te421 = proposal_ids == ["S25B-TE421-K"]

    if has_proposal_id:
        mask_uh016 = tb_tgt["proposal_id"] == "S25B-UH016-A"
        mask_uh022_p0 = (tb_tgt["proposal_id"] == "S25A-UH022-A") & (
            tb_tgt["priority"] == 0
        )
        mask_uh041 = tb_tgt["proposal_id"] == "S25B-UH041-A"
        mask_te007_ra = (tb_tgt["proposal_id"] == "S26A-TE007-G") & (
            tb_tgt["ra"] < 210
        )

        tb_tgt["single_exptime"][mask_uh016] = 27000.0
        tb_tgt["exptime"][mask_uh022_p0] = 12000.0
        tb_tgt["priority"][mask_uh016 & (tb_tgt["input_catalog_id"] == 10289)] += 10

        tb_tgt["exptime"][mask_uh041 & np.isin(tb_tgt["exptime"], [14400.0, 19800.0])] = 450.0
        tb_tgt["exptime"][mask_uh041 & np.isin(tb_tgt["exptime"], [28800.0, 39600.0])] = 900.0
        tb_tgt["exptime"][mask_uh041 & np.isin(tb_tgt["exptime"], [43200.0, 59400.0])] = 1350.0
        tb_tgt["exptime"][mask_te007_ra] = 2700.0

    single_exptime_values = np.unique(tb_tgt["single_exptime"])
    if len(single_exptime_values) > 1:
        logger.error(
            "[S1] Multiple single-exptime are given. Not accepted now (240709)."
        )
        return None

    tb_tgt.meta["single_exptime"] = single_exptime_values[0]
    single_exptime_seconds = tb_tgt.meta["single_exptime"]
    tb_tgt["exptime_PPP"] = (
        np.ceil(tb_tgt["exptime"] / single_exptime_seconds) * single_exptime_seconds
    )

    if is_only_s25a_uh006:
        extra_targets_primary = Table.read(
            "/home/wanqqq/examples/run_2503/S25A-UH006-B/input/Sanders_Extra_PFS_Targets_2025A_rev.csv"
        )
        extra_targets_secondary = Table.read(
            "/home/wanqqq/examples/run_2503/S25A-UH006-B/input/PFS_EDFN_March22_rev.csv"
        )

        extra_targets_primary["ob_code"] = extra_targets_primary["ob_code"].astype("str")
        extra_targets_secondary["ob_code"] = extra_targets_secondary["ob_code"].astype("str")

        for col in ["filter_g", "filter_r", "filter_i", "filter_z", "filter_y"]:
            tb_tgt[col] = tb_tgt[col].astype("str")
            extra_targets_primary[col] = extra_targets_primary[col].astype("str")
            extra_targets_secondary[col] = extra_targets_secondary[col].astype("str")

        for col in [
            "psf_flux_g",
            "psf_flux_r",
            "psf_flux_i",
            "psf_flux_z",
            "psf_flux_y",
        ]:
            tb_tgt[col] = tb_tgt[col].astype(float)
            extra_targets_primary[col] = extra_targets_primary[col].astype(float)
            extra_targets_secondary[col] = extra_targets_secondary[col].astype(float)

        for col in [
            "psf_flux_error_g",
            "psf_flux_error_r",
            "psf_flux_error_i",
            "psf_flux_error_z",
            "psf_flux_error_y",
        ]:
            tb_tgt[col] = tb_tgt[col].astype(float)
            extra_targets_primary[col] = extra_targets_primary[col].astype(float)
            extra_targets_secondary[col] = extra_targets_secondary[col].astype(float)

        tb_tgt = vstack([tb_tgt, extra_targets_primary, extra_targets_secondary])
        tb_tgt["exptime_PPP"] = (
            np.ceil(tb_tgt["exptime"] / single_exptime_seconds) * single_exptime_seconds
        )

        if has_proposal_id:
            tb_tgt["identify_code"] = np.char.add(
                np.char.add(tb_tgt["proposal_id"].astype(str), "_"),
                tb_tgt["ob_code"].astype(str),
            )
        else:
            tb_tgt["identify_code"] = tb_tgt["ob_code"].astype(str)

        tb_tgt["exptime_assign"] = 0.0
        tb_tgt["exptime_done"] = 0.0
        tb_tgt["i2_mag"] = (tb_tgt["psf_flux_i"] * u.nJy).to(u.ABmag)
        tb_tgt["exptime_PPP"][tb_tgt["i2_mag"] < 24] = 900
        tb_tgt["exptime_PPP"][tb_tgt["i2_mag"] >= 24] = 1800

        logger.info(f"Input target list: {tb_tgt}")

    if is_only_s25a_uh041:
        tb_tgt["exptime_PPP"] = 900

    if is_only_s25b_te421:
        tb_tgt["exptime_PPP"] = 900

    return tb_tgt


def load_user_ppc_table(path_ppc):
    """Read one or more user-supplied PPC tables into a single table."""
    if path_ppc.endswith(".ecsv"):
        return Table.read(path_ppc)

    from glob import glob

    tables = []
    for index, file in enumerate(glob(path_ppc + "*"), start=1):
        tbl = Table.read(file)
        tbl["ppc_code"] = [f"{code}_{index}" for code in tbl["ppc_code"]]
        tables.append(tbl)

    return vstack(tables, join_type="outer")


def query_user_ppc_from_db(tgt_db, proposal_ids):
    """Query user-pointing metadata for the given proposal IDs from target DB."""
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


def build_ppc_meta_array(tb_ppc_tem, resolution=None):
    """Convert a PPC table into the metadata array used by PPP."""
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
    """Attach PPC metadata from a table to the target-table metadata."""
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
    """Load user PPCs and attach them to the target-table metadata."""
    tb_ppc_tem = load_user_ppc_table(path_ppc)

    apply_ppc_metadata(tb_tgt, tb_tgt_l, tb_tgt_m, tb_ppc_tem, "usr")

    logger.info(f"[S1] PPC list is read from {path_ppc}.")


def apply_db_ppc_metadata(tb_tgt, tb_tgt_l, tb_tgt_m, para_db):
    """Load user PPC metadata from target DB and attach it to target metadata."""
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

def read_target(mode, params):
    """Read and normalize the PPP target table.

    Parameters
    ----------
    mode : str
        Observation mode label such as ``classic`` or ``queue``.
    params : dict
        PPP target-loading configuration.

    Returns
    -------
    tuple
        Full target table plus low- and medium-resolution subsets.
    """
    time_start = time.time()
    logger.info("[S1] Read targets started (PPP)")

    # Shared constants for early returns and column normalization.
    empty_result = (Table(), Table(), Table())

    # Step 1: load the raw target table from a local file or from the target DB.
    tb_tgt = load_raw_target_table(params)
    if tb_tgt is None:
        return empty_result

    # Step 2: standardize core columns used 
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

    # Step 3: add missing columns needed before proposal-specific adjustments.
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

    # Step 4: apply all proposal-specific target adjustments in one place.
    tb_tgt = apply_proposal_target_adjustments(tb_tgt)
    if tb_tgt is None:
        return empty_result

    logger.info(
        f"[S1] The single exptime is set to {tb_tgt.meta['single_exptime']:.2f} sec."
    )

    tb_tgt.meta["PPC"] = np.array([])
    tb_tgt.meta["PPC_origin"] = "auto"

    # Step 5: check visibility of targets when requested, and remove those that are not visible in the given time windows.
    if params["visibility_check"]:
        tb_tgt = visibility_checker(
            tb_tgt,
            params["obstimes"],
            params["starttimes"],
            params["stoptimes"],
        )

    # Step 6: Split the sample by spectral resolution.
    tb_tgt_l = tb_tgt[tb_tgt["resolution"] == "L"]
    tb_tgt_m = tb_tgt[tb_tgt["resolution"] == "M"]

    # Step 7: Load PPCs from local path first; otherwise try DB.
    if len(params["localPath_ppc"]) > 0:
        apply_user_ppc_metadata(
            tb_tgt,
            tb_tgt_l,
            tb_tgt_m,
            params["localPath_ppc"],
        )
    else:
        apply_db_ppc_metadata(
        tb_tgt,
        tb_tgt_l,
        tb_tgt_m,
        params["DBPath_tgt"],
        )

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


def PPP_centers(
    tb_tgt,
    n_ppc,
    weight_params=(1.5, 0, 0),
    use_multiprocessing=True,
    peak_pa=120.0,
):
    """Determine PPC centers for a single-program PPP run."""
    time_start = time.time()
    rng = np.random.default_rng(0)
    logger.info("[S2] Determine pointing centers started")

    ppc_list = []

    # If PPCs are already provided by the user, keep them as-is.
    if tb_tgt.meta["PPC_origin"] == "usr":
        logger.warning(
            f"[S2] PPCs from usr adopted (takes {round(time.time()-time_start,3):.2f} sec)."
        )
        return tb_tgt.meta["PPC"]

    # Nothing to optimize if no targets are available.
    if len(tb_tgt) == 0:
        logger.warning("[S2] no targets")
        return np.array(ppc_list)

    if n_ppc == 0:
        logger.warning("[S2] no PPC to be determined")
        return np.array(ppc_list)

    # Prepare the target table for PPC determination, and select the remaining targets that still need to be observed.
    tb_tgt = _prepare_tb_tgt_for_ppc(tb_tgt, weight_params)
    single_exptime = tb_tgt.meta["single_exptime"]
    tb_tgt_remaining = _select_tb_tgt_remaining(tb_tgt)
    if len(tb_tgt_remaining) == 0:
        logger.warning("[S2] no remaining targets")
        return np.array(ppc_list)

    proposal_id = str(tb_tgt_remaining["proposal_id"][0])
    fh_goal = float(tb_tgt_remaining["allocated_time"][0])

    while len(tb_tgt_remaining) > 0 and len(ppc_list) < n_ppc:
        fh_done = _calculate_fh_done_by_proposal(tb_tgt).get(proposal_id, 0.0)
        if fh_done >= fh_goal:
            # Break if FH goal has been achieved
            break

        candidate_pointings = []

        # Find KDE peak in each cluster of remaining targets and run netflow to assign targets in the FoV of this peak location; keep those with assigned targets as candidates.
        for tb_tgt_cluster in target_clustering(tb_tgt_remaining, 1.38):
            tb_tgt_cluster_remaining = _select_tb_tgt_remaining(tb_tgt_cluster)
            if len(tb_tgt_cluster_remaining) == 0:
                continue

            # Re-sample to limit the size of targets to run KDE to save time; this may cause some randomness.
            tb_tgt_cluster_sample = _sample_tb_tgt_rows(
                tb_tgt_cluster_remaining, 200, rng
            )

            # Run KDE to find the peak location for this cluster
            _, _, _, peak_x, peak_y = KDE(tb_tgt_cluster_sample, use_multiprocessing)

            # Run netflow to assign targets in the FoV of this peak location
            index_ = PFS_FoV(peak_x, peak_y, peak_pa, tb_tgt_cluster_remaining)
            tb_tgt_cluster_in_fov = tb_tgt_cluster_remaining[list(index_)]
            if len(tb_tgt_cluster_in_fov) == 0:
                continue

            assigned_target_ids = fiber_allocate(
                tb_tgt_cluster_in_fov,
                single_ppc_mode=True,
                ppc_candidate=(peak_x, peak_y, peak_pa),
            )

            # Do a local shift if no target is assigned to this peak location, which may happen when the peak is close to the edge of the FoV
            shift_iter = 0
            while len(assigned_target_ids) == 0 and shift_iter < 2:
                peak_x += rng.uniform(-0.15, 0.15)
                peak_y += rng.uniform(-0.15, 0.15)
                assigned_target_ids = fiber_allocate(
                    tb_tgt_cluster_in_fov,
                    single_ppc_mode=True,
                    ppc_candidate=(peak_x, peak_y, peak_pa),
                    observation_time="2026-03-24T13:00:00Z",
                )
                shift_iter += 1

            if len(assigned_target_ids) == 0:
                continue

            index_assign = np.isin(
                tb_tgt_cluster_remaining["identify_code"], assigned_target_ids
            )
            total_assigned_weight = float(
                np.sum(tb_tgt_cluster_remaining["weight"][index_assign])
            )
            candidate_pointings.append(
                (
                    peak_x,
                    peak_y,
                    peak_pa,
                    total_assigned_weight,
                    assigned_target_ids,
                )
            )

        if len(candidate_pointings) == 0:
            logger.warning(
                "[S2] No valid PPC candidates found; stop automatic PPC determination."
            )
            break

        # Sort candidate pointings by total assigned weight and select the best one
        best_pointing = max(candidate_pointings, key=lambda candidate: candidate[3])

        assigned_mask = np.isin(tb_tgt["identify_code"], best_pointing[4])
        tb_tgt["exptime_PPP"][assigned_mask] -= single_exptime
        tb_tgt["exptime_done"][assigned_mask] += single_exptime

        fh_done = _calculate_fh_done_by_proposal(tb_tgt).get(proposal_id, 0.0)
        ppc_list.append(
            np.array(
                [
                    len(ppc_list),
                    best_pointing[0],
                    best_pointing[1],
                    best_pointing[2],
                    best_pointing[3],
                    fh_done / fh_goal if fh_goal > 0 else 0.0,
                    len(best_pointing[4]) / 2394.0,
                ],
                dtype=object,
            )
        )

        tb_tgt_remaining = _select_tb_tgt_remaining(tb_tgt)
        if len(tb_tgt_remaining) > 0:
            tb_tgt_remaining = _prepare_tb_tgt_for_ppc(tb_tgt_remaining, weight_params)

        print(
            f"PPC_{len(ppc_list):3d}: {len(tb_tgt)-len(tb_tgt_remaining):5d}/{len(tb_tgt):10d} targets are finished (w={best_pointing[3]:.2f}, FH={fh_done:.2f}/{fh_goal:.2f})."
        )

    # Sort the final PPC list by total assigned weight and keep the top n_ppc ones.
    ppc_list_final = sorted(ppc_list, key=lambda x: x[4], reverse=True)[:n_ppc]

    ppc_list_final = np.array(ppc_list_final, dtype=object)

    logger.info(
        f"[S2] Determine pointing centers done ( nppc = {len(ppc_list_final):.0f}; takes {round(time.time()-time_start,3)} sec)"
    )

    return ppc_list_final


def objective_single_program_ppc_assignment(trial_ppc, tb_tgt, ppc_pa=0.0):
    """Objective function used by `PPC_centers_single`.

    This keeps the PPP-specific scoring behavior, including the special
    priority remapping for `S25A-UH006-B`.
    """
    ppc_ra, ppc_dec = trial_ppc
    assigned_target_ids = fiber_allocate(
        tb_tgt,
        single_ppc_mode=True,
        ppc_candidate=(ppc_ra, ppc_dec, ppc_pa),
    )
    assigned_mask = np.isin(tb_tgt["identify_code"], assigned_target_ids)

    proposal_ids = sorted(set(np.asarray(tb_tgt["proposal_id"], dtype=str)))
    if proposal_ids == ["S25A-UH006-B"]:
        tracked_priorities = list(range(20, 30)) + [999]
        emphasized_priority = 21
    else:
        tracked_priorities = list(range(10)) + [999]
        emphasized_priority = 1

    priority_values = np.asarray(tb_tgt["priority"])
    assigned_counts = {}
    total_counts = {}
    for priority in tracked_priorities:
        priority_mask = priority_values == priority
        assigned_counts[priority] = int(np.sum(assigned_mask & priority_mask))
        total_counts[priority] = int(np.sum(priority_mask))

    priority_summary = ", ".join(
        f"N{priority} = {assigned_counts[priority]}/{total_counts[priority]}"
        for priority in tracked_priorities
    )
    print(
        f"{ppc_ra}, {ppc_dec}, {ppc_pa}, "
        f"Nall = {len(assigned_target_ids)}/{len(tb_tgt)}, {priority_summary}"
    )

    score = (
        1.0 * len(assigned_target_ids)
        + 0.5 * assigned_counts[emphasized_priority]
        + 1.50 * assigned_counts[999]
    )
    return -score


def PPC_centers_single(_tb_tgt, n_ppc, weight_para):

    time_start = time.time()
    logger.info("[S2] Determine pointing centers started")

    # _tb_tgt = rank_recalculate(_tb_tgt)
    # _tb_tgt = count_local_number(_tb_tgt)
    # _tb_tgt = weight(_tb_tgt, 1,1,1)

    ppc_lst = []

    para_sci, para_exp, para_n = weight_para
    _tb_tgt = rank_recalculate(_tb_tgt)
    _tb_tgt = count_local_number(_tb_tgt)
    _tb_tgt = weight(_tb_tgt, para_sci, para_exp, para_n)
    single_exptime_ = _tb_tgt.meta["single_exptime"]
    print(single_exptime_)

    _tb_tgt_ = _tb_tgt[_tb_tgt["exptime_PPP"] > 0]
    _tb_tgt_["exptime_done"] = 0.0

    pslID_ = sorted(set(_tb_tgt_["proposal_id"]))

    FH_goal = [
        _tb_tgt_["allocated_time"][_tb_tgt_["proposal_id"] == tt][0] for tt in pslID_
    ]

    tb_fh = Table([pslID_, FH_goal], names=["proposal_id", "FH_goal"])
    tb_fh["FH_done"] = 0.0
    tb_fh["N_done"] = 0.0

    while (len(_tb_tgt_) > 0) and (len(ppc_lst) < n_ppc):
        if list(set(_tb_tgt["proposal_id"])) == ["S25A-UH006-B"]:
            if len(ppc_lst) == 0:
                initial_guess = [150.08189537, 2.18829806, 92.51180584]
            elif len(ppc_lst) == 1:
                initial_guess = [150.08220377, 2.18805709, 92.677787]
            elif len(ppc_lst) == 2:
                initial_guess = [270.0, 66.0, 90.0]

            if len(ppc_lst) == 1:
                fixed_ppc_pa = initial_guess[2]
                result = minimize(
                    partial(
                        objective_single_program_ppc_assignment,
                        tb_tgt=_tb_tgt_,
                        ppc_pa=fixed_ppc_pa,
                    ),
                    initial_guess[:2],
                    method="Nelder-Mead",
                )
                print(result.x)
                ra, dec, pa = result.x[0], result.x[1], fixed_ppc_pa
            elif len(ppc_lst) == 0:
                ra, dec, pa = [150.08220377, 2.18805709, 92.677787]
            elif len(ppc_lst) == 2:
                ra, dec, pa = [270.29782837, 65.7456042, 94.62414553]

        elif list(set(_tb_tgt["proposal_id"])) == ["S25B-UH041-A"]:
            #"""
            if len(ppc_lst) ==0:
                ra, dec, pa = [36.49583333, -4.49444444, 0]
            elif len(ppc_lst) ==1:
                _tb_tgt_ = _tb_tgt_[(_tb_tgt_["priority"] == 999) | (_tb_tgt_["exptime_PPP"] < 57600.0)]
                ra, dec, pa = [36.49583333, -4.49444444, 0]
            elif len(ppc_lst) ==2:
                _tb_tgt_ = _tb_tgt_[(_tb_tgt_["priority"] == 999) | (_tb_tgt_["exptime_PPP"] < 43200.0)]
                ra, dec, pa = [36.49583333, -4.49444444, 0]
            elif len(ppc_lst) ==3:
                _tb_tgt_ = _tb_tgt_[(_tb_tgt_["priority"] == 999) | (_tb_tgt_["exptime_PPP"] < 28800.0)]
                ra, dec, pa = [36.49583333, -4.49444444, 0]
        elif list(set(_tb_tgt["proposal_id"])) == ["S26A-TE007-G"]:
            central_ra, central_dec = 203.879558,59.047015
            pa = 120
                
            def optimize_pointing_objective(coords):
                test_ra, test_dec = coords
                
                # Run netflow to get assigned target IDs
                assigned_ids = fiber_allocate(
                    _tb_tgt_,
                    single_ppc_mode=True,
                    ppc_candidate=(test_ra, test_dec, pa),
                )
                
                # Get indices of assigned targets
                index_assign = np.isin(_tb_tgt_["identify_code"], assigned_ids)
                
                # Count P0-P2 targets (priority 0-2)
                assigned_targets = _tb_tgt_[index_assign]
                n_p0_p2 = np.sum(assigned_targets["priority"] <= 1)
                n_p0_p2_norm = n_p0_p2 / (_tb_tgt_["priority"] <= 1).sum()  # normalized by total P0-P2 targets
                
                # Total allocated targets
                n_total = len(assigned_ids) 
                n_total_norm = len(assigned_ids) / len(_tb_tgt_)  # normalized by total targets/
                
                # Normalized score with minimum as 2
                # Weight P0-P2 targets more heavily
                score = - (5 * n_p0_p2_norm + n_total_norm)
                
                print(f"Testing RA={test_ra:.6f}, Dec={test_dec:.6f}: "
                      f"P0-P2={n_p0_p2}, Total={n_total}, Score={score:.3f}")
                
                return score
            
            # Use L-BFGS-B method which supports bounds
            from scipy.optimize import minimize
            
            initial_guess = [central_ra, central_dec]
            bounds = [(central_ra - 0.1, central_ra + 0.1), 
                     (central_dec - 0.1, central_dec + 0.1)]
            
            print(f"\nOptimizing pointing position around RA={central_ra}, Dec={central_dec}")
            print(f"Search range: ±0.1 degrees")
            
            result = minimize(
                optimize_pointing_objective,
                initial_guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1.0, "eps": 0.01}
            )
            
            # Store optimized position for reuse
            optimized_ra, optimized_dec = result.x
            
            print(f"\nOptimal position found: RA={optimized_ra:.6f}, Dec={optimized_dec:.6f}")
            print(f"Optimization result: {result.message}")
            ra, dec, pa = optimized_ra, optimized_dec, pa
        elif list(set(_tb_tgt["proposal_id"])) == ["S26A-UH022-A"]:
            if len(ppc_lst) ==0:
                ra, dec, pa = 150.029167, 2.195718, 0
            elif len(ppc_lst) ==1:
                _tb_tgt_ = _tb_tgt_[(_tb_tgt_["priority"] == 999) | (_tb_tgt_["exptime_PPP"] < 57600.0)]
                ra, dec, pa = 150.029167, 2.195718, 0
            elif len(ppc_lst) ==2:
                _tb_tgt_ = _tb_tgt_[(_tb_tgt_["priority"] == 999) | (_tb_tgt_["exptime_PPP"] < 43200.0)]
                ra, dec, pa = 150.029167, 2.195718, 0
            elif len(ppc_lst) ==3:
                _tb_tgt_ = _tb_tgt_[(_tb_tgt_["priority"] == 999) | (_tb_tgt_["exptime_PPP"] < 28800.0)]
                ra, dec, pa = 150.029167, 2.195718, 0
            #"""

        elif list(set(_tb_tgt["proposal_id"])) == ["S26A-091"]:
            print("now it is for 091")
            #"""
            # Only optimize once for the first PPC
            if len(ppc_lst) < 2:
                # Local optimization for S26A-UH022-A pointing position
                # Optimize within 0.1 degrees to maximize P0-P2 and total allocated targets
                central_ra, central_dec = 150.119167, 2.205833
                pa = 0
                print("now it is for 091")
                
                def optimize_pointing_objective(coords):
                    test_ra, test_dec = coords
                    
                    # Run netflow to get assigned target IDs
                    assigned_ids = fiber_allocate(
                        _tb_tgt_,
                        single_ppc_mode=True,
                        ppc_candidate=(test_ra, test_dec, pa),
                    )
                    
                    # Get indices of assigned targets
                    index_assign = np.isin(_tb_tgt_["identify_code"], assigned_ids)
                    
                    # Count P0-P2 targets (priority 0-2)
                    assigned_targets = _tb_tgt_[index_assign]
                    n_p0_p2 = np.sum(assigned_targets["priority"] <= 2)
                    n_p0_p2_norm = n_p0_p2 / (_tb_tgt_["priority"] <= 2).sum()  # normalized by total P0-P2 targets
                    
                    # Total allocated targets
                    n_total = len(assigned_ids) 
                    n_total_norm = len(assigned_ids) / len(_tb_tgt_)  # normalized by total targets/
                    
                    # Normalized score with minimum as 2
                    # Weight P0-P2 targets more heavily
                    score = - (5 * n_p0_p2_norm + n_total_norm)
                    
                    print(f"Testing RA={test_ra:.6f}, Dec={test_dec:.6f}: "
                          f"P0-P2={n_p0_p2}, Total={n_total}, Score={score:.3f}")
                    
                    return score
                
                # Use L-BFGS-B method which supports bounds
                #from scipy.optimize import minimize
                
                initial_guess = [central_ra, central_dec]
                bounds = [(central_ra - 1, central_ra + 1), 
                         (central_dec - 0.5, central_dec + 0.5)]
                
                print(f"\nOptimizing pointing position around RA={central_ra}, Dec={central_dec}")
                print(f"Search range: ±0.1 degrees")
                
                result = minimize(
                    optimize_pointing_objective,
                    initial_guess,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"ftol": 1.0, "eps": 0.01}
                )
                
                # Store optimized position for reuse
                optimized_ra, optimized_dec = result.x
                
                print(f"\nOptimal position found: RA={optimized_ra:.6f}, Dec={optimized_dec:.6f}")
                print(f"Optimization result: {result.message}")
                        
            #"""
            if len(ppc_lst) ==0:
                ra, dec, pa = optimized_ra, optimized_dec, 0
            elif len(ppc_lst) ==1:
                ra, dec, pa = optimized_ra, optimized_dec, 0
            #"""

        else:
            tb_tgt_t_group = target_clustering(_tb_tgt_, 1.38)

            _tb_tgt_t = tb_tgt_t_group[0]

            _df_tgt_t = Table.to_pandas(_tb_tgt_t)
            n_tgt = min(200, len(_tb_tgt_t))
            _df_tgt_t = _df_tgt_t.sample(n_tgt, ignore_index=True, random_state=1)
            _tb_tgt_t_1 = Table.from_pandas(_df_tgt_t)

            X_, Y_, obj_dis_sig_, peak_x, peak_y = KDE(_tb_tgt_t_1, False)

            initial_guess = [peak_x, peak_y]  # , 0]
            result = minimize(
                partial(
                    objective_single_program_ppc_assignment,
                    tb_tgt=_tb_tgt_t,
                    ppc_pa=0.0,
                ),
                initial_guess,
                method="Nelder-Mead",
                options={"xatol": 0.01, "fatol": 0.001},
            )
            print(result.x)
            ra, dec = result.x[0], result.x[1]
            pa = 120.0

        lst_tgtID_assign = fiber_allocate(
            _tb_tgt_,
            single_ppc_mode=True,
            ppc_candidate=(ra, dec, pa),
        )
        index_in = PFS_FoV(ra, dec, pa, _tb_tgt_)

        ppc_lst.append(
            np.array([
                len(ppc_lst),
                ra,
                dec,
                pa,
                0,
                lst_tgtID_assign,
                len(lst_tgtID_assign) / 2394.0 * 100.0,
            ], dtype=object)
        )

        index_assign = np.isin(_tb_tgt_["identify_code"], lst_tgtID_assign)

        from collections import Counter
        print(dict(Counter(_tb_tgt_["exptime_PPP"][index_in])))
        print(dict(Counter(_tb_tgt_["exptime_PPP"][index_assign])))

        _tb_tgt_["exptime_PPP"][
            index_assign
        ] -= single_exptime_  # targets in the PPC observed with single_exptime sec

        _tb_tgt_["exptime_done"][index_assign] += single_exptime_

        _tb_tgt_["priority"][_tb_tgt_["exptime_done"] > 0] = 999

        if list(set(_tb_tgt["proposal_id"])) == ["S25A-UH006-B"]:
            _tb_tgt_["priority"][
                (_tb_tgt_["exptime_done"] == 0) & (_tb_tgt_["exptime_PPP"] == 900)
            ] += 20

        _tb_tgt_ = _tb_tgt_[(_tb_tgt_["exptime_PPP"] > 0)]
        print(sum(_tb_tgt_["priority"] == 999))

    ppc_lst_fin = np.array(ppc_lst)

    resol = _tb_tgt["resolution"][0]
    pslid_ = _tb_tgt["proposal_id"][0]
    if pslid_ == "S25A-UH006-B":
        ppc_code = ["cla_L_uh006_" + str(n + 1) for n in np.arange(n_ppc)]
    else:
        ppc_code = [
            f"cla_{resol}_{pslid_.split('-')[1]}_{str(n+1)}" for n in np.arange(n_ppc)
        ]
    ppc_ra = ppc_lst_fin[:, 1]
    ppc_dec = ppc_lst_fin[:, 2]
    ppc_pa = ppc_lst_fin[:, 3]
    ppc_equinox = ["J2000"] * n_ppc
    ppc_priority = [0] * n_ppc
    ppc_priority_usr = [0] * n_ppc
    ppc_exptime = [900.0] * n_ppc
    ppc_totaltime = [1200.0] * n_ppc
    ppc_resolution = [resol] * n_ppc
    ppc_fibAlloFrac = ppc_lst_fin[:, -1]
    ppc_tgtAllo = ppc_lst_fin[:, -2]
    ppc_comment = [""] * n_ppc

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

    # ppcList.write("/home/wanqqq/examples/run_2503/S25A-UH006-B/output/ppp/ppcList.ecsv", format="ascii.ecsv", overwrite=True)
    ppcList.write("/home/wanqqq/workDir_pfs/S26A/run_2603/S26A-TE007-G/output_20260306/ppp/ppcList_1by1.ecsv", format="ascii.ecsv", overwrite=True)

    logger.info(
        f"[S2] Determine pointing centers done ( nppc = {len(ppc_lst_fin):.0f}; takes {round(time.time()-time_start,3)} sec)"
    )

    return ppc_lst_fin

def classic_build_classdict(_tb_tgt):
    classdict = queue_build_classdict()
    classdict.update(
        {
            f"sci_P{priority}": {
                "nonObservationCost": non_observation_cost,
                "partialObservationCost": _NETFLOW_PARTIAL_OBSERVATION_COST,
                "calib": False,
            }
            for priority, non_observation_cost in _NETFLOW_PRIORITY_COSTS.items()
        }
    )
    classdict["cal"] = {
        "numRequired": 0,
        "nonObservationCost": _NETFLOW_CALIBRATION_COST,
        "calib": True,
    }
    classdict["sky"] = {
        "numRequired": 0,
        "nonObservationCost": _NETFLOW_CALIBRATION_COST,
        "calib": True,
    }

    return classdict

def optimize_non_observation_costs(
    _tb_tgt,
    ppc_lst,
    seed=0,
    scale_range=(0.05, 50.0),
    weight_total=1.0,
    weight_p0=3.0,
    weight_p1=2.0,
    focus_max=10,
    otime="2026-03-24T13:00:00Z",
    debug=False,
):
    """Optimize PPP non-observation costs for science priorities.

    The optimizer rescales the base non-observation costs for science
    priorities 0-10, re-runs netflow, and scores the result using normalized
    assignment counts. The PPP/classic class definitions themselves stay the
    same; only their non-observation costs are varied.

    Returns
    -------
    tuple
        ``(best_classdict, best_metrics)`` where ``best_metrics`` includes the
        best score, assignment counts, and fitted scale factors.
    """
    base_classdict = classic_build_classdict(_tb_tgt)
    science_priorities = np.arange(11, dtype=int)
    science_keys = [f"sci_P{priority}" for priority in science_priorities]
    base_non_obs_costs = np.array(
        [base_classdict[key]["nonObservationCost"] for key in science_keys],
        dtype=float,
    )
    partial_cost_caps = np.array(
        [base_classdict[key].get("partialObservationCost", np.inf) for key in science_keys],
        dtype=float,
    )
    scale_group_index = np.array([0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3], dtype=int)

    target_priorities = np.asarray(_tb_tgt["priority"], dtype=int)
    priority_by_code = {
        code: priority
        for code, priority in zip(np.asarray(_tb_tgt["identify_code"], dtype=str), target_priorities)
    }

    total_available = int(np.sum((target_priorities >= 0) & (target_priorities <= focus_max)))
    p0_available = int(np.sum(target_priorities == 0))
    p1_available = int(np.sum(target_priorities == 1))

    def _build_cost_classdict(base_classdict, non_obs_costs):
        """Apply trial non-observation costs to a copy of the base classdict."""
        classdict = {key: value.copy() for key, value in base_classdict.items()}
        for key, cost in non_obs_costs.items():
            if key in classdict:
                classdict[key]["nonObservationCost"] = float(cost)
                if "partialObservationCost" in classdict[key]:
                    classdict[key]["nonObservationCost"] = min(
                        classdict[key]["nonObservationCost"],
                        classdict[key]["partialObservationCost"],
                    )
        return classdict

    def _extract_assigned_ids(res, tgt_lst_netflow):
        """Collect assigned target IDs from one netflow solution."""
        assigned_ids = {
            tgt_lst_netflow[tidx].ID
            for vis in res
            for tidx in vis.keys()
        }
        return assigned_ids

    def _score_assignment_counts(assigned_ids, focus_max=10):
        """Score the assigned targets with cached priority lookups."""
        if not assigned_ids:
            return 0, 0, 0

        assigned_priorities = np.fromiter(
            (priority_by_code[code] for code in assigned_ids if code in priority_by_code),
            dtype=int,
        )
        if assigned_priorities.size == 0:
            return 0, 0, 0

        total = int(np.sum((assigned_priorities >= 0) & (assigned_priorities <= focus_max)))
        p0 = int(np.sum(assigned_priorities == 0))
        p1 = int(np.sum(assigned_priorities == 1))

        return total, p0, p1

    def _sorted_counts(counter_like):
        """Sort priorities numerically while keeping 999 at the end."""
        keys = sorted(key for key in counter_like.keys() if key != 999)
        if 999 in counter_like:
            keys.append(999)
        return {key: counter_like[key] for key in keys}

    def build_costs(scales):
        """Convert four grouped scale factors into per-priority costs."""
        scaled_costs = base_non_obs_costs * np.asarray(scales, dtype=float)[scale_group_index]
        clipped_costs = np.minimum(scaled_costs, partial_cost_caps)
        return {
            key: float(cost)
            for key, cost in zip(science_keys, clipped_costs)
        }

    evaluation_cache = {}

    def evaluate_scales(scales):
        """Run one optimizer evaluation and cache repeated scale vectors."""
        scales = np.asarray(scales, dtype=float)
        cache_key = tuple(scales.tolist())
        if cache_key in evaluation_cache:
            return evaluation_cache[cache_key]

        non_obs_costs = build_costs(scales)
        classdict = _build_cost_classdict(base_classdict, non_obs_costs)

        res, _, tgt_lst_netflow = run_netflow(
            ppc_lst,
            _tb_tgt,
            observation_time=otime,
            classdict_override=classdict,
        )

        assigned_ids = _extract_assigned_ids(res, tgt_lst_netflow)
        total, p0, p1 = _score_assignment_counts(assigned_ids, focus_max=focus_max)
        total_norm = total / total_available if total_available > 0 else 0.0
        p0_norm = p0 / p0_available if p0_available > 0 else 0.0
        p1_norm = p1 / p1_available if p1_available > 0 else 0.0
        score = (
            weight_total * total_norm
            + weight_p0 * p0_norm
            + weight_p1 * p1_norm
        )

        evaluation = {
            "classdict": classdict,
            "non_obs_costs": non_obs_costs,
            "assigned_ids": assigned_ids,
            "total": total,
            "p0": p0,
            "p1": p1,
            "score": score,
        }
        evaluation_cache[cache_key] = evaluation

        if debug:
            from collections import Counter

            logger.info(_sorted_counts(Counter(target_priorities)))
            logger.info(
                _sorted_counts(Counter(priority_by_code[code] for code in assigned_ids))
            )
            logger.info(
                _sorted_counts(
                    {
                        int(key.replace("sci_P", "")): value
                        for key, value in non_obs_costs.items()
                    }
                )
            )

        return evaluation

    def objective(scales):
        return -evaluate_scales(scales)["score"]

    x0 = np.ones(4)
    scale_bounds = [scale_range] * len(x0)
    result = minimize(objective, x0, method="Powell", bounds=scale_bounds)

    best_scales = result.x
    best_eval = evaluate_scales(best_scales)
    best_costs = best_eval["non_obs_costs"]
    best_classdict = best_eval["classdict"]

    best = {
        "score": best_eval["score"],
        "total": best_eval["total"],
        "p0": best_eval["p0"],
        "p1": best_eval["p1"],
        "non_obs_costs": best_costs,
        "scales": best_scales,
        "success": bool(result.success),
        "message": result.message,
    }

    return best_classdict, best


def fiber_allocation_classic(
    tb_tgt,
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    classdict_override=None,
):
    """Run shared fiber allocation and convert results to the classic PPP schema.

    The actual optimization is delegated to `PPP_queue.fiber_allocate`. This
    wrapper keeps only the PPP/classic-specific pieces:
    - classic `ppc_code` naming
    - restoring user PPC priorities when PPCs came from user input
    """

    output_columns = [
        "ppc_code",
        "ppc_ra",
        "ppc_dec",
        "ppc_pa",
        "ppc_priority",
        "ppc_fiber_usage_frac",
        "ppc_allocated_targets",
        "ppc_resolution",
    ]
    output_dtypes = [
        np.str_,
        np.float64,
        np.float64,
        np.float64,
        np.float64,
        np.float64,
        object,
        np.str_,
    ]

    time_start = time.time()
    logger.info("[S3] Run netflow started")

    ppc_list = tb_tgt.meta.get("PPC")
    if ppc_list is None or len(ppc_list) == 0:
        logger.warning("[S3] No PPC has been determined")
        return []

    if len(tb_tgt) == 0:
        logger.warning("[S3] No targets")
        return []

    resolution = tb_tgt["resolution"][0]
    proposal_token = tb_tgt["proposal_id"][0].split("-")[1]

    # Restrict the allocation input to targets that lie in at least one PPC FoV.
    # This keeps the netflow call smaller and avoids repeated table scanning.
    target_indices_in_group = set()
    for ppc_row in ppc_list:
        target_indices_in_group.update(
            PFS_FoV(ppc_row[1], ppc_row[2], ppc_row[3], tb_tgt)
        )

    if len(target_indices_in_group) == 0:
        tb_ppc_netflow = Table(names=output_columns, dtype=output_dtypes)
    else:
        tb_tgt_inuse = tb_tgt[sorted(target_indices_in_group)]

        logger.info(
            f"[S3] nppc = {len(ppc_list):5d}, n_tgt = {len(tb_tgt_inuse):6d}"
        )

        tb_tgt_inuse.meta["PPC"] = ppc_list
        tb_ppc_group = fiber_allocate(
            tb_tgt_inuse,
            num_reserved_fibers=num_reserved_fibers,
            fiber_non_allocation_cost=fiber_non_allocation_cost,
            classdict_override=classdict_override,
        )

        if len(tb_ppc_group) == 0:
            tb_ppc_netflow = Table(names=output_columns, dtype=output_dtypes)
        else:
            tb_ppc_netflow = tb_ppc_group.copy(copy_data=True)
            # Modify ppc_code in classic mode
            tb_ppc_netflow["ppc_code"] = [
                f"cla_{resolution}_{proposal_token}_{pointing_index}"
                for pointing_index in range(1, len(tb_ppc_netflow) + 1)
            ]
            tb_ppc_netflow = tb_ppc_netflow[output_columns]

    # Use user PPC priorities if PPCs came from user input, to preserve any manual adjustments
    if tb_tgt.meta["PPC_origin"] == "usr":
        lst_ppc_usr = tb_tgt.meta["PPC"]
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
        tb_ppc_netflow["ppc_priority"] = tb_ppc_netflow["ppc_priority_usr"]

    logger.info(
        f"[S3] Run netflow done (takes {round(time.time() - time_start, 3)} sec)"
    )

    return tb_ppc_netflow


def _run_classic_for_resolution(
    tb_tgt_resolution,
    ppc_list,
    weight_params,
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    optimize_costs=False,
    random_seed=2,
    resolution_label=None,
):
    tb_tgt_netflow = Table.copy(tb_tgt_resolution)
    tb_tgt_netflow.meta["PPC"] = ppc_list

    tb_tgt_netflow = rank_recalculate(tb_tgt_netflow)
    tb_tgt_netflow = count_local_number(tb_tgt_netflow)
    tb_tgt_netflow = weight(tb_tgt_netflow, *weight_params)

    if optimize_costs:
        best_classdict, best_metrics = optimize_non_observation_costs(
            tb_tgt_netflow,
            ppc_list,
            seed=random_seed,
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


def _combine_classic_outputs(
    nppc_l,
    nppc_m,
    tb_ppc_l,
    tb_ppc_m,
    tb_tgt_l,
    tb_tgt_m,
    tb_tgt_l_source,
    tb_tgt_m_source,
):
    if nppc_l > 0:
        if nppc_m > 0:
            return vstack([tb_ppc_l, tb_ppc_m]), vstack([tb_tgt_l, tb_tgt_m])

        if len(tb_tgt_m_source) > 0:
            logger.warning("no allocated time for MR")
        return tb_ppc_l.copy(), tb_tgt_l.copy()

    if nppc_m > 0:
        if len(tb_tgt_l_source) > 0:
            logger.warning("no allocated time for LR")
        return tb_ppc_m.copy(), tb_tgt_m.copy()

    raise ValueError("Please specify n_pcc_l or n_pcc_m")


def _export_classic_outputs(tb_ppc, tb_tgt, output_dir):
    tb_tgt_export = tb_tgt.copy(copy_data=True)
    special_priority_mask = (
        (tb_tgt_export["proposal_id"] == "S25B-UH016-A")
        & (tb_tgt_export["input_catalog_id"] == 10289)
    )
    if np.any(special_priority_mask):
        tb_tgt_export["priority"][special_priority_mask] -= 10

    queue_export_output_tables(
        tb_ppc,
        tb_tgt_export,
        output_dir=output_dir,
    )


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
    optimize_costs=False,
):
    global bench
    bench = bench_info
    queue_module.bench = bench_info

    tb_tgt, tb_tgt_l, tb_tgt_m = read_target(
        readtgt_con["mode_readtgt"], readtgt_con["para_readtgt"]
    )
    tb_sel_l = tb_tgt_l
    tb_sel_m = tb_tgt_m

    para_sci_l, para_exp_l, para_n_l = [1.5, 0, 0]
    para_sci_m, para_exp_m, para_n_m = [1.5, 0, 0]
    random_seed = 2
    multi_process = True

    unique_proposal_ids = set(tb_tgt["proposal_id"])
    single_proposal_id = next(iter(unique_proposal_ids), None)
    lr_only_proposals = {
        "S25B-116N",
        "S25B-UH041-A",
        "S26A-UH022-A",
        "S26A-TE007-G",
    }
    mr_only_proposals = {"S25B-TE421-K", "S26A-091"}

    if len(unique_proposal_ids) == 1 and single_proposal_id in lr_only_proposals:
        ppc_lst_l = PPC_centers_single(
            tb_sel_l,
            nppc_l,
            [para_sci_l, para_exp_l, para_n_l],
        )
        tb_ppc_tot, tb_tgt_tot = _run_classic_for_resolution(
            tb_tgt_l,
            ppc_lst_l,
            (1, 0, 0),
            num_reserved_fibers=numReservedFibers,
            fiber_non_allocation_cost=fiberNonAllocationCost,
        )
        _export_classic_outputs(tb_ppc_tot, tb_tgt_tot, dirName)
        return None

    if len(unique_proposal_ids) == 1 and single_proposal_id in mr_only_proposals:
        ppc_lst_m = PPC_centers_single(
            tb_sel_m,
            nppc_m,
            [para_sci_m, para_exp_m, para_n_m],
        )
        tb_ppc_tot, tb_tgt_tot = _run_classic_for_resolution(
            tb_tgt_m,
            ppc_lst_m,
            (1, 0, 0),
            num_reserved_fibers=numReservedFibers,
            fiber_non_allocation_cost=fiberNonAllocationCost,
        )
        _export_classic_outputs(tb_ppc_tot, tb_tgt_tot, dirName)
        return None

    ppc_lst_l = PPP_centers(
        tb_sel_l,
        nppc_l,
        [para_sci_l, para_exp_l, para_n_l],
        multi_process,
    )
    tb_ppc_l_fin, tb_tgt_l_fin = _run_classic_for_resolution(
        tb_tgt_l,
        ppc_lst_l,
        (para_sci_l, para_exp_l, para_n_l),
        num_reserved_fibers=numReservedFibers,
        fiber_non_allocation_cost=fiberNonAllocationCost,
        optimize_costs=optimize_costs,
        random_seed=random_seed,
        resolution_label="LR",
    )

    ppc_lst_m = PPP_centers(
        tb_sel_m,
        nppc_m,
        [para_sci_m, para_exp_m, para_n_m],
        multi_process,
    )
    tb_ppc_m_fin, tb_tgt_m_fin = _run_classic_for_resolution(
        tb_tgt_m,
        ppc_lst_m,
        (para_sci_m, para_exp_m, para_n_m),
        num_reserved_fibers=numReservedFibers,
        fiber_non_allocation_cost=fiberNonAllocationCost,
        optimize_costs=optimize_costs,
        random_seed=random_seed,
        resolution_label="MR",
    )

    tb_ppc_tot, tb_tgt_tot = _combine_classic_outputs(
        nppc_l,
        nppc_m,
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
        dirName,
    )

    # CR_tot, CR_tot_, sub_tot = complete_ppc(tb_tgt_tot, "compOFpsl_n")

    # plotCR(CR_tot_, sub_tot, tb_ppc_tot, dirName=dirName, show_plots=show_plots)
