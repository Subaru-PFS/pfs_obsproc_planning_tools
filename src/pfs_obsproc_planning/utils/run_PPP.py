#!/usr/bin/env python3
"""Pointing-priority planning helpers for PPP target selection.

This module contains the core routines used to rank targets, estimate
high-density pointing seeds, optimize PPC centers, and build PPC tables for
both queue and classic single-program planning flows.
"""

import multiprocessing
import os
import time
import warnings
from functools import partial

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from loguru import logger
from matplotlib.path import Path
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

from .classic_for_single_proposal import (
    _DEFAULT_SINGLE_PROGRAM_PRIORITY_POLICY,
    _get_proposal_policy,
)
from .run_netflow import fiber_allocate

warnings.filterwarnings("ignore")
np.random.seed(0)

_DEFAULT_PPP_WEIGHT_PARAMS = (2, 0, 0)


# -----------------------------------------------------------------------------
# Shared target-scoring helpers
# -----------------------------------------------------------------------------


def count_local_number(_tb_tgt):
    """Annotate each target with a coarse local sky-density count."""
    if len(_tb_tgt) == 0:
        return _tb_tgt
    ra_index = np.asarray(_tb_tgt["ra"], dtype=float).astype(int)
    dec_index = (np.asarray(_tb_tgt["dec"], dtype=float) + 40).astype(int)
    count_bin = np.zeros((131, 361), dtype=int)
    np.add.at(count_bin, (dec_index, ra_index), 1)
    _tb_tgt["local_count"] = count_bin[dec_index, ra_index]
    return _tb_tgt


def rank_recalculate(_tb_tgt):
    """Convert proposal rank and priority into an exponential planning weight."""
    if len(_tb_tgt) == 0:
        return _tb_tgt
    rank_values = np.asarray(_tb_tgt["rank"], dtype=float)
    priority_values = np.asarray(_tb_tgt["priority"], dtype=int)
    unique_ranks, rank_index = np.unique(rank_values, return_inverse=True)
    previous_distinct_ranks = np.concatenate(([0.0], unique_ranks[:-1]))
    interval_lower = 0.55 * unique_ranks + 0.45 * previous_distinct_ranks
    interval_step = 0.05 * (unique_ranks - previous_distinct_ranks)
    sci_usr_ranktot = (
        interval_lower[rank_index] + (9 - priority_values) * interval_step[rank_index]
    )
    _tb_tgt["rank_fin"] = np.exp(sci_usr_ranktot)
    return _tb_tgt


def weight(_tb_tgt, para_sci, para_exp, para_n):
    """Build the combined PPP weight from science, exposure, and density terms."""
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
    """Cluster nearby targets so the densest group can seed PPC optimization."""
    if len(_tb_tgt) == 0:
        return []
    target_coordinates = np.radians(
        np.column_stack(
            (
                np.asarray(_tb_tgt["dec"], dtype=float),
                np.asarray(_tb_tgt["ra"], dtype=float),
            )
        )
    )
    db = DBSCAN(eps=np.radians(sep), min_samples=1, metric="haversine").fit(
        target_coordinates
    )
    labels = db.labels_
    unique_labels, inverse_labels = np.unique(labels, return_inverse=True)
    cluster_weights = np.bincount(
        inverse_labels, weights=np.asarray(_tb_tgt["rank_fin"], dtype=float)
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
    """Return indices of targets that fall inside the hexagonal PFS field."""
    if len(_tb_tgt) == 0:
        return np.array([], dtype=int)
    target_coordinates = np.column_stack(
        (
            np.asarray(_tb_tgt["ra"], dtype=float),
            np.asarray(_tb_tgt["dec"], dtype=float),
        )
    )
    ppc_center = SkyCoord(ppc_ra * u.deg, ppc_dec * u.deg)
    hexagon = ppc_center.directional_offset_by(
        [30 + PA, 90 + PA, 150 + PA, 210 + PA, 270 + PA, 330 + PA, 30 + PA] * u.deg,
        1.38 / 2.0 * u.deg,
    )
    ra_h = hexagon.ra.deg
    dec_h = hexagon.dec.deg
    wrap_mask = np.fabs(ra_h - ppc_ra) > 180
    if np.any(wrap_mask):
        if ra_h[wrap_mask][0] > 180:
            ra_h[wrap_mask] -= 360
        else:
            ra_h[wrap_mask] += 360
    polygon = Path(np.column_stack((ra_h, dec_h)))
    return np.where(polygon.contains_points(target_coordinates))[0]


# -----------------------------------------------------------------------------
# KDE-based pointing seed search
# -----------------------------------------------------------------------------


def KDE_xy(_tb_tgt, X, Y):
    """Evaluate the KDE map for a subset of targets on the supplied grid."""
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
    """Estimate the highest-density pointing seed using a KDE over the sky grid."""
    if len(_tb_tgt) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    if len(_tb_tgt) == 1:
        return (
            _tb_tgt["ra"].data[0],
            _tb_tgt["dec"].data[0],
            np.nan,
            _tb_tgt["ra"].data[0],
            _tb_tgt["dec"].data[0],
        )
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
    if ra_step < 0.5 and dec_step < 0.5:
        X_, Y_ = np.mgrid[ra_low:ra_up:101j, dec_low:dec_up:101j]
    elif dec_step < 0.5:
        X_, Y_ = np.mgrid[0:360:721j, dec_low:dec_up:101j]
    elif ra_step < 0.5:
        X_, Y_ = np.mgrid[ra_low:ra_up:101j, -40:90:261j]
    else:
        X_, Y_ = np.mgrid[0:360:721j, -40:90:261j]
    if multiProcesing:
        threads_count = 4
        thread_n = min(threads_count, round(len(_tb_tgt) * 0.5))
        with multiprocessing.Pool(thread_n) as pool:
            kde_maps = pool.map(
                partial(KDE_xy, X=X_, Y=Y_), np.array_split(_tb_tgt, thread_n)
            )
        Z = np.sum(kde_maps, axis=0)
    else:
        Z = KDE_xy(_tb_tgt, X_, Y_)
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
    """Score a candidate PPC center by the targets it can allocate."""
    ppc_ra, ppc_dec = trial_ppc
    assigned_target_ids = fiber_allocate(
        _tb_tgt, single_ppc_mode=True, ppc_candidate=(ppc_ra, ppc_dec, ppc_pa)
    )
    assigned_mask = np.isin(_tb_tgt["identify_code"], assigned_target_ids)
    priority_values = np.asarray(_tb_tgt["priority"])
    tracked_priorities = list(range(10)) + [999]
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
        f"{ppc_ra}, {ppc_dec}, {ppc_pa}, Nall = {n_assigned_total}/{len(_tb_tgt)}, {priority_summary}"
    )
    score = (
        1.0 * n_assigned_total + 0.5 * assigned_counts[0] + 1.10 * assigned_counts[999]
    )
    return -score


def _prepare_tb_tgt_for_ppc(_tb_tgt, weight_params):
    """Apply all ranking and weighting steps before PPC optimization."""
    science_weight, exposure_weight, density_weight = weight_params
    _tb_tgt = rank_recalculate(_tb_tgt)
    _tb_tgt = count_local_number(_tb_tgt)
    _tb_tgt = weight(_tb_tgt, science_weight, exposure_weight, density_weight)
    return _tb_tgt


def _select_tb_tgt_remaining(_tb_tgt, proposal_ids=None):
    """Select targets that still need exposure, optionally for chosen proposals."""
    tb_tgt_remaining = _tb_tgt[_tb_tgt["exptime_PPP"] > 0]
    if proposal_ids is not None:
        tb_tgt_remaining = tb_tgt_remaining[
            np.isin(tb_tgt_remaining["proposal_id"], proposal_ids)
        ]
    return tb_tgt_remaining


def _initialize_tb_proposal_progress(_tb_tgt):
    """Initialize per-proposal bookkeeping for PPP completion tracking."""
    tb_tgt_remaining = _select_tb_tgt_remaining(_tb_tgt)
    proposal_ids = sorted(set(tb_tgt_remaining["proposal_id"]))
    proposal_fh_goal = [
        tb_tgt_remaining["allocated_time"][
            tb_tgt_remaining["proposal_id"] == proposal_id
        ][0]
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
    """Randomly down-sample a target table when KDE seeding would be too large."""
    if len(tb_tgt_input) <= max_rows:
        return tb_tgt_input
    sample_indices = rng.choice(len(tb_tgt_input), size=max_rows, replace=False)
    return tb_tgt_input[np.sort(sample_indices)]


def _select_ppc_seed(tb_tgt_remaining, rng, use_multiprocessing):
    """Pick the strongest target cluster and derive an initial PPC seed from it."""
    tb_tgt_groups = target_clustering(tb_tgt_remaining, 1.38)
    tb_tgt_group_primary = tb_tgt_groups[0]
    tb_tgt_group_sampled = _sample_tb_tgt_rows(tb_tgt_group_primary, 200, rng)
    _, _, _, initial_ra, initial_dec = KDE(tb_tgt_group_sampled, use_multiprocessing)
    return tb_tgt_group_primary, initial_ra, initial_dec


def _calculate_tb_tgt_credit_seconds(tb_tgt_assigned):
    """Cap credited exposure at the requested exposure time for each target."""
    requested_exptime = np.asarray(tb_tgt_assigned["exptime"], dtype=float)
    exptime_done = np.asarray(tb_tgt_assigned["exptime_done"], dtype=float)
    credited_exptime = exptime_done.copy()
    overdone_mask = exptime_done > requested_exptime
    credited_exptime[overdone_mask] = requested_exptime[overdone_mask]
    return credited_exptime


def _calculate_fh_done_by_proposal(_tb_tgt):
    """Aggregate credited observing time in fiber-hours by proposal."""
    credited_exptime = _calculate_tb_tgt_credit_seconds(_tb_tgt)
    proposal_ids = np.asarray(_tb_tgt["proposal_id"], dtype=str)
    unique_proposal_ids, inverse_indices = np.unique(proposal_ids, return_inverse=True)
    credited_fh = np.bincount(inverse_indices, weights=credited_exptime) / 3600.0
    return {
        proposal_id: fh_done
        for proposal_id, fh_done in zip(unique_proposal_ids, credited_fh)
    }


def _summarize_tb_tgt_assignment(_tb_tgt, tb_tgt_assigned_mask):
    """Summarize one PPC assignment step for logging and proposal accounting."""
    tb_tgt_assigned = _tb_tgt[tb_tgt_assigned_mask]
    assigned_credit_seconds = _calculate_tb_tgt_credit_seconds(tb_tgt_assigned)
    total_assigned_weight = float(
        np.sum(np.asarray(tb_tgt_assigned["rank_fin"], dtype=float))
    )
    proposal_ids = np.asarray(tb_tgt_assigned["proposal_id"], dtype=str)
    unique_proposal_ids, inverse_indices = np.unique(proposal_ids, return_inverse=True)
    credited_fh = np.bincount(inverse_indices, weights=assigned_credit_seconds) / 3600.0
    assigned_fh_by_proposal = {
        proposal_id: fh_done
        for proposal_id, fh_done in zip(unique_proposal_ids, credited_fh)
    }
    return (
        tb_tgt_assigned,
        assigned_credit_seconds,
        total_assigned_weight,
        assigned_fh_by_proposal,
    )


def _update_tb_proposal_progress(tb_proposal_progress, _tb_tgt):
    """Refresh proposal completion counters after one PPC assignment."""
    fh_done_by_proposal = _calculate_fh_done_by_proposal(_tb_tgt)
    for proposal_id in tb_proposal_progress["proposal_id"]:
        proposal_progress_mask = tb_proposal_progress["proposal_id"] == proposal_id
        proposal_mask = _tb_tgt["proposal_id"] == proposal_id
        tb_proposal_progress["N_psl"].data[proposal_progress_mask] = np.sum(
            proposal_mask
        )
        tb_proposal_progress["FH_done"].data[
            proposal_progress_mask
        ] = fh_done_by_proposal.get(proposal_id, 0.0)
        tb_proposal_progress["N_done"].data[proposal_progress_mask] = np.sum(
            _tb_tgt["exptime_PPP"][proposal_mask] <= 0
        )
        tb_proposal_progress["N_obs"].data[proposal_progress_mask] = np.sum(
            _tb_tgt["exptime_PPP"][proposal_mask] < _tb_tgt["exptime"][proposal_mask]
        )


def _build_ppc_list_table(final_ppc_records, _tb_tgt, backup):
    """Build the queue-mode PPC table written to PPP outputs."""
    ppc_weights = final_ppc_records[:, 4]
    weight_for_qplan = np.arange(1, len(final_ppc_records) + 1, dtype=int)
    n_ppc_final = len(final_ppc_records)
    resolution = _tb_tgt["resolution"][0]
    if backup:
        ppc_codes = [
            f"que_{resolution}_{time.strftime('%y%m%d')}_{int(index + 1)}_backup"
            for index in range(n_ppc_final)
        ]
    else:
        ppc_codes = [
            f"que_{resolution}_{time.strftime('%y%m%d')}_{int(index + 1)}"
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


# -----------------------------------------------------------------------------
# Queue-mode PPC optimization
# -----------------------------------------------------------------------------


def PPP_centers(
    _tb_tgt,
    n_ppc,
    weight_params=[2, 0, 0],
    random_seed=0,
    use_multiprocessing=True,
    backup=False,
    fixed_ppc_pa=0.0,
):
    """Determine PPC centers for queue-mode planning across multiple proposals."""
    start_time = time.time()
    rng = np.random.default_rng(random_seed)
    logger.info("[S2] Determine pointing centers started")
    ppc_records = []
    if len(_tb_tgt) == 0:
        logger.warning("[S2] no targets")
        return np.array(ppc_records), Table()
    if n_ppc == 0:
        logger.warning("[S2] no PPC to be determined")
        return np.array(ppc_records), Table()
    single_exptime = _tb_tgt.meta["single_exptime"]
    _tb_tgt = _prepare_tb_tgt_for_ppc(_tb_tgt, weight_params)
    tb_proposal_progress, proposal_fh_goal = _initialize_tb_proposal_progress(_tb_tgt)
    total_fh_goal = float(np.sum(proposal_fh_goal))
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
        _tb_tgt["priority"][_tb_tgt["exptime_done"] > 0] = 999
        tb_tgt_group_primary, initial_ra, initial_dec = _select_ppc_seed(
            tb_tgt_remaining, rng, use_multiprocessing
        )
        optimization_result = minimize(
            objective_ppc_assignment,
            [initial_ra, initial_dec],
            args=(tb_tgt_group_primary, fixed_ppc_pa),
            method="Nelder-Mead",
            options={"xatol": 0.1, "fatol": 0.1, "maxiter": 25, "maxfev": 25},
        )
        print(f"The optimal PPC center: {optimization_result.x}")
        best_ppc_ra, best_ppc_dec = optimization_result.x[0], optimization_result.x[1]
        best_ppc_pa = fixed_ppc_pa
        assigned_target_ids = fiber_allocate(
            tb_tgt_remaining,
            single_ppc_mode=True,
            ppc_candidate=(best_ppc_ra, best_ppc_dec, best_ppc_pa),
        )
        retry_count = 0
        while len(assigned_target_ids) == 0 and retry_count < 2:
            best_ppc_ra += rng.uniform(-0.15, 0.15)
            best_ppc_dec += rng.uniform(-0.15, 0.15)
            assigned_target_ids = fiber_allocate(
                tb_tgt_remaining,
                single_ppc_mode=True,
                ppc_candidate=(best_ppc_ra, best_ppc_dec, best_ppc_pa),
                observation_time=None,
            )
            retry_count += 1
        tb_tgt_assigned_mask = np.isin(_tb_tgt["identify_code"], assigned_target_ids)
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
            )
        )
        proposal_mask = np.isin(_tb_tgt["proposal_id"], incomplete_proposal_ids)
        n_partially_observed_before_filter = sum(
            (_tb_tgt["exptime_done"] > 0) * proposal_mask
        )
        _update_tb_proposal_progress(tb_proposal_progress, _tb_tgt)
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
    if len(ppc_records) == 0:
        logger.warning("[S2] No valid PPC centers were determined")
        return np.array(ppc_records), Table()
    elif len(ppc_records) > n_ppc:
        final_ppc_records = sorted(ppc_records, key=lambda x: x[4], reverse=True)[
            :n_ppc
        ]
    else:
        final_ppc_records = sorted(ppc_records, key=lambda x: x[4], reverse=True)
    final_ppc_records = np.asarray(final_ppc_records, dtype=object)
    if final_ppc_records.ndim == 1:
        final_ppc_records = final_ppc_records.reshape(1, -1)
    else:
        sort_order = np.argsort(-np.asarray(final_ppc_records[:, 4], dtype=float))
        final_ppc_records = final_ppc_records[sort_order]
    ppc_list_table = _build_ppc_list_table(final_ppc_records, _tb_tgt, backup)
    logger.info(
        f"[S2] Determine pointing centers done ( nppc = {len(final_ppc_records):.0f}; takes {round(time.time()-start_time,3)} sec)"
    )
    return final_ppc_records, ppc_list_table


def objective_single_program_ppc_assignment(trial_ppc, tb_tgt, ppc_pa=0.0):
    """Score a classic-mode PPC candidate for a single proposal."""
    ppc_ra, ppc_dec = trial_ppc
    assigned_target_ids = fiber_allocate(
        tb_tgt, single_ppc_mode=True, ppc_candidate=(ppc_ra, ppc_dec, ppc_pa)
    )
    assigned_mask = np.isin(tb_tgt["identify_code"], assigned_target_ids)
    proposal_policy = _get_proposal_policy(_single_program_proposal_id(tb_tgt))
    tracked_priorities = proposal_policy.get(
        "tracked_priorities",
        _DEFAULT_SINGLE_PROGRAM_PRIORITY_POLICY["tracked_priorities"],
    )
    emphasized_priority = proposal_policy.get(
        "emphasized_priority",
        _DEFAULT_SINGLE_PROGRAM_PRIORITY_POLICY["emphasized_priority"],
    )
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
        f"{ppc_ra}, {ppc_dec}, {ppc_pa}, Nall = {len(assigned_target_ids)}/{len(tb_tgt)}, {priority_summary}"
    )
    score = (
        1.0 * len(assigned_target_ids)
        + 0.5 * assigned_counts[emphasized_priority]
        + 1.50 * assigned_counts[999]
    )
    return -score


def _single_program_proposal_id(tb_tgt):
    """Return the unique proposal ID when a table contains one proposal only."""
    proposal_ids = sorted(set(np.asarray(tb_tgt["proposal_id"], dtype=str)))
    if len(proposal_ids) == 1:
        return proposal_ids[0]
    return None


def _filter_single_program_targets(tb_tgt, proposal_id, ppc_index):
    """Apply proposal-specific target filtering before a classic PPC step."""
    threshold = (
        _get_proposal_policy(proposal_id)
        .get("single_program_filter_thresholds", {})
        .get(ppc_index)
    )
    if threshold is None:
        return tb_tgt
    priority_mask = tb_tgt["priority"] == 999
    remaining_exptime_mask = tb_tgt["exptime_PPP"] < threshold
    return tb_tgt[priority_mask | remaining_exptime_mask]


def _optimize_fixed_pa_single_program_pointing(
    tb_tgt, central_ra, central_dec, fixed_ppc_pa, priority_limit, bounds, label
):
    """Optimize a single-program pointing around a fixed position-angle seed."""

    def optimize_pointing_objective(coords):
        """Evaluate how well a local pointing covers high-priority targets."""
        test_ra, test_dec = coords
        assigned_ids = fiber_allocate(
            tb_tgt,
            single_ppc_mode=True,
            ppc_candidate=(test_ra, test_dec, fixed_ppc_pa),
        )
        index_assign = np.isin(tb_tgt["identify_code"], assigned_ids)
        assigned_targets = tb_tgt[index_assign]
        priority_mask = tb_tgt["priority"] <= priority_limit
        n_priority_targets_total = int(np.sum(priority_mask))
        n_priority_targets_assigned = int(
            np.sum(assigned_targets["priority"] <= priority_limit)
        )
        n_total = len(assigned_ids)
        priority_norm = (
            n_priority_targets_assigned / n_priority_targets_total
            if n_priority_targets_total > 0
            else 0.0
        )
        total_norm = n_total / len(tb_tgt) if len(tb_tgt) > 0 else 0.0
        score = -(5 * priority_norm + total_norm)
        print(
            f"Testing RA={test_ra:.6f}, Dec={test_dec:.6f}: P<= {priority_limit}={n_priority_targets_assigned}, Total={n_total}, Score={score:.3f}"
        )
        return score

    print(
        f"\nOptimizing pointing position for {label} around RA={central_ra}, Dec={central_dec}"
    )
    result = minimize(
        optimize_pointing_objective,
        [central_ra, central_dec],
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1.0, "eps": 0.01},
    )
    optimized_ra, optimized_dec = result.x
    print(f"\nOptimal position found: RA={optimized_ra:.6f}, Dec={optimized_dec:.6f}")
    print(f"Optimization result: {result.message}")
    return optimized_ra, optimized_dec, fixed_ppc_pa


def _optimize_single_program_from_initial_guess(tb_tgt, initial_guess):
    """Optimize a single-program pointing starting from a proposal-defined seed."""
    special_ppc_pa = initial_guess[2]
    result = minimize(
        partial(
            objective_single_program_ppc_assignment,
            tb_tgt=tb_tgt,
            ppc_pa=special_ppc_pa,
        ),
        initial_guess[:2],
        method="Nelder-Mead",
    )
    print(result.x)
    return result.x[0], result.x[1], special_ppc_pa


def _select_special_single_program_pointing(
    tb_tgt_current, proposal_id, ppc_index, fixed_ppc_pa
):
    """Select any proposal-specific pointing rule before falling back to default."""
    proposal_policy = _get_proposal_policy(proposal_id)
    tb_tgt_current = _filter_single_program_targets(
        tb_tgt_current, proposal_id, ppc_index
    )
    single_program_pointings = proposal_policy.get("single_program_pointings", {})
    pointing_spec = single_program_pointings.get(ppc_index)
    if pointing_spec is not None:
        if pointing_spec["mode"] == "fixed":
            return tb_tgt_current, (
                pointing_spec["ra"],
                pointing_spec["dec"],
                pointing_spec["pa"],
            )
        if pointing_spec["mode"] == "optimize_initial_guess":
            return tb_tgt_current, _optimize_single_program_from_initial_guess(
                tb_tgt_current, pointing_spec["initial_guess"]
            )
    single_program_fixed_pointing = proposal_policy.get("single_program_fixed_pointing")
    if single_program_fixed_pointing is not None:
        ra, dec = single_program_fixed_pointing
        return tb_tgt_current, (ra, dec, fixed_ppc_pa)
    optimization_policy = proposal_policy.get("single_program_optimization")
    if optimization_policy is not None:
        max_ppc_count = optimization_policy.get("max_ppc_count")
        if max_ppc_count is None or ppc_index < max_ppc_count:
            return tb_tgt_current, _optimize_fixed_pa_single_program_pointing(
                tb_tgt_current,
                optimization_policy["central_ra"],
                optimization_policy["central_dec"],
                fixed_ppc_pa,
                priority_limit=optimization_policy["priority_limit"],
                bounds=optimization_policy["bounds"],
                label=proposal_id,
            )
    return tb_tgt_current, None


def _determine_default_single_program_pointing(tb_tgt, fixed_ppc_pa):
    """Derive the default classic pointing by clustering plus local optimization."""
    tb_tgt_groups = target_clustering(tb_tgt, 1.38)
    tb_tgt_primary = tb_tgt_groups[0]
    df_tgt_primary = Table.to_pandas(tb_tgt_primary)
    n_tgt = min(200, len(tb_tgt_primary))
    df_tgt_primary = df_tgt_primary.sample(n_tgt, ignore_index=True, random_state=1)
    tb_tgt_sample = Table.from_pandas(df_tgt_primary)
    _, _, _, peak_x, peak_y = KDE(tb_tgt_sample, False)
    result = minimize(
        partial(
            objective_single_program_ppc_assignment,
            tb_tgt=tb_tgt_primary,
            ppc_pa=fixed_ppc_pa,
        ),
        [peak_x, peak_y],
        method="Nelder-Mead",
        options={"xatol": 0.01, "fatol": 0.001},
    )
    print(result.x)
    return result.x[0], result.x[1], fixed_ppc_pa


def _build_classic_ppc_list_table(final_ppc_records, tb_tgt, n_ppc):
    """Build the PPC table for classic single-program planning output."""
    resol = tb_tgt["resolution"][0]
    proposal_id = tb_tgt["proposal_id"][0]
    proposal_policy = _get_proposal_policy(proposal_id)
    custom_prefix = proposal_policy.get("classic_ppc_prefix")
    if custom_prefix is not None:
        ppc_code = [f"{custom_prefix}_{n + 1}" for n in np.arange(n_ppc)]
    else:
        ppc_code = [
            f"cla_{resol}_{proposal_id.split('-')[1]}_{n + 1}" for n in np.arange(n_ppc)
        ]
    return Table(
        [
            ppc_code,
            final_ppc_records[:, 1],
            final_ppc_records[:, 2],
            final_ppc_records[:, 3],
            ["J2000"] * n_ppc,
            [0] * n_ppc,
            [0] * n_ppc,
            [900.0] * n_ppc,
            [1200.0] * n_ppc,
            [resol] * n_ppc,
            final_ppc_records[:, -1],
            final_ppc_records[:, -2],
            [""] * n_ppc,
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


# -----------------------------------------------------------------------------
# Classic single-program PPC optimization
# -----------------------------------------------------------------------------


def PPP_centers_for_single_program(
    _tb_tgt,
    n_ppc,
    weight_para=_DEFAULT_PPP_WEIGHT_PARAMS,
    fixed_ppc_pa=0.0,
    write_ppc_list=False,
    output_dir=None,
):
    """Determine PPC centers for classic planning of a single proposal."""
    time_start = time.time()
    logger.info("[S2] Determine pointing centers started")
    ppc_lst = []
    para_sci, para_exp, para_n = weight_para
    tb_tgt = rank_recalculate(_tb_tgt)
    tb_tgt = count_local_number(tb_tgt)
    tb_tgt = weight(tb_tgt, para_sci, para_exp, para_n)
    single_exptime = tb_tgt.meta["single_exptime"]
    print(single_exptime)
    tb_tgt_current = tb_tgt[tb_tgt["exptime_PPP"] > 0]
    tb_tgt_current["exptime_done"] = 0.0
    proposal_id = _single_program_proposal_id(tb_tgt)
    while len(tb_tgt_current) > 0 and len(ppc_lst) < n_ppc:
        ppc_index = len(ppc_lst)
        tb_tgt_current, pointing = _select_special_single_program_pointing(
            tb_tgt_current, proposal_id, ppc_index, fixed_ppc_pa
        )
        if len(tb_tgt_current) == 0:
            logger.warning(
                "[S2] no remaining targets after proposal-specific filtering"
            )
            break
        if pointing is None:
            pointing = _determine_default_single_program_pointing(
                tb_tgt_current, fixed_ppc_pa
            )
        ra, dec, pa = pointing
        lst_tgtID_assign = fiber_allocate(
            tb_tgt_current, single_ppc_mode=True, ppc_candidate=(ra, dec, pa)
        )
        index_in = PFS_FoV(ra, dec, pa, tb_tgt_current)
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
                ],
                dtype=object,
            )
        )
        index_assign = np.isin(tb_tgt_current["identify_code"], lst_tgtID_assign)
        from collections import Counter

        print(dict(Counter(tb_tgt_current["exptime_PPP"][index_in])))
        print(dict(Counter(tb_tgt_current["exptime_PPP"][index_assign])))
        tb_tgt_current["exptime_PPP"][index_assign] -= single_exptime
        tb_tgt_current["exptime_done"][index_assign] += single_exptime
        tb_tgt_current["priority"][tb_tgt_current["exptime_done"] > 0] = 999
        proposal_policy = _get_proposal_policy(proposal_id)
        remaining_priority_increment_value = proposal_policy.get(
            "remaining_priority_increment_value"
        )
        remaining_priority_increment_exptime_ppp = proposal_policy.get(
            "remaining_priority_increment_exptime_ppp"
        )
        if (
            remaining_priority_increment_value is not None
            and remaining_priority_increment_exptime_ppp is not None
        ):
            tb_tgt_current["priority"][
                (tb_tgt_current["exptime_done"] == 0)
                & (
                    tb_tgt_current["exptime_PPP"]
                    == remaining_priority_increment_exptime_ppp
                )
            ] += remaining_priority_increment_value
        tb_tgt_current = tb_tgt_current[tb_tgt_current["exptime_PPP"] > 0]
        print(sum(tb_tgt_current["priority"] == 999))
    ppc_lst_fin = np.array(ppc_lst)
    ppcList = _build_classic_ppc_list_table(ppc_lst_fin, tb_tgt, n_ppc)
    if write_ppc_list:
        if output_dir is None:
            raise ValueError("output_dir must be provided when write_ppc_list=True")
        ppcList.write(
            os.path.join(output_dir, "ppcList_1by1.ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )
    logger.info(
        f"[S2] Determine pointing centers done ( nppc = {len(ppc_lst_fin):.0f}; takes {round(time.time()-time_start,3)} sec)"
    )
    return ppc_lst_fin
