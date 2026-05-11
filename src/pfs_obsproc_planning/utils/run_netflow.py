#!/usr/bin/env python3

import time
from datetime import datetime, timedelta

import ets_fiber_assigner.netflow as nf
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table, join
from astropy.time import Time
from loguru import logger
from ics.cobraOps.TargetGroup import TargetGroup

bench = None
cobra_location_group = None
min_sky_targets_per_location = None
location_group_penalty = None
cobra_instrument_region = None
min_sky_targets_per_instrument_region = None
instrument_region_penalty = None
black_dot_penalty_cost = None
cobraSafetyMargin = 0.1
_COBRA_FEATURE_FLAGS = None
_CLASSIC_PPC_TARGET_RADIUS_DEG = 2.0
_CLASSIC_PPC_CLUSTER_LINK_RADIUS_DEG = 2.0 * _CLASSIC_PPC_TARGET_RADIUS_DEG

_QUEUE_NETFLOW_PRIORITY_COSTS = {
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

_QUEUE_NETFLOW_CLASSDICT_TEMPLATE = {
    **{
        f"sci_P{priority}": {
            "nonObservationCost": non_observation_cost,
            "partialObservationCost": 200,
            "calib": False,
        }
        for priority, non_observation_cost in _QUEUE_NETFLOW_PRIORITY_COSTS.items()
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

_CLASSIC_NETFLOW_CALIBRATION_COST = 2000
_CLASSIC_FIXED_COST_PATTERNS = {
    "baseline": np.array(
        [5e4, 5e4, 2e4, 1e4, 9e3, 8e3, 7e3, 6e3, 5e3, 1e3, 100],
        dtype=float,
    ),
    "linear_100_to_1": np.array(
        [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1], dtype=float
    ),
    "linear_1000_to_10": np.array(
        [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 10], dtype=float
    ),
    "steep_exp": np.array(
        [50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 10],
        dtype=float,
    ),
    "highlight_p0_p1": np.array(
        [50000, 50000, 5000, 3000, 2000, 1000, 500, 200, 100, 50, 10],
        dtype=float,
    ),
    "highlight_p0_p2": np.array(
        [50000, 30000, 15000, 5000, 3000, 1500, 800, 400, 200, 100, 20],
        dtype=float,
    ),
    "gentle_tail": np.array(
        [50000, 25000, 12000, 6000, 3000, 1500, 800, 400, 200, 100, 50],
        dtype=float,
    ),
}


def set_bench(bench_model):
    global bench
    bench = bench_model


def select_good_observation_time(
    ppc_ra,
    ppc_dec,
    dates_local=None,
    local_times_hst=None,
    min_elevation=30.0,
    max_elevation=75.0,
):
    """Pick a coarse but reasonable UTC observation time for one or more PPCs.

    The search is intentionally simple: try one representative night in each
    season and a few nighttime local HST times (21:00, 00:00, 03:00 by default).
    Choose the sampled UTC time that maximizes how many PPC centers are within
    the requested elevation range. If multiple times tie, prefer the one with
    the highest summed valid elevation. If none match, fall back to the time
    with the highest mean elevation across the supplied PPC centers.
    """

    ppc_ra_values = np.atleast_1d(np.asarray(ppc_ra, dtype=float))
    ppc_dec_values = np.atleast_1d(np.asarray(ppc_dec, dtype=float))
    if ppc_ra_values.shape != ppc_dec_values.shape:
        raise ValueError("ppc_ra and ppc_dec must have the same shape")

    if dates_local is None:
        dates_local = [
            "2026-01-15",
            "2026-03-15",
            "2026-05-15",
            "2026-07-15",
            "2026-09-15",
            "2026-11-15",
        ]
    if local_times_hst is None:
        local_times_hst = [21, 0, 3]

    candidate_datetimes_utc = []
    for date_local in dates_local:
        local_midnight = datetime.fromisoformat(f"{date_local}T00:00:00")
        for local_hour in local_times_hst:
            local_datetime = local_midnight + timedelta(hours=float(local_hour))
            utc_datetime = local_datetime + timedelta(hours=10)
            candidate_datetimes_utc.append(utc_datetime)

    sample_times = Time(candidate_datetimes_utc, scale="utc")

    subaru = EarthLocation.of_site("Subaru Telescope")
    elevations = np.vstack(
        [
            np.asarray(
                SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                .transform_to(AltAz(obstime=sample_times, location=subaru))
                .alt.deg,
                dtype=float,
            )
            for ra, dec in zip(ppc_ra_values, ppc_dec_values)
        ]
    )

    valid_mask = (elevations >= float(min_elevation)) & (
        elevations <= float(max_elevation)
    )
    visible_counts = np.sum(valid_mask, axis=0)
    n_ppc = len(ppc_ra_values)

    if np.any(visible_counts > 0):
        candidate_indices = np.flatnonzero(visible_counts == np.max(visible_counts))
        valid_elevation_sums = np.sum(
            np.where(valid_mask[:, candidate_indices], elevations[:, candidate_indices], 0.0),
            axis=0,
        )
        best_index = int(candidate_indices[np.argmax(valid_elevation_sums)])
        logger.debug(
            "Selected observation_time={} with {}/{} PPCs visible in elevation range {:.1f}-{:.1f} deg".format(
                sample_times[best_index].strftime("%Y-%m-%dT%H:%M:%SZ"),
                int(visible_counts[best_index]),
                int(n_ppc),
                float(min_elevation),
                float(max_elevation),
            )
        )
    else:
        best_index = int(np.argmax(np.mean(elevations, axis=0)))
        logger.warning(
            "No sampled observation time keeps any PPC within elevation {:.1f}-{:.1f} deg; using {} with mean elevation {:.2f} deg across {} PPCs".format(
                float(min_elevation),
                float(max_elevation),
                sample_times[best_index].strftime("%Y-%m-%dT%H:%M:%SZ"),
                float(np.mean(elevations[:, best_index])),
                int(n_ppc),
            )
        )

    best_elevations = elevations[:, best_index]
    if n_ppc == 1:
        best_elevations = float(best_elevations[0])

    return sample_times[best_index].strftime("%Y-%m-%dT%H:%M:%SZ"), best_elevations


def _resolve_observation_time(observation_time, ppc_list):
    if observation_time is not None:
        return observation_time

    representative_ra = np.asarray(ppc_list[:, 1], dtype=float)
    representative_dec = np.asarray(ppc_list[:, 2], dtype=float)
    resolved_time, _ = select_good_observation_time(
        representative_ra,
        representative_dec,
    )
    return resolved_time


def _filter_targets_near_ppc(tb_tgt, ppc_centers, radius_deg=2.0):
    if len(tb_tgt) == 0 or len(ppc_centers) == 0:
        return tb_tgt

    target_coords = SkyCoord(
        ra=np.asarray(tb_tgt["ra"], dtype=float) * u.deg,
        dec=np.asarray(tb_tgt["dec"], dtype=float) * u.deg,
    )
    keep_mask = np.zeros(len(tb_tgt), dtype=bool)

    for ppc_ra, ppc_dec in ppc_centers:
        center_coord = SkyCoord(ra=float(ppc_ra) * u.deg, dec=float(ppc_dec) * u.deg)
        keep_mask |= target_coords.separation(center_coord).deg <= float(radius_deg)

    return tb_tgt[keep_mask]


def _cluster_classic_ppc_index_groups(ppc_list, link_radius_deg=_CLASSIC_PPC_CLUSTER_LINK_RADIUS_DEG):
    """Split PPCs into connected sky clusters based on center separation.

    Two PPCs are connected when their centers are within `link_radius_deg`.
    Using 4 deg by default matches the case where 2 deg target-search radii can
    overlap, so disconnected fields are solved independently while nearby PPCs
    are still optimized together.
    """

    if len(ppc_list) == 0:
        return []

    if len(ppc_list) == 1:
        return [np.array([0], dtype=int)]

    coords = SkyCoord(
        ra=np.asarray(ppc_list[:, 1], dtype=float) * u.deg,
        dec=np.asarray(ppc_list[:, 2], dtype=float) * u.deg,
    )

    visited = np.zeros(len(ppc_list), dtype=bool)
    groups = []

    for start_index in range(len(ppc_list)):
        if visited[start_index]:
            continue

        queue = [start_index]
        visited[start_index] = True
        group_indices = []

        while queue:
            current_index = queue.pop()
            group_indices.append(current_index)

            separations = coords[current_index].separation(coords).deg
            neighbor_indices = np.flatnonzero(
                (~visited) & (separations <= float(link_radius_deg))
            )
            for neighbor_index in neighbor_indices:
                visited[neighbor_index] = True
                queue.append(int(neighbor_index))

        groups.append(np.array(sorted(group_indices), dtype=int))

    return groups


def _iter_classic_ppc_groups(tb_tgt, ppc_list):
    """Yield classic PPC groups to solve together.

    User-provided classic PPCs are split into disconnected sky clusters so that
    far-away fields do not enter the same netflow solve.
    """

    if len(ppc_list) == 0:
        return []

    if tb_tgt.meta.get("PPC_origin") == "usr":
        ppc_index_groups = _cluster_classic_ppc_index_groups(ppc_list)
    else:
        ppc_index_groups = [np.arange(len(ppc_list), dtype=int)]

    grouped_entries = []
    for group_number, group_indices in enumerate(ppc_index_groups, start=1):
        ppc_group = np.asarray(ppc_list[group_indices], dtype=object)
        tb_tgt_group = _select_classic_netflow_targets(tb_tgt, ppc_group)
        grouped_entries.append((group_number, group_indices, ppc_group, tb_tgt_group))

    return grouped_entries


def build_netflow_targets(tb_tgt, for_single_ppc=False):
    netflow_targets = []
    proposal_class_keys = []
    single_exptime = tb_tgt.meta["single_exptime"]
    exposure_column = "exptime_PPP"
    use_cobra_feature_flag = tb_tgt.meta.get("cobra_feature_flag", True)

    for tb_tgt_row in tb_tgt:
        target_exptime = single_exptime if for_single_ppc else tb_tgt_row[exposure_column]
        target_priority = int(tb_tgt_row["priority"])
        qa_reference_arm = tb_tgt_row["qa_reference_arm"]

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
                pmra=tb_tgt_row["pmra"],
                pmdec=tb_tgt_row["pmdec"],
                parallax=tb_tgt_row["parallax"],
                epoch=float(str(tb_tgt_row["equinox"]).lstrip("J")),
            )
        )
        proposal_class_keys.append(f"sci_P{target_priority}")

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
    return {
        class_key: class_config.copy()
        for class_key, class_config in _QUEUE_NETFLOW_CLASSDICT_TEMPLATE.items()
    }


def classic_build_classdict(cost_values=None):
    if cost_values is None:
        cost_values = _CLASSIC_FIXED_COST_PATTERNS["baseline"]

    cost_values = np.asarray(cost_values, dtype=float)
    science_priorities = range(11)
    if len(cost_values) != len(list(science_priorities)):
        raise ValueError("Classic fixed cost patterns must provide costs for priorities P0-P10")
    partial_observation_cost = float(np.max(cost_values))

    classdict = build_classdict()
    for class_key in list(classdict.keys()):
        if class_key.startswith("sci_P"):
            del classdict[class_key]
    classdict.update(
        {
            f"sci_P{priority}": {
                "nonObservationCost": float(non_observation_cost),
                "partialObservationCost": partial_observation_cost,
                "calib": False,
            }
            for priority, non_observation_cost in zip(science_priorities, cost_values)
        }
    )
    classdict["cal"] = {
        "numRequired": 0,
        "nonObservationCost": _CLASSIC_NETFLOW_CALIBRATION_COST,
        "calib": True,
    }
    classdict["sky"] = {
        "numRequired": 0,
        "nonObservationCost": _CLASSIC_NETFLOW_CALIBRATION_COST,
        "calib": True,
    }
    return classdict


def cobra_move_cost(dist):
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
    observation_time=None,
    for_single_ppc=False,
    classdict_override=None,
):
    telescope_ra = ppc_list[:, 1]
    telescope_dec = ppc_list[:, 2]
    telescope_pa = ppc_list[:, 3]

    netflow_targets, proposal_fh_limits = build_netflow_targets(
        tb_tgt, for_single_ppc=for_single_ppc
    )
    class_dict = (
        classdict_override if classdict_override is not None else build_classdict()
    )
    observation_time = _resolve_observation_time(observation_time, ppc_list)

    telescopes = [
        nf.Telescope(telescope_ra[index], telescope_dec[index], telescope_pa[index], observation_time)
        for index in range(len(telescope_ra))
    ]
    focal_plane_positions = [
        telescope.get_fp_positions(netflow_targets) for telescope in telescopes
    ]

    n_visit = len(telescope_ra)
    single_exptime = tb_tgt.meta["single_exptime"]
    visit_costs = [0] * n_visit

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
        cobra_feature_flags = _get_cobra_feature_flags()
    else:
        cobra_feature_flags = None

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

    res = [{} for _ in range(min(n_visit, len(telescope_ra)))]
    for key, value in problem._vardict.items():
        if key.startswith("Tv_Cv_"):
            visited = problem.value(value) > 0
            if visited:
                _, _, tidx, cidx, ivis = key.split("_")
                res[int(ivis)][int(tidx)] = int(cidx)

    for ivis, (vis, tp) in enumerate(zip(res, focal_plane_positions)):
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
                if dist > bench.cobras.L1[i] + bench.cobras.L2[i]:
                    logger.warning(
                        f"(CobraId={i}) Distance from the center exceeds L1+L2 ({dist} mm)"
                    )

    return res, telescopes, netflow_targets


def fiber_allocate(
    tb_tgt,
    single_ppc_mode=False,
    ppc_candidate=None,
    observation_time=None,
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    classdict_override=None,
    backup=False,
    queue_mode=True,
):
    if single_ppc_mode:
        if ppc_candidate is None:
            raise ValueError("ppc_candidate must be provided when single_ppc_mode=True")

        ppc_ra, ppc_dec, ppc_pa = ppc_candidate
        tb_tgt = _filter_targets_near_ppc(
            tb_tgt,
            [(ppc_ra, ppc_dec)],
            radius_deg=_CLASSIC_PPC_TARGET_RADIUS_DEG,
        )
        if len(tb_tgt) == 0:
            logger.warning(
                "No targets within {:.1f} deg of PPC center ({:.6f}, {:.6f})".format(
                    float(_CLASSIC_PPC_TARGET_RADIUS_DEG),
                    float(ppc_ra),
                    float(ppc_dec),
                )
            )
            return []
        ppc_list = np.array([[0, ppc_ra, ppc_dec, ppc_pa, 0]], dtype=object)

        assignments, telescopes, netflow_targets = run_netflow(
            ppc_list,
            tb_tgt,
            for_single_ppc=True,
            observation_time=observation_time,
            classdict_override=classdict_override,
        )

        return [
            netflow_targets[target_idx].ID
            for assignment_map in assignments
            for target_idx in assignment_map
        ]

    elif ("PPC" not in tb_tgt.meta) or (len(tb_tgt.meta["PPC"]) == 0):
        logger.warning("[S3] No PPC has been determined")
        return []

    elif len(tb_tgt) == 0:
        logger.warning("[S3] No targets")
        return []

    else:
        time_start = time.time()
        logger.info("[S3] Run netflow started")

        ppc_records = []
        ppc_list = tb_tgt.meta["PPC"]
        tb_tgt = _filter_targets_near_ppc(
            tb_tgt,
            ppc_list[:, 1:3],
            radius_deg=_CLASSIC_PPC_TARGET_RADIUS_DEG,
        )
        if len(tb_tgt) == 0:
            logger.warning(
                "[S3] No targets within {:.1f} deg of any PPC center".format(
                    float(_CLASSIC_PPC_TARGET_RADIUS_DEG)
                )
            )
            return []
        today_date = datetime.now().strftime("%y%m%d")
        resolution = tb_tgt["resolution"][0]

        tb_tgt_in_group = tb_tgt

        logger.info(
            f"[S3] Group {1:3d}: nppc = {len(ppc_list):5d}, n_tgt = {len(tb_tgt_in_group):6d}"
        )
        if queue_mode:
            logger.info("[S3] Skipping multi-pointing netflow in queue mode")
        else:
            assignments, telescopes, netflow_targets = run_netflow(
                ppc_list,
                tb_tgt_in_group,
                num_reserved_fibers=num_reserved_fibers,
                fiber_non_allocation_cost=fiber_non_allocation_cost,
                observation_time=observation_time,
                classdict_override=classdict_override,
            )

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
                    ppc_priority = np.nan
                else:
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

        if len(ppc_records) == 0:
            logger.warning("[S3] Netflow returned no PPC allocations")
            return []

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

        tb_ppc_netflow = tb_ppc_netflow[
            np.argsort(np.asarray(tb_ppc_netflow["ppc_priority_usr"], dtype=float))[::-1]
        ]

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

        tb_ppc_netflow["ppc_priority"] = np.arange(1, len(tb_ppc_netflow) + 1, dtype=int)

        logger.info(
            f"[S3] Run netflow done (takes {round(time.time() - time_start, 3)} sec)"
        )

        return tb_ppc_netflow


def check_netflow_assign_exptime(tb_tgt, tb_ppc_netflow):
    if len(tb_ppc_netflow) == 0:
        return tb_tgt

    tb_tgt["exptime_assign"] = 0
    single_exptime = int(tb_tgt.meta["single_exptime"])
    target_index_by_id = {
        identify_code: index
        for index, identify_code in enumerate(np.asarray(tb_tgt["identify_code"], dtype=str))
    }

    for ppc_row in tb_ppc_netflow:
        assigned_indices = [
            target_index_by_id[target_id]
            for target_id in ppc_row["ppc_allocated_targets"]
            if target_id in target_index_by_id
        ]
        if assigned_indices:
            tb_tgt["exptime_assign"].data[assigned_indices] += single_exptime

    return tb_tgt


def _select_classic_netflow_targets(tb_tgt, ppc_list):
    tb_tgt_filtered = _filter_targets_near_ppc(
        tb_tgt,
        ppc_list[:, 1:3],
        radius_deg=_CLASSIC_PPC_TARGET_RADIUS_DEG,
    )
    if len(tb_tgt_filtered) == 0:
        return tb_tgt_filtered

    tb_tgt_inuse = tb_tgt_filtered.copy(copy_data=True)
    tb_tgt_inuse.meta = dict(tb_tgt.meta)
    tb_tgt_inuse.meta["PPC"] = ppc_list
    return tb_tgt_inuse


def optimize_non_observation_costs(
    _tb_tgt,
    ppc_lst,
    otime=None,
    debug=False,
):
    weight_total = 1.0
    weight_p0 = 3.0
    weight_p1 = 2.0
    focus_max = 10

    tb_tgt_inuse = _select_classic_netflow_targets(_tb_tgt, ppc_lst)
    if len(tb_tgt_inuse) == 0:
        logger.warning("[S3] No classic netflow targets remain after PPC filtering during cost optimization")
        return classic_build_classdict(), {
            "score": 0.0,
            "total": 0,
            "p0": 0,
            "p1": 0,
            "non_obs_costs": {
                f"sci_P{priority}": float(cost)
                for priority, cost in enumerate(_CLASSIC_FIXED_COST_PATTERNS["baseline"])
            },
            "pattern_name": "baseline",
            "success": False,
            "message": "No classic netflow targets remain after PPC filtering",
        }

    science_keys = [f"sci_P{priority}" for priority in range(11)]
    evaluation_counter = 0

    target_priorities = np.asarray(tb_tgt_inuse["priority"], dtype=int)
    unsupported_priorities = sorted(set(target_priorities[target_priorities > 10]))
    if unsupported_priorities:
        raise ValueError(
            "Classic fixed cost optimization only supports priorities P0-P10; found {}".format(
                unsupported_priorities
            )
        )
    priority_by_code = {
        code: priority
        for code, priority in zip(np.asarray(tb_tgt_inuse["identify_code"], dtype=str), target_priorities)
    }

    total_available = int(np.sum((target_priorities >= 0) & (target_priorities <= focus_max)))
    p0_available = int(np.sum(target_priorities == 0))
    p1_available = int(np.sum(target_priorities == 1))

    def _build_cost_classdict(cost_values):
        return classic_build_classdict(cost_values)

    def _extract_assigned_ids(res, tgt_lst_netflow):
        return {
            tgt_lst_netflow[tidx].ID
            for vis in res
            for tidx in vis.keys()
        }

    def _score_assignment_counts(assigned_ids):
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

    def _assigned_priority_counts(assigned_ids, min_priority=0, max_priority=9):
        counts = {priority: 0 for priority in range(min_priority, max_priority + 1)}
        for code in assigned_ids:
            priority = priority_by_code.get(code)
            if priority in counts:
                counts[priority] += 1
        return counts

    def evaluate_pattern(pattern_name, cost_values):
        nonlocal evaluation_counter
        evaluation_counter += 1

        classdict = _build_cost_classdict(cost_values)
        non_obs_costs = {
            key: float(classdict[key]["nonObservationCost"])
            for key in science_keys
        }

        assigned_ids = set()
        for group_number, group_indices, ppc_group, tb_tgt_group in _iter_classic_ppc_groups(
            tb_tgt_inuse,
            ppc_lst,
        ):
            if len(tb_tgt_group) == 0:
                logger.warning(
                    "[S3] Cost pattern {} group {} has no targets within {:.1f} deg of provided PPCs; skipping group".format(
                        pattern_name,
                        group_number,
                        float(_CLASSIC_PPC_TARGET_RADIUS_DEG),
                    )
                )
                continue

            res, _, tgt_lst_netflow = run_netflow(
                ppc_group,
                tb_tgt_group,
                observation_time=otime,
                classdict_override=classdict,
            )

            assigned_ids.update(_extract_assigned_ids(res, tgt_lst_netflow))
        total, p0, p1 = _score_assignment_counts(assigned_ids)
        assigned_priority_counts = _assigned_priority_counts(assigned_ids)
        total_norm = total / total_available if total_available > 0 else 0.0
        p0_norm = p0 / p0_available if p0_available > 0 else 0.0
        p1_norm = p1 / p1_available if p1_available > 0 else 0.0
        score = weight_total * total_norm + weight_p0 * p0_norm + weight_p1 * p1_norm

        print(
            "Cost pattern {} (iter {}): assigned P0-9 = {} (score={:.4f}, total={}, P0={}, P1={})".format(
                pattern_name,
                evaluation_counter,
                ", ".join(
                    f"P{priority}:{count}"
                    for priority, count in assigned_priority_counts.items()
                ),
                score,
                total,
                p0,
                p1,
            )
        )

        evaluation = {
            "pattern_name": pattern_name,
            "classdict": classdict,
            "non_obs_costs": non_obs_costs,
            "assigned_ids": assigned_ids,
            "assigned_priority_counts": assigned_priority_counts,
            "total": total,
            "p0": p0,
            "p1": p1,
            "score": score,
        }

        if debug:
            from collections import Counter

            logger.info(dict(sorted(Counter(target_priorities).items())))
            logger.info(
                dict(sorted(Counter(priority_by_code[code] for code in assigned_ids).items()))
            )
            logger.info(
                {
                    int(key.replace("sci_P", "")): value
                    for key, value in non_obs_costs.items()
                }
            )

        return evaluation

    best_eval = None
    for pattern_name, cost_values in _CLASSIC_FIXED_COST_PATTERNS.items():
        candidate_eval = evaluate_pattern(pattern_name, cost_values)
        if best_eval is None or candidate_eval["score"] > best_eval["score"]:
            best_eval = candidate_eval

    print(
        "Best cost pattern: {} (score={:.4f})".format(
            best_eval["pattern_name"],
            best_eval["score"],
        )
    )

    best_costs = best_eval["non_obs_costs"]
    best_classdict = best_eval["classdict"]

    best = {
        "score": best_eval["score"],
        "total": best_eval["total"],
        "p0": best_eval["p0"],
        "p1": best_eval["p1"],
        "non_obs_costs": best_costs,
        "pattern_name": best_eval["pattern_name"],
        "success": True,
        "message": "Completed fixed-pattern cost search",
    }

    return best_classdict, best


def fiber_allocation_classic(
    tb_tgt,
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    classdict_override=None,
):
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

    ppc_records = []
    grouped_ppcs = _iter_classic_ppc_groups(tb_tgt, ppc_list)
    if len(grouped_ppcs) > 1:
        logger.info(
            "[S3] Split {} provided PPCs into {} disconnected clusters before classic netflow".format(
                len(ppc_list),
                len(grouped_ppcs),
            )
        )

    for group_number, group_indices, ppc_group, tb_tgt_group in grouped_ppcs:
        logger.info(
            "[S3] Group {:3d}: nppc = {:5d}, n_tgt = {:6d}".format(
                group_number,
                len(ppc_group),
                len(tb_tgt_group),
            )
        )

        if len(tb_tgt_group) == 0:
            logger.warning(
                "[S3] Group {:3d} has no targets within {:.1f} deg of its PPCs; keeping empty allocations".format(
                    group_number,
                    float(_CLASSIC_PPC_TARGET_RADIUS_DEG),
                )
            )
            for ppc_row in ppc_group:
                ppc_records.append(
                    [
                        float(ppc_row[1]),
                        float(ppc_row[2]),
                        float(ppc_row[3]),
                        np.nan,
                        0.0,
                        [],
                        resolution,
                    ]
                )
            continue

        tb_ppc_group = fiber_allocate(
            tb_tgt_group,
            num_reserved_fibers=num_reserved_fibers,
            fiber_non_allocation_cost=fiber_non_allocation_cost,
            classdict_override=classdict_override,
            queue_mode=False,
        )

        if len(tb_ppc_group) == 0:
            for ppc_row in ppc_group:
                ppc_records.append(
                    [
                        float(ppc_row[1]),
                        float(ppc_row[2]),
                        float(ppc_row[3]),
                        np.nan,
                        0.0,
                        [],
                        resolution,
                    ]
                )
            continue

        for ppc_row in tb_ppc_group:
            ppc_records.append(
                [
                    float(ppc_row["ppc_ra"]),
                    float(ppc_row["ppc_dec"]),
                    float(ppc_row["ppc_pa"]),
                    float(ppc_row["ppc_priority_usr"]),
                    float(ppc_row["ppc_fiber_usage_frac"]),
                    list(ppc_row["ppc_allocated_targets"]),
                    str(ppc_row["ppc_resolution"]),
                ]
            )

    if len(ppc_records) == 0:
        tb_ppc_netflow = Table(names=output_columns, dtype=output_dtypes)
    else:
        tb_ppc_netflow = Table(
            rows=ppc_records,
            names=[
                "ppc_ra",
                "ppc_dec",
                "ppc_pa",
                "ppc_priority",
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
        tb_ppc_netflow["ppc_code"] = [
            f"cla_{resolution}_{proposal_token}_{pointing_index}"
            for pointing_index in range(1, len(tb_ppc_netflow) + 1)
        ]
        tb_ppc_netflow = tb_ppc_netflow[output_columns]

    if tb_tgt.meta["PPC_origin"] == "usr":
        if len(tb_ppc_netflow) == 0:
            logger.warning(
                "[S3] Skip merging user PPC priorities because netflow produced no allocations"
            )
            logger.info(
                f"[S3] Run netflow done (takes {round(time.time() - time_start, 3)} sec)"
            )
            return tb_ppc_netflow

        lst_ppc_usr = tb_tgt.meta["PPC"]
        tb_ppc_usr = Table(
            lst_ppc_usr[:, 1:],
            names=["ppc_ra", "ppc_dec", "ppc_pa", "ppc_priority_usr"],
        )
        for column_name in ["ppc_ra", "ppc_dec", "ppc_pa", "ppc_priority_usr"]:
            tb_ppc_usr[column_name] = np.asarray(tb_ppc_usr[column_name], dtype=float)
        for column_name in ["ppc_ra", "ppc_dec", "ppc_pa"]:
            tb_ppc_netflow[column_name] = np.asarray(
                tb_ppc_netflow[column_name], dtype=float
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
