#!/usr/bin/env python3

import time
from datetime import datetime

import ets_fiber_assigner.netflow as nf
import numpy as np
from astropy.table import Table, join
from loguru import logger
from scipy.optimize import minimize

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

_CLASSIC_NETFLOW_PRIORITY_COSTS = {
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
_CLASSIC_NETFLOW_PARTIAL_OBSERVATION_COST = 5e4
_CLASSIC_NETFLOW_CALIBRATION_COST = 2000


def set_bench(bench_model):
    global bench
    bench = bench_model


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


def classic_build_classdict(_tb_tgt=None):
    classdict = build_classdict()
    classdict.update(
        {
            f"sci_P{priority}": {
                "nonObservationCost": non_observation_cost,
                "partialObservationCost": _CLASSIC_NETFLOW_PARTIAL_OBSERVATION_COST,
                "calib": False,
            }
            for priority, non_observation_cost in _CLASSIC_NETFLOW_PRIORITY_COSTS.items()
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
    observation_time="2026-01-10T10:00:00Z",
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

    return res, telescopes, netflow_targets


def fiber_allocate(
    tb_tgt,
    single_ppc_mode=False,
    ppc_candidate=None,
    observation_time="2026-01-10T10:00:00Z",
    num_reserved_fibers=0,
    fiber_non_allocation_cost=0.0,
    classdict_override=None,
    backup=False,
):
    if single_ppc_mode:
        if ppc_candidate is None:
            raise ValueError("ppc_candidate must be provided when single_ppc_mode=True")

        ppc_ra, ppc_dec, ppc_pa = ppc_candidate
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
        from .run_PPP import PFS_FoV

        time_start = time.time()
        logger.info("[S3] Run netflow started")

        ppc_records = []
        ppc_list = tb_tgt.meta["PPC"]
        today_date = datetime.now().strftime("%y%m%d")
        resolution = tb_tgt["resolution"][0]

        target_indices_in_group = set()
        for ppc_row in ppc_list:
            target_indices_in_group.update(
                PFS_FoV(ppc_row[1], ppc_row[2], ppc_row[3], tb_tgt)
            )

        if len(target_indices_in_group) > 0:
            tb_tgt_in_group = tb_tgt[sorted(target_indices_in_group)]

            logger.info(
                f"[S3] Group {1:3d}: nppc = {len(ppc_list):5d}, n_tgt = {len(tb_tgt_in_group):6d}"
            )

            assignments, telescopes, netflow_targets = run_netflow(
                ppc_list,
                tb_tgt_in_group,
                num_reserved_fibers=num_reserved_fibers,
                fiber_non_allocation_cost=fiber_non_allocation_cost,
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


def optimize_non_observation_costs(
    _tb_tgt,
    ppc_lst,
    scale_range=(0.05, 50.0),
    weight_total=1.0,
    weight_p0=3.0,
    weight_p1=2.0,
    focus_max=10,
    otime="2026-03-24T13:00:00Z",
    debug=False,
):
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
        return {
            tgt_lst_netflow[tidx].ID
            for vis in res
            for tidx in vis.keys()
        }

    def _score_assignment_counts(assigned_ids, focus_max=10):
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
        keys = sorted(key for key in counter_like.keys() if key != 999)
        if 999 in counter_like:
            keys.append(999)
        return {key: counter_like[key] for key in keys}

    def build_costs(scales):
        scaled_costs = base_non_obs_costs * np.asarray(scales, dtype=float)[scale_group_index]
        clipped_costs = np.minimum(scaled_costs, partial_cost_caps)
        return {
            key: float(cost)
            for key, cost in zip(science_keys, clipped_costs)
        }

    evaluation_cache = {}

    def evaluate_scales(scales):
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
        score = weight_total * total_norm + weight_p0 * p0_norm + weight_p1 * p1_norm

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
            logger.info(_sorted_counts(Counter(priority_by_code[code] for code in assigned_ids)))
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

    from .run_PPP import PFS_FoV

    resolution = tb_tgt["resolution"][0]
    proposal_token = tb_tgt["proposal_id"][0].split("-")[1]

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
            tb_ppc_netflow["ppc_code"] = [
                f"cla_{resolution}_{proposal_token}_{pointing_index}"
                for pointing_index in range(1, len(tb_ppc_netflow) + 1)
            ]
            tb_ppc_netflow = tb_ppc_netflow[output_columns]

    if tb_tgt.meta["PPC_origin"] == "usr":
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
