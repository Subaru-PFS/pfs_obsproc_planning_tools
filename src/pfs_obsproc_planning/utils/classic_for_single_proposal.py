#!/usr/bin/env python3

import numpy as np
from astropy import units as u
from astropy.table import Table, vstack
from loguru import logger

_DEFAULT_SINGLE_PROGRAM_PRIORITY_POLICY = {
    "tracked_priorities": list(range(10)) + [999],
    "emphasized_priority": 1,
}


def get_single_proposal_requirements(proposal_id):
    proposalId = str(proposal_id) if proposal_id is not None else None

    if proposalId == "S25A-UH006-B":
        return {
            "tracked_priorities": list(range(20, 30)) + [999],
            "emphasized_priority": 21,
            "single_program_pointings": {
                0: {"mode": "fixed", "ra": 150.08220377, "dec": 2.18805709, "pa": 92.677787},
                1: {
                    "mode": "optimize_initial_guess",
                    "initial_guess": [150.08220377, 2.18805709, 92.677787],
                },
                2: {"mode": "fixed", "ra": 270.29782837, "dec": 65.7456042, "pa": 94.62414553},
            },
            "classic_ppc_prefix": "cla_L_uh006",
            "remaining_priority_increment_value": 20,
            "remaining_priority_increment_exptime_ppp": 900,
            "extra_target_paths": [
                "/home/wanqqq/examples/run_2503/S25A-UH006-B/input/Sanders_Extra_PFS_Targets_2025A_rev.csv",
                "/home/wanqqq/examples/run_2503/S25A-UH006-B/input/PFS_EDFN_March22_rev.csv",
            ],
            "i2_mag_split": {"threshold": 24, "bright": 900, "faint": 1800},
        }

    if proposalId == "S25B-UH016-A":
        return {
            "single_exptime_override": 27000.0,
            "input_catalog_priority_offsets": {10289: 10},
            "export_priority_restore": {10289: 10},
        }

    if proposalId == "S25A-UH022-A":
        return {
            "priority_exptime_overrides": {0: 12000.0},
        }

    if proposalId == "S25A-UH041-A":
        return {
            "force_exptime_ppp": 900,
        }

    if proposalId == "S25B-UH041-A":
        return {
            "exptime_remap": {
                450.0: [14400.0, 19800.0],
                900.0: [28800.0, 39600.0],
                1350.0: [43200.0, 59400.0],
            },
            "single_program_filter_thresholds": {1: 57600.0, 2: 43200.0, 3: 28800.0},
            "single_program_fixed_pointing": (36.49583333, -4.49444444),
            "single_program_mode": "LR",
        }

    if proposalId == "S25B-TE421-K":
        return {
            "force_exptime_ppp": 900,
            "single_program_mode": "MR",
        }

    if proposalId == "S25B-116N":
        return {
            "single_program_mode": "LR",
        }

    if proposalId == "S26A-UH022-A":
        return {
            "single_program_filter_thresholds": {1: 57600.0, 2: 43200.0, 3: 28800.0},
            "single_program_fixed_pointing": (150.029167, 2.195718),
            "single_program_mode": "LR",
        }

    if proposalId == "S26A-TE007-G":
        return {
            "ra_less_than_exptime_override": {"ra_max": 210.0, "exptime": 2700.0},
            "single_program_optimization": {
                "central_ra": 203.879558,
                "central_dec": 59.047015,
                "priority_limit": 1,
                "bounds": [(203.779558, 203.979558), (58.947015, 59.147015)],
            },
            "single_program_mode": "LR",
        }

    if proposalId == "S26A-091":
        return {
            "single_program_optimization": {
                "central_ra": 150.119167,
                "central_dec": 2.205833,
                "priority_limit": 2,
                "bounds": [(149.119167, 151.119167), (1.705833, 2.705833)],
                "max_ppc_count": 2,
            },
            "single_program_mode": "MR",
        }

    return {}


def _get_proposal_policy(proposal_id):
    return get_single_proposal_requirements(proposal_id)


def _get_single_program_mode(proposal_id):
    if proposal_id is None:
        return None
    return _get_proposal_policy(proposal_id).get("single_program_mode")


def _get_single_program_ppc_pa(proposal_id, default=0.0):
    if proposal_id is None:
        return default
    return _get_proposal_policy(proposal_id).get("ppc_pa", default)


def _get_import_user_ppc_from_db(proposal_id, default=True):
    if proposal_id is None:
        return default
    return _get_proposal_policy(proposal_id).get("import_user_ppc_from_db", default)


def _apply_configured_row_level_adjustments(tb_tgt):
    if "proposal_id" not in tb_tgt.colnames:
        return tb_tgt

    for proposal_id in sorted(set(tb_tgt["proposal_id"])):
        policy = _get_proposal_policy(str(proposal_id))
        proposal_mask = tb_tgt["proposal_id"] == proposal_id

        single_exptime_override = policy.get("single_exptime_override")
        if single_exptime_override is not None:
            tb_tgt["single_exptime"][proposal_mask] = single_exptime_override

        for priority_value, exptime_value in policy.get(
            "priority_exptime_overrides", {}
        ).items():
            priority_mask = tb_tgt["priority"] == priority_value
            tb_tgt["exptime"][proposal_mask & priority_mask] = exptime_value

        for input_catalog_id, priority_offset in policy.get(
            "input_catalog_priority_offsets", {}
        ).items():
            catalog_mask = tb_tgt["input_catalog_id"] == input_catalog_id
            tb_tgt["priority"][proposal_mask & catalog_mask] += priority_offset

        for remapped_exptime, source_exptimes in policy.get("exptime_remap", {}).items():
            source_mask = np.isin(tb_tgt["exptime"], source_exptimes)
            tb_tgt["exptime"][proposal_mask & source_mask] = remapped_exptime

        ra_exptime_override = policy.get("ra_less_than_exptime_override")
        if ra_exptime_override is not None:
            ra_mask = tb_tgt["ra"] < ra_exptime_override["ra_max"]
            tb_tgt["exptime"][proposal_mask & ra_mask] = ra_exptime_override["exptime"]

    return tb_tgt


def _append_single_program_extra_targets(tb_tgt, policy, has_proposal_id, single_exptime_seconds):
    extra_tables = [Table.read(path) for path in policy.get("extra_target_paths", [])]
    if len(extra_tables) == 0:
        return tb_tgt

    for extra_targets in extra_tables:
        extra_targets["ob_code"] = extra_targets["ob_code"].astype("str")

        for col in ["filter_g", "filter_r", "filter_i", "filter_z", "filter_y"]:
            tb_tgt[col] = tb_tgt[col].astype("str")
            extra_targets[col] = extra_targets[col].astype("str")

        for col in ["psf_flux_g", "psf_flux_r", "psf_flux_i", "psf_flux_z", "psf_flux_y"]:
            tb_tgt[col] = tb_tgt[col].astype(float)
            extra_targets[col] = extra_targets[col].astype(float)

        for col in [
            "psf_flux_error_g",
            "psf_flux_error_r",
            "psf_flux_error_i",
            "psf_flux_error_z",
            "psf_flux_error_y",
        ]:
            tb_tgt[col] = tb_tgt[col].astype(float)
            extra_targets[col] = extra_targets[col].astype(float)

        tb_tgt = vstack([tb_tgt, extra_targets])

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

    i2_mag_split = policy.get("i2_mag_split")
    if i2_mag_split is not None:
        tb_tgt["i2_mag"] = (tb_tgt["psf_flux_i"] * u.nJy).to(u.ABmag)
        tb_tgt["exptime_PPP"][tb_tgt["i2_mag"] < i2_mag_split["threshold"]] = i2_mag_split["bright"]
        tb_tgt["exptime_PPP"][tb_tgt["i2_mag"] >= i2_mag_split["threshold"]] = i2_mag_split["faint"]

    logger.info(f"Input target list: {tb_tgt}")
    return tb_tgt


def _apply_single_program_proposal_adjustments(tb_tgt, proposal_id, has_proposal_id):
    policy = _get_proposal_policy(proposal_id)
    single_exptime_seconds = tb_tgt.meta["single_exptime"]

    if policy.get("extra_target_paths"):
        tb_tgt = _append_single_program_extra_targets(
            tb_tgt,
            policy,
            has_proposal_id,
            single_exptime_seconds,
        )

    force_exptime_ppp = policy.get("force_exptime_ppp")
    if force_exptime_ppp is not None:
        tb_tgt["exptime_PPP"] = force_exptime_ppp

    return tb_tgt


def apply_proposal_target_adjustments(tb_tgt):
    has_proposal_id = "proposal_id" in tb_tgt.colnames
    proposal_ids = sorted(set(tb_tgt["proposal_id"])) if has_proposal_id else []
    tb_tgt = _apply_configured_row_level_adjustments(tb_tgt)

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

    if len(proposal_ids) == 1:
        tb_tgt = _apply_single_program_proposal_adjustments(
            tb_tgt,
            str(proposal_ids[0]),
            has_proposal_id,
        )

    return tb_tgt
