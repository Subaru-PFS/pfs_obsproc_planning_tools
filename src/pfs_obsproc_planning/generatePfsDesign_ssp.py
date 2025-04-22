#!/usr/bin/env python3
# generatePfsDesign_ssp.py : PPP+qPlan+SFA

import os, sys
# skip print out message
sys.stdout = open(os.devnull, 'w')

import warnings
from datetime import datetime, timedelta

import pytz

hawaii_tz = pytz.timezone("Pacific/Honolulu")

import git
import numpy as np
import pandas as pd
import toml
from astropy.table import Table, vstack
from logzero import logger

warnings.filterwarnings("ignore")

import ets_fiber_assigner.netflow as nf
from pfs_design_tool import reconfigure_fibers_ppp as sfa
from pfs_design_tool.pointing_utils import nfutils

from .opefile import OpeFile


def read_conf(conf):
    config = toml.load(conf)

    # get some parameters from environmet variables
    # if not in the config file and already set as enviroment variables
    if ("pfs_utils_dir" not in config["sfa"]) and (
        os.environ.get("PFS_UTILS_DIR") is not None
    ):
        config["sfa"]["pfs_utils_dir"] = os.environ.get("PFS_UTILS_DIR")
        logger.info(
            f"Setting config['sfa']['pfs_utils_dir'] from environment as {config['sfa']['pfs_utils_dir']}"
        )

    if ("pfs_instdata_dir" not in config["sfa"]) and (
        os.environ.get("PFS_INSTDATA_DIR") is not None
    ):
        config["sfa"]["pfs_instdata_dir"] = os.environ.get("PFS_INSTDATA_DIR")
        logger.info(
            f"Setting config['sfa']['pfs_instdata_dir'] from environment as {config['sfa']['pfs_instdata_dir']}"
        )

    if ("cobra_coach_dir" not in config["sfa"]) and (
        os.environ.get("COBRA_COACH_DIR") is not None
    ):
        config["sfa"]["cobra_coach_dir"] = os.environ.get("COBRA_COACH_DIR")
        logger.info(
            f"Setting config['sfa']['cobra_coach_dir'] from environment as {config['sfa']['cobra_coach_dir']}"
        )

    if ("template" not in config["ope"]) and (
        os.environ.get("OPE_TEMPLATE") is not None
    ):
        config["ope"]["template"] = os.environ.get("OPE_TEMPLATE", None)
        logger.info(
            f"Setting config['ope']['template'] from environmentas {config['ope']['template']}"
        )

    if "guidestar_mag_min" not in config["sfa"]:
        config["sfa"]["guidestar_mag_min"] = 12
    if "guidestar_mag_max" not in config["sfa"]:
        config["sfa"]["guidestar_mag_max"] = 22
    if "guidestar_neighbor_mag_min" not in config["sfa"]:
        config["sfa"]["guidestar_neighbor_mag_min"] = 21
    if "guidestar_minsep_deg" not in config["sfa"]:
        config["sfa"]["guidestar_minsep_deg"] = 0.0002778

    return config


def check_versions(package, repo_url, repo_path, version_desire):
    """Clone the dependent package from repo_url to repo_path, and checkout to version_desire branch/tag."""

    def clone_repo(repo_url, repo_path="repo"):
        """Clone a GitHub repository if it’s not already cloned."""
        if not os.path.exists(repo_path):
            git.Repo.clone_from(repo_url, repo_path)
            logger.info(f"({package}) Cloned repository to {repo_path}")
        else:
            logger.info(f"({package}) Repository already exists at {repo_path}")

    def fetch_all_branches_and_tags(repo):
        """Fetch all branches and tags from a repository."""
        repo.remotes.origin.fetch()
        logger.info(f"({package}) Fetched all branches and tags.")

    def get_branches_and_tags(repo):
        """Get a list of all branches and tags."""
        branches = [ref.name for ref in repo.remote().refs]
        tags = [tag.name for tag in repo.tags]
        return branches, tags

    def get_current_tag_branch(repo):
        """Get the current tag of the repository if HEAD matches a tag."""
        # Get the current commit
        current_commit = repo.head.commit

        # Find if the current commit matches any tag
        for tag in repo.tags:
            if tag.commit == current_commit:
                logger.info(f"({package}) Current tag = {tag.name}")

        # Find if the current commit matches any branch
        for ref in repo.remote().refs:
            if ref.commit == current_commit:
                logger.info(f"({package}) Current branch = {ref.name}")

    def checkout_version(repo, version):
        """Checkout a specified branch or tag."""
        version = version.strip()
        if version == "":
            logger.info(f"({package}) Do not change the current branch/tag.")
            get_current_tag_branch(repo)
        elif version in [ref.name for ref in repo.remote().refs] or version in [
            tag.name for tag in repo.tags
        ]:
            repo.git.checkout(version)
            logger.info(f"({package}) Checked out to {version}.")
            get_current_tag_branch(repo)
        else:
            logger.warning(
                f"({package}) Version '{version}' not found in branches or tags."
            )
            get_current_tag_branch(repo)

    # Step 1: Clone the repository if not done already
    clone_repo(repo_url, repo_path)

    # Step 2: Load the repository and fetch all branches and tags
    repo = git.Repo(repo_path)
    fetch_all_branches_and_tags(repo)

    # Step 3: Get list of branches and tags
    get_branches_and_tags(repo)

    # Step 4: Checkout the specified branch or tag
    checkout_version(repo, version_desire)


class GeneratePfsDesign_ssp(object):
    def __init__(self, config, workDir=".", repoDir=None):
        self.config = config
        self.workDir = workDir
        self.repoDir = repoDir

        ## configuration file ##
        self.conf = read_conf(os.path.join(self.workDir, self.config))

        self.cobraCoachDir = os.path.join(self.conf["sfa"]["cobra_coach_dir"])

        # check if pfs_instdata exists; if no, clone from GitHub when not found; if version specified, switch to it
        repo_url = "https://github.com/Subaru-PFS/pfs_instdata.git"
        repo_path = self.conf["sfa"]["pfs_instdata_dir"]
        version_desire = self.conf["sfa"]["pfs_instdata_ver"]

        check_versions("pfs_instdata", repo_url, repo_path, version_desire)

        # check if pfs_utils exists; if no, clone from GitHub when not found; if version specified, switch to it
        repo_url = "https://github.com/Subaru-PFS/pfs_utils.git"
        try:
            import pfs.utils

            repo_path = os.path.join(pfs.utils.__path__[0], "../../../")
            os.environ["PFS_UTILS_DIR"] = os.path.join(
                pfs.utils.__path__[0], "../../../"
            )
        except:
            repo_path = self.conf["sfa"]["pfs_utils_dir"]
        version_desire = self.conf["sfa"]["pfs_utils_ver"]

        check_versions("pfs_utils", repo_url, repo_path, version_desire)

        return None

    def update_config(self):
        self.conf = read_conf(os.path.join(self.workDir, self.config))

    def ssp_tgt_validate(self, tb, ppc_code, tgt_type):
        validate_success = True

        # fmt: off
        # check whether required columns included
        expected_science = {
            "obj_id"           : {"dtype": int  , "default": None     },
            "ra"               : {"dtype": float, "default": None     },
            "dec"              : {"dtype": float, "default": None     },
            "pmra"             : {"dtype": float, "default": 0.0      },
            "pmdec"            : {"dtype": float, "default": 0.0      },
            "parallax"         : {"dtype": float, "default": 1.0e-7   },
            "epoch"            : {"dtype": str  , "default": "J2000.0"},
            "target_type_id"   : {"dtype": int  , "default": None     },
            "input_catalog_id" : {"dtype": int  , "default": None     },
            "ob_code"          : {"dtype": str  , "default": None     },
            "proposal_id"      : {"dtype": str  , "default": None     },
            "priority"         : {"dtype": int  , "default": None     },
            "effective_exptime": {"dtype": float, "default": None     },
            "filter_g"         : {"dtype": str  , "default": None     },
            "filter_r"         : {"dtype": str  , "default": None     },
            "filter_i"         : {"dtype": str  , "default": None     },
            "filter_z"         : {"dtype": str  , "default": None     },
            "filter_y"         : {"dtype": str  , "default": None     },
            "psf_flux_g"       : {"dtype": float, "default": None     },
            "psf_flux_r"       : {"dtype": float, "default": None     },
            "psf_flux_i"       : {"dtype": float, "default": None     },
            "psf_flux_z"       : {"dtype": float, "default": None     },
            "psf_flux_y"       : {"dtype": float, "default": None     },
            "psf_flux_error_g" : {"dtype": float, "default": None     },
            "psf_flux_error_r" : {"dtype": float, "default": None     },
            "psf_flux_error_i" : {"dtype": float, "default": None     },
            "psf_flux_error_z" : {"dtype": float, "default": None     },
            "psf_flux_error_y" : {"dtype": float, "default": None     },
            "cobraId"          : {"dtype": int  , "default": None     },
            "pfi_X"            : {"dtype": float, "default": None     },
            "pfi_Y"            : {"dtype": float, "default": None     },
        }

        expected_sky = {
            "obj_id"          : {"dtype": int  , "default": None},
            "ra"              : {"dtype": float, "default": None},
            "dec"             : {"dtype": float, "default": None},
            "target_type_id"  : {"dtype": int  , "default": None},
            "input_catalog_id": {"dtype": int  , "default": None},
            "cobraId"         : {"dtype": int  , "default": None},
            "pfi_X"           : {"dtype": float, "default": None},
            "pfi_Y"           : {"dtype": float, "default": None},
        }

        expected_fluxstd = {
            "obj_id"          : {"dtype": int  , "default": None     },
            "ra"              : {"dtype": float, "default": None     },
            "dec"             : {"dtype": float, "default": None     },
            "epoch"           : {"dtype": str  , "default": "J2000.0"},
            "pmra"            : {"dtype": float, "default": 0.0      },
            "pmdec"           : {"dtype": float, "default": 0.0      },
            "parallax"        : {"dtype": float, "default": 1.0e-7   },
            "target_type_id"  : {"dtype": int  , "default": None     },
            "input_catalog_id": {"dtype": int  , "default": None     },
            "prob_f_star"     : {"dtype": float, "default": 0.0      },
            "filter_g"        : {"dtype": str  , "default": None     },
            "filter_r"        : {"dtype": str  , "default": None     },
            "filter_i"        : {"dtype": str  , "default": None     },
            "filter_z"        : {"dtype": str  , "default": None     },
            "filter_y"        : {"dtype": str  , "default": None     },
            "psf_flux_g"      : {"dtype": float, "default": None     },
            "psf_flux_r"      : {"dtype": float, "default": None     },
            "psf_flux_i"      : {"dtype": float, "default": None     },
            "psf_flux_z"      : {"dtype": float, "default": None     },
            "psf_flux_y"      : {"dtype": float, "default": None     },
            "psf_flux_error_g": {"dtype": float, "default": None     },
            "psf_flux_error_r": {"dtype": float, "default": None     },
            "psf_flux_error_i": {"dtype": float, "default": None     },
            "psf_flux_error_z": {"dtype": float, "default": None     },
            "psf_flux_error_y": {"dtype": float, "default": None     },
            "cobraId"         : {"dtype": int  , "default": None     },
            "pfi_X"           : {"dtype": float, "default": None     },
            "pfi_Y"           : {"dtype": float, "default": None     },
        }
        # fmt: on

        if tgt_type == "science":
            req_cols = list(expected_science.keys())
            expected = expected_science
        elif tgt_type == "sky":
            req_cols = list(expected_sky.keys())
            expected = expected_sky
        elif tgt_type == "fluxstd":
            req_cols = list(expected_fluxstd.keys())
            expected = expected_fluxstd

        missing_cols = [col for col in req_cols if col not in tb.colnames]
        if missing_cols:
            validate_success = False
            logger.error(
                f"[Validation of tgtLists] The following required columns are missing ({ppc_code, tgt_type}): {missing_cols}"
            )

        # check if datatype of column is correct, and default value is set for no input
        for col, col_info in expected.items():
            if col in tb.colnames:
                col_dtype = tb[col].dtype
                expected_dtype = col_info["dtype"]

                # For string types, expected_dtype == str; check dtype.kind for Unicode ('U') or bytes ('S').
                if expected_dtype == str:
                    # Check for string: dtype.kind should be 'U' (Unicode) or 'S' (bytes)
                    if col_dtype.kind not in ["U", "S"]:
                        validate_success = False
                        logger.error(
                            f"Column '{col}' expected to be a string type but got {col_dtype}"
                        )
                elif expected_dtype == float:
                    # Check for any floating type (np.float32, np.float64, or Python float)
                    if not np.issubdtype(col_dtype, np.floating):
                        validate_success = False
                        logger.error(
                            f"Column '{col}' expected to be a float type but got {col_dtype}"
                        )
                elif expected_dtype == int:
                    # Check for any integer type
                    if not np.issubdtype(col_dtype, np.integer):
                        validate_success = False
                        logger.error(
                            f"Column '{col}' expected to be an int type but got {col_dtype}"
                        )

                # If a default is provided, check that at least one value is non-missing.
                default_val = col_info["default"]
                if default_val is not None:
                    # For numeric columns, consider a value missing if it is NaN.
                    for i, val in enumerate(tb[col]):
                        if val is None:
                            logger.error(
                                f"Column '{col}' has None; expected default {default_val}."
                            )
                            validate_success = False
                            break
                        # Check for NaN (only applicable if the value is a float).
                        if isinstance(val, float) and np.isnan(val):
                            logger.error(
                                f"Column '{col}' has np.nan; expected default {default_val}."
                            )
                            validate_success = False
                            break

        # check no duplicated cobraId
        unique_vals, counts = np.unique(tb["cobraId"], return_counts=True)
        duplicates = unique_vals[counts > 1]
        if len(duplicates) > 0:
            for dup_val in duplicates:
                dup_obj_ids = list(tb["obj_id"][tb["cobraId"] == dup_val])
                logger.error(
                    f"[Validation of tgtLists] Found duplicates in 'cobraId' ({ppc_code, tgt_type}): cobraId={dup_val} assigned to {dup_obj_ids}"
                )
            validate_success = False

        # check flux columns
        filter_category_sci = {
            "g": ["g_hsc", "g_ps1", "g_sdss", "bp_gaia"],
            "r": ["r_old_hsc", "r2_hsc", "r_ps1", "r_sdss", "g_gaia"],
            "i": ["i_old_hsc", "i2_hsc", "i_ps1", "i_sdss", "rp_gaia"],
            "z": ["z_hsc", "z_ps1", "z_sdss"],
            "y": ["y_hsc", "y_ps1"],
            "j": [],
        }

        filter_category_fluxstd = {
            "g": ["g_ps1", "bp_gaia"],
            "r": ["r_ps1", "g_gaia"],
            "i": ["i_ps1", "rp_gaia"],
            "z": ["z_ps1"],
            "y": ["y_ps1"],
            "j": [],
        }

        if tgt_type == "science":
            filter_category = filter_category_sci
        elif tgt_type == "fluxstd":
            filter_category = filter_category_fluxstd

        for band in ["g", "r", "i", "z", "y"]:
            if tgt_type == "sky":
                continue

            col_name = f"filter_{band}"
            valid_values = set(
                filter_category.get(band, [])
            )  # e.g. ["g_hsc", "g_ps1", "g_sdss", "bp_gaia"] for band="g"

            invalid_rows = []
            for i, val in enumerate(tb[col_name]):
                if np.ma.is_masked(val):
                    continue
                if val == None:
                    continue
                if val not in valid_values:
                    invalid_rows.append((i, val))

            if invalid_rows:
                validate_success = False
                invalid_str = ", ".join(
                    f"Row {row_idx} => '{bad_val}'" for row_idx, bad_val in invalid_rows
                )
                logger.error(
                    f"[Validation of tgtLists] Invalid values in flux column '{col_name}' ({ppc_code}, {tgt_type})"  #: {invalid_str} "
                )

        # check flux in at least one band / all bands are there for science / fluxstd
        flux_cols = [
            "psf_flux_g",
            "psf_flux_r",
            "psf_flux_i",
            "psf_flux_z",
            "psf_flux_y",
        ]

        if tgt_type == "science":
            flux_data = np.array([tb[col].data.astype(float) for col in flux_cols])
            valid_mask = np.any(
                flux_data > 0, axis=0
            )  # flux in at least one band should be there
            invalid_rows = np.where(~valid_mask)[0]
            if len(invalid_rows) > 0:
                validate_success = False
                logger.error(
                    f"[Validation of tgtLists] Rows lack flux info ({ppc_code}, {tgt_type}): {list(invalid_rows)}"
                )

        elif tgt_type == "fluxstd":
            flux_data = np.array([tb[col].data for col in flux_cols])
            valid_mask = np.all(
                flux_data > 0, axis=0
            )  # flux in all bands should be there
            invalid_rows = np.where(~valid_mask)[0]
            if len(invalid_rows) > 0:
                validate_success = False
                logger.error(
                    f"[Validation of tgtLists] Rows lack flux info ({ppc_code}, {tgt_type}): {list(invalid_rows)}"
                )

        # check duplicated obj_id / ob_code
        df = tb.to_pandas()  # If tb is already a DataFrame, skip this line.

        if tgt_type == "science":
            duplicates_mask = df.duplicated(subset=["ob_code"], keep=False)
            if duplicates_mask.any():
                validate_success = False
                duplicated_rows = df[duplicates_mask]["ob_code"]
                logger.error(
                    f"[Validation of tgtLists] Found duplicates in 'ob_code' ({ppc_code}, {tgt_type}):\n{duplicated_rows}"
                )

        duplicates_mask = df.duplicated(subset=["obj_id"], keep=False)
        if duplicates_mask.any():
            validate_success = False
            duplicated_rows = df[duplicates_mask]["obj_id"]
            logger.error(
                f"[Validation of tgtLists] Found duplicates in 'obj_id' ({ppc_code}, {tgt_type}):\n{duplicated_rows}"
            )

        # check proposal_id, input_catalog_id & target_type
        if tgt_type == "science":
            logger.info(
                f"{ppc_code} ({tgt_type}): psl_id = {set(tb['proposal_id'])}, tgt_type = {set(tb['target_type_id'])}, catId = {set(tb['input_catalog_id'])}"
            )

            proposal_id = set(tb["proposal_id"])
            proposal_id_req = {"S25A-OT02"}
            if proposal_id != proposal_id_req:
                validate_success = False
                logger.error(
                    f"[Validation of tgtLists] Proposal_id is incorrect (should be S25A-OT02; {ppc_code}, {tgt_type}): {proposal_id}"
                )

            target_type = set(tb["target_type_id"])
            if target_type != {1}:
                validate_success = False
                logger.error(
                    f"[Validation of tgtLists] Target_type for science is incorrect (should be 1; {ppc_code}, {tgt_type}): {target_type}"
                )

            catId = set(tb["input_catalog_id"])
            unexpected_Id = catId - {10091, 10092, 10093}
            if len(unexpected_Id) > 0:
                validate_success = False
                logger.error(
                    f"[Validation of tgtLists] Incorrect catId (should be 10091/2/3; {ppc_code}, {tgt_type}): {unexpected_Id}"
                )

        elif tgt_type == "sky":
            logger.info(
                f"{ppc_code} ({tgt_type}): tgt_type = {set(tb['target_type_id'])}, catId = {set(tb['input_catalog_id'])}"
            )

            target_type = set(tb["target_type_id"])
            if target_type != {2}:
                validate_success = False
                logger.error(
                    f"[Validation of tgtLists] Target_type for sky is incorrect (should be 2; {ppc_code}, {tgt_type}): {target_type}"
                )

            catId = set(tb["input_catalog_id"])
            unexpected_Id = catId - {1006, 1007, 10091, 10092, 10093}
            if len(unexpected_Id) > 0:
                validate_success = False
                logger.error(
                    f"[Validation of tgtLists] Incorrect catId (should be 1006/7 or 10091/2/3; {ppc_code}, {tgt_type}): {unexpected_Id}"
                )

        elif tgt_type == "fluxstd":
            logger.info(
                f"{ppc_code} ({tgt_type}): tgt_type = {set(tb['target_type_id'])}, catId = {set(tb['input_catalog_id'])}"
            )

            target_type = set(tb["target_type_id"])
            if target_type != {3}:
                validate_success = False
                logger.error(
                    f"[Validation of tgtLists] Target_type for fluxstd is incorrect (should be 3; {ppc_code}, {tgt_type}): {target_type}"
                )

            catId = set(tb["input_catalog_id"])
            unexpected_Id = catId - {3006, 10091, 10092, 10093}
            if len(unexpected_Id) > 0:
                validate_success = False
                logger.error(
                    f"[Validation of tgtLists] Incorrect catId (should be 3006 or 10091/2/3; {ppc_code}, {tgt_type}): {unexpected_Id}"
                )

        return validate_success

    def read_tgt(self, ppc_code, WG):
        logger.info(f"[{WG}] Reading in target lists for pointing - {ppc_code}")
        filepath_sci = os.path.join(
            self.workDir, "targets", WG, "science", f"{ppc_code}.ecsv"
        )
        filepath_sky = os.path.join(
            self.workDir, "targets", WG, "sky", f"{ppc_code}.ecsv"
        )
        filepath_fluxstd = os.path.join(
            self.workDir, "targets", WG, "fluxstd", f"{ppc_code}.ecsv"
        )

        if not os.path.isfile(filepath_sci):
            logger.error(f"[read_tgt] Missing science file for {ppc_code}: {filepath_sci}")
            return Table(), Table(), Table()
        else:
            tb_sci = Table.read(filepath_sci)

        if not os.path.isfile(filepath_sky):
            logger.error(f"[read_tgt] Missing sky file for {ppc_code}: {filepath_sky}")
            return Table(), Table(), Table()
        else:
            tb_sky = Table.read(filepath_sky)
            
        if not os.path.isfile(filepath_fluxstd):
            logger.error(f"[read_tgt] Missing fluxstd file for {ppc_code}: {filepath_fluxstd}")
            return Table(), Table(), Table()
        else:
            tb_fluxstd = Table.read(filepath_fluxstd)

        tb_sci["cidx"] = tb_sci["cobraId"] - 1
        tb_sky["cidx"] = tb_sky["cobraId"] - 1
        tb_fluxstd["cidx"] = tb_fluxstd["cobraId"] - 1

        # validate input lists
        validate_success_sci = self.ssp_tgt_validate(tb_sci, ppc_code, "science")
        validate_success_sky = self.ssp_tgt_validate(tb_sky, ppc_code, "sky")
        validate_success_fluxstd = self.ssp_tgt_validate(
            tb_fluxstd, ppc_code, "fluxstd"
        )

        if validate_success_sci and validate_success_sky and validate_success_fluxstd:
            logger.info(f"[Validation of tgtLists] Validation passed ({ppc_code})")

        return tb_sci, tb_sky, tb_fluxstd

    def ssp_ppc_validate(self, tb):
        validate_success = True

        # check if all required columns included in ppcList
        req_cols = [
            "ppc_code",
            "ppc_ra",
            "ppc_dec",
            "ppc_pa",
            "ppc_resolution",
            "ppc_obstime",
            "ppc_exptime",
            "ppc_nframes",
        ]

        missing_cols = [col for col in req_cols if col not in tb.colnames]
        if missing_cols:
            validate_success = False
            logger.error(
                f"[Validation of ppcList] The following required columns are missing: {missing_cols}"
            )

        # check if no duplicated ppc_code in ppcList
        df = tb.to_pandas()

        duplicates_mask = df.duplicated(subset=["ppc_code"], keep=False)
        if duplicates_mask.any():
            validate_success = False
            duplicated_rows = df[duplicates_mask]["ppc_code"]
            logger.error(
                f"[Validation of ppcList] Found duplicates in 'ppc_code':\n{duplicated_rows}"
            )

        # check if nframes > 1
        if np.any(tb["ppc_nframes"] <= 1):
            validate_success = False
            logger.error("[Validation of ppcList] ppc_nframes should be more than 1")

        # check N_guide star for each pointing
        ppc_obstime_utc = []
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        ]
        for hst_string in tb["ppc_obstime"]:
            dt_naive = None
            for format_str in formats:
                try:
                    dt_naive = datetime.strptime(hst_string, format_str)
                except (ValueError, TypeError):
                    continue

            if dt_naive is None:
                raise ValueError(
                    f"Time data '{hst_string}' does not match any known format"
                )

            dt_hst = hawaii_tz.localize(dt_naive)
            dt_utc = dt_hst.astimezone(pytz.utc)
            ppc_obstime_utc.append(dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))

        tb["ppc_obstime_utc"] = ppc_obstime_utc

        for tb_ppc_t in tb:
            code = tb_ppc_t['ppc_code']
            guidestars = sfa.designutils.generate_guidestars_from_gaiadb(
                tb_ppc_t["ppc_ra"],
                tb_ppc_t["ppc_dec"],
                tb_ppc_t["ppc_pa"],
                tb_ppc_t["ppc_obstime_utc"],  # obstime should be in UTC
                telescope_elevation=None,
                conf=self.conf,
                guidestar_mag_min=self.conf["sfa"]["guidestar_mag_min"],
                guidestar_mag_max=self.conf["sfa"]["guidestar_mag_max"],
                guidestar_neighbor_mag_min=self.conf["sfa"][
                    "guidestar_neighbor_mag_min"
                ],
                guidestar_minsep_deg=self.conf["sfa"]["guidestar_minsep_deg"],
            )

            # build a list of (cam_id, count)
            counts = [(cam+1, int((guidestars.agId == cam).sum())) for cam in range(6)]

            # single info line
            counts_str = ", ".join(f"AG‑Cam‑{cam}={cnt}" for cam, cnt in counts)
            logger.info(f"[Validation of ppcList] ({code}) {counts_str}")

            # individual warnings for any zero counts
            for cam, cnt in counts:
                if cnt == 0:
                    validate_success = False
                    logger.warning(f"[Validation of ppcList] ({code}) AG‑Cam‑{cam} has zero guide stars")

        return validate_success

    def makedesign(self, WG):
        logger.info(f"[For SSP] Make design for {WG}")

        # read ppcList.ecsv
        tb_ppc = Table.read(os.path.join(self.workDir, "targets", WG, "ppcList.ecsv"))

        mask = np.array(
            [
                (isinstance(val, str) and val.strip().lower() != "nan")
                or not (isinstance(val, str))
                for val in tb_ppc["ppc_obstime"]
            ]
        )  # True for pointings that are not NaN or "nan" in obstime
        tb_ppc = tb_ppc[mask]

        # Convert each timestamp from HST to UTC
        ppc_obstime_utc = []
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        ]
        for hst_string in tb_ppc["ppc_obstime"]:
            dt_naive = None
            for format_str in formats:
                try:
                    dt_naive = datetime.strptime(hst_string, format_str)
                except (ValueError, TypeError):
                    continue

            if dt_naive is None:
                raise ValueError(
                    f"Time data '{hst_string}' does not match any known format"
                )

            dt_hst = hawaii_tz.localize(dt_naive)
            dt_utc = dt_hst.astimezone(pytz.utc)
            ppc_obstime_utc.append(dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))

        tb_ppc["ppc_obstime_utc"] = ppc_obstime_utc
        tb_ppc["pfsDesignId"] = np.zeros(len(tb_ppc), dtype=np.int64)
        tb_ppc["pfsDesignId_hex"] = np.zeros(len(tb_ppc), dtype="U64")

        validate_success_ppc = self.ssp_ppc_validate(tb_ppc)

        for tb_ppc_t in tb_ppc:
            ppc_code = tb_ppc_t["ppc_code"]

            tb_sci, tb_sky, tb_fluxstd = self.read_tgt(ppc_code, WG)

            df_sci = Table.to_pandas(tb_sci[:])
            df_fluxstds = Table.to_pandas(tb_fluxstd[:])
            df_sky = Table.to_pandas(tb_sky[:])

            # make target list and dict of {tgt_idx: cidx}
            targets = []
            vis = {}

            def vis_generator(df):
                vis_ = {}
                for i in range(df.index.size):
                    vis_.update({i: int(df["cidx"].values[i])})
                return vis_

            target1 = nfutils.register_objects(df_sci, target_class="sci")
            vis1 = vis_generator(df_sci)

            target2 = nfutils.register_objects(df_fluxstds, target_class="cal")
            vis2 = vis_generator(df_fluxstds)

            target3 = nfutils.register_objects(df_sky, target_class="sky")
            vis3 = vis_generator(df_sky)

            targets += target1
            targets += target2
            targets += target3

            vis2_update = {k + len(vis1): v for k, v in vis2.items()}
            vis3_update = {k + len(vis1) + len(vis2): v for k, v in vis3.items()}

            vis.update(vis1)
            vis.update(vis2_update)
            vis.update(vis3_update)

            # make class_dict (not sure really needed or not)
            class_dict = {
                # Priorities correspond to the magnitudes of bright stars (in most case for the 2022 June Engineering)
                "sci_P0": {
                    "nonObservationCost": 5e10,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P1": {
                    "nonObservationCost": 4e10,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P2": {
                    "nonObservationCost": 2e10,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P3": {
                    "nonObservationCost": 1e10,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P4": {
                    "nonObservationCost": 1e9,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P5": {
                    "nonObservationCost": 1e8,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P6": {
                    "nonObservationCost": 1e7,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P7": {
                    "nonObservationCost": 1e6,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P8": {
                    "nonObservationCost": 1e5,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P9": {
                    "nonObservationCost": 1e4,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P10": {
                    "nonObservationCost": 1e3,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P11": {
                    "nonObservationCost": 100,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P12": {
                    "nonObservationCost": 10,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P13": {
                    "nonObservationCost": 5,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "sci_P9999": {  # fillers
                    "nonObservationCost": 1,
                    "partialObservationCost": 1e11,
                    "calib": False,
                },
                "cal": {
                    "numRequired": 200,
                    "nonObservationCost": 6e10,
                    "calib": True,
                },
                "sky": {
                    "numRequired": 400,
                    "nonObservationCost": 6e10,
                    "calib": True,
                },
            }
            target_class_dict = {}
            # for i in range(1, 13, 1):
            for i in range(0, 14, 1):
                target_class_dict[f"sci_P{i}"] = 1
            target_class_dict = {
                **target_class_dict,
                **dict(sci_P9999=1, sky=2, cal=3),
            }

            # calculate targets' position on focal plane for the pointing
            telescopes = [
                nf.Telescope(
                    tb_ppc_t["ppc_ra"],
                    tb_ppc_t["ppc_dec"],
                    tb_ppc_t["ppc_pa"],
                    tb_ppc_t["ppc_obstime_utc"],
                )
            ]  # ppc_obstime should be converted into UTC here
            is_no_target = False
            target_fppos = [tele.get_fp_positions(targets) for tele in telescopes]

            # skip running netflow
            """
            (
                vis,
                tp,
                tel,
                tgt,
                tgt_class_dict,
                is_no_target,
                bench,
            ) = nfutils.fiber_allocation(
                df_sci,
                df_fluxstds,
                df_sky,
                tb_ppc_t["ppc_ra"].value[0],
                tb_ppc_t["ppc_dec"].value[0],
                tb_ppc_t["ppc_pa"].value[0],
                200,
                400,
                "2025-01-24",
                self.conf["netflow"]["use_gurobi"],
                dict(self.conf["gurobi"]) if self.conf["netflow"]["use_gurobi"] else None,
                self.conf["sfa"]["pfs_instdata_dir"],
                self.conf["sfa"]["cobra_coach_dir"],
                None,
                self.conf["sfa"]["sm"],
                self.conf["sfa"]["dot_margin"],
                None,
                None,
                None,
                location_group_penalty=None,
                cobra_instrument_region=None,
                min_sky_targets_per_instrument_region=None,
                instrument_region_penalty=None,
                num_reserved_fibers=0,
                fiber_non_allocation_cost=0.0,
                df_filler=None,
                force_exptime=900.0,
                two_stage=self.conf["netflow"]["two_stage"],
                design_ready=True,
            )
            #"""

            # make design
            tp = target_fppos[0]
            tel = telescopes[0]
            tgt = targets
            tgt_class_dict = target_class_dict
            bench = None  # do not set bench as it is determined when running netflow

            if tb_ppc_t["ppc_resolution"] == "L":
                arm_ = "brn"
            if tb_ppc_t["ppc_resolution"] == "M":
                arm_ = "bmn"

            design = sfa.designutils.generate_pfs_design(
                df_sci,
                df_fluxstds,
                df_sky,
                vis,
                tp,
                tel,
                tgt,
                tgt_class_dict,
                bench,
                arms=arm_,
                df_filler=None,
                is_no_target=is_no_target,
                design_name=ppc_code,
                pfs_instdata_dir=self.conf["sfa"]["pfs_instdata_dir"],
                obs_time=tb_ppc_t["ppc_obstime_utc"],
            )

            # add guiders
            guidestars = sfa.designutils.generate_guidestars_from_gaiadb(
                tb_ppc_t["ppc_ra"],
                tb_ppc_t["ppc_dec"],
                tb_ppc_t["ppc_pa"],
                tb_ppc_t["ppc_obstime_utc"],  # obstime should be in UTC
                telescope_elevation=None,
                conf=self.conf,
                guidestar_mag_min=self.conf["sfa"]["guidestar_mag_min"],
                guidestar_mag_max=self.conf["sfa"]["guidestar_mag_max"],
                guidestar_neighbor_mag_min=self.conf["sfa"][
                    "guidestar_neighbor_mag_min"
                ],
                guidestar_minsep_deg=self.conf["sfa"]["guidestar_minsep_deg"],
                # guidestar_mag_min=12,
                # guidestar_mag_max=22,
                # guidestar_neighbor_mag_min=21,
                # guidestar_minsep_deg=0.0002778,
                # # gaiadb_epoch=2015.0,
                # # gaiadb_input_catalog_id=2,
            )
            design.guideStars = guidestars

            # show assigned targets
            logger.info(
                f"[Make design] pfsDesign file {design.filename} is created in the {os.path.join(self.workDir,'pfs_designs', WG)} directory."
            )
            logger.info(
                "Number of SCIENCE fibers: {:}".format(
                    len(np.where(design.targetType == 1)[0])
                )
            )
            logger.info(
                "Number of FLUXSTD fibers: {:}".format(
                    len(np.where(design.targetType == 3)[0])
                )
            )
            logger.info(
                "Number of SKY fibers: {:}".format(
                    len(np.where(design.targetType == 2)[0])
                )
            )
            logger.info("Number of AG stars: {:}".format(len(guidestars.objId)))

            # write design to output folder
            design.write(
                dirName=os.path.join(self.workDir, self.conf["ope"]["designPath"], WG),
                fileName=design.filename,
            )

            logger.info(
                f"DesignId = {design.pfsDesignId} (0x{design.pfsDesignId:016x})"
            )
            tb_ppc["pfsDesignId"][tb_ppc["ppc_code"] == ppc_code] = str(
                design.pfsDesignId
            )
            tb_ppc["pfsDesignId_hex"][
                tb_ppc["ppc_code"] == ppc_code
            ] = f"0x{design.pfsDesignId:016x}"

        tb_ppc["design_filename"] = [
            f"pfsDesign-{ii}.fits" for ii in tb_ppc["pfsDesignId_hex"]
        ]
        tb_ppc_ = tb_ppc[
            "ppc_code",
            "ppc_ra",
            "ppc_dec",
            "ppc_pa",
            "design_filename",
            "pfsDesignId",
            "ppc_exptime",
            "ppc_nframes",
            "ppc_obstime_utc",
            "ppc_obstime",
        ]
        df_ppc_ = Table.to_pandas(tb_ppc_)
        df_ppc_.to_csv(
            os.path.join(
                self.workDir,
                self.conf["ope"]["designPath"],
                f"{WG}_summary_reconfigure.csv",
            ),
            index=False,
        )

        return tb_ppc

    def makeope(self, tb_ppc):
        def is_valid_date_format(date_string):
            """Check if string starts with a valid YYYY-MM-DD date format."""
            if not isinstance(date_string, str):
                return False
            try:
                # Just validate the first 10 chars as a date
                datetime.strptime(date_string[:10], "%Y-%m-%d")
                return True
            except (ValueError, TypeError):
                return False

        def parse_datetime(date_string):
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
            ]
            # check if the date_string is in one of the possible formats
            for format_str in formats:
                try:
                    return datetime.strptime(date_string, format_str)
                except (ValueError, TypeError):
                    continue
            # If none of the formats worked
            raise ValueError(
                f"Time data '{date_string}' does not match any known format"
            )

        # Define the Hawaii timezone (HST is always UTC-10, no DST)
        hawaii_tz = pytz.timezone("Pacific/Honolulu")

        def hst_to_utc(hst_string):
            """Convert HST datetime string to UTC datetime strings (full and date only)."""
            try:
                dt_naive = parse_datetime(hst_string)
                dt_utc = hawaii_tz.localize(dt_naive).astimezone(pytz.utc)
                return dt_utc.strftime("%Y-%m-%d %H:%M:%S"), dt_utc.strftime("%Y-%m-%d")
            except ValueError:
                return np.nan, np.nan

        # Convert ppc_obstime (HST) to UTC
        _obstime_utc = []
        _obsdate_utc = []
        for hst_string in tb_ppc["ppc_obstime"]:
            full_time, date_only = hst_to_utc(hst_string)
            _obstime_utc.append(full_time)
            _obsdate_utc.append(date_only)

        # Update dataframe
        tb_ppc["ppc_obsdate"] = [
            row[:10] if is_valid_date_format(row) else np.nan
            for row in tb_ppc["ppc_obstime"]
        ]
        tb_ppc["ppc_obstime_utc_from_hst"] = _obstime_utc
        tb_ppc["ppc_obsdate_utc_from_hst"] = _obsdate_utc

        # Get unique dates more concisely
        def get_unique_dates(date_column):
            """Extract unique valid dates from a column."""
            return sorted(
                {date[:10] for date in date_column if is_valid_date_format(date)}
            )

        obsdates = get_unique_dates(tb_ppc["ppc_obstime"])
        obsdates_utc = get_unique_dates(tb_ppc["ppc_obsdate_utc_from_hst"])

        ## ope file generation ##
        ope = OpeFile(conf=self.conf, workDir=self.workDir)

        for obsdate_utc in obsdates_utc:
            logger.info(f"[Make ope] generating ope file for {obsdate_utc} (UTC)...")
            template_file = (
                self.conf["ope"]["template"]
                if os.path.exists(self.conf["ope"]["template"])
                else None
            )
            ope.loadTemplate(filename=template_file)  # initialize
            ope.update_obsdate(obsdate_utc, utc=True)  # update observation date

            tb_ppc_t = tb_ppc[tb_ppc["ppc_obsdate_utc_from_hst"] == obsdate_utc]

            tb_ppc_t["obsdate_in_hst"] = tb_ppc_t["ppc_obsdate"]
            tb_ppc_t["obsdate_in_utc"] = tb_ppc_t["ppc_obsdate_utc_from_hst"]
            tb_ppc_t["obstime_in_utc"] = tb_ppc_t["ppc_obstime_utc_from_hst"]

            if np.unique(tb_ppc_t["obsdate_in_utc"]).size > 1:
                logger.error(
                    f"Multiple unique UTC times found for {obsdate_utc} ({tb_ppc_t['obstime_in_utc_from_hst']=}) . This may lead to unexpected behavior."
                )
                raise ValueError(
                    f"Multiple unique UTC times found for the same UTC obsdate ({obsdate_utc}). Please check the input ppc_obstime values."
                )

            tb_ppc_t["pfs_design_id"] = tb_ppc_t["pfsDesignId"]
            tb_ppc_t["obstime_in_hst"] = tb_ppc_t["ppc_obstime"]
            tb_ppc_t["single_exptime"] = tb_ppc_t["ppc_exptime"]
            tb_ppc_t["n_split_frame"] = tb_ppc_t["ppc_nframes"]

            info = Table.to_pandas(
                tb_ppc_t[
                    "ppc_code",
                    "obsdate_in_hst",
                    "obstime_in_utc",
                    "pfs_design_id",
                    "ppc_ra",
                    "ppc_dec",
                    "obstime_in_hst",
                    "single_exptime",
                    "n_split_frame",
                ]
            )

            info = info.sort_values(by="obstime_in_utc", ascending=True).values.tolist()
            ope.update_design(info)
            ope.write()  # save file

    def ssp_input_validation(self):
        for wg_ in self.conf["ssp"]["WG"]:
            logger.info(f"[Validation of input] {wg_}")

            # read ppcList.ecsv
            tb_ppc = Table.read(
                os.path.join(self.workDir, "targets", wg_, "ppcList.ecsv")
            )

            validate_success_ppc = self.ssp_ppc_validate(tb_ppc)

            if validate_success_ppc:
                logger.info(f"[Validation of ppcList] Validation passed ({wg_})")

            for tb_ppc_t in tb_ppc:
                ppc_code = tb_ppc_t["ppc_code"]
                tb_sci, tb_sky, tb_fluxstd = self.read_tgt(ppc_code, wg_)

        return None

    def runSFA_ssp(self):
        tb_ppc_ = []
        for wg_ in self.conf["ssp"]["WG"]:
            tb_ppc_.append(self.makedesign(wg_))

        tb_ppc_mix = vstack(tb_ppc_)
        logger.info(f"ppclist updated:\n{tb_ppc_mix}")
        self.makeope(tb_ppc_mix)
        return None

    def ssp_obsplan_update(self):
        tb_ppc_ = []
        for wg_ in self.conf["ssp"]["WG"]:
            df_ppc_ = pd.read_csv(
                os.path.join(
                    self.workDir,
                    self.conf["ope"]["designPath"],
                    f"{wg_}_summary_reconfigure.csv",
                ),
            )
            tb_ppc_.append(Table.from_pandas(df_ppc_))

        tb_ppc_mix = vstack(tb_ppc_)
        logger.info(f"ppclist updated:\n{tb_ppc_mix}")
        self.makeope(tb_ppc_mix)
        return None

    def runValidation(self):
        from . import validation

        ## update config before run SFA ##
        self.update_config()

        for wg_ in self.conf["ssp"]["WG"]:
            parentPath = os.path.join(self.workDir, self.conf["ope"]["designPath"], wg_)
            figpath = os.path.join(
                self.workDir, self.conf["ope"]["validationPath"], wg_
            )

            validation.validation(
                parentPath,
                figpath,
                True,
                False,
                self.conf["ssp"]["ssp"],
            )

            logger.info(f"validation plots saved under {figpath}")

        return None
