#!/usr/bin/env python3
# generatePfsDesign.py : PPP+qPlan+SFR

import argparse
import os
import warnings
from datetime import timedelta, datetime
import pytz

hawaii_tz = pytz.timezone("Pacific/Honolulu")

import git
import numpy as np
import pandas as pd
import toml
from astropy.table import Table, vstack
from logzero import logger

warnings.filterwarnings("ignore")

from .opefile import OpeFile
from pfs_design_tool.pointing_utils import nfutils
import ets_fiber_assigner.netflow as nf
from pfs_design_tool import reconfigure_fibers_ppp as sfa


def read_conf(conf):
    config = toml.load(conf)
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


class GeneratePfsDesign(object):
    def __init__(self, config, workDir=".", repoDir=None):
        self.config = config
        self.workDir = workDir
        self.repoDir = repoDir
        self.obs_dates = ["2023-05-20"]

        ## configuration file ##
        self.conf = read_conf(os.path.join(self.workDir, self.config))

        ## define directory of outputs from each component ##
        if self.conf["ssp"]["ssp"] == False:
            ## set obs_dates
            self.obs_dates = self.conf["qplan"]["obs_dates"]

            self.inputDirPPP = os.path.join(self.workDir, self.conf["ppp"]["inputDir"])
            self.outputDirPPP = os.path.join(
                self.workDir, self.conf["ppp"]["outputDir"]
            )
            self.outputDirQplan = os.path.join(
                self.workDir, self.conf["qplan"]["outputDir"]
            )
            self.cobraCoachDir = os.path.join(
                self.workDir, self.conf["sfa"]["cobra_coach_dir"]
            )

            # create input/output directories when not exist
            for d in [
                self.inputDirPPP,
                self.outputDirPPP,
                self.outputDirQplan,
                self.cobraCoachDir,
                os.path.join(self.workDir, self.conf["ope"]["designPath"]),
            ]:
                if not os.path.exists(d):
                    logger.info(f"{d} is not found and created")
                    os.makedirs(d, exist_ok=True)
                else:
                    logger.info(f"{d} exists")

            # looks like cobra_coach_dir must be in a full absolute path
            self.conf["sfa"]["cobra_coach_dir_orig"] = self.conf["sfa"][
                "cobra_coach_dir"
            ]
            self.conf["sfa"]["cobra_coach_dir"] = self.cobraCoachDir
        else:
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
            os.environ["PFS_UTILS_DIR"] = os.path.join(pfs.utils.__path__[0], "../../../")
        except:
            repo_path = self.conf["sfa"]["pfs_utils_dir"]
        version_desire = self.conf["sfa"]["pfs_utils_ver"]

        check_versions("pfs_utils", repo_url, repo_path, version_desire)

        """
        instdata_dir = self.conf["sfa"]["pfs_instdata_dir"]
        if os.path.exists(instdata_dir):
            logger.info(f"pfs_instdata found: {instdata_dir}")
        else:
            if not os.path.exists(
                os.path.join(self.workDir, os.path.basename(instdata_dir))
            ):
                logger.info(
                    f"pfs_instdata not found at {instdata_dir}, clone from GitHub as {os.path.join(self.workDir, os.path.basename(instdata_dir))}"
                )
                _ = git.Repo.clone_from(
                    "https://github.com/Subaru-PFS/pfs_instdata.git",
                    os.path.join(self.workDir, os.path.basename(instdata_dir)),
                    branch="master",
                )
            else:
                logger.info(
                    f"pfs_instdata found at {os.path.join(self.workDir, os.path.basename(instdata_dir))}, reuse it"
                )

            self.conf["sfa"]["pfs_instdata_dir_orig"] = self.conf["sfa"][
                "pfs_instdata_dir"
            ]
            self.conf["sfa"]["pfs_instdata_dir"] = os.path.join(
                self.workDir, os.path.basename(instdata_dir)
            )
        #"""

        return None

    def update_config(self):
        self.conf = read_conf(os.path.join(self.workDir, self.config))

    """
    def update_obs_dates(self, obs_dates):
        if type(obs_dates) == list:
            self.obs_dates = obs_dates
        else:
            raise ("specify obs_dates as a list")
    #"""

    def runPPP(self, n_pccs_l, n_pccs_m, show_plots=False):
        from . import PPP

        ## update config before run PPP ##
        self.update_config()

        ## read sample##
        readtgt_con = {
            "mode_readtgt": self.conf["ppp"]["mode"],
            "para_readtgt": {
                "localPath_tgt": self.conf["ppp"]["localPath_tgt"],
                "localPath_ppc": self.conf["ppp"]["localPath_ppc"],
                "DBPath_tgt": [
                    self.conf["targetdb"]["db"]["dialect"],
                    self.conf["targetdb"]["db"]["user"],
                    self.conf["targetdb"]["db"]["password"],
                    self.conf["targetdb"]["db"]["host"],
                    self.conf["targetdb"]["db"]["port"],
                    self.conf["targetdb"]["db"]["dbname"],
                ],
                "sql_query": self.conf["ppp"]["sql_query"],
                "DBPath_qDB": self.conf["queuedb"]["filepath"],
                "visibility_check": self.conf["ppp"]["visibility_check"],
                "obstimes": np.array(
                    [
                        [
                            self.conf["qplan"]["start_time"],
                            self.conf["qplan"]["stop_time"],
                        ]
                    ]
                ),
            },
        }

        cobra_coach, bench_info = nfutils.getBench(
            self.conf["sfa"]["pfs_instdata_dir"],
            self.conf["sfa"]["cobra_coach_dir"],
            None,
            self.conf["sfa"]["sm"],
            self.conf["sfa"]["dot_margin"],
        )

        # reserve fibers for calibration targets?
        if self.conf["ppp"]["reserveFibers"] == True:
            num_reserved_fibers = int(
                self.conf["sfa"]["n_sky"] + self.conf["sfa"]["n_fluxstd"]
            )
            fiber_non_allocation_cost = self.conf["ppp"]["fiberNonAllocationCost"]
        else:
            num_reserved_fibers = 0
            fiber_non_allocation_cost = 0.0
        logger.info(f"{num_reserved_fibers} fibers reserved for calibration targets")

        PPP.run(
            bench_info,
            readtgt_con,
            n_pccs_l,
            n_pccs_m,
            dirName=self.outputDirPPP,
            numReservedFibers=num_reserved_fibers,
            fiberNonAllocationCost=fiber_non_allocation_cost,
            show_plots=show_plots,
        )

        ## check output ##
        data_ppp = np.load(
            os.path.join(self.outputDirPPP, "obj_allo_tot.npy"), allow_pickle=True
        )

        return None

    def runQPlan(self, plotVisibility=False):
        ## update config before run qPlan ##
        self.update_config()

        ## import qPlanner module ##
        from . import qPlan

        ## read output from PPP ##
        self.df_qplan, self.sdlr, self.figs_qplan = qPlan.run(
            self.conf,
            "ppcList.ecsv",
            inputDirName=self.outputDirPPP,
            outputDirName=self.outputDirQplan,
            plotVisibility=plotVisibility,
        )

        ## qPlan result ##
        self.resQPlan = {
            ppc_code: (obstime, ppc_ra, ppc_dec)
            for obstime, ppc_code, ppc_ra, ppc_dec in zip(
                self.df_qplan["obstime"],
                self.df_qplan["ppc_code"],
                self.df_qplan["ppc_ra"],
                self.df_qplan["ppc_dec"],
            )
        }

        if plotVisibility is True:
            return self.figs_qplan
        else:
            return None

    def runSFA(self, clearOutput=False):
        from . import SFA

        ## update config before run SFA ##
        self.update_config()

        ## get a list of OBs ##
        t = Table.read(os.path.join(self.outputDirPPP, "obList.ecsv"))
        proposal_ids = t["proposal_id"]
        ob_codes = t["ob_code"]
        ob_obj_ids = t["ob_obj_id"]
        ob_cat_ids = t["ob_cat_id"]
        ob_ras = t["ob_ra"]
        ob_decs = t["ob_dec"]
        ob_pmras = np.array([float(ii) for ii in t["ob_pmra"]])
        ob_pmdecs = np.array([float(ii) for ii in t["ob_pmdec"]])
        ob_parallaxs = np.array([float(ii) for ii in t["ob_parallax"]])
        ob_equinoxs = t["ob_equinox"]
        ob_priorities = t["ob_priority"]
        ob_single_exptimes = t["ob_single_exptime"]
        ob_filter_gs = t["ob_filter_g"]
        ob_filter_rs = t["ob_filter_r"]
        ob_filter_is = t["ob_filter_i"]
        ob_filter_zs = t["ob_filter_z"]
        ob_filter_ys = t["ob_filter_y"]
        ob_psf_flux_gs = t["ob_psf_flux_g"]
        ob_psf_flux_rs = t["ob_psf_flux_r"]
        ob_psf_flux_is = t["ob_psf_flux_i"]
        ob_psf_flux_zs = t["ob_psf_flux_z"]
        ob_psf_flux_ys = t["ob_psf_flux_y"]
        ob_psf_flux_error_gs = t["ob_psf_flux_error_g"]
        ob_psf_flux_error_rs = t["ob_psf_flux_error_r"]
        ob_psf_flux_error_is = t["ob_psf_flux_error_i"]
        ob_psf_flux_error_zs = t["ob_psf_flux_error_z"]
        ob_psf_flux_error_ys = t["ob_psf_flux_error_y"]
        obList = {
            f"{proposal_id}_{ob_code}": [
                proposal_id,
                ob_code,
                ob_obj_id,
                ob_cat_id,
                ob_ra,
                ob_dec,
                ob_pmra,
                ob_pmdec,
                ob_parallax,
                ob_equinox,
                "sci_P%d" % (int(ob_priority)),
                ob_single_exptime,
                ob_filter_g,
                ob_filter_r,
                ob_filter_i,
                ob_filter_z,
                ob_filter_y,
                ob_psf_flux_g,
                ob_psf_flux_r,
                ob_psf_flux_i,
                ob_psf_flux_z,
                ob_psf_flux_y,
                ob_psf_flux_error_g,
                ob_psf_flux_error_r,
                ob_psf_flux_error_i,
                ob_psf_flux_error_z,
                ob_psf_flux_error_y,
            ]
            for proposal_id, ob_code, ob_obj_id, ob_cat_id, ob_ra, ob_dec, ob_pmra, ob_pmdec, ob_parallax, ob_equinox, ob_priority, ob_single_exptime, ob_filter_g, ob_filter_r, ob_filter_i, ob_filter_z, ob_filter_y, ob_psf_flux_g, ob_psf_flux_r, ob_psf_flux_i, ob_psf_flux_z, ob_psf_flux_y, ob_psf_flux_error_g, ob_psf_flux_error_r, ob_psf_flux_error_i, ob_psf_flux_error_z, ob_psf_flux_error_y in zip(
                proposal_ids,
                ob_codes,
                ob_obj_ids,
                ob_cat_ids,
                ob_ras,
                ob_decs,
                ob_pmras,
                ob_pmdecs,
                ob_parallaxs,
                ob_equinoxs,
                ob_priorities,
                ob_single_exptimes,
                ob_filter_gs,
                ob_filter_rs,
                ob_filter_is,
                ob_filter_zs,
                ob_filter_ys,
                ob_psf_flux_gs,
                ob_psf_flux_rs,
                ob_psf_flux_is,
                ob_psf_flux_zs,
                ob_psf_flux_ys,
                ob_psf_flux_error_gs,
                ob_psf_flux_error_rs,
                ob_psf_flux_error_is,
                ob_psf_flux_error_zs,
                ob_psf_flux_error_ys,
            )
        }
        logger.info(len(obList))

        ## get a list of assigned OBs ## FIXME (maybe we don't need to use this)
        data_ppp = Table.read(os.path.join(self.outputDirPPP, "ppcList.ecsv"))
        # print(len(data_ppp))
        # print(t[:4])

        ## check the number of assigned fibers ##
        for i in range(len(data_ppp)):
            print(data_ppp[i]["ppc_code"], len(data_ppp[i]["ppc_allocated_targets"]))
            # print(data_ppp[0])

        ## get a list of assigned targets combined with qPlan info ##
        data = []
        for i in range(len(data_ppp)):
            ppc_code = data_ppp[i]["ppc_code"]
            ppc_ra = data_ppp[i]["ppc_ra"]
            ppc_dec = data_ppp[i]["ppc_dec"]
            ppc_pa = data_ppp[i]["ppc_pa"]
            ob_unique_id = data_ppp[i]["ppc_allocated_targets"]
            if ppc_code in self.resQPlan.keys():
                res = self.resQPlan[ppc_code]
                obstime = res[0].tz_convert("UTC")
                obsdate_in_hst = obstime.date() - timedelta(days=1)
                for oid in ob_unique_id:
                    data.append(
                        [ppc_code, ppc_ra, ppc_dec, ppc_pa, oid]
                        + obList[oid]
                        + [obstime.strftime("%Y-%m-%d %X")]
                        + [obsdate_in_hst.strftime("%Y-%m-%d")]
                    )

        ## write to csv ##
        filename = "ppp+qplan_output.csv"
        header = "pointing,ra_center,dec_center,pa_center,ob_unique_code,proposal_id,ob_code,obj_id,cat_id,ra_target,dec_target,pmra_target,pmdec_target,parallax_target,equinox_target,target_class,ob_single_exptime,filter_g,filter_r,filter_i,filter_z,filter_y,psf_flux_g,psf_flux_r,psf_flux_i,psf_flux_z,psf_flux_y,psf_flux_error_g,psf_flux_error_r,psf_flux_error_i,psf_flux_error_z,psf_flux_error_y,obstime,obsdate_in_hst"
        np.savetxt(
            os.path.join(self.outputDirPPP, filename),
            data,
            fmt="%s",
            delimiter=",",
            comments="",
            header=header,
        )
        ## curate csv (FIXME) ##
        df = pd.read_csv(os.path.join(self.outputDirPPP, filename))
        df = df.replace("[]", "")
        df.to_csv(os.path.join(self.outputDirPPP, filename), index=False)

        ## run SFA ##
        filename = "ppp+qplan_output.csv"
        df = pd.read_csv(os.path.join(self.outputDirPPP, filename))

        listPointings, dictPointings, pfsDesignIds, observation_dates_in_hst = SFA.run(
            self.conf,
            workDir=self.workDir,
            repoDir=self.repoDir,
            clearOutput=clearOutput,
        )

        ## ope file generation ##
        ope = OpeFile(conf=self.conf, workDir=self.workDir)
        for obsdate in self.obs_dates:
            logger.info(f"generating ope file for {obsdate}...")
            ope.loadTemplate()  # initialize
            ope.update_obsdate(obsdate)  # update observation date
            info = []
            for pointing, (k, v), observation_date_in_hst in zip(
                listPointings, pfsDesignIds.items(), observation_dates_in_hst
            ):
                if observation_date_in_hst == obsdate:
                    res = self.resQPlan[pointing]
                    info.append(
                        [
                            pointing,
                            obsdate,
                            k,
                            v,
                            res[1].replace(":", ""),
                            res[2].replace(":", ""),
                            k,
                            dictPointings[pointing.lower()]["single_exptime"],
                            self.conf["ope"]["n_split_frame"],
                        ]
                    )
            info = pd.DataFrame(
                info,
                columns=[
                    "ppc_code",
                    "obsdate_in_hst",
                    "obstime_in_utc",
                    "pfs_design_id",
                    "ppc_ra",
                    "ppc_dec",
                    "obstime_in_hst",
                    "single_exptime",
                    "n_split_frame",
                ],
            )
            info["obstime_in_hst"] = pd.to_datetime(info["obstime_in_hst"], utc=True)
            info["obstime_in_hst"] = (
                info["obstime_in_hst"]
                .dt.tz_convert("Pacific/Honolulu")
                .dt.strftime("%Y/%m/%d %H:%M:%S")
            )
            info = info.sort_values(by="obstime_in_utc", ascending=True).values.tolist()
            ope.update_design(info)
            ope.write()  # save file
        # for pointing, (k,v) in zip(listPointings, pfsDesignIds.items()):
        #    ope.loadTemplate() # initialize
        #    ope.update(pointing=pointing, dictPointings=dictPointings, designId=v, observationTime=k) # update contents
        #    ope.write() # save file

        return None

    def runSFA_ssp(self):
        def ssp_tgt_validate(self, tb, ppc_code, tgt_type):
            validate_success = True

            # check whether required columns included
            if tgt_type == "science":
                req_cols = [
                    "obj_id",
                    "ra",
                    "dec",
                    "pmra",
                    "pmdec",
                    "parallax",
                    "epoch",
                    "target_type_id",
                    "input_catalog_id",
                    "ob_code",
                    "proposal_id",
                    "priority",
                    "effective_exptime",
                    "filter_g",
                    "filter_r",
                    "filter_i",
                    "filter_z",
                    "filter_y",
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
                    "cobraId",
                    "pfi_X",
                    "pfi_Y",
                ]
            elif tgt_type == "sky":
                req_cols = [
                    "obj_id",
                    "ra",
                    "dec",
                    "target_type_id",
                    "input_catalog_id",
                    "cobraId",
                    "pfi_X",
                    "pfi_Y",
                ]
            elif tgt_type == "fluxstd":
                req_cols = [
                    "obj_id",
                    "ra",
                    "dec",
                    "epoch",
                    "pmra",
                    "pmdec",
                    "parallax",
                    "target_type_id",
                    "input_catalog_id",
                    "prob_f_star",
                    "filter_g",
                    "filter_r",
                    "filter_i",
                    "filter_z",
                    "filter_y",
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
                    "cobraId",
                    "pfi_X",
                    "pfi_Y",
                ]
            missing_cols = [col for col in req_cols if col not in tb.colnames]
            if missing_cols:
                validate_success = False
                logger.error(
                    f"[Validation of input] The following required columns are missing ({ppc_code, tgt_type}): {missing_cols}"
                )

            # check no duplicated cobraId
            unique_vals, counts = np.unique(tb["cobraId"], return_counts=True)
            duplicates = unique_vals[counts > 1]
            if len(duplicates) > 0:
                for dup_val in duplicates:
                    dup_obj_ids = list(tb["obj_id"][tb["cobraId"] == dup_val])
                    logger.error(
                        f"[Validation of input] Found duplicates in 'cobraId' ({ppc_code, tgt_type}): cobraId={dup_val} assigned to {dup_obj_ids}"
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
                    if np.ma.is_masked(val): continue
                    if val not in valid_values:
                        invalid_rows.append((i, val))

                if invalid_rows:
                    validate_success = False
                    invalid_str = ", ".join(
                        f"Row {row_idx} => '{bad_val}'"
                        for row_idx, bad_val in invalid_rows
                    )
                    logger.error(
                        f"[Validation of input] Invalid values in flux column '{col_name}' ({ppc_code}, {tgt_type}): {invalid_str} "
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
                flux_data = np.array([tb[col].data for col in flux_cols])
                valid_mask = np.any(
                    flux_data > 0, axis=0
                )  # flux in at least one band should be there
                invalid_rows = np.where(~valid_mask)[0]
                if len(invalid_rows) > 0:
                    validate_success = False
                    logger.error(
                        f"[Validation of input] Rows lack flux info ({ppc_code}, {tgt_type}): {list(invalid_rows)}"
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
                        f"[Validation of input] Rows lack flux info ({ppc_code}, {tgt_type}): {list(invalid_rows)}"
                    )

            # check duplicated obj_id / ob_code
            df = tb.to_pandas()  # If tb is already a DataFrame, skip this line.

            if tgt_type == "science":
                duplicates_mask = df.duplicated(subset=["ob_code"], keep=False)
                if duplicates_mask.any():
                    validate_success = False
                    duplicated_rows = df[duplicates_mask]["ob_code"]
                    logger.error(
                        f"[Validation of input] Found duplicates in 'ob_code' ({ppc_code}, {tgt_type}):\n{duplicated_rows}"
                    )

            duplicates_mask = df.duplicated(subset=["obj_id"], keep=False)
            if duplicates_mask.any():
                validate_success = False
                duplicated_rows = df[duplicates_mask]["obj_id"]
                logger.error(
                    f"[Validation of input] Found duplicates in 'obj_id' ({ppc_code}, {tgt_type}):\n{duplicated_rows}"
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
                        f"[Validation of input] Proposal_id is incorrect (should be S25A-OT02; {ppc_code}, {tgt_type}): {proposal_id}"
                    )

                target_type = set(tb["target_type_id"])
                if target_type != {1}:
                    validate_success = False
                    logger.error(
                        f"[Validation of input] Target_type for science is incorrect (should be 1; {ppc_code}, {tgt_type}): {target_type}"
                    )

                catId = set(tb["input_catalog_id"])
                unexpected_Id = catId - {10091, 10092, 10093}
                if len(unexpected_Id) > 0:
                    validate_success = False
                    logger.error(
                        f"[Validation of input] Incorrect catId (should be 10091/2/3; {ppc_code}, {tgt_type}): {unexpected_Id}"
                    )

            elif tgt_type == "sky":
                logger.info(
                    f"{ppc_code} ({tgt_type}): tgt_type = {set(tb['target_type_id'])}, catId = {set(tb['input_catalog_id'])}"
                )

                target_type = set(tb["target_type_id"])
                if target_type != {2}:
                    validate_success = False
                    logger.error(
                        f"[Validation of input] Target_type for sky is incorrect (should be 2; {ppc_code}, {tgt_type}): {target_type}"
                    )

                catId = set(tb["input_catalog_id"])
                unexpected_Id = catId - {1006, 1007, 10091, 10092, 10093}
                if len(unexpected_Id) > 0:
                    validate_success = False
                    logger.error(
                        f"[Validation of input] Incorrect catId (should be 1006/7 or 10091/2/3; {ppc_code}, {tgt_type}): {unexpected_Id}"
                    )

            elif tgt_type == "fluxstd":
                logger.info(
                    f"{ppc_code} ({tgt_type}): tgt_type = {set(tb['target_type_id'])}, catId = {set(tb['input_catalog_id'])}"
                )

                target_type = set(tb["target_type_id"])
                if target_type != {3}:
                    validate_success = False
                    logger.error(
                        f"[Validation of input] Target_type for fluxstd is incorrect (should be 3; {ppc_code}, {tgt_type}): {target_type}"
                    )

                catId = set(tb["input_catalog_id"])
                unexpected_Id = catId - {3006, 10091, 10092, 10093}
                if len(unexpected_Id) > 0:
                    validate_success = False
                    logger.error(
                        f"[Validation of input] Incorrect catId (should be 3006 or 10091/2/3; {ppc_code}, {tgt_type}): {unexpected_Id}"
                    )

            return validate_success

        def read_tgt(self, ppc_code, WG):
            logger.info(f"[For SSP] Reading in target lists for pointing - {ppc_code}")
            filepath_sci = (
                self.workDir + "/targets/" + WG + "/science/" + ppc_code + ".ecsv"
            )
            filepath_sky = (
                self.workDir + "/targets/" + WG + "/sky/" + ppc_code + ".ecsv"
            )
            filepath_fluxstd = (
                self.workDir + "/targets/" + WG + "/fluxstd/" + ppc_code + ".ecsv"
            )

            tb_sci = Table.read(filepath_sci)
            tb_sky = Table.read(filepath_sky)
            tb_fluxstd = Table.read(filepath_fluxstd)

            tb_sci["cidx"] = tb_sci["cobraId"] - 1
            tb_sky["cidx"] = tb_sky["cobraId"] - 1
            tb_fluxstd["cidx"] = tb_fluxstd["cobraId"] - 1

            # validate input lists
            validate_success_sci = ssp_tgt_validate(self, tb_sci, ppc_code, "science")
            validate_success_sky = ssp_tgt_validate(self, tb_sky, ppc_code, "sky")
            validate_success_fluxstd = ssp_tgt_validate(
                self, tb_fluxstd, ppc_code, "fluxstd"
            )

            if (
                validate_success_sci
                and validate_success_sky
                and validate_success_fluxstd
            ):
                logger.info(f"[Validation of input] Validation passed ({ppc_code})")

            return tb_sci, tb_sky, tb_fluxstd

        def makedesign(self, WG):
            logger.info(f"[For SSP] Make design for {WG}")

            # read ppcList.ecsv
            tb_ppc = Table.read(self.workDir + "/targets/" + WG + "/ppcList.ecsv")

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

            missing_cols = [col for col in req_cols if col not in tb_ppc.colnames]
            if missing_cols:
                logger.error(
                    f"[Validation of ppcList] The following required columns are missing: {missing_cols}"
                )

            # Convert each timestamp from HST to UTC
            ppc_obstime_utc = []
            for hst_string in tb_ppc["ppc_obstime"]:
                try:
                    dt_naive = datetime.strptime(hst_string, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        dt_naive = datetime.strptime(hst_string, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        dt_naive = datetime.strptime(hst_string, "%Y-%m-%d %H:%M:%S.%f")
                dt_hst = hawaii_tz.localize(dt_naive)
                dt_utc = dt_hst.astimezone(pytz.utc)
                ppc_obstime_utc.append(dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))

            tb_ppc["ppc_obstime_utc"] = ppc_obstime_utc
            tb_ppc["pfsDesignId"] = np.zeros(len(tb_ppc), dtype=np.int64)
            tb_ppc["pfsDesignId_hex"] = np.zeros(len(tb_ppc), dtype="U64")

            # ensure no duplicated ppc_code in ppcList
            df_ppc = tb_ppc.to_pandas()  # If tb is already a DataFrame, skip this line.

            duplicates_mask = df_ppc.duplicated(subset=["ppc_code"], keep=False)
            if duplicates_mask.any():
                duplicated_rows = df_ppc[duplicates_mask]["ppc_code"]
                logger.error(
                    f"[Validation of ppcList] Found duplicates in 'ppc_code':\n{duplicated_rows}"
                )

            for tb_ppc_t in tb_ppc:
                ppc_code = tb_ppc_t["ppc_code"]

                tb_sci, tb_sky, tb_fluxstd = read_tgt(self, ppc_code, WG)

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
                bench = (
                    None  # do not set bench as it is determined when running netflow
                )

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
                )

                # add guiders
                guidestars = sfa.designutils.generate_guidestars_from_gaiadb(
                    tb_ppc_t["ppc_ra"],
                    tb_ppc_t["ppc_dec"],
                    tb_ppc_t["ppc_pa"],
                    tb_ppc_t["ppc_obstime_utc"],  # obstime should be in UTC
                    telescope_elevation=None,
                    conf=self.conf,
                    guidestar_mag_min=12,
                    guidestar_mag_max=22,
                    guidestar_neighbor_mag_min=21,
                    guidestar_minsep_deg=0.0002778,
                    # gaiadb_epoch=2015.0,
                    # gaiadb_input_catalog_id=2,
                )
                design.guideStars = guidestars

                # show assigned targets
                logger.info(
                    f"[Make design] pfsDesign file {design.filename} is created in the {self.workDir + 'pfs_designs/' + WG + '/'} directory."
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
                    dirName=self.workDir + "pfs_designs/" + WG + "/",
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
            tb_ppc_ = Table(
                tb_ppc[
                    "ppc_code",
                    "ppc_ra",
                    "ppc_dec",
                    "ppc_pa",
                    "design_filename",
                    "ppc_obstime_utc",
                ],
                names=[
                    "pointing",
                    "ra_center",
                    "dec_center",
                    "pa_center",
                    "design_filename",
                    "observation_time",
                ],
            )
            df_ppc_ = Table.to_pandas(tb_ppc_)
            df_ppc_.to_csv(
                os.path.join(self.workDir, f"pfs_designs/{WG}_summary_reconfigure.csv"),
                index=False,
            )

            return tb_ppc

        def makeope(tb_ppc):
            ## ope file generation ##
            ope = OpeFile(conf=self.conf, workDir=self.workDir)
            obsdates = list(set([row[:10] for row in tb_ppc["ppc_obstime"]]))
            tb_ppc["ppc_obsdate"] = [row[:10] for row in tb_ppc["ppc_obstime"]]
            for obsdate in obsdates:
                logger.info(f"[Make ope] generating ope file for {obsdate}...")
                ope.loadTemplate()  # initialize
                ope.update_obsdate(obsdate)  # update observation date

                tb_ppc_t = tb_ppc[tb_ppc["ppc_obsdate"] == obsdate]

                tb_ppc_t["obsdate_in_hst"] = tb_ppc_t["ppc_obsdate"]

                # Define the Hawaii timezone (HST is always UTC-10, no DST)
                hawaii_tz = pytz.timezone("Pacific/Honolulu")

                # Convert each timestamp from HST to UTC
                ppc_obstime_utc = []
                for hst_string in tb_ppc_t["ppc_obstime"]:
                    try:
                        dt_naive = datetime.strptime(hst_string, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        try:
                            dt_naive = datetime.strptime(hst_string, "%Y-%m-%dT%H:%M:%SZ")
                        except ValueError:
                            dt_naive = datetime.strptime(hst_string, "%Y-%m-%d %H:%M:%S.%f")
                    dt_hst = hawaii_tz.localize(dt_naive)
                    dt_utc = dt_hst.astimezone(pytz.utc)
                    ppc_obstime_utc.append(dt_utc.strftime("%Y-%m-%d %H:%M:%S"))

                tb_ppc_t["obstime_in_utc"] = ppc_obstime_utc

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

                info = info.sort_values(
                    by="obstime_in_utc", ascending=True
                ).values.tolist()
                ope.update_design(info)
                ope.write()  # save file

        tb_ppc_ = []
        for wg_ in self.conf["ssp"]["WG"]:
            tb_ppc_.append(makedesign(self, wg_))

        tb_ppc_mix = vstack(tb_ppc_)
        logger.info(f"ppclist updated:\n{tb_ppc_mix}")
        makeope(tb_ppc_mix)
        return None

    def runValidation(self):
        from . import validation

        ## update config before run SFA ##
        self.update_config()

        if self.conf["ssp"]["ssp"]:
            for wg_ in self.conf["ssp"]["WG"]:
                parentPath = self.workDir + "pfs_designs/" + wg_
                figpath = self.workDir + "validations/" + wg_

                validation.validation(
                    parentPath,
                    figpath,
                    True,
                    False,
                    self.conf["ssp"]["ssp"],
                )

                logger.info(f"validation plots saved under {figpath}")

        else:
            parentPath = os.path.join(
                self.workDir, self.conf["validation"]["parentPath"]
            )
            figpath = os.path.join(self.workDir, self.conf["validation"]["figpath"])

            validation.validation(
                parentPath,
                figpath,
                self.conf["validation"]["savefig"],
                self.conf["validation"]["showfig"],
                self.conf["ssp"]["ssp"],
            )

            logger.info(f"validation plots saved under {figpath}")

        return None


def get_arguments():
    parser = argparse.ArgumentParser()

    # workDir
    parser.add_argument(
        "--workDir",
        type=str,
        default=".",
        help="directory for working (default: current directory)",
    )
    # repoDir
    parser.add_argument(
        "--repoDir",
        type=str,
        default=".",
        help="directory for repository (default: current directory)",
    )
    # config
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="configuration file (default: config.toml)",
    )
    # n_pccs_l
    parser.add_argument(
        "--n_pccs_l",
        type=int,
        default=10,
        help="the number of pointings in LR (default: 10)",
    )
    # n_pccs_m
    parser.add_argument(
        "--n_pccs_m",
        type=int,
        default=10,
        help="the number of pointings in MR (default: 10)",
    )
    # skip_ppp
    parser.add_argument(
        "--skip_ppp",
        action="store_true",
        help="skip the PPP processing? (default: False)",
    )
    # skip_qplan
    parser.add_argument(
        "--skip_qplan",
        action="store_true",
        help="skip the qPlan processing? (default: False)",
    )
    # skip_sfa
    parser.add_argument(
        "--skip_sfa",
        action="store_true",
        help="skip the SFA processing? (default: False)",
    )
    # obs_dates
    parser.add_argument(
        "--obs_dates",
        required=True,
        nargs="*",
        type=str,
        default="2023-05-20",
        help="A list of observation dates (default: 2023-05-20)",
    )
    # show_plots
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="show plots of the PPP results? (default: False)",
    )

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    # print(args)

    gpd = GeneratePfsDesign(args.config, args.workDir, args.repoDir)

    ## run PPP ##
    if args.skip_ppp is False:
        gpd.runPPP(args.n_pccs_l, args.n_pccs_m, args.show_plots)

    ## run queuePlanner ##

    if args.skip_qplan == False:
        gpd.runQPlan(args.obs_dates)

    ## run SFA.py
    if args.skip_sfa == False:
        gpd.runSFA()

    return 0


if __name__ == "__main__":
    main()
