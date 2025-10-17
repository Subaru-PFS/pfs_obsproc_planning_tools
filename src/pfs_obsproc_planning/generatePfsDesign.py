#!/usr/bin/env python3
# generatePfsDesign.py : PPP+qPlan+SFR

import argparse
import os, sys
import warnings
from datetime import timedelta, datetime, date
import pytz
from dateutil import parser as ps
import time

hawaii_tz = pytz.timezone("Pacific/Honolulu")

import git
import numpy as np
import pandas as pd
import toml
from astropy.table import Table, vstack
from astropy.coordinates import Angle
import astropy.units as u
from loguru import logger

warnings.filterwarnings("ignore")

from .opefile import OpeFile
from pfs_design_tool.pointing_utils import nfutils
import ets_fiber_assigner.netflow as nf
from pfs_design_tool import reconfigure_fibers_ppp as sfa

def read_conf(conf):
    config = toml.load(conf)
    return config

def clear_folder(folder):
    import shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def check_versions(package, repo_path, version_desire):
    """Ensure the repository at repo_path is at the desired version."""

    def fetch_all_branches_and_tags(repo):
        """Fetch all branches and tags from a repository."""
        repo.remotes.origin.fetch()
        logger.info(f"({package}) Fetched all branches and tags.")

    def get_commit_time(repo, version):
        """Get the commit time of a branch or tag."""
        if version in [
            ref.name for ref in repo.remote().refs
        ]:  # Check if it's a branch
            commit_time = repo.commit(version).committed_date
        elif version in [tag.name for tag in repo.tags]:  # Check if it's a tag
            commit_time = repo.commit(version).committed_date
        else:
            return None  # Invalid version
        return commit_time

    def compare_commit_times(current_commit_time, desired_commit_time):
        """Compare commit times to determine if current is older than desired."""
        if current_commit_time is None:
            return True  # If no current commit time, always update
        return (
            current_commit_time < desired_commit_time
        )  # Check if the current version's commit is older

    def get_current_version(repo):
        """Get the current branch or tag version."""
        current_commit = repo.head.commit
        # Find if the current commit matches any tag
        for tag in repo.tags:
            if tag.commit == current_commit:
                logger.info(f"({package}) Current tag = {tag.name}")
                return tag.name  # Return the tag name if found

        # If no tag, return the current branch name
        for ref in repo.remote().refs:
            if ref.commit == current_commit:
                logger.info(f"({package}) Current branch = {ref.name}")
                return ref.name  # Return the branch name if found

        return None  # If no matching commit found

    def checkout_version(repo, version):
        """Checkout a specified branch or tag."""
        version = version.strip()
        if version == "":
            logger.info(f"({package}) Do not change the current branch/tag.")
            return  # Do nothing if no version is provided

        current_commit_time = get_commit_time(repo, "HEAD")
        desired_commit_time = get_commit_time(repo, version)

        if compare_commit_times(current_commit_time, desired_commit_time):
            if version in [ref.name for ref in repo.remote().refs]:
                # Checkout the remote branch and track it locally
                repo.git.checkout(f"-b {version} origin/{version}")
                logger.info(f"({package}) Checked out and tracking {version}.")
            elif version in [tag.name for tag in repo.tags]:
                # Checkout the tag (no tracking needed for tags)
                repo.git.checkout(version)
                logger.info(f"({package}) Checked out to tag {version}.")
            elif repo.commit(version):
                # If it's a commit hash, check out the commit directly
                repo.git.checkout(version)
                logger.info(f"({package}) Checked out to commit {version}.")
            else:
                logger.warning(
                    f"({package}) Version '{version}' not found in branches or tags."
                )
        else:
            logger.info(f"({package}) Current version is up-to-date with {version}.")

    # Step 1: Load the repository from the given path
    repo = git.Repo(repo_path)

    # Step 2: Fetch all branches and tags
    fetch_all_branches_and_tags(repo)
    get_current_version(repo)

    # Step 3: Checkout the specified branch or tag
    checkout_version(repo, version_desire)

    return None


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

            self.today = date.today().strftime("%Y%m%d")
            self.outputDir = os.path.join(self.workDir, f"output_{self.today}")
            self.inputDirPPP = os.path.join(self.workDir, self.conf["ppp"]["inputDir"])
            self.outputDirPPP = os.path.join(
                self.outputDir, "ppp"
            )
            self.outputDirQplan = os.path.join(
                self.outputDir, "qplan"
            )
            self.outputDirDesign = os.path.join(
                self.outputDir, "design"
            )
            self.outputDirOpe = os.path.join(
                self.outputDir, "ope"
            )
            self.cobraCoachDir = os.path.join(
                self.workDir, self.conf["sfa"]["cobra_coach_dir"]
            )
            # create input/output directories when not exist
            for d in [
                self.outputDir,
                self.inputDirPPP,
                self.outputDirPPP,
                self.outputDirQplan,
                self.cobraCoachDir,
                self.outputDirDesign,
                self.outputDirOpe,
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

            try:
                self.df_runtime = pd.read_csv(self.outputDir + "/runtime.csv")
            except FileNotFoundError:
                self.df_runtime = pd.DataFrame(np.array([[0, 0, 0]]), columns=["runtime_ppp", "runtime_qplan", "runtime_sfa"])

        # check versions of dependent packages
        def check_version_pfs(self, package):
            if self.conf["packages"]["check_version"]:
                try:
                    repo_path = self.conf["packages"][package + "_dir"]
                    version_desire = self.conf["packages"][package + "_ver"]
                    check_versions(package, repo_path, version_desire)
                except KeyError:
                    logger.warning(f"Path of {package} not found in {self.config}")
                return None
            else:
                return None

        for package_ in [
            "pfs_utils",
            "ets_pointing",
            "ets_shuffle",
            "pfs_datamodel",
            "ics_cobraCharmer",
            "ics_cobraOps",
            "ets_fiberalloc",
            "pfs_instdata",
            "ets_target_database",
            "ics_fpsActor",
            "spt_operational_database",
            "qplan",
        ]:
            check_version_pfs(self, package_)

        import pfs.utils

        repo_path = os.path.join(pfs.utils.__path__[0], "../../../")
        os.environ["PFS_UTILS_DIR"] = os.path.join(pfs.utils.__path__[0], "../../../")

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

    def runPPP(self, n_pccs_l, n_pccs_m, backup=False, show_plots=False):
        if "queue" in self.workDir:
            from . import PPP_queue as PPP
        else:
            from . import PPP

        time_start = time.time()

        ## update config before run PPP ##
        self.update_config()

        ## read sample##
        if backup:
            proposalId_ = self.conf["ppp"]["proposalIds_backup"]
            visibility_check_ = False
            obstimes_ = ["2025-06-22"]
            starttimes_ = ["2025-06-23 03:00:00"]
            stoptimes_ = ["2025-06-23 05:00:00"]
        elif not backup:
            proposalId_ = self.conf["ppp"]["proposalIds"]
            visibility_check_ = self.conf["ppp"]["visibility_check"]
            obstimes_ = self.conf["qplan"]["obs_dates"]
            starttimes_ = self.conf["qplan"]["start_time"]
            stoptimes_ = self.conf["qplan"]["stop_time"]
            
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
                "visibility_check": visibility_check_,
                "proposalIds": proposalId_,
                "obstimes": obstimes_,
                "starttimes": starttimes_,
                "stoptimes": stoptimes_,
            },
        }

        bench_info = nfutils.getBench(
            self.conf["packages"]["pfs_instdata_dir"],
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
        else:
            num_reserved_fibers = 0
        fiber_non_allocation_cost = self.conf["ppp"]["fiberNonAllocationCost"]
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
            backup=backup
            
        )

        ## check output ##
        data_ppp = np.load(
            os.path.join(self.outputDirPPP, "obj_allo_tot.npy"), allow_pickle=True
        )

        time_ppp = time.time() - time_start
        self.df_runtime["runtime_ppp"] = time_ppp

        return None

    def runQPlan(self, plotVisibility=False):
        time_start = time.time()
        
        ## update config before run qPlan ##
        self.update_config()

        ## import qPlanner module ##
        from . import qPlan
        
        #"""
        try:
            self.df_qplan = pd.read_csv(os.path.join(self.outputDirQplan, "result.csv"))
            obstimes = [pd.to_datetime(obstime_str) for obstime_str in self.df_qplan["obstime"]]
            self.resQPlan = {
                ppc_code: (obstime, ppc_ra, ppc_dec)
                for obstime, ppc_code, ppc_ra, ppc_dec in zip(
                    obstimes,
                    self.df_qplan["ppc_code"],
                    self.df_qplan["ppc_ra"],
                    self.df_qplan["ppc_dec"],
                )
            }
            return None
        except FileNotFoundError:
            self.df_qplan, self.sdlr, self.figs_qplan, self.tw_start, self.tw_stop = qPlan.run(
                self.conf,
                "ppcList.ecsv",
                inputDirName=self.outputDirPPP,
                outputDirName=self.outputDirQplan,
                plotVisibility=plotVisibility,
            )
            self.resQPlan = {
                ppc_code: (obstime, ppc_ra, ppc_dec)
                for obstime, ppc_code, ppc_ra, ppc_dec in zip(
                    self.df_qplan["obstime"],
                    self.df_qplan["ppc_code"],
                    self.df_qplan["ppc_ra"],
                    self.df_qplan["ppc_dec"],
                )
            }               

            if self.conf["ppp"]["daily_plan"]:
                logger.info(f"Now running for the daily planning")

                self.df_qplan["obstime_hst"] = self.df_qplan["obstime"].dt.tz_convert("US/Hawaii") 
                self.df_qplan["obstime_stop"] = self.df_qplan["obstime_hst"] + timedelta(minutes=21)
                df_window = (self.df_qplan).copy()
                
                starttime_backup = []
                stoptime_backup = []
                
                for tw_start, tw_stop in zip(self.tw_start, self.tw_stop):
                    # convert to Timestamp (in case they are strings)
                    tw_start = pd.Timestamp(tw_start)
                    tw_stop = pd.Timestamp(tw_stop)

                    # filter df_window to only rows inside this available window
                    df_sub = df_window[(df_window["obstime_hst"] >= tw_start) &
                                       (df_window["obstime_hst"] <= tw_stop)].copy()
                    df_sub.sort_values("obstime_hst", inplace=True)

                    if df_sub.empty:
                        # no obs inside this window → whole window is a gap
                        if tw_stop - tw_start > timedelta(minutes=10):
                            starttime_backup.append(tw_start)
                            stoptime_backup.append(tw_stop)
                            print(f"Gap: start at {tw_start}, stop at {tw_stop}")
                        continue

                    # check gap from tw_start → first observation
                    first_obs = df_sub["obstime_hst"].iloc[0]
                    if first_obs - tw_start > timedelta(minutes=10):
                        starttime_backup.append(tw_start)
                        stoptime_backup.append(first_obs)
                        print(f"Gap: start at {tw_start}, stop at {first_obs}")
                
                    # check gaps between consecutive observations
                    for i in range(len(df_sub) - 1):
                        gap = df_sub["obstime_hst"].iloc[i + 1] - df_sub["obstime_stop"].iloc[i]
                        if gap > timedelta(minutes=10):
                            starttime_backup.append(df_sub["obstime_stop"].iloc[i])
                            stoptime_backup.append(df_sub["obstime_hst"].iloc[i + 1])
                            print(f"Gap: start at {df_sub['obstime_stop'].iloc[i]}, stop at {df_sub['obstime_hst'].iloc[i + 1]}")
                
                    # check gap from last observation → tw_stop
                    last_stop = df_sub["obstime_stop"].iloc[-1]
                    if tw_stop - last_stop > timedelta(minutes=10):
                        starttime_backup.append(last_stop)
                        stoptime_backup.append(tw_stop)
                        print(f"Gap: start at {last_stop}, stop at {tw_stop}")
        
                if len(starttime_backup) > 0:
                    self.runPPP(1, 1, show_plots=False, backup=True)
        
                    self.df_qplan_, self.sdlr_, self.figs_qplan_ = qPlan.run(
                        self.conf,
                        "ppcList_backup.ecsv",
                        inputDirName=self.outputDirPPP,
                        outputDirName=self.outputDirQplan,
                        plotVisibility=plotVisibility,
                        starttime_backup=starttime_backup,
                        stoptime_backup=stoptime_backup,
                    )[:3]
                    self.df_qplan = pd.concat([self.df_qplan, self.df_qplan_], ignore_index=True)
                    self.resQPlan_ = {
                        ppc_code: (obstime, ppc_ra, ppc_dec)
                        for obstime, ppc_code, ppc_ra, ppc_dec in zip(
                            self.df_qplan_["obstime"],
                            self.df_qplan_["ppc_code"],
                            self.df_qplan_["ppc_ra"],
                            self.df_qplan_["ppc_dec"],
                        )
                    }
                    self.resQPlan = {**self.resQPlan, **self.resQPlan_}
                #"""
        
            (self.df_qplan).to_csv(os.path.join(self.outputDirQplan, "result.csv"))
    
            if plotVisibility is True:
                time_qplan = time.time() - time_start
                self.df_runtime["runtime_qplan"] = time_qplan
                return self.figs_qplan
            else:
                time_qplan = time.time() - time_start
                self.df_runtime["runtime_qplan"] = time_qplan
                return None

        # for test design generation at different obstime
        #self.resQPlan = {"PPC_L_uh006_1": (pd.to_datetime("2025-03-23T11:40:10.422Z", utc=True), 150.08220377, 2.18805709 ),
        #                "PPC_L_uh006_2": (pd.to_datetime("2025-03-24T11:13:28.779Z", utc=True), 150.08220377, 2.18805709),}

    def runSFA(self, clearOutput=False):
        time_start = time.time()
        
        from . import SFA

        ## update config before run SFA ##
        self.update_config()

        ## get a list of OBs ##
        t1 = Table.read(os.path.join(self.outputDirPPP, "obList.ecsv"))
        try:
            t2 = Table.read(os.path.join(self.outputDirPPP, "obList_backup.ecsv"))
        except:
            t2 = Table()
        t = vstack([t1, t2])
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
        ob_total_flux_gs = t["ob_total_flux_g"]
        ob_total_flux_rs = t["ob_total_flux_r"]
        ob_total_flux_is = t["ob_total_flux_i"]
        ob_total_flux_zs = t["ob_total_flux_z"]
        ob_total_flux_ys = t["ob_total_flux_y"]
        ob_total_flux_error_gs = t["ob_total_flux_error_g"]
        ob_total_flux_error_rs = t["ob_total_flux_error_r"]
        ob_total_flux_error_is = t["ob_total_flux_error_i"]
        ob_total_flux_error_zs = t["ob_total_flux_error_z"]
        ob_total_flux_error_ys = t["ob_total_flux_error_y"]
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
                ob_total_flux_g,
                ob_total_flux_r,
                ob_total_flux_i,
                ob_total_flux_z,
                ob_total_flux_y,
                ob_total_flux_error_g,
                ob_total_flux_error_r,
                ob_total_flux_error_i,
                ob_total_flux_error_z,
                ob_total_flux_error_y,
            ]
            for proposal_id, ob_code, ob_obj_id, ob_cat_id, ob_ra, ob_dec, ob_pmra, ob_pmdec, ob_parallax, ob_equinox, ob_priority, ob_single_exptime, ob_filter_g, ob_filter_r, ob_filter_i, ob_filter_z, ob_filter_y, ob_psf_flux_g, ob_psf_flux_r, ob_psf_flux_i, ob_psf_flux_z, ob_psf_flux_y, ob_psf_flux_error_g, ob_psf_flux_error_r, ob_psf_flux_error_i, ob_psf_flux_error_z, ob_psf_flux_error_y,ob_total_flux_g, ob_total_flux_r, ob_total_flux_i, ob_total_flux_z, ob_total_flux_y, ob_total_flux_error_g, ob_total_flux_error_r, ob_total_flux_error_i, ob_total_flux_error_z, ob_total_flux_error_y in zip(
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
                ob_total_flux_gs,
                ob_total_flux_rs,
                ob_total_flux_is,
                ob_total_flux_zs,
                ob_total_flux_ys,
                ob_total_flux_error_gs,
                ob_total_flux_error_rs,
                ob_total_flux_error_is,
                ob_total_flux_error_zs,
                ob_total_flux_error_ys,
            )
        }
        logger.info(len(obList))

        ## get a list of assigned OBs ## FIXME (maybe we don't need to use this)
        tb_ppp = Table.read(os.path.join(self.outputDirPPP, "ppcList.ecsv"))
        try:
            tb_ppp_backup = Table.read(os.path.join(self.outputDirPPP, "ppcList_backup.ecsv"))
        except:
            tb_ppp_backup = Table()
        data_ppp = vstack([tb_ppp, tb_ppp_backup])
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
        header = "pointing,ra_center,dec_center,pa_center,ob_unique_code,proposal_id,ob_code,obj_id,cat_id,ra_target,dec_target,pmra_target,pmdec_target,parallax_target,equinox_target,target_class,ob_single_exptime,filter_g,filter_r,filter_i,filter_z,filter_y,psf_flux_g,psf_flux_r,psf_flux_i,psf_flux_z,psf_flux_y,psf_flux_error_g,psf_flux_error_r,psf_flux_error_i,psf_flux_error_z,psf_flux_error_y,total_flux_g,total_flux_r,total_flux_i,total_flux_z,total_flux_y,total_flux_error_g,total_flux_error_r,total_flux_error_i,total_flux_error_z,total_flux_error_y,obstime,obsdate_in_hst"
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
        df['filter_g'] = df['filter_g'].apply(lambda x: "none" if x in [0.0,"0.0", "[]","--"] else x)
        df['filter_r'] = df['filter_r'].apply(lambda x: "none" if x in [0.0,"0.0","[]","--"] else x)
        df['filter_i'] = df['filter_i'].apply(lambda x: "none" if x in [0.0,"0.0","[]","--"] else x)
        df['filter_z'] = df['filter_z'].apply(lambda x: "none" if x in [0.0,"0.0","[]","--"] else x)
        df['filter_y'] = df['filter_y'].apply(lambda x: "none" if x in [0.0,"0.0","[]","--"] else x)
        #df = df.replace("[]", "")
        df.replace(9e-05, np.nan, inplace=True)
        df.to_csv(os.path.join(self.outputDirPPP, filename), index=False)

        ## run SFA ##
        filename = "ppp+qplan_output.csv"
        df = pd.read_csv(os.path.join(self.outputDirPPP, filename))

        clear_folder(self.outputDirDesign)
        listPointings, dictPointings, pfsDesignIds, observation_dates_in_hst = SFA.run(
            self.conf,
            workDir=self.outputDir,
            repoDir=self.repoDir,
            clearOutput=clearOutput,
        )

        ## ope file generation ##
        clear_folder(self.outputDirOpe)
        ope = OpeFile(conf=self.conf, workDir=self.outputDir)
        for obsdate in self.obs_dates:
            date_t = ps.parse(f"{obsdate} 12:00 HST")
            date_today = ps.parse(f"{self.today} 12:00 HST")
    
            if date_today > date_t:
                continue
            
            logger.info(f"generating ope file for {obsdate}...")
            ope.loadTemplate()  # initialize
            ope.update_obsdate(obsdate)  # update observation date
            info = []
            for pointing, (k, v), observation_date_in_hst in zip(
                listPointings, pfsDesignIds.items(), observation_dates_in_hst
            ):
                if observation_date_in_hst == obsdate:
                    res = self.resQPlan[pointing]
                    ppc_ra_ = res[1]
                    ppc_dec_ = res[2]

                    if isinstance(ppc_ra_, str) and (':' in ppc_ra_):
                        ppc_ra_t = ppc_ra_.replace(":", "")
                        ppc_dec_t = ppc_dec_.replace(":", "")
                    elif isinstance(ppc_ra_, float):
                        ppc_ra_t = Angle(ppc_ra_ * u.deg).to_string(unit=u.hourangle, sep='', precision=3, pad=True)
                        ppc_dec_t = Angle(ppc_dec_ * u.deg).to_string(unit=u.deg, sep='', alwayssign=True, precision=2, pad=True)

                    info.append(
                        [
                            pointing,
                            obsdate,
                            k,
                            v,
                            ppc_ra_t,
                            ppc_dec_t,
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

            if self.conf["ppp"]["daily_plan"]:
                break
                
        # for pointing, (k,v) in zip(listPointings, pfsDesignIds.items()):
        #    ope.loadTemplate() # initialize
        #    ope.update(pointing=pointing, dictPointings=dictPointings, designId=v, observationTime=k) # update contents
        #    ope.write() # save file

        time_sfa = time.time() - time_start
        self.df_runtime["runtime_sfa"] = time_sfa

        return None

    def runValidation(self):
        from . import validation

        ## update config before run SFA ##
        self.update_config()

        parentPath = self.outputDir
        figpath = os.path.join(self.outputDir, "figure_pfsDesign_validation")

        if not os.path.exists(figpath):
            os.makedirs(figpath)

        clear_folder(figpath)

        validation.validation(
            parentPath,
            figpath,
            self.conf["validation"]["savefig"],
            self.conf["validation"]["showfig"],
            self.conf["ssp"]["ssp"],
            self.conf,
        )
        
        logger.info(f"validation plots saved under {figpath}")

        if "queue" in self.workDir:
            from . import completion_check
            completion_check.run(self.conf, self.outputDir)
            
        logger.info(f"{self.df_runtime}")

        (self.df_runtime).to_csv((self.outputDir + "/runtime.csv"), index = False)

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
