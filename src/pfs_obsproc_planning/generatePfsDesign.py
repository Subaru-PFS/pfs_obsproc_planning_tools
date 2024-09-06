#!/usr/bin/env python3
# generatePfsDesign.py : PPP+qPlan+SFR

import argparse
import os
import warnings
from datetime import timedelta

import git
import numpy as np
import pandas as pd
import toml
from astropy.table import Table
from logzero import logger

warnings.filterwarnings("ignore")

from .opefile import OpeFile
from pfs_design_tool.pointing_utils import nfutils


def read_conf(conf):
    config = toml.load(conf)
    return config


class GeneratePfsDesign(object):
    def __init__(self, config, workDir=".", repoDir=None):
        self.config = config
        self.workDir = workDir
        self.repoDir = repoDir
        self.obs_dates = ["2023-05-20"]

        ## configuration file ##
        self.conf = read_conf(os.path.join(self.workDir, self.config))

        ## set obs_dates
        self.obs_dates = self.conf["qplan"]["obs_dates"]

        ## define directory of outputs from each component ##
        self.inputDirPPP = os.path.join(self.workDir, self.conf["ppp"]["inputDir"])
        self.outputDirPPP = os.path.join(self.workDir, self.conf["ppp"]["outputDir"])
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
        self.conf["sfa"]["cobra_coach_dir_orig"] = self.conf["sfa"]["cobra_coach_dir"]
        self.conf["sfa"]["cobra_coach_dir"] = self.cobraCoachDir

        # check if pfs_instdata exists and clone from GitHub when not found
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
        ob_pmras = t["ob_pmra"]
        ob_pmdecs = t["ob_pmdec"]
        ob_parallaxs = t["ob_parallax"]
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

    def runValidation(self):
        from . import validation

        ## update config before run SFA ##
        self.update_config()

        parentPath = os.path.join(self.workDir, self.conf["validation"]["parentPath"])
        figpath = os.path.join(self.workDir, self.conf["validation"]["figpath"])
        validation.validation(
            parentPath,
            figpath,
            self.conf["validation"]["savefig"],
            self.conf["validation"]["showfig"],
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
