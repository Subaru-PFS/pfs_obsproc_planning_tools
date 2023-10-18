#!/usr/bin/env python3
# generatePfsDesign.py : PPP+qPlan+SFR

import argparse
import os
import sys
import warnings
from datetime import timedelta

import git
import numpy as np
import pandas as pd
import toml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

warnings.filterwarnings('ignore')

from .opefile import OpeFile


def read_conf(conf):
    config = toml.load(conf)
    return config

class GeneratePfsDesign(object):
    def __init__(self, config, workDir=".", repoDir=None):
        self.config = config
        self.workDir = workDir
        self.repoDir = repoDir
        self.obs_dates = ['2023-05-20']

        ## configuration file ##
        self.conf = read_conf(os.path.join(self.workDir, self.config))

        ## define directory of outputs from each component ##
        self.inputDirPPP = os.path.join(self.workDir, self.conf["ppp"]["inputDir"])
        self.outputDirPPP = os.path.join(self.workDir, self.conf["ppp"]["outputDir"])
        self.outputDirQplan = os.path.join(
            self.workDir, self.conf["qplan"]["outputDir"]
        )
        self.cobraCoachDir = os.path.join(self.workDir, self.conf['sfa']['cobra_coach_dir'])

        # create input/output directories when not exist
        for d in [self.inputDirPPP, self.outputDirPPP, self.outputDirQplan,
                  self.cobraCoachDir,
                  os.path.join(self.workDir, self.conf['ope']['designPath'])]:
            if not os.path.exists(d):
                print(f"{d} is not found and created")
                os.makedirs(d, exist_ok=True)
            else:
                print(f"{d} exists")

        # looks like cobra_coach_dir must be in a full absolute path
        self.conf['sfa']['cobra_coach_dir_orig'] = self.conf['sfa']['cobra_coach_dir']
        self.conf['sfa']['cobra_coach_dir'] = self.cobraCoachDir

        # check if pfs_instdata exists and clone from GitHub when not found
        instdata_dir = self.conf['sfa']['pfs_instdata_dir']
        if os.path.exists(instdata_dir):
            print(f"pfs_instdata found: {instdata_dir}")
        else:
            print(f"pfs_instdata not found at {instdata_dir}, clone from GitHub as {os.path.join(self.workDir, os.path.basename(instdata_dir))}")
            _ = git.Repo.clone_from("https://github.com/Subaru-PFS/pfs_instdata.git",
                                       os.path.join(self.workDir, os.path.basename(instdata_dir)),
                                       branch='master')
            self.conf['sfa']['pfs_instdata_dir_orig'] = self.conf['sfa']['pfs_instdata_dir']
            self.conf['sfa']['pfs_instdata_dir'] = os.path.join(self.workDir, os.path.basename(instdata_dir))

        return None

    def update_obs_dates(self, obs_dates):
        if type(obs_dates)==list:
            self.obs_dates = obs_dates
        else:
            raise('specify obs_dates as a list')

    def runPPP(self, n_pccs_l, n_pccs_m, show_plots=False):
        from . import PPP

        ## check input target list ##
        # df = pd.read_csv(os.path.join(self.workDir, 'input/mock_sim.csv'))
        # print(df[:5])

        ## read sample from local path ##
        if self.conf["ppp"]["mode"] == "local":
            readsamp_con = {
                "mode": "local",
                "localPath": os.path.join(
                    self.workDir, f"{self.inputDirPPP}/mock_sim.csv"
                ),
            }
        else:
            readsamp_con = {
                "mode": self.conf["ppp"]["mode"],
                "dialect": self.conf["targetdb"]["db"]["dialect"],
                "user": self.conf["targetdb"]["db"]["user"],
                "pwd": self.conf["targetdb"]["db"]["password"],
                "host": self.conf["targetdb"]["db"]["host"],
                "port": self.conf["targetdb"]["db"]["port"],
                "dbname": self.conf["targetdb"]["db"]["dbname"],
                "sql_query": self.conf["ppp"]["sql_query"],
            }

        ## define exposure time ##
        onsourceT_L = (
            self.conf["ppp"]["TEXP_NOMINAL"] * n_pccs_l
        )  # in sec (assuming 300 PPCs given)  --  LR
        onsourceT_M = (
            self.conf["ppp"]["TEXP_NOMINAL"] * n_pccs_m
        )  # in sec (assuming 0 PPCs given)  --  MR

        PPP.run(
            readsamp_con,
            onsourceT_L,
            onsourceT_M,
            iter1_on=False,
            dirName=self.outputDirPPP,
            show_plots=show_plots,
        )

        ## check output ##
        data_ppp = np.load(
            os.path.join(self.outputDirPPP, "obj_allo_tot.npy"), allow_pickle=True
        )

        return None


    def runQPlan(self, obs_dates=['2023-05-20'], plotVisibility=False):

        if obs_dates is not ['2023-05-20']:
            self.update_obs_dates(obs_dates)

        ## import qPlanner module ##
        from . import qPlan

        # outputDir = self.conf["qplan"]["outputDir"]
        ## read output from PPP ##
        self.df_qplan, self.sdlr, self.figs_qplan =qPlan.run('ppcList.ecsv', obs_dates, inputDirName=self.outputDirPPP, outputDirName=self.outputDirQplan, plotVisibility=plotVisibility)

        ## qPlan result ##
        self.resQPlan = {ppc_code: (obstime, ppc_ra, ppc_dec) for obstime, ppc_code, ppc_ra, ppc_dec in zip(self.df_qplan['obstime'], self.df_qplan['ppc_code'], self.df_qplan['ppc_ra'], self.df_qplan['ppc_dec'])}
        # print(len(self.resQPlan))

        if plotVisibility is True:
            return self.figs_qplan
        else:
            return None

    def runSFA(self, clearOutput=False):
        ## setup python path ##
        # sys.path.append(os.path.join(self.repoDir, "ets_pointing/pfs_design_tool"))
        from pfs_design_tool.pointing_utils import dbutils, designutils, nfutils

        from . import SFA

        ## define directory of outputs from each component ##


        ## get a list of OBs ##
        t = Table.read(os.path.join(self.outputDirPPP, "obList.ecsv"))
        proposal_ids = t["proposal_id"]
        ob_codes = t["ob_code"]
        ob_obj_ids = t["ob_obj_id"]
        ob_ras = t["ob_ra"]
        ob_decs = t["ob_dec"]
        ob_pmras = t["ob_pmra"]
        ob_pmdecs = t["ob_pmdec"]
        ob_parallaxs = t["ob_parallax"]
        ob_equinoxs = t["ob_equinox"]
        ob_priorities = t["ob_priority"]
        obList = {
            f"{proposal_id}{ob_code}": [
                proposal_id,
                ob_code,
                ob_obj_id,
                ob_ra,
                ob_dec,
                ob_pmra,
                ob_pmdec,
                ob_parallax,
                ob_equinox,
                "sci_P%d" % (int(ob_priority)),
            ]
            for proposal_id, ob_code, ob_obj_id, ob_ra, ob_dec, ob_pmra, ob_pmdec, ob_parallax, ob_equinox, ob_priority in zip(
                proposal_ids,
                ob_codes,
                ob_obj_ids,
                ob_ras,
                ob_decs,
                ob_pmras,
                ob_pmdecs,
                ob_parallaxs,
                ob_equinoxs,
                ob_priorities,
            )
        }
        print(len(obList))

        ## get a list of assigned OBs ## FIXME (maybe we don't need to use this)
        data_ppp = np.load(
            os.path.join(self.outputDirPPP, "obj_allo_tot.npy"), allow_pickle=True
        )
        # print(len(data_ppp))
        # print(t[:4])

        ## check the number of assigned fibers ##
        for i in range(len(data_ppp)):
            print(data_ppp[i][0], len(data_ppp[i][7]))
            # print(data_ppp[0])

        ## get a list of assigned targets combined with qPlan info ##
        data = []
        for i in range(len(data_ppp)):
            ppc_code = data_ppp[i][0]
            ppc_ra = data_ppp[i][2]
            ppc_dec = data_ppp[i][3]
            ppc_pa = data_ppp[i][4]
            ob_unique_id = data_ppp[i][7]
            if ppc_code in self.resQPlan.keys():
                res = self.resQPlan[ppc_code]
                obstime = res[0].tz_convert('UTC')
                obsdate_in_hst = (obstime.date() - timedelta(days=1))
                for oid in ob_unique_id:
                    data.append([ppc_code, ppc_ra, ppc_dec, ppc_pa, oid] + obList[oid] + [obstime.strftime('%Y-%m-%d %X')] + [obsdate_in_hst.strftime('%Y-%m-%d')])

        ## write to csv ##
        filename = 'ppp+qplan_outout.csv'
        header='pointing,ra_center,dec_center,pa_center,ob_unique_code,proposal_id,ob_code,obj_id,ra_target,dec_target,pmra_target,pmdec_target,parallax_target,equinox_target,target_class,obstime,obsdate_in_hst'
        np.savetxt(os.path.join(self.outputDirPPP, filename), data , fmt='%s',
        delimiter=',', comments='', header=header)

        ## run SFA ##
        filename = 'ppp+qplan_outout.csv'
        df = pd.read_csv(os.path.join(self.outputDirPPP, filename))

        listPointings, dictPointings, pfsDesignIds, observation_dates_in_hst = SFA.run(self.conf, workDir=self.workDir, repoDir=self.repoDir, clearOutput=clearOutput)

        ## ope file generation ##
        ope = OpeFile(conf=self.conf, workDir=self.workDir)
        for obsdate in self.obs_dates:
            ope.loadTemplate() # initialize
            ope.update_obsdate(obsdate) # update observation date
            info = []
            for pointing, (k,v), observation_date_in_hst in zip(listPointings, pfsDesignIds.items(), observation_dates_in_hst):
                if observation_date_in_hst == obsdate:
                    res = self.resQPlan[pointing]
                    info.append([pointing, obsdate, k, v, res[1], res[2]])
            ope.update_design(info)
            ope.write() # save file
        #for pointing, (k,v) in zip(listPointings, pfsDesignIds.items()):
        #    ope.loadTemplate() # initialize
        #    ope.update(pointing=pointing, dictPointings=dictPointings, designId=v, observationTime=k) # update contents
        #    ope.write() # save file

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
