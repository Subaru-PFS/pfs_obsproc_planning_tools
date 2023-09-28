#!/usr/bin/env python3
# generatePfsDesign.py : PPP+qPlan+SFR

import os
import sys
import numpy as np
import pandas as pd
import toml
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import argparse

import warnings
warnings.filterwarnings('ignore')

def read_conf(conf):
    config = toml.load(conf)
    return config

class OpeFile(object):

    def __init__(self, conf, workDir):
        self.template = os.path.join(workDir, conf['ope']['template'])
        self.outfilePath = os.path.join(workDir, conf['ope']['outfilePath'])
        self.runName = conf['ope']['runName']
        self.designPath = conf['ope']['designPath']
        #self.loadTemplate(self.template)
        return None

    def loadTemplate(self, filename=None):
        if filename == None:
            filename = self.template
        self.contents = ""
        with open(filename, 'r') as file:
            for line in file:
                self.contents += line

    def update(self, pointing, dictPointings, designId, observationTime):
        def convRaDec(ra, dec):
            if dec>0:
                decsgn = '+'
            else:
                decsgn = '-'
            ra /= 15
            ra1 = int(ra)
            ra2 = int((ra - ra1) * 60)
            ra3 = (ra - ra1 - ra2/60.) * 3600.
            ra_new = '%02d%02d%.3f' % (ra1, ra2, ra3)
            dec = abs(dec)
            dec1 = int(dec)
            dec2 = int((dec-dec1) * 60)
            dec3 = (dec - dec1 - dec2 / 60.) * 3600.
            dec_new = '%s%02d%02d%.2f' % (decsgn, dec1, dec2, dec3)
            return ra_new, dec_new

        # name of OPE file
        self.outfile = os.path.join(self.outfilePath, f'{pointing}.ope')
      
        # update PFSDSGNDIR
        self.contents_updated = self.contents.replace('PFSDSGNDIR="/data/pfsDesign/"', f'PFSDSGNDIR="{self.designPath}"')

        # update HEADER
        # OBSERVATION_FILE_NAME
        repl1 = 'OBSERVATION_FILE_NAME=template_pfs.ope'
        repl2 = f'OBSERVATION_FILE_NAME=pointing.ope'
        self.contents_updated = self.contents_updated.replace(repl1, repl2)

        # OBSERVATION_START_DATE
        obsdate = observationTime.split('T')[0].replace('-','.')
        obstime = observationTime.split('T')[1].replace('Z','')

        repl1 = 'OBSERVATION_START_DATE=2023.07.01'
        repl2 = f'OBSERVATION_START_DATE={obsdate}'
        self.contents_updated = self.contents_updated.replace(repl1, repl2)

        repl1 = 'OBSERVATION_END_DATE=2023.07.31'
        repl2 = f'OBSERVATION_END_DATE={obsdate}'
        self.contents_updated = self.contents_updated.replace(repl1, repl2)

        repl1 = 'OBSERVATION_START_TIME=17:00:00'
        repl2 = f'OBSERVATION_START_TIME={obstime}'
        self.contents_updated = self.contents_updated.replace(repl1, repl2)

        repl1 = 'OBSERVATION_END_TIME=06:00:00'
        repl2 = f'OBSERVATION_END_TIME={obstime}'
        self.contents_updated = self.contents_updated.replace(repl1, repl2)

        # update FIELD NAME (header part)
        repl1 = '# [FIELD_NAME] 05:28:40.1 +35:49:26'
        repl2 = f'# [{pointing}] 05:28:40.1 +35:49:26'
        self.contents_updated = self.contents_updated.replace(repl1, repl2)

        # update FIELD NAME (contents part)
        repl1 = 'FIELD_NAME=OBJECT="FIELD_NAME" RA=FIELD_RA DEC=FIELD_DEC EQUINOX=2000.0'
        ra = float(dictPointings[pointing.lower()]["ra_center"])
        dec = float(dictPointings[pointing.lower()]["dec_center"])
        ra_new, dec_new = convRaDec(ra, dec)
        repl2 = f'{pointing}=OBJECT="{pointing}" RA={ra_new} DEC={dec_new} EQUINOX=2000.0'
        self.contents_updated = self.contents_updated.replace(repl1, repl2)

        # update SETUPFIELD part
        repl1 = 'SetupField $DEF_PFSENG DESIGN_ID="designId" AG=OFF OFFSET_RA=0 OFFSET_DEC=0'
        repl2 = f'SetupField $DEF_PFSENG DESIGN_ID="{designId}" AG=OFF OFFSET_RA=0 OFFSET_DEC=0'
        self.contents_updated = self.contents_updated.replace(repl1, repl2)

    def write(self):
        with open(self.outfile, 'w') as file:
            file.write(self.contents_updated)
            
class GeneratePfsDesign(object):

    def __init__(self, config, workDir, repoDir):
        self.config = config
        self.workDir = workDir
        self.repoDir = repoDir
        return None

    def runPPP(self, n_pccs_l, n_pccs_m, show_plots=False):
        import PPP

        conf = read_conf(os.path.join(self.workDir, self.config))

        ## check input target list ##
        #df = pd.read_csv(os.path.join(self.workDir, 'input/mock_sim.csv'))
        #print(df[:5])

        ## read sample from local path ##
        if conf['ppp']['mode']=='local':
            readsamp_con={'mode':'local', 'localPath':os.path.join(self.workDir, 'input/mock_sim.csv')}
        else:
            readsamp_con={'mode':conf['ppp']['mode'], 'dialect':conf['targetdb']['db']['dialect'], 'user':conf['targetdb']['db']['user'], 'pwd':conf['targetdb']['db']['password'], 'host':conf['targetdb']['db']['host'], 'port':conf['targetdb']['db']['port'], 'dbname':conf['targetdb']['db']['dbname'], 'sql_query':conf['ppp']['sql_query']}

        ## define exposure time ##
        onsourceT_L=conf['ppp']['TEXP_NOMINAL']*n_pccs_l # in sec (assuming 300 PPCs given)  --  LR
        onsourceT_M=conf['ppp']['TEXP_NOMINAL']*n_pccs_m # in sec (assuming 0 PPCs given)  --  MR

        PPP.run(readsamp_con, onsourceT_L, onsourceT_M, iter1_on=False, dirName=self.workDir, show_plots=show_plots)

        ## check output ##
        data_ppp = np.load(os.path.join(self.workDir, 'output/obj_allo_tot.npy'), allow_pickle=True)

        return None


    def runQPlan(self, obs_dates):

        ## import qPlanner module ##
        import qPlan

        ## read output from PPP ##
        df=qPlan.run('output/ppcList.ecsv', obs_dates, dirName=self.workDir)

        ## qPlan result ##
        self.resQPlan = {ppc_code: obstime for obstime, ppc_code in zip(df['obstime'], df['ppc_code'])}
        # print(len(self.resQPlan))

        return None

    def runSFA(self, clearOutput=False):

        ## configuration file ##
        conf = read_conf(os.path.join(self.workDir, self.config))

        ## setup python path ##
        sys.path.append(os.path.join(self.repoDir, 'ets_pointing/pfs_design_tool'))
        import pointing_utils.dbutils as dbutils
        import pointing_utils.nfutils as nfutils
        import pointing_utils.designutils as designutils

        import SFA

        ## get a list of OBs ##
        t = Table.read(os.path.join(self.workDir, 'output/obList.ecsv'))
        proposal_ids=t['proposal_id']
        ob_codes=t['ob_code']
        ob_obj_ids=t['ob_obj_id']
        ob_ras=t['ob_ra']
        ob_decs=t['ob_dec']
        ob_pmras = t['ob_pmra']
        ob_pmdecs = t['ob_pmdec']
        ob_parallaxs = t['ob_parallax']
        ob_equinoxs = t['ob_equinox']
        ob_priorities=t['ob_priority']
        obList = {f'{proposal_id}{ob_code}': [proposal_id, ob_code, ob_obj_id, ob_ra, ob_dec, ob_pmra, ob_pmdec, ob_parallax, ob_equinox, 'sci_P%d' % (int(ob_priority))] for proposal_id, ob_code, ob_obj_id, ob_ra, ob_dec, ob_pmra, ob_pmdec, ob_parallax, ob_equinox, ob_priority in zip(proposal_ids, ob_codes, ob_obj_ids, ob_ras, ob_decs, ob_pmras, ob_pmdecs, ob_parallaxs, ob_equinoxs, ob_priorities)}
        print(len(obList))

        ## get a list of assigned OBs ## FIXME (maybe we don't need to use this)
        data_ppp = np.load(os.path.join(self.workDir, 'output/obj_allo_tot.npy'), allow_pickle=True)
        #print(len(data_ppp))
        #print(t[:4])

        ## check the number of assigned fibers ##
        for i in range(len(data_ppp)):
            print(data_ppp[i][0], len(data_ppp[i][7]))
            #print(data_ppp[0])

        ## get a list of assigned targets combined with qPlan info ##
        data = []
        for i in range(len(data_ppp)):
            ppc_code = data_ppp[i][0]
            ppc_ra = data_ppp[i][2]
            ppc_dec = data_ppp[i][3]
            ppc_pa = data_ppp[i][4]
            ob_unique_id = data_ppp[i][7]
            if ppc_code in self.resQPlan.keys():
                obstime=self.resQPlan[ppc_code].tz_convert('UTC')
                for oid in ob_unique_id:
                    data.append([ppc_code, ppc_ra, ppc_dec, ppc_pa, oid] + obList[oid] + [obstime.strftime('%Y-%m-%d %X')])

        ## write to csv ##
        filename = 'ppp+qplan_outout.csv'
        header='pointing,ra_center,dec_center,pa_center,ob_unique_code,proposal_id,ob_code,obj_id,ra_target,dec_target,pmra_target,pmdec_target,parallax_target,equinox_target,target_class,obstime'
        np.savetxt(os.path.join(self.workDir, 'output', filename), data , fmt='%s', 
        delimiter=',', comments='', header=header)

        ## run SFA ##
        listPointings, dictPointings, pfsDesignIds = SFA.run(conf, workDir=self.workDir, repoDir=self.repoDir, clearOutput=clearOutput)

        ## ope file generation ##
        ope = OpeFile(conf=conf, workDir=self.workDir)
        for pointing, (k,v) in zip(listPointings, pfsDesignIds.items()):
            ope.loadTemplate() # initialize
            ope.update(pointing=pointing, dictPointings=dictPointings, designId=v, observationTime=k) # update contents
            ope.write() # save file

        return None


def get_arguments():
    parser = argparse.ArgumentParser()

    # workDir
    parser.add_argument(
        "--workDir",
        type=str,
        default='.',
        help="directory for working (default: current directory)",
    )
    # repoDir
    parser.add_argument(
        "--repoDir",
        type=str,
        default='.',
        help="directory for repository (default: current directory)",
    )
    # config
    parser.add_argument(
        "--config",
        type=str,
        default='config.toml',
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
    if args.skip_ppp == False:
        gpd.runPPP(args.n_pccs_l, args.n_pccs_m, args.show_plots)

    ## run queuePlanner ##
    gpd.runQPlan(args.obs_dates)

    ## run SFA.py
    gpd.runSFA()

    return 0

if __name__ == "__main__":
    main()
