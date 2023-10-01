#!/usr/bin/env python3
# opefile.py : PPP+qPlan+SFR

import os
import sys
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

import warnings
warnings.filterwarnings('ignore')

class OpeFile(object):

    def __init__(self, conf, workDir):
        self.template = os.path.join(workDir, conf['ope']['template'])
        self.outfilePath = os.path.join(workDir, conf['ope']['outfilePath'])
        self.runName = conf['ope']['runName']
        self.designPath = os.path.join(workDir, conf['ope']['designPath'])
        self.exptime_ppp = conf['ppp']['TEXP_NOMINAL']
        #self.loadTemplate(self.template)
        return None

    def loadTemplate(self, filename=None):
        if filename == None:
            filename = self.template
        self.contents1 = ""
        self.contents2 = ""
        self.contents3 = ""
        with open(filename, 'r') as file:
            science_part = 0
            for line in file:
                if line == "### Basic commands for science exposure ###\n":
                    science_part += 1
                if line == "### Calibrations ###\n":
                    science_part += 1
                if science_part == 0:
                    self.contents1 += line
                elif science_part == 1:
                    self.contents2 += line
                elif science_part == 2:
                    self.contents3 += line

    def update_obsdate(self, obsdate):
        # name of OPE file
        self.outfile = os.path.join(self.outfilePath, f'{obsdate}.ope')
      
        # update PFSDSGNDIR
        self.contents1_updated = self.contents1.replace('PFSDSGNDIR="/data/pfsDesign/"', f'PFSDSGNDIR="{self.designPath}"')

        # update HEADER
        # OBSERVATION_FILE_NAME
        repl1 = 'OBSERVATION_FILE_NAME=template_pfs.ope'
        repl2 = f'OBSERVATION_FILE_NAME={obsdate}.ope'
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)

        # OBSERVATION_START_DATE
        obsdate_new = obsdate.replace('-','.')
        repl1 = 'OBSERVATION_START_DATE=2023.07.01'
        repl2 = f'OBSERVATION_START_DATE={obsdate_new}'
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)

        # OBSERVATION_END_DATE
        obsdate2 = Time(obsdate) + TimeDelta(1.0 * u.day)
        obsdate2_new = obsdate2.strftime('%Y.%m.%d')
        repl1 = 'OBSERVATION_END_DATE=2023.07.31'
        repl2 = f'OBSERVATION_END_DATE={obsdate2_new}'
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)

    def update_design(self, info):
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

        # update FIELD NAME (header part)
        repl1 = '# [FIELD_NAME] 05:28:40.1 +35:49:26'
        repl2 = ""
        for i, val in enumerate(info):
            repl2 +=f'# [{val[0]}] {val[4]} {val[5]}\n'
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)
        
        # update FIELD NAME (contents part)
        repl1 = 'FIELD_NAME=OBJECT="FIELD_NAME" RA=FIELD_RA DEC=FIELD_DEC EQUINOX=2000.0'
        repl2 = ""
        for i, val in enumerate(info):
            repl2 += f'FIELD_NAME=OBJECT="{val[0]}" RA={val[4]} DEC={val[5]} EQUINOX=2000.0\n'
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)

        # update "Basic commands for science exposure" part
        self.contents2_updated = ""
        for i, val in enumerate(info):
            tmpl = self.contents2

            ''' add PPC code '''
            repl1 = '### Basic commands for science exposure ###'
            repl2 = f'### Basic commands for science exposure ({val[0]}) ###'
            tmpl = tmpl.replace(repl1, repl2)

            ''' add pfsDesignId '''
            repl1 = 'SetupField $DEF_PFSENG DESIGN_ID="designId" AG=OFF OFFSET_RA=0 OFFSET_DEC=0'
            repl2 = f'SetupField $DEF_PFSENG DESIGN_ID={val[3]:#013x} AG=OFF OFFSET_RA=0 OFFSET_DEC=0'
            tmpl = tmpl.replace(repl1, repl2)

            self.contents2_updated += tmpl

    def write(self):
        with open(self.outfile, 'w') as file:
            file.write(self.contents1_updated)
            file.write(self.contents2_updated)
            file.write(self.contents3)
            
