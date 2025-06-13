#!/usr/bin/env python3
# opefile.py : PPP+qPlan+SFR

import os
import sys
import warnings

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time, TimeDelta

warnings.filterwarnings("ignore")


class OpeFile(object):
    def __init__(self, conf, workDir):
        self.conf = conf
        if os.path.exists(os.path.join(workDir, self.conf["ope"]["template"])):
            self.template = os.path.join(workDir, self.conf["ope"]["template"])
        elif os.path.exists(self.conf["ope"]["template"]):
            self.template = self.conf["ope"]["template"]
        else:
            raise FileNotFoundError(
                f"OPE file template {self.conf['ope']['template']} not found in {workDir} or current directory."
            )
        if not self.conf["ssp"]["ssp"]:
            self.outfilePath = os.path.join(workDir, "ope")
            self.designPath = os.path.join(workDir, "design")
        elif self.conf["ssp"]["ssp"]:
            self.outfilePath = os.path.join(workDir, self.conf["ope"]["outfilePath"])
            self.designPath = os.path.join(workDir, self.conf["ope"]["designPath"])
        # self.runName = conf["ope"]["runName"]  # not used in the current implementation
        # self.exptime_ppp = conf["ppp"]["TEXP_NOMINAL"]
        # self.loadTemplate(self.template)

        for d in [self.outfilePath, self.designPath]:
            if not os.path.exists(d):
                print(f"{d} is not found and created")
                os.makedirs(d, exist_ok=True)
            else:
                print(f"{d} is found")

        return None

    def loadTemplate(self, filename=None):
        if filename is None:
            filename = self.template
        self.contents1 = ""
        self.contents2 = ""
        self.contents2_main = ""
        self.contents3 = ""
        with open(filename, "r") as file:
            science_part = 0
            for line in file:
                if line == "### SCIENCE:START ###\n":
                    science_part += 1
                if line == "### SCIENCE:END  ###\n":
                    science_part += 1
                if science_part == 0:
                    self.contents1 += line
                elif science_part == 1:
                    self.contents2 += line
                elif science_part == 2:
                    self.contents3 += line
                if line.startswith(("# SETUPFIELD WITH", "SETUPFIELD", "# Check Auto Guiding", "## Get spectrum", "GETOBJECT")):
                    self.contents2_main += line
                    if line.startswith(("SETUPFIELD", "# Check Auto Guiding")):
                        self.contents2_main += "\n" 

    def update_obsdate(self, obsdate, utc=False):
        obsdate_orig = obsdate
        if utc:
            # NOTE: A night in HST is safely assumed to be always the same day in UTC
            # convert obsdate (YYYY-MM-DD) to UTC and subtract 1 day
            obstime_hst = Time(obsdate) - TimeDelta(1.0 * u.day)
            obsdate = obstime_hst.strftime("%Y-%m-%d")

        # name of OPE file
        if self.conf["ssp"]["ssp"]:
            if obsdate in self.conf["ope"]["backup_dates"]:
                self.outfile = os.path.join(self.outfilePath, f"{obsdate}_backup.ope")
            else:
                self.outfile = os.path.join(self.outfilePath, f"{obsdate}.ope")
        else:
            self.outfile = os.path.join(self.outfilePath, f"{obsdate}.ope")

        # update PFSDSGNDIR
        self.contents1_updated = self.contents1.replace(
            'PFSDSGNDIR="/data/pfsDesign/"', f'PFSDSGNDIR="{self.designPath}"'
        )

        # update HEADER
        # OBSERVATION_FILE_NAME
        repl1 = "OBSERVATION_FILE_NAME=template_pfs.ope"
        repl2 = f"OBSERVATION_FILE_NAME={obsdate}.ope"
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)

        # OBSERVATION_START_DATE
        obsdate_new = obsdate.replace("-", ".")
        repl1 = "OBSERVATION_START_DATE=2023.07.01"
        repl2 = f"OBSERVATION_START_DATE={obsdate_new}"
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)

        # OBSERVATION_END_DATE
        obsdate2 = Time(obsdate) + TimeDelta(1.0 * u.day)
        obsdate2_new = obsdate2.strftime("%Y.%m.%d")
        repl1 = "OBSERVATION_END_DATE=2023.7.31"
        repl2 = f"OBSERVATION_END_DATE={obsdate2_new}"
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)

    def update_design(self, info):
        def convRaDec(ra, dec):
            if dec > 0:
                decsgn = "+"
            else:
                decsgn = "-"
            ra /= 15
            ra1 = int(ra)
            ra2 = int((ra - ra1) * 60)
            ra3 = (ra - ra1 - ra2 / 60.0) * 3600.0
            ra_new = "%02d%02d%.3f" % (ra1, ra2, ra3)
            dec = abs(dec)
            dec1 = int(dec)
            dec2 = int((dec - dec1) * 60)
            dec3 = (dec - dec1 - dec2 / 60.0) * 3600.0
            dec_new = "%s%02d%02d%.2f" % (decsgn, dec1, dec2, dec3)
            return ra_new, dec_new

        # update FIELD NAME (header part)
        repl1 = "# [FIELD_NAME] 05:28:40.1 +35:49:26"
        repl2 = ""
        for i, val in enumerate(info):
            repl2 += f"# [{val[0]}] {val[4]} {val[5]}\n"
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)

        # update FIELD NAME (contents part)
        repl1 = (
            'FIELD_NAME=OBJECT="FIELD_NAME" RA=FIELD_RA DEC=FIELD_DEC EQUINOX=2000.0'
        )
        repl2 = ""
        object_names = []
        for i, val in enumerate(info):
            repl2_single_line = (
                f'{val[0]}=OBJECT="{val[0]}" RA={val[4]} DEC={val[5]} EQUINOX=2000.0'
            )
            # register the OBJECT if it is not already in the list
            if repl2_single_line not in repl2:
                object_names.append(val[0])
                repl2 += repl2_single_line + "\n"
        if len(object_names) != len(set(object_names)):
            raise ValueError(
                f"Duplicate PPC codes with different coordinates and/or EQUINOX found ppc_code={object_names}"
            )
        self.contents1_updated = self.contents1_updated.replace(repl1, repl2)

        # remove unnecessary words
        self.contents1_updated = self.contents1_updated.replace(
            "#!!! MODIFICATION NEEDED !!!#", ""
        )
        self.contents1_updated = self.contents1_updated.replace(
            "#!!! WHOLE LIST NEED TO BE MODIFIED !!!#", ""
        )

        # update "Science Exposure" part
        self.contents2_updated = ""
        for i, val in enumerate(info):
            tmpl = self.contents2
            tmpl_longexp = self.contents2_main
            total_exptime = val[7]
            nframe = val[8]
            single_exptime = total_exptime/nframe
            nframe_long = int(np.ceil(1800.0 / single_exptime))              

            # add PPC code
            repl1 = "### SCIENCE:START ###"
            repl2 = f"### {val[0]} ###\n### OBSTIME: {val[6]} ###"
            tmpl = tmpl.replace(repl1, repl2)
            tmpl_longexp = tmpl_longexp.replace(repl1, repl2)

            # add pfsDesignId
            repl1 = 'DESIGN_ID="designId"'
            repl2 = f'DESIGN_ID="0x{val[3]:016x}"'
            tmpl = tmpl.replace(repl1, repl2)
            tmpl_longexp = tmpl_longexp.replace(repl1, repl2)

            # add objectname
            repl1 = '"objectname"'
            repl2 = f'"{val[0]}"'
            tmpl = tmpl.replace(repl1, repl2)
            tmpl_longexp = tmpl_longexp.replace(repl1, repl2)

            # add exptime
            repl1 = '"exptime"'
            # if split_frame is true, separate each frame into n sub-frames with an exptime of exptime/n
            if nframe <= nframe_long:
                repl2 = f"{single_exptime} NFRAME={nframe}"
            else:
                repl2 = f"{single_exptime} NFRAME={nframe_long}"
            tmpl = tmpl.replace(repl1, repl2)

            # remove unnecessary words
            repl1 = "# SETUPFIELD WITH cobra convergence                     #!!! MODIFICATION NEEDED: designId, objectname !!!#"
            repl2 = "# SETUPFIELD WITH cobra convergence"
            tmpl = tmpl.replace(repl1, repl2)
            tmpl_longexp = tmpl_longexp.replace(repl1, repl2)

            repl1 = "## Get spectrum                                         #!!! MODIFICATION NEEDED: objectname !!!#"
            repl2 = "## Get spectrum"
            tmpl = tmpl.replace(repl1, repl2)
            tmpl_longexp = tmpl_longexp.replace(repl1, repl2)

            self.contents2_updated += tmpl

            if nframe > nframe_long:
                repl1 = '"exptime"'
                nframe -= nframe_long
                while nframe > 0:
                    repl2 = f"{single_exptime} NFRAME={nframe_long}"
                    if nframe <= nframe_long:
                        repl2 = f"{single_exptime} NFRAME={nframe}"
                    tmpl_longexp = tmpl_longexp.replace(repl1, repl2)
                    self.contents2_updated += tmpl_longexp + "\n\n"
                    nframe -= nframe_long
                
            self.contents3 = self.contents3.replace("### SCIENCE:END  ###", "")

    def write(self):
        with open(self.outfile, "w") as file:
            file.write(self.contents1_updated)
            file.write(self.contents2_updated)
            file.write(self.contents3)
