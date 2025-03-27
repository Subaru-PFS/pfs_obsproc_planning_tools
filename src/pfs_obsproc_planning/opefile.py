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
        if os.path.exists(os.path.join(workDir, conf["ope"]["template"])):
            self.template = os.path.join(workDir, conf["ope"]["template"])
        elif os.path.exists(conf["ope"]["template"]):
            self.template = conf["ope"]["template"]
        else:
            raise FileNotFoundError(
                f"OPE file template {conf['ope']['template']} not found in {workDir} or current directory."
            )
        self.outfilePath = os.path.join(workDir, conf["ope"]["outfilePath"])
        # self.runName = conf["ope"]["runName"]  # not used in the current implementation
        self.designPath = os.path.join(workDir, conf["ope"]["designPath"])
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

    def update_obsdate(self, obsdate):
        # name of OPE file
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

            # add PPC code
            repl1 = "### SCIENCE:START ###"
            repl2 = f"### {val[0]} ###\n### OBSTIME: {val[6]} ###"
            tmpl = tmpl.replace(repl1, repl2)

            # add pfsDesignId
            repl1 = 'DESIGN_ID="designId"'
            repl2 = f'DESIGN_ID="0x{val[3]:016x}"'
            tmpl = tmpl.replace(repl1, repl2)

            # add objectname
            repl1 = '"objectname"'
            repl2 = f'"{val[0]}"'
            tmpl = tmpl.replace(repl1, repl2)

            # add exptime
            repl1 = '"exptime"'
            repl2 = f"{val[7]}"
            if val[8] > 1:
                # if split_frame is true, separate each frame into n sub-frames with an exptime of exptime/n
                repl2 = f"{val[7]/val[8]} NFRAME={val[8]}"
            tmpl = tmpl.replace(repl1, repl2)

            # remove unnecessary words
            repl1 = "# SETUPFIELD WITH cobra convergence                     #!!! MODIFICATION NEEDED: designId, objectname !!!#"
            repl2 = "# SETUPFIELD WITH cobra convergence"
            tmpl = tmpl.replace(repl1, repl2)

            repl1 = "## Get spectrum                                         #!!! MODIFICATION NEEDED: objectname !!!#"
            repl2 = "## Get spectrum"
            tmpl = tmpl.replace(repl1, repl2)

            self.contents2_updated += tmpl

            self.contents3 = self.contents3.replace("### SCIENCE:END  ###", "")

    def write(self):
        with open(self.outfile, "w") as file:
            file.write(self.contents1_updated)
            file.write(self.contents2_updated)
            file.write(self.contents3)
