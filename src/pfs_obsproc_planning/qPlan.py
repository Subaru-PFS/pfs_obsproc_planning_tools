#!/usr/bin/env python3
# qPlan.py : queuePlanner

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

warnings.filterwarnings("ignore")


def make_schedule_table(schedule):
    data = [
        (slot.start_time, slot.ob.name)
        for slot in schedule
        if slot.ob is not None and not slot.ob.derived
    ]
    df = pd.DataFrame(data, columns=["datetime", "ob_code"])
    return df


from io import BytesIO

from ginga.misc.Bunch import Bunch
from ginga.misc.log import get_logger
from IPython.core.display import display
from IPython.display import Image
from qplan.entity import (
    PPC_OB,
    EnvironmentConfiguration,
    PPCConfiguration,
    Program,
    Schedule,
    StaticTarget,
    TelescopeConfiguration,
)
from qplan.plots import airmass
from qplan.Scheduler import Scheduler
from qplan.util.site import site_subaru as observer

def run(conf, ppcList, obs_dates, inputDirName=".", outputDirName=".", plotVisibility=False):
    # log file will let us debug scheduling, check it for error and debug messages
    # NOTE: any python (stdlib) logging compatible logger will work
    logger = get_logger(
        "qplan", level=10, log_file=os.path.join(outputDirName, "sched.log")
    )

    # create one fake program, for PPP, everything runs under one "program"
    proposal = "S24B-QN017"
    pgm = Program(proposal, rank=10.0, hours=2000.0, category="open")
    pgms = {proposal: pgm}
    print(pgms)
    # apriori_info can convey how much time has already been scheduled for this program
    apriori_info = {proposal: dict(sched_time=0.0)}
    # print(apriori_info)

    # These  be used for all PPCs ("BOB"s)
    telcfg = TelescopeConfiguration(focus="P_OPT2")
    telcfg.min_el_deg = 30.0
    telcfg.max_el_deg = 85.0
    telcfg.min_rot_deg = -174.0
    telcfg.max_rot_deg = 174.0
    # PPCConfiguration -- PFS Pointing Center
    # if you have different times for your pointing centers, create a different one
    # for each exp_time, PA or resolution
    inscfg = PPCConfiguration(exp_time=15 * 60.0, pa=0.0, resolution="low")
    # can be used later to constrain, if desired, for now set to allow anything
    envcfg = EnvironmentConfiguration(seeing=99.0, transparency=0.0)

    # these are typical weights for HSC, but PFS has no filters
    # most weights are normalized to 0-1, but we use a higher value for delay
    # sometimes.
    # weights indicate "how important" something is to the scheduler
    # slew: 0 any slew is ok between OBs, >0 preference for shorter slews
    # delay: 0 any delay is ok between OBs, may delay to observe a higher ranked OB
    #        that will become visible shortly, >0 preference for smaller or no delays
    # rank: 0 rank not important, >0 preference for higher ranked programs
    # priority: (*only considered when looking at two OBs from the same program*)
    #           0 priority disregarded, >0 user's priority considered
    weights = conf['qplan']['weight']

    # create and initialize qplan scheduler
    sdlr = Scheduler(logger, observer)
    sdlr.set_weights(weights)
    sdlr.set_programs_info(pgms, False)
    sdlr.set_apriori_program_info(apriori_info)
    sdlr.set_scheduling_params(Bunch(limit_filter=None, allow_delay=True))

    # this defines the initial conditions during scheduling:
    # telescope parked, dome open, current sky conditions, etc.
    cur_data = Bunch(
        filters=[],
        cur_az=-90.0,
        cur_el=89.0,
        cur_rot=0.0,
        seeing=1.0,
        transparency=0.9,
        dome="open",
        categories=["open"],
        instruments=["PPC"],
    )

    # you can build your target list any way you want
    # see the loop below for details
    # Here I use an example table like we are considering for Phase 1 & 2
    # OB code	Priority	Effective Exp Time	Resolution	RA	DEC	Equinox	Object ID	Catalog ID	Comment
    #

    # get PPC list
    tab = Table.read(os.path.join(inputDirName, ppcList))
    tgt_tbl = ""
    for t in tab:
        c = SkyCoord(t["ppc_ra"], t["ppc_dec"], unit="deg")
        ra = c.ra.hms
        dec = c.dec.dms
        line = "  "
        line += f"{t['ppc_code']}\t"
        line += "0\t"
        line += "15\t"
        line += "L\t"
        line += f"{int(ra.h)}:{int(abs(ra.m))}:{abs(ra.s)}\t"
        line += f"{int(dec.d)}:{int(abs(dec.m))}:{abs(dec.s)}\t"
        line += "2000\t"
        line += f"design_{t['ppc_code']}\t"
        line += f"catalog_{t['ppc_code']}\t"
        line += f"{proposal}  extracted from ppcList"
        line += "\n"
        tgt_tbl += line

    obs = []
    for line in tgt_tbl.split("\n"):
        line = line.strip()
        if len(line) == 0:
            continue
        (
            ob_code,
            priority,
            exp_time,
            resolution,
            ra,
            dec,
            eq,
            obj_id,
            cat_id,
            comment,
        ) = line.split("\t")
        exp_time = float(exp_time) * 60.0  # assume table is in MINUTES
        print(dec)
        tgt = StaticTarget(
            name=ob_code, ra=ra, dec=dec, equinox=float(eq), comment=comment
        )
        ob = PPC_OB(
            id=ob_code,
            program=pgm,
            target=tgt,
            telcfg=telcfg,
            inscfg=inscfg,
            envcfg=envcfg,
            # total_time should really include instrument overheads
            # acct_time is time we charge to the PI
            acct_time=exp_time,
            total_time=exp_time,
            comment=f"{ra} / {dec}",
        )
        obs.append(ob)

    # let's now schedule a single night
    # for this function, we will make "schedule records" that
    # the scheduler will turn into actual Schedule objects
    # NOTE: for this function, start night is ALWAYS the official start of
    #       observation at sunset
    #       if scheduling a second half just set the start time to 00:30:00 etc
    rec = []
    for date in obs_dates:
        rec.append(
            Bunch(
                date=date,  # date HST
                starttime="18:30:00",  # time HST
                stoptime="05:30:00",  # time HST
                categories=["open"],
                skip=False,
                note="",
                data=cur_data,
            )
        )

    sdlr.set_schedule_info(rec)
    # set OB list to schedule from
    sdlr.set_oblist_info(obs)

    # now scheduling...
    sdlr.schedule_all()

    # get a summary report of what happened
    print("Summary_report:")
    print(sdlr.summary_report)

    # unschedulable OBs are available here
    print("unschedulable:", sdlr.unschedulable)

    # completed programs
    print("completed:", sdlr.completed)

    # incomplete programs
    # "obs" gives the unobserved OBs, "obcount" is the total number of OBs in the program,
    # "sched_time" is the scheduled time (including overheads charged to PI) in SEC,
    # "total_time" is TAC awarded time in SEC
    print("uncompleted:", sdlr.uncompleted)

    # individual schedules are available at `schedules`
    sch = sdlr.schedules

    # iterate over a schedule
    # if there is unused time at the end of a schedule, the "ob" attribute will be None
    slots = []
    obs_allo = []
    for s in sch:
        # print(s.printed())
        for slot in s.slots:
            slots.append(slot)
            if slot.ob is not None:
                print(
                    slot.start_time,
                    slot.stop_time,
                    slot.ob,
                    slot.ob.name,
                    slot.ob.comment,
                )
                obs_allo.append(slot.ob)
    # print('print slots...')
    # print(slots)
    # print(dir(obs_allo[0]))
    # print(type(slots[0].start_time))

    # the type of slot.ob will be a PPC_OB if it is not a delay or some other derived OB
    # for PPC_OBs, there will always be a setup and teardown part--these are normally used
    # by the OPE file generation code to properly set up status items, etc. on Gen2 for
    # the instrument to read and populate its FITS headers with queue keywords
    # We need to accurately set the proper setup and teardown times in qplan so that the
    # schedules are reasonably accurate in time

    def make_schedule_table(schedule):
        data = [
            (slot.start_time, slot.ob.name, slot.ob.target.ra, slot.ob.target.dec)
            for slot in schedule
            if slot.ob is not None and not slot.ob.derived
        ]
        targets = [
            (slot.start_time, slot.ob.target)
            for slot in schedule
            if slot.ob is not None and not slot.ob.derived
        ]
        df = pd.DataFrame(data, columns=["obstime", "ppc_code", "ppc_ra", "ppc_dec"])
        return df, targets

    df, targets = make_schedule_table(slots)
    df.to_csv(os.path.join(outputDirName, "result.csv"))
    print(df)

    # plot visibility plots for each night
    if plotVisibility == True:
        figs = []
        for obs_date in obs_dates:
            t = observer.get_date(obs_date)
            observer.set_date(t)
            sunset = observer.sunset()
            sunrise = observer.sunrise() + timedelta(days=1)
            print(sunset, sunrise)

            target_data = []
            for t, v in targets:
                if t > sunset and t < sunrise:
                    info_list = observer.get_target_info(v)
                    target_data.append(Bunch(history=info_list, target=v))
            if len(target_data) > 0:
                amp = airmass.AirMassPlot(800, 600, logger=logger)
                from matplotlib.backends.backend_agg import (
                    FigureCanvasAgg as FigureCanvas,
                )

                canvas = FigureCanvas(amp.fig)
                amp.plot_altitude(observer, target_data, observer.timezone)
                buf2 = BytesIO()
                canvas.print_figure(buf2, format="png")
                Image(data=bytes(buf2.getvalue()), format="png", embed=True)
                figs.append(amp)
                display(amp.fig)
    else:
        figs = None

    return df, sdlr, figs
