#!/usr/bin/env python3
# qPlan.py : queuePlanner

import os, sys
import warnings
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo

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
#from IPython.core.display import display
#from IPython.display import Image
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
from dateutil import parser


def run(conf, ppcList, inputDirName=".", outputDirName=".", plotVisibility=False, starttime_backup=[], stoptime_backup=[]):
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
    telcfg.min_el_deg = 32.0
    telcfg.max_el_deg = 75.0
    telcfg.min_rot_deg = -174.0
    telcfg.max_rot_deg = 174.0
    # PPCConfiguration -- PFS Pointing Center
    # if you have different times for your pointing centers, create a different one
    # for each exp_time, PA or resolution
    # inscfg = PPCConfiguration(exp_time=15 * 60.0, pa=0.0, resolution="low")
    # can be used later to constrain, if desired, for now set to allow anything
    # envcfg = EnvironmentConfiguration(seeing=99.0, transparency=0.0)

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
    weights = conf["qplan"]["weight"]

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
    overhead_add = 0.0
    if conf["ope"]["n_split_frame"] > 1:
        overhead_add = (
            float(conf["ope"]["n_split_frame"]) - 1
        ) * 60.0  # splitting into 1 more frame adds an readout time of ~60 seconds

    tab = Table.read(os.path.join(inputDirName, ppcList))
    tgt_tbl = ""
    for t in tab:
        c = SkyCoord(t["ppc_ra"], t["ppc_dec"], unit="deg")
        ra = c.ra.to_string(unit=u.hourangle, sep=":", precision=2, pad=True)
        dec = c.dec.to_string(sep=":", precision=2, pad=True)
        line = "  "
        line += f"{t['ppc_code']}\t"
        line += f"{t['ppc_priority_usr']}\t"
        line += f"{t['ppc_exptime'] + float(conf['qplan']['overhead'])*60.0 + overhead_add}\t"
        line += f"{t['ppc_pa']}\t"
        line += f"{t['ppc_resolution']}\t"
        line += f"{ra}\t"
        line += f"{dec}\t"
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
            pa,
            resolution,
            ra,
            dec,
            eq,
            obj_id,
            cat_id,
            comment,
        ) = line.split("\t")
        # pa =70
        # exp_time = float(exp_time) * 60.0  # assume table is in MINUTES
        exp_time = float(exp_time)  # exptime is in seconds
        pa = float(pa)
        #"""
        if "backup" in ob_code:
            priority = 100
        else:
            priority = 0
        #"""
        priority = float(priority) 

        if resolution == "L":
            resolution = "low"
        elif resolution == "M":
            resolution = "medium"

        tgt = StaticTarget(
            name=ob_code, ra=ra, dec=dec, equinox=float(eq), comment=comment
        )
        
        """
        if len(conf["qplan"]["start_time"]) > 0:
            start_time_too = datetime.strptime(
                conf["qplan"]["start_time"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=ZoneInfo("US/Hawaii"))
            start_time_too = start_time_too.astimezone(ZoneInfo("UTC"))
        else:
            start_time_too = None

        if len(conf["qplan"]["stop_time"]) > 0:
            stop_time_too = datetime.strptime(
                conf["qplan"]["stop_time"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=ZoneInfo("US/Hawaii"))
            stop_time_too = stop_time_too.astimezone(ZoneInfo("UTC"))
        else:
            stop_time_too = None
        #"""
        start_time_too = None
        stop_time_too = None
        
        ob = PPC_OB(
            id=ob_code,
            program=pgm,
            target=tgt,
            telcfg=telcfg,
            inscfg=PPCConfiguration(exp_time=exp_time, pa=pa, resolution=resolution),
            envcfg=EnvironmentConfiguration(
                seeing=99.0,
                transparency=0.0,
                lower_time_limit=start_time_too,
                upper_time_limit=stop_time_too,
            ),
            # total_time should really include instrument overheads
            # acct_time is time we charge to the PI
            acct_time=exp_time,
            total_time=exp_time,
            priority=priority,
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

    start_time_list = conf["qplan"].get("start_time", [])
    stop_time_list = conf["qplan"].get("stop_time", [])

    today = date.today().strftime("%Y-%m-%d")
    for date_ in sorted(conf["qplan"]["obs_dates"], key=lambda d: parser.parse(d)):
        date_t = parser.parse(f"{date_} 12:00 HST")
        date_today = parser.parse(f"{today} 12:00 HST")

        if date_today > date_t:
            continue
            
        observer.set_date(date_t)
        default_start_time = observer.evening_twilight_18()
        default_stop_time = observer.morning_twilight_18() + timedelta(minutes=30) #extend TW18 by 30 min for real operation, just in case

        start_override = None
        stop_override = None
        
        for item in start_time_list:
            next_date = (datetime.strptime(date_, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if (date_ in item) and parser.parse(f"{item} HST") > default_start_time:
                start_override = parser.parse(f"{item} HST")
            elif (next_date in item) and parser.parse(f"{item} HST") < default_stop_time:
                start_override = parser.parse(f"{item} HST")

        for item in stop_time_list:
            next_date = (datetime.strptime(date_, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if (date_ in item) and parser.parse(f"{item} HST") > default_start_time:
                stop_override = parser.parse(f"{item} HST")
            elif (next_date in item) and parser.parse(f"{item} HST") < default_stop_time:
                stop_override = parser.parse(f"{item} HST")

        if start_override is not None:
            start_time = start_override
        else:
            start_time = default_start_time

        if stop_override is not None:
            stop_time = stop_override
        else:
            stop_time = default_stop_time

        print(f"{date_}: start obs. at {start_time}, stop obs. at {stop_time}")

        if start_time == default_start_time and stop_time == default_stop_time:
            # Calculate refocus start time as datetime
            time_refocus_start = default_start_time + timedelta(minutes=(23.0 + float(conf['qplan']['overhead'])) * 2)
            
            # Then compute stop time based on start time
            time_refocus_stop = time_refocus_start + timedelta(minutes=10.0)

            print(time_refocus_start, time_refocus_stop)

            #"""
            if len(starttime_backup) == 0:
                rec.append(
                    Bunch(
                        date=date_,  # date HST
                        starttime=start_time,  # time HST
                        stoptime=parser.parse(f"{time_refocus_start.strftime('%Y-%m-%d %H:%M:%S')} HST"),  # time HST
                        categories=["open"],
                        skip=False,
                        note="",
                        data=cur_data,
                    )
                )
                rec.append(
                    Bunch(
                        date=date_,  # date HST
                        starttime=parser.parse(f"{time_refocus_stop.strftime('%Y-%m-%d %H:%M:%S')} HST"),  # time HST
                        stoptime=stop_time,  # time HST
                        categories=["open"],
                        skip=False,
                        note="",
                        data=cur_data,
                    )
                )
            else:
                for ii in range(len(starttime_backup)):
                    if starttime_backup[ii] < default_start_time:
                        starttime_backup[ii] = default_start_time
                    if stoptime_backup[ii] > default_stop_time:
                        stoptime_backup[ii] = default_stop_time
                        
                    if time_refocus_start >= starttime_backup[ii] and time_refocus_stop <= stoptime_backup[ii]:
                        rec.append(
                            Bunch(
                                date=date_,  # date HST
                                starttime=starttime_backup[ii],  # time HST
                                stoptime=parser.parse(f"{time_refocus_start.strftime('%Y-%m-%d %H:%M:%S')} HST"),  # time HST
                                categories=["open"],
                                skip=False,
                                note="",
                                data=cur_data,
                            )
                        )
                        rec.append(
                            Bunch(
                                date=date_,  # date HST
                                starttime=parser.parse(f"{time_refocus_stop.strftime('%Y-%m-%d %H:%M:%S')} HST"),  # time HST
                                stoptime=stoptime_backup[ii],  # time HST
                                categories=["open"],
                                skip=False,
                                note="",
                                data=cur_data,
                            )
                        )
                    else:
                        rec.append(
                            Bunch(
                                date=date_,  # date HST
                                starttime=starttime_backup[ii],
                                stoptime=stoptime_backup[ii],  # time HST
                                categories=["open"],
                                skip=False,
                                note="",
                                data=cur_data,
                            )
                        ) 
            #"""
            
        else:
            if len(starttime_backup) == 0:
                rec.append(
                    Bunch(
                        date=date_,  # date HST
                        starttime=start_time,  # time HST
                        stoptime=stop_time,  # time HST
                        categories=["open"],
                        skip=False,
                        note="",
                        data=cur_data,
                    )
                )
            else:
                for ii in range(len(starttime_backup)):
                    if starttime_backup[ii] < default_start_time:
                        starttime_backup[ii] = default_start_time
                    if stoptime_backup[ii] > default_stop_time:
                        stoptime_backup[ii] = default_stop_time

                    rec.append(
                        Bunch(
                            date=date_,  # date HST
                            starttime=starttime_backup[ii],
                            stoptime=stoptime_backup[ii],  # time HST
                            categories=["open"],
                            skip=False,
                            note="",
                            data=cur_data,
                        )
                    ) 
                    
        if conf["ppp"]["daily_plan"]:
            break

    if len(rec) == 0:
        print("Error: No time slots available.")
        sys.exit(1)

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
            (
                slot.start_time,
                slot.ob.name,
                slot.ob.target.ra,
                slot.ob.target.dec,
                slot.ob.inscfg.pa,
                slot.ob.priority,
            )
            for slot in schedule
            if slot.ob is not None and not slot.ob.derived
        ]
        targets = [
            (slot.start_time, slot.ob.target)
            for slot in schedule
            if slot.ob is not None and not slot.ob.derived
        ]
        df = pd.DataFrame(
            data,
            columns=[
                "obstime",
                "ppc_code",
                "ppc_ra",
                "ppc_dec",
                "ppc_pa",
                "ppc_priority",
            ],
        )
        return df, targets

    df, targets = make_schedule_table(slots)
    print(df)

    # plot visibility plots for each night
    if plotVisibility == True:
        figs = []
        for obs_date in conf["qplan"]["obs_dates"]:
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
