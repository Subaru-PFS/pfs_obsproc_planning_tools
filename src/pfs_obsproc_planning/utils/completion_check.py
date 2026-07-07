#!/usr/bin/env python3
# completion_check.py : Subaru Fiber Allocation software
import os
import random
import warnings
from datetime import datetime, time, timedelta
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, join
from astropy.io import fits
from astropy import units as u
from matplotlib.backends.backend_pdf import PdfPages

from loguru import logger

warnings.filterwarnings("ignore")


def run(conf, workDir="."):
    # read obList
    tb_tgt = Table.read(os.path.join(workDir, "ppp/obList_tot.ecsv"))
    try:
        tb_tgt_backup = Table.read(os.path.join(workDir, "ppp/obList_backup.ecsv"))
    except Exception:
        tb_tgt_backup = Table()
    tb_tgt = vstack([tb_tgt, tb_tgt_backup])

    # read ppclist
    tb_ppc = Table.read(os.path.join(workDir, "ppp/ppcList.ecsv"))
    try:
        tb_ppc_backup = Table.read(os.path.join(workDir, "ppp/ppcList_backup.ecsv"))
    except Exception:
        tb_ppc_backup = Table()
    tb_ppc = vstack([tb_ppc, tb_ppc_backup])

    # read target list in qDB
    today_str = datetime.today().strftime("%Y%m%d")
    base = os.path.join(workDir, "ppp", f"tgt_queueDB_{today_str}")
    main_csv = f"{base}.csv"
    backup_csv = f"{base}_backup.csv"

    # --- read main table ---
    try:
        tb_queue = Table.read(main_csv)
    except Exception as e:
        logger.warning(f"[EET] Could not read {main_csv}: {e}")
        tb_queue = Table()

    # --- read backup table ---
    try:
        tb_queue_backup = Table.read(backup_csv)
        tb_queue_backup["ob_code"] = tb_queue_backup["ob_code"].astype(str)
    except Exception as e:
        logger.error(f"[EET] Could not read {backup_csv}: {e}")
        tb_queue_backup = Table()

    # --- combine ---
    if len(tb_queue) > 0 and len(tb_queue_backup) > 0:
        tb_queue = vstack([tb_queue, tb_queue_backup])
    elif len(tb_queue) == 0 and len(tb_queue_backup) > 0:
        tb_queue = tb_queue_backup

    pdf = PdfPages(os.path.join(workDir, "check-S26A-queue.pdf"))

    plot_ppc(conf, tb_tgt, tb_ppc, pdf)
    plot_assign(conf, workDir, pdf)
    plot_schedule(workDir, pdf)
    plot_EET(workDir, tb_queue, pdf)
    plot_CR(conf, tb_tgt, tb_queue, workDir, pdf)
    save_proposal_sum_csv(conf, workDir)

    pdf.close()


def save_proposal_sum_csv(conf, workDir):
    """Save per-date per-proposal executed pointing counts (nppc) as CSV.

    The output file is saved in the same folder as the completion PDF:
    ``{workDir}/proposal_sum.csv``.
    """
    try:
        from pfs_design_tool.pointing_utils.dbutils import connect_qadb
    except Exception as e:
        logger.error(f"[proposal_sum] Failed to import connect_qadb: {e}")
        return None

    # infer semester from proposal IDs in config (e.g. S26A)
    proposal_ids = conf.get("ppp", {}).get("proposalIds", []) + conf.get("ppp", {}).get(
        "proposalIds_backup", []
    )
    if len(proposal_ids) == 0:
        logger.warning("[proposal_sum] No proposalIds found in config.")
        return None

    semester_code = str(proposal_ids[0]).split("-")[0]

    output_path = os.path.join(workDir, "proposal_nppc.csv")

    # Find all semester queue runs (e.g. /work/wanqqq/run_2603/S26A-queue,
    # /work/wanqqq/run_2605/S26A-queue) under the common run root.
    base_dir = "/work/wanqqq"
    run_queue_dirs = sorted(glob(os.path.join(base_dir, "run_*", f"{semester_code}-queue")))

    # Map design FITS filename -> absolute path from all
    # /work/wanqqq/run_YYMM/Sxxx-queue/output_YYMMDD/design/*.fits
    design_path_map = {}
    for run_queue_dir in run_queue_dirs:
        for fits_path in glob(os.path.join(run_queue_dir, "output_*", "design", "pfsDesign-0x*.fits")):
            design_path_map[os.path.basename(fits_path)] = fits_path

    if len(design_path_map) == 0:
        logger.warning(f"[proposal_nppc] No design FITS found for {semester_code} under {base_dir}")
        pd.DataFrame(columns=["date"]).to_csv(output_path, index=False)
        return None

    # Query qaDB: executed visits with pfs_design_id and started_at
    conn = connect_qadb(conf)
    try:
        sql = """
        SELECT
            pfs_visit.pfs_visit_id,
            pfs_visit.pfs_design_id,
            onsite_processing_status.started_at
        FROM exposure_time
            JOIN pfs_visit ON exposure_time.pfs_visit_id = pfs_visit.pfs_visit_id
            JOIN onsite_processing_status ON onsite_processing_status.pfs_visit_id = pfs_visit.pfs_visit_id
        WHERE pfs_visit.pfs_design_id IS NOT NULL AND pfs_visit.pfs_visit_id >=129587
        ORDER BY onsite_processing_status.started_at ASC, pfs_visit.pfs_visit_id ASC;
        """

        with conn.cursor() as cur:
            cur.execute(sql)
            df_design_done = pd.DataFrame(
                cur.fetchall(),
                columns=["pfs_visit_id", "pfs_design_id", "started_at"],
            )
    except Exception as e:
        logger.error(f"[proposal_nppc] Failed to query qaDB: {e}")
        return None
    finally:
        conn.close()

    if df_design_done.empty:
        logger.warning("[proposal_nppc] qaDB returned no executed designs.")
        pd.DataFrame(columns=["date"]).to_csv(output_path, index=False)
        return None

    # Convert each visit timestamp into HST night date:
    # e.g., 2026-05-14 05:00 belongs to night 2026-05-13.
    def hst_night_date(ts):
        t = pd.to_datetime(ts)
        if pd.isna(t):
            return None
        if t.hour < 12:
            t = t - pd.Timedelta(days=1)
        return t.date().isoformat()

    # Treat adjacent visits of the same design within the same HST night
    # as one pointing (2 adjacent visits => 1 count).
    # For a run length N, pointing count is ceil(N/2) == (N + 1) // 2.
    visit_runs = {}
    prev_key = None
    run_len = 0

    for _, row in df_design_done.iterrows():
        design_id = row["pfs_design_id"]
        if pd.isna(design_id):
            continue

        obs_date = hst_night_date(row["started_at"])
        if obs_date is None:
            continue

        try:
            design_id_int = int(design_id)
        except Exception:
            continue

        key = (obs_date, design_id_int)
        if key == prev_key:
            run_len += 1
        else:
            if prev_key is not None and run_len > 0:
                visit_runs[prev_key] = visit_runs.get(prev_key, 0) + (run_len + 1) // 2
            prev_key = key
            run_len = 1

    if prev_key is not None and run_len > 0:
        visit_runs[prev_key] = visit_runs.get(prev_key, 0) + (run_len + 1) // 2

    # Cache proposal IDs present in each design FITS once.
    design_proposals_cache = {}

    # Accumulate nppc(date, proposal_id) from collapsed pointing counts.
    # Count design if proposal appears in targetType==1
    # (excluding observed filler proposal S25A-000QF).
    per_date_counts = {}

    for (obs_date, design_id_int), n_pointings in visit_runs.items():
        try:
            fname = f"pfsDesign-0x{design_id_int:016x}.fits"
        except Exception:
            continue

        filepath = design_path_map.get(fname)
        if filepath is None:
            continue

        if design_id_int not in design_proposals_cache:
            try:
                with fits.open(filepath, memmap=True) as hdul:
                    data = hdul[1].data
                    if data is None or len(data) == 0:
                        design_proposals_cache[design_id_int] = []
                        continue

                    mask_science = data["targetType"] == 1
                    raw_ids = np.unique(data["proposalId"][mask_science])
                    proposal_ids_in_design = []
                    for raw_pid in raw_ids:
                        if isinstance(raw_pid, (bytes, np.bytes_)):
                            pid = raw_pid.decode("utf-8", errors="ignore").strip()
                        else:
                            pid = str(raw_pid).strip()

                        if pid.startswith(semester_code) and pid != "S25A-000QF":
                            proposal_ids_in_design.append(pid)

                    design_proposals_cache[design_id_int] = proposal_ids_in_design
            except Exception:
                continue

        proposal_ids_in_design = design_proposals_cache.get(design_id_int, [])

        if obs_date not in per_date_counts:
            per_date_counts[obs_date] = {}

        for pid in proposal_ids_in_design:
            per_date_counts[obs_date][pid] = per_date_counts[obs_date].get(pid, 0) + n_pointings

    # Build wide table: one row per date, one column per proposal_id
    configured_semester_proposals = {
        str(pid) for pid in proposal_ids if str(pid).startswith(semester_code)
    }
    counted_proposals = {
        pid for date_counts in per_date_counts.values() for pid in date_counts.keys()
    }
    all_proposals = sorted(configured_semester_proposals | counted_proposals)

    rows = []
    for date_str in sorted(per_date_counts.keys()):
        row = {"date": date_str}
        for pid in all_proposals:
            row[pid] = per_date_counts[date_str].get(pid, 0)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("date").reset_index(drop=True)
    else:
        df_out = pd.DataFrame(columns=["date"] + all_proposals)

    df_out.to_csv(output_path, index=False)
    logger.info(f"[proposal_sum] Saved {output_path}")

    return None


def plot_ppc(conf, tb_tgt, tb_ppc, pdf):
    """
    Plot the distribution of targets and PPC positions with PFS FoV hexagon overlays.
    Saves the plot to the given pdf.
    """

    def PFS_FoV_plot(ppc_ra, ppc_dec, PA, line_color, line_width, line_st):
        """
        Draw hexagonal field-of-view for each PPC center.
        """
        for ra, dec, pa in zip(ppc_ra, ppc_dec, PA):
            center = SkyCoord(ra * u.deg, dec * u.deg)
            # Hexagon: 6 corners plus one to close, rotated by pa
            angles = np.array([30, 90, 150, 210, 270, 330, 30]) + pa
            hexagon = center.directional_offset_by(angles * u.deg, 1.38 / 2.0 * u.deg)
            ra_h, dec_h = hexagon.ra.deg, hexagon.dec.deg
            # Correct wrap-around at RA=0/360 if needed
            diff = np.abs(ra_h - center.ra.deg)
            if np.any(diff > 180):
                ra_h = np.where(ra_h > 180, ra_h - 360, ra_h)
            plt.plot(
                ra_h,
                dec_h,
                color=line_color,
                lw=line_width,
                ls=line_st,
                alpha=0.5,
                zorder=5,
            )

    plt.figure(figsize=(8, 3))

    proposal_ids = conf["ppp"]["proposalIds"]
    color_list = [
        (random.random(), random.random(), random.random()) for _ in proposal_ids
    ]

    # Plot targets by proposal
    for idx, proposal_id in enumerate(proposal_ids):
        targets = tb_tgt[tb_tgt["proposal_id"] == proposal_id]
        plt.plot(
            targets["ob_ra"],
            targets["ob_dec"],
            "o",
            mfc=color_list[idx],
            mec="none",
            ms=5,
            alpha=0.8,
            label=proposal_id,
        )

    # Overlay all PPC hexagons
    PFS_FoV_plot(
        tb_ppc["ppc_ra"],
        tb_ppc["ppc_dec"],
        tb_ppc["ppc_pa"],
        line_color="k",
        line_width=1,
        line_st="-",
    )

    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("Target & PPC Distribution")
    plt.legend(fontsize=8, loc="best", markerscale=0.7)
    pdf.savefig(bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plot_assign(conf, workDir, pdf):
    """
    Plots fiber assignment summary and a stacked bar chart of fiber types per field.
    """
    # 1. Gather SFA summary from all design FITS files
    pfsdeg_files = sorted(glob(os.path.join(workDir, "design", "*.fits")))
    sfa_info = []
    for fits_file in pfsdeg_files:
        hdul = fits.open(fits_file)
        hdr = hdul[1].header
        data = hdul[1].data
        n_guides = len(hdul[3].data) if len(hdul) > 3 else 0

        # Fiber counts by type
        n_sci = np.sum(
            (data["targettype"] == 1)
            & ~np.isin(data["proposalID"], conf["sfa"]["proposalIds_obsFiller"])
        )
        n_filler = np.sum(
            (data["targettype"] == 1)
            & np.isin(data["proposalID"], conf["sfa"]["proposalIds_obsFiller"])
        )
        n_sky = np.sum(data["targettype"] == 2)
        n_fstar = np.sum(data["targettype"] == 3)
        n_blank = np.sum((data["targettype"] == 4) & (data["fiberStatus"] == 1))
        frac_sci = n_sci / 2394.0 * 100

        # SFA info for each field
        sfa_info.append(
            [
                hdr["DSGN_NAM"],
                f"0x{hdr['W_PFDSGN']:016x}",
                float(hdr["RA"]),
                float(hdr["DEC"]),
                float(hdr["POSANG"]),
                n_sci,
                n_filler,
                n_sky,
                n_fstar,
                n_blank,
                n_guides,
                frac_sci,
            ]
        )
        hdul.close()

    # 2. Convert to Astropy Table for easy handling
    col_names = [
        "ppc_code",
        "designId",
        "ppc_ra",
        "ppc_dec",
        "ppc_pa",
        "N_sci",
        "N_filler",
        "N_sky",
        "N_fstar",
        "N_blank",
        "N_guide",
        "N_sci_frac",
    ]
    tb_sfa = Table(rows=sfa_info, names=col_names)
    tb_sfa["field"] = [
        f"({ra:.2f}, {dec:.2f}, {pa:.2f})"
        for ra, dec, pa in zip(tb_sfa["ppc_ra"], tb_sfa["ppc_dec"], tb_sfa["ppc_pa"])
    ]
    tb_sfa = tb_sfa.group_by("N_blank")

    # 3. Plot summary table
    df = tb_sfa.to_pandas()
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.3 + 1))
    ax.axis("tight")
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="best")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    pdf.savefig(bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # 4. Plot fiber assignment stacked bar chart
    N = len(tb_sfa)
    ind = np.arange(N)
    width = 0.9

    plt.figure(figsize=(15, 3))
    plt.bar(
        ind,
        tb_sfa["N_sci"] + tb_sfa["N_filler"] + tb_sfa["N_sky"] + tb_sfa["N_fstar"],
        width=width,
        color="orange",
        label="Filler targets",
        alpha=1,
    )
    plt.bar(
        ind,
        tb_sfa["N_sci"] + tb_sfa["N_sky"] + tb_sfa["N_fstar"],
        width=width,
        color="dodgerblue",
        label="Blank sky",
        alpha=1,
    )
    plt.bar(
        ind,
        tb_sfa["N_sci"] + tb_sfa["N_fstar"],
        width=width,
        color="darkgreen",
        label="Flux calibrators",
        alpha=1,
    )
    plt.bar(
        ind,
        tb_sfa["N_sci"],
        width=width,
        color="tomato",
        label="Science targets",
        alpha=1,
    )

    plt.plot([-1, N], [2394, 2394], "--", color="tomato", lw=2, label="Nmax=2394")
    plt.legend(loc="best", bbox_to_anchor=(1.08, 0.7), fontsize=12)
    plt.title("Fiber assignment (queue)")
    plt.xticks(ind, tb_sfa["field"], rotation=75)
    plt.ylabel("N(used fiber)")
    plt.xlim(-0.5, N)
    plt.ylim(0, 2500)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Secondary axis for science fraction
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(ind, tb_sfa["N_sci_frac"], "-.", color="yellow", lw=2, label="N_sci frac")
    ax2.set_ylabel("N_sci / N_max [%]")
    ax2.set_ylim(0, 100)
    ax2.grid()
    ax2.legend()

    pdf.savefig(bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plot_schedule(workDir, pdf):
    """
    Plot the observation schedule for each night, handling both standard and partial (late-night) schedules.
    """
    schedule_csv = os.path.join(workDir, "qplan", "result.csv")
    df_qplan = pd.read_csv(schedule_csv)
    df_qplan["obstime"] = pd.to_datetime(df_qplan["obstime"].str[:19])
    df_qplan["obstime_end"] = df_qplan["obstime"] + pd.Timedelta(seconds=1260)
    df_qplan["alpha"] = 1 - df_qplan["ppc_priority"] / df_qplan["ppc_priority"].max()

    date_list = sorted(set(df_qplan["obstime"].dt.date.tolist()))
    for obs_date in date_list:
        # Define observation window for the night.
        obs_window_start = datetime.combine(obs_date, time(18, 30))
        obs_window_end = datetime.combine(obs_date, time(5, 30)) + timedelta(days=1)

        xlim_start, xlim_stop = [obs_window_start, obs_window_end]

        # Filter for this night.
        df_window = df_qplan[
            (df_qplan["obstime"] >= obs_window_start)
            & (df_qplan["obstime"] <= obs_window_end)
        ]

        plt.figure(figsize=(10, 0.5))

        if df_window.empty:
            obs_window_start = datetime.combine(obs_date, time(0, 0))
            obs_window_end = datetime.combine(obs_date, time(5, 30))

            # Filter for this night.
            df_window = df_qplan[
                (df_qplan["obstime"] >= obs_window_start)
                & (df_qplan["obstime"] <= obs_window_end)
            ]

            if df_window.empty:
                obs_window_start = datetime.combine(obs_date, time(18, 30))
                obs_window_end = datetime.combine(obs_date, time(5, 30)) + timedelta(
                    days=1
                )

                plt.plot(
                    [obs_window_start, obs_window_end],
                    [1, 1],
                    ls="-",
                    color="white",
                    alpha=0.8,
                    lw=30,
                    solid_capstyle="butt",
                )
                plt.title(f"Schedule for the night {obs_date} (HST)", fontsize=10)
            else:
                xlim_start, xlim_stop = [
                    obs_window_end - timedelta(hours=11),
                    obs_window_end,
                ]
                obs_date -= timedelta(days=1)

        for _, row in df_window.iterrows():
            # Plot vertical red lines at start and end times.
            color_ = "gray" if "backup" in row["ppc_code"] else "tomato"

            plt.plot(
                [row["obstime"], row["obstime"]],
                [0, 2],
                ls="-",
                color="r",
                alpha=1,
                lw=1,
                solid_capstyle="butt",
            )
            plt.plot(
                [row["obstime_end"], row["obstime_end"]],
                [0, 2],
                ls="-",
                color="r",
                alpha=1,
                lw=1,
                solid_capstyle="butt",
            )
            # Plot a horizontal bar for the observation.
            plt.plot(
                [row["obstime"], row["obstime_end"]],
                [1, 1],
                ls="-",
                color=color_,
                alpha=0.8,
                lw=30,
                solid_capstyle="butt",
            )

            # Add annotation: place text at the center of the bar.
            mid_time = row["obstime"] + (row["obstime_end"] - row["obstime"]) / 2
            annotation_text = (
                f"({row['ppc_ra']:.2f}, {row['ppc_dec']:.2f}, {row['ppc_pa']:.1f})"
            )
            # Place the text slightly above the bar (y=1.1)
            plt.text(
                mid_time,
                1.06,
                annotation_text,
                fontsize=6,
                ha="center",
                va="bottom",
                rotation=90,
            )

            plt.title(f"Schedule for the night {obs_date} (HST)", fontsize=10, pad=80)
        plt.xticks(fontsize=8, rotation=45)
        plt.xlim(xlim_start, xlim_stop)
        plt.ylim(0.95, 1.05)
        plt.yticks([], [])
        plt.tick_params(
            axis="x",
            which="both",
            labelbottom=True,
            labeltop=False,
            bottom=True,
            top=False,
        )
        pdf.savefig(bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plot_EET(workDir, tb_queue, pdf):
    """
    Plot a pairwise correlation matrix (Seaborn pairplot) of effective exposure times
    for each arm, sampled from today's queueDB.
    """
    if len(tb_queue) == 0:
        return None
        
    # 1. Convert to pandas DataFrame and downsample for visualization clarity        
    df = tb_queue.to_pandas()
    if len(df) > 0:
        df = df.sample(frac=0.05, random_state=42).reset_index(drop=True)

    # 2. Select columns for correlation analysis (B, R, M, N arms)
    cols = [
        "eff_exptime_done_real_b",
        "eff_exptime_done_real_r",
        "eff_exptime_done_real_m",
        "eff_exptime_done_real_n",
    ]
    for c in cols:
        if c not in df.columns:
            logger.error(f"[EET] Missing column: {c}")
            return

    # 3. Create Seaborn pairplot and add one-to-one line in each subplot
    g = sns.pairplot(
        df[cols],
        corner=True,
        diag_kind=None,
        plot_kws={"marker": ".", "color": "orange"},
    )

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax = g.axes[i, j]
            if ax is not None:
                x_min, x_max = df[cols[i]].min(), df[cols[i]].max()
                y_min, y_max = df[cols[j]].min(), df[cols[j]].max()
                line_min = min(x_min, y_min)
                line_max = max(x_max, y_max)
                ax.plot(
                    [line_min, line_max],
                    [line_min, line_max],
                    color="k",
                    linestyle="--",
                    lw=2,
                    zorder=10,
                )

    # 4. Save to PDF
    plt.tight_layout()
    pdf.savefig(bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plot_CR(conf, tb_tgt, tb_queue, workDir, pdf):
    """
    Plot completion rates (bar charts) for each proposal.
    """

    # --- Prepare target table ---
    if len(tb_queue) > 0:
        tb_tgt = join(tb_tgt, tb_queue,
                        keys_left=["proposal_id", "ob_code"],
                        keys_right=["psl_id", "ob_code"],
                        join_type="left")
        tb_tgt.rename_column("ob_code_1", "ob_code")
    else:
        tb_tgt["ob_exptime_usr"] = 0.0
        tb_tgt["eff_exptime_done_real"] = 0.0
        tb_tgt["exptime_done_real"] = 0.0
    
    exptime_usr = np.ma.filled(tb_tgt["ob_exptime_usr"], 0.0)
    exptime_done_real = np.ma.filled(tb_tgt["eff_exptime_done_real"], 0.0)

    exptime_usr = exptime_usr.astype(float)
    exptime_done_real = exptime_done_real.astype(float)
    
    tb_tgt["eff_exptime_done_rec"] = np.minimum(exptime_usr, exptime_done_real)

    #cols_to_remove = [c for c in tb_tgt.colnames if c in tb_queuedb.colnames and "ob_code" not in c]
    #tb_tgt.remove_columns(cols_to_remove)

    tb_tgt["exptime_assign"] = 0.0

    # --- Calculate expected exposure time ---
    pfsdeg_files = glob(os.path.join(workDir, "design/*.fits"))
    tb_tgt["exptime_exp"] = 0
    for file in pfsdeg_files:
        hdul = fits.open(file)
        obcode_assign = [
            row["obCode"] for row in hdul[1].data if row["obCode"] != "N/A"
        ]
        mask = np.isin(tb_tgt["ob_code"].data, obcode_assign)
        tb_tgt["exptime_exp"][mask] += 900

    # --- Proposal ID lists ---
    all_psl_ids = conf["ppp"]["proposalIds"] + conf["ppp"]["proposalIds_backup"]

    # --- Collect stats per proposal ---
    fh_tot, fh_alloc, fh_com, fh_achieve, fh_exe, fh_exp = [], [], [], [], [], []
    for psl_id in all_psl_ids:
        queue_ = tb_queue[tb_queue["psl_id"] == psl_id]
        tgt_ = tb_tgt[tb_tgt["proposal_id"] == psl_id]

        fh_tot_ = np.sum(tgt_["ob_exptime_usr"]) / 3600.0
        fh_allo_ = list(set(tgt_["allocated_time_tac"]))[0] if len(tgt_) > 0 else 0
        fh_com_ = (
            np.sum(
                tgt_["eff_exptime_done_rec"][
                    tgt_["eff_exptime_done_rec"] >= tgt_["ob_exptime_usr"]
                ]
            )
            / 3600.0
        )
        fh_now_ = np.sum(tgt_["eff_exptime_done_rec"]) / 3600.0
        fh_real_ = np.sum(tgt_["exptime_done_real"]) / 3600.0
        fh_exp_ = np.sum(tgt_["exptime_exp"]) / 3600.0 + fh_now_

        logger.info(
            f"{psl_id}, FH_tot={fh_tot_:.2f}, FH_alloc={fh_allo_}, FH_com={fh_com_:.2f}, FH_achieve={fh_now_:.2f}, FH_exe={fh_real_:.2f}, FH_exp={fh_exp_:.2f}, CR={fh_now_/fh_allo_*100 if fh_allo_ else 0:.2f}%"
        )

        fh_tot.append(fh_tot_)
        fh_alloc.append(fh_allo_)
        fh_com.append(fh_com_)
        fh_achieve.append(fh_now_)
        fh_exe.append(fh_real_)
        fh_exp.append(fh_exp_)

    # --- Prepare groups for plotting ---
    ids_B = conf["ppp"]["proposalIds"]
    ids_C = [pid for pid in conf["ppp"]["proposalIds_backup"] if pid.endswith("QN")]
    ids_F = [pid for pid in conf["ppp"]["proposalIds_backup"] if pid.endswith("QF")]
    split_B = len(ids_B)
    split_C = split_B + len(ids_C)

    def _plot_group(
        ax, indices, label_ids, fh_exp, fh_alloc, fh_com, fh_achieve, fh_exe
    ):
        bar_height = 0.15
        bars_exp = ax.barh(
            indices + 4 * bar_height,
            fh_exp,
            bar_height,
            label="FH_exp",
            color="lightblue",
            alpha=0.8,
        )
        bars_alloc = ax.barh(
            indices + 0 * bar_height,
            fh_alloc,
            bar_height,
            label="FH_alloc",
            color="#FDB863",
            alpha=0.7,
        )
        bars_com = ax.barh(
            indices + 1 * bar_height,
            fh_com,
            bar_height,
            label="FH_com",
            color="#80B1D3",
            alpha=0.7,
        )
        bars_achieve = ax.barh(
            indices + 2 * bar_height,
            fh_achieve,
            bar_height,
            label="FH_achieve",
            color="tomato",
            alpha=1,
        )
        bars_exe = ax.barh(
            indices + 3 * bar_height,
            fh_exe,
            bar_height,
            label="FH_exe",
            color="gray",
            alpha=0.3,
        )

        # Annotate bars
        for i, bar in enumerate(bars_achieve):
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2.0
            ratio = (
                (fh_achieve[i] / fh_alloc[i]) * 100 if fh_alloc[i] else fh_achieve[i]
            )
            text_ = f"{ratio:.0f}%" if fh_alloc[i] else f"{ratio:.1f}"
            ax.text(
                x + 0.05 * (fh_alloc[i] or 1),
                y,
                text_,
                va="center",
                fontsize=9,
                color="tomato",
                fontweight="bold",
            )

        for i, bar in enumerate(bars_com):
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2.0
            ratio = (fh_com[i] / fh_alloc[i]) * 100 if fh_alloc[i] else fh_com[i]
            text_ = f"{ratio:.0f}%" if fh_alloc[i] else f"{ratio:.1f}"
            ax.text(x + 0.05 * (fh_alloc[i] or 1), y, text_, va="center", fontsize=8)

        for i, bar in enumerate(bars_exe):
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2.0
            ratio = (fh_exe[i] / fh_alloc[i]) * 100 if fh_alloc[i] else fh_exe[i]
            text_ = f"{ratio:.0f}%" if fh_alloc[i] else f"{ratio:.1f}"
            ax.text(x + 0.05 * (fh_alloc[i] or 1), y, text_, va="center", fontsize=8)

        for i, bar in enumerate(bars_exp):
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2.0
            ratio = (fh_exp[i] / fh_alloc[i]) * 100 if fh_alloc[i] else fh_exp[i]
            text_ = f"{ratio:.0f}%" if fh_alloc[i] else f"{ratio:.1f}"
            ax.text(x + 0.05 * (fh_alloc[i] or 1), y, text_, va="center", fontsize=8)

        ax.set_yticks(indices + 2 * bar_height)
        ax.set_yticklabels(label_ids)
        ax.set_xlabel("FH (hours)")
        ax.set_ylabel("Proposal ID")
        ax.legend()
        ax.invert_yaxis()

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))

    # Grade B
    indices_B = np.arange(len(ids_B))
    _plot_group(
        axes[0],
        indices_B,
        ids_B,
        fh_exp[:split_B],
        fh_alloc[:split_B],
        fh_com[:split_B],
        fh_achieve[:split_B],
        fh_exe[:split_B],
    )
    axes[0].set_title("Completion rates (Grade B)")

    # Grade C
    indices_C = np.arange(len(ids_C))
    _plot_group(
        axes[1],
        indices_C,
        ids_C,
        fh_exp[split_B:split_C],
        fh_alloc[split_B:split_C],
        fh_com[split_B:split_C],
        fh_achieve[split_B:split_C],
        fh_exe[split_B:split_C],
    )
    axes[1].set_title("Completion rates (Grade C)")

    # Grade F
    indices_F = np.arange(len(ids_F))
    _plot_group(
        axes[2],
        indices_F,
        ids_F,
        fh_exp[split_C:],
        fh_alloc[split_C:],
        fh_com[split_C:],
        fh_achieve[split_C:],
        fh_exe[split_C:],
    )
    axes[2].set_title("Completion rates (Grade F)")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
