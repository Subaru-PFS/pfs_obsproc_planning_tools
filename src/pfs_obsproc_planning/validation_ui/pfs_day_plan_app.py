import csv
import ast
import glob
import re
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from bs4 import BeautifulSoup
import pandas as pd
import panel as pn

# Compact/scale the calendar popup so it doesn't take excessive screen
# space. The CSS is kept in a separate file `styles.css` in this
# package directory; we load it here so Panel still receives the raw
# CSS at startup.
css_path = Path(__file__).parent / "styles.css"
css_text = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

pn.extension(
    "tabulator",
    notifications=True,
    raw_css=[css_text],
)

# Path to CSV produced by the daily processing pipeline. Adjust as needed.
CSV_PATH = "/home/wanqiu/data/HE/PFS_frame/git/work/wanqqq/daily_process_status.csv"
PROPOSAL_SUM_CSV_PATH = "/home/wanqiu/data/HE/PFS_frame/git/work/wanqqq/run_2605/S26A-queue/proposal_sum.csv"
HIGHLIGHT_STYLE = (
    "background-color: #FCE59F;"
    "font-weight: bold;"
)

# ------------------------
# Functions
# ------------------------
def color_status(status: str) -> str:
    """Return CSS color for status text."""
    return {
        "done": "green",
        "running": "orange",
        "pending": "gray",
    }[status]


def semester_code_for_date(selected_date):
    if selected_date.month == 1:
        return f"S{(selected_date.year - 1) % 100:02d}B"
    if selected_date.month <= 7:
        return f"S{selected_date.year % 100:02d}A"
    return f"S{selected_date.year % 100:02d}B"


def proposal_stat_csv_path(selected_date):
    semester_code = semester_code_for_date(selected_date)
    yymm = selected_date.strftime("%y%m")
    ymd = selected_date.strftime("%Y%m%d")
    return (
        "/home/wanqiu/data/HE/PFS_frame/git/work/wanqqq/"
        f"run_{yymm}/{semester_code}-queue/output_{ymd}/proposal_stat_{yymm}{selected_date.strftime('%d')}.csv"
    )


def load_status() -> pd.DataFrame:
    """Read CSV and normalise types used by the UI.

    - parse date/time columns so we can format them easily
    - add `date_obj` column (date-only) for the DatePicker enabled dates
    - sort by the textual `date` column to keep presentation stable
    """
    df = pd.read_csv(
        CSV_PATH,
        comment="#",
        parse_dates=[
            "time_qaDB",
            "time_queueDB",
            "time_daily_processing",
            "time_validation_complete",
            "last_updated",
        ],
    )
    # keep a date-only column for the DatePicker widget
    df["date_obj"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date")


# Keep dataframe in a mutable container so callbacks can replace it.
df_holder = {"df": load_status()}

# Frequently used derived values
df = df_holder["df"]
available_dates = sorted(df["date_obj"])  # list of date objects
latest_date = available_dates[-1]

# read validation HTML path for a given date
def validation_html_path(selected_date):
    yymm = selected_date.strftime("%y%m")
    ymd = selected_date.strftime("%Y%m%d")

    base_dir = f"/home/wanqiu/data/HE/PFS_frame/git/work/wanqqq/run_{yymm}"

    pattern = (
        f"{base_dir}/*queue/"
        f"output_{ymd}/"
        f"figure_pfsDesign_validation/"
        f"validation_report.html"
    )

    matches = glob.glob(pattern)

    if len(matches) == 0:
        pn.state.notifications.warning(  # type: ignore[union-attr]
            f"No validation file found for {selected_date}.",
            duration=4000,
        )
        return None

    if len(matches) > 1:
        # Optional: warn if multiple matches
        pn.state.notifications.warning(  # type: ignore[union-attr]
            f"Multiple validation reports found for {selected_date}. "
            f"Using the first one.",
            duration=5000,
        )

    return matches[0]

def find_highlighted_cells(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    cell_highlight = set()

    # 1. Find all <style> tags
    for style_tag in soup.find_all("style"):
        if not style_tag.string:
            continue

        css_text = style_tag.string

        # 2. Parse CSS rules: selectors { body }
        for selectors, body in re.findall(r"(.*?)\{(.*?)\}", css_text, re.S):
            # 3. Keep only rules with the target background color
            if re.search(
                rf"background-color\s*:\s*{re.escape('hotpink')}\b",
                body,
            ):
                # 4. Extract row/col indices from selectors
                for row, col in re.findall(r"row(\d+)_col(\d+)", selectors):
                    cell_highlight.add((int(row), int(col)))

    return cell_highlight

# ------------------------
# Widgets
# ------------------------
# DatePicker is initialised with the latest available date and the set
# of enabled dates to prevent selecting dates not present in the CSV.
date_picker = pn.widgets.DatePicker(
    name="Date",
    value=latest_date,
    enabled_dates=available_dates,
    width=200,
)

# Simple refresh button; label is updated with the last update time.
refresh_btn = pn.widgets.Button(
    name="Refresh",
    button_type="primary",
    button_style="outline",
)

confirm_btn = pn.widgets.Button(
    name="Validation complete (SA)",
    button_type="success",
    button_style="outline",
)


# ------------------------
# Actions / Callbacks
# ------------------------
def refresh(event) -> None:
    """Reload CSV, update available dates and refresh the button label.

    This callback is bound to `refresh_btn.on_click` and mutates
    `df_holder["df"]` so the reactive view reads the new data.
    """
    # reload data
    df_holder["df"] = load_status()
    df = df_holder["df"]

    # update widgets that depend on available dates
    available_dates = sorted(df["date_obj"])
    latest_date = available_dates[-1]

    date_picker.enabled_dates = available_dates
    date_picker.latest_date = latest_date  # type: ignore[attr-defined]

    # show last update time in the refresh button label
    valid_last_updated = df["last_updated"].dropna()
    if not valid_last_updated.empty:
        latest_time = valid_last_updated.max().strftime("%H:%M")
        refresh_btn.name = f"Refresh (Last update at {latest_time})"
    else:
        refresh_btn.name = "Refresh"


refresh_btn.on_click(refresh)


def confirm_validation_complete(event) -> None:
    selected_date = date_picker.value
    now = datetime.now()

    df_csv = pd.read_csv(CSV_PATH, comment="#")

    if "time_validation_complete" not in df_csv.columns:
        insert_at = (
            df_csv.columns.get_loc("last_updated")
            if "last_updated" in df_csv.columns
            else len(df_csv.columns)
        )
        df_csv.insert(insert_at, "time_validation_complete", "")
    else:
        df_csv["time_validation_complete"] = df_csv[
            "time_validation_complete"
        ].astype("object")

    if "last_updated" in df_csv.columns:
        df_csv["last_updated"] = df_csv["last_updated"].astype("object")

    mask = pd.to_datetime(df_csv["date"]).dt.date == selected_date
    if not mask.any():
        pn.state.notifications.warning(  # type: ignore[union-attr]
            f"No information found for {selected_date}.",
            duration=4000,
        )
        return

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    df_csv.loc[mask, "time_validation_complete"] = timestamp
    df_csv.loc[mask, "last_updated"] = timestamp
    df_csv.to_csv(CSV_PATH, index=False)

    refresh(None)
    refresh_btn.name = f"Refresh (Last update at {now.strftime('%H:%M')})"
    refresh_btn.param.trigger("clicks")

    pn.state.notifications.success(  # type: ignore[union-attr]
        f"Validation complete recorded for {selected_date} at {now.strftime('%H:%M:%S')}.",
        duration=5000,
    )


confirm_btn.on_click(confirm_validation_complete)


# ------------------------
# Reactive view
# ------------------------
@pn.depends(date_picker, refresh_btn)  # type: ignore[call-arg]
def status_view(selected_date, _):
    """Return an HTML pane summarising statuses for the selected date.

    The second dependency (refresh_btn) is used as a cheap invalidation
    trigger so the view updates after a refresh.
    """
    df = df_holder["df"]
    # select the single row for the chosen date
    row = df[df["date_obj"] == selected_date].iloc[0]

    def show_time(time):
        return f" ({time.strftime('%H:%M')})" if pd.notna(time) else ""

    # Build HTML parts; colour the status text for quick scanning.
    validation_status = (
        "done" if pd.notna(row["time_validation_complete"]) else "pending"
    )
    parts = [
        f"<b>qaDB: <span style='color:{color_status(row['status_qaDB'])}'>{row['status_qaDB']}</b>{show_time(row['time_qaDB'])}</span>",
        f"<b>queueDB: <span style='color:{color_status(row['status_queueDB'])}'>{row['status_queueDB']}</b>{show_time(row['time_queueDB'])}</span>",
        f"<b>designGenerator: <span style='color:{color_status(row['status_daily_processing'])}'>{row['status_daily_processing']}</b>{show_time(row['time_daily_processing'])}</span>",
        f"<b>Validation (SA): <span style='color:{color_status(validation_status)}'>{validation_status}</b>{show_time(row['time_validation_complete'])}</span>",
    ]

    text = " | ".join(parts)

    return pn.pane.HTML(
        f"<div style='font-size:20px; font-weight:normal;'>{text}</div>"
    )

@pn.depends(date_picker, refresh_btn)  # type: ignore[call-arg]
def validation_view(selected_date, _):
    html_path = validation_html_path(selected_date)

    if html_path is None:
        return None

    df = pd.read_html(html_path, index_col=0)[0]
    df = df.map(
        lambda x: "{:.2f}".format(x) if isinstance(x, float) else x
    ) # re-format floats to 2 decimal places

    validation_dir = Path(html_path).parent

    def on_row_select(event):
        if not event.new:
            return

        row = event.new[0]
        design_id = df.iloc[row]["designId"]
        pdf_path = validation_dir / f"check_{design_id}.pdf"

        if pdf_path.exists():
            pdf_pane.object = str(pdf_path)
            tabs.active = 1   # switch to "Validation Figure" tab
        else:
            pdf_pane.object = None

    cell_highlight = find_highlighted_cells(html_path) # find highlighted cells


    def styler_from_cell_highlight(df, cell_highlight):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        for (row, col) in cell_highlight:
            if row < len(df.index) and col < len(df.columns):
                styles.iat[row, col] = HIGHLIGHT_STYLE

        return df.style.apply(lambda _: styles, axis=None)
    
    styler = styler_from_cell_highlight(df, cell_highlight)


    # Create Tabulator without passing `columns` (some Panel builds
    # reject columns at init). Use `apply` to set Tabulator options,
    # including per-column HTML formatter and disabling escaping.
    tab = pn.widgets.Tabulator(
        value=styler,
        layout="fit_data_table",
        pagination="remote",
        page_size=25,
        sizing_mode="stretch_width",
        theme="bulma",
        theme_classes=["table-striped"],
        header_filters=True,
        show_index=False,
        disabled=True,
        #selection=list(rows_flagged),
        #selectable="checkbox",
    )
    tab.param.watch(on_row_select, "selection")

    return tab


@pn.depends(date_picker, refresh_btn)  # type: ignore[call-arg]
def observation_progress_view(selected_date, _):
    start_date = selected_date if selected_date is not None else datetime.now().date()

    semester_code = semester_code_for_date(start_date)

    program_id = f"{semester_code}-999QN"
    url = (
        f"https://www1.subaru.nao.ac.jp/operation/opecenter/"
        f"ObsProgram{semester_code}.html"
    )

    try:
        with urlopen(url) as response:
            html = response.read()
    except URLError as exc:
        return pn.pane.HTML(
            (
                "<div style='font-size:18px;'>"
                "Observation Progress<br>"
                f"<span style='color:#b00020;'>Failed to load {url}: {exc}</span>"
                "</div>"
            ),
            sizing_mode="stretch_width",
        )

    soup = BeautifulSoup(html, "html.parser")
    semester_year = 2000 + int(semester_code[1:3])
    remaining_nights = 0.0
    remaining_hours = 0.0
    remaining_segments = []
    gB_nppc_req = {}
    gB_nppc = {}
    gB_nppc_tonight = {}
    gB_fh_alloc = {}
    gB_fh_achieved = {}
    gB_fh_expected = {}
    gB_fh_executed = {}
    gB_n_target_completed = {}
    gB_n_target_partial = {}
    gB_fh_needed_partial = {}
    gB_priority_total = {}
    gB_priority_observed = {}

    try:
        with open(PROPOSAL_SUM_CSV_PATH, "r", encoding="utf-8") as f:
            sheet_rows = csv.DictReader(f)
            for sheet_row in sheet_rows:
                if (sheet_row.get("grade") or "").strip() != "B":
                    continue

                proposal_id = (sheet_row.get("proposal_id") or "").strip()
                if not proposal_id.startswith(semester_code):
                    continue

                comment = (sheet_row.get("comment") or "").strip()
                match = re.search(r"nppc\s*=\s*(\d+)", comment)
                if match:
                    gB_nppc_req[proposal_id] = int(match.group(1))
                    gB_nppc[proposal_id] = 0
                    gB_nppc_tonight[proposal_id] = 0
                    gB_fh_alloc[proposal_id] = 0.0
                    gB_fh_achieved[proposal_id] = 0.0
                    gB_fh_expected[proposal_id] = 0.0
                    gB_fh_executed[proposal_id] = 0.0
                    gB_n_target_completed[proposal_id] = 0
                    gB_n_target_partial[proposal_id] = 0
                    gB_fh_needed_partial[proposal_id] = 0.0
                    gB_priority_total[proposal_id] = [0] * 10
                    gB_priority_observed[proposal_id] = [0] * 10
    except Exception:
        pn.state.notifications.warning(  # type: ignore[union-attr]
            f"Could not read proposal summary file: {PROPOSAL_SUM_CSV_PATH}",
            duration=5000,
        )
        gB_nppc_req = {}
        gB_nppc = {}

    if gB_nppc_req:
        proposal_stat_path = proposal_stat_csv_path(start_date)

        try:
            df_stat = pd.read_csv(proposal_stat_path)
            df_stat = df_stat.set_index("proposal_id")
            for proposal_id in gB_nppc_req:
                if proposal_id not in df_stat.index:
                    continue

                row = df_stat.loc[proposal_id]
                gB_nppc[proposal_id] = int(pd.to_numeric(row.get("nppc_observed", 0), errors="coerce"))
                gB_nppc_tonight[proposal_id] = int(pd.to_numeric(row.get("nppc_tonight", 0), errors="coerce"))

                gB_fh_alloc[proposal_id] = float(pd.to_numeric(row.get("fh_allocated", 0.0), errors="coerce"))
                gB_fh_achieved[proposal_id] = float(pd.to_numeric(row.get("fh_achieved", 0.0), errors="coerce"))
                gB_fh_expected[proposal_id] = float(pd.to_numeric(row.get("fh_expected_after_tonight", 0.0), errors="coerce"))
                gB_fh_executed[proposal_id] = float(pd.to_numeric(row.get("fh_executed", 0.0), errors="coerce"))

                gB_n_target_completed[proposal_id] = int(pd.to_numeric(row.get("n_target_completed", 0), errors="coerce"))
                gB_n_target_partial[proposal_id] = int(pd.to_numeric(row.get("n_target_partial", 0), errors="coerce"))
                gB_fh_needed_partial[proposal_id] = float(pd.to_numeric(row.get("fh_needed_to_complete_partial_now", 0.0), errors="coerce"))

                try:
                    gB_priority_total[proposal_id] = [
                        int(x) for x in ast.literal_eval(
                            str(row.get("priority_total_P0toP9", "[0,0,0,0,0,0,0,0,0,0]"))
                        )
                    ]
                except Exception:
                    gB_priority_total[proposal_id] = [0] * 10
                try:
                    gB_priority_observed[proposal_id] = [
                        int(x) for x in ast.literal_eval(
                            str(row.get("priority_observed_P0toP9", "[0,0,0,0,0,0,0,0,0,0]"))
                        )
                    ]
                except Exception:
                    gB_priority_observed[proposal_id] = [0] * 10
        except Exception:
            pn.state.notifications.warning(  # type: ignore[union-attr]
                f"Could not read proposal stat file: {proposal_stat_path}",
                duration=5000,
            )

    for row in soup.find_all("tr"):
        cells = [" ".join(cell.stripped_strings) for cell in row.find_all(["td", "th"])]
        if not cells:
            continue

        date_text = next(
            (cell for cell in cells if re.match(r"^\d{1,2}/\d{1,2}$", cell)),
            None,
        )
        if date_text is None:
            continue

        month, day = map(int, date_text.split("/"))
        row_year = semester_year + 1 if semester_code.endswith("B") and month == 1 else semester_year
        row_date = datetime(row_year, month, day).date()
        if row_date < start_date:
            continue

        row_text = " ".join(cells)

        for time_range, fraction_text, segment_text in re.findall(
            r"(\d{1,2}:\d{2}-\d{1,2}:\d{2})\s*\{([0-9.]+)\}\s*(.*?)(?=\d{1,2}:\d{2}-\d{1,2}:\d{2}\s*\{|$)",
            row_text,
        ):
            if "[Observation Canceled]" in segment_text:
                continue

            fraction = float(fraction_text)
            start_text, end_text = time_range.split("-")
            start_hour, start_minute = map(int, start_text.split(":"))
            end_hour, end_minute = map(int, end_text.split(":"))
            start_total_minutes = start_hour * 60 + start_minute
            end_total_minutes = end_hour * 60 + end_minute
            if end_total_minutes < start_total_minutes:
                end_total_minutes += 24 * 60

            duration_hours = (end_total_minutes - start_total_minutes) / 60.0

            if f"(Arai; {program_id})" in segment_text:
                remaining_hours += duration_hours
                remaining_nights += fraction
                remaining_segments.append((row_date, time_range, fraction))

    remaining_pointings = round(remaining_hours * 3)
    description = "&#10;".join(
        f"{date.strftime('%Y-%m-%d')}, {time_range}, {fraction:.2f} night"
        for date, time_range, fraction in remaining_segments
    )
    gB_remaining_pointings = sum(
        gB_nppc_req[proposal_id] - gB_nppc.get(proposal_id, 0)
        for proposal_id in gB_nppc_req
    )
    gB_remaining_alert_style = (
        "color:#b91c1c; font-weight:bold;"
        if gB_remaining_pointings > remaining_pointings * 0.8
        else ""
    )
    gB_table = ""
    gB_fh_table = ""
    gB_details = ""
    if gB_nppc_req:
        proposal_cells = "".join(
            f"<td style='padding:4px 8px; border:1px solid #ddd;'><b>{proposal_id}</b></td>"
            for proposal_id in gB_nppc_req
        )
        pointing_cells = "".join(
            (
                f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:center;'>{obs} ({(100.0 * obs / req):.0f}%)</td>"
                if req > 0
                else f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:center;'>{obs} (0%)</td>"
            )
            for proposal_id in gB_nppc_req
            for obs, req in [(gB_nppc.get(proposal_id, 0), gB_nppc_req[proposal_id])]
        )
        requested_cells = "".join(
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:center;'>{gB_nppc_req[proposal_id]}</td>"
            for proposal_id in gB_nppc_req
        )
        tonight_cells = "".join(
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:center; color:#b45309; background:#fff7ed; font-weight:bold;'>+{gB_nppc_tonight.get(proposal_id, 0)}</td>"
            for proposal_id in gB_nppc_req
        )
        gB_table = (
            "<table style='margin-top:10px; border-collapse:collapse; font-size:14px;'>"
            "<tr><th style='padding:4px 8px; border:1px solid #ddd; background:#f7f7f7;'>Proposal ID</th>"
            f"{proposal_cells}</tr>"
            "<tr><th style='padding:4px 8px; border:1px solid #ddd; background:#f7f7f7;'>Nppc (observed)</th>"
            f"{pointing_cells}</tr>"
            "<tr><th style='padding:4px 8px; border:1px solid #ddd; background:#ffedd5; color:#9a3412;'>Nppc (tonight)</th>"
            f"{tonight_cells}</tr>"
            "<tr><th style='padding:4px 8px; border:1px solid #ddd; background:#f7f7f7;'>Nppc (uploader)</th>"
            f"{requested_cells}</tr>"
            "</table>"
        )

        fh_rows = []
        for proposal_id in gB_nppc_req:
            fh_alloc = gB_fh_alloc.get(proposal_id, 0.0)
            fh_ach = gB_fh_achieved.get(proposal_id, 0.0)
            fh_exp = gB_fh_expected.get(proposal_id, 0.0)
            fh_exe = gB_fh_executed.get(proposal_id, 0.0)
            # Per-proposal normalization: allocated FH defines bar length.
            scale = max(1.0, fh_alloc)
            ach_w = min(100.0, 100.0 * fh_ach / scale)
            exp_w = min(100.0, 100.0 * fh_exp / scale)
            alloc_w = min(100.0, 100.0 * fh_alloc / scale)
            exe_x = 100.0 * fh_exe / scale
            ach_frac = (100.0 * fh_ach / fh_alloc) if fh_alloc > 0 else 0.0
            ach_frac_text = f"{ach_frac:.0f}%"
            ach_frac_color = "#15803d" if ach_frac >= 90.0 else "#333"
            fh_rows.append(
                "<div style='margin:8px 0 12px 0;'>"
                f"<div style='font-size:13px; margin-bottom:4px;'><b>{proposal_id}</b> | "
                f"<span style='color:#6b7280;'>Alloc</span> <span style='color:#374151; font-weight:bold;'>{fh_alloc:.1f}</span> | "
                f"<span style='color:#1d4ed8;'>Ach</span> <span style='color:#1d4ed8; font-weight:bold;'>{fh_ach:.1f}</span> | "
                f"<span style='color:#15803d;'>Exp</span> <span style='color:#15803d; font-weight:bold;'>{fh_exp:.1f}</span> | "
                f"<span style='color:#c2410c;'>Exe</span> <span style='color:#c2410c; font-weight:bold;'>{fh_exe:.1f}</span></div>"
                "<div style='display:flex; align-items:center; gap:8px; max-width:760px;'>"
                "<div style='position:relative; height:14px; border:1px solid #ccc; background:#f5f5f5; border-radius:4px; width:620px;'>"
                f"<div style='position:absolute; left:0; top:0; height:100%; width:{alloc_w:.1f}%; background:#d9d9d9; border-radius:4px;'></div>"
                f"<div style='position:absolute; left:0; top:0; height:100%; width:{exp_w:.1f}%; background:#a5d6a7; border-radius:4px; opacity:0.9;'></div>"
                f"<div style='position:absolute; left:0; top:0; height:100%; width:{ach_w:.1f}%; background:#64b5f6; border-radius:4px; opacity:0.95;'></div>"
                f"<div style='position:absolute; left:{exe_x:.1f}%; top:-2px; height:18px; border-left:3px solid #ff9800;'></div>"
                "</div>"
                f"<div style='min-width:52px; font-size:12px; color:{ach_frac_color}; font-weight:bold; text-align:right;'>{ach_frac_text}</div>"
                "</div>"
                "</div>"
            )
        gB_fh_table = (
            "<div style='margin-top:12px; padding:10px; border:1px solid #ddd; border-radius:6px; background:#fbfbfb;'>"
            "<div style='font-weight:bold; margin-bottom:6px;'>Current Observation pregressing for Grade B</div>"
            + "".join(fh_rows)
            + "</div>"
        )

        detail_lines = []
        for proposal_id in gB_nppc_req:
            p_total = gB_priority_total.get(proposal_id, [0] * 10)
            p_obs = gB_priority_observed.get(proposal_id, [0] * 10)
            highest_p = next((idx for idx, n in enumerate(p_total) if n > 0), None)
            if highest_p is None:
                p_label = "N_P-"
                p_obs_text = "0"
                p_tot_text = "0"
            else:
                p_label = f"N_P{highest_p}"
                p_obs_text = str(p_obs[highest_p])
                p_tot_text = str(p_total[highest_p])

            fh_remaining = gB_fh_alloc.get(proposal_id, 0.0) - gB_fh_achieved.get(proposal_id, 0.0)
            fh_needed = gB_fh_needed_partial.get(proposal_id, 0.0)
            alert_style = "color:#b91c1c; font-weight:bold;" if fh_needed > fh_remaining * 0.8 else ""
            detail_lines.append(
                f"<li><b>{proposal_id}</b>: "
                f"N_complete = <b>{gB_n_target_completed.get(proposal_id, 0)}</b>, "
                f"N_partial = <b>{gB_n_target_partial.get(proposal_id, 0)}</b> "
                f"(<b style='{alert_style}'>{fh_needed:.1f}</b> FHs further needed, "
                f"<b style='{alert_style}'>{fh_remaining:.1f}</b> FHs remaining), "
                f"{p_label} = <b>{p_obs_text}</b>/<b>{p_tot_text}</b>"
                "</li>"
            )
        gB_details = (
            "<div style='margin-top:10px; padding:10px; border:1px solid #ddd; border-radius:6px; background:#fff;'>"
            "<div style='font-weight:bold; margin-bottom:6px;'>Target-base progressing</div>"
            "<ul style='margin:0 0 0 18px; font-size:14px; line-height:1.6;'>"
            + "".join(detail_lines)
            + "</ul></div>"
        )

    return pn.pane.HTML(
        (
            "<div style='font-size:18px;'>"
            f"<span title='{description}'>"
            f"There are <b>{remaining_nights:.2f}</b> nights = "
            f"<b>{remaining_pointings}</b> pointings remaining ({semester_code})."
            " <span style='color:#6A5AA3; font-weight:bold; cursor:help;'>?</span>"
            "</span>"
            "<br>"
            f"<span style='{gB_remaining_alert_style}'><b>{gB_remaining_pointings}</b> pointings still needed to complete grade B programs (assuming good weather conditions and uploader simulation results).</span>"
            f"{gB_table}"
            f"{gB_fh_table}"
            f"{gB_details}"
            "</div>"
        ),
        sizing_mode="stretch_width",
    )


@pn.depends(date_picker, refresh_btn)  # type: ignore[call-arg]
def visibility_view(selected_date, _):
    html_path = validation_html_path(selected_date)
    if html_path is None:
        return pn.pane.HTML(
            "<div style='font-size:18px; color:#666;'>Visibility</div>",
            sizing_mode="stretch_width",
        )

    figure_path = Path(html_path).with_name("visibility_plot.png")
    if not figure_path.exists():
        return pn.pane.HTML(
            "<div style='font-size:18px; color:#666;'>Visibility figure not found.</div>",
            sizing_mode="stretch_width",
        )

    return pn.pane.PNG(figure_path.read_bytes(), sizing_mode="stretch_width")

# ------------------------
# Layout
# ------------------------
pdf_pane = pn.pane.PDF(
    None,
    sizing_mode="stretch_width",
    height=800,
)
tabs = pn.Tabs(
    ("Validation Table", validation_view),
    ("Design Figure", pdf_pane),
    ("Observation Progress", observation_progress_view),
    ("Visibility", visibility_view),
)

template = pn.template.BootstrapTemplate(
    title="Validation of PFS Queue Planning",
    sidebar=[
        refresh_btn,
        confirm_btn,
        date_picker,
    ],
    sidebar_width=290,
    theme="default",
    header_background="#6A5AA3",
)

template.main.append(status_view)
template.main.append(tabs)

template.servable()