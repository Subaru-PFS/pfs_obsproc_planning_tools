import asyncio
import copy
import html
import json
import re
import shlex
import tomllib
from datetime import datetime, timedelta
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import panel as pn


pn.extension(
    "tabulator",
    notifications=True,
    raw_css=[
        ".config-help-cursor {"
        "cursor: url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 28 28'%3E%3Ccircle cx='11' cy='11' r='8.5' fill='white' stroke='%23333' stroke-width='1.8'/%3E%3Ctext x='11' y='15' text-anchor='middle' font-family='Arial, sans-serif' font-size='14' font-weight='700' fill='%23333'%3E%3F%3C/text%3E%3C/svg%3E\") 11 11, help !important;"
        "}"
    ],
)

SEMESTER_OPTIONS = ["S26A", "S26B"]
DEFAULT_SEMESTER = "S26B"
EXCLUDED_PROPOSAL_IDS = {"S26B-OT02", "S26B-EN16"}
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "examples" / "config_example.toml"
)
PROGRAM_SHEET_LINKS_PATH = (
    Path.home()
    / "config"
    / "pfs_obsproc_planning"
    / "program_sheet_links.json"
)
TARGET_INSERT_DIR = (
    Path.home() / "config" / "pfs_obsproc_planning" / "targetdb_insert"
)
TARGET_INSERT_STATUS_PATH = TARGET_INSERT_DIR / "status.json"
TARGETDB_DEV_CONFIG = Path(
    "/work/monodera/Subaru-PFS/database_configs/config_targetdb_dev.toml"
)
TARGETDB_PRODUCTION_CONFIG = Path(
    "/work/monodera/Subaru-PFS/database_configs/config_targetdb_production.toml"
)

SECTION_TITLE_STYLE = (
    "font-size: 15px; font-weight: 700; margin: 0 0 8px 0;"
)
SEMESTER_TITLE_STYLE = (
    "font-size: 18px; font-weight: 700; color: #124a78; "
    "background: #e8f3fc; border-left: 5px solid #1976d2; "
    "padding: 10px 12px; margin: 0 0 10px 0;"
)
SUBTITLE_STYLE = "font-size: 16px; font-weight: 700; margin: 0;"
CONFIG_DESCRIPTIONS = {
    "ppp": {
        "mode": "Execution or planning mode for PPP. In this UI it switches queue vs classic behavior.",
        "proposalIds": "Proposal IDs used for planning or target selection.",
        "proposalIds_backup": "Backup proposal IDs used for daily-plan gap filling.",
        "localPath": "Local data path used when inputs are read from files instead of a database.",
        "localPath_tgt": "Local path to target input data when not reading targets from the database.",
        "localPath_ppc": "Local path to PPC-related input data when using local files.",
        "TEXP_NOMINAL": "Nominal exposure time used for planning or scaling calculations.",
        "sql_query": "SQL query used to fetch targets or proposals from the database.",
        "inputDir": "Directory containing PPP input files.",
        "outputDir": "Output directory for PPP products.",
        "reserveFibers": "Whether to reserve fibers for calibration targets such as sky and flux standards.",
        "fiberNonAllocationCost": "Penalty term for leaving fibers unallocated.",
        "visibility_check": "Whether PPP should enforce observation-window visibility checks.",
        "daily_plan": "Enable daily-plan mode, including backup filling of schedule gaps.",
    },
    "qplan": {
        "outputDir": "Output directory for quick-planning results.",
        "overhead": "Per-observation overhead used by the planner, typically in minutes.",
        "obs_dates": "Observation dates in HST for which planning is performed.",
        "start_time": "Start times of the available observation windows in HST.",
        "stop_time": "Stop times of the available observation windows in HST.",
    },
    "qplan.weight": {
        "slew": "Weight for slew or telescope movement cost.",
        "delay": "Weight for delay or overhead time.",
        "filter": "Penalty weight for filter changes.",
        "rank": "Weight for target rank score.",
        "priority": "Weight for proposal or target priority.",
        "w_slew": "Weight for slew or telescope movement cost.",
        "w_delay": "Weight for delay or overhead time.",
        "w_filterchange": "Penalty weight for filter changes.",
        "w_rank": "Weight for target rank score.",
        "w_priority": "Weight for proposal or target priority.",
    },
    "targetdb.sky": {
        "version": "Sky catalog version or versions to use.",
    },
    "targetdb.fluxstd": {
        "version": "Flux-standard catalog version or versions to use.",
    },
    "queuedb": {
        "filepath": "Path to the queue database or queue template file used by PPP.",
    },
    "schemacrawler": {
        "SCHEMACRAWLERDIR": "Path to the SchemaCrawler distribution used for schema inspection or documentation.",
    },
    "netflow": {
        "use_gurobi": "Enable Gurobi in the network-flow solver step.",
        "two_stage": "Whether to use a two-stage optimization strategy.",
        "cobra_location_group_n": "Number of cobra location groups used in optimization.",
        "min_sky_targets_per_location": "Minimum number of sky targets required per location group.",
        "location_group_penalty": "Penalty for violating location-group related constraints.",
        "cobra_safety_margin": "Safety margin applied to cobra geometry constraints.",
    },
    "gurobi": {
        "seed": "Random seed for reproducibility.",
        "presolve": "Gurobi presolve level.",
        "method": "Gurobi algorithm or method selector.",
        "degenmoves": "Degeneracy handling strategy.",
        "heuristics": "Heuristic effort level.",
        "mipfocus": "MIP focus mode.",
        "mipgap": "Relative MIP gap tolerance.",
        "PreSOS2Encoding": "Preprocessing toggle for SOS2 encoding.",
        "PreSOS1Encoding": "Preprocessing toggle for SOS1 encoding.",
        "threads": "Maximum number of Gurobi threads to use.",
    },
    "sfa": {
        "n_sky": "Number of sky fibers or sky targets to sample.",
        "sky_random": "Whether to randomize sky target selection.",
        "n_sky_random": "Candidate pool size when random sky selection is enabled.",
        "reduce_sky_targets": "Whether to down-select sky targets to the configured target count.",
        "pfs_instdata_dir": "Instrument data directory.",
        "cobra_coach_dir": "CobraCoach data directory.",
        "cobra_coach_module_version": "Specific CobraCoach module version; use None for the default behavior.",
        "sm": "Spectrograph modules to include.",
        "dot_margin": "Margin used for dot or collision checks in the instrument geometry.",
        "dot_penalty": "Penalty profile for dot or collision handling.",
        "arms": "Enabled spectrograph arms, typically a combination of b, r, and n.",
        "guidestar_mag_min": "Bright magnitude limit for guide stars.",
        "guidestar_mag_max": "Faint magnitude limit for guide stars.",
        "guidestar_neighbor_mag_min": "Neighbor magnitude threshold used to avoid guide-star contamination.",
        "guidestar_minsep_deg": "Minimum angular separation from neighboring sources, in degrees.",
        "n_fluxstd": "Number of flux-standard stars to select.",
        "fluxstd_mag_max": "Faint magnitude limit for flux standards.",
        "fluxstd_mag_min": "Bright magnitude limit for flux standards.",
        "fluxstd_mag_filter": "Band or filter used for flux-standard magnitude selection.",
        "good_fluxstd": "Whether to require a high-quality flag for flux standards.",
        "fluxstd_min_prob_f_star": "Minimum stellar probability for flux standards.",
        "fluxstd_min_teff": "Minimum effective temperature allowed for flux standards.",
        "fluxstd_max_teff": "Maximum effective temperature allowed for flux standards.",
        "fluxstd_flags_dist": "Whether to exclude flux standards using distance-related flags.",
        "fluxstd_flags_ebv": "Whether to exclude flux standards using extinction flags.",
        "filler": "Enable filler-target allocation.",
        "proposalIds_obsFiller": "Proposal IDs used for filler observations.",
        "filler_mag_min": "Bright magnitude limit for filler targets.",
        "filler_mag_max": "Faint magnitude limit for filler targets.",
        "filler_random": "Whether to randomize filler target selection.",
        "n_fillers_random": "Candidate pool size when random filler selection is enabled.",
        "reduce_fillers": "Whether to down-select filler candidates.",
        "multiprocessing": "Enable multiprocessing in the SFA step.",
        "dup_obs_filler_remove": "Remove filler targets already observed in duplicate observations.",
        "obs_filler_done_remove": "Remove filler targets already completed in previous observations.",
        "fill_unassign_radius_check": "Radius used when checking for unassigned filler targets.",
        "fill_unassign": "Whether to fill remaining unassigned fibers.",
        "fill_unassign_radius": "Search radius for filling unassigned fibers.",
        "fill_unassign_gaia_mag": "Gaia magnitude limit used while filling unassigned fibers.",
        "fill_unassign_pslId": "Proposal or sample list ID used for unassigned-fiber filling.",
        "raster": "Enable raster pattern generation.",
        "raster_mag_min": "Bright magnitude limit for raster selection.",
        "raster_mag_max": "Faint magnitude limit for raster selection.",
    },
    "packages": {
        "config_path": "Path to an additional TOML file that is merged into the main config.",
        "check_version": "Whether dependent package versions should be checked before running.",
        "pfs_datamodel_ver": "Desired version of pfs_datamodel.",
        "pfs_utils_ver": "Desired version of pfs_utils.",
        "ics_cobraCharmer_ver": "Desired version of ics_cobraCharmer.",
        "ics_cobraOps_ver": "Desired version of ics_cobraOps.",
        "ets_fiberalloc_ver": "Desired version of ets_fiberalloc.",
        "pfs_instdata_ver": "Desired version of pfs_instdata.",
        "ets_pointing_ver": "Desired version of ets_pointing.",
        "pfs_obsproc_planning_ver": "Desired version of pfs_obsproc_planning_tools.",
        "ets_shuffle_ver": "Desired version of ets_shuffle.",
        "ets_target_database_ver": "Desired version of ets_target_database.",
        "ics_fpsActor_ver": "Desired version of ics_fpsActor.",
        "spt_operational_database_ver": "Desired version of spt_operational_database.",
        "qplan_ver": "Desired version of qplan.",
        "pfs_instdata_dir": "Local repository path for pfs_instdata.",
        "ets_pointing_dir": "Local repository path for ets_pointing.",
        "ets_shuffle_dir": "Local repository path for ets_shuffle.",
        "pfs_utils_dir": "Local repository path for pfs_utils.",
        "pfs_datamodel_dir": "Local repository path for pfs_datamodel.",
        "ics_cobraCharmer_dir": "Local repository path for ics_cobraCharmer.",
        "ics_cobraOps_dir": "Local repository path for ics_cobraOps.",
        "ets_fiberalloc_dir": "Local repository path for ets_fiberalloc.",
        "ets_target_database_dir": "Local repository path for ets_target_database.",
        "ics_fpsActor_dir": "Local repository path for ics_fpsActor.",
        "spt_operational_database_dir": "Local repository path for spt_operational_database.",
        "qplan_dir": "Local repository path for qplan.",
    },
    "ope": {
        "template": "OPE template file to populate.",
        "outfilePath": "Output directory for generated OPE files.",
        "designPath": "Output directory for design artifacts.",
        "runName": "Run name or prefix included in outputs.",
        "n_split_frame": "Number of sub-exposures or frame splits per observation.",
    },
    "validation": {
        "savefig": "Whether validation plots should be saved.",
        "showfig": "Whether validation plots should be displayed interactively.",
        "save_unassign_toobright": "Whether to save diagnostics for unassigned targets that are too bright.",
        "planet_minsep_deg": "Minimum separation (deg) from naked-eye planets before a pointing is flagged (default 3.0).",
    },
    "ssp": {
        "ssp": "Whether the run is for SSP mode.",
    },
}


def format_value(value):
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def format_toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped_value = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )
        return f'"{escaped_value}"'
    if isinstance(value, list):
        return "[" + ", ".join(format_toml_value(item) for item in value) + "]"
    return str(value)


def dumps_toml(data):
    lines = []

    def write_section(section_data, prefix=None):
        scalar_items = []
        nested_items = []

        for key, value in section_data.items():
            if isinstance(value, dict):
                nested_items.append((key, value))
            else:
                scalar_items.append((key, value))

        if prefix is not None:
            lines.append(f"[{prefix}]")

        for key, value in scalar_items:
            lines.append(f"{key} = {format_toml_value(value)}")

        if prefix is not None and (scalar_items or nested_items):
            lines.append("")

        for key, value in nested_items:
            nested_prefix = f"{prefix}.{key}" if prefix else key
            write_section(value, nested_prefix)

    write_section(data)
    return "\n".join(lines).rstrip() + "\n"


def get_parameter_description(section_name, parameter_name):
    return CONFIG_DESCRIPTIONS.get(section_name, {}).get(parameter_name, "")


def format_parameter_cell(section_name, parameter_name):
    description = get_parameter_description(section_name, parameter_name)
    parameter_text = html.escape(str(parameter_name))
    if not description:
        return f"<div style='font-weight: 600;'>{parameter_text}</div>"

    description_text = html.escape(description)
    return (
        f"<div title='{description_text}' "
        "class='config-help-cursor' "
        "style='font-weight: 600; display: inline-block; user-select: none; text-decoration: underline; text-decoration-style: dotted;'>"
        f"{parameter_text}</div>"
    )


def flatten_sections(data, prefix=""):
    sections = []
    rows = []

    for key, value in data.items():
        if isinstance(value, dict):
            child_prefix = f"{prefix}.{key}" if prefix else key
            sections.extend(flatten_sections(value, child_prefix))
        else:
            rows.append(
                {
                    "parameter": format_parameter_cell(prefix, key),
                    "value": format_value(value),
                }
            )

    if rows:
        section_name = prefix if prefix else "root"
        sections.insert(0, (section_name, pd.DataFrame(rows)))

    return sections
def flatten_column_name(column_name):
    if isinstance(column_name, tuple):
        return " ".join(str(part) for part in column_name if str(part) != "nan").strip()
    return str(column_name)


class HTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self._current_table = None
        self._current_row = None
        self._current_cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
        elif tag == "br" and self._current_cell is not None:
            self._current_cell.append(" ")

    def handle_data(self, data):
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            cell_text = " ".join("".join(self._current_cell).split())
            self._current_row.append(cell_text)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None and self._current_table is not None:
            if any(cell for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None


def parse_html_tables(html_text):
    parser = HTMLTableParser()
    parser.feed(html_text)
    return parser.tables


def parse_schedule_datetime(date_text, time_text, semester_code=DEFAULT_SEMESTER):
    month_text, day_text = str(date_text).split("/")
    hour_text, minute_text = time_text.split(":")
    year = 2000 + int(semester_code[1:3])
    if semester_code.endswith("B") and int(month_text) == 1:
        year += 1
    return datetime(
        year,
        int(month_text),
        int(day_text),
        int(hour_text),
        int(minute_text),
    )


def build_schedule_datetimes(
    date_text, start_time_text, end_time_text, semester_code=DEFAULT_SEMESTER
):
    start_datetime = parse_schedule_datetime(date_text, start_time_text, semester_code)
    end_datetime = parse_schedule_datetime(date_text, end_time_text, semester_code)
    return start_datetime, end_datetime


def build_actual_qplan_datetime(obs_date, time_text):
    actual_datetime = datetime.strptime(
        f"{obs_date.strftime('%Y-%m-%d')} {time_text}", "%Y-%m-%d %H:%M"
    )
    if actual_datetime.hour < 12:
        actual_datetime += timedelta(days=1)
    return actual_datetime.strftime("%Y-%m-%d %H:%M:%S")


def parse_observation_text(value):
    entries = []
    if not value.strip():
        return entries

    pattern = re.compile(
        r"^\s*(?P<date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<start>\d{2}:\d{2})-(?P<end>\d{2}:\d{2})\s*$"
    )

    for line in value.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue

        match = pattern.fullmatch(stripped_line)
        if match is None:
            return None

        obs_date = datetime.strptime(match.group("date"), "%Y-%m-%d")

        entries.append(
            {
                "date": match.group("date"),
                "start": build_actual_qplan_datetime(obs_date, match.group("start")),
                "stop": build_actual_qplan_datetime(obs_date, match.group("end")),
            }
        )

    return entries


def extract_pfs_schedule_entries(schedule_text):
    entry_pattern = re.compile(
        r"(?P<start>\d{1,2}:\d{2})-(?P<end>\d{1,2}:\d{2})\s*\{[^}]*\}\s*"
        r"PFS\b[^()]*\([^;()]*;\s*(?P<proposal>S\d{2}[AB]-[A-Za-z0-9-]+)\)",
        re.IGNORECASE,
    )
    return entry_pattern.finditer(schedule_text)


def fetch_pfs_schedule_data(semester_code=DEFAULT_SEMESTER):
    semester_url = (
        "https://www1.subaru.nao.ac.jp/operation/opecenter/"
        f"ObsProgram{semester_code}.html"
    )
    queue_only_id = f"{semester_code}-999QN"
    request = Request(semester_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        html_text = response.read().decode("utf-8", errors="ignore")

    schedule_table = None
    date_column = None
    schedule_column = None

    for table in parse_html_tables(html_text):
        for row_index, row in enumerate(table):
            if "Date" in row and "Night Observation Schedule" in row:
                date_column = row.index("Date")
                schedule_column = row.index("Night Observation Schedule") + 1
                schedule_table = table[row_index + 1 :]
                break
        if schedule_table is not None:
            break

    if schedule_table is None:
        raise ValueError("Could not find the Subaru schedule table.")

    schedule_by_proposal = {}

    for row in schedule_table:
        if len(row) <= max(date_column, schedule_column):
            continue

        date_text = str(row[date_column]).strip()
        schedule_text = str(row[schedule_column]).strip()

        if not re.fullmatch(r"\d{1,2}/\d{1,2}", date_text):
            continue

        for match in extract_pfs_schedule_entries(schedule_text):
            proposal_id = match.group("proposal").upper()
            if proposal_id in EXCLUDED_PROPOSAL_IDS:
                continue

            start_datetime, end_datetime = build_schedule_datetimes(
                date_text,
                match.group("start"),
                match.group("end"),
                semester_code,
            )

            schedule_by_proposal.setdefault(proposal_id, []).append(
                {
                    "date": date_text,
                    "start": start_datetime,
                    "end": end_datetime,
                    "time_range": f"{match.group('start')}-{match.group('end')}",
                }
            )

    classic_ids = sorted(
        proposal_id
        for proposal_id in schedule_by_proposal
        if proposal_id != queue_only_id and proposal_id not in EXCLUDED_PROPOSAL_IDS
    )
    return classic_ids, schedule_by_proposal


class PFSConfigApp:
    def __init__(self):
        self.config_path = DEFAULT_CONFIG_PATH
        self.config_data = {}
        self.classic_ids = []
        self.schedule_by_proposal = {}
        self.default_observation_text = ""
        self.id_source_label = ""
        self._mode_guard = False
        self._proposal_guard = False
        self._program_sheet_url_guard = False
        self.program_sheet_links = self._load_program_sheet_links()
        self.target_insert_status = self._load_target_insert_status()
        self.proposal_source_data = pd.DataFrame()

        self.path_input = pn.widgets.TextInput(
            name="config file",
            value=str(self.config_path),
            sizing_mode="stretch_width",
        )
        self.reload_button = pn.widgets.Button(
            name="Reload config",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.refresh_ids_button = pn.widgets.Button(
            name="Refresh proposal IDs",
            button_type="default",
            sizing_mode="stretch_width",
        )
        self.semester_select = pn.widgets.Select(
            name="",
            options=SEMESTER_OPTIONS,
            value=DEFAULT_SEMESTER,
            sizing_mode="stretch_width",
        )
        self.program_status_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["proposal_id", "status", "inserted_at"]),
            show_index=False,
            disabled=True,
            selectable="checkbox",
            pagination=None,
            layout="fit_data_stretch",
            sizing_mode="stretch_width",
            height=220,
        )
        self.program_status_table.style.apply(
            self._style_program_status_row,
            axis=1,
        )
        self.insert_programs_button = pn.widgets.Button(
            name="Insert checked programs",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.insert_summary = pn.pane.Markdown(sizing_mode="stretch_width")
        self.production_confirmation = pn.widgets.Checkbox(
            name="I confirm insertion into the production database",
            value=False,
        )
        self.run_insert_button = pn.widgets.Button(
            name="Run insertion",
            button_type="primary",
            disabled=True,
            width=140,
        )
        self.insert_log = pn.pane.Str(
            "",
            styles={
                "background": "#111827",
                "color": "#f3f4f6",
                "font-family": "monospace",
                "white-space": "pre-wrap",
                "padding": "12px",
                "overflow-y": "auto",
            },
            height=480,
            sizing_mode="stretch_width",
        )
        self.insert_modal = pn.Column(
            pn.pane.HTML(f"<div style=\"{SECTION_TITLE_STYLE}\">Insert targets</div>"),
            self.insert_summary,
            self.production_confirmation,
            self.run_insert_button,
            self.insert_log,
            sizing_mode="stretch_width",
        )
        self._pending_proposals_path = None
        self._pending_input_catalogs_path = None
        self._pending_uploads_path = None
        self._pending_proposal_ids = []
        self._pending_has_user_pointings = False
        self.mode_selector = pn.widgets.CheckButtonGroup(
            name="Select mode",
            options=["queue", "classic"],
            value=["queue"],
            button_type="success",
            button_style="solid",
            sizing_mode="stretch_width",
        )
        self.proposal_select = pn.widgets.Select(
            name="proposal ID",
            options=[self._queue_only_id()],
            value=self._queue_only_id(),
            disabled=True,
            sizing_mode="stretch_width",
        )
        self.observation_text = pn.widgets.TextAreaInput(
            name="observation durations",
            disabled=True,
            height=140,
            description="All times are in HST. \n Please note that the date is the scheduled night, not the actual observation date. For example, an observation from 00:00 to 05:00 on the night of 2026-03-15 should be entered as `2026-03-15 00:00-05:00`, not `2026-03-16 00:00-05:00`.",
            placeholder="HST",
            sizing_mode="stretch_width",
        )
        self.refresh_observation_button = pn.widgets.Button(
            name="Refresh default",
            button_type="default",
            sizing_mode="stretch_width",
        )
        self.save_config_button = pn.widgets.Button(
            name="Save the config",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.save_path_input = pn.widgets.TextInput(
            name="save path",
            value=str(self.config_path),
            sizing_mode="stretch_width",
        )
        self.confirm_save_button = pn.widgets.Button(
            name="Save",
            button_type="primary",
            width=100,
        )
        self.cancel_save_button = pn.widgets.Button(
            name="Cancel",
            button_type="default",
            width=100,
        )
        self.save_dialog = pn.Column(
            pn.pane.HTML(f"<div style=\"{SUBTITLE_STYLE}\">Save config</div>"),
            self.save_path_input,
            pn.Row(self.confirm_save_button, self.cancel_save_button),
            visible=False,
            sizing_mode="stretch_width",
        )
        self.program_sheet_url_input = pn.widgets.TextInput(
            name="Google Spreadsheet link",
            value=self.program_sheet_links.get(DEFAULT_SEMESTER, ""),
            placeholder="https://docs.google.com/spreadsheets/d/.../edit#gid=0",
            sizing_mode="stretch_width",
        )
        self.program_sheet_view = pn.pane.HTML(
            sizing_mode="stretch_both",
            min_height=760,
        )
        self.program_details_panel = pn.Column(
            self.program_sheet_url_input,
            self.program_sheet_view,
            sizing_mode="stretch_both",
        )
        self.left_status = pn.pane.Markdown(sizing_mode="stretch_width")
        self.right_panel = pn.Column(sizing_mode="stretch_both", scroll=True)

        self.reload_button.on_click(self._reload_config)
        self.refresh_ids_button.on_click(self._refresh_ids)
        self.refresh_observation_button.on_click(self._reset_observation_text)
        self.save_config_button.on_click(self._open_save_dialog)
        self.confirm_save_button.on_click(self._save_config)
        self.cancel_save_button.on_click(self._close_save_dialog)
        self.insert_programs_button.on_click(self._prepare_target_insertion)
        self.production_confirmation.param.watch(
            self._on_production_confirmation, "value"
        )
        self.run_insert_button.on_click(self._run_target_insertion)
        self.program_sheet_url_input.param.watch(
            self._on_program_sheet_url_change, "value"
        )
        self.mode_selector.param.watch(self._on_mode_change, "value")
        self.semester_select.param.watch(self._on_semester_change, "value")
        self.proposal_select.param.watch(self._on_proposal_change, "value")
        self.observation_text.param.watch(self._on_observation_text_change, "value")

        self._refresh_ids(notify=False)
        self._load_config(self.config_path, notify=False)
        self._render_program_sheet()
        pn.state.onload(self._load_initial_program_status)

    def _notify(self, message, level="info"):
        if pn.state.notifications:
            getattr(pn.state.notifications, level)(message)

    def _load_program_sheet_links(self):
        try:
            return json.loads(PROGRAM_SHEET_LINKS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_program_sheet_links(self):
        PROGRAM_SHEET_LINKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROGRAM_SHEET_LINKS_PATH.write_text(
            json.dumps(self.program_sheet_links, indent=2) + "\n",
            encoding="utf-8",
        )

    def _load_target_insert_status(self):
        try:
            return json.loads(TARGET_INSERT_STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _style_program_status_row(row):
        if row["status"] == "✗":
            return ["background-color: tomato; color: white"] * len(row)
        return [""] * len(row)

    def _save_target_insert_status(self):
        TARGET_INSERT_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TARGET_INSERT_STATUS_PATH.write_text(
            json.dumps(self.target_insert_status, indent=2) + "\n",
            encoding="utf-8",
        )

    def _google_sheet_csv_url(self, sheet_name):
        sheet_url = self._google_sheet_web_url()
        if sheet_url is None:
            return None
        spreadsheet_id = re.fullmatch(
            r"/spreadsheets/d/([^/]+)(?:/.*)?", urlparse(sheet_url).path
        ).group(1)
        return (
            f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?"
            f"tqx=out:csv&sheet={quote(sheet_name)}"
        )

    def _read_google_sheet_csv(self, sheet_name):
        csv_url = self._google_sheet_csv_url(sheet_name)
        if csv_url is None:
            return None
        request = Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=20) as response:
            return pd.read_csv(BytesIO(response.read()), keep_default_na=False)

    def _load_initial_program_status(self):
        self._refresh_program_status_table(notify=True)

    def _refresh_program_status_table(self, notify=True):
        semester_key = self.semester_select.value.lower()
        sheet_name = "proposals"
        if self._google_sheet_csv_url(sheet_name) is None:
            self.proposal_source_data = pd.DataFrame()
            self.program_status_table.value = pd.DataFrame(
                columns=["proposal_id", "status", "inserted_at"]
            )
            return

        try:
            proposals = self._read_google_sheet_csv(sheet_name)
            if "proposal_id" not in proposals.columns:
                raise ValueError("The proposals worksheet has no proposal_id column.")
        except (OSError, ValueError) as error:
            if notify:
                self._notify(f"Could not load proposal table: {error}", "error")
            return

        self.proposal_source_data = proposals
        semester_status = self.target_insert_status.get(semester_key, {})
        status_rows = []
        for proposal_id in proposals["proposal_id"].astype(str):
            inserted_at = semester_status.get(proposal_id, "")
            status_rows.append(
                {
                    "proposal_id": proposal_id,
                    "status": "✓" if inserted_at else "✗",
                    "inserted_at": inserted_at,
                }
            )
        status_rows.sort(key=lambda row: row["status"] == "✓")
        self.program_status_table.value = pd.DataFrame(status_rows)
        self.program_status_table.selection = []

    def _selected_program_ids(self):
        selection = list(self.program_status_table.selection)
        if not selection:
            return []
        selected_rows = self.program_status_table.value.iloc[selection]
        return selected_rows["proposal_id"].astype(str).tolist()

    def _prepare_target_insertion(self, event=None):
        proposal_ids = self._selected_program_ids()
        if not proposal_ids:
            self._notify("Check at least one program to insert.", "warning")
            return

        semester_key = self.semester_select.value.lower()
        try:
            input_catalogs = self._read_google_sheet_csv("Input catalogs")
            if "proposal_id" not in input_catalogs.columns:
                raise ValueError(
                    "The input_catalogs worksheet has no proposal_id column."
                )
        except (OSError, ValueError) as error:
            self._notify(f"Could not load input catalogs: {error}", "error")
            return

        proposal_mask = self.proposal_source_data["proposal_id"].astype(str).isin(
            proposal_ids
        )
        input_catalog_mask = input_catalogs["proposal_id"].astype(str).isin(
            proposal_ids
        )
        selected_proposals = self.proposal_source_data.loc[proposal_mask]
        selected_input_catalogs = input_catalogs.loc[input_catalog_mask]
        if selected_input_catalogs.empty:
            self._notify("No matching input catalogs were found.", "error")
            return

        semester_dir = TARGET_INSERT_DIR / semester_key
        uploads_dir = semester_dir / "uploads"
        proposals_path = semester_dir / f"proposals_{semester_key}.csv"
        input_catalogs_path = semester_dir / f"input_catalogs_{semester_key}.csv"
        try:
            uploads_dir.mkdir(parents=True, exist_ok=True)
            selected_proposals.to_csv(proposals_path, index=False)
            selected_input_catalogs.to_csv(input_catalogs_path, index=False)
        except OSError as error:
            self._notify(f"Could not save insertion CSV files: {error}", "error")
            return

        pointings = selected_input_catalogs.get("is_user_pointing")
        if pointings is None:
            has_user_pointings = False
        else:
            has_user_pointings = pointings.astype(str).str.lower().isin(
                {"true", "1", "yes", "y"}
            ).any()

        self._pending_proposals_path = proposals_path
        self._pending_input_catalogs_path = input_catalogs_path
        self._pending_uploads_path = uploads_dir
        self._pending_proposal_ids = proposal_ids
        self._pending_has_user_pointings = has_user_pointings
        self.insert_summary.object = (
            f"**Semester:** {self.semester_select.value}  \n"
            f"**Programs:** {', '.join(proposal_ids)}  \n"
            f"**Input catalogs:** {len(selected_input_catalogs)}  \n"
            f"**User pointings:** {'yes' if has_user_pointings else 'no'}"
        )
        self.production_confirmation.value = False
        self.run_insert_button.disabled = True
        self.insert_log.object = ""
        self.template.open_modal()

    def _on_production_confirmation(self, event):
        self.run_insert_button.disabled = not event.new

    def _target_insert_commands(self):
        proposals_path = str(self._pending_proposals_path)
        input_catalogs_path = str(self._pending_input_catalogs_path)
        uploads_path = str(self._pending_uploads_path)
        dev_config = str(TARGETDB_DEV_CONFIG)
        production_config = str(TARGETDB_PRODUCTION_CONFIG)
        commands = [
            ["pfs-targetdb-cli", "insert", proposals_path, "-c", dev_config, "--table", "proposal", "--commit"],
            ["pfs-targetdb-cli", "insert", proposals_path, "-c", production_config, "--table", "proposal", "--commit"],
            ["pfs-targetdb-cli", "insert", input_catalogs_path, "-c", dev_config, "--table", "input_catalog", "--commit", "--verbose"],
            ["pfs-targetdb-cli", "insert", input_catalogs_path, "-c", production_config, "--table", "input_catalog", "--commit", "--verbose"],
            ["pfs-targetdb-cli", "transfer-targets-api", input_catalogs_path, "-c", production_config, "--local-dir", uploads_path],
            ["pfs-targetdb-cli", "insert-targets", input_catalogs_path, "-c", dev_config, "--data-dir", uploads_path, "--commit"],
            ["pfs-targetdb-cli", "insert-targets", input_catalogs_path, "-c", production_config, "--data-dir", uploads_path, "--commit"],
        ]
        if self._pending_has_user_pointings:
            commands.extend(
                [
                    ["pfs-targetdb-cli", "insert-pointings", input_catalogs_path, "-c", dev_config, "--data-dir", uploads_path, "--verbose", "--commit"],
                    ["pfs-targetdb-cli", "insert-pointings", input_catalogs_path, "-c", production_config, "--data-dir", uploads_path, "--verbose", "--commit"],
                ]
            )
        return commands

    async def _run_target_insertion(self, event=None):
        if not self.production_confirmation.value:
            return

        self.run_insert_button.disabled = True
        self.production_confirmation.disabled = True
        log_lines = []
        succeeded = True
        for command in self._target_insert_commands():
            log_lines.append(f"$ {shlex.join(command)}")
            self.insert_log.object = "\n".join(log_lines)
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except OSError as error:
                log_lines.append(str(error))
                succeeded = False
                break

            async for output in process.stdout:
                log_lines.append(output.decode(errors="replace").rstrip())
                self.insert_log.object = "\n".join(log_lines)
            return_code = await process.wait()
            log_lines.append(f"[exit code {return_code}]")
            self.insert_log.object = "\n".join(log_lines)
            if return_code != 0:
                succeeded = False
                break

        if succeeded:
            inserted_at = datetime.now().astimezone().isoformat(timespec="seconds")
            semester_key = self.semester_select.value.lower()
            semester_status = self.target_insert_status.setdefault(semester_key, {})
            for proposal_id in self._pending_proposal_ids:
                semester_status[proposal_id] = inserted_at
            try:
                self._save_target_insert_status()
            except OSError as error:
                log_lines.append(f"Could not save local status: {error}")
            self._refresh_program_status_table(notify=False)
            log_lines.append(f"Completed successfully at {inserted_at}")
        else:
            log_lines.append("Stopped after command failure; status was not updated.")

        self.insert_log.object = "\n".join(log_lines)
        self.production_confirmation.disabled = False

    def _set_program_sheet_url_for_semester(self):
        self._program_sheet_url_guard = True
        self.program_sheet_url_input.value = self.program_sheet_links.get(
            self.semester_select.value, ""
        )
        self._program_sheet_url_guard = False
        self._render_program_sheet()

    def _google_sheet_web_url(self):
        sheet_url = self.program_sheet_url_input.value.strip()
        parsed_url = urlparse(sheet_url)
        match = re.fullmatch(r"/spreadsheets/d/([^/]+)(?:/.*)?", parsed_url.path)
        if parsed_url.netloc != "docs.google.com" or match is None:
            return None

        return sheet_url

    def _render_program_sheet(self):
        sheet_url = self._google_sheet_web_url()
        if sheet_url is None:
            self.program_sheet_view.object = ""
            return

        escaped_url = html.escape(sheet_url, quote=True)
        self.program_sheet_view.object = (
            f'<iframe src="{escaped_url}" '
            'style="width: 100%; height: 78vh; min-height: 760px; border: 0;" '
            'loading="lazy"></iframe>'
        )

    def _on_program_sheet_url_change(self, event):
        if self._program_sheet_url_guard:
            return

        if not event.new.strip():
            self.program_sheet_links.pop(self.semester_select.value, None)
            self.program_sheet_view.object = ""
        else:
            self.program_sheet_links[self.semester_select.value] = event.new.strip()
            self._render_program_sheet()

        try:
            self._save_program_sheet_links()
        except OSError as error:
            self._notify(f"Could not save spreadsheet link: {error}", "error")
            return

        self._refresh_program_status_table()

    def _current_mode(self):
        if len(self.mode_selector.value) == 1:
            return self.mode_selector.value[0]
        return "queue"

    def _queue_only_id(self):
        return f"{self.semester_select.value}-999QN"

    def _set_mode(self, mode):
        self._mode_guard = True
        self.mode_selector.value = [mode]
        self._mode_guard = False

    def _set_proposal_value(self, value):
        self._proposal_guard = True
        self.proposal_select.value = value
        self._proposal_guard = False

    def _refresh_ids(self, event=None, notify=True):
        semester_code = self.semester_select.value
        try:
            self.classic_ids, self.schedule_by_proposal = fetch_pfs_schedule_data(
                semester_code
            )
            if not self.classic_ids:
                raise ValueError("No matching PFS proposal IDs found on the schedule page.")
            self.id_source_label = (
                f"Loaded {len(self.classic_ids)} classic proposal IDs from the "
                f"{semester_code} Subaru schedule."
            )
            if notify:
                self._notify(
                    f"Fetched proposal IDs from the {semester_code} schedule page.",
                    "success",
                )
        except Exception:
            self.classic_ids = []
            self.schedule_by_proposal = {}
            self.id_source_label = (
                f"Could not fetch proposal IDs from the {semester_code} schedule page."
            )
            if notify:
                self._notify(
                    f"Could not fetch proposal IDs from the {semester_code} schedule page.",
                    "warning",
                )

        self._apply_mode_to_widgets(self._current_mode(), rerender=False)
        self._update_observation_text()
        self._update_status()

    def _load_config(self, config_path, notify=True):
        try:
            with Path(config_path).expanduser().open("rb") as stream:
                self.config_data = tomllib.load(stream)
        except FileNotFoundError:
            self.config_data = {
                "ppp": {"mode": "queue", "proposalIds": [self._queue_only_id()]}
            }
            if notify:
                self._notify(f"Config file not found: {config_path}", "error")
        except tomllib.TOMLDecodeError as error:
            self.config_data = {
                "ppp": {"mode": "queue", "proposalIds": [self._queue_only_id()]}
            }
            if notify:
                self._notify(f"Invalid TOML: {error}", "error")
        else:
            if notify:
                self._notify(f"Loaded config: {config_path}", "success")

        self.config_path = Path(config_path).expanduser()
        self._sync_controls_from_config()
        self._render_config_panel()
        self._update_status()

    def _reload_config(self, event=None):
        self._load_config(self.path_input.value, notify=True)

    def _open_save_dialog(self, event=None):
        self._activate_config_tab()
        self.save_path_input.value = str(self.config_path)
        self.save_dialog.visible = True

    def _close_save_dialog(self, event=None):
        self.save_dialog.visible = False

    def _save_config(self, event=None):
        save_path = Path(self.save_path_input.value).expanduser()
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(dumps_toml(self.config_data), encoding="utf-8")
        except Exception as error:
            self._notify(f"Could not save config: {error}", "error")
            return

        self._notify(f"Saved config to {save_path}", "success")
        self.save_dialog.visible = False

    def _sync_controls_from_config(self):
        ppp_config = self.config_data.setdefault("ppp", {})
        mode = ppp_config.get("mode", "queue")
        mode = "classic" if str(mode).lower() == "classic" else "queue"
        if mode == "classic":
            ppp_config["daily_plan"] = False
        proposal_ids = ppp_config.get("proposalIds", [])

        selected_proposal = None
        if mode == "classic":
            for proposal_id in proposal_ids:
                if proposal_id != self._queue_only_id():
                    selected_proposal = proposal_id
                    break

        if selected_proposal is None and self.classic_ids:
            selected_proposal = self.classic_ids[0]

        self._set_mode(mode)
        self._apply_mode_to_widgets(mode, selected_proposal=selected_proposal, rerender=False)

    def _classic_options(self):
        return self.classic_ids

    def _apply_mode_to_widgets(self, mode, selected_proposal=None, rerender=True):
        ppp_config = self.config_data.setdefault("ppp", {})
        ppp_config["mode"] = mode
        if mode == "classic":
            ppp_config["daily_plan"] = False

        if mode == "queue":
            queue_only_id = self._queue_only_id()
            self.proposal_select.options = [queue_only_id]
            self.proposal_select.disabled = True
            self._set_proposal_value(queue_only_id)
            ppp_config["proposalIds"] = [queue_only_id]
        else:
            classic_options = self._classic_options()
            self.proposal_select.options = classic_options
            self.proposal_select.disabled = False

            if selected_proposal in classic_options:
                proposal_value = selected_proposal
            else:
                existing_value = None
                for proposal_id in ppp_config.get("proposalIds", []):
                    if proposal_id in classic_options:
                        existing_value = proposal_id
                        break
                proposal_value = existing_value or (classic_options[0] if classic_options else None)

            if proposal_value is not None:
                self._set_proposal_value(proposal_value)
                ppp_config["proposalIds"] = [proposal_value]
            else:
                self._set_proposal_value(None)
                ppp_config["proposalIds"] = []

        self._update_observation_text()

        if rerender:
            self._render_config_panel()
            self._update_status()

    def _sync_qplan_from_observation_text(self):
        qplan_config = self.config_data.setdefault("qplan", {})
        parsed_entries = parse_observation_text(self.observation_text.value)
        if parsed_entries is None:
            return False

        observation_dates = []
        for entry in parsed_entries:
            if entry["date"] not in observation_dates:
                observation_dates.append(entry["date"])

        qplan_config["obs_dates"] = observation_dates
        qplan_config["start_time"] = [entry["start"] for entry in parsed_entries]
        qplan_config["stop_time"] = [entry["stop"] for entry in parsed_entries]
        return True

    def _format_observation_entry(self, entry):
        display_date = entry["start"].strftime("%Y-%m-%d")
        start_text = entry["start"].strftime("%H:%M")
        end_text = entry["end"].strftime("%H:%M")
        return f"{display_date} {start_text}-{end_text}"

    def _update_observation_text(self):
        proposal_id = self.proposal_select.value
        schedule_entries = sorted(
            self.schedule_by_proposal.get(proposal_id, []),
            key=lambda entry: entry["start"],
        )

        if not schedule_entries:
            self.default_observation_text = ""
            self.observation_text.disabled = True
            self.observation_text.value = ""
            self.refresh_observation_button.disabled = True
            self._sync_qplan_from_observation_text()
            return

        self.default_observation_text = "\n".join(
            self._format_observation_entry(entry) for entry in schedule_entries
        )
        self.observation_text.disabled = False
        self.observation_text.value = self.default_observation_text
        self.refresh_observation_button.disabled = False
        self._sync_qplan_from_observation_text()

    def _reset_observation_text(self, event=None):
        self._activate_config_tab()
        self.observation_text.value = self.default_observation_text
        if self._sync_qplan_from_observation_text():
            self._render_config_panel()
            self._update_status()

    def _on_mode_change(self, event):
        if self._mode_guard:
            return

        self._activate_config_tab()

        new_value = list(event.new)
        old_value = list(event.old) if event.old is not None else []

        if not new_value:
            self._set_mode(old_value[0] if old_value else "queue")
            return

        if len(new_value) > 1:
            added = [item for item in new_value if item not in old_value]
            selected_mode = added[-1] if added else new_value[-1]
            self._set_mode(selected_mode)
            new_value = [selected_mode]

        self._apply_mode_to_widgets(new_value[0])

    def _on_semester_change(self, event):
        self._refresh_ids(notify=True)
        self._set_program_sheet_url_for_semester()
        self._refresh_program_status_table()
        self._render_config_panel()

    def _on_proposal_change(self, event):
        if self._proposal_guard or self._current_mode() != "classic":
            return

        self._activate_config_tab()
        proposal_id = event.new
        if proposal_id:
            self.config_data.setdefault("ppp", {})["proposalIds"] = [proposal_id]
            self._update_observation_text()
            self._render_config_panel()
            self._update_status()

    def _on_observation_text_change(self, event):
        self._activate_config_tab()
        if self._sync_qplan_from_observation_text():
            self._render_config_panel()

    def _activate_config_tab(self):
        if hasattr(self, "main_tabs"):
            self.main_tabs.active = 1

    def _render_config_panel(self):
        display_config = copy.deepcopy(self.config_data)
        if self._current_mode() == "queue":
            display_config.setdefault("ppp", {}).pop("proposalIds", None)

        sections = flatten_sections(display_config)
        accordion_items = []

        for section_name, frame in sections:
            table = pn.widgets.Tabulator(
                frame,
                show_index=False,
                disabled=False,
                pagination=None,
                layout="fit_data_stretch",
                formatters={"parameter": {"type": "html"}},
                sizing_mode="stretch_width",
                height=min(400, 36 * (len(frame) + 1)),
            )
            accordion_items.append((section_name, table))

        header = pn.pane.HTML(
            (
                f"<div style=\"{SECTION_TITLE_STYLE}\">Config parameters</div>"
            ),
            sizing_mode="stretch_width",
        )
        accordion = pn.Accordion(*accordion_items, active=[0], sizing_mode="stretch_width")
        self.right_panel[:] = [header, accordion]

    def _update_status(self):
        mode = self._current_mode()
        proposal_id = self.proposal_select.value or "-"
        self.left_status.object = (
            f"**Current mode:** {mode}  \n"
            f"**Selected proposal ID:** {proposal_id}  \n"
            f"**Proposal ID source:** {self.id_source_label}"
        )

    def view(self):
        sidebar = pn.Column(
            pn.pane.HTML(
                f"<div style=\"{SEMESTER_TITLE_STYLE}\">Select semester</div>"
            ),
            self.semester_select,
            pn.Spacer(height=12),
            pn.pane.HTML(f"<div style=\"{SECTION_TITLE_STYLE}\">Program status</div>"),
            self.program_status_table,
            self.insert_programs_button,
            pn.Spacer(height=12),
            pn.pane.HTML(f"<div style=\"{SECTION_TITLE_STYLE}\">Make config</div>"),
            self.mode_selector,
            pn.Spacer(height=12),
            self.proposal_select,
            pn.Spacer(height=12),
            self.observation_text,
            pn.Spacer(height=8),
            self.refresh_observation_button,
            pn.Spacer(height=12),
            self.save_config_button,
            pn.Spacer(height=8),
            self.save_dialog,
            sizing_mode="stretch_width",
        )

        self.main_tabs = pn.Tabs(
            ("Program details", self.program_details_panel),
            ("Config parameters", self.right_panel),
            sizing_mode="stretch_both",
        )

        template = pn.template.FastListTemplate(
            title="PFS Observation Planning Tools",
            sidebar=[sidebar],
            main=[self.main_tabs],
            modal=[self.insert_modal],
            sidebar_width=360,
            theme_toggle=False,
        )
        self.template = template
        return template

app = PFSConfigApp().view()
app.servable()

