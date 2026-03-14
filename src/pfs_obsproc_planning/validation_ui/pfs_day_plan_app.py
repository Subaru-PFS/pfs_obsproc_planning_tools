import pandas as pd
import panel as pn
from datetime import datetime, timedelta
import glob
import os
from bs4 import BeautifulSoup
import re
from pathlib import Path

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
CSV_PATH = "/work/wanqqq/daily_process_status.csv"
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

    base_dir = f"/work/wanqqq/run_{yymm}"

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
    latest_time = sorted(df["last_updated"])[-1].strftime("%H:%M")
    refresh_btn.name = f"Refresh (Last update at {latest_time})"


refresh_btn.on_click(refresh)




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
    parts = [
        f"<b>qaDB: <span style='color:{color_status(row['status_qaDB'])}'>{row['status_qaDB']}</b>{show_time(row['time_qaDB'])}</span>",
        f"<b>queueDB: <span style='color:{color_status(row['status_queueDB'])}'>{row['status_queueDB']}</b>{show_time(row['time_queueDB'])}</span>",
        f"<b>designGenerator: <span style='color:{color_status(row['status_daily_processing'])}'>{row['status_daily_processing']}</b>{show_time(row['time_daily_processing'])}</span>",
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

    # collect rows that have any highlighted cells
    rows_flagged = {row for (row, _) in cell_highlight}


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
    ("Validation Figure", pdf_pane),
)

template = pn.template.BootstrapTemplate(
    title="Validation of PFS Queue Planning",
    sidebar=[
        refresh_btn,
        date_picker,
    ],
    sidebar_width=290,
    theme="default",
    header_background="#6A5AA3",
)

template.main.append(status_view)
template.main.append(tabs)

template.servable()