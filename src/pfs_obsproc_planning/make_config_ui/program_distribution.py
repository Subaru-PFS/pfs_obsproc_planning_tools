#!/usr/bin/env python3
"""Target distribution figures for programs inserted into targetdb.

No database connection information is stored in this module. Host, port, user,
password and database name are read at runtime from the ``[targetdb.db]`` table
of the TOML configuration, either directly or through the secondary file
referenced by ``[packages] config_path``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sqlalchemy as sa  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body  # noqa: E402
from astropy.time import Time  # noqa: E402

SUBARU_LOCATION = EarthLocation(
    lat=19.825556 * u.deg,
    lon=-155.476111 * u.deg,
    height=4139 * u.m,
)

HST = "Pacific/Honolulu"
DEFAULT_PA = 0.0
DEFAULT_CONE_RADIUS_DEG = 1.105
DEFAULT_SKY_VERSIONS = ("20230209",)
DEFAULT_FLUXSTD_VERSIONS = ("3.5",)
DEFAULT_FLUXSTD_PROBABILITY_RANGE = (0.5, 1.0)
DEFAULT_FLUXSTD_MAG_RANGE = (17.0, 19.0)
DEFAULT_FLUXSTD_TEFF_RANGE = (5000.0, 8000.0)
CLUSTER_EPS_DEG = 5.0
CATALOG_SAMPLE_LIMIT = 20000
ELEVATION_LIMITS = (32.0, 75.0)
TARGET_EPOCH = 2016.0


def _load_toml(path):
    with Path(path).expanduser().open("rb") as stream:
        return tomllib.load(stream)


def resolve_targetdb_params(config_data, config_path=None):
    """Return the ``[targetdb.db]`` mapping for the active configuration.

    The lookup order mirrors ``generatePfsDesign.read_conf``: the primary config
    wins, otherwise the secondary file named by ``[packages] config_path`` is
    consulted. Nothing is defaulted, so a misconfigured deployment fails loudly
    instead of silently pointing at the wrong database.
    """
    db_params = (config_data.get("targetdb") or {}).get("db")
    if db_params:
        return dict(db_params)

    secondary = (config_data.get("packages") or {}).get("config_path")
    if not secondary:
        raise ValueError(
            "No targetdb connection settings found. Add a [targetdb.db] section "
            "to the config file, or point [packages] config_path at the database "
            "configuration file."
        )

    secondary_path = Path(secondary).expanduser()
    if not secondary_path.is_absolute() and config_path is not None:
        secondary_path = (
            Path(config_path).expanduser().resolve().parent / secondary_path
        ).resolve()

    if not secondary_path.exists():
        raise ValueError(f"Database configuration file not found: {secondary_path}")

    db_params = (_load_toml(secondary_path).get("targetdb") or {}).get("db")
    if not db_params:
        raise ValueError(f"No [targetdb.db] section found in {secondary_path}.")
    return dict(db_params)


def create_targetdb_engine(db_params):
    """Build an SQLAlchemy engine from configuration-supplied parameters."""
    missing = [key for key in ("dialect", "host", "dbname") if not db_params.get(key)]
    if missing:
        raise ValueError(
            f"Incomplete [targetdb.db] settings; missing: {', '.join(missing)}"
        )

    url = sa.engine.URL.create(
        drivername=str(db_params["dialect"]),
        username=db_params.get("user"),
        password=db_params.get("password"),
        host=db_params.get("host"),
        port=db_params.get("port"),
        database=db_params.get("dbname"),
    )
    return sa.create_engine(url)


_TARGET_COLUMNS = (
    "t.ob_code, t.obj_id, t.ra, t.dec, t.priority, t.effective_exptime, "
    "t.single_exptime, t.qa_reference_arm, t.is_medium_resolution, t.proposal_id"
)


def _read_sql(engine, statement, params):
    with engine.connect() as connection:
        rows = connection.execute(statement, params).mappings().all()
    return pd.DataFrame(rows)


def query_program_targets(engine, proposal_id):
    """Return the active targets registered for a single proposal."""
    statement = sa.text(
        f"SELECT {_TARGET_COLUMNS} FROM target AS t "
        "JOIN input_catalog AS c ON t.input_catalog_id = c.input_catalog_id "
        "WHERE t.proposal_id = :proposal_id AND c.active"
    )
    return _read_sql(engine, statement, {"proposal_id": proposal_id})


def query_neighbour_targets(engine, proposal_ids, ra, dec, radius_deg):
    """Return targets of other proposals inside a cone, for overlap checks.

    The cone restriction keeps the query bounded: without it this would scan
    every target of every proposal in the semester.
    """
    proposal_ids = [str(value) for value in proposal_ids if str(value).strip()]
    if not proposal_ids:
        return pd.DataFrame(columns=["proposal_id", "ra", "dec"])

    statement = sa.text(
        "SELECT t.proposal_id, t.ra, t.dec FROM target AS t "
        "JOIN input_catalog AS c ON t.input_catalog_id = c.input_catalog_id "
        "WHERE c.active AND t.proposal_id IN :proposal_ids "
        "AND q3c_radial_query(t.ra, t.dec, :ra, :dec, :radius)"
    ).bindparams(sa.bindparam("proposal_ids", expanding=True))

    return _read_sql(
        engine,
        statement,
        {
            "proposal_ids": proposal_ids,
            "ra": float(ra),
            "dec": float(dec),
            "radius": float(radius_deg),
        },
    )


def _versions_clause(versions, params):
    if not versions:
        return "", None
    values = [str(value) for value in versions if str(value).strip()]
    if not values:
        return "", None
    params["versions"] = values
    return " AND version IN :versions", sa.bindparam("versions", expanding=True)


def query_sky(
    engine,
    ra,
    dec,
    radius_deg,
    versions=DEFAULT_SKY_VERSIONS,
    limit=CATALOG_SAMPLE_LIMIT,
):
    """Return sky positions inside a cone around the field centre."""
    params = {
        "ra": float(ra),
        "dec": float(dec),
        "radius": float(radius_deg),
        "limit": int(limit),
    }
    clause, binding = _versions_clause(versions, params)
    statement = sa.text(
        "SELECT ra, dec FROM sky "
        "WHERE q3c_radial_query(ra, dec, :ra, :dec, :radius)"
        f"{clause} LIMIT :limit"
    )
    if binding is not None:
        statement = statement.bindparams(binding)
    return _read_sql(engine, statement, params)


def query_fluxstd(
    engine,
    ra,
    dec,
    radius_deg,
    versions=DEFAULT_FLUXSTD_VERSIONS,
    probability_range=DEFAULT_FLUXSTD_PROBABILITY_RANGE,
    magnitude_range=DEFAULT_FLUXSTD_MAG_RANGE,
    temperature_range=DEFAULT_FLUXSTD_TEFF_RANGE,
    limit=CATALOG_SAMPLE_LIMIT,
):
    """Return usable flux standard stars inside a cone around the field centre."""
    params = {
        "ra": float(ra),
        "dec": float(dec),
        "radius": float(radius_deg),
        "limit": int(limit),
        "prob_f_star_min": float(probability_range[0]),
        "prob_f_star_max": float(probability_range[1]),
        "psf_mag_g_min": float(magnitude_range[0]),
        "psf_mag_g_max": float(magnitude_range[1]),
        "teff_brutus_min": float(temperature_range[0]),
        "teff_brutus_max": float(temperature_range[1]),
    }
    version_clause, binding = _versions_clause(versions, params)

    statement = sa.text(
        "SELECT ra, dec, psf_mag_g, prob_f_star, teff_brutus FROM fluxstd "
        "WHERE q3c_radial_query(ra, dec, :ra, :dec, :radius)"
        " AND prob_f_star BETWEEN :prob_f_star_min AND :prob_f_star_max"
        " AND psf_mag_g BETWEEN :psf_mag_g_min AND :psf_mag_g_max"
        " AND teff_brutus BETWEEN :teff_brutus_min AND :teff_brutus_max"
        f"{version_clause} LIMIT :limit"
    )
    if binding is not None:
        statement = statement.bindparams(binding)
    return _read_sql(engine, statement, params)


def moon_illumination(time):
    """Illuminated fraction of the lunar disc at ``time`` (0-1)."""
    sun = get_body("sun", time, location=SUBARU_LOCATION)
    moon = get_body("moon", time, location=SUBARU_LOCATION)
    elongation = sun.separation(moon)
    phase_angle = np.arctan2(
        sun.distance * np.sin(elongation),
        moon.distance - sun.distance * np.cos(elongation),
    )
    return float((1.0 + np.cos(phase_angle)) / 2.0)


def compute_night_curves(nights, ra, dec, pa=DEFAULT_PA, step_minutes=10):
    """Elevation, InR, Moon separation and illumination for allocated windows.

    ``nights`` is a sequence of mappings with ``date``, ``start`` and ``stop``
    entries, where the times are naive HST timestamps. Only the allocated
    windows are sampled, so the curves show the scheduled time only.
    """
    from pfs.utils.coordinates.DistortionCoefficients import radec_to_subaru

    target = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
    curves = {}

    for night in nights:
        start_hst = pd.Timestamp(night["start"], tz=HST)
        stop_hst = pd.Timestamp(night["stop"], tz=HST)
        if stop_hst <= start_hst:
            stop_hst += pd.Timedelta(days=1)

        times_utc = pd.date_range(
            start=start_hst.tz_convert("UTC"),
            end=stop_hst.tz_convert("UTC"),
            freq=f"{step_minutes}min",
            tz="UTC",
        )
        if len(times_utc) == 0:
            continue

        elevations = []
        rotator_angles = []
        for timestamp in times_utc:
            _, elevation, rotator = radec_to_subaru(
                float(ra),
                float(dec),
                float(pa),
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                TARGET_EPOCH,
                0,
                0,
                1e-7,
                inr=None,
                log=False,
                returnRaDec=False,
            )
            elevations.append(elevation)
            rotator_angles.append(rotator)

        astropy_times = Time(times_utc.to_pydatetime())
        altaz_frame = AltAz(obstime=astropy_times, location=SUBARU_LOCATION)
        moon_altaz = get_body(
            "moon", astropy_times, location=SUBARU_LOCATION
        ).transform_to(altaz_frame)
        separations = target.transform_to(altaz_frame).separation(moon_altaz).deg

        curves[night["date"]] = {
            "time": times_utc.tz_convert(HST),
            "el": np.asarray(elevations, dtype=float),
            "inr": np.asarray(rotator_angles, dtype=float),
            "moon_separation": np.asarray(separations, dtype=float),
            "moon_illumination": moon_illumination(
                astropy_times[len(astropy_times) // 2]
            ),
        }

    return curves


def cluster_targets(frame, eps_deg=CLUSTER_EPS_DEG):
    """Split a target list into spatially separated groups.

    Returns a list of ``(cluster_id, frame)`` pairs. Points DBSCAN labels as
    noise are kept in a single fallback group so that no target is silently
    dropped from the figure.
    """
    from sklearn.cluster import DBSCAN

    if frame.empty:
        return []

    coordinates = frame[["ra", "dec"]].to_numpy(dtype=float)
    if len(coordinates) < 2:
        return [(0, frame)]

    labels = DBSCAN(eps=eps_deg, min_samples=2).fit(coordinates).labels_
    tagged = frame.assign(cluster_id=labels)

    groups = [
        (int(cluster_id), tagged[tagged["cluster_id"] == cluster_id])
        for cluster_id in sorted(set(labels))
        if cluster_id != -1
    ]
    if not groups:
        return [(0, tagged)]

    noise = tagged[tagged["cluster_id"] == -1]
    if not noise.empty:
        groups.append((-1, noise))
    return groups


def _plot_sky_panel(axis, cluster, neighbours, title, ra_limits, dec_limits):
    priorities = pd.to_numeric(cluster["priority"], errors="coerce")
    top_priority = priorities.min()
    highest = cluster[priorities == top_priority]
    others = cluster[priorities != top_priority]

    if not neighbours.empty:
        colours = plt.cm.tab10
        for index, proposal in enumerate(sorted(neighbours["proposal_id"].unique())):
            subset = neighbours[neighbours["proposal_id"] == proposal]
            axis.scatter(
                subset["ra"],
                subset["dec"],
                s=2,
                facecolors=colours(index % 10),
                edgecolors="none",
                alpha=0.6,
                label=f"{proposal} ({len(subset)})",
            )

    axis.scatter(
        others["ra"],
        others["dec"],
        c="red",
        s=18,
        edgecolors="none",
        alpha=0.8,
        label=f"P > {top_priority:g} ({len(others)})",
    )
    axis.scatter(
        highest["ra"],
        highest["dec"],
        c="black",
        s=18,
        edgecolors="none",
        alpha=0.9,
        label=f"P = {top_priority:g} ({len(highest)})",
    )

    axis.set_xlabel("RA [deg]")
    axis.set_ylabel("Dec [deg]")
    axis.set_xlim(*ra_limits)
    axis.set_ylim(*dec_limits)
    axis.set_title(title)
    axis.legend(loc="best", fontsize=7, framealpha=0.85)


def _plot_histograms(axis_exptime, axis_priority, axis_arm, cluster):
    exposure = pd.to_numeric(cluster["effective_exptime"], errors="coerce").dropna()
    if not exposure.empty:
        axis_exptime.hist(exposure, bins=20, color="red", alpha=0.6)
    axis_exptime.set_title("Exptime distribution", fontsize=10)
    axis_exptime.set_xlabel("effective_exptime [s]", fontsize=8)

    priorities = pd.to_numeric(cluster["priority"], errors="coerce").dropna()
    if not priorities.empty:
        priorities = priorities[
            (priorities >= 0) & (priorities <= 9) & (priorities % 1 == 0)
        ].astype(int)
        axis_priority.hist(
            priorities,
            bins=np.arange(-0.5, 10.5, 1),
            color="red",
            alpha=0.6,
        )
    axis_priority.set_title("Priority distribution", fontsize=10)
    axis_priority.set_xlabel("priority", fontsize=8)
    axis_priority.set_xlim(-0.5, 9.5)
    axis_priority.set_xticks(range(10))

    arms = cluster["qa_reference_arm"].dropna().astype(str)
    arms = arms[arms.str.strip() != ""]
    if not arms.empty:
        counts = arms.value_counts().sort_index()
        axis_arm.bar(counts.index, counts.values, color="red", alpha=0.6)
    axis_arm.set_title("qa_reference_arm distribution", fontsize=10)
    axis_arm.set_xlabel("qa_reference_arm", fontsize=8)


def _plot_night_panel(figure, subgrid, index, date, curve, is_last):
    axis_elevation = figure.add_subplot(subgrid[index, 0])
    axis_rotator = axis_elevation.twinx()

    axis_elevation.plot(curve["time"], curve["el"], color="tab:blue", lw=1.8)
    for limit in ELEVATION_LIMITS:
        axis_elevation.axhline(limit, color="gray", ls="--", lw=0.8)
    axis_elevation.set_ylabel("EL (deg)", color="tab:blue", fontsize=8)
    axis_elevation.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)

    axis_rotator.plot(curve["time"], curve["inr"], color="tab:red", ls="--", lw=1.4)
    axis_rotator.set_ylabel("INR (deg)", color="tab:red", fontsize=8)
    axis_rotator.tick_params(axis="y", labelcolor="tab:red", labelsize=7)

    # Moon separation is annotated sparsely so the labels stay readable.
    separations = curve["moon_separation"]
    sample_count = len(separations)
    for position in np.linspace(0, sample_count - 1, min(5, sample_count)).astype(int):
        axis_elevation.annotate(
            f"{separations[position]:.0f}°",
            (curve["time"][position], curve["el"][position]),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=7,
            color="dimgray",
            ha="center",
        )

    axis_elevation.set_title(
        f"{date} (HST) — Moon {curve['moon_illumination'] * 100:.0f}%"
        f", min sep {np.min(separations):.0f}°",
        fontsize=9,
    )
    axis_elevation.tick_params(axis="x", rotation=30, labelsize=7)
    if is_last:
        axis_elevation.set_xlabel("Time (HST)", fontsize=8)


def _plot_calibration_panel(axis, frame, colour, label, ra_limits, dec_limits):
    if frame.empty:
        axis.text(
            0.5,
            0.5,
            f"No {label} returned",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="gray",
        )
    else:
        axis.scatter(
            frame["ra"],
            frame["dec"],
            s=4,
            c=colour,
            edgecolors="none",
            alpha=0.6,
        )
    axis.set_xlabel("RA [deg]", fontsize=8)
    axis.set_ylabel("Dec [deg]", fontsize=8)
    axis.set_xlim(*ra_limits)
    axis.set_ylim(*dec_limits)
    axis.set_title(f"{label} ({len(frame)})", fontsize=10)
    axis.tick_params(labelsize=7)


def plot_cluster_figure(
    proposal_id,
    cluster,
    neighbours,
    night_curves,
    sky_frame,
    fluxstd_frame,
    output_path,
    title_suffix="",
):
    """Render and save one distribution figure, returning the output path."""
    ra_limits = (cluster["ra"].min() - 0.1, cluster["ra"].max() + 0.1)
    dec_limits = (cluster["dec"].min() - 0.1, cluster["dec"].max() + 0.1)

    night_count = max(len(night_curves), 1)
    figure = plt.figure(figsize=(19, 10))
    outer = figure.add_gridspec(2, 1, height_ratios=[2.6, 1.0], hspace=0.32)
    top = outer[0].subgridspec(
        3, 3, width_ratios=[2.2, 1.0, 1.6], wspace=0.38, hspace=0.55
    )
    bottom = outer[1].subgridspec(1, 2, wspace=0.22)

    _plot_sky_panel(
        figure.add_subplot(top[:, 0]),
        cluster,
        neighbours,
        f"{proposal_id}{title_suffix}",
        ra_limits,
        dec_limits,
    )
    _plot_histograms(
        figure.add_subplot(top[0, 1]),
        figure.add_subplot(top[1, 1]),
        figure.add_subplot(top[2, 1]),
        cluster,
    )

    night_grid = top[:, 2].subgridspec(night_count, 1, hspace=0.75)
    if night_curves:
        for index, (date, curve) in enumerate(night_curves.items()):
            _plot_night_panel(
                figure, night_grid, index, date, curve, index == night_count - 1
            )
    else:
        axis = figure.add_subplot(night_grid[0, 0])
        axis.text(
            0.5,
            0.5,
            "No allocated nights found",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        axis.set_axis_off()

    _plot_calibration_panel(
        figure.add_subplot(bottom[0, 0]),
        sky_frame,
        "tab:green",
        "sky",
        ra_limits,
        dec_limits,
    )
    _plot_calibration_panel(
        figure.add_subplot(bottom[0, 1]),
        fluxstd_frame,
        "tab:purple",
        "fluxstd",
        ra_limits,
        dec_limits,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_program_figures(
    proposal_id,
    nights,
    config_data,
    config_path,
    output_dir,
    neighbour_proposal_ids=(),
    pa=DEFAULT_PA,
    cone_radius_deg=DEFAULT_CONE_RADIUS_DEG,
):
    """Create the distribution figures for one program.

    Returns ``(paths, messages)`` where ``paths`` lists the written PNG files
    and ``messages`` carries non-fatal notes such as skipped calibration
    queries.
    """
    db_params = resolve_targetdb_params(config_data, config_path)
    engine = create_targetdb_engine(db_params)
    messages = []

    try:
        targets = query_program_targets(engine, proposal_id)
        if targets.empty:
            raise ValueError(
                f"No active targets found in targetdb for {proposal_id}."
            )

        others = [
            value
            for value in neighbour_proposal_ids
            if str(value) != str(proposal_id)
        ]

        output_dir = Path(output_dir)
        groups = cluster_targets(targets)
        written = []

        for cluster_id, cluster in groups:
            centre_ra = float(cluster["ra"].median())
            centre_dec = float(cluster["dec"].median())

            neighbours = query_neighbour_targets(
                engine, others, centre_ra, centre_dec, cone_radius_deg
            )
            try:
                sky_frame = query_sky(
                    engine, centre_ra, centre_dec, cone_radius_deg
                )
            except Exception as error:  # calibration panels are optional
                sky_frame = pd.DataFrame(columns=["ra", "dec"])
                messages.append(f"sky query failed: {error}")
            try:
                fluxstd_frame = query_fluxstd(
                    engine,
                    centre_ra,
                    centre_dec,
                    cone_radius_deg,
                )
            except Exception as error:
                fluxstd_frame = pd.DataFrame(columns=["ra", "dec"])
                messages.append(f"fluxstd query failed: {error}")

            night_curves = compute_night_curves(
                nights, centre_ra, centre_dec, pa=pa
            )

            if len(groups) > 1:
                suffix = f"_cluster{cluster_id}"
                title_suffix = f" — cluster {cluster_id}"
            else:
                suffix = ""
                title_suffix = ""

            written.append(
                plot_cluster_figure(
                    proposal_id,
                    cluster,
                    neighbours,
                    night_curves,
                    sky_frame,
                    fluxstd_frame,
                    output_dir / f"{proposal_id}{suffix}.png",
                    title_suffix=title_suffix,
                )
            )

        return written, messages
    finally:
        engine.dispose()
