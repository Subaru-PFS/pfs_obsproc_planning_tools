# Configuration parameter descriptions (`config.toml`) 

This document explains each parameter specified in `config.toml` (see example in [config.toml](examples/workdir_template/config.toml)). 

---

## Table of Contents
- [ppp](#ppp)
- [qplan](#qplan)
- [qplan.weight](#qplanweight)
- [targetdb.db](#targetdbdb)
- [targetdb.sky](#targetdbsky)
- [targetdb.fluxstd](#targetdbfluxstd)
- [gaiadb](#gaiadb)
- [schemacrawler](#schemacrawler)
- [netflow](#netflow)
- [gurobi](#gurobi)
- [sfa](#sfa)
- [ope](#ope)

---


## `[ppp]`

| Parameter      | Allowed Values (Type) | Default         | Description |
|----------------|------------------------|-----------------|-------------|
| `mode`         | string (e.g., `'db'`, `'local'`) | `'db'` | Execution mode. Typically determines whether inputs are pulled from a database (`'db'`) or local files (`'local'`). |
| `localPath`    | string (path)         | `''`            | Local data path used when not in DB mode. Empty string means unused. |
| `TEXP_NOMINAL` | number (float, seconds) | `900.0`       | Nominal exposure time used for planning or scaling calculations. |
| `sql_query`    | string (SQL)          | *(long SELECT)* | SQL used when `mode='db'` to fetch targets/proposals. |
| `inputDir`     | string (path)         | `'input'`       | Directory containing input files for this stage. |
| `outputDir`    | string (path)         | `'output/ppp'`  | Output directory for `ppp` artifacts. |

**Example**
```toml
[ppp]
mode = 'db'
localPath = ''
TEXP_NOMINAL = 900.0
sql_query = "SELECT ob_code,ra,dec,pmra,pmdec,parallax,epoch,priority,effective_exptime,is_medium_resolution,proposal.proposal_id,rank,grade FROM target JOIN proposal ON target.proposal_id=proposal.proposal_id;"
inputDir = 'input'
outputDir = 'output/ppp'
```

---

## `[qplan]`

| Parameter   | Allowed Values (Type) | Default          | Description |
|-------------|------------------------|------------------|-------------|
| `outputDir` | string (path)          | `'output/qplan'` | Output directory for quick-planning results. |

**Example**
```toml
[qplan]
outputDir = 'output/qplan'
```

---

## `[qplan.weight]`

Weights for the objective function in the planner.

| Parameter  | Allowed Values (Type) | Default | Description |
|------------|------------------------|---------|-------------|
| `slew`     | number (float)         | `0.2`   | Weight for slew (movement) cost. |
| `delay`    | number (float)         | `3.0`   | Weight for delay/overhead time. |
| `filter`   | number (float)         | `0.0`   | Penalty for filter changes (0 disables). |
| `rank`     | number (float)         | `0.85`  | Weight for target rank score. |
| `priority` | number (float)         | `0.1`   | Weight for proposal/target priority. |

**Example**
```toml
[qplan.weight]
slew = 0.2
delay = 3.0
filter = 0.0
rank = 0.85
priority = 0.1
```

---

## `[targetdb.db]`

Database connection settings for the target database.

| Parameter  | Allowed Values (Type) | Default         | Description |
|------------|------------------------|-----------------|-------------|
| `host`     | string (hostname)      | `"example.com"`| Database host. |
| `port`     | integer                | `5432`          | Database port. |
| `dbname`   | string                 | `"dbname"`     | Database name. |
| `user`     | string (username)      | `"username"`   | Database user. |
| `password` | string                 | `"password"`   | Database password (use secret storage in production). |
| `dialect`  | string                 | `"postgresql"` | SQL dialect/driver identifier. |

**Example**
```toml
[targetdb.db]
host = "example.com"
port = 5432
dbname = "dbname"
user = "username"
password = "password"
dialect = "postgresql"
```

---


# `[targetdb.sky]`

| Parameter | Allowed Values (Type) | Default        | Description |
|-----------|------------------------|----------------|-------------|
| `version` | array of strings       | `["20221031"]` | Sky catalog version(s) to use. |

**Example**
```toml
[targetdb.sky]
version = ["20221031"]
```

---


## `[targetdb.fluxstd]`

| Parameter | Allowed Values (Type) | Default  | Description |
|-----------|------------------------|----------|-------------|
| `version` | array of strings       | `["2.1"]` | Flux-standard catalog version(s). |

**Example**
```toml
[targetdb.fluxstd]
version = ["2.1"]
```

---


## `[gaiadb]`

Connection details for a Gaia (or Gaia-like) catalog database.

| Parameter  | Allowed Values (Type) | Default         | Description |
|------------|------------------------|-----------------|-------------|
| `host`     | string (hostname)      | `"example.com"`| Database host. |
| `port`     | integer                | `5432`          | Port. |
| `dbname`   | string                 | `"dbname"`     | DB name. |
| `user`     | string                 | `"username"`   | Username. |
| `password` | string                 | `"password"`   | Password. |

**Example**
```toml
[gaiadb]
host = "example.com"
port = 5432
dbname = "dbname"
user = "username"
password = "password"
```

---

## `[schemacrawler]`

| Parameter         | Allowed Values (Type) | Default                                       | Description |
|-------------------|------------------------|-----------------------------------------------|-------------|
| `SCHEMACRAWLERDIR`| string (path)          | `"../../../schemacrawler-16.15.7-distribution/"` | Path to SchemaCrawler distribution used for schema introspection or docs generation. |

**Example**
```toml
[schemacrawler]
SCHEMACRAWLERDIR = "../../../schemacrawler-16.15.7-distribution/"
```

---


## `[netflow]`

| Parameter    | Allowed Values (Type) | Default | Description |
|--------------|------------------------|---------|-------------|
| `use_gurobi` | boolean                | `true`  | Enable solving with Gurobi in the network-flow step. |

**Example**
```toml
[netflow]
use_gurobi = true
```

---

## `[gurobi]`

Solver controls for Gurobi. See your project’s usage for the precise meaning of each switch.

| Parameter         | Allowed Values (Type) | Default  | Description |
|-------------------|------------------------|----------|-------------|
| `seed`            | integer                | `0`      | Random seed for reproducibility. |
| `presolve`        | integer (0/1/2/auto)   | `1`      | Presolve level. |
| `method`          | integer (algorithm id) | `0`      | Algorithm selection (auto/method-specific depending on model type). |
| `degenmoves`      | integer                | `0`      | Degeneracy handling strategy. |
| `heuristics`      | number (0.0–1.0)       | `0.6`    | Heuristic effort level. |
| `mipfocus`        | integer (0–3)          | `0`      | MIP focus (balance vs. bound vs. feasible vs. optimality). |
| `mipgap`          | number (float, ≥0)     | `5.0e-3` | Relative MIP gap tolerance. |
| `PreSOS2Encoding` | integer (0/1)          | `0`      | Preprocessing toggle for SOS2 encoding. |
| `PreSOS1Encoding` | integer (0/1)          | `0`      | Preprocessing toggle for SOS1 encoding. |
| `threads`         | integer (≥1)           | `4`      | Maximum threads to use. |

**Example**
```toml
[gurobi]
seed = 0
presolve = 1
method = 0
degenmoves = 0
heuristics = 0.6
mipfocus = 0
mipgap = 5.0e-3
PreSOS2Encoding = 0
PreSOS1Encoding = 0
threads = 4
```

---

## `[sfa]`

Selection and facility assignment parameters (naming suggests sky/flux standard handling and instrument geometry).

| Parameter                      | Allowed Values (Type) | Default        | Description |
|--------------------------------|------------------------|----------------|-------------|
| `n_sky`                        | integer                | `100`          | Number of sky fibers/targets to sample. |
| `sky_random`                   | boolean                | `false`        | Randomize sky target selection. |
| `n_sky_random`                 | integer                | `1000`         | Candidate pool size when `sky_random=true`. |
| `reduce_sky_targets`           | boolean                | `true`         | Down-select sky targets to meet `n_sky`. |
| `pfs_instdata_dir`             | string (path)          | `"pfs_instdata"` | Instrument data directory. |
| `cobra_coach_dir`              | string (path)          | `"cobracoach"`   | CobraCoach data directory. |
| `cobra_coach_module_version`   | string or `"None"`    | `"None"`      | Specific CobraCoach module version tag; `"None"` to auto/select default. |
| `sm`                           | array of integers      | `[1, 2, 3, 4]` | Spectrograph modules to include. |
| `dot_margin`                   | number (float)         | `1.0`          | Margin for dot/collision checks (instrument geometry). |
| `dot_penalty`                  | string or `"None"`    | `"None"`      | Penalty profile for dot/collision; `"None"` disables additional penalty. |
| `arms`                         | string (e.g., `'brn'`) | `'brn'`        | Enabled arms; combination of `b`(blue), `r`(red), `n`(NIR). |
| `guidestar_mag_min`            | number (float)         | `12.0`         | Minimum magnitude (bright limit) for guide stars. |
| `guidestar_mag_max`            | number (float)         | `19.0`         | Maximum magnitude (faint limit) for guide stars. |
| `guidestar_neighbor_mag_min`   | number (float)         | `21.0`         | Neighbor magnitude threshold to avoid contamination. |
| `guidestar_minsep_deg`         | number (float, deg)    | `0.0002778`    | Minimum separation from neighbors (degrees). |
| `n_fluxstd`                    | integer                | `100`          | Number of flux standard stars to select. |
| `fluxstd_mag_max`              | number (float)         | `18.0`         | Faint limit for flux standards. |
| `fluxstd_mag_min`              | number (float)         | `15.0`         | Bright limit for flux standards. |
| `fluxstd_mag_filter`           | string                 | `"g"`         | Band/filter for flux standard magnitude selection. |
| `good_fluxstd`                 | boolean                | `false`        | Require high-quality flag for flux standards. |
| `fluxstd_min_prob_f_star`      | number (0–1)           | `0.5`          | Minimum stellar probability for flux standards. |
| `fluxstd_flags_dist`           | boolean                | `false`        | Exclude by distance flags. |
| `fluxstd_flags_ebv`            | boolean                | `false`        | Exclude by E(B–V) extinction flags. |
| `raster`                       | boolean                | `true`         | Enable raster pattern generation. |
| `raster_mag_min`               | number (float)         | `16.0`         | Bright limit for raster selection. |
| `raster_mag_max`               | number (float)         | `20.0`         | Faint limit for raster selection. |

**Example**
```toml
[sfa]
n_sky = 100
sky_random = false
n_sky_random = 1000
reduce_sky_targets = true
pfs_instdata_dir = "pfs_instdata"
cobra_coach_dir = "cobracoach"
cobra_coach_module_version = "None"
sm = [1, 2, 3, 4]
dot_margin = 1.0
dot_penalty = "None"
arms = 'brn'
guidestar_mag_min = 12.0
guidestar_mag_max = 19.0
guidestar_neighbor_mag_min = 21.0
guidestar_minsep_deg = 0.0002778
n_fluxstd = 100
fluxstd_mag_max = 18.0
fluxstd_mag_min = 15.0
fluxstd_mag_filter = "g"
good_fluxstd = false
fluxstd_min_prob_f_star = 0.5
fluxstd_flags_dist = false
fluxstd_flags_ebv = false
raster = true
raster_mag_min = 16.0
raster_mag_max = 20.0
```

---

## `[ope]`

| Parameter    | Allowed Values (Type) | Default                               | Description |
|--------------|------------------------|---------------------------------------|-------------|
| `template`   | string (path)          | `"templates/template_pfs_v2023-09-12.ope"` | OPE template file to populate. |
| `outfilePath`| string (path)          | `"output/ope"`                        | Output directory for generated OPE files. |
| `designPath` | string (path)          | `"output/design"`                     | Output directory for design artifacts. |
| `runName`    | string                 | `"example"`                           | Run name/prefix included in outputs. |

**Example**
```toml
[ope]
template = "templates/template_pfs_v2023-09-12.ope"
outfilePath = "output/ope"
designPath = "output/design"
runName = "example"
```

---
