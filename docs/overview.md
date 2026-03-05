# Overview: 


PFS Obsproc Planning Tool ("Integration code") is an integrated software that handles the entire workflow from loading related PFS databases to generating observation plans and producing fiber‑allocation designs (pfsDesign files). This document provides an overview of the core script of the system, `generatePfsDesign.py`.


## Main Workflow of `generatePfsDesign.py`


The script performs the following key steps:

1. **Configuration and Setup**
	- Loads configuration parameters from a TOML file.
	- Sets up working directories for input, output, and intermediate results.
	- Checks and manages versions of dependent packages and repositories.

2. **PPP (PFS Preparation Pipeline)**
	- Selects and prioritizes science targets from databases or local files.
	- Allocates fibers to targets, calibrators, and fillers according to configuration.
	- Outputs target allocation results for subsequent planning.

3. **Queue Planning (qPlan)**
	- Schedules observations by optimizing the sequence and timing of pointings.
	- Considers constraints such as overheads, visibility, and priorities.
	- Identifies gaps in the schedule and can trigger backup planning if needed.

4. **SFA (Spectrograph Fiber Assignment)**
	- Assigns fibers to science targets, sky, and flux calibrators for each pointing.
	- Generates detailed assignment tables and summary CSVs.
	- Prepares data for design file generation.

5. **Design and OPE File Generation**
	- Creates PFS design files and OPE (Observation Preparation Environment) files for each scheduled observation.
	- Uses templates and configuration to ensure files are ready for instrument operations.

6. **Validation**
	- Runs validation routines to check the integrity and quality of the generated plans and files.
	- Produces diagnostic plots and summary reports.

## Usage

The script can be run as a standalone command-line tool or imported as a module. Please follow the instruction in [README.md](../README.md). Typical usage involves specifying the working directory, configuration file, and desired number of pointings:

```shell
python src/pfs_obsproc_planning/generatePfsDesign.py --workDir <workdir> --config config.toml --obs_dates 2026-01-17
```

Or programmatically:

```python
from pfs_obsproc_planning import generatePfsDesign

gpd = generatePfsDesign.GeneratePfsDesign("config.toml", workDir="workdir_example")
gpd.runPPP(n_pccs_l=5, n_pccs_m=0)
gpd.runQPlan(obs_dates=["2026-01-17"])
gpd.runSFA(clearOutput=True)
```

## Inputs and Outputs

- **Inputs:** Configuration file (`config.toml`), target lists, pointing lists, and database connections.
- **Outputs:** Target allocation tables, observation schedules, PFS design files, OPE files, and validation plots.

## Modular Design

The script is organized into modular methods:
- `runPPP()`: Determination of pointing centers.
- `runQPlan()`: Executes the queue planner to create observation schedule.
- `runSFA()`: Performs fiber assignment and generates design files.
- `runValidation()`: Validates the outputs.

Each step can be run independently or as part of the full workflow.

## Customization

- The workflow is highly configurable via the TOML config file.
- Supports both queue and classical observation modes.
- Allows for backup planning and flexible scheduling.

## References

- See the [README.md](../README.md) for installation and setup instructions.
- Example configuration files and work directory templates are provided in the `examples/` folder.
- For detailed parameter descriptions, refer to the comments in the config file and the script source.

---
