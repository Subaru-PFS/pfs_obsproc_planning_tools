#!/usr/bin/env python3
"""Manage PFS-related package installations based on a TOML config file.

Subcommands:
    sync      Check versions and install/update packages as needed.
    validate  Check versions and report status without installing.

Usage:
    NOTE: Do NOT use `uv run` without --no-sync, as uv will overwrite package
    versions from uv.lock before/after execution. Use one of the following:

    # Option 1: venv Python directly (recommended)
    .venv/bin/python scripts/pfs_packages.py sync examples/package_config_example.toml
    .venv/bin/python scripts/pfs_packages.py sync examples/package_config_example.toml --force
    .venv/bin/python scripts/pfs_packages.py sync examples/package_config_example.toml --dry-run
    .venv/bin/python scripts/pfs_packages.py validate examples/package_config_example.toml
    .venv/bin/python scripts/pfs_packages.py validate examples/package_config_example.toml --strict

    # Option 2: uv run with --no-sync
    uv run --no-sync python scripts/pfs_packages.py sync examples/package_config_example.toml
    uv run --no-sync python scripts/pfs_packages.py validate examples/package_config_example.toml
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from loguru import logger

# Mapping: TOML key (without _ver suffix) -> (pip package name, GitHub repo URL)
PACKAGE_REGISTRY: dict[str, tuple[str, str]] = {
    "pfs_datamodel": (
        "pfs-datamodel",
        "https://github.com/Subaru-PFS/datamodel.git",
    ),
    "pfs_utils": (
        "pfs-utils",
        "https://github.com/Subaru-PFS/pfs_utils.git",
    ),
    "ics_cobraCharmer": (
        "ics_cobraCharmer",
        "https://github.com/Subaru-PFS/ics_cobraCharmer.git",
    ),
    "ics_cobraOps": (
        "ics-cobraOps",
        "https://github.com/Subaru-PFS/ics_cobraOps.git",
    ),
    "ets_fiberalloc": (
        "ets-fiber-assigner",
        "https://github.com/Subaru-PFS/ets_fiberalloc.git",
    ),
    "pfs_instdata": (
        "pfs_instdata",
        "https://github.com/Subaru-PFS/pfs_instdata.git",
    ),
    "ets_pointing": (
        "pfs_design_tool",
        "https://github.com/Subaru-PFS/ets_pointing.git",
    ),
    "pfs_obsproc_planning": (
        "pfs_obsproc_planning",
        "https://github.com/Subaru-PFS/pfs_obsproc_planning_tools.git",
    ),
    "ets_shuffle": (
        "ets_shuffle",
        "https://github.com/Subaru-PFS/ets_shuffle.git",
    ),
    "ets_target_database": (
        "ets_target_database",
        "https://github.com/Subaru-PFS/ets_target_database.git",
    ),
    "spt_operational_database": (
        "opdb",
        "https://github.com/Subaru-PFS/spt_operational_database.git",
    ),
    "qplan": (
        "qplan",
        "https://github.com/naojsoft/qplan.git",
    ),
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_installed_info(pip_name: str) -> dict | None:
    """Return installation info for a package, or None if not installed.

    Returns a dict with keys:
      - "version": the declared version string
      - "revision": the git requested_revision (if installed from git), or None
      - "commit_id": the full git commit id (if installed from git), or None
    """
    try:
        dist = importlib.metadata.distribution(pip_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    info: dict = {
        "version": dist.metadata["Version"],
        "revision": None,
        "commit_id": None,
    }

    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        try:
            data = json.loads(direct_url_text)
            vcs_info = data.get("vcs_info", {})
            info["revision"] = vcs_info.get("requested_revision")
            info["commit_id"] = vcs_info.get("commit_id")
        except json.JSONDecodeError:
            pass

    return info


def is_version_match(info: dict, required: str) -> bool:
    """Return True if the installed package matches the required version/commit."""
    if info["revision"] is not None and info["revision"] == required:
        return True
    if info["commit_id"] is not None:
        commit_id: str = info["commit_id"]
        if commit_id == required or commit_id.startswith(required):
            return True
    return info["version"] == required


def get_remote_head_commit(url: str) -> str | None:
    """Return the current HEAD commit hash of the remote repo's default branch."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", url, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        line = result.stdout.strip()
        if line:
            return line.split("\t")[0]
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to query remote HEAD for {}: {}", url, e)
    return None


def load_packages(config_path: Path) -> dict[str, str]:
    """Load and return the [packages] section from the TOML config."""
    if not config_path.exists():
        logger.error("Config file not found: {}", config_path)
        sys.exit(1)
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    packages = config.get("packages", {})
    if not packages:
        logger.warning("No [packages] section found in {}", config_path)
    return packages


# ---------------------------------------------------------------------------
# install subcommand
# ---------------------------------------------------------------------------


def install_package(pip_name: str, url: str, version: str | None, dry_run: bool) -> None:
    """Install a package from a git URL at the specified version/tag/commit.

    If version is None or empty, install from the default branch HEAD.
    """
    git_url = f"git+{url}@{version}" if version else f"git+{url}"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        git_url,
    ]
    logger.info("Installing {} from {} ...", pip_name, git_url)
    if dry_run:
        logger.info("[dry-run] Would run: {}", " ".join(cmd))
        return
    try:
        subprocess.run(cmd, check=True)
        logger.success("Successfully installed {}", pip_name)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install {}: {}", pip_name, e)
        sys.exit(1)


def cmd_sync(args: argparse.Namespace) -> None:
    packages = load_packages(args.config)

    for key, version in packages.items():
        pkg_key = key.removesuffix("_ver")

        if pkg_key not in PACKAGE_REGISTRY:
            logger.warning("Unknown package '{}' — not in PACKAGE_REGISTRY, skipping", pkg_key)
            continue

        pip_name, repo_url = PACKAGE_REGISTRY[pkg_key]

        if not version:
            # No version pinned — update only if already installed and behind remote HEAD
            logger.info(
                "{}: no version specified, checking against remote default branch HEAD ...",
                pkg_key,
            )
            remote_commit = get_remote_head_commit(repo_url)
            if remote_commit is None:
                logger.warning("Could not determine remote HEAD for {}, skipping", pkg_key)
                continue
            info = get_installed_info(pip_name)
            if info is None:
                logger.info("{} is not installed, skipping", pip_name)
                continue
            if not args.force and info["commit_id"] == remote_commit:
                logger.info("{} is already at remote HEAD ({})", pip_name, remote_commit[:12])
                continue
            if not args.force:
                logger.info(
                    "{} is not at remote HEAD (installed: {}, remote: {})",
                    pip_name,
                    info["commit_id"][:12] if info["commit_id"] else "unknown",
                    remote_commit[:12],
                )
            install_package(pip_name, repo_url, version=None, dry_run=args.dry_run)
            continue

        if not args.force:
            info = get_installed_info(pip_name)
            if info is None:
                logger.info("{} is not installed", pip_name)
            elif is_version_match(info, version):
                logger.info(
                    "{} is already up-to-date (required: {}, revision: {}, commit: {})",
                    pip_name,
                    version,
                    info["revision"],
                    info["commit_id"],
                )
                continue
            else:
                logger.info(
                    "{} version mismatch (required: {}, installed revision: {}, commit: {})",
                    pip_name,
                    version,
                    info["revision"],
                    info["commit_id"],
                )

        install_package(pip_name, repo_url, version, dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# validate subcommand
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> None:
    packages = load_packages(args.config)

    issues: list[str] = []

    for key, version in packages.items():
        pkg_key = key.removesuffix("_ver")

        if pkg_key not in PACKAGE_REGISTRY:
            logger.warning("Unknown package '{}' — not in PACKAGE_REGISTRY, skipping", pkg_key)
            continue

        pip_name, repo_url = PACKAGE_REGISTRY[pkg_key]
        info = get_installed_info(pip_name)

        if not version:
            # No version pinned — compare against remote HEAD if installed
            if info is None:
                logger.info("{}: not installed, skipping (no version pinned)", pip_name)
                continue
            logger.info(
                "{}: no version specified, checking against remote default branch HEAD ...",
                pkg_key,
            )
            remote_commit = get_remote_head_commit(repo_url)
            if remote_commit is None:
                logger.warning("Could not determine remote HEAD for {}, skipping", pkg_key)
                continue
            if info["commit_id"] == remote_commit:
                logger.success("{}: OK (at remote HEAD {})", pip_name, remote_commit[:12])
            else:
                msg = (
                    f"{pip_name}: MISMATCH"
                    f" (installed: {info['commit_id'][:12] if info['commit_id'] else 'unknown'},"
                    f" remote HEAD: {remote_commit[:12]})"
                )
                logger.error(msg)
                issues.append(msg)
            continue

        if info is None:
            msg = f"{pip_name}: NOT INSTALLED (required: {version})"
            if args.strict:
                logger.error(msg)
                issues.append(msg)
            else:
                logger.warning(msg)
        elif is_version_match(info, version):
            logger.success(
                "{}: OK (required: {}, revision: {}, commit: {})",
                pip_name,
                version,
                info["revision"],
                info["commit_id"],
            )
        else:
            msg = (
                f"{pip_name}: MISMATCH"
                f" (required: {version},"
                f" installed revision: {info['revision']},"
                f" commit: {info['commit_id']})"
            )
            logger.error(msg)
            issues.append(msg)

    if issues:
        logger.error("{} issue(s) found.", len(issues))
        sys.exit(1)
    else:
        logger.success("All checked packages are up-to-date.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage PFS package installations from a TOML config file."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sync
    p_sync = subparsers.add_parser("sync", help="Sync packages to versions specified in config.")
    p_sync.add_argument("config", type=Path, help="Path to the TOML config file")
    p_sync.add_argument(
        "--force", action="store_true", help="Reinstall all packages regardless of current version"
    )
    p_sync.add_argument(
        "--dry-run", action="store_true", help="Show what would be installed without installing"
    )
    p_sync.set_defaults(func=cmd_sync)

    # validate
    p_validate = subparsers.add_parser(
        "validate", help="Check package versions and report status."
    )
    p_validate.add_argument("config", type=Path, help="Path to the TOML config file")
    p_validate.add_argument(
        "--strict",
        action="store_true",
        help="Treat NOT_INSTALLED as an error (exits with code 1)",
    )
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
