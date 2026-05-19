#!/usr/bin/env python3
"""Download one of the three benchmark dataset suites from Kaggle.

The three suites match the paper's experimental design:
  IIS3D     : Integrated IDS Security 3 Datasets
  ICS3D     : Integrated Cloud Security 3 Datasets
  IDS-PQC   : IDS Encrypted PQC Datasets

Usage
-----
    python scripts/download_datasets.py --suite iis3d --out ./data/iis3d
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import click

try:
    from ssl_graph_anomaly.utils import get_logger
except Exception:  # pragma: no cover - utils may be partially loaded
    import logging

    def get_logger(name: str):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


LOG = get_logger(__name__)

KAGGLE_SLUGS = {
    "iis3d": "rogerpanel/integrated-idps-security-3-datasets",
    "ics3d": "rogerpanel/integrated-cloud-security-3-datasets",
    "ids_pqc": "rogerpanel/ids-encrypted-pqc-datasets",
}


def _check_kaggle_available() -> bool:
    if shutil.which("kaggle") is None:
        LOG.error("`kaggle` CLI not found on PATH.")
        click.echo(
            "\nThe Kaggle CLI is required to download these suites.\n"
            "Install via:    pip install kaggle\n"
            "Authenticate by placing your kaggle.json in ~/.kaggle/ "
            "(see https://github.com/Kaggle/kaggle-api for details).\n"
            "Then re-run this script.",
            err=True,
        )
        return False
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        LOG.warning(
            "~/.kaggle/kaggle.json not found; API call will likely fail."
        )
    return True


def _kaggle_download(slug: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)]
    LOG.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise click.ClickException(
            f"kaggle download failed with code {result.returncode}"
        )
    zips = list(dest.glob("*.zip"))
    if not zips:
        raise click.ClickException("No zip archive found after download.")
    return zips[0]


def _unzip(archive: Path, dest: Path) -> None:
    LOG.info("Unzipping %s -> %s", archive, dest)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)
    archive.unlink()


def _summarise(dest: Path) -> None:
    files = sorted(p for p in dest.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    LOG.info("Downloaded %d files (%.2f MB) to %s",
             len(files), total_bytes / (1024 ** 2), dest)
    for p in files[:20]:
        rel = p.relative_to(dest)
        LOG.info("  %s (%.1f KB)", rel, p.stat().st_size / 1024.0)
    if len(files) > 20:
        LOG.info("  ... and %d more files", len(files) - 20)


@click.command()
@click.option(
    "--suite",
    type=click.Choice(list(KAGGLE_SLUGS.keys()), case_sensitive=False),
    required=True,
    help="Which benchmark suite to download.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Destination directory.",
)
def main(suite: str, out: Path) -> None:
    """Download a benchmark suite from Kaggle and unpack it."""
    suite = suite.lower()
    slug = KAGGLE_SLUGS[suite]
    LOG.info("Downloading suite=%s slug=%s -> %s", suite, slug, out)

    if not _check_kaggle_available():
        sys.exit(2)

    archive = _kaggle_download(slug, out)
    _unzip(archive, out)
    _summarise(out)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
