#!/usr/bin/env python3
"""Merge monthly Micromet Zarr stores into one CF NetCDF file.

Shortwave radiation is selected by default. Temperature can instead be merged
with ``--product temperature``. Input stores must share the same spatial grid.
Data remain lazy while the stores are opened and concatenated; Dask reads each
chunk only when the NetCDF file is written.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


OUTPUTS_DIR = Path(
    "/mnt/CEPH_PROJECTS/SNOWCOP/Dataset_eotraining/Micromet/outputs"
)
PRODUCTS = {
    "shortwave": {
        "directory": OUTPUTS_DIR / "SW",
        "prefix": "shortwave",
        "variable": "SW",
        "title": "Topographically corrected downscaled shortwave radiation",
    },
    "temperature": {
        "directory": OUTPUTS_DIR / "Temperature_biascorr",
        "prefix": "temperature",
        "variable": "t2m",
        "title": "Bias-corrected downscaled air temperature",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product",
        choices=PRODUCTS,
        default="shortwave",
        help="Micromet product to merge (default: shortwave)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing monthly stores (default: product's standard path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: <input-dir>/<product>_downscaled_<years>.nc)",
    )
    parser.add_argument(
        "--start",
        metavar="YYYY[-MM]",
        help="Optional first year or month to include, e.g. 2020 or 2020-04",
    )
    parser.add_argument(
        "--end",
        metavar="YYYY[-MM]",
        help="Optional last year or month to include, e.g. 2023 or 2023-12",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(1, 10),
        default=4,
        help="NetCDF DEFLATE compression level, 1-9 (default: 4)",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Allow missing calendar months (the default is to stop on gaps)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and validate inputs without writing the NetCDF file",
    )
    return parser.parse_args()


def parse_period(
    value: str | None, option: str, *, end_of_year: bool = False
) -> pd.Period | None:
    if value is None:
        return None
    if re.fullmatch(r"\d{4}", value):
        month = 12 if end_of_year else 1
        return pd.Period(f"{value}-{month:02d}", freq="M")
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise SystemExit(f"{option} must have the form YYYY or YYYY-MM: {value!r}")
    try:
        month = pd.Period(value, freq="M")
    except ValueError as exc:
        raise SystemExit(
            f"{option} must have the form YYYY or YYYY-MM: {value!r}"
        ) from exc
    if str(month) != value:
        raise SystemExit(f"{option} must have the form YYYY or YYYY-MM: {value!r}")
    return month


def discover_stores(
    input_dir: Path,
    prefix: str,
    start: pd.Period | None,
    end: pd.Period | None,
) -> list[tuple[pd.Period, Path]]:
    store_pattern = f"{prefix}_downscaled_????_??.zarr"
    store_re = re.compile(rf"{re.escape(prefix)}_downscaled_(\d{{4}})_(\d{{2}})\.zarr$")
    stores: list[tuple[pd.Period, Path]] = []
    for path in input_dir.glob(store_pattern):
        match = store_re.fullmatch(path.name)
        if match is None:
            continue
        month = pd.Period(f"{match.group(1)}-{match.group(2)}", freq="M")
        if (start is None or month >= start) and (end is None or month <= end):
            stores.append((month, path))

    stores.sort(key=lambda item: item[0])
    if not stores:
        raise FileNotFoundError(f"No stores matching {store_pattern!r} in {input_dir}")

    months = [month for month, _ in stores]
    if len(months) != len(set(months)):
        raise ValueError("More than one input store was found for the same month")
    return stores


def validate_month_sequence(months: list[pd.Period]) -> None:
    expected = list(pd.period_range(months[0], months[-1], freq="M"))
    missing = sorted(set(expected) - set(months))
    if missing:
        formatted = ", ".join(str(month) for month in missing)
        raise ValueError(f"Missing monthly store(s): {formatted}")


def open_and_merge(paths: list[Path], variable: str, title: str) -> xr.Dataset:
    print(f"Opening {len(paths)} monthly stores...")
    datasets = [
        xr.open_zarr(path, consolidated=True, chunks={"time": 30}) for path in paths
    ]
    merged = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        join="exact",
        combine_attrs="override",
    ).sortby("time")

    if variable not in merged:
        raise ValueError(
            f"The input stores do not contain the expected {variable!r} variable"
        )
    if tuple(merged[variable].dims) != ("time", "y", "x"):
        raise ValueError(
            f"Unexpected {variable} dimensions: {merged[variable].dims}"
        )

    times = pd.DatetimeIndex(merged.time.values)
    if times.hasnans:
        raise ValueError("The merged time coordinate contains missing values")
    duplicates = times[times.duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"Duplicate timestamps found: {duplicates.tolist()}")
    if not times.is_monotonic_increasing:
        raise ValueError("The merged time coordinate is not increasing")

    # Make the projected CRS discoverable by CF-aware and geospatial readers.
    if "spatial_ref" in merged:
        merged[variable].attrs["grid_mapping"] = "spatial_ref"
    merged.attrs.update(
        title=title,
        source="Monthly Micromet Zarr stores",
        history="Merged from monthly Zarr stores with merge_temperature_zarr.py",
    )
    return merged


def write_netcdf(
    dataset: xr.Dataset,
    variable: str,
    output: Path,
    level: int,
    overwrite: bool,
) -> None:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output} (use --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"Incomplete output already exists: {temporary}")

    # Discard Zarr-specific encodings before supplying NetCDF encodings.
    for array in dataset.variables.values():
        array.encoding.clear()
    y_size, x_size = dataset.sizes["y"], dataset.sizes["x"]
    encoding = {
        variable: {
            "dtype": "float32",
            "zlib": True,
            "complevel": level,
            "shuffle": True,
            "chunksizes": (min(30, dataset.sizes["time"]), y_size, x_size),
        },
        "time": {
            "dtype": "int32",
            "units": "days since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        },
    }

    print(f"Writing {output} ...")
    try:
        dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        with xr.open_dataset(temporary, engine="netcdf4") as check:
            if check.sizes != dataset.sizes:
                raise RuntimeError(
                    f"Output dimensions differ: {dict(check.sizes)} != {dict(dataset.sizes)}"
                )
            if not np.array_equal(check.time.values, dataset.time.values):
                raise RuntimeError("Output time coordinate differs from the input")
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    size_mib = output.stat().st_size / 1024**2
    print(f"Created {output} ({size_mib:.1f} MiB)")


def main() -> None:
    args = parse_args()
    product = PRODUCTS[args.product]
    input_dir = args.input_dir or product["directory"]
    start = parse_period(args.start, "--start")
    end = parse_period(args.end, "--end", end_of_year=True)
    if start is not None and end is not None and start > end:
        raise SystemExit("--start must be earlier than or equal to --end")

    stores = discover_stores(input_dir, product["prefix"], start, end)
    months = [month for month, _ in stores]
    if not args.allow_gaps:
        validate_month_sequence(months)

    print(f"Selected {len(stores)} stores from {months[0]} through {months[-1]}")
    merged = open_and_merge(
        [path for _, path in stores], product["variable"], product["title"]
    )
    print(
        f"Validated {merged.sizes['time']} timesteps on a "
        f"{merged.sizes['y']} x {merged.sizes['x']} grid"
    )
    if args.dry_run:
        print("Dry run complete; no output was written.")
        return

    output = args.output
    if output is None:
        output = input_dir / (
            f"{product['prefix']}_downscaled_"
            f"{months[0].year}_{months[-1].year}.nc"
        )
    write_netcdf(
        merged,
        product["variable"],
        output,
        args.compression_level,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
