#!/usr/bin/env python3
"""Merge monthly daily ERA5 files into one STAC-ready compressed NetCDF.

The source files are expected to follow ``era_YYYY_MM_daily.nc``. By default,
all available files are merged. ``--start`` and ``--end`` accept either a year
or a year-month, for example ``--start 2020 --end 2023``.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_INPUT = Path(
    "/mnt/CEPH_PROJECTS/SNOWCOP/Dataset_eotraining/Micromet/inputs/climate"
)
FILE_PATTERN = "era_????_??_daily.nc"
FILE_RE = re.compile(r"era_(\d{4})_(\d{2})_daily\.nc$")
SOURCE_TIME_DIMENSION = "valid_time"
TIME_DIMENSION = "time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory containing monthly files (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: <input-dir>/era5_daily_<years>.nc)",
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
        "--variables",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Variables to retain, e.g. --variables tp t2m (default: all)",
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
        help="Allow missing months or days (the default is to stop on gaps)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Open and validate inputs without writing the output",
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
        period = pd.Period(value, freq="M")
    except ValueError as exc:
        raise SystemExit(
            f"{option} must have the form YYYY or YYYY-MM: {value!r}"
        ) from exc
    if str(period) != value:
        raise SystemExit(f"{option} must have the form YYYY or YYYY-MM: {value!r}")
    return period


def discover_files(
    input_dir: Path,
    start: pd.Period | None,
    end: pd.Period | None,
) -> list[tuple[pd.Period, Path]]:
    files: list[tuple[pd.Period, Path]] = []
    for path in input_dir.glob(FILE_PATTERN):
        match = FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        month = pd.Period(f"{match.group(1)}-{match.group(2)}", freq="M")
        if (start is None or month >= start) and (end is None or month <= end):
            files.append((month, path))

    files.sort(key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(f"No files matching {FILE_PATTERN!r} in {input_dir}")

    months = [month for month, _ in files]
    if len(months) != len(set(months)):
        raise ValueError("More than one ERA5 file was found for the same month")
    return files


def validate_month_sequence(months: list[pd.Period]) -> None:
    expected = set(pd.period_range(months[0], months[-1], freq="M"))
    missing = sorted(expected - set(months))
    if missing:
        raise ValueError(
            "Missing monthly file(s): " + ", ".join(str(month) for month in missing)
        )


def open_and_merge(
    paths: list[Path],
    allow_gaps: bool,
    variables: list[str] | None = None,
) -> xr.Dataset:
    print(f"Opening {len(paths)} monthly ERA5 files...")
    dataset = xr.open_mfdataset(
        paths,
        engine="netcdf4",
        decode_coords="all",
        combine="nested",
        concat_dim=SOURCE_TIME_DIMENSION,
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        join="exact",
        combine_attrs="override",
        chunks={SOURCE_TIME_DIMENSION: 31},
    ).sortby(SOURCE_TIME_DIMENSION)

    required_dimensions = {SOURCE_TIME_DIMENSION, "latitude", "longitude"}
    missing_dimensions = required_dimensions - set(dataset.dims)
    if missing_dimensions:
        raise ValueError(
            f"Missing expected dimension(s): {sorted(missing_dimensions)}"
        )

    times = pd.DatetimeIndex(dataset[SOURCE_TIME_DIMENSION].values)
    if times.hasnans:
        raise ValueError("The merged time coordinate contains missing values")
    duplicates = times[times.duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"Duplicate timestamps found: {duplicates.tolist()}")
    if not times.is_monotonic_increasing:
        raise ValueError("The merged time coordinate is not increasing")

    if not allow_gaps:
        expected_times = pd.date_range(times[0], times[-1], freq="D")
        missing_times = expected_times.difference(times)
        if len(missing_times):
            preview = ", ".join(str(value.date()) for value in missing_times[:10])
            suffix = " ..." if len(missing_times) > 10 else ""
            raise ValueError(f"Missing daily timestep(s): {preview}{suffix}")

    source_time_variables = [
        name
        for name, array in dataset.data_vars.items()
        if SOURCE_TIME_DIMENSION in array.dims
    ]
    if not source_time_variables:
        raise ValueError("No time-varying ERA5 variables were found")

    if variables:
        variables = list(dict.fromkeys(variables))
        unknown = sorted(set(variables) - set(source_time_variables))
        if unknown:
            raise ValueError(
                f"Unknown requested variable(s): {unknown}. Available variables: "
                f"{source_time_variables}"
            )
        dataset = dataset[variables]

    # raster2stac detects these conventional dimension names. In particular,
    # it does not recognize ERA5's native ``valid_time`` as temporal.
    dataset = dataset.rename(
        {
            SOURCE_TIME_DIMENSION: TIME_DIMENSION,
            "latitude": "y",
            "longitude": "x",
        }
    )
    dataset.attrs.update(
        title="Daily ERA5 meteorological data for Lo Aguirre",
        source="Merged monthly ERA5 NetCDF files",
        history="Merged with merge_era5_netcdf.py",
        openeo_x_dim="x",
        openeo_y_dim="y",
        openeo_temporal_dims=[TIME_DIMENSION],
        openeo_band_dims=["bands"],
        openeo_other_dims=[],
    )
    return dataset


def netcdf_encoding(dataset: xr.Dataset, level: int) -> dict[str, dict]:
    encoding: dict[str, dict] = {}
    for name, array in dataset.data_vars.items():
        if TIME_DIMENSION not in array.dims:
            continue
        chunksizes = tuple(
            min(31, dataset.sizes[dimension])
            if dimension == TIME_DIMENSION
            else dataset.sizes[dimension]
            for dimension in array.dims
        )
        encoding[name] = {
            "zlib": True,
            "complevel": level,
            "shuffle": True,
            "chunksizes": chunksizes,
        }

    encoding[TIME_DIMENSION] = {
        "dtype": "int32",
        "units": "days since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
    }
    return encoding


def write_netcdf(
    dataset: xr.Dataset,
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

    # Remove source-file chunking and compression settings before defining the
    # encoding of the merged output.
    for array in dataset.variables.values():
        array.encoding.clear()

    print(f"Writing {output} ...")
    try:
        dataset.to_netcdf(
            temporary,
            engine="netcdf4",
            encoding=netcdf_encoding(dataset, level),
        )
        with xr.open_dataset(temporary, engine="netcdf4") as check:
            if check.sizes != dataset.sizes:
                raise RuntimeError(
                    f"Output dimensions differ: {dict(check.sizes)} != "
                    f"{dict(dataset.sizes)}"
                )
            if not np.array_equal(
                check[TIME_DIMENSION].values,
                dataset[TIME_DIMENSION].values,
            ):
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
    start = parse_period(args.start, "--start")
    end = parse_period(args.end, "--end", end_of_year=True)
    if start is not None and end is not None and start > end:
        raise SystemExit("--start must be earlier than or equal to --end")

    files = discover_files(args.input_dir, start, end)
    months = [month for month, _ in files]
    if not args.allow_gaps:
        validate_month_sequence(months)

    print(f"Selected {len(files)} files from {months[0]} through {months[-1]}")
    dataset = open_and_merge(
        [path for _, path in files],
        allow_gaps=args.allow_gaps,
        variables=args.variables,
    )
    variables = [
        name
        for name, array in dataset.data_vars.items()
        if TIME_DIMENSION in array.dims
    ]
    print(
        f"Validated {dataset.sizes[TIME_DIMENSION]} daily timesteps; "
        f"variables: {', '.join(variables)}"
    )
    if args.dry_run:
        print("Dry run complete; no output was written.")
        return

    output = args.output
    if output is None:
        output = args.input_dir / (
            f"era5_daily_{months[0].year}_{months[-1].year}.nc"
        )
    write_netcdf(dataset, output, args.compression_level, args.overwrite)


if __name__ == "__main__":
    main()
