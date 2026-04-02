"""Inspect the GS2 build metadata stored in a .out.nc file."""

import xarray as xr
from gs2ew.utils.gs2_metadata import get_metadata, get_version, requires_version

NC_FILE = "results.out.nc"

with xr.open_dataset(NC_FILE) as ds:
    # ── Full metadata summary ──────────────────────────────────────────────
    print("=== Metadata ===")
    for key, value in get_metadata(ds).items():
        print(f"  {key}: {value}")

    # ── Parsed version ─────────────────────────────────────────────────────
    print()
    version = get_version(ds)
    print(f"=== Parsed version ===")
    print(f"  {version}")
    print(f"  major={version.major}, minor={version.minor}, patch={version.patch}")
    print(f"  commits_since_tag={version.commits_since_tag}")
    print(f"  commit_hash={version.commit_hash!r}")
    print(f"  branch={version.branch!r}")
    print(f"  git_state={version.git_state!r}")

    # ── Version comparison examples ────────────────────────────────────────
    print()
    print("=== Version comparisons ===")

    for example in ["8.1.0", "8.2.0", "8.2.1", "8.2.1-500-gabcdef123", "9.0.0"]:
        from gs2ew.utils.gs2_metadata import _parse_version_string
        parsed = _parse_version_string(example)
        print(f"  file version > {example}? {version > parsed}")

    # ── requires_version guard ─────────────────────────────────────────────
    print()
    print("=== requires_version guard ===")
    try:
        requires_version(ds, "8.1.0")
        print("  requires_version(ds, '8.1.0') → passed")
    except RuntimeError as e:
        print(f"  requires_version(ds, '8.1.0') → {e}")

    try:
        requires_version(ds, "9.0.0")
        print("  requires_version(ds, '9.0.0') → passed")
    except RuntimeError as e:
        print(f"  requires_version(ds, '9.0.0') → {e}")
