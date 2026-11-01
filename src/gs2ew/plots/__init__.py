"""Plotting routines for GS2 output.

Importing this package registers every plot with :mod:`gs2ew.bundle`, so a
bundle can be replotted or reproduced by name.
"""

from gs2ew.plots import (  # noqa: F401
    fields,
    transfer_spectral,
    transfer_summary,
    transfer_theta,
    transfer_time,
    transfer_tp,
    weights,
)
