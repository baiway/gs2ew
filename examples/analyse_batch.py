"""Batch post-processing on ARCHER2.

This is now a thin wrapper: `gs2ew batch` does the walking, plotting and
error handling. Kept as a worked example of the SLURM script around it.

Usage
-----
    gs2ew batch <project_root> -o <output_root> --skip cbc --movies

Submit with:

```
#!/bin/bash

#SBATCH --job-name=analysis
#SBATCH --output=analysis%j.log
#SBATCH --time=2:00:00        # Movies dominate the runtime; lower this once
                              # you know how long the batch actually takes.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# `e281` is the Plasma Physics Consortium project code
#SBATCH --account=e281
#SBATCH --partition=serial
#SBATCH --qos=serial

VENV_PATH="/work/e281/e281/bc1264/gk/gs2ew/.venv"
PROJECT_ROOT="/work/e281/e281/bc1264/tobias-reproduce-v2"

source "${VENV_PATH}/bin/activate"

gs2ew batch "${PROJECT_ROOT}" \
    -o "${PROJECT_ROOT}/analysis" \
    --skip cbc \
    --tstart saturation \
    --klim 0.6
```

Outputs land under `<output_root>/<case>`, mirroring each run's path relative
to the project root, so runs that share an input-file name do not collide.
The CLI selects a headless matplotlib backend for you.
"""

import sys

from gs2ew.cli import main

if __name__ == "__main__":
    sys.exit(main(["batch", *sys.argv[1:]]))
