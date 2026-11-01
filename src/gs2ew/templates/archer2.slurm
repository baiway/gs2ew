#!/bin/bash
# Run GS2 on ARCHER2. Copy this into the run directory, then from there:
#
#   sbatch --job-name=NAME archer2.slurm INPUT.in
#
# GS2 writes its output beside INPUT.in, and its screen output goes to
# screen_output.txt. Override the resources on the command line rather than
# editing this file, e.g. `sbatch --nodes=2 --time=02:00:00 ...`.

#SBATCH --job-name=gs2
#SBATCH --output=%x-%j.log
#SBATCH --time=00-24:00:00              # Time limit (DD-HH:MM:SS)
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=128           # 6 * 128 = 768 cores
#SBATCH --cpus-per-task=1

# `e281-ypimcf` is the Plasma Physics Consortium project code;
# use `e281` when resource reallocated
#SBATCH --account=e281-ypimcf
#SBATCH --partition=standard
#SBATCH --qos=standard

# The compute nodes can't see /home, so both of these must be under /work.
# The environment file is extras/gs2-archer2.sh, which loads the modules GS2
# was compiled with.
GS2_EXEC="${GS2_EXEC:-/work/e281/e281/bc1264/gk/gs2/bin/gs2}"
GS2_ENV="${GS2_ENV:-/work/e281/e281/bc1264/.gs2.sh}"

input_file="${1:?usage: sbatch archer2.slurm INPUT.in}"

source "$GS2_ENV"
# After the modules, which can fail harmlessly. pipefail so a failed srun
# isn't hidden by tee succeeding.
set -eo pipefail

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
srun --distribution=block:block --hint=nomultithread \
    "$GS2_EXEC" "$input_file" | tee screen_output.txt
