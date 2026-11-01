#!/usr/bin/env bash
# Run GS2 on Viking. Copy this into the run directory, then from there:
#
#   sbatch --job-name=NAME viking.slurm INPUT.in
#
# GS2 writes its output beside INPUT.in, and its screen output goes to
# screen_output.txt. Override the resources on the command line rather than
# editing this file, e.g. `sbatch --ntasks=256 --time=02:00:00 ...`. For
# email when the job ends, add `--mail-user=ADDRESS`.

#SBATCH --job-name=gs2
#SBATCH --partition=nodes
#SBATCH --time=00-12:00:00              # Time limit (DD-HH:MM:SS)
#SBATCH --ntasks=512                    # Number of MPI tasks
#SBATCH --cpus-per-task=1               # CPU cores per MPI task
#SBATCH --mem-per-cpu=5200              # In megabytes
#SBATCH --account=pet-gspt-2019
#SBATCH --mail-type=END,FAIL
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

# The environment file is extras/gs2-viking.sh, which loads the modules GS2
# was compiled with.
GS2_EXEC="${GS2_EXEC:-$HOME/scratch/gk/gs2/bin/gs2}"
GS2_ENV="${GS2_ENV:-$HOME/.gs2.sh}"

input_file="${1:?usage: sbatch viking.slurm INPUT.in}"

source "$GS2_ENV"
# After the modules, which can fail harmlessly. pipefail so a failed srun
# isn't hidden by tee succeeding.
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
srun "$GS2_EXEC" "$input_file" | tee screen_output.txt
