# Set `GK_SYSTEM` and `MAKEFLAGS` for compiling GS2
export GK_SYSTEM="archer2" MAKEFLAGS="-I Makefiles"

module load PrgEnv-gnu
module load cray-python/3.10.10
module load cray-fftw/3.3.10.5
module load cray-hdf5/1.12.2.7
module load cray-netcdf/4.9.0.7

export TESTEXEC="srun -n 128 --hint=nomultithread --distribution=block:block"
