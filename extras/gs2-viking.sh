# Set GS2 environment variables
export MAKEFLAGS=-IMakefiles
export GK_SYSTEM=viking

# Modules to compile GS2 (working as of 29th May 2025)
module purge
module load Python/3.12.3-GCCcore-13.3.0    # Python (for tests)
module load foss/2023b                      # GCC, incl. MPI, OpenMP, FFTW and BLAS
module load HDF5/1.14.3-gompi-2023b         # Parallel HDF5
module load netCDF/4.9.2-gompi-2023b        # NetCDF
module load netCDF-Fortran/4.6.1-gompi-2023b
module load git                             # Git
