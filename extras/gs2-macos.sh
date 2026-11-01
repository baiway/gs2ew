# Set `GK_SYSTEM` and `MAKEFLAGS` for compiling GS2 on an Apple-silicon Mac.
# Set -j to the number of cores you want make to use.
export GK_SYSTEM=macos-M-series MAKEFLAGS='-j 6 -I Makefiles'
