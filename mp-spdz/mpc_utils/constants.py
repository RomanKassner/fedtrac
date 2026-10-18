"""Shared constants for MPC programs."""

START_ACT = 1
END_ACT = 2

# Public width the edge filter compares aggregated DFG cells at; must match AGG_CELL_BITS in src/constants.rs.
AGG_CELL_BITS = 41

# Fixed-point scale of the in-MPC L2 row normalisation; must match NORM_SCALE in src/constants.rs.
NORM_SCALE = 1 << 17
