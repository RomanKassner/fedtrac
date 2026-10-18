"""Dense trigram encoding over the user-activity alphabet: consecutive triples of real events, no START/END wrap."""

from Compiler.types import sint

from .constants import END_ACT
from .encoding_bigram import demux_histogram

# First user-activity id; PAD, START and END are reserved in src/types.rs.
FIRST_USER_ACT = END_ACT + 1


def user_dim(dict_size):
    """Trigram cube alphabet size: the user activities only."""
    return dict_size - FIRST_USER_ACT


def extract_features(sort_keys, sort_vals, max_real, dict_size, pad_sentinel_val,
                     output_row):
    """Accumulate the dense trigram counts into output_row."""
    du = user_dim(dict_size)
    du_cubed = du * du * du
    du_sq = du * du

    if max_real < 3:
        return

    # The remap parks pads at index 0 with zero mass; a slot carrying a reserved id would fold live mass.
    is_real = [sort_keys[k] != sint(pad_sentinel_val) for k in range(max_real)]
    av = [sort_vals[k] - FIRST_USER_ACT * is_real[k] for k in range(max_real)]


    tri_idx = []
    tri_val = []
    for k in range(max_real - 2):
        tri_idx.append(av[k] * du_sq + av[k + 1] * du + av[k + 2])
        tri_val.append(is_real[k] * is_real[k + 1] * is_real[k + 2])

    tri_hist = demux_histogram(tri_idx, tri_val, du_cubed)
    for b in range(du_cubed):
        output_row[b] = tri_hist[b]
