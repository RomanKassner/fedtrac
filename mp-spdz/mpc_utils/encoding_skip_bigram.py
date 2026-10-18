"""Skip-bigram encoding: 2*d^2 cells, the direct bigrams and the pairs at distance 2 in the extended trace."""

from Compiler.types import sint

from .constants import END_ACT, START_ACT
from .encoding_bigram import demux_histogram


def extract_features(sort_keys, sort_vals, max_real, dict_size, pad_sentinel_val,
                     output_row):
    """Write the 2*d^2 skip-bigram histogram into output_row: block 0 the direct pairs, block 1 the distance-2 pairs."""
    d = dict_size
    block = d * d
    total_bins = 2 * block
    skip_base = sint(block)

    bg_idx = []
    bg_val = []

    # Block 0: the bigram contributions, as in encoding_bigram.
    first_real = (sort_keys[0] != sint(pad_sentinel_val))
    bg_idx.append(sint(START_ACT * d) + sort_vals[0])
    bg_val.append(first_real)

    for k in range(max_real - 1):
        a_real = (sort_keys[k]     != sint(pad_sentinel_val))
        b_real = (sort_keys[k + 1] != sint(pad_sentinel_val))
        bg_idx.append(sort_vals[k] * d + sort_vals[k + 1])
        bg_val.append(a_real * b_real)

    for k in range(max_real):
        is_real_k = (sort_keys[k] != sint(pad_sentinel_val))
        if k < max_real - 1:
            next_is_pad = (sort_keys[k + 1] == sint(pad_sentinel_val))
        else:
            next_is_pad = sint(1)
        is_last = is_real_k * next_is_pad
        bg_idx.append(sort_vals[k] * d + sint(END_ACT))
        bg_val.append(is_last)

    # Block 1: the distance-2 contributions.
    if max_real >= 2:
        second_real = (sort_keys[1] != sint(pad_sentinel_val))
        bg_idx.append(skip_base + sint(START_ACT * d) + sort_vals[1])
        bg_val.append(second_real)

    # (START, END) when the trace has one real event, which the twin extract_skip_bigram counts too.
    if max_real >= 2:
        second_is_pad = (sort_keys[1] == sint(pad_sentinel_val))
    else:
        second_is_pad = sint(1)
    bg_idx.append(skip_base + sint(START_ACT * d + END_ACT))
    bg_val.append(first_real * second_is_pad)

    for k in range(max_real - 2):
        a_real = (sort_keys[k]     != sint(pad_sentinel_val))
        c_real = (sort_keys[k + 2] != sint(pad_sentinel_val))
        bg_idx.append(skip_base + sort_vals[k] * d + sort_vals[k + 2])
        bg_val.append(a_real * c_real)

    # (second-to-last real activity, END).
    for k in range(1, max_real):
        is_real_k = (sort_keys[k] != sint(pad_sentinel_val))
        if k < max_real - 1:
            next_is_pad = (sort_keys[k + 1] == sint(pad_sentinel_val))
        else:
            next_is_pad = sint(1)
        is_last = is_real_k * next_is_pad
        bg_idx.append(skip_base + sort_vals[k - 1] * d + sint(END_ACT))
        bg_val.append(is_last)

    # Width: every index is below 2*d^2 by construction, the demux kernel's precondition.
    hist = demux_histogram(bg_idx, bg_val, total_bins)
    for b in range(total_bins):
        output_row[b] = hist[b]
