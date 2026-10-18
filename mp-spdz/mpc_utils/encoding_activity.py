"""Bag-of-activities encoding; twin of src/phases/p2_bigrams.rs::extract_activity (START and END are not counted)."""

from Compiler.types import sint


def extract_features(sort_keys, sort_vals, max_real, dict_size, pad_sentinel_val,
                     output_row):
    """Accumulate per-activity counts into output_row[a]."""
    for k in range(max_real):
        is_real = (sort_keys[k] != sint(pad_sentinel_val))
        act = sort_vals[k]
        for a in range(dict_size):
            output_row[a] += is_real * (act == sint(a))
