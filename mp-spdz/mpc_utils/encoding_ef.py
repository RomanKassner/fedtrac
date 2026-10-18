"""Eventually-follows encoding: output_row[b*d+a] counts "b strictly before a" in [START] + real events + [END]."""

from Compiler.types import sint

from .constants import END_ACT, START_ACT


def extract_features(sort_keys, sort_vals, max_real, dict_size, pad_sentinel_val, output_row):
    """Accumulate the eventually-follows counts into output_row."""
    d = dict_size
    prefix = [sint(0)] * d

    real_flags = [(sort_keys[k] != sint(pad_sentinel_val)) for k in range(max_real)]

    # Virtual START event, real iff the case has a real event; the gate keeps an all-pad case all-zero, which would
    # otherwise carry one EF cell and form a spurious shared fingerprint class.
    any_real = sint(1)
    for flag in real_flags:
        any_real = any_real * (sint(1) - flag)
    any_real = sint(1) - any_real if real_flags else sint(0)
    prefix[START_ACT] = any_real

    # Real events of the merged trace.
    for k in range(max_real):
        is_real = real_flags[k]
        act = sort_vals[k]

        gated_one_hot = [is_real * (act == sint(a)) for a in range(d)]

        for b in range(d):
            for a in range(d):
                output_row[b * d + a] += prefix[b] * gated_one_hot[a]

        for a in range(d):
            prefix[a] += gated_one_hot[a]

    # Virtual END event, always real and of public activity END_ACT.
    for b in range(d):
        output_row[b * d + END_ACT] += prefix[b]
