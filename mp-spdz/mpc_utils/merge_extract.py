"""Round-1 input read, per-case merge tree and extraction, shared by the two extract programs."""

import math

from Compiler.library import for_range_opt_multithread
from Compiler.types import sint

from . import encoding_bigram
from .merge_schedule import normalize_partial_lens
from .utils import next_pow2


def read_party_inputs(n_parties, n_cases, partial_lens):
    """Read interleaved (activity, timestamp) pairs from each party into one flat array of n_cases * L_p entries."""
    lens = normalize_partial_lens(n_parties, partial_lens)
    activities, timestamps = [], []
    for p in range(n_parties):
        acts = sint.Array(n_cases * lens[p])
        ts = sint.Array(n_cases * lens[p])
        for i in range(n_cases * lens[p]):
            acts[i] = sint.get_input_from(p)
            ts[i] = sint.get_input_from(p)
        activities.append(acts)
        timestamps.append(ts)
    return activities, timestamps


def merge_and_extract(
    activities, timestamps,
    n_parties, n_cases, partial_lens, dict_size, n_threads,
    encoding_is_bigram, encoding_extract, enc_dim,
    pad_composite, level_schedules,
):
    """Pruned bitonic merge per case, then the DFG bigrams and the chosen encoding; the two alias at ENCODING=0."""
    lens = normalize_partial_lens(n_parties, partial_lens)
    n_parties_padded = next_pow2(n_parties)
    n_levels = int(math.log2(n_parties_padded))
    party_slots = next_pow2(max(lens))
    total_slots = party_slots * n_parties_padded
    max_real = sum(lens)
    actual_bigrams = dict_size * dict_size

    share_buffers = encoding_is_bigram

    bigram_matrix = sint.Matrix(n_cases, actual_bigrams)
    encoding_matrix = bigram_matrix if share_buffers else sint.Matrix(n_cases, enc_dim)

    @for_range_opt_multithread(n_threads, n_cases * actual_bigrams)
    def _(flat):
        bigram_matrix[flat // actual_bigrams][flat % actual_bigrams] = sint(0)

    if not share_buffers:
        @for_range_opt_multithread(n_threads, n_cases * enc_dim)
        def _(flat):
            encoding_matrix[flat // enc_dim][flat % enc_dim] = sint(0)

    @for_range_opt_multithread(n_threads, n_cases)
    def _(case_idx):
        sort_keys = [sint(pad_composite) for _ in range(total_slots)]
        sort_vals = [sint(0) for _ in range(total_slots)]
        for p in range(n_parties):
            for k in range(lens[p]):
                event_idx = case_idx * lens[p] + k
                slot = p * party_slots + k
                act = activities[p][event_idx]
                ts = timestamps[p][event_idx]
                # PAD events sort to the end; the composite key breaks timestamp ties by activity id.
                is_pad = (act == 0)
                sort_keys[slot] = is_pad.if_else(pad_composite, ts * dict_size + act)
                sort_vals[slot] = act

        for lv in range(n_levels):
            reversal_swaps, merge_steps = level_schedules[lv]
            for i, j in reversal_swaps:
                sort_keys[i], sort_keys[j] = sort_keys[j], sort_keys[i]
                sort_vals[i], sort_vals[j] = sort_vals[j], sort_vals[i]
            for static_swaps, oblivious_pairs in merge_steps:
                for idx1, idx2 in static_swaps:
                    sort_keys[idx1] = sort_keys[idx2]
                    sort_vals[idx1] = sort_vals[idx2]
                    sort_keys[idx2] = sint(pad_composite)
                    sort_vals[idx2] = sint(0)
                if oblivious_pairs:
                    ka = [sort_keys[a] for a, _ in oblivious_pairs]
                    kb = [sort_keys[b] for _, b in oblivious_pairs]
                    va = [sort_vals[a] for a, _ in oblivious_pairs]
                    vb = [sort_vals[b] for _, b in oblivious_pairs]
                    swaps = [(a > b) for a, b in zip(ka, kb)]
                    # cond_swap: one product serves both outputs of a pair, so a comparator costs two mults, not four.
                    nk = [s.cond_swap(a, b) for s, a, b in zip(swaps, ka, kb)]
                    nv = [s.cond_swap(a, b) for s, a, b in zip(swaps, va, vb)]
                    for (a, b), (k1, k2), (v1, v2) in zip(
                        oblivious_pairs, nk, nv
                    ):
                        sort_keys[a], sort_keys[b] = k1, k2
                        sort_vals[a], sort_vals[b] = v1, v2

        # Only the first sum(L_p) slots of the merged array can hold real events.
        merged_keys = sort_keys[:max_real]
        merged_vals = sort_vals[:max_real]
        encoding_bigram.extract_features(
            merged_keys, merged_vals, max_real, dict_size, pad_composite,
            bigram_matrix[case_idx],
        )
        if not share_buffers:
            encoding_extract(
                merged_keys, merged_vals, max_real, dict_size, pad_composite,
                encoding_matrix[case_idx],
            )

    return bigram_matrix, encoding_matrix
