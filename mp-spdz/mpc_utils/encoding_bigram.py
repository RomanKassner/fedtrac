"""Bigram (directly-follows) encoding: a d^2 histogram of pure counts, which doubles as the DFG slab."""

from Compiler.types import sint

from .constants import END_ACT, START_ACT


def demux_histogram(idx_list, val_list, n_buckets):
    """Distribute secret values into n_buckets by secret index; every index must lie in [0, n_buckets)."""
    m = len(idx_list)
    if m == 0:
        return [sint(0) for _ in range(n_buckets)]
    # Width: (n-1).bit_length() is exact at every n, where ceil(log2(n)) can hand back one bit too few.
    n_bits = max(1, (n_buckets - 1).bit_length())
    all_bits = [idx_list[j].bit_decompose(n_bits) for j in range(m)]

    # MSB-first, so a prefix whose smallest completion exceeds n_buckets-1 is massless and its mult is skipped.
    nodes = [(0, list(val_list))]
    for level in range(n_bits):
        bit_pos = n_bits - 1 - level  # bit_decompose returns LSB-first
        mask = [all_bits[j][bit_pos] for j in range(m)]
        new_nodes = []
        for prefix, row in nodes:
            p1 = prefix * 2 + 1
            if (p1 << bit_pos) >= n_buckets:
                new_nodes.append((prefix * 2, row))
            else:
                t = [row[j] * mask[j] for j in range(m)]
                new_nodes.append((prefix * 2, [row[j] - t[j] for j in range(m)]))
                new_nodes.append((p1, t))
        nodes = new_nodes

    out = [None] * n_buckets
    for prefix, row in nodes:
        total = row[0]
        for j in range(1, m):
            total = total + row[j]
        out[prefix] = total
    return out


def _edge_contributions(sort_keys, sort_vals, max_real, dict_size,
                        pad_sentinel_val):
    """Consecutive-pair cascade entries plus the START-row and END-column histograms, and the reusable is_real bits."""
    d = dict_size
    is_real = [sort_keys[k] != sint(pad_sentinel_val) for k in range(max_real)]

    pair_idx, pair_val = [], []
    for k in range(max_real - 1):
        pair_idx.append(sort_vals[k] * d + sort_vals[k + 1])
        pair_val.append(is_real[k] * is_real[k + 1])

    # START to first activity: public row START_ACT, d-bucket demux.
    start_hist = demux_histogram([sort_vals[0]], [is_real[0]], d)

    # Last activity to END: public column END_ACT; is_last reuses is_real rather than a second comparison.
    end_idx, end_val = [], []
    for k in range(max_real):
        next_is_pad = (sint(1) - is_real[k + 1]) if k < max_real - 1 else sint(1)
        end_idx.append(sort_vals[k])
        end_val.append(is_real[k] * next_is_pad)
    end_hist = demux_histogram(end_idx, end_val, d)

    return pair_idx, pair_val, start_hist, end_hist, is_real


def _assemble(output_row, dict_size, pair_hist, start_hist, end_hist):
    """Write the d^2 row; START and END are absent from the merged stream, so adding their histograms is exact."""
    d = dict_size
    for b in range(d * d):
        output_row[b] = pair_hist[b]
    for col in range(d):
        output_row[START_ACT * d + col] = output_row[START_ACT * d + col] + start_hist[col]
    for row_a in range(d):
        output_row[row_a * d + END_ACT] = output_row[row_a * d + END_ACT] + end_hist[row_a]


def extract_features(sort_keys, sort_vals, max_real, dict_size, pad_sentinel_val,
                     output_row):
    """Write the pure-count d^2 bigram histogram into output_row."""
    pair_idx, pair_val, start_hist, end_hist, _ = _edge_contributions(
        sort_keys, sort_vals, max_real, dict_size, pad_sentinel_val,
    )
    pair_hist = demux_histogram(pair_idx, pair_val, dict_size * dict_size)
    _assemble(output_row, dict_size, pair_hist, start_hist, end_hist)
