"""Compile-time pruned bitonic merge schedule: only a real-real pair costs an oblivious compare-swap."""
from __future__ import annotations

import math

from .utils import next_pow2


def normalize_partial_lens(n_parties: int, partial_lens) -> list[int]:
    """Accept a scalar (uniform) or per-party list; validate shape."""
    if isinstance(partial_lens, int):
        return [partial_lens] * n_parties
    lens = list(partial_lens)
    assert len(lens) == n_parties, (len(lens), n_parties)
    assert all(n >= 1 for n in lens), lens
    return lens


def build_pruned_merge_schedule(n_parties: int, partial_lens):
    """Return one (reversal_swaps, merge_steps) per level plus the swap counts the configuration banner prints."""
    lens = normalize_partial_lens(n_parties, partial_lens)
    n_parties_padded = next_pow2(n_parties)
    party_slots = next_pow2(max(lens))
    total_slots = party_slots * n_parties_padded
    n_levels = int(math.log2(n_parties_padded))

    pad = [True] * total_slots
    for p in range(n_parties):
        for k in range(lens[p]):
            pad[p * party_slots + k] = False

    schedules = []
    totals = {"reversal": 0, "oblivious": 0, "static": 0}

    for lv in range(n_levels):
        block = party_slots * (2 ** (lv + 1))
        half = block // 2
        n_blocks = total_slots // block
        level_reversal: list[tuple[int, int]] = []
        level_merge_steps: list[tuple[list, list]] = []

        for g in range(n_blocks):
            base = g * block
            right_start, right_end = base + half, base + block

            # Reverse the right half into a bitonic sequence, then cascade pruned half-cleaners.
            rev_swaps: list[tuple[int, int]] = []
            for k in range(half // 2):
                i, j = right_start + k, right_end - 1 - k
                if i < j and not (pad[i] and pad[j]):
                    rev_swaps.append((i, j))
                    pad[i], pad[j] = pad[j], pad[i]
            level_reversal.extend(rev_swaps)

            step = half
            while step > 0:
                statics: list[tuple[int, int]] = []
                oblivious: list[tuple[int, int]] = []
                for i in range(base, base + block, step * 2):
                    for k2 in range(step):
                        a, b = i + k2, i + k2 + step
                        if b < base + block:
                            if pad[a] and pad[b]:
                                pass  # both PAD, no-op
                            elif not pad[a] and pad[b]:
                                pass  # real < PAD, already correct
                            elif pad[a] and not pad[b]:
                                statics.append((a, b))
                                pad[a], pad[b] = False, True
                            else:
                                oblivious.append((a, b))
                level_merge_steps.append((statics, oblivious))
                step //= 2

        totals["reversal"] += len(level_reversal)
        totals["oblivious"] += sum(len(o) for _, o in level_merge_steps)
        totals["static"] += sum(len(s) for s, _ in level_merge_steps)
        schedules.append((level_reversal, level_merge_steps))

    return schedules, totals
