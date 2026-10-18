"""Helpers for the secure programs; every encoding_* submodule exposes one extract_features with the same signature."""

from . import (
    encoding_activity,
    encoding_bigram,
    encoding_ef,
    encoding_skip_bigram,
    encoding_trigram,
)
from .constants import END_ACT, START_ACT
from .merge_extract import merge_and_extract, read_party_inputs
from .merge_schedule import build_pruned_merge_schedule
from .utils import expected_enc_dim, next_pow2, profile_timers

__all__ = ['next_pow2', 'expected_enc_dim', 'profile_timers',
           'START_ACT', 'END_ACT',
           'build_pruned_merge_schedule',
           'merge_and_extract', 'read_party_inputs',
           'encoding_bigram', 'encoding_ef',
           'encoding_activity', 'encoding_trigram', 'encoding_skip_bigram']