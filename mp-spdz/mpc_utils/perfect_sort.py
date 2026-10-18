"""reveal_sort for a key that is a permutation of 0..n-1: one secret shuffle, one open, one clear scatter (HICT14)."""

from Compiler import instructions, library, types
from Compiler.types import cint

from .utils import next_pow2


def _clear_sum(vec, n):
    """Log-depth sum of a clear vector, in local clear operations."""
    m = next_pow2(n)
    buf = cint.Array(m)
    if m > n:
        buf.assign_all(0)
    buf.assign_vector(cint(vec), base=0)
    width = m
    while width > 1:
        half = width // 2
        buf.assign_vector(
            buf.get_vector(base=0, size=half) + buf.get_vector(base=half, size=half),
            base=0,
        )
        width = half
    return buf[0]


def reveal_sort_checked(k, D):
    """Sort payload D by the perfect key k, so that D_final[k[i]] = D_old[i]; crashes if k is not a permutation."""
    library.get_program().reading('sorting', 'HICT14')
    n = len(k)
    assert n == len(D)
    library.break_point()
    # Leakage: the one opened vector is sigma o k for a fresh shuffle sigma, uniform on S_n for any bijection k, so it
    # discloses nothing; the handle is fresh per call and deleted, since reuse would leak relative permutations.
    shuffle = types.sint.get_secure_shuffle(n)
    k_prime = k.get_vector().secure_permute(shuffle).reveal()
    idx = types.Array.create_from(k_prime)

    # The guard is not optional: the scatter is unbounded, so an out-of-range key writes outside the array and a
    # duplicate key drops a payload row. Both checks are clear-side work on a vector the primitive opens anyway.
    rng_bad = _clear_sum((idx.get_vector() < 0) + (idx.get_vector() >= n), n)
    library.runtime_error_if(
        rng_bad != 0,
        "FATAL reveal_sort: %s key(s) outside [0, %s) - key is not a "
        "permutation; scatter would write outside the array", rng_bad, n)
    library.break_point()

    # Coverage: scatter 1 to every key position; sum == n iff no duplicates.
    seen = cint.Array(n)
    seen.assign_all(0)
    seen.assign_slice_vector(idx, cint(1, size=n))
    covered = _clear_sum(seen.get_vector(), n)
    library.runtime_error_if(
        covered != n,
        "FATAL reveal_sort: key covers %s of %s slots (duplicate keys; "
        "scatter would drop payload rows)", covered, n)
    library.break_point()

    D.secure_permute(shuffle)
    library.break_point()
    D.assign_slice_vector(idx, D.get_vector())
    library.break_point()
    instructions.delshuffle(shuffle)
