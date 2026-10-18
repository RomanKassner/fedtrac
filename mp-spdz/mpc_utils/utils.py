"""Small compile-time helpers shared by the secure programs."""


def profile_timers(profile):
    """(tick, tock) for a program's PROFILE argument, no-ops at PROFILE=0; main tape only, a timer has one owner."""
    if not profile:
        return (lambda _timer_id: None), (lambda _timer_id: None)
    from Compiler.library import start_timer, stop_timer
    return start_timer, stop_timer


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def expected_enc_dim(encoding_flag, dict_size):
    """The width each ENCODING flag compiles at; twin of pipeline::mpc::shared::encoding_dim."""
    if encoding_flag == 3:                      # activity
        return dict_size
    if encoding_flag == 4:                      # trigram: user alphabet only
        du = dict_size - 3
        return du * du * du
    if encoding_flag == 5:                      # skip-bigram
        return 2 * dict_size * dict_size
    return dict_size * dict_size                # bigram (0) / ef (1)
