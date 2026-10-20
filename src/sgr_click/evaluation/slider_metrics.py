"""Diagnostics for the native integer slider, independent of annotation levels."""
import numpy as np


def slider_diagnostics(masks, window=5, still_threshold=.005, jump_threshold=.05):
    """Changes use XOR, normalized by the foreground union over the trajectory.

    A rolling window starts at EVERY native tick, not just multiples of ten.
    The accumulated motion also detects equal-area swaps and out-and-back motion.
    These thresholds are diagnostic conventions, not published standard metrics.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or len(masks) != 101:
        raise ValueError('Expected 101 masks at native slider positions 0..100')
    if not 1 <= window <= 100:
        raise ValueError('Window must lie in 1..100')
    if not 0 <= still_threshold < jump_threshold <= 1:
        raise ValueError('Expected 0 <= still < jump <= 1')
    support = max(int(np.any(masks, axis=0).sum()), 1)
    changes = np.count_nonzero(masks[1:] ^ masks[:-1], axis=(1, 2)) / support
    motion = np.r_[0., np.cumsum(changes)]
    window_changes = motion[window:] - motion[:-window]
    still = window_changes <= still_threshold + 1e-12
    jumps = changes > jump_threshold + 1e-12
    # Maximum native slider span whose total motion stays below the same limit.
    longest, left = 0, 0
    for right in range(1, 101):
        while motion[right] - motion[left] > still_threshold + 1e-12:
            left += 1
        longest = max(longest, right - left)
    return dict(
        stagnant_window_rate=float(still.mean()),
        jump_step_rate=float(jumps.mean()),
        has_stagnation=bool(still.any()), has_jump=bool(jumps.any()),
        longest_stagnant_span=int(longest),
        max_step_change=float(changes.max()),
        p95_step_change=float(np.quantile(changes, .95)),
        mean_step_change=float(changes.mean()),
        trajectory_support_pixels=support,
        step_changes=changes.tolist(),
        stagnant_window_starts=np.flatnonzero(still).tolist(),
        jump_destination_ticks=(np.flatnonzero(jumps) + 1).tolist())
