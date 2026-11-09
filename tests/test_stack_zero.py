"""Where the stack's zero comes from.

On Andy's 2024 stack (Nov 2026, frame 35 the reference, no bias frames: the black level from each
RAW file) every long exposure carried the same offset against the reference, in stack units
R 6.3, G 9.5, B 8.3 x 1e-5 (from 1/4 s to 8 s): the reference frame's black level was about
1.4 ADU in 15000 out. The stack took its zero from the reference, so its sky came out below zero
by that much (Enhance: "sky removed ... -0.000116, -8.15e-05, -9.21e-05" after white balance),
black beyond the corona with Keep the sky, and the sky model "removed" a negative sky. The
longest exposures set the zero now: there a black-level error of an ADU or two is divided by a
gain of 100-500 and does not matter."""

import numpy as np

from tse_core.stack import FrameInfo, StackParams, match_exposures

SKY = np.array([2.0e-4, 3.0e-4, 2.5e-4])     # the true sky, in reference units
REF_ERR = np.array([-6.3e-5, -9.5e-5, -8.3e-5])  # the reference frame's black-level error


def _frames():
    n = 160
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n / 2, yy - n / 2) / 12.0 + 1.0
    corona = 0.02 * r ** -3                    # reference units
    gains = [0.06, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0]
    reduced, infos = {}, {}
    rng = np.random.default_rng(1)
    for i, g in enumerate(gains):
        err = REF_ERR if g == 1.0 else rng.normal(0, 3e-6, 3)  # the others: a fraction of an ADU
        v = g * (corona[..., None] + SKY[None, None, :]) + err[None, None, :]
        ok = (v.mean(axis=2) < 0.85) & (v.mean(axis=2) > 0.0005)
        reduced[i] = (np.clip(v, 0, 1).astype(np.float32), ok)
        infos[i] = FrameInfo(i, f"f{i}", g, [[1, 0, 0], [0, 1, 0]], 0.0, 0.0, 1.0)
    return reduced, infos, gains


def test_the_longest_exposures_set_the_stacks_zero():
    reduced, infos, gains = _frames()
    msgs = []
    m = match_exposures(reduced, infos, 2, StackParams(match_smooth=0.0), msgs.append)
    longest = max(m, key=lambda i: m[i].gain)
    v, ok = reduced[longest]
    # The sky as the stack sees it in the longest frame (its darkest usable pixel), in reference units.
    far = (v.mean(axis=2) == v.mean(axis=2)[ok].min())
    seen = np.array([float(((v[..., c] - m[longest].offsets[c]) / m[longest].gain)[far].mean()) for c in range(3)])
    corona_min = 0.02 * (np.hypot(80, 80) / 12.0 + 1.0) ** -3
    assert np.allclose(seen - corona_min, SKY, atol=5e-6), (seen - corona_min, SKY)
    assert any("zero" in s for s in msgs), msgs
