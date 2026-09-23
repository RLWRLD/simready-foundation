# SPDX-License-Identifier: Apache-2.0
"""Whether a multi-body asset is still one object: measured on the solver's particles.

The drop and press verdicts are read off the merged particle set and say nothing about the bodies
staying where they belong -- measured, a loaded polybag whose film lay flat through its filling
read `pass`. These are the measurements that would have said so, printed in every setup:

  seal_median_mm / seal_max_mm   the gap across the asset's authored seal pairs at the last frame
                                 (the polybag authors 2.0 mm median, 10.6 mm max)
  escaped_nodes                  nodes of an enclosed body that ended up below the enclosing
                                 body's own lowest node near them -- a filling that fell out of
                                 the bottom of its bag

They are measurements, not verdicts: no word is attached to them here.
"""
import numpy as np

# How far, in the plane, a node of the enclosing body has to be from an enclosed node to count as
# "above/below it". The asset's own resolution: passed in as its contact size.


def seal_gap(q, pairs):
    """(median, max) distance across the authored seal pairs, or (None, None) without any."""
    if len(pairs) == 0:
        return None, None
    d = np.linalg.norm(q[pairs[:, 0]] - q[pairs[:, 1]], axis=1)
    return float(np.median(d)), float(d.max())


def enclosure(bodies):
    """(outer, inner) index arrays: the body whose authored box contains the other's, or None."""
    if len(bodies) != 2:
        return None
    (pa, qa), (pb, qb) = bodies
    def contains(outer, inner):
        return bool((outer.min(0) <= inner.min(0) + 1e-9).all() and (outer.max(0) >= inner.max(0) - 1e-9).all())
    if contains(qa, qb):
        return pa, pb
    if contains(qb, qa):
        return pb, pa
    return None


def escaped(q, outer, inner, reach):
    """How many nodes of `inner` sit below every node of `outer` within `reach` of them in the
    plane -- below the bag's floor, not inside it. Chunked so a 10k x 8k pair stays in memory."""
    qo, qi = q[outer], q[inner]
    count = 0
    for start in range(0, len(qi), 512):
        block = qi[start:start + 512]
        d2 = ((block[:, None, :2] - qo[None, :, :2]) ** 2).sum(-1)
        near = d2 <= reach * reach
        lowest = np.where(near, qo[None, :, 2], np.inf).min(1)
        count += int(((block[:, 2] < lowest - 1e-4) & np.isfinite(lowest)).sum())
    return count


def result_tokens(q, pairs, bodies, reach):
    """The RESULT tokens for this frame: '' when the asset has nothing to measure this way."""
    out = []
    median, worst = seal_gap(q, pairs)
    if median is not None:
        out.append(f"seal_median_mm={median * 1000:.2f} seal_max_mm={worst * 1000:.2f}")
    pair = enclosure(bodies)
    if pair is not None:
        out.append(f"escaped_nodes={escaped(q, pair[0], pair[1], reach)}")
    return " ".join(out)
