"""Carry an asset's render mesh along with the elements that are actually simulated.

A deformable asset often ships two meshes. The solver moves one -- tetrahedra for a soft body,
triangles for a cloth -- and that is what the physics means. The one a person recognises may be
another: finer, with the UVs and the textures on it, which no solver touches.

Showing only the simulated surface is honest and it is what you want when the question is "what
did the solver do". Showing the render mesh is what you want when the question is "what does this
look like". So both are produced, and binding one to the other is this file's whole job.

The binding is made once, in the rest pose, and it is a rule about elements rather than about any
particular asset: a vertex inside a tetrahedron is four barycentric weights, a vertex near a
triangle is three weights plus how far off the surface it sits. Moving it afterwards replays that
description against wherever its element is now. Nothing here knows what the asset is.
"""
import numpy as np

TET, TRI = 4, 3
# How far a barycentric weight may go negative and still count as on the element rather than off
# it. Barycentric weights are dimensionless and sum to one, so this is a fraction of the element
# itself and needs no length scale.
INSIDE_TOLERANCE = 1e-6


def _as_elements(elements):
    elements = np.asarray(elements, dtype=np.int64)
    if elements.ndim != 2 or elements.shape[1] not in (TET, TRI):
        raise ValueError(f"elements must be (n, 4) tetrahedra or (n, 3) triangles, got {elements.shape}")
    return elements


def surface_faces(elements):
    """The triangles to draw for these elements.

    For tetrahedra that is every face exactly one of them owns -- drawing all four faces of every
    tetrahedron draws the inside too, and the result reads as a lump rather than as the object.
    A triangle mesh is already its own surface.
    """
    elements = _as_elements(elements)
    if elements.shape[1] == TRI:
        return [tuple(int(i) for i in tri) for tri in elements]
    counts = {}
    for tet in elements:
        for face in ((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)):
            key = tuple(sorted(int(tet[i]) for i in face))
            counts[key] = counts.get(key, 0) + 1
    return [face for face, n in counts.items() if n == 1]


def _tet_weights(points, corners):
    """Barycentric weights within each candidate tetrahedron, allowed to go negative."""
    origin = corners[:, 0, :]
    basis = np.stack([corners[:, 1, :] - origin, corners[:, 2, :] - origin, corners[:, 3, :] - origin], axis=-1)
    # A degenerate tetrahedron has no inverse; it also cannot contain anything, so it is left to
    # lose the containment test rather than crash the solve.
    determinant = np.linalg.det(basis)
    safe = np.abs(determinant) > 1e-18
    weights = np.full((len(points), TET), -np.inf)
    if safe.any():
        solved = np.linalg.solve(basis[safe], (points[safe] - origin[safe])[..., None])[..., 0]
        weights[safe] = np.concatenate([(1.0 - solved.sum(axis=1))[:, None], solved], axis=1)
    return weights


def _tri_weights(points, corners):
    """Barycentric weights within each candidate triangle's plane, plus the distance off it.

    A cloth has no inside to be in, so "which element holds this vertex" becomes "which triangle
    is it over, and how far above". The offset is kept and replayed along the triangle's current
    normal, which is what carries a render mesh's thickness on a surface that has none.
    """
    a, b, c = corners[:, 0, :], corners[:, 1, :], corners[:, 2, :]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    area = np.linalg.norm(normal, axis=1)
    safe = area > 1e-18
    weights = np.full((len(points), TRI), -np.inf)
    offsets = np.zeros(len(points))
    if safe.any():
        unit = normal[safe] / area[safe][:, None]
        relative = points[safe] - a[safe]
        offsets[safe] = (relative * unit).sum(axis=1)
        flat = relative - offsets[safe][:, None] * unit
        d00 = (ab[safe] * ab[safe]).sum(axis=1)
        d01 = (ab[safe] * ac[safe]).sum(axis=1)
        d11 = (ac[safe] * ac[safe]).sum(axis=1)
        d20 = (flat * ab[safe]).sum(axis=1)
        d21 = (flat * ac[safe]).sum(axis=1)
        denominator = d00 * d11 - d01 * d01
        denominator[np.abs(denominator) < 1e-30] = np.inf
        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        weights[safe] = np.stack([1.0 - v - w, v, w], axis=1)
    return weights, offsets


def bind(render_points, node_points, elements, candidates=32):
    """Describe each render vertex in terms of one element, in the rest pose.

    Returns a binding to hand back to `deform`. Done once; `deform` is what runs per frame.
    """
    from scipy.spatial import cKDTree

    render_points = np.asarray(render_points, dtype=np.float64)
    node_points = np.asarray(node_points, dtype=np.float64)
    elements = _as_elements(elements)
    kind = elements.shape[1]
    centroids = node_points[elements].mean(axis=1)
    _, nearby = cKDTree(centroids).query(render_points, k=min(candidates, len(centroids)))
    nearby = np.atleast_2d(nearby)
    if nearby.shape[0] == 1 and len(render_points) != 1:
        nearby = nearby.T

    chosen = np.zeros(len(render_points), dtype=np.int64)
    weights = np.zeros((len(render_points), kind))
    offsets = np.zeros(len(render_points))
    best = np.full(len(render_points), -np.inf)
    settled = np.zeros(len(render_points), dtype=bool)
    for column in range(nearby.shape[1]):
        pending = ~settled
        if not pending.any():
            break
        rows = np.flatnonzero(pending)
        candidate = nearby[pending, column]
        corners = node_points[elements[candidate]]
        if kind == TET:
            w, off = _tet_weights(render_points[pending], corners), np.zeros(len(rows))
        else:
            w, off = _tri_weights(render_points[pending], corners)
        # How well the vertex sits in this element: inside means every weight is 0 or more. For a
        # triangle the offset is expected, so only the in-plane weights decide.
        # A render vertex usually sits *on* the simulated surface, where one barycentric weight
        # is zero and floating point puts it a hair below. Counting those as outside made 12118
        # of 17186 look misplaced on two meshes whose bounding boxes agree to a hundredth of a
        # millimetre. The tolerance is relative to the element, so it means the same thing at any
        # scale.
        score = w.min(axis=1)
        better = score > best[pending]
        chosen[rows[better]] = candidate[better]
        weights[rows[better]] = w[better]
        offsets[rows[better]] = off[better]
        best[rows[better]] = score[better]
        settled[rows[score >= -INSIDE_TOLERANCE]] = True

    # Whatever stayed outside every candidate rides its nearest element instead of being dropped.
    outside = best < -INSIDE_TOLERANCE
    if outside.any():
        clamped = np.clip(weights[outside], 0.0, None)
        weights[outside] = clamped / np.maximum(clamped.sum(axis=1, keepdims=True), 1e-12)
    return {"kind": kind, "element": chosen, "weights": weights, "offset": offsets,
            "outside": int(outside.sum())}


def deform(binding, elements, node_points):
    """Where the bound vertices are now."""
    elements = _as_elements(elements)
    nodes = np.asarray(node_points, dtype=np.float64)
    corners = nodes[elements[binding["element"]]]
    placed = np.einsum("ij,ijk->ik", binding["weights"], corners)
    if binding["kind"] == TRI and np.any(binding["offset"]):
        normal = np.cross(corners[:, 1, :] - corners[:, 0, :], corners[:, 2, :] - corners[:, 0, :])
        length = np.linalg.norm(normal, axis=1)
        length[length < 1e-18] = np.inf
        placed = placed + binding["offset"][:, None] * (normal / length[:, None])
    return placed
