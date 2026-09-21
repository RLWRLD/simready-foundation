"""Carry an asset's render mesh along with the tetrahedra that are actually simulated.

A deformable asset ships two meshes. The solver moves the coarse tetrahedral one -- 3074 points
for this banana -- and that is what the physics means. The one a person recognises is the other:
17186 points, with the UVs and the textures on it, which no solver touches.

Showing only the simulated surface is honest and it is what you want when the question is "what
did the solver do". Showing only the render mesh is what you want when the question is "what does
this look like". So both are produced, and binding one to the other is this file's whole job.

Each render vertex is bound once, in the rest pose, to the tetrahedron that holds it, as four
barycentric weights. Moving it afterwards is a weighted sum of that tetrahedron's four corners --
smooth, because it is linear inside each element, and exactly right wherever the two meshes agree.
A vertex outside the tetrahedral hull (the render mesh is usually a little proud of it) binds to
the nearest tetrahedron with its weights clamped, which carries it rigidly with that element
rather than leaving it behind.
"""
import numpy as np


def surface_faces(tet_indices):
    """The outside of a tetrahedral mesh: every triangular face that exactly one tetrahedron owns.

    Drawing all four faces of every tetrahedron draws the inside too, and the result reads as a
    lump rather than as the object.
    """
    counts = {}
    for tet in tet_indices:
        for face in ((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)):
            key = tuple(sorted(int(tet[i]) for i in face))
            counts[key] = counts.get(key, 0) + 1
    return [face for face, n in counts.items() if n == 1]


def _barycentric(points, corners):
    """Weights of each point within its candidate tetrahedron, allowed to go negative."""
    origin = corners[:, 0, :]
    basis = np.stack([corners[:, 1, :] - origin, corners[:, 2, :] - origin, corners[:, 3, :] - origin], axis=-1)
    # A degenerate (flat) tetrahedron has no inverse; it also cannot contain anything, so it is
    # left to lose the containment test rather than crash the solve.
    determinant = np.linalg.det(basis)
    safe = np.abs(determinant) > 1e-18
    weights = np.full((len(points), 4), -np.inf)
    if safe.any():
        solved = np.linalg.solve(basis[safe], (points[safe] - origin[safe])[..., None])[..., 0]
        weights[safe] = np.concatenate([(1.0 - solved.sum(axis=1))[:, None], solved], axis=1)
    return weights


def bind(render_points, node_points, tet_indices, candidates=32):
    """Bind each render vertex to one tetrahedron, in the rest pose.

    Returns (tet index per vertex, 4 weights per vertex). Done once; `deform` is what runs per
    frame.
    """
    from scipy.spatial import cKDTree

    render_points = np.asarray(render_points, dtype=np.float64)
    node_points = np.asarray(node_points, dtype=np.float64)
    tet_indices = np.asarray(tet_indices, dtype=np.int64)
    centroids = node_points[tet_indices].mean(axis=1)
    _, nearby = cKDTree(centroids).query(render_points, k=min(candidates, len(centroids)))
    nearby = np.atleast_2d(nearby)

    chosen = np.zeros(len(render_points), dtype=np.int64)
    weights = np.zeros((len(render_points), 4))
    settled = np.zeros(len(render_points), dtype=bool)
    best_score = np.full(len(render_points), -np.inf)
    for column in range(nearby.shape[1]):
        pending = ~settled
        if not pending.any():
            break
        tets = nearby[pending, column]
        w = _barycentric(render_points[pending], node_points[tet_indices[tets]])
        # How far inside the tetrahedron the point is: 0 or more means it is in there.
        score = w.min(axis=1)
        better = score > best_score[pending]
        rows = np.flatnonzero(pending)[better]
        chosen[rows] = tets[better]
        weights[rows] = w[better]
        best_score[rows] = score[better]
        settled[np.flatnonzero(pending)[score >= 0.0]] = True

    # Whatever stayed outside every candidate rides its nearest element instead of being dropped.
    outside = best_score < 0.0
    if outside.any():
        clamped = np.clip(weights[outside], 0.0, None)
        weights[outside] = clamped / np.maximum(clamped.sum(axis=1, keepdims=True), 1e-12)
    return chosen, weights, int(outside.sum())


def deform(chosen, weights, tet_indices, node_points):
    """Where the bound vertices are now."""
    corners = np.asarray(node_points, dtype=np.float64)[np.asarray(tet_indices)[chosen]]
    return np.einsum("ij,ijk->ik", weights, corners)
