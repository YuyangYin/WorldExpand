#!/usr/bin/env python3
"""
Polyhedra face geometries for tangent image cutting.

Supports:
  - 6 faces:  cube map
  - 20 faces: icosahedron (default)

All return the same format: list of dicts with 'center', 'u', 'v', 'half_angle', 'face_idx'.
"""

import numpy as np


def _normalize(v):
    """Normalize array of vectors to unit length along last axis."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _build_tangent_basis(center):
    """Build orthonormal tangent basis (u, v) for a face center direction."""
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(center, up_hint)) > 0.99:
        up_hint = np.array([1.0, 0.0, 0.0])
    u = np.cross(up_hint, center)
    u = u / np.linalg.norm(u)
    v = np.cross(center, u)
    v = v / np.linalg.norm(v)
    return u, v


# ── Cube (6 faces) ──────────────────────────────────────────────────

def _cube_faces(padding_ratio=0.2):
    """6 cube map faces: +X, -X, +Y, -Y, +Z, -Z."""
    directions = [
        np.array([ 1,  0,  0], dtype=np.float64),
        np.array([-1,  0,  0], dtype=np.float64),
        np.array([ 0,  1,  0], dtype=np.float64),
        np.array([ 0, -1,  0], dtype=np.float64),
        np.array([ 0,  0,  1], dtype=np.float64),
        np.array([ 0,  0, -1], dtype=np.float64),
    ]
    # Cube face half-angle = atan(1) = pi/4 = 45 degrees
    base_half_angle = np.pi / 4.0
    half_angle = base_half_angle * (1.0 + padding_ratio)

    results = []
    for fi, center in enumerate(directions):
        u, v = _build_tangent_basis(center)
        results.append({
            'center': center,
            'u': u,
            'v': v,
            'half_angle': half_angle,
            'face_idx': fi,
        })
    return results


# ── Icosahedron (20 faces) ──────────────────────────────────────────

def _icosahedron_vertices():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float64)
    return _normalize(verts)


def _icosahedron_faces_idx():
    return [
        (0, 11, 5),  (0, 5, 1),   (0, 1, 7),   (0, 7, 10),  (0, 10, 11),
        (1, 5, 9),   (5, 11, 4),  (11, 10, 2),  (10, 7, 6),  (7, 1, 8),
        (3, 9, 4),   (3, 4, 2),   (3, 2, 6),    (3, 6, 8),   (3, 8, 9),
        (4, 9, 5),   (2, 4, 11),  (6, 2, 10),   (8, 6, 7),   (9, 8, 1),
    ]


def _faces_from_verts_and_indices(verts, faces_idx, padding_ratio):
    """Build face info list from vertices and face index tuples."""
    results = []
    for fi, idx_tuple in enumerate(faces_idx):
        vs = [verts[i] for i in idx_tuple]
        center = sum(vs)
        center = center / np.linalg.norm(center)

        angles = [np.arccos(np.clip(np.dot(center, v), -1, 1)) for v in vs]
        base_half_angle = max(angles)
        half_angle = base_half_angle * (1.0 + padding_ratio)

        u, v = _build_tangent_basis(center)
        results.append({
            'center': center,
            'u': u,
            'v': v,
            'half_angle': half_angle,
            'face_idx': fi,
        })
    return results


def _icosahedron_faces(padding_ratio=0.2):
    """20 icosahedron faces."""
    verts = _icosahedron_vertices()
    faces_idx = _icosahedron_faces_idx()
    return _faces_from_verts_and_indices(verts, faces_idx, padding_ratio)


# ── Subdivided icosahedron (80 faces) ───────────────────────────────

def _front_face(half_angle):
    """Build a single face looking exactly along +z (OpenCV forward).

    This guarantees that face_00 is always the standard forward-looking
    camera so that Mirror's cam_align (which uses the first camera as
    reference) produces an upright coordinate system.
    """
    center = np.array([0.0, 0.0, 1.0])
    u = np.array([1.0, 0.0, 0.0])   # right
    v = np.array([0.0, 1.0, 0.0])   # up  (OpenCV y-down → col1 = -v)
    return {
        'center': center,
        'u': u,
        'v': v,
        'half_angle': half_angle,
        'face_idx': 0,
    }

# ── Public API ──────────────────────────────────────────────────────

SUPPORTED_FACES = {6, 20}

def get_face_infos(num_faces=20, padding_ratio=0.2, prepend_front=False):
    """
    Get face geometry for the specified number of faces.

    Args:
        num_faces: 6 (cube) or 20 (icosahedron)
        padding_ratio: overlap padding ratio
        prepend_front: if True, insert an absolute forward-looking face
            (center=[0,0,1]) as face_00 before all polyhedron faces.
            Total face count becomes num_faces + 1.

    Returns:
        list of dicts with 'center', 'u', 'v', 'half_angle', 'face_idx'
    """
    if num_faces == 6:
        faces = _cube_faces(padding_ratio)
    elif num_faces == 20:
        faces = _icosahedron_faces(padding_ratio)
    else:
        raise ValueError(f"Unsupported num_faces={num_faces}. Must be one of {SUPPORTED_FACES}")

    if prepend_front:
        # Use the same half_angle as the polyhedron faces for consistency
        ref_half_angle = faces[0]['half_angle']
        front = _front_face(ref_half_angle)
        # Re-index: front gets 0, original faces shift by 1
        for i, f in enumerate(faces):
            f['face_idx'] = i + 1
        faces = [front] + faces

    return faces
