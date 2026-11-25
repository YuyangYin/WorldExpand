"""2x4 cropped-repeat cubemap padding layout — ERP <-> cube grid conversion.

Layout v2 matches the cubemap arrangement expected by the released checkpoints.

Conventions (unchanged from v1)
-------------------------------
* World coordinates: +X right, +Y up, +Z forward (right-handed).
* ERP: v=0 is the north pole (+Y), v=H-1 is the south pole (-Y),
  u=0 is back (-Z), u=W/2 is front (+Z), u=W/4 is left (-X),
  u=3W/4 is right (+X). i.e. ``u = (theta + pi) / (2pi) * (W-1)``
  with ``theta = atan2(x, z)`` and ``v = (pi/2 - phi) / pi * (H-1)``
  with ``phi = asin(y)``.
* Cube face indices (used by FACE_*):
    0 = +X (right) ; image up = +Y, image right = -Z
    1 = -X (left)  ; image up = +Y, image right = +Z
    2 = +Y (top)   ; image up = +Z (toward front), image right = +X    (*)
    3 = -Y (bot)   ; image up = -Z (toward back),  image right = +X    (*)
    4 = +Z (front) ; image up = +Y, image right = +X
    5 = -Z (back)  ; image up = +Y, image right = -X
    (*) The internal sampler ``_erp_sample_six_faces`` follows the convention
    above; before atlas assembly we vertically-flip the +Y / -Y faces so they
    match ``py360convert.e2p(yaw=0, pitch=±90)`` (which we verified is exactly
    a vertical flip of our internal convention). After the flip:
        FACE_PY (post-flip): row 0 = -Z (back) seam, row S-1 = +Z (front) seam
        FACE_NY (post-flip): row 0 = +Z (front) seam, row S-1 = -Z (back) seam

Grid layout — 2x4 cropped-repeat with vertical buffer
-----------------------------------------------------
::

    H_grid = 2 * face_size + 2 * grid_pad        (was 2*S in v1)
    W_grid = 4 * face_size +     grid_pad        (unchanged)

Vertical bands (S = face_size, half = S/2, P = grid_pad):

    rows [0,             half + P)        TOP band       (height = S/2 + P)
    rows [half + P,      half + P + S)    EQUATOR band   (height = S)
    rows [half + P + S,  2*S + 2*P)       BOTTOM band    (height = S/2 + P)

Horizontal columns (4 equator faces in L, F, R, B order):

    cols [pad + col*S,  pad + (col+1)*S)  for col in 0..3

Plus a ``grid_pad`` LEFT WRAP COLUMN at cols [0, pad), copied from the rightmost
content cols [content_w, content_w + pad). Same as v1.

Top band content (POST-FLIP +Y face, rotated and cropped per column):

    col 0 (L): np.rot90(face_PY_postflip, k=1)  src rows [S-(half+P), S)
    col 1 (F): np.rot90(face_PY_postflip, k=0)  src rows [S-(half+P), S)
    col 2 (R): np.rot90(face_PY_postflip, k=3)  src rows [S-(half+P), S)
    col 3 (B): np.rot90(face_PY_postflip, k=2)  src rows [S-(half+P), S)

Bottom band content (POST-FLIP -Y face, rotated and cropped per column):

    col 0 (L): np.rot90(face_NY_postflip, k=3)  src rows [0, half+P)
    col 1 (F): np.rot90(face_NY_postflip, k=0)  src rows [0, half+P)
    col 2 (R): np.rot90(face_NY_postflip, k=1)  src rows [0, half+P)
    col 3 (B): np.rot90(face_NY_postflip, k=2)  src rows [0, half+P)

Equator band content:

    col 0 (L): face_NX  (no rotation)
    col 1 (F): face_PZ
    col 2 (R): face_PX
    col 3 (B): face_NZ

The +Y / -Y face content appears 4× redundantly in the atlas (once per equator
column, each with a different rotation). Adjacent strips of a pole face land in
the atlas under different rotations -- this is by design (the colleague's
training data distribution). Adjacent strips overlap at the
pole-quadrant diagonals because each strip is half + pad rows tall (not just
half).

Verification
------------
``training/data/verify_cube_layout.py`` runs roundtrip ERP -> atlas -> ERP and
also has an ``--against-colleague`` mode that imports the colleague's
``build_2x4_grid`` directly and asserts pixel-parity (atol=1 uint8) for the
forward path. The inverse (``atlas_to_erp``) has no colleague-side reference
implementation; we verify it by roundtrip only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


VALID_LAYOUTS = ("2x4_padding",)   # name kept; geometry redefined 2026-05-29


# Face indices into the (B, 6, C, S, S) tensor returned by
# ``_erp_sample_six_faces``. This is our OWN convention defined in this
# module; do NOT confuse with face indices used elsewhere in the repo.
FACE_PX = 0  # +X / right
FACE_NX = 1  # -X / left
FACE_PY = 2  # +Y / top
FACE_NY = 3  # -Y / bottom
FACE_PZ = 4  # +Z / front
FACE_NZ = 5  # -Z / back


_REQUIRED_MULTIPLE = 16


# Per-column rotations for the 2x4 layout. Order: column 0..3 (L, F, R, B).
# k is the np.rot90 / torch.rot90 argument: 0=no rot, 1=CCW 90, 2=180, 3=CW 90.
# These match the colleague's TOP_LAYOUT / BOTTOM_LAYOUT exactly.
TOP_LAYOUT_K = (1, 0, 3, 2)     # rotations for FACE_PY (post-flip) per column
BOTTOM_LAYOUT_K = (3, 0, 1, 2)  # rotations for FACE_NY (post-flip) per column

EQUATOR_ORDER = ("L", "F", "R", "B")
EQUATOR_FACE_INDEX = {
    "L": FACE_NX,
    "F": FACE_PZ,
    "R": FACE_PX,
    "B": FACE_NZ,
}


def validate_dims(face_size: int, grid_pad: int) -> None:
    """Both ``face_size`` and ``grid_pad`` must be multiples of 16 (mask_encoder
    Conv2d stride 16 acts directly on the grid; the VAE 8x downsample +
    transformer 2x2 patch packing are stricter than 16). ``face_size`` must
    also be even (the half-strip is S/2)."""
    if face_size % _REQUIRED_MULTIPLE != 0:
        raise ValueError(
            f"cube_face_size={face_size} must be divisible by "
            f"{_REQUIRED_MULTIPLE} (mask_encoder Conv2d stride). "
            f"Common safe values: 256 / 384 / 480 / 512 / 768."
        )
    if grid_pad % _REQUIRED_MULTIPLE != 0:
        raise ValueError(
            f"cube_grid_pad={grid_pad} must be divisible by "
            f"{_REQUIRED_MULTIPLE}. Common safe values: 16 / 32 / 64."
        )
    if face_size % 2 != 0:
        raise ValueError(
            f"cube_face_size={face_size} must be even (half-strip = S/2)."
        )


def grid_shape(face_size: int, grid_pad: int) -> tuple[int, int]:
    """Return (H_grid, W_grid) for the 2x4 cropped-repeat padding layout (v2).

    H_grid = 2 * face_size + 2 * grid_pad   (vertical buffers added by v2)
    W_grid = 4 * face_size +     grid_pad   (left wrap column, same as v1)
    """
    return (2 * face_size + 2 * grid_pad, 4 * face_size + grid_pad)


# ---------------------------------------------------------------------------
# ERP -> six cube faces (canonical convention, unchanged from v1).
# These functions match py360convert.e2p for the four equator faces; for
# +Y / -Y the output differs by a vertical flip -- corrected at assembly time.
# ---------------------------------------------------------------------------
def _face_grid(face_size: int, face_index: int, device, dtype) -> torch.Tensor:
    """Per-pixel normalized 3D direction vectors for one cube face,
    using viewer-from-inside convention. Returns (face_size, face_size, 3)."""
    rng = torch.linspace(-1.0, 1.0, face_size, device=device, dtype=dtype)
    s = rng.view(1, -1).expand(face_size, -1)         # +1 = right col, -1 = left col
    t = -rng.view(-1, 1).expand(-1, face_size)        # +1 = top row,   -1 = bottom row
    one = torch.ones_like(s)

    if face_index == FACE_PX:    # +X: up=+Y => t->y; right=-Z => s->-z
        xyz = torch.stack([one,  t, -s], dim=-1)
    elif face_index == FACE_NX:  # -X: up=+Y, right=+Z
        xyz = torch.stack([-one, t,  s], dim=-1)
    elif face_index == FACE_PY:  # +Y: up=+Z, right=+X
        xyz = torch.stack([s,  one,  t], dim=-1)
    elif face_index == FACE_NY:  # -Y: up=-Z, right=+X
        xyz = torch.stack([s, -one, -t], dim=-1)
    elif face_index == FACE_PZ:  # +Z: up=+Y, right=+X
        xyz = torch.stack([s,  t,  one], dim=-1)
    elif face_index == FACE_NZ:  # -Z: up=+Y, right=-X
        xyz = torch.stack([-s, t, -one], dim=-1)
    else:
        raise ValueError(f"face_index must be in [0, 6), got {face_index}")

    return xyz / xyz.norm(dim=-1, keepdim=True)


def _erp_sample_six_faces(erp: torch.Tensor, face_size: int) -> torch.Tensor:
    """Sample 6 cube faces from a standard 2:1 ERP using the canonical
    convention (FACE_PY row 0 = +Z seam, FACE_NY row 0 = -Z seam).

    erp: (B, C, H, W). Returns (B, 6, C, face_size, face_size).
    """
    if erp.dim() == 3:
        erp = erp.unsqueeze(0)
    B, _, H, W = erp.shape
    device, dtype = erp.device, erp.dtype

    faces_list = []
    for fi in range(6):
        xyz = _face_grid(face_size, fi, device, dtype)
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        theta = torch.atan2(x, z)
        phi = torch.asin(y.clamp(-1.0, 1.0))
        u = (theta + math.pi) / (2 * math.pi) * (W - 1)
        v = (math.pi / 2 - phi) / math.pi * (H - 1)
        grid_u = u / (W - 1) * 2.0 - 1.0
        grid_v = v / (H - 1) * 2.0 - 1.0
        grid = torch.stack([grid_u, grid_v], dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        face = F.grid_sample(
            erp, grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        faces_list.append(face)
    return torch.stack(faces_list, dim=1)  # (B, 6, C, S, S)


def _flip_pole_faces(faces: torch.Tensor) -> torch.Tensor:
    """Convert FACE_PY / FACE_NY from our canonical convention to py360convert
    convention (vertical flip). After this step:

        FACE_PY: row 0 = -Z (back), row S-1 = +Z (front)
        FACE_NY: row 0 = +Z (front), row S-1 = -Z (back)

    All other faces unchanged. We do this so the downstream rotation +
    cropping match the colleague's ``build_2x4_grid`` bit-for-bit.
    """
    out = faces.clone()
    out[:, FACE_PY] = out[:, FACE_PY].flip(-2)
    out[:, FACE_NY] = out[:, FACE_NY].flip(-2)
    return out


# ---------------------------------------------------------------------------
# Forward: ERP -> 2x4 cropped-repeat cube grid
# ---------------------------------------------------------------------------
def erp_to_2x4_padding_grid(
    erp: torch.Tensor,
    face_size: int,
    grid_pad: int,
    interp_mode: str = "bilinear",
) -> torch.Tensor:
    """Sample a 2:1 standard ERP tensor into the 2x4 cropped-repeat padding grid (v2).

    ``erp`` : (B, C, H, W) or (C, H, W). Standard convention (north pole at v=0).
    Will be resized to (2*face_size, 4*face_size) first if it isn't already.

    Returns: (B, C, H_grid, W_grid) or (C, H_grid, W_grid) where
    H_grid = 2*face_size + 2*grid_pad and W_grid = 4*face_size + grid_pad.
    """
    validate_dims(face_size, grid_pad)
    squeeze = (erp.dim() == 3)
    if squeeze:
        erp = erp.unsqueeze(0)

    _, _, H, W = erp.shape
    target_H = 2 * face_size
    target_W = 4 * face_size
    if (H, W) != (target_H, target_W):
        if interp_mode == "nearest":
            erp = F.interpolate(erp, size=(target_H, target_W), mode="nearest")
        else:
            erp = F.interpolate(
                erp, size=(target_H, target_W),
                mode=interp_mode, align_corners=False,
            )

    faces = _erp_sample_six_faces(erp, face_size)
    faces = _flip_pole_faces(faces)

    B, _, C, S, _ = faces.shape
    half = face_size // 2
    pad = grid_pad
    H_grid, W_grid = grid_shape(face_size, grid_pad)
    content_w = 4 * face_size

    grid = faces.new_zeros((B, C, H_grid, W_grid))

    # Top band: dst rows [0, half+pad), src rows [S-(half+pad), S)
    src_y0_top, src_y1_top = S - (half + pad), S
    dst_y0_top, dst_y1_top = 0, half + pad
    for col, k in enumerate(TOP_LAYOUT_K):
        rotated = torch.rot90(faces[:, FACE_PY], k=k, dims=(-2, -1))
        x0 = pad + col * S
        grid[:, :, dst_y0_top:dst_y1_top, x0:x0 + S] = rotated[..., src_y0_top:src_y1_top, :]

    # Equator band: dst rows [half+pad, half+pad+S), full S rows of equator face
    dst_y0_eq, dst_y1_eq = half + pad, half + pad + S
    for col, edge in enumerate(EQUATOR_ORDER):
        x0 = pad + col * S
        grid[:, :, dst_y0_eq:dst_y1_eq, x0:x0 + S] = faces[:, EQUATOR_FACE_INDEX[edge]]

    # Bottom band: dst rows [half+pad+S, 2*S+2*pad), src rows [0, half+pad)
    src_y0_bot, src_y1_bot = 0, half + pad
    dst_y0_bot, dst_y1_bot = half + pad + S, 2 * S + 2 * pad
    for col, k in enumerate(BOTTOM_LAYOUT_K):
        rotated = torch.rot90(faces[:, FACE_NY], k=k, dims=(-2, -1))
        x0 = pad + col * S
        grid[:, :, dst_y0_bot:dst_y1_bot, x0:x0 + S] = rotated[..., src_y0_bot:src_y1_bot, :]

    # Left wrap column: copy the rightmost ``pad`` content columns.
    grid[:, :, :, :pad] = grid[:, :, :, content_w:content_w + pad]

    if squeeze:
        grid = grid.squeeze(0)
    return grid


# ---------------------------------------------------------------------------
# 4-equator-face row layout ("triptych" variant, 2026-06).
# ---------------------------------------------------------------------------
# A single horizontal strip of the four equator cube faces in [L, F, R, B]
# order (NO ±Y poles, NO pole-padding, NO left-wrap column). This is the
# per-row building block for the 3-row triptych canvas (top = GT, middle =
# warp, bottom = start) used by the single-canvas spatial-inpainting variant.
#
# The faces are the *exact* equator faces of the verified ``2x4_padding``
# layout (same convention, same EQUATOR_ORDER / EQUATOR_FACE_INDEX), so the
# geometry matches the rest of the repo. We deliberately keep it minimal:
#   * no poles               -> only the equatorial 90deg-FOV band is modelled.
#   * no inter-face padding   -> faces are simply concatenated edge-to-edge.
#   * no wrap column          -> the L/B seam is a hard adjacency (the trainer
#                                encodes each row independently, so a wrap
#                                column would only matter for ERP continuity,
#                                which this layout does not promise).
#
# Returned strip shape: (B, C, face_size, 4*face_size). With face_size=512 the
# strip is 512 x 2048; VAE/8 -> 64 x 256; the trainer stacks three such latent
# rows along height into one (C, 3*64, 256) canvas latent.

FOUR_FACE_ORDER = EQUATOR_ORDER  # ("L", "F", "R", "B")


def validate_face_size(face_size: int) -> None:
    """``face_size`` must be a multiple of 16 (VAE 8x + transformer 2x2 patch)
    and even. Mirrors ``validate_dims`` minus the ``grid_pad`` checks."""
    if face_size % _REQUIRED_MULTIPLE != 0:
        raise ValueError(
            f"cube_face_size={face_size} must be divisible by "
            f"{_REQUIRED_MULTIPLE} (VAE 8x downsample + transformer 2x2 "
            f"patch packing). Common safe values: 256 / 384 / 480 / 512."
        )
    if face_size % 2 != 0:
        raise ValueError(f"cube_face_size={face_size} must be even.")


def grid_shape_4face(face_size: int) -> tuple[int, int]:
    """(H, W) of one 4-equator-face row: ``(face_size, 4 * face_size)``."""
    return (face_size, 4 * face_size)


def erp_to_4face_row(
    erp: torch.Tensor,
    face_size: int,
) -> torch.Tensor:
    """Sample a 2:1 ERP into a horizontal strip of the 4 equator cube faces.

    ``erp`` : (B, C, H, W) or (C, H, W), standard convention (north pole at
    v=0). Sampled directly at ``face_size`` (no pre-resize needed -- the
    underlying ``_erp_sample_six_faces`` grid_samples at the requested face
    resolution).

    Returns ``(B, C, face_size, 4*face_size)`` or ``(C, face_size,
    4*face_size)``: faces concatenated left-to-right as [L, F, R, B] =
    [-X, +Z, +X, -Z].
    """
    validate_face_size(face_size)
    squeeze = (erp.dim() == 3)
    if squeeze:
        erp = erp.unsqueeze(0)

    faces = _erp_sample_six_faces(erp, face_size)  # (B, 6, C, S, S)
    cols = [faces[:, EQUATOR_FACE_INDEX[edge]] for edge in EQUATOR_ORDER]
    row = torch.cat(cols, dim=-1)                  # (B, C, S, 4S)

    if squeeze:
        row = row.squeeze(0)
    return row


# ---------------------------------------------------------------------------
# Inverse: 4-equator-face row -> ERP  (triptych variant, M4 inference)
# ---------------------------------------------------------------------------
def four_face_row_to_erp(
    row: torch.Tensor,
    face_size: int,
    erp_h: int,
    erp_w: int,
    interp_mode: str = "bilinear",
    return_mask: bool = False,
):
    """Inverse of :func:`erp_to_4face_row`: re-project a 4-equator-face strip
    ``[L, F, R, B]`` back to a standard 2:1 ERP.

    The 4 equator faces only cover the equatorial ``±45°`` latitude band, so the
    poles (``|latitude| > 45°``) have NO source pixels. Those ERP rows are filled
    with zeros (black) and flagged ``0`` in the coverage mask. This is the
    expected behaviour for the triptych variant (no sky / no ground), see
    ``docs/triptych_4face_variant.md`` §7.

    Args:
        row: ``(B, C, S, 4S)`` or ``(C, S, 4S)`` strip, faces concatenated
            left-to-right as ``[L, F, R, B] = [-X, +Z, +X, -Z]`` (exactly what
            :func:`erp_to_4face_row` produces). ``S == face_size``.
        face_size: per-face pixel size ``S``.
        erp_h, erp_w: target ERP shape (``erp_w == 2 * erp_h`` is standard).
        interp_mode: ``"bilinear"`` (default) or ``"nearest"``.
        return_mask: when True, also return a ``(B, 1, erp_h, erp_w)`` (or
            ``(1, erp_h, erp_w)``) float coverage mask (1 = covered equator band,
            0 = pole / no data).

    Returns:
        ``erp`` of shape ``(B, C, erp_h, erp_w)`` / ``(C, erp_h, erp_w)``, or
        ``(erp, mask)`` when ``return_mask=True``.
    """
    validate_face_size(face_size)
    squeeze = (row.dim() == 3)
    if squeeze:
        row = row.unsqueeze(0)
    B, _, H_row, W_row = row.shape
    S = face_size
    if (H_row, W_row) != (S, 4 * S):
        raise ValueError(
            f"row spatial dims {H_row}x{W_row} != expected {S}x{4 * S} "
            f"for face_size={face_size}"
        )

    device, dtype = row.device, row.dtype
    eps = 1e-12

    # ERP pixel -> world direction (same convention as _erp_sample_six_faces /
    # atlas_to_erp: theta = atan2(x, z), v measured from the north pole).
    j_erp = torch.arange(erp_w, device=device, dtype=dtype)
    i_erp = torch.arange(erp_h, device=device, dtype=dtype)
    u_erp = j_erp.view(1, -1).expand(erp_h, -1)
    v_erp = i_erp.view(-1, 1).expand(-1, erp_w)
    theta = u_erp / max(erp_w - 1, 1) * (2.0 * math.pi) - math.pi
    phi = (math.pi / 2.0) - v_erp / max(erp_h - 1, 1) * math.pi
    cos_phi = torch.cos(phi)
    x = cos_phi * torch.sin(theta)
    y = torch.sin(phi)
    z = cos_phi * torch.cos(theta)
    ax, ay, az = x.abs(), y.abs(), z.abs()

    # Equator-face classification. Pole pixels (ay dominates) get no face and
    # are masked out below.
    on_x = (ax >= ay) & (ax >= az)
    on_z = (az > ay) & (az >= ax)
    on_PX = on_x & (x >= 0)
    on_NX = on_x & (x < 0)
    on_PZ = on_z & (z >= 0)
    on_NZ = on_z & (z < 0)
    covered = on_PX | on_NX | on_PZ | on_NZ

    def _st_to_face(s_, t_):
        i_f = (1.0 - t_) * (S - 1) * 0.5
        j_f = (s_ + 1.0) * (S - 1) * 0.5
        return i_f, j_f

    y_atl = torch.zeros_like(x)
    x_atl = torch.zeros_like(x)

    # NX -> slot L (col 0): s = z/ax, t = y/ax
    inv = 1.0 / ax.clamp(min=eps)
    i_f, j_f = _st_to_face(z * inv, y * inv)
    y_atl = torch.where(on_NX, i_f, y_atl)
    x_atl = torch.where(on_NX, 0 * S + j_f, x_atl)

    # PZ -> slot F (col 1): s = x/az, t = y/az
    inv = 1.0 / az.clamp(min=eps)
    i_f, j_f = _st_to_face(x * inv, y * inv)
    y_atl = torch.where(on_PZ, i_f, y_atl)
    x_atl = torch.where(on_PZ, 1 * S + j_f, x_atl)

    # PX -> slot R (col 2): s = -z/ax, t = y/ax
    inv = 1.0 / ax.clamp(min=eps)
    i_f, j_f = _st_to_face(-z * inv, y * inv)
    y_atl = torch.where(on_PX, i_f, y_atl)
    x_atl = torch.where(on_PX, 2 * S + j_f, x_atl)

    # NZ -> slot B (col 3): s = -x/az, t = y/az
    inv = 1.0 / az.clamp(min=eps)
    i_f, j_f = _st_to_face(-x * inv, y * inv)
    y_atl = torch.where(on_NZ, i_f, y_atl)
    x_atl = torch.where(on_NZ, 3 * S + j_f, x_atl)

    grid_x = x_atl / max(W_row - 1, 1) * 2.0 - 1.0
    grid_y = y_atl / max(H_row - 1, 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    erp = F.grid_sample(
        row, grid,
        mode=interp_mode, padding_mode="border", align_corners=True,
    )
    mask = covered.to(dtype).view(1, 1, erp_h, erp_w)
    erp = erp * mask  # zero the uncovered poles

    if squeeze:
        erp = erp.squeeze(0)
        mask = mask.squeeze(0)
    if return_mask:
        return erp, mask
    return erp


# ---------------------------------------------------------------------------
# Inverse: 2x4 cropped-repeat cube grid -> ERP
# ---------------------------------------------------------------------------
def _rot90_forward_coords(i: torch.Tensor, j: torch.Tensor, k: int, S: int):
    """Apply forward np.rot90(..., k) to face-local pixel coords (i, j).
    Returns (u, v) = rotated-face coords such that ``rotated[u, v] = face[i, j]``.

    Derivation:
        np.rot90(M, k=1)[i, j] = M[j, S-1-i]   ⇒ given M[i_old, j_old], the
            corresponding pixel in the rotated array is at (S-1-j_old, i_old).
        np.rot90(M, k=2)[i, j] = M[S-1-i, S-1-j]
        np.rot90(M, k=3)[i, j] = M[S-1-j, i]   ⇒ rotated coords: (j_old, S-1-i_old)
    """
    if k == 0:
        return i, j
    if k == 1:
        return (S - 1) - j, i
    if k == 2:
        return (S - 1) - i, (S - 1) - j
    if k == 3:
        return j, (S - 1) - i
    raise ValueError(f"k must be in {{0,1,2,3}}, got {k}")


def atlas_to_erp(
    atlas: torch.Tensor,
    face_size: int,
    grid_pad: int,
    erp_h: int,
    erp_w: int,
    interp_mode: str = "bilinear",
) -> torch.Tensor:
    """Inverse of ``erp_to_2x4_padding_grid``: re-project a v2 cube atlas back
    to a standard 2:1 ERP.

    Strategy: for each ERP pixel (u, v) we compute the world direction (x, y, z),
    classify into one of the 6 cube faces by argmax(|x|, |y|, |z|), then derive
    face-local pixel coords (i_face, j_face) in the **post-flip** convention
    (matching what's stored in the atlas). For pole faces we additionally choose
    a primary atlas column based on which equator seam the (s, t) is closest to,
    apply that column's forward rotation to (i_face, j_face) -> (u, v), and
    finally compute (atlas_y, atlas_x) accounting for the band's src-row offset.

    Args:
        atlas: ``(B, C, H_grid, W_grid)`` or ``(C, H_grid, W_grid)``.
        face_size, grid_pad: must match erp_to_2x4_padding_grid's args.
        erp_h, erp_w: target ERP shape. ``erp_w == 2 * erp_h`` is standard.
        interp_mode: ``"bilinear"`` (default) or ``"nearest"``.

    Returns: ``(B, C, erp_h, erp_w)`` or ``(C, erp_h, erp_w)``.
    """
    validate_dims(face_size, grid_pad)
    squeeze = (atlas.dim() == 3)
    if squeeze:
        atlas = atlas.unsqueeze(0)
    B, _, H_grid, W_grid = atlas.shape
    expected_h, expected_w = grid_shape(face_size, grid_pad)
    if (H_grid, W_grid) != (expected_h, expected_w):
        raise ValueError(
            f"atlas spatial dims {H_grid}x{W_grid} != expected "
            f"{expected_h}x{expected_w} for face_size={face_size}, "
            f"grid_pad={grid_pad}"
        )

    device, dtype = atlas.device, atlas.dtype
    S = face_size
    pad = grid_pad
    half = S // 2

    # Atlas band y-offsets (forward layout, for inverse coordinate math)
    eq_y0 = half + pad             # equator band starts at this atlas row
    eq_y1 = eq_y0 + S              # ... and ends here
    # TOP band: src_rows = [S-(half+pad), S), dst_rows = [0, half+pad)
    #   ⇒ atlas_y = u_rotated - (S - (half+pad)) = u - S + half + pad
    top_offset = S - (half + pad)  # subtract this from u to get atlas_y
    # BOTTOM band: src_rows = [0, half+pad), dst_rows = [eq_y1, eq_y1 + half+pad)
    #   ⇒ atlas_y = u_rotated + eq_y1
    bot_offset = eq_y1             # add this to u to get atlas_y

    # ------------------------------------------------------------------
    # ERP pixel -> world direction
    # ------------------------------------------------------------------
    j_erp = torch.arange(erp_w, device=device, dtype=dtype)
    i_erp = torch.arange(erp_h, device=device, dtype=dtype)
    u_erp = j_erp.view(1, -1).expand(erp_h, -1)
    v_erp = i_erp.view(-1, 1).expand(-1, erp_w)
    theta = u_erp / max(erp_w - 1, 1) * (2.0 * math.pi) - math.pi
    phi = (math.pi / 2.0) - v_erp / max(erp_h - 1, 1) * math.pi
    cos_phi = torch.cos(phi)
    x = cos_phi * torch.sin(theta)
    y = torch.sin(phi)
    z = cos_phi * torch.cos(theta)
    ax = x.abs()
    ay = y.abs()
    az = z.abs()

    on_x = (ax >= ay) & (ax >= az)
    on_y = (~on_x) & (ay >= az)
    on_z = (~on_x) & (~on_y)
    on_PX = on_x & (x >= 0)
    on_NX = on_x & (x < 0)
    on_PY = on_y & (y >= 0)
    on_NY = on_y & (y < 0)
    on_PZ = on_z & (z >= 0)
    on_NZ = on_z & (z < 0)

    # ------------------------------------------------------------------
    # Per-face atlas-coord derivation
    # ------------------------------------------------------------------
    y_atl = torch.zeros_like(x)
    x_atl = torch.zeros_like(x)
    eps = 1e-12

    def _st_to_face_canon(s_, t_):
        """Canonical face-local (i, j) ∈ [0, S) from face-local (s, t) ∈ [-1, 1].

        Matches ``_face_grid`` convention: i = (1 - t) * (S-1) / 2,
        j = (s + 1) * (S-1) / 2.
        """
        i_f = (1.0 - t_) * (S - 1) * 0.5
        j_f = (s_ + 1.0) * (S - 1) * 0.5
        return i_f, j_f

    # ---- PX (slot R, col=2) -----------------------------------------
    inv = 1.0 / ax.clamp(min=eps)
    s_ = -z * inv
    t_ =  y * inv
    i_f, j_f = _st_to_face_canon(s_, t_)
    y_atl = torch.where(on_PX, eq_y0 + i_f, y_atl)
    x_atl = torch.where(on_PX, pad + 2 * S + j_f, x_atl)

    # ---- NX (slot L, col=0) -----------------------------------------
    inv = 1.0 / ax.clamp(min=eps)
    s_ =  z * inv
    t_ =  y * inv
    i_f, j_f = _st_to_face_canon(s_, t_)
    y_atl = torch.where(on_NX, eq_y0 + i_f, y_atl)
    x_atl = torch.where(on_NX, pad + 0 * S + j_f, x_atl)

    # ---- PZ (slot F, col=1) -----------------------------------------
    inv = 1.0 / az.clamp(min=eps)
    s_ =  x * inv
    t_ =  y * inv
    i_f, j_f = _st_to_face_canon(s_, t_)
    y_atl = torch.where(on_PZ, eq_y0 + i_f, y_atl)
    x_atl = torch.where(on_PZ, pad + 1 * S + j_f, x_atl)

    # ---- NZ (slot B, col=3) -----------------------------------------
    inv = 1.0 / az.clamp(min=eps)
    s_ = -x * inv
    t_ =  y * inv
    i_f, j_f = _st_to_face_canon(s_, t_)
    y_atl = torch.where(on_NZ, eq_y0 + i_f, y_atl)
    x_atl = torch.where(on_NZ, pad + 3 * S + j_f, x_atl)

    # ---- PY (top face, appears 4× rotated, choose primary by closest seam)
    # Canonical PY face: xyz = (s, +1, t) with s = x/y, t = z/y (y > 0).
    # After flip (post-_flip_pole_faces), face row 0 is -Z (back) and
    # face col 0 is -X (left). The mapping (s_canon, t_canon) -> face-local
    # canonical pixel is _st_to_face_canon(s_canon, t_canon). Then we convert
    # to py360 (post-flip) coords:
    #   i_post = (S-1) - i_canon
    #   j_post = j_canon
    # Primary column logic uses the SAME (s_canon, t_canon):
    #   prim_F (col 1): t >=  |s|
    #   prim_B (col 3): -t >= |s|
    #   prim_L (col 0): -s >  |t|
    #   prim_R (col 2):  s >  |t|
    # Per column, apply forward rot90(k) to (i_post, j_post) -> (u, v),
    # then atlas_y = u - top_offset, atlas_x = pad + col*S + v.
    inv_y = 1.0 / ay.clamp(min=eps)
    s_PY = x * inv_y
    t_PY = z * inv_y
    abs_s = s_PY.abs()
    abs_t = t_PY.abs()
    prim_F = (t_PY >= abs_s)
    prim_B = (~prim_F) & (-t_PY >= abs_s)
    prim_L = (~prim_F) & (~prim_B) & (s_PY <= 0)
    prim_R = (~prim_F) & (~prim_B) & (~prim_L)
    i_canon, j_canon = _st_to_face_canon(s_PY, t_PY)
    i_post = (S - 1) - i_canon
    j_post = j_canon

    # rot90 forward per col:
    # col 0 (L), k=1: (u, v) = (S-1-j_post, i_post)
    # col 1 (F), k=0: (u, v) = (i_post, j_post)
    # col 2 (R), k=3: (u, v) = (j_post, S-1-i_post)
    # col 3 (B), k=2: (u, v) = (S-1-i_post, S-1-j_post)
    u_PY = torch.where(
        prim_F, i_post,
        torch.where(
            prim_B, (S - 1) - i_post,
            torch.where(
                prim_L, (S - 1) - j_post,
                j_post,
            ),
        ),
    )
    v_PY = torch.where(
        prim_F, j_post,
        torch.where(
            prim_B, (S - 1) - j_post,
            torch.where(
                prim_L, i_post,
                (S - 1) - i_post,
            ),
        ),
    )
    col_PY = torch.where(
        prim_F, torch.full_like(i_post, 1),
        torch.where(
            prim_B, torch.full_like(i_post, 3),
            torch.where(
                prim_L, torch.full_like(i_post, 0),
                torch.full_like(i_post, 2),
            ),
        ),
    )
    y_PY = u_PY - top_offset
    x_PY = pad + col_PY * S + v_PY
    y_atl = torch.where(on_PY, y_PY, y_atl)
    x_atl = torch.where(on_PY, x_PY, x_atl)

    # ---- NY (bottom face, appears 4× rotated)
    # Canonical NY face: xyz = (s, -1, -t) with s = x/|y|, t = -z/|y| (y < 0).
    # After flip, face row 0 is +Z (front), row S-1 is -Z (back); col 0 is
    # -X (left), col S-1 is +X (right).
    # Primary column choice (uses canonical s, t):
    #   prim_F (col 1): -t >= |s|        # FRONT side, z > 0
    #   prim_B (col 3):  t >= |s|        # BACK side, z < 0
    #   prim_L (col 0): -s >  |t|
    #   prim_R (col 2):  s >  |t|
    # BOTTOM_LAYOUT_K = (3, 0, 1, 2) per col. atlas_y = u + bot_offset.
    inv_y = 1.0 / ay.clamp(min=eps)
    s_NY = x * inv_y
    t_NY = -z * inv_y
    abs_s = s_NY.abs()
    abs_t = t_NY.abs()
    prim_F = (-t_NY >= abs_s)
    prim_B = (~prim_F) & (t_NY >= abs_s)
    prim_L = (~prim_F) & (~prim_B) & (s_NY <= 0)
    prim_R = (~prim_F) & (~prim_B) & (~prim_L)
    i_canon, j_canon = _st_to_face_canon(s_NY, t_NY)
    i_post = (S - 1) - i_canon
    j_post = j_canon

    # rot90 forward per col:
    # col 0 (L), k=3: (u, v) = (j_post, S-1-i_post)
    # col 1 (F), k=0: (u, v) = (i_post, j_post)
    # col 2 (R), k=1: (u, v) = (S-1-j_post, i_post)
    # col 3 (B), k=2: (u, v) = (S-1-i_post, S-1-j_post)
    u_NY = torch.where(
        prim_F, i_post,
        torch.where(
            prim_B, (S - 1) - i_post,
            torch.where(
                prim_L, j_post,
                (S - 1) - j_post,
            ),
        ),
    )
    v_NY = torch.where(
        prim_F, j_post,
        torch.where(
            prim_B, (S - 1) - j_post,
            torch.where(
                prim_L, (S - 1) - i_post,
                i_post,
            ),
        ),
    )
    col_NY = torch.where(
        prim_F, torch.full_like(i_post, 1),
        torch.where(
            prim_B, torch.full_like(i_post, 3),
            torch.where(
                prim_L, torch.full_like(i_post, 0),
                torch.full_like(i_post, 2),
            ),
        ),
    )
    y_NY = u_NY + bot_offset
    x_NY = pad + col_NY * S + v_NY
    y_atl = torch.where(on_NY, y_NY, y_atl)
    x_atl = torch.where(on_NY, x_NY, x_atl)

    # ------------------------------------------------------------------
    # grid_sample
    # ------------------------------------------------------------------
    grid_x = x_atl / max(W_grid - 1, 1) * 2.0 - 1.0
    grid_y = y_atl / max(H_grid - 1, 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    erp = F.grid_sample(
        atlas, grid,
        mode=interp_mode, padding_mode="border", align_corners=True,
    )
    if squeeze:
        erp = erp.squeeze(0)
    return erp


# ---------------------------------------------------------------------------
# Inverse: 2x4 cropped-repeat cube grid -> 6 canonical cube faces
# ---------------------------------------------------------------------------
# Diagnostic-only path. Recovers full S×S cube faces from the cropped-repeat
# 2x4 atlas (the actual model output during inference). Equator faces are a
# direct slice; ±Y faces require fusing 4 partial-rotation strips.
#
# This is the *inverse* of `erp_to_2x4_padding_grid` in face-space. The
# fused ±Y faces have detail loss inside the central 2*pad blend region
# and along the |y|=|x| polar diagonal, but reach the ground-truth pixels
# everywhere else. See CubeMap_Related_Work/2x4_padding_strategy.md §7-§9.

def _polar_blend_alpha_mask(
    face_size: int,
    feather: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Returns (1, 1, S, S) alpha mask. alpha=1 → use FB; alpha=0 → use LR.

    For pixel at face-local (y_norm, x_norm) ∈ [-1, 1]^2:
        d = |y_norm| - |x_norm|
        alpha = clip((d + feather) / (2 * feather), 0, 1)

    Top/bottom edges of face (|y|>>|x|): alpha → 1, trust FB strip pair.
    Left/right edges of face (|x|>>|y|): alpha → 0, trust LR strip pair.
    Diagonal: narrow feather blend.
    """
    rng = torch.linspace(-1.0, 1.0, face_size, device=device, dtype=dtype)
    y = rng.abs().view(-1, 1)
    x = rng.abs().view(1, -1)
    d = y - x
    alpha = ((d + feather) / (2.0 * feather)).clamp(0.0, 1.0)
    return alpha.view(1, 1, face_size, face_size)


def _blend_fb_strips(
    top_part: torch.Tensor,
    bot_part: torch.Tensor,
    face_size: int,
    pad: int,
) -> torch.Tensor:
    """Linearly alpha-blend two partial face strips along the row axis.

    `top_part` (B, C, half+pad, S) is placed at face rows [0, half+pad).
    `bot_part` (B, C, half+pad, S) is placed at face rows [S-(half+pad), S).
    They overlap in face rows [half-pad, half+pad) (= 2*pad rows in the
    middle); pure regions on the outside are taken verbatim from each
    strip.

    Returns (B, C, S, S).
    """
    half = face_size // 2
    out = top_part.new_zeros(*top_part.shape[:-2], face_size, face_size)

    # Pure top: face rows [0, half-pad)  ← top_part rows [0, half-pad)
    if half - pad > 0:
        out[..., 0:half - pad, :] = top_part[..., 0:half - pad, :]
    # Pure bot: face rows [half+pad, S)  ← bot_part rows [2*pad, half+pad)
    if face_size - (half + pad) > 0:
        out[..., half + pad:face_size, :] = bot_part[..., 2 * pad:half + pad, :]

    overlap_h = 2 * pad
    if overlap_h > 0:
        # Overlap region: face rows [half-pad, half+pad)
        # top_part local rows [half-pad, half+pad)
        # bot_part local rows [0, 2*pad)
        ot = top_part[..., half - pad:half + pad, :]
        ob = bot_part[..., 0:2 * pad, :]
        alpha = torch.linspace(
            1.0, 0.0, overlap_h,
            device=top_part.device, dtype=top_part.dtype,
        )
        alpha = alpha.view(*([1] * (top_part.dim() - 2)), overlap_h, 1)
        out[..., half - pad:half + pad, :] = ot * alpha + ob * (1.0 - alpha)
    return out


def _blend_lr_strips(
    left_part: torch.Tensor,
    right_part: torch.Tensor,
    face_size: int,
    pad: int,
) -> torch.Tensor:
    """Column-axis analog of _blend_fb_strips.

    `left_part` (B, C, S, half+pad)  → face cols [0, half+pad)
    `right_part`(B, C, S, half+pad)  → face cols [S-(half+pad), S)

    Returns (B, C, S, S).
    """
    half = face_size // 2
    out = left_part.new_zeros(*left_part.shape[:-2], face_size, face_size)

    if half - pad > 0:
        out[..., :, 0:half - pad] = left_part[..., :, 0:half - pad]
    if face_size - (half + pad) > 0:
        out[..., :, half + pad:face_size] = right_part[..., :, 2 * pad:half + pad]

    overlap_w = 2 * pad
    if overlap_w > 0:
        ol = left_part[..., :, half - pad:half + pad]
        orr = right_part[..., :, 0:2 * pad]
        alpha = torch.linspace(
            1.0, 0.0, overlap_w,
            device=left_part.device, dtype=left_part.dtype,
        )
        alpha = alpha.view(*([1] * (left_part.dim() - 2)), 1, overlap_w)
        out[..., :, half - pad:half + pad] = ol * alpha + orr * (1.0 - alpha)
    return out


def atlas_to_six_faces(
    atlas: torch.Tensor,
    face_size: int,
    grid_pad: int,
    feather: float = 0.10,
) -> dict:
    """Recover 6 cube faces in our canonical convention from a 2x4_padding atlas.

    `atlas` : (B, C, H_grid, W_grid) or (C, H_grid, W_grid) with
        H_grid = 2*S + 2*pad and W_grid = 4*S + pad.

    Returns a dict {name: tensor (B, C, S, S) or (C, S, S)} for keys::
        "front" (FACE_PZ), "back" (FACE_NZ),
        "left"  (FACE_NX), "right"(FACE_PX),
        "top"   (FACE_PY canonical, NOT post-flip),
        "bottom"(FACE_NY canonical, NOT post-flip).

    Equator faces are exact-pixel from the atlas. ±Y faces are reconstructed
    via FB strip fusion + LR strip fusion + polar alpha blend (per
    CubeMap_Related_Work/2x4_padding_strategy.md).
    """
    validate_dims(face_size, grid_pad)
    squeeze = atlas.dim() == 3
    if squeeze:
        atlas = atlas.unsqueeze(0)
    _, _, H_grid, W_grid = atlas.shape
    expected_H, expected_W = grid_shape(face_size, grid_pad)
    assert (H_grid, W_grid) == (expected_H, expected_W), (
        f"atlas shape mismatch: got ({H_grid},{W_grid}), "
        f"expected ({expected_H},{expected_W})"
    )

    S = face_size
    pad = grid_pad
    half = S // 2

    eq_y0, eq_y1 = half + pad, half + pad + S
    top_y0, top_y1 = 0, half + pad
    bot_y0, bot_y1 = half + pad + S, 2 * S + 2 * pad

    def col_x(col: int) -> int:
        return pad + col * S

    # ---- 1) Equator faces — direct slice (atlas pixels are face pixels) ----
    left_face  = atlas[..., eq_y0:eq_y1, col_x(0):col_x(0) + S].clone()
    front_face = atlas[..., eq_y0:eq_y1, col_x(1):col_x(1) + S].clone()
    right_face = atlas[..., eq_y0:eq_y1, col_x(2):col_x(2) + S].clone()
    back_face  = atlas[..., eq_y0:eq_y1, col_x(3):col_x(3) + S].clone()

    # ---- 2) Top face: 4 strips of FACE_PY (post-flip) recovery ----
    # TOP_LAYOUT_K = (1, 0, 3, 2) — col 0/1/2/3 = L/F/R/B, k = 1/0/3/2
    # Strips slice rows [S-(half+pad), S) of rot90(PY_postflip, k).
    top_l_strip = atlas[..., top_y0:top_y1, col_x(0):col_x(0) + S]  # k=1
    top_f_strip = atlas[..., top_y0:top_y1, col_x(1):col_x(1) + S]  # k=0
    top_r_strip = atlas[..., top_y0:top_y1, col_x(2):col_x(2) + S]  # k=3
    top_b_strip = atlas[..., top_y0:top_y1, col_x(3):col_x(3) + S]  # k=2

    # FB axis: top_b → top half of PY_postflip (after rot180); top_f → bot half
    py_top_half = torch.rot90(top_b_strip, k=2, dims=(-2, -1))   # (B, C, half+pad, S)
    py_bot_half = top_f_strip                                     # (B, C, half+pad, S)
    top_fb = _blend_fb_strips(py_top_half, py_bot_half, S, pad)  # (B, C, S, S)

    # LR axis: top_l → left half cols (rot90 k=3), top_r → right half cols (rot90 k=1)
    py_left_half  = torch.rot90(top_l_strip, k=3, dims=(-2, -1))  # (B, C, S, half+pad)
    py_right_half = torch.rot90(top_r_strip, k=1, dims=(-2, -1))  # (B, C, S, half+pad)
    top_lr = _blend_lr_strips(py_left_half, py_right_half, S, pad)

    # Polar alpha blend
    alpha = _polar_blend_alpha_mask(S, feather, atlas.device, atlas.dtype)
    top_postflip = top_fb * alpha + top_lr * (1.0 - alpha)

    # Undo _flip_pole_faces() to return canonical FACE_PY
    top_face = top_postflip.flip(-2)

    # ---- 3) Bottom face: 4 strips of FACE_NY (post-flip) recovery ----
    # BOTTOM_LAYOUT_K = (3, 0, 1, 2) — col 0/1/2/3 = L/F/R/B, k = 3/0/1/2
    # Strips slice rows [0, half+pad) of rot90(NY_postflip, k).
    bot_l_strip = atlas[..., bot_y0:bot_y1, col_x(0):col_x(0) + S]  # k=3
    bot_f_strip = atlas[..., bot_y0:bot_y1, col_x(1):col_x(1) + S]  # k=0
    bot_r_strip = atlas[..., bot_y0:bot_y1, col_x(2):col_x(2) + S]  # k=1
    bot_b_strip = atlas[..., bot_y0:bot_y1, col_x(3):col_x(3) + S]  # k=2

    # FB axis: bot_f → top half of NY_postflip; bot_b → bot half (after rot180)
    ny_top_half = bot_f_strip
    ny_bot_half = torch.rot90(bot_b_strip, k=2, dims=(-2, -1))
    bottom_fb = _blend_fb_strips(ny_top_half, ny_bot_half, S, pad)

    # LR axis: bot_l → left half (rot90 k=1), bot_r → right half (rot90 k=3)
    ny_left_half  = torch.rot90(bot_l_strip, k=1, dims=(-2, -1))
    ny_right_half = torch.rot90(bot_r_strip, k=3, dims=(-2, -1))
    bottom_lr = _blend_lr_strips(ny_left_half, ny_right_half, S, pad)

    bottom_postflip = bottom_fb * alpha + bottom_lr * (1.0 - alpha)
    bottom_face = bottom_postflip.flip(-2)

    out = {
        "front": front_face,
        "back":  back_face,
        "left":  left_face,
        "right": right_face,
        "top":   top_face,
        "bottom": bottom_face,
    }
    if squeeze:
        out = {k: v.squeeze(0) for k, v in out.items()}
    return out
