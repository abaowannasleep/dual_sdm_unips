"""Photometric prior utilities for the LS-prior teacher branch.

This module implements the "physics sanity" layer of the LS-prior /
teacher-student plan (see ``LS_prior_teacher_student_dev_doc_v2.md``).
It is intentionally *standalone* — no network code is touched here, so
the helpers can be exercised on raw tensors / numpy data with nothing
more than ``torch`` available.

The conventions follow the v2 development document:

* ``S`` (clean shading) has layout ``[B, 3, H, W, N_max]`` with the last
  axis being the per-shot dimension. Channels are ordered ``R, G, B``.
* ``light_param`` has layout ``[B, N_max, 3, 3]``. The leading-most axis
  is the shot index; the inner ``3 x 3`` matrix is
  ``[l_R; l_G; l_B]`` — i.e. row ``c`` is the light direction that
  illuminates colour channel ``c`` in that shot.
* ``mask`` has layout ``[B, 1, H, W]`` with binary values in ``{0, 1}``.
* ``n_pred`` / ``n_gt`` have layout ``[B, 3, H, W]`` with unit-norm
  vectors over the masked region.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Camera-response normalisation (Section 3.2 of the v2 doc)
# ---------------------------------------------------------------------------
def normalize_camera_response(
    S_rgb: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Divide each colour channel by its mask-mean.

    The clean shading is rendered with ``albedo == 1`` but every colour
    channel still carries an unknown response factor ``k_c`` (light
    intensity x quantum efficiency x camera gain). Per-channel
    normalisation absorbs this unknown scalar so the stacked LS
    problem becomes well-posed.

    Parameters
    ----------
    S_rgb: ``[B, 3, H, W]`` (or any leading shape with a channel dim 1).
    mask:  broadcastable to ``S_rgb`` — typically ``[B, 1, H, W]``.

    Returns
    -------
    Tensor with the same shape as ``S_rgb`` where each channel has unit
    mean over the masked region.
    """
    if S_rgb.dim() != 4 or S_rgb.shape[1] != 3:
        raise ValueError(
            f"normalize_camera_response expects [B, 3, H, W]; got {tuple(S_rgb.shape)}"
        )

    m = mask.to(S_rgb.dtype)
    if m.dim() != 4:
        raise ValueError(
            f"normalize_camera_response expects mask [B, 1, H, W]; got {tuple(mask.shape)}"
        )

    masked_sum = (S_rgb * m).sum(dim=[2, 3], keepdim=True)
    masked_cnt = m.sum(dim=[2, 3], keepdim=True).clamp(min=1.0)
    chan_mean = masked_sum / masked_cnt
    return S_rgb / (chan_mean + eps)


# ---------------------------------------------------------------------------
# Condition number via singular values (avoid torch.linalg.cond per the plan)
# ---------------------------------------------------------------------------
def light_matrix_cond(L: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return ``sigma_max / sigma_min`` for one or more light matrices.

    ``torch.linalg.cond`` has been observed to be noisy on tall CUDA
    matrices; the explicit ratio via :func:`torch.linalg.svdvals` is
    stable in both fp32 and fp64.

    Parameters
    ----------
    L: ``[..., m, n]`` light direction matrix. Typical use is
       ``[3*n, 3]`` (stacked observations) or batched ``[B, 3*n, 3]``.

    Returns
    -------
    cond: ``[...]`` (one scalar per leading-batch element).
    """
    sv = torch.linalg.svdvals(L)
    sv_max = sv.amax(dim=-1)
    sv_min = sv.amin(dim=-1)
    return sv_max / (sv_min + eps)


# ---------------------------------------------------------------------------
# Multi-shot least-squares normal estimation
# ---------------------------------------------------------------------------
def _validate_inputs(
    S: torch.Tensor,
    L: torch.Tensor,
    mask: torch.Tensor,
    nImgArray: torch.Tensor,
) -> None:
    if S.dim() != 5 or S.shape[1] != 3:
        raise ValueError(
            f"S must have shape [B, 3, H, W, N_max]; got {tuple(S.shape)}"
        )
    if L.dim() != 4 or L.shape[-1] != 3 or L.shape[-2] != 3:
        raise ValueError(
            f"L must have shape [B, N_max, 3, 3]; got {tuple(L.shape)}"
        )
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(
            f"mask must have shape [B, 1, H, W]; got {tuple(mask.shape)}"
        )
    B = S.shape[0]
    if L.shape[0] != B or mask.shape[0] != B:
        raise ValueError(
            "S, L and mask must share batch size; got "
            f"S:{S.shape[0]}, L:{L.shape[0]}, mask:{mask.shape[0]}"
        )
    if nImgArray.shape != (B,):
        raise ValueError(
            f"nImgArray must be 1-D with B elements; got {tuple(nImgArray.shape)}"
        )


def _stack_observations(
    S_b: torch.Tensor,
    L_b: torch.Tensor,
    mask_b: torch.Tensor,
    n_b: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(I, L_stack)`` for a single batch element.

    Each shot is normalised independently (different shots may have
    different exposure or gain). Output shapes are ``[HW, 3*n_b]`` for
    ``I`` and ``[3*n_b, 3]`` for ``L_stack``.
    """
    # S_b: [3, H, W, n_b]; convert to [n_b, 3, H, W] so the
    # normalisation helper sees a standard NCHW tensor.
    S_shots = S_b.permute(3, 0, 1, 2).contiguous()  # [n_b, 3, H, W]
    mask_exp = mask_b.expand(n_b, -1, -1, -1)        # [n_b, 1, H, W]
    S_norm = normalize_camera_response(S_shots, mask_exp, eps=eps)

    H, W = S_b.shape[1], S_b.shape[2]
    # Collapse (shot, channel) into a single observation axis so each
    # pixel ends up with ``3 * n_b`` scalar observations.
    I_obs = S_norm.permute(2, 3, 0, 1).reshape(H * W, 3 * n_b)

    L_stack = L_b[:n_b].reshape(3 * n_b, 3)
    L_stack = F.normalize(L_stack, dim=-1, eps=eps)
    return I_obs, L_stack


def _stack_observations_raw(
    S_b: torch.Tensor,
    L_b: torch.Tensor,
    n_b: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as :func:`_stack_observations` but without camera-response
    normalisation. Used when the caller has already pre-conditioned
    shading or when the synthetic ``k_c == 1`` assumption holds.

    The observation order in the flattened axis must match the row
    order of ``L_stack`` (which is ``(shot, channel)`` after
    reshaping). We therefore permute to ``[H, W, n_b, 3]`` before
    flattening — mirroring :func:`_stack_observations`.
    """
    H, W = S_b.shape[1], S_b.shape[2]
    # S_b: [3, H, W, N_max] -> select n_b shots -> [3, H, W, n_b]
    S_sel = S_b[..., :n_b]
    # -> [H, W, n_b, 3] so reshape(H*W, 3*n_b) yields (shot, channel) order.
    S_perm = S_sel.permute(1, 2, 3, 0).contiguous()
    I_obs = S_perm.reshape(H * W, 3 * n_b)

    L_stack = L_b[:n_b].reshape(3 * n_b, 3)
    L_stack = F.normalize(L_stack, dim=-1, eps=eps)
    return I_obs, L_stack


def ls_normal_multi_shot(
    S: torch.Tensor,
    L: torch.Tensor,
    mask: torch.Tensor,
    nImgArray: torch.Tensor,
    eps: float = 1e-6,
    normalize_response: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-shot least-squares photometric stereo (no shadow weighting).

    The forward model assumed at each pixel is
    ``I_c^{(k)} = (n . l_c^{(k)})`` after per-channel camera-response
    normalisation. Stacking ``3 * n_b`` observations per pixel and
    solving for ``n`` by ordinary least squares yields a tall system
    that admits an analytic Moore-Penrose solution per pixel; the
    returned ``n_ls`` is the L2-normalised result, masked.

    Parameters
    ----------
    S:         ``[B, 3, H, W, N_max]`` clean shading.
    L:         ``[B, N_max, 3, 3]`` light directions; row ``c`` is the
               light dir for colour channel ``c`` of that shot.
    mask:      ``[B, 1, H, W]`` object mask.
    nImgArray: ``[B]`` integer-valued tensor with the number of valid
               shots per batch element. Values in ``[1, N_max]``.
    normalize_response: when ``True`` (default), apply per-channel
        mask-mean normalisation before stacking observations. Set to
        ``False`` for analytic round-trip tests where the synthetic
        data already obeys ``k_c == 1``.

    Returns
    -------
    n_ls: ``[B, 3, H, W]`` LS normal (zeroed outside the mask).
    cond: ``[B]`` condition number ``sigma_max / sigma_min`` of the
          stacked light matrix.
    """
    _validate_inputs(S, L, mask, nImgArray)

    B, _, H, W, _ = S.shape
    device = S.device
    dtype = S.dtype

    n_ls = torch.zeros(B, 3, H, W, device=device, dtype=dtype)
    cond = torch.zeros(B, device=device, dtype=dtype)

    for b in range(B):
        n_b = int(nImgArray[b].item())
        if n_b <= 0:
            cond[b] = torch.tensor(float("inf"), device=device, dtype=dtype)
            continue

        if normalize_response:
            I_obs, L_stack = _stack_observations(
                S[b], L[b], mask[b : b + 1], n_b, eps
            )
        else:
            I_obs, L_stack = _stack_observations_raw(S[b], L[b], n_b, eps)

        # Ordinary least squares: solve L_stack @ n = I_obs.T per pixel.
        # ``torch.linalg.lstsq`` returns ``g`` of shape ``[3, HW]``.
        sol = torch.linalg.lstsq(L_stack, I_obs.transpose(0, 1))
        g = sol.solution  # [3, HW]
        n = F.normalize(g.transpose(0, 1), dim=-1, eps=eps)  # [HW, 3]
        n_ls[b] = n.reshape(H, W, 3).permute(2, 0, 1)

        cond[b] = light_matrix_cond(L_stack)

    n_ls = n_ls * mask
    return n_ls, cond


def ls_normal_multi_shot_robust(
    S: torch.Tensor,
    L: torch.Tensor,
    mask: torch.Tensor,
    nImgArray: torch.Tensor,
    shadow_thresh: float = 0.05,
    eps: float = 1e-6,
    ridge: float = 1e-6,
    normalize_response: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted LS that down-weights attached-shadow observations.

    Observations whose post-normalisation intensity falls below
    ``shadow_thresh`` are zero-weighted. The remaining observations
    are solved via the per-pixel normal equations
    ``(L^T W L) n = L^T W I`` (with a small ridge for numerical
    stability). Falls back to the plain LS solution if a pixel has
    fewer than three positive observations.

    Returns
    -------
    n_ls, cond — same conventions as :func:`ls_normal_multi_shot`.
    """
    _validate_inputs(S, L, mask, nImgArray)
    if shadow_thresh < 0.0:
        raise ValueError("shadow_thresh must be >= 0")

    B, _, H, W, _ = S.shape
    device = S.device
    dtype = S.dtype

    n_ls = torch.zeros(B, 3, H, W, device=device, dtype=dtype)
    cond = torch.zeros(B, device=device, dtype=dtype)

    eye3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)

    for b in range(B):
        n_b = int(nImgArray[b].item())
        if n_b <= 0:
            cond[b] = torch.tensor(float("inf"), device=device, dtype=dtype)
            continue

        I_obs, L_stack = (
            _stack_observations(S[b], L[b], mask[b : b + 1], n_b, eps)
            if normalize_response
            else _stack_observations_raw(S[b], L[b], n_b, eps)
        )  # I_obs:[HW, K], L_stack:[K, 3], K = 3*n_b

        # Per-observation weight: 1 if above threshold, else 0.
        W_b = (I_obs > shadow_thresh).to(dtype)  # [HW, K]

        # AtA, Atb per pixel.
        WI = (W_b * I_obs).unsqueeze(-1)               # [HW, K, 1]
        WL = W_b.unsqueeze(-1) * L_stack.unsqueeze(0)  # [HW, K, 3]
        AtA = WL.transpose(1, 2) @ WL                  # [HW, 3, 3]
        AtA = AtA + ridge * eye3                       # ridge regularise
        Atb = WL.transpose(1, 2) @ WI                  # [HW, 3, 1]

        # Pixels with no valid observations fall back to plain LS.
        valid_count = W_b.sum(dim=1)                   # [HW]
        degenerate = valid_count < 3.0

        g = torch.linalg.solve(AtA, Atb).squeeze(-1)   # [HW, 3]

        if degenerate.any():
            # Use plain LS for those pixels — gives some answer rather
            # than a NaN-laden tensor when the whole pixel is in
            # shadow.
            fallback_sol = torch.linalg.lstsq(L_stack, I_obs.transpose(0, 1))
            g_fallback = fallback_sol.solution.transpose(0, 1)  # [HW, 3]
            g = torch.where(
                degenerate.unsqueeze(-1).expand_as(g), g_fallback, g
            )

        n = F.normalize(g, dim=-1, eps=eps)
        n_ls[b] = n.reshape(H, W, 3).permute(2, 0, 1)
        cond[b] = light_matrix_cond(L_stack)

    n_ls = n_ls * mask
    return n_ls, cond


# ---------------------------------------------------------------------------
# Normal losses / metrics / mask ops
# ---------------------------------------------------------------------------
def normal_cos_loss(
    n_pred: torch.Tensor,
    n_gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cosine normal loss ``mean_{mask}(1 - <n_pred, n_gt>)``.

    Supports an optional per-pixel ``weight`` (shape ``[B, 1, H, W]``)
    which is multiplied into the loss before mask averaging.
    """
    cos = (n_pred * n_gt).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
    loss = 1.0 - cos
    if weight is not None:
        loss = loss * weight
    loss = loss * mask
    return loss.sum() / (mask.sum() + eps)


def angular_error_deg(
    n_pred: torch.Tensor,
    n_gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean angular error in degrees over the masked region."""
    cos = (n_pred * n_gt).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
    err = torch.acos(cos) * (180.0 / math.pi)
    err = err * mask
    return err.sum() / (mask.sum() + eps)


def normal_gradient_loss(
    n_pred: torch.Tensor,
    n_gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L1 loss on spatial gradients of normals."""
    def _grad(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    gx_p, gy_p = _grad(n_pred)
    gx_g, gy_g = _grad(n_gt)
    mx = mask[..., :, 1:] * mask[..., :, :-1]
    my = mask[..., 1:, :] * mask[..., :-1, :]

    lx = (torch.abs(gx_p - gx_g) * mx).sum() / (mx.sum() * 3 + eps)
    ly = (torch.abs(gy_p - gy_g) * my).sum() / (my.sum() * 3 + eps)
    return lx + ly


def erode_mask(mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Binary mask erosion via max-pooling on the inverted mask."""
    if k < 1:
        raise ValueError("kernel size k must be >= 1")
    inv = 1.0 - mask
    inv_dilated = F.max_pool2d(inv, kernel_size=k, stride=1, padding=k // 2)
    return 1.0 - inv_dilated


# ---------------------------------------------------------------------------
# Teacher-S1' helpers (see Teacher_S1_prime_dev_doc.md)
# ---------------------------------------------------------------------------
def canonicalize_light_param(
    light_param: torch.Tensor, num_shots: int, eps: float = 1e-8
) -> torch.Tensor:
    """Convert ``light_param`` to the canonical ``[B, N, 3, 3]`` layout.

    Supported input shapes:
        * ``[B, N, 3, 3]``  (already canonical)
        * ``[B, 9*N, 1, 1]`` (packed 4-D)
        * ``[B, 9*N]``       (flat 2-D)
        * ``[B, N, 9]``      (per-shot 9-vector)

    The inner ``3 x 3`` matrix encodes ``[RGB-channel, xyz]`` and each
    row is L2-normalised so the returned directions are unit vectors.
    """
    B = light_param.shape[0]

    if light_param.dim() == 4 and light_param.shape[-2:] == (3, 3):
        L = light_param
    elif light_param.dim() == 4 and light_param.shape[1] == 9 * num_shots:
        L = light_param.view(B, num_shots, 3, 3)
    elif light_param.dim() == 2 and light_param.shape[1] == 9 * num_shots:
        L = light_param.view(B, num_shots, 3, 3)
    elif (
        light_param.dim() == 3
        and light_param.shape[1] == num_shots
        and light_param.shape[2] == 9
    ):
        L = light_param.view(B, num_shots, 3, 3)
    else:
        raise ValueError(
            f"Unsupported light_param shape: {tuple(light_param.shape)}, "
            f"num_shots={num_shots}"
        )

    return F.normalize(L, dim=-1, eps=eps)


def build_light_weighted_feature(
    S: torch.Tensor,
    light_param: torch.Tensor,
    nImgArray: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build the light-weighted geometric feature ``G``.

    For each shot ``k``::

        G_k(p) = sum_c S_c^k(p) * l_c^k

    where ``S_k`` is the per-channel clean shading and ``l_c^k`` is the
    light direction for colour channel ``c`` in that shot.  Equivalent
    to ``torch.einsum("bchw,bcx->bxhw", S_k, L_k)`` per shot.

    Parameters
    ----------
    S:           ``[B, 3, H, W, N]`` clean shading.
    light_param: ``[B, N, 3, 3]`` or any packed variant accepted by
                 :func:`canonicalize_light_param`.
    nImgArray:   optional ``[B]`` per-batch valid-shot count.  When
                 provided, invalid shots are zeroed in the output.

    Returns
    -------
    G: ``[B, 3, H, W, N]`` light-weighted geometric feature.
    """
    if S.dim() != 5 or S.shape[1] != 3:
        raise ValueError(
            f"S must have shape [B, 3, H, W, N]; got {tuple(S.shape)}"
        )

    B, _, H, W, N = S.shape
    L = canonicalize_light_param(light_param, num_shots=N, eps=eps)  # [B, N, 3, 3]
    target_dtype = torch.promote_types(
        S.dtype if torch.is_floating_point(S) else torch.get_default_dtype(),
        L.dtype if torch.is_floating_point(L) else torch.get_default_dtype(),
    )
    S = S.to(dtype=target_dtype)
    L = L.to(dtype=target_dtype)

    # Reorder S to [B, N, 3, H, W] so we can batch the einsum across shots.
    S_perm = S.permute(0, 4, 1, 2, 3).contiguous()  # [B, N, 3, H, W]
    # G_xyz = Σ_c S_c * l_c  -> einsum over channel ``c``.
    G_perm = torch.einsum("bnchw,bncx->bnxhw", S_perm, L)  # [B, N, 3, H, W]
    G = G_perm.permute(0, 2, 3, 4, 1).contiguous()  # [B, 3, H, W, N]

    if nImgArray is not None:
        idx = torch.arange(N, device=S.device).unsqueeze(0)  # [1, N]
        n_b = nImgArray.to(S.device).view(B, 1)               # [B, 1]
        valid = (idx < n_b).to(S.dtype).view(B, 1, 1, 1, N)
        G = G * valid

    return G


__all__ = [
    "normalize_camera_response",
    "light_matrix_cond",
    "ls_normal_multi_shot",
    "ls_normal_multi_shot_robust",
    "normal_cos_loss",
    "angular_error_deg",
    "normal_gradient_loss",
    "erode_mask",
    "canonicalize_light_param",
    "build_light_weighted_feature",
]
