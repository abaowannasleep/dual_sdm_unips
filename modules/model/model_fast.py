"""
Fast / efficient drop-in replacement for ``model.DualInputNet``.

This module preserves the *exact* public interface of
``modules.model.model.DualInputNet`` (constructor signature, ``forward``,
``test``, ``load_pretrained_encoders``, freeze helpers, return shapes) but
swaps the costly O(N^2) hand-rolled multi-head attention used inside
``modules.model.transformer`` for ``F.scaled_dot_product_attention`` (SDPA).

The low-level SDPA attention primitives (``MultiHeadAttentionBlock``, ``SAB``,
``PMA``, ``CommunicationBlock``, ``AggregationBlock``) live in
``transformer_sdpa.py``; this file contains only the higher-level model
components and the top-level ``DualInputNetFast`` class.

Why this is faster
------------------
* SDPA dispatches to fused FlashAttention / memory-efficient kernels on
  CUDA when available.
* The hand-rolled implementation in ``transformer.MultiHeadSelfAttentionBlock``
  performs explicit ``torch.cat(... .split(...))`` head-splitting, manual
  ``softmax`` and ``bmm`` calls, and an extra ``Dropout`` on the attention
  matrix; SDPA does all of this in a single fused kernel.
Weight compatibility
--------------------
All sub-module attribute names are kept identical to the originals
(``fc_q``, ``fc_k``, ``fc_v``, ``ln0``, ``ln1``, ``fc_o1``, ``fc_o2``,
``enc``, ``dec``, ``mab``, ``S``, ``comm``, ``aggregation``,
``prediction_normal``, ``image_encoder``, ``glc_upsample``,
``glc_aggregation``, ``regressor``).  Existing
checkpoints saved by ``DualInputNet`` therefore load via
``load_pretrained_encoders`` (with ``strict=False``); any legacy
``window_regressor.*`` / ``window_residual_refiner.*`` entries written
during the (now removed) window-attention experiment are silently
skipped.

Numerically the new attention is mathematically equivalent to the
original "Normal" attention path (we explicitly pre-scale Q so SDPA's
``1/sqrt(d_head)`` scale combines to the same ``1/sqrt(dim_V)`` that the
original code uses).  Fine-tuning is therefore expected to be unaffected.
"""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import mode_change, make_index_tensor
from . import convnext
from . import uper
from ..utils import gauss_filter
from ..utils.ind2sub import ind2coords
from .decompose_tensors import divide_tensor_spatial, merge_tensor_spatial
from .transformer_sdpa import SAB, PMA, CommunicationBlock, AggregationBlock
from .photometric_prior import build_light_weighted_feature


SHALLOW_DISTILLATION_MODULE_PATHS = {
    "pp_conv1": "image_encoder.per_pixel_branch.conv.0",
    "pp_conv2": "image_encoder.per_pixel_branch.conv.2",
    # ConvNeXt is called for both the resized image and the tiled image.
    # ShallowFeatureHooks keeps the first activation, i.e. the resized path.
    "convnext_stem": (
        "image_encoder.backbone.backbone.0.downsample_layers.0"
    ),
    "convnext_stage0": "image_encoder.backbone.backbone.0.stages.0",
}


class ShallowFeatureHooks:
    """Capture named shallow activations without changing model forward APIs."""

    def __init__(self, model, layer_names):
        self.features = {}
        self.enabled = False
        self._handles = []

        for layer_name in layer_names:
            if layer_name not in SHALLOW_DISTILLATION_MODULE_PATHS:
                supported = ", ".join(SHALLOW_DISTILLATION_MODULE_PATHS)
                raise ValueError(
                    f"Unknown shallow KD layer {layer_name!r}. "
                    f"Supported layers: {supported}."
                )
            module_path = SHALLOW_DISTILLATION_MODULE_PATHS[layer_name]
            try:
                module = model.get_submodule(module_path)
            except (AttributeError, KeyError, RuntimeError) as exc:
                raise ValueError(
                    f"Shallow KD layer {layer_name!r} is unavailable at "
                    f"{module_path!r}."
                ) from exc
            self._handles.append(
                module.register_forward_hook(self._make_hook(layer_name))
            )

    def _make_hook(self, layer_name):
        def hook(_module, _inputs, output):
            # ConvNeXt modules run twice; retain the resized-image activation.
            if self.enabled and layer_name not in self.features:
                self.features[layer_name] = output

        return hook

    def clear(self):
        self.features.clear()

    @contextmanager
    def capture(self):
        self.clear()
        self.enabled = True
        try:
            yield self.features
        finally:
            self.enabled = False

    def close(self):
        self.enabled = False
        self.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Image feature extractor / fusion / encoder
# ---------------------------------------------------------------------------

class ImageFeatureExtractor(nn.Module):
    """Same as ``model.ImageFeatureExtractor`` -- ConvNeXt backbone."""

    def __init__(self, input_nc):
        super().__init__()
        self.out_channels = (96, 192, 384, 768)
        # Wrap in Sequential so the ``backbone.0.*`` parameter prefix matches
        # the original ``ImageFeatureExtractor`` (which uses ``nn.Sequential``).
        self.backbone = nn.Sequential(convnext.ConvNeXt(in_chans=input_nc))

    def forward(self, x):
        return self.backbone(x)


class ImageFeatureFusion(nn.Module):
    """Vectorised replacement for ``model.ImageFeatureFusion``.

    Differences vs. the original:

    * Self-attention inside ``CommunicationBlock`` uses SDPA.
    * The per-batch Python ``for b in range(batch_size)`` loop is
      eliminated *when all batch elements share the same number of
      images* (the common bucketed-training case): we then run the
      communication block exactly once on a rearranged tensor.
    * For variable image counts we fall back to the original per-batch
      loop -- still correct, just less vectorised.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.fusion = uper.UPerHead(in_channels=in_channels)
        self.num_comm_enc = [0, 1, 2, 4]

        attn = []
        for i in range(len(in_channels)):
            if self.num_comm_enc[i] > 0:
                attn.append(
                    CommunicationBlock(
                        in_channels[i],
                        num_enc_sab=self.num_comm_enc[i],
                        dim_hidden=in_channels[i],
                        ln=True,
                        dim_feedforward=in_channels[i],
                    )
                )
        # Use ModuleList so we can index numerically; original uses
        # Sequential (also indexable). Naming "comm" matches the original.
        self.comm = nn.ModuleList(attn)

    def forward(self, glc, nImgArray, return_attention=False):
        batch_size = len(nImgArray)
        n_list = [int(n) for n in (nImgArray.tolist() if torch.is_tensor(nImgArray) else nImgArray)]
        sum_nimg = sum(n_list)
        same_n = (len(set(n_list)) == 1) and batch_size > 0

        out_fuse = []
        attn_cnt = 0
        for k in range(len(glc)):
            if self.num_comm_enc[k] > 0:
                in_fuse = glc[k]
                _, C, H, W = in_fuse.shape
                # ((K+1)*sum_nimg, C, H, W) -> (K+1, sum_nimg, C, H, W)
                in_fuse = in_fuse.reshape(-1, sum_nimg, C, H, W).permute(0, 3, 4, 1, 2)
                K_plus = in_fuse.shape[0]
                # -> (K_plus*H*W, sum_nimg, C)
                in_fuse = in_fuse.reshape(K_plus * H * W, sum_nimg, C)
                M = K_plus * H * W

                if same_n:
                    n_b = n_list[0]
                    # Reinterpret sum_nimg = batch_size * n_b and run the comm
                    # block once over (batch_size * M, n_b, C).  The original
                    # code processes each batch element separately because
                    # different batch elements may have different ``n_b`` --
                    # when they don't, this is exactly equivalent.
                    feats = (
                        in_fuse.view(M, batch_size, n_b, C)
                        .transpose(0, 1)
                        .reshape(batch_size * M, n_b, C)
                    )
                    feats = self.comm[attn_cnt](feats)
                    feats = (
                        feats.view(batch_size, M, n_b, C)
                        .transpose(0, 1)
                        .reshape(M, sum_nimg, C)
                    )
                else:
                    # Variable image counts -- fall back to per-batch loop
                    # (matches original behaviour exactly).
                    feats_list = []
                    ids = 0
                    for b in range(batch_size):
                        n_b = n_list[b]
                        feat = in_fuse[:, ids:ids + n_b, :]
                        feat = self.comm[attn_cnt](feat)
                        feats_list.append(feat)
                        ids += n_b
                    feats = torch.cat(feats_list, dim=1)

                # Restore spatial layout: (M, sum_nimg, C) -> ((K+1)*sum_nimg, C, H, W)
                feats = feats.reshape(K_plus, H * W, sum_nimg, C).permute(0, 2, 3, 1)
                feats = feats.reshape(K_plus * sum_nimg, C, H, W)
                out_fuse.append(feats)
                attn_cnt += 1
            else:
                out_fuse.append(glc[k])

        return self.fusion(out_fuse)


class PerPixelBranch(nn.Module):
    """Lightweight per-pixel feature extractor that keeps the full ``H x W``
    resolution (no downsampling).

    Complementary to the ConvNeXt ``ImageFeatureExtractor`` backbone:
      * backbone   -> abstract / global features (4x ~ 32x downsampled)
      * this branch -> pixel-level photometric details (zero downsampling)

    Design notes (see ``fix1_per_pixel_branch_dev_doc.md``):
      * Shallow (3 layers) and moderately wide, to avoid duplicating
        backbone capacity.
      * 3x3 convolutions naturally overlap, so no grid artefacts.
      * Input is the same ``[N, input_nc, H, W]`` tensor that feeds the
        backbone, so the encoder dataloader output is reused as-is.
    """

    def __init__(self, in_ch=4, out_ch=64, mid_ch=32):
        super().__init__()
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_ch, 3, padding=1),
        )

    def forward(self, x):
        # x: [N, in_ch, H, W] -> [N, out_ch, H, W] (resolution preserved).
        return self.conv(x)


class ScaleInvariantSpatialLightImageEncoder(nn.Module):
    """Same forward logic as the original encoder, fast fusion underneath."""

    def __init__(
        self,
        input_nc,
        use_per_pixel_branch=False,
        pp_out_ch=64,
        pp_input_nc=None,
    ):
        super().__init__()
        self.backbone = ImageFeatureExtractor(input_nc)
        self.fusion = ImageFeatureFusion(self.backbone.out_channels)
        self.feat_dim = 256

        # [fix1] Per-pixel branch: parallel zero-downsampling conv stack that
        # preserves pixel-level photometric detail lost in the ConvNeXt stem.
        self.use_per_pixel_branch = use_per_pixel_branch
        self.pp_out_ch = pp_out_ch if use_per_pixel_branch else 0
        self.pp_input_nc = pp_input_nc if pp_input_nc is not None else input_nc
        if use_per_pixel_branch:
            self.per_pixel_branch = PerPixelBranch(
                in_ch=self.pp_input_nc, out_ch=pp_out_ch
            )
        else:
            self.per_pixel_branch = None

    def forward(
        self,
        x,
        nImgArray,
        canonical_resolution,
        return_intermediate=False,
        pp_input=None,
    ):
        N, C, H, W = x.shape
        canon = int(canonical_resolution)
        if H % canon != 0 or W % canon != 0:
            canon = math.gcd(H, W)
            if canon == 0:
                canon = min(H, W)
        mosaic_scale = H // canon
        K = mosaic_scale * mosaic_scale

        # (1a) resize x to (Hc, Wc)
        x_resized = F.interpolate(x, size=(canon, canon), mode="bilinear", align_corners=True)

        # (1b) decompose x into K x K of (Hc, Wc) non-overlapped blocks
        x_grid = divide_tensor_spatial(x, block_size=canon, method="tile_stride")
        x_grid = x_grid.permute(1, 0, 2, 3, 4).reshape(-1, C, canon, canon)

        # (2a) feature extraction at canonical resolution
        back_feats = self.backbone(x_resized)
        f_resized = self.fusion(back_feats, nImgArray)
        f_resized = f_resized.reshape(1, N, self.feat_dim, canon // 4 * canon // 4)
        del x_resized

        # (2b) feature extraction over tiles
        grid_back_feats = self.backbone(x_grid)
        gf = self.fusion(grid_back_feats, nImgArray)
        gf = gf.reshape(K, N, gf.shape[1], canon // 4, canon // 4)
        glc_grid = merge_tensor_spatial(gf, method="tile_stride")
        del x_grid

        # (3) upsample
        glc_resized = F.interpolate(
            f_resized.reshape(N, self.feat_dim, canon // 4, canon // 4),
            size=(H // 4, W // 4),
            mode="bilinear",
            align_corners=True,
        )
        del f_resized

        glc = glc_resized + glc_grid

        # [fix1] Per-pixel branch operates on the full-resolution input ``x``
        # (same tensor that feeds the backbone), so each output spatial
        # location aligns 1:1 with an input pixel.  Returns ``None`` when
        # disabled so downstream consumers can branch on it cheaply.
        pp_feat = None
        if self.use_per_pixel_branch:
            if pp_input is None:
                pp_input = x
            pp_feat = self.per_pixel_branch(pp_input)

        if return_intermediate:
            back_feats = {"global": back_feats, "grid": grid_back_feats}
            return glc, back_feats, pp_feat
        return glc, pp_feat


# ---------------------------------------------------------------------------
# Per-pixel heads (drop-in replacements; weight names match originals)
# ---------------------------------------------------------------------------

class GLC_Upsample(nn.Module):
    """Replacement for ``model.GLC_Upsample``."""

    def __init__(self, input_nc, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024):
        super().__init__()
        self.comm = CommunicationBlock(
            input_nc,
            num_enc_sab=num_enc_sab,
            dim_hidden=dim_hidden,
            ln=True,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, x):
        return self.comm(x)


class GLC_Aggregation(nn.Module):
    """Replacement for ``model.GLC_Aggregation``."""

    def __init__(
        self,
        input_nc,
        num_agg_transformer=2,
        dim_aggout=384,
        dim_feedforward=1024,
    ):
        super().__init__()
        self.aggregation = AggregationBlock(
            dim_input=input_nc,
            num_enc_sab=num_agg_transformer,
            num_outputs=1,
            dim_hidden=dim_aggout,
            dim_feedforward=dim_feedforward,
            num_heads=8,
            ln=True,
            attention_dropout=0.1,
        )

    def forward(self, x):
        return self.aggregation(x)


class PredictionHead(nn.Module):
    """Same as ``model.PredictionHead``."""

    def __init__(self, dim_input, dim_output):
        super().__init__()
        modules_regression = [
            nn.Linear(dim_input, dim_input // 2),
            nn.ReLU(),
            nn.Linear(dim_input // 2, dim_output),
        ]
        self.regression = nn.Sequential(*modules_regression)

    def forward(self, x):
        return self.regression(x)


class Regressor(nn.Module):
    """Replacement for ``model.Regressor``.

    Uses a ``CommunicationBlock`` (SDPA-backed) for the per-pixel
    self-attention -- this is the dominant cost in the original network
    because the sequence length here equals the number of sampled pixels
    in a chunk (often thousands).
    """

    def __init__(self, input_nc, num_enc_sab=1, dim_feedforward=256):
        super().__init__()
        self.comm = CommunicationBlock(
            input_nc,
            num_enc_sab=num_enc_sab,
            dim_hidden=input_nc,
            ln=True,
            dim_feedforward=dim_feedforward,
        )
        self.prediction_normal = PredictionHead(input_nc, 3)

    def forward(self, x, num_sample_set, n_imgs=None):
        # ``x`` is (M, N_imgs, C); ``num_sample_set`` is the per-chunk sample
        # count.  We attend across the M dim grouped into windows of size
        # ``num_sample_set`` -- identical semantics to the original.
        if x.shape[0] % num_sample_set == 0:
            x_ = x.reshape(-1, num_sample_set, x.shape[1])
            x_ = self.comm(x_)
            x = x_.reshape(-1, x.shape[1])
        else:
            ids = list(range(x.shape[0]))
            num_split = len(ids) // num_sample_set
            x_1 = x[: num_split * num_sample_set, :].reshape(-1, num_sample_set, x.shape[1])
            x_1 = self.comm(x_1).reshape(-1, x.shape[1])
            x_2 = x[num_split * num_sample_set:, :].reshape(1, -1, x.shape[1])
            x_2 = self.comm(x_2).reshape(-1, x.shape[1])
            x = torch.cat([x_1, x_2], dim=0)

        return self.prediction_normal(x)



def mean_valid_shots(x, nImgArray):
    """Average valid shots along the final ``Nmax`` shot dimension.

    Args:
        x: Tensor with shape ``[B, C, H, W, Nmax]``.
        nImgArray: Valid-shot counts with shape ``[B]``.

    Returns:
        Tensor with shape ``[B, C, H, W]``.

    Only shots where ``shot_id < nImgArray[b]`` contribute to each sample's
    mean; padded shots are masked out.
    """
    B, _, _, _, Nmax = x.shape
    n_img = nImgArray.to(x.device)
    counts = n_img.to(x.dtype).clamp_min(1).view(B, 1, 1, 1)
    shot_ids = torch.arange(Nmax, device=x.device).view(1, 1, 1, 1, Nmax)
    valid = (shot_ids < n_img.view(B, 1, 1, 1, 1)).to(x.dtype)
    return (x * valid).sum(dim=-1) / counts


class LightCalibNet(nn.Module):
    """Per-shot, per-channel global light-direction estimator.

    Input ``I`` is a multispectral RGB stack with shape
    ``[B, 3, H, W, Nmax]``; ``nImgArray`` contains valid-shot counts.  The
    output is ``l_hat`` with shape ``[B, Nmax, 3, 3]`` and unit-norm xyz
    directions for each RGB channel.
    """

    def __init__(self, in_ch: int = 3, base: int = 64, work_res: int = 256):
        super().__init__()
        self.work_res = int(work_res)
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1),
            nn.GroupNorm(8, base), nn.GELU(),
            nn.Conv2d(base, base * 2, 3, 2, 1),
            nn.GroupNorm(8, base * 2), nn.GELU(),
            nn.Conv2d(base * 2, base * 2, 3, 2, 1),
            nn.GroupNorm(8, base * 2), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(base * 2, base * 2), nn.GELU(),
            nn.Linear(base * 2, 3 * 3),
        )

    def forward(self, I: torch.Tensor, nImgArray=None) -> torch.Tensor:
        B, C, H, W, Nmax = I.shape
        x = I.permute(0, 4, 1, 2, 3).reshape(B * Nmax, C, H, W)
        if (H, W) != (self.work_res, self.work_res):
            x = F.interpolate(
                x, size=(self.work_res, self.work_res),
                mode="bilinear", align_corners=False,
            )
        f = self.feat(x).mean(dim=(2, 3))
        l = self.head(f).reshape(B, Nmax, 3, 3)
        return F.normalize(l, dim=-1, eps=1e-6)



# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------

class DualInputNet(nn.Module):
    """Drop-in faster replacement for ``model.DualInputNet``.

    The constructor remains backward-compatible with ``DualInputNet`` and the
    ``forward`` signature and return-value shapes are unchanged -- swap the
    import and existing training pipelines work unchanged.  (The legacy inference-only
    ``test`` method has been removed; it was never used by any caller of
    this fast variant.)

    Light input is selected with two mutually exclusive switches:
    ``use_light_calib_model`` predicts directions inside ``forward`` with
    ``LightCalibNet``; ``use_light`` consumes real directions passed as
    ``light_param``.  If both are enabled, ``use_light_calib_model`` takes
    precedence.
    """

    def __init__(self, pixel_samples, device,
                 use_light_calib_model=False,
                 use_light=False,
                 use_per_pixel_branch=False,
                 pp_out_ch=64,
                 light_calib_base=64,
                 light_calib_work_res=256,
                 **deprecated_options):
        super().__init__()
        self.device = device
        self.pixel_samples = pixel_samples
        self.glc_smoothing = True
        # [Teacher-S1'] Light-weighted input feature G = Σ_c S_c * l_c.
        # ``use_light_calib_model`` predicts lights with LightCalibNet inside
        # forward(); ``use_light`` consumes real light directions supplied to
        # forward().  Both light paths use the SG encoder input.  Without a
        # light path the encoder falls back to the ordinary 4-channel input.
        self.use_light_calib_model = bool(use_light_calib_model)
        # Mutually exclusive light sources: LightCalibNet predicts the active
        # directions itself, so it takes precedence if both switches are on.
        self.use_light = bool(use_light) and not self.use_light_calib_model
        self.uses_light_input = self.use_light_calib_model or self.use_light
        self.light_input_mode = "SG"
        self.light_calib = None
        if self.use_light_calib_model:
            self.light_calib = LightCalibNet(
                in_ch=3, base=light_calib_base, work_res=light_calib_work_res
            )
        # Encoder input channels: 7 for SG (S+G+M), 4 otherwise (with M).
        if self.uses_light_input:
            self.input_dim = 7
        else:
            self.input_dim = 4

        # [fix1] Per-pixel branch switch; the branch output ``pp_out_ch`` is
        # concatenated into the decoder chunk token alongside ``o_ids`` and
        # ``glc_ids`` (see ``fix1_per_pixel_branch_dev_doc.md``).
        self.use_per_pixel_branch = use_per_pixel_branch
        self.pp_out_ch = pp_out_ch if use_per_pixel_branch else 0
        self.image_encoder = ScaleInvariantSpatialLightImageEncoder(
            self.input_dim,
            use_per_pixel_branch=use_per_pixel_branch,
            pp_out_ch=pp_out_ch,
            pp_input_nc=self.input_dim,
        ).to(self.device)

        self.rgb_dim = 3
        # [fix1] Per-pixel branch features are concatenated into the token
        # decoder, so ``upsample_in`` grows by ``pp_out_ch`` when enabled.
        upsample_in = 256 + self.rgb_dim + self.pp_out_ch
        self.glc_upsample = GLC_Upsample(
            upsample_in, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024
        ).to(self.device)
        self.glc_aggregation = GLC_Aggregation(
            upsample_in,
            num_agg_transformer=2,
            dim_aggout=384,
            dim_feedforward=1024,
        ).to(self.device)
        self.regressor = Regressor(
            384, num_enc_sab=1, dim_feedforward=1024
        ).to(self.device)

        # Cache for the Gaussian smoothing filter (channels, kernel_size).
        self._smoothing_cache = {}

    # ------------------------------------------------------------------
    # Freeze helpers (same names + signatures as DualInputNet)
    # ------------------------------------------------------------------
    def freeze_image_encoder(self, freeze=True):
        for param in self.image_encoder.parameters():
            param.requires_grad = not freeze

    def freeze_glc_upsample(self, freeze=True):
        for param in self.glc_upsample.parameters():
            param.requires_grad = not freeze

    def freeze_glc_aggregation(self, freeze=True):
        for param in self.glc_aggregation.parameters():
            param.requires_grad = not freeze

    def setup_finetuning(
        self,
        freeze_image_encoder=True,
        freeze_upsample=True,
        freeze_aggregation=False,
    ):
        self.freeze_image_encoder(freeze_image_encoder)
        self.freeze_glc_upsample(freeze_upsample)
        self.freeze_glc_aggregation(freeze_aggregation)
        for param in self.regressor.parameters():
            param.requires_grad = True

    def no_grad(self):
        mode_change(self.image_encoder, False)
        mode_change(self.glc_upsample, False)
        mode_change(self.glc_aggregation, False)
        mode_change(self.regressor, False)

    # ------------------------------------------------------------------
    # Pretrained weight loading -- mirrors ``DualInputNet.load_pretrained_encoders``
    # ------------------------------------------------------------------
    def load_pretrained_encoders(self, state_dict):
        """Load weights from a ``DualInputNet`` (or ``DualInputNetFast``) checkpoint.

        Handles two layouts:
            1. Single-encoder checkpoints with ``image_encoder.*`` keys.
            2. Old dual-encoder checkpoints with ``shading_encoder.*`` (used as
               fallback when no ``image_encoder.*`` keys are present).

        Any legacy ``refiner.*`` / ``dense_decoder.*`` keys from older
        checkpoints are silently skipped (those modules have been removed).

        Loading is non-strict: extra keys (e.g. ``shading_encoder.*``) and
        any minor shape-incompatible entries are tolerated and reported.
        """
        if any(k.startswith("module.") for k in state_dict.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        else:
            state = state_dict

        enc_state = {
            k.replace("image_encoder.", "", 1): v
            for k, v in state.items()
            if k.startswith("image_encoder.")
        }
        if not enc_state:
            enc_state = {
                k.replace("shading_encoder.", "", 1): v
                for k, v in state.items()
                if k.startswith("shading_encoder.")
            }
            if enc_state:
                print("[Pretrain] No image_encoder keys found; falling back to shading_encoder keys")

        if enc_state:
            # [Teacher-S1'-local] When SG light input is enabled the encoder
            # input dim grows from 4 (S+M / G+M) to 7 (S+G+M).  The ConvNeXt
            # stem ``backbone.0.downsample_layers.0.0.weight`` has shape
            # ``[96, in_chans, 4, 4]`` -- its ``in_chans`` axis mismatches the
            # checkpoint, and PyTorch's ``load_state_dict(strict=False)`` does
            # NOT forgive size mismatches (it forgives only missing/unexpected
            # keys); we therefore reshape the stem-conv weight ourselves
            # before loading.  Behaviour matches the decoder-side handling
            # below: extra input channels get a small initialization (so the network
            # initially behaves as if the new channels were near-zero, but
            # is not stuck in a symmetric all-zero regime); shrinking
            # truncates the leading channels.  See ``S1_local_dev_doc.md`` §7.
            stem_key = "backbone.backbone.0.downsample_layers.0.0.weight"
            cur_stem = dict(self.image_encoder.named_parameters()).get(stem_key)
            if stem_key in enc_state and cur_stem is not None:
                old_w = enc_state[stem_key]
                new_in = cur_stem.shape[1]
                old_in = old_w.shape[1]
                if new_in != old_in:
                    if new_in < old_in:
                        print(
                            f"[Pretrain] encoder.{stem_key}: checkpoint "
                            f"in_chans ({old_in}) > model ({new_in}); "
                            f"truncating leading channels"
                        )
                        enc_state[stem_key] = old_w.narrow(1, 0, new_in).contiguous()
                    else:
                        new_shape = list(old_w.shape)
                        new_shape[1] = new_in
                        new_w = torch.empty(
                            *new_shape, dtype=old_w.dtype, device=old_w.device,
                        )
                        new_w.narrow(1, 0, old_in).copy_(old_w)
                        torch.nn.init.normal_(
                            new_w.narrow(1, old_in, new_in - old_in),
                            mean=0.0, std=1e-3,
                        )
                        enc_state[stem_key] = new_w
                        print(
                            f"[Pretrain] encoder.{stem_key}: expanded "
                            f"in_chans {old_in} -> {new_in} "
                            f"(extra {new_in - old_in} channels small initialization)"
                        )

            cur_state = self.image_encoder.state_dict()
            shape_mismatch_keys = []
            keys_to_drop = []
            for key in list(enc_state.keys()):
                if key not in cur_state:
                    continue
                if tuple(enc_state[key].shape) != tuple(cur_state[key].shape):
                    shape_mismatch_keys.append(
                        (key, tuple(enc_state[key].shape), tuple(cur_state[key].shape))
                    )
                    keys_to_drop.append(key)
            for key in keys_to_drop:
                del enc_state[key]
            if shape_mismatch_keys:
                print(
                    "[Pretrain] encoder skipped shape-mismatched keys "
                    f"(kept model init): {shape_mismatch_keys[:5]}"
                )

            enc_msg = self.image_encoder.load_state_dict(enc_state, strict=False)
            print(f"[Pretrain] encoder keys loaded: {len(enc_state)}")
            if enc_msg.missing_keys:
                print(f"[Pretrain] encoder missing keys: {len(enc_msg.missing_keys)}")
            if enc_msg.unexpected_keys:
                print(f"[Pretrain] encoder unexpected keys: {len(enc_msg.unexpected_keys)}")
        else:
            print("[Pretrain] WARNING: no encoder keys found in checkpoint")

        for prefix, mod, first_linear_root in (
            ("glc_upsample.", self.glc_upsample, "comm.enc.0.mab"),
            ("glc_aggregation.", self.glc_aggregation, "aggregation.enc.0.mab"),
            ("regressor.", self.regressor, None),
        ):
            if mod is None:
                # Module disabled; report any keys that we would otherwise
                # have consumed and move on.
                sub = {k for k in state if k.startswith(prefix)}
                if sub:
                    print(f"[Pretrain] {prefix[:-1]} disabled; skipping {len(sub)} checkpoint keys")
                continue
            sub = {k.replace(prefix, "", 1): v for k, v in state.items() if k.startswith(prefix)}
            if sub:
                # When loading checkpoints whose decoder token input dim
                # differs from the current configuration (for example because
                # the legacy fix1 variant injected ``pp_out_ch`` extra channels into the
                # first MAB), every parameter whose shape depends on
                # ``dim_in`` of that first MAB has to be reshaped before
                # load_state_dict, otherwise PyTorch raises a RuntimeError
                # (``strict=False`` only forgives missing/unexpected keys,
                # not size mismatches):
                #   * fc_q/fc_k/fc_v.weight : Linear(dim_in -> dim_hidden),
                #     extra input columns use small initialization so the rest of the
                #     layer keeps its learned semantics (zero-init can trap
                #     symmetries).
                #   * ln0.weight / ln0.bias : LayerNorm(dim_in), extra
                #     dimensions are initialised to the LayerNorm defaults
                #     (weight=1, bias=0) so the new channels pass through
                #     unchanged at load time.
                if first_linear_root is not None:
                    # (sub_name, axis_for_dim_in, pad_init_fn)
                    # - axis_for_dim_in: which tensor axis carries dim_in.
                    # - pad_init_fn(tensor_slice): fills the new slice in
                    #   place.  Each function takes the new (uninitialised)
                    #   slice tensor that needs to be populated.
                    expand_specs = (
                        (
                            "fc_q.weight", 1,
                            lambda t: torch.nn.init.normal_(t, mean=0.0, std=1e-3),
                        ),
                        (
                            "fc_k.weight", 1,
                            lambda t: torch.nn.init.normal_(t, mean=0.0, std=1e-3),
                        ),
                        (
                            "fc_v.weight", 1,
                            lambda t: torch.nn.init.normal_(t, mean=0.0, std=1e-3),
                        ),
                        # LayerNorm(dim_in) defaults: weight=1, bias=0.
                        ("ln0.weight", 0, lambda t: t.fill_(1.0)),
                        ("ln0.bias", 0, lambda t: t.fill_(0.0)),
                    )
                    cur_params = dict(mod.named_parameters())
                    for sub_name, axis, pad_init in expand_specs:
                        key = f"{first_linear_root}.{sub_name}"
                        if key not in sub:
                            continue
                        old_w = sub[key]
                        cur_param = cur_params.get(key)
                        if cur_param is None:
                            continue
                        new_in = cur_param.shape[axis]
                        old_in = old_w.shape[axis]
                        if new_in == old_in:
                            continue
                        if new_in < old_in:
                            print(
                                f"[Pretrain] {prefix[:-1]}.{key}: checkpoint "
                                f"dim ({old_in}) > model ({new_in}); "
                                f"truncating extra entries along axis {axis}"
                            )
                            sub[key] = old_w.narrow(axis, 0, new_in).contiguous()
                            continue
                        new_shape = list(old_w.shape)
                        new_shape[axis] = new_in
                        new_w = torch.empty(
                            *new_shape, dtype=old_w.dtype, device=old_w.device,
                        )
                        new_w.narrow(axis, 0, old_in).copy_(old_w)
                        pad_init(new_w.narrow(axis, old_in, new_in - old_in))
                        sub[key] = new_w
                        print(
                            f"[Pretrain] {prefix[:-1]}.{key}: expanded "
                            f"dim {old_in} -> {new_in} along axis {axis} "
                            f"(extra {new_in - old_in} entries default-init)"
                        )

                msg = mod.load_state_dict(sub, strict=False)
                print(f"[Pretrain] {prefix[:-1]} keys loaded: {len(sub)}")
                if msg.missing_keys:
                    print(f"[Pretrain] {prefix[:-1]} missing keys: {len(msg.missing_keys)}")
                if msg.unexpected_keys:
                    print(f"[Pretrain] {prefix[:-1]} unexpected keys: {len(msg.unexpected_keys)}")

        # Silently swallow any legacy ``refiner.*`` / ``dense_decoder.*`` keys
        # from older checkpoints; the modules have been removed.
        legacy_consumed = {"refiner.", "dense_decoder."}
        legacy_keys = [k for k in state if any(k.startswith(p) for p in legacy_consumed)]
        if legacy_keys:
            print(f"[Pretrain] Skipping {len(legacy_keys)} legacy refiner/dense_decoder keys")

        consumed_prefixes = (
            "image_encoder.",
            "shading_encoder.",
            "glc_upsample.",
            "glc_aggregation.",
            "regressor.",
            "refiner.",
            "dense_decoder.",
        )
        skipped = [k for k in state if not any(k.startswith(p) for p in consumed_prefixes)]
        if skipped:
            print(f"[Pretrain] Skipped {len(skipped)} unrecognised keys: {skipped[:5]}...")

    # ------------------------------------------------------------------
    # Helper: cached Gaussian filter
    # ------------------------------------------------------------------
    def _get_smoothing(self, channels, kernel_size, sigma, device):
        cache_key = (channels, kernel_size)
        if cache_key not in self._smoothing_cache:
            self._smoothing_cache[cache_key] = gauss_filter.gauss_filter(
                channels, kernel_size, sigma
            ).to(device)
        return self._smoothing_cache[cache_key]

    # ------------------------------------------------------------------
    # Training-time forward (matches ``DualInputNet.forward``)
    # ------------------------------------------------------------------
    def forward(
        self,
        I,
        S,
        M,
        nImgArray,
        canonical_resolution,
        decoder_resolution,
        useShading=False,
        light_param=None,
        return_intermediate=False,
    ):
        decoder_resolution = int(decoder_resolution[0, 0].item())
        canonical_resolution = int(canonical_resolution[0, 0].item())
        decoder_imgsize = (decoder_resolution, decoder_resolution)
        H_dec, W_dec = decoder_imgsize
        B, C, H, W, Nmax = I.shape
        nImgArray_cpu = nImgArray.detach().cpu().tolist()
        img_index = make_index_tensor(Nmax, nImgArray_cpu, device=self.device)

        M_enc = M.repeat(1, Nmax, 1, 1).reshape(B * Nmax, 1, H, W)
        M_dec = F.interpolate(M, size=decoder_imgsize, mode="nearest")
        nout = torch.zeros(B, H_dec * W_dec, 3, device=self.device)
        upglc_dim = 256
        x_dim = 384

        # Pre-compute valid pixel indices for every batch element and
        # randomly permute them so the chunk-level set attention sees
        # spatially uncorrelated pixels (the original "random attention"
        # path that pre-dates the window-attention experiment).
        M_flat = M_dec.reshape(B, H_dec * W_dec)
        all_ids = [torch.nonzero(M_flat[b] > 0, as_tuple=True)[0] for b in range(B)]
        n_valid_counts = [len(ids_b) for ids_b in all_ids]
        same_count = (len(set(n_valid_counts)) == 1)
        if same_count and B > 1:
            shared_perm = torch.randperm(n_valid_counts[0], device=self.device)
            all_ids = [ids_b[shared_perm] for ids_b in all_ids]
        else:
            all_ids = [
                ids_b[torch.randperm(len(ids_b), device=ids_b.device)]
                for ids_b in all_ids
            ]

        # Select input data based on useShading flag (single unified path).
        # ``input_data`` (3-channel) is also used for the per-pixel decoder
        # observation ``o_ids``; the encoder-side input may differ in the
        # Teacher-S1' light-weighted variants below.
        input_data = S if (useShading and S is not None) else I

        # [Teacher-S1'] Build the encoder input core.
        #   light-calib -> [S, G(l_hat), M] with l_hat predicted from I
        #   use-light   -> [S, G(L), M] with L supplied by the caller
        #   no-light    -> [input_data, M]
        active_light_param = None
        if self.use_light_calib_model:
            active_light_param = self.light_calib(I, nImgArray)
        elif self.use_light:
            active_light_param = light_param

        if self.uses_light_input:
            assert useShading, (
                "Teacher-S1' light-weighted input only supports useShading=True."
            )
            assert S is not None, "S is required for light-weighted input."
            assert active_light_param is not None, (
                "light_param is required when light input is enabled."
            )
            G = build_light_weighted_feature(
                S, active_light_param.to(self.device), nImgArray=nImgArray
            )  # [B, 3, H, W, Nmax]
            enc_core = torch.cat([S, G], dim=1)  # [B, 6, H, W, Nmax]
        else:
            enc_core = input_data

        enc_core_C = enc_core.shape[1]
        enc_input = enc_core.permute(0, 4, 1, 2, 3).reshape(B * Nmax, enc_core_C, H, W)
        data = torch.cat([enc_input * M_enc, M_enc], dim=1)
        data = data[img_index, :, :, :]
        if return_intermediate:
            glc, backfeats, pp_feat = self.image_encoder(
                data, nImgArray, canonical_resolution, return_intermediate=True,
            )
        else:
            glc, pp_feat = self.image_encoder(
                data, nImgArray, canonical_resolution, return_intermediate=False,
            )
        # [fix1] ``pp_feat`` is ``None`` when the per-pixel branch is
        # disabled.  When enabled it has shape ``[sum_nimg, pp_out_ch, H, W]``
        # at the original input resolution (no downsampling).
        # Always apply the Gaussian smoothing of the global-local feature map
        # (Teacher-S1' design: ``smoothing`` is no longer toggleable from the
        # forward signature; ``self.glc_smoothing`` remains the single flag).
        if self.glc_smoothing:
            f_scale = decoder_resolution // canonical_resolution
            kernel_size = 10 * f_scale + 1
            smoothing_op = self._get_smoothing(glc.shape[1], kernel_size, 1, glc.device)
            glc = smoothing_op(glc)

        if return_intermediate:
            mid_results = {"glc": glc, "backfeats": backfeats}
            if self.uses_light_input:
                mid_results["input_mode"] = self.light_input_mode
                mid_results["light_param"] = active_light_param
            total_upglc = torch.zeros(B, H_dec * W_dec, Nmax, upglc_dim, device=self.device)
            total_aggx = torch.zeros(B, H_dec * W_dec, x_dim, device=self.device)
        p = 0  # cumulative image offset into the flat glc feature map
        for b in range(B):
            batch_data = input_data[b].permute(3, 0, 1, 2).contiguous()
            valid_mask = img_index[b * Nmax:(b + 1) * Nmax]
            batch_data_valid = batch_data[valid_mask]

            n_valid = int(nImgArray_cpu[b])
            if n_valid <= 0 or batch_data_valid.shape[0] == 0:
                continue
            target_glc = slice(p, p + n_valid)
            target_local = slice(0, n_valid)
            p += n_valid

            ids = all_ids[b]

            if len(ids) > self.pixel_samples:
                num_split = len(ids) // self.pixel_samples + 1
                idset = torch.split(ids, (len(ids) + num_split - 1) // num_split)
            else:
                idset = [ids]

            data_dec = F.interpolate(batch_data_valid, size=decoder_imgsize, mode="bilinear", align_corners=False)
            o_ = data_dec.reshape(n_valid, C, H_dec * W_dec).permute(2, 0, 1)
            glc_batch = glc[target_glc]

            # [fix1] Per-pixel branch features at the original input
            # resolution.  Sliced per batch element so each chunk samples
            # the matching shots; ``None`` when the branch is disabled.
            pp_batch = pp_feat[target_glc] if pp_feat is not None else None

            hw_tensor = torch.tensor((H_dec, W_dec), device=ids.device, dtype=torch.long)
            for ids_chunk in idset:
                o_ids = o_[ids_chunk, :, :]
                coords = ind2coords(hw_tensor, ids_chunk).expand(n_valid, -1, -1, -1)

                glc_ids = F.grid_sample(glc_batch, coords, mode="bilinear", align_corners=False)
                glc_ids = glc_ids.reshape(n_valid, -1, len(ids_chunk)).permute(2, 0, 1)

                # [fix1] Sample the per-pixel branch features at the same
                # decoder coordinates.  ``coords`` is normalised to [-1,1]
                # so grid_sample adapts to the full input resolution of
                # ``pp_batch`` automatically.
                pp_ids = None
                if pp_batch is not None:
                    pp_ids = F.grid_sample(
                        pp_batch, coords, mode="bilinear", align_corners=False,
                    )
                    pp_ids = pp_ids.reshape(
                        n_valid, -1, len(ids_chunk)
                    ).permute(2, 0, 1)

                extra_feats = []
                if pp_ids is not None:
                    extra_feats.append(pp_ids)
                x = torch.cat([o_ids, glc_ids] + extra_feats, dim=2)
                upglc_ids = self.glc_upsample(x)
                x = torch.cat([o_ids, upglc_ids] + extra_feats, dim=2)
                x = self.glc_aggregation(x)

                if return_intermediate:
                    total_upglc[b, ids_chunk, target_local, :] = upglc_ids
                    total_aggx[b, ids_chunk, :] = x
                x_n = self.regressor(x, len(ids_chunk), n_imgs=n_valid)
                nout[b, ids_chunk, :] = x_n

        nout = nout.permute(0, 2, 1).reshape(B, 3, H_dec, W_dec)
        nout = F.normalize(nout, dim=1, eps=1e-6)

        if return_intermediate:
            mid_results["upglc"] = total_upglc
            mid_results["aggx"] = total_aggx
            return mid_results, nout
        return nout
