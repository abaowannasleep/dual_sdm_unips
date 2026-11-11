"""Image-only student inference wrapper for the efficient DualInputNet variant."""

from __future__ import annotations

import torch

from .model_fast import DualInputNet


class StudentModel(DualInputNet):
    """Student inference model whose forward API accepts only an image stack.

    The model keeps the efficient per-pixel branch / light-calibration options
    from :class:`model_fast.DualInputNet`, but callers configure mask, image
    counts, resolutions and optional light directions through
    :meth:`set_inference_context` before calling ``forward(img)``.  No shading
    tensor is accepted; if a light-weighted encoder is enabled, the image stack
    itself is used as the photometric input.
    """

    def __init__(
        self,
        pixel_samples,
        device,
        canonical_resolution=128,
        decoder_resolution=256,
        **kwargs,
    ):
        super().__init__(pixel_samples=pixel_samples, device=device, **kwargs)
        self.default_canonical_resolution = int(canonical_resolution)
        self.default_decoder_resolution = int(decoder_resolution)
        self._mask = None
        self._n_img_array = None
        self._light_param = None

    def set_inference_context(
        self,
        mask=None,
        nImgArray=None,
        canonical_resolution=None,
        decoder_resolution=None,
        light_param=None,
    ):
        self._mask = mask
        self._n_img_array = nImgArray
        if canonical_resolution is not None:
            self.default_canonical_resolution = int(canonical_resolution)
        if decoder_resolution is not None:
            self.default_decoder_resolution = int(decoder_resolution)
        self._light_param = light_param

    def forward(self, img):
        B, _, H, W, Nmax = img.shape
        device = img.device
        mask = self._mask
        if mask is None:
            mask = torch.ones(B, 1, H, W, device=device, dtype=img.dtype)
        else:
            mask = mask.to(device=device, dtype=img.dtype)

        n_img_array = self._n_img_array
        if n_img_array is None:
            n_img_array = torch.full((B,), Nmax, device=device, dtype=torch.long)
        else:
            n_img_array = n_img_array.to(device=device, dtype=torch.long).clamp(max=Nmax)

        canonical = torch.tensor(
            [[self.default_canonical_resolution]], device=device, dtype=torch.long
        )
        decoder = torch.tensor(
            [[self.default_decoder_resolution]], device=device, dtype=torch.long
        )
        use_shading_path = bool(self.uses_light_input)
        light_param = self._light_param.to(device) if self._light_param is not None else None
        normal = super().forward(
            img,
            img,
            mask,
            n_img_array,
            canonical,
            decoder,
            useShading=use_shading_path,
            light_param=light_param,
            return_intermediate=False,
        )
        return normal


__all__ = ["StudentModel"]
