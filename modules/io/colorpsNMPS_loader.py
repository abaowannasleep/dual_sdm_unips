"""
ColorPS dataset loader for the NeuralMPS architecture.

This module is a port of NeuralMPS' ``datasets/my_npy_dataloader.py`` to the
ColorPS dataset layout (same directory structure as ``MultiColorPS`` in
``modules/io/multicolorps.py``) with one important addition: each class
directory may carry a ``lights.txt`` file describing the calibrated lights
for every image in the class.

``lights.txt`` is **optional**. When present it is parsed and exposed as
calibrated light directions (used by NeuralMPS). When
absent (e.g. for the calibration-free SpectraMPS architecture) the
loader silently substitutes zero-filled placeholder tensors of the same
shape for ``dirs`` so callers that ignore those fields (such
as SpectraMPS) continue to work without modification.

`lights.txt` format
-------------------
One line per image (line ``k`` corresponds to ``k.png``). Each line starts
with three light-vector fields separated by ``;``::

    (x_R, y_R, z_R); (x_G, y_G, z_G); (x_B, y_B, z_B)

* The first three fields are the **3-D world-space positions** of the
  R/G/B point lights illuminating that image (one position per RGB
  channel). The object is assumed to be approximately centred at the
  world origin, so each position doubles as a far-field unit *direction*
  pointing from the object towards the light after L2 normalisation.
  This is the standard "distant-light" approximation; near-field per-pixel
  falloff is **not** modelled here (see ``LS_prior_teacher_student_dev_doc_v2.md``).
We L2-normalise each position vector to unit length (yielding a far-field
direction). Any trailing fields after the three light vectors are ignored.

Per-sample outputs (NeuralMPS layout)
-------------------------------------
::

    img              : torch.FloatTensor  (3*N, H, W)
    dirs             : torch.FloatTensor  (3*N, 1, 1)
        per-image mean light direction; slot ``3k+a`` holds component
        ``a`` (x/y/z) of the (channel-averaged) direction for image ``k``.
        Beware that averaging three unit vectors can produce a tiny
        vector when the channel directions are nearly antipodal — use
        ``dirs_per_channel`` if exact R/G/B directions are required.
    dirs_per_channel : torch.FloatTensor  (9*N, 1, 1)
        full per-channel light directions; reshape to ``(N, 3, 3)`` as
        ``(image, RGB-channel, xyz)``.
    ints             : torch.FloatTensor  (3*N, 1, 1)  all-ones compatibility placeholder
    normal           : torch.FloatTensor  (3,   H, W)  in [-1, 1]
    mask             : torch.FloatTensor  (1,   H, W)  in [0, 1]
    obj              : str                              class name

For compatibility with the existing SDM trainer we also expose
``ColorPSNMPS.collate_sdm`` (a custom ``collate_fn``) and an alternative
output mode (``return_format='sdm'``) that returns the same five-tuple as
``MultiColorPS`` (``I, N, M, S, nImgArray``) plus the calibrated lights as a
6th element and an all-ones compatibility placeholder as a 7th element. In
the SDM path ``dirs`` is the **per-channel** tensor (shape ``(9*N, 1, 1)``,
reshape to ``(N, 3, 3)``) so that downstream LS solvers receive the true
R/G/B-to-light correspondence needed by the DV0-1 round-trip sanity check.
"""

from __future__ import division

import os
import re
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data


# Regex to parse a single 3-vector inside parentheses, e.g.
#   "(2.12e-16, 3.46, 2.0)"
_VEC_RE = re.compile(
    r"\(\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*,"
    r"\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*,"
    r"\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*\)"
)


def parse_lights_txt(path):
    """Parse a ``lights.txt`` file into a light-position array.

    Returns
    -------
    positions : np.ndarray of shape ``(K, 3, 3)``
        ``positions[k, c]`` is the **3-D world-space position** of the
        point light illuminating image ``k`` on RGB channel ``c``. The
        object is assumed to be approximately centred at the world
        origin, so each position acts as a far-field unit direction
        (pointing from the object towards the light) once L2-normalised
        by :func:`_normalise_lights`. Near-field per-pixel falloff is
        **not** modelled here.
    """
    dirs = []
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            vecs = _VEC_RE.findall(line)
            if len(vecs) < 3:
                raise ValueError(
                    f"Line in {path!r} does not contain 3 light vectors: {line!r}"
                )
            d = np.array(
                [[float(v) for v in vec] for vec in vecs[:3]],
                dtype=np.float32,
            )  # (3, 3)
            dirs.append(d)

    if not dirs:
        raise ValueError(f"No light entries parsed from {path!r}")
    return np.stack(dirs, axis=0)


# Minimum acceptable norm for a light-position vector. Anything below
# this is almost certainly a data bug (e.g. a light placed at the world
# origin or a parse error) and would otherwise be silently rescued by an
# epsilon, producing a near-arbitrary unit direction. We raise instead
# so that the caller learns about the problem at load time.
_MIN_LIGHT_NORM = 1e-3


def _normalise_lights(dirs):
    """Convert world-space point-light **positions** to far-field unit
    direction vectors.

    The conversion is the standard "object at origin" far-field
    approximation: ``l = pos / ||pos||``. ``l`` therefore points from
    the object towards the light, which is the convention expected by
    the LS solver ``S = max(0, n . l)``.

    ``dirs`` : (..., 3, 3) per-channel light positions, last dim is xyz.

    Returns
    -------
    dirs_norm : np.ndarray, same shape as ``dirs``, unit-norm along last
        dim.

    Raises
    ------
    ValueError
        If any input position has a norm smaller than
        :data:`_MIN_LIGHT_NORM` (which would otherwise yield a
        near-arbitrary unit direction after dividing by ``eps``).
    """
    dirs = np.asarray(dirs, dtype=np.float32)
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    bad_mask = norm[..., 0] < _MIN_LIGHT_NORM
    if np.any(bad_mask):
        bad = np.argwhere(bad_mask)
        raise ValueError(
            f"Encountered light position(s) with norm < {_MIN_LIGHT_NORM} at "
            f"indices {bad.tolist()} — refusing to silently rescue with an "
            "epsilon as that would yield an unstable far-field direction. "
            "Check the corresponding lights.txt rows."
        )
    dirs_norm = dirs / norm
    return dirs_norm


def _crop_bbox(mask, margin):
    """Compute a square-ish crop region from a binary mask."""
    h0, w0 = mask.shape
    rows, cols = np.nonzero(mask)
    if len(rows) == 0 or len(cols) == 0:
        return 0, h0, 0, w0
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    row, col = rmax - rmin, cmax - cmin

    if row > col:
        r_s, r_e = rmin - margin, rmax + margin
        c_s = max(cmin - (row - col) // 2 - margin, 0)
        c_e = min(cmax + (row - col) // 2 + margin, w0)
    else:
        r_s = max(rmin - (col - row) // 2 - margin, 0)
        r_e = min(rmax + (col - row) // 2 + margin, h0)
        c_s, c_e = cmin - margin, cmax + margin

    r_s, r_e = max(r_s, 0), min(r_e, h0)
    c_s, c_e = max(c_s, 0), min(c_e, w0)
    return r_s, r_e, c_s, c_e


class ColorPSNMPS(data.Dataset):
    """ColorPS dataset loader returning NeuralMPS-style tensors.

    Directory layout (per-class)::

        data_root/
            class_1/
                1.png, 2.png, ...               (N RGB images)
                1_shading.png, 2_shading.png ... (optional)
                mask.png
                normal.png
                lights.txt
            class_2/
                ...

    Parameters
    ----------
    data_root : str
        Root directory containing class folders.
    split : {'train', 'test'}
        ``'train'``: one sample per class per epoch. ``num_images`` images
        are drawn at ``__getitem__`` time from the configured pool
        (``train_img_min``..``train_img_max``).
        ``'test'``: contiguous, non-overlapping windows across the
        ``test_img_min``..``test_img_max`` pool, with each window of size
        ``num_images`` producing one sample.
    num_images : int
        Number of images per sample. Used as the fixed image count for
        both train and test (test windows are also of this size).
    max_image_resolution : int
        Maximum spatial resolution (longest side).
    mask_margin : int
        Margin (in pixels) added around the mask bounding box.
    return_format : {'nmps', 'sdm'}
        Output tuple format. ``'nmps'`` (default) returns NeuralMPS-style
        tensors ``(img, dirs, ints, normal, mask, obj)``. ``'sdm'`` returns
        the SDM trainer's 5-tuple ``(I, N, M, S, nImgArray)`` plus
        ``(dirs, ints)`` for a 7-element tuple.
    require_lights : bool, optional
        If True (default for ``return_format='nmps'``), classes without a
        ``lights.txt`` file are skipped and a missing file at
        ``__getitem__`` time is treated as an error. If False (default for
        ``return_format='sdm'``), missing ``lights.txt`` files are
        tolerated: such classes are still included and zero-filled
        placeholder ``dirs`` tensors are returned for them.
        This is intended for calibration-free architectures such as
        SpectraMPS which ignore ``dirs`` / ``ints`` entirely.

    Notes
    -----
    Following the upstream NeuralMPS design we keep the per-sample image
    count constant (== ``num_images``) for every batch element; the
    network does not require a variable N. Direction vectors read from
    ``lights.txt`` are normalised here to unit norm.
    """

    TRAIN_IMG_MIN = 1
    TRAIN_IMG_MAX = 10
    TEST_IMG_MIN = 11
    TEST_IMG_MAX = 32

    def __init__(
        self,
        data_root,
        split="train",
        num_images=4,
        mask_margin=16,
        max_image_resolution=512,
        return_format="nmps",
        train_img_min=None,
        train_img_max=None,
        test_img_min=None,
        test_img_max=None,
        keep_ratio=1.0,
        require_lights=None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if return_format not in ("nmps", "sdm"):
            raise ValueError(f"return_format must be 'nmps' or 'sdm', got {return_format!r}")

        self.data_root = data_root
        self.split = split
        self.num_images = int(num_images)
        self.mask_margin = int(mask_margin)
        self.max_image_resolution = int(max_image_resolution)
        self.return_format = return_format
        # Default: require lights for the calibrated NMPS path, tolerate
        # missing lights for the calibration-free SDM/SpectraMPS path.
        if require_lights is None:
            require_lights = (return_format == "nmps")
        self.require_lights = bool(require_lights)

        if train_img_min is not None:
            self.TRAIN_IMG_MIN = int(train_img_min)
        if train_img_max is not None:
            self.TRAIN_IMG_MAX = int(train_img_max)
        if test_img_min is not None:
            self.TEST_IMG_MIN = int(test_img_min)
        if test_img_max is not None:
            self.TEST_IMG_MAX = int(test_img_max)

        # Scan classes.
        class_dirs = sorted(
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        )

        self.samples = []  # list of (class_path, img_numbers_or_None)
        for class_dir in class_dirs:
            class_path = os.path.join(data_root, class_dir)
            # ``lights.txt`` is only required when calibrated lights are
            # needed (NMPS); for calibration-free use (SpectraMPS) we
            # tolerate classes without it.
            has_lights = os.path.exists(os.path.join(class_path, "lights.txt"))
            if self.require_lights and not has_lights:
                continue
            # Sanity-check: a class is only usable if it has a mask and a
            # ground-truth normal map.
            if not (
                os.path.exists(os.path.join(class_path, "mask.png"))
                and os.path.exists(os.path.join(class_path, "normal.png"))
            ):
                continue
            if split == "test":
                pool = self.TEST_IMG_MAX - self.TEST_IMG_MIN + 1
                if not (1 <= self.num_images <= pool):
                    raise ValueError(
                        f"num_images={self.num_images} not in [1, {pool}] for test pool "
                        f"{self.TEST_IMG_MIN}..{self.TEST_IMG_MAX}"
                    )
                samples_per_class = pool // self.num_images
                for s in range(samples_per_class):
                    start = self.TEST_IMG_MIN + s * self.num_images
                    img_nums = list(range(start, start + self.num_images))
                    self.samples.append((class_path, img_nums))
            else:
                self.samples.append((class_path, None))

        if keep_ratio < 1.0:
            random.shuffle(self.samples)
            keep = max(1, int(len(self.samples) * keep_ratio))
            self.samples = self.samples[:keep]

        print(f"[ColorPSNMPS:{split}] {len(self.samples)} samples from {data_root}")

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------------------------
    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _bit_depth(self, dtype):
        if dtype == np.uint8:
            return 255.0
        if dtype == np.uint16:
            return 65535.0
        return 1.0

    def __getitem__(self, index):
        class_path, img_numbers = self.samples[index]
        obj = os.path.basename(class_path)

        # ---------------- Lights -----------------------------------------
        lights_path = os.path.join(class_path, "lights.txt")
        if os.path.exists(lights_path):
            dirs_all = parse_lights_txt(lights_path)
            # 1-based image numbers → 0-based row index into dirs_all.
            num_light_rows = dirs_all.shape[0]
            has_lights = True
        elif self.require_lights:
            raise FileNotFoundError(
                f"lights.txt is required but missing in {class_path!r}. "
                f"Pass require_lights=False (or use return_format='sdm') to "
                f"tolerate datasets without calibrated lights."
            )
        else:
            dirs_all = None
            num_light_rows = None
            has_lights = False

        # ---------------- Choose image numbers ---------------------------
        # The network operates on a fixed ``num_images`` per sample; in
        # train mode we draw that many image numbers (without replacement)
        # from the configured training pool each epoch.
        if img_numbers is None:
            pool = list(range(self.TRAIN_IMG_MIN, self.TRAIN_IMG_MAX + 1))
            if self.num_images > len(pool):
                raise ValueError(
                    f"num_images={self.num_images} exceeds the training pool size "
                    f"{len(pool)} (range {self.TRAIN_IMG_MIN}..{self.TRAIN_IMG_MAX})."
                )
            img_numbers = sorted(random.sample(pool, self.num_images))
        n = len(img_numbers)

        # Index lights for the chosen images. ``lights.txt`` is expected
        # to have one row per available image (1-indexed); we fail fast
        # if a requested image number has no corresponding row, rather
        # than silently wrapping around with modulo. When lights are
        # absent we substitute zero-filled placeholders of the correct
        # shape (this is fine for callers that ignore ``dirs``,
        # e.g. SpectraMPS).
        if has_lights:
            max_requested = max(img_numbers)
            if max_requested > num_light_rows:
                raise ValueError(
                    f"lights.txt in {class_path!r} has {num_light_rows} rows but "
                    f"image number {max_requested} was requested. Each image "
                    f"k.png must have a corresponding row k in lights.txt."
                )
            sel_idx = [num - 1 for num in img_numbers]
            dirs_sel = dirs_all[sel_idx]               # (n, 3, 3)
            dirs_sel = _normalise_lights(dirs_sel)
        else:
            dirs_sel = np.zeros((n, 3, 3), dtype=np.float32)

        # ---------------- Geometry (normal + mask) -----------------------
        normal_path = os.path.join(class_path, "normal.png")
        mask_path = os.path.join(class_path, "mask.png")
        normal = self._load_image(normal_path)
        mask = self._load_image(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        r_s, r_e, c_s, c_e = _crop_bbox(mask, self.mask_margin)
        normal = normal[r_s:r_e, c_s:c_e, :]
        mask = mask[r_s:r_e, c_s:c_e]

        # ---------------- Images + optional shading ----------------------
        imgs, shadings = [], []
        for num in img_numbers:
            img_p = os.path.join(class_path, f"{num}.png")
            img = self._load_image(img_p)[r_s:r_e, c_s:c_e, :]
            imgs.append(img)
            shading_p = os.path.join(class_path, f"{num}_shading.png")
            if os.path.exists(shading_p):
                sh = self._load_image(shading_p)[r_s:r_e, c_s:c_e, :]
            else:
                sh = None
            shadings.append(sh)

        # ---------------- Resize to a multiple of 256 --------------------
        long_side = max(imgs[0].shape[0], imgs[0].shape[1])
        h = int(np.floor(long_side / 256) * 256)
        h = min(max(h, 256), self.max_image_resolution)
        w = h


        # 分别 resize
        imgs_r = [cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC) for im in imgs]
        shadings_r = [
            cv2.resize(s, (w, h), interpolation=cv2.INTER_CUBIC) if s is not None else None
            for s in shadings
        ]
        normal = cv2.resize(normal, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = np.float32(cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC) > 0.5)

        # ---------------- Normalise to float [0, 1]，每个分量用自己的 bit_depth ----------------------
        imgs_r = [np.float32(im) / self._bit_depth(im.dtype) for im in imgs_r]
        shadings_r = [
            (np.float32(s) / self._bit_depth(s.dtype)) if s is not None else None
            for s in shadings_r
        ]
        normal_bit_depth = self._bit_depth(normal.dtype)
        normal = np.float32(normal) / normal_bit_depth
        normal = 2.0 * normal - 1.0
        normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8)

        # ---------------- Build NeuralMPS-style tensors ------------------
        # img: (n, h, w, 3) -> (3*n, h, w)  (one image after another, RGB stacked on C)
        img_np = np.stack(imgs_r, axis=0)                          # (n, h, w, 3)
        img_np = np.transpose(img_np, (0, 3, 1, 2))                # (n, 3, h, w)
        img_np = img_np * mask[None, None, :, :]                   # apply mask
        img_3n = img_np.reshape(n * 3, h, w)                       # (3n, h, w)

        # ----------------------------------------------------------------
        # Direction layout (matches upstream NeuralMPS' ``(3*N, 1, 1)``):
        # the channel dim of ``imgs (3*N, H, W)`` is split into N groups of
        # 3; the matching 3-slice of ``dirs`` must carry the (dx, dy, dz)
        # of one direction. Upstream's multispectral data has *one*
        # direction per spectral channel; ColorPS has 3 directions per
        # image (one per RGB channel) so we collapse to a single
        # representative direction per image for the legacy NMPS ``dirs``
        # tensor and keep the full per-channel directions in
        # ``dirs_per_channel`` (also surfaced in the ``'sdm'`` return as
        # the primary ``dirs`` tensor).
        #
        # We use a mean-of-unit-vectors collapse here for the NMPS path
        # only. Averaging three nearly antipodal unit vectors can produce
        # a tiny vector whose direction is dominated by numerical noise.
        # We therefore guard against that case and fall back to the green
        # (middle) channel's direction, which is always well-defined.
        # ----------------------------------------------------------------
        dirs_per_image = dirs_sel.mean(axis=1)                      # (n, 3)
        per_img_norm = np.linalg.norm(dirs_per_image, axis=1, keepdims=True)
        degenerate = (per_img_norm[:, 0] < _MIN_LIGHT_NORM)
        if np.any(degenerate):
            # Fall back to the per-image green channel (index 1) for the
            # degenerate rows. This keeps the NMPS-format ``dirs`` tensor
            # well-defined; consumers that need the exact R/G/B
            # directions should read ``dirs_per_channel`` instead.
            dirs_per_image[degenerate] = dirs_sel[degenerate, 1]
            per_img_norm = np.linalg.norm(dirs_per_image, axis=1, keepdims=True)
        dirs_per_image = dirs_per_image / per_img_norm
        dirs_3n_flat = dirs_per_image.reshape(n * 3, 1, 1).astype(np.float32)
        # Full per-channel directions kept around as a flat ``(9*N, 1, 1)``
        # tensor so callers that need the exact R/G/B directions per
        # image can reshape to ``(N, 3, 3)`` themselves. Inner layout is
        # ``(image, RGB-channel, xyz)``.
        dirs_per_channel_flat = dirs_sel.reshape(-1, 1, 1).astype(np.float32)

        # All-ones placeholder retained for callers that still accept the
        # legacy intensity tensor; no light-intensity data is read or applied.
        ints_3n_flat = np.ones((n * 3, 1, 1), dtype=np.float32)

        # ---------------- Convert to torch tensors -----------------------
        img_t = torch.from_numpy(img_3n).float()
        dirs_t = torch.from_numpy(dirs_3n_flat).float()
        dirs_pc_t = torch.from_numpy(dirs_per_channel_flat).float().view(-1, 1, 1)
        ints_t = torch.from_numpy(ints_3n_flat).float()
        normal_t = torch.from_numpy(np.transpose(normal, (2, 0, 1))).float()
        mask_t = torch.from_numpy(mask[None, :, :]).float()

        # --------------- SDM-compatible 5-tuple --------------------------
        if self.return_format == "sdm":
            # I, S: (3, h, w, N) - replicate channel-first order; we already
            # have (n, 3, h, w) in img_np (pre-mask we re-use imgs_r).
            I_np = np.stack(imgs_r, axis=-1)                       # (h, w, 3, n)
            I_np = np.transpose(I_np, (2, 0, 1, 3))                # (3, h, w, n)
            if any(s is not None for s in shadings_r):
                S_list = []
                for i_s, s in enumerate(shadings_r):
                    if s is None:
                        # Fall back to the matching image (already normalised
                        # to float [0, 1]) rather than the first image so the
                        # i-th shading slot stays aligned with the i-th view.
                        S_list.append(np.float32(imgs_r[i_s]))
                    else:
                        S_list.append(np.float32(s) / bit_depth)
                S_np = np.stack(S_list, axis=-1)                  # (h, w, 3, n)
                S_np = np.transpose(S_np, (2, 0, 1, 3))           # (3, h, w, n)
            else:
                S_np = I_np.copy()

            # In the SDM path we expose the **per-channel** R/G/B light
            # directions as ``dirs`` (shape ``(9*N, 1, 1)``, reshape to
            # ``(N, 3, 3)`` as ``(image, RGB-channel, xyz)``) so the
            # downstream LS solver can use the true per-channel
            # correspondence required by DV0-1. The legacy per-image
            # mean tensor is still available as ``dirs_pc_t.view(-1, 1,
            # 1)[::3]`` if a caller really wants it.
            return (
                torch.from_numpy(I_np).float(),
                normal_t,
                mask_t,
                torch.from_numpy(S_np).float(),
                n,
                dirs_pc_t,
                ints_t,
            )

        # --------------- Default NeuralMPS dict --------------------------
        return {
            "img": img_t,                  # (3n, h, w)
            "dirs": dirs_t,                # (3n, 1, 1) — per image, repeated 3x
            "dirs_per_channel": dirs_pc_t, # (9n, 1, 1) — full per-channel dirs
            "ints": ints_t,                # (3n, 1, 1)
            "normal": normal_t,            # (3,  h, w)
            "mask": mask_t,                # (1,  h, w)
            "obj": obj,
            "n_imgs": n,
        }


__all__ = ["ColorPSNMPS", "parse_lights_txt"]
