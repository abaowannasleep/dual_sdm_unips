"""
MultiColorPS Dataset Loader
Loads N RGB images, normal maps, shading maps, and masks from ColorPS dataset
"""

import os
import numpy as np
import cv2
import random
import torch
import torch.utils.data as data

# Reuse the ColorPS ``lights.txt`` parser / normaliser so that the light
# directions surfaced here follow exactly the same convention as
# ``ColorPSNMPS`` (world-space point-light positions L2-normalised to
# far-field unit vectors).
from .colorpsNMPS_loader import (
    _normalise_lights,
    parse_lights_txt,
)
# random.seed(42)


def _imread_required(path, flags, role):
    image = cv2.imread(path, flags)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Failed to read {role} image: {path}")
    return image


class MultiColorPS(data.Dataset):
    # Image-number ranges (1-based file naming). These are class-level
    # defaults; instances may override them via ``__init__`` kwargs (e.g.
    # ``test_img_min``) when the on-disk dataset uses a different layout.
    TRAIN_IMG_MIN = 1
    TRAIN_IMG_MAX = 10
    TEST_IMG_MIN = 11
    TEST_IMG_MAX = 32

    # --- Train-time photometric augmentation parameter ranges ---
    # Rationale:
    #   * AUG_GAIN_RANGE (per-channel multiplicative gain): models white-balance
    #     / color-temperature / spectral-sensitivity drift across cameras.
    #     A ±20% swing covers realistic WB variation without driving pixels
    #     out of a sensible exposure range.
    #   * AUG_NOISE_STD_RANGE (std of additive Gaussian noise on normalized
    #     [0,1] images): models CMOS read/shot noise. Upper bound 0.02 keeps
    #     SNR roughly above 34 dB, mild enough not to wash out photometric
    #     cues used for normal estimation.
    #   * AUG_CROSSTALK_OFF (off-diagonal of the 3x3 RGB mixing matrix) and
    #     AUG_CROSSTALK_DIAG (diagonal): simulate ~5% spectral leakage
    #     between color filters, which is typical for real Bayer sensors.
    AUG_GAIN_RANGE = (0.8, 1.2)
    AUG_NOISE_STD_RANGE = (0.0, 0.02)
    AUG_CROSSTALK_OFF = (-0.05, 0.05)
    AUG_CROSSTALK_DIAG = (0.95, 1.05)
    SHADING_BLENDER_GAMMA = 1.231
    SHADING_BLENDER_BIAS_OVER_GAIN = -0.086

    def __init__(self, data_root, type='train', num_images=4, mask_margin=16, max_image_resolution=2048, keep_ratio=1, num_epochs=None,
                 test_img_min=None, test_img_max=None, train_img_min=None, train_img_max=None,
                 augment=None, augment_shading=None, shading_process=None):
        """
        Multi-image ColorPS Dataset Loader
        Loads N RGB images, normal maps, shading maps, and masks from ColorPS dataset

        Sampling behavior:
            - type='train': one sample per class **per epoch**. The number of
              images N for that sample is drawn at ``__getitem__`` time from a
              distribution that depends on the current training epoch.

              When ``num_epochs`` is None (legacy / single-stage training),
              the distribution is the static one used historically:
                  P(N=1)         = 30%
                  P(N in {2,3})  = 30%
                  P(N in {4,5})  = 20%
                  P(N in {6..10})= 20%

              When ``num_epochs`` is provided (and ``set_epoch`` is called at
              the start of each epoch), the distribution becomes a linear
              ramp-up on N=1:
                  P(N=1) grows linearly from 0.0 at epoch 0 to 0.5 at the
                  final epoch (``num_epochs - 1``). The remaining
                  ``1 - P(N=1)`` is split among the other buckets in their
                  original 3:2:2 ratio:
                      P({2,3})   = (1 - P(N=1)) * 3/7
                      P({4,5})   = (1 - P(N=1)) * 2/7
                      P({6..10}) = (1 - P(N=1)) * 2/7
              The N images are drawn (without replacement) from image numbers
              1..10 (file names ``1.png``..``10.png``).
            - type='test': multiple non-overlapping samples per class, each
              containing ``num_images`` images. Samples are taken as contiguous
              windows across images 11..32 (22 images), producing
              ``floor(22 / num_images)`` samples per class.
              ``num_images`` must be in [1, 22].

        Args:
            data_root: Root directory containing class folders
            type: Dataset split – 'train' or 'test'
            num_images: N - number of images per sample. Used in test mode as
                the fixed window size. In train mode it is ignored because the
                size is drawn from the distribution above.
            mask_margin: Margin for mask processing
            max_image_resolution: Maximum resolution for images
            keep_ratio: Fraction of samples to retain (0 < keep_ratio <= 1).
                Samples are shuffled before truncation so a random subset is
                kept. Useful for quick training runs on a subset of data.
            augment: Enable train-time photometric augmentation for input
                images. Defaults to train mode only.
            augment_shading: Apply the same sampled photometric augmentation
                parameters to shading maps. Defaults to the value of
                ``augment`` so image/shading domains stay aligned.
            shading_process: Optional shading preprocessing mode. Set to
                ``'blender_like'`` (or True) to map clean synthetic shading
                towards the measured Blender fit (gamma=1.231, b/k=-0.086).

        Data structure:
            data_root/
                class_1/
                    1.png, 2.png, ..., 32.png (RGB images)
                    1_shading.png, 2_shading.png, ..., 32_shading.png
                    mask.png
                    normal.png
                class_2/
                    ...

        Returns per sample:
            I: (3, h, w, N) - RGB images with N image count dimension
            N: (3, h, w) - Normal map (unit vectors in [-1, 1])
            M: (1, h, w) - Mask
            S: (3, h, w, N) - Shading maps in [0, 1]
            numberOfImages: N
            L: (N, 3, 3) - per-image R/G/B unit light directions (xyz on
                the last axis), parsed from ``<class>/lights.txt`` using
                the same convention as ``ColorPSNMPS`` (world-space
                point-light positions L2-normalised to far-field unit
                vectors). If ``lights.txt`` is missing for a class, a
                zero-filled placeholder of the same shape is returned;
                consumers that don't need lights (e.g. the legacy 5-tuple
                callers) can simply ignore this 6th element.
        """
        self.data_root = data_root
        self.num_images = num_images
        self.mask_margin = mask_margin
        self.max_image_resolution = max_image_resolution
        self.samples = []  # list of (class_path, [img_numbers]) for test, or (class_path, None) for train
        self._warned_missing_shading = False
        self.type = type
        self.num_epochs = num_epochs
        self._current_epoch = 0
        # Enable photometric augmentation by default in train mode only.
        # ``augment=False`` lets callers explicitly disable it (e.g. for
        # debugging / reproducibility); ``augment=True`` forces it on.
        self.augment = (type == 'train') if augment is None else bool(augment)
        self.augment_shading = self.augment if augment_shading is None else bool(augment_shading)
        self.shading_process = self._normalise_shading_process(shading_process)

        # Per-instance overrides for image-number ranges. Falling back to the
        # class-level defaults preserves backward compatibility for callers
        # that don't pass these kwargs.
        if train_img_min is not None:
            self.TRAIN_IMG_MIN = int(train_img_min)
        if train_img_max is not None:
            self.TRAIN_IMG_MAX = int(train_img_max)
        if test_img_min is not None:
            self.TEST_IMG_MIN = int(test_img_min)
        if test_img_max is not None:
            self.TEST_IMG_MAX = int(test_img_max)

        if self.TEST_IMG_MIN > self.TEST_IMG_MAX:
            raise ValueError(
                f'test_img_min ({self.TEST_IMG_MIN}) must be <= '
                f'test_img_max ({self.TEST_IMG_MAX}).'
            )
        if self.TRAIN_IMG_MIN > self.TRAIN_IMG_MAX:
            raise ValueError(
                f'train_img_min ({self.TRAIN_IMG_MIN}) must be <= '
                f'train_img_max ({self.TRAIN_IMG_MAX}).'
            )

        if type == 'test':
            test_pool = self.TEST_IMG_MAX - self.TEST_IMG_MIN + 1  # 22
            if not (1 <= num_images <= test_pool):
                raise ValueError(
                    f'For test, num_images ({num_images}) must be in '
                    f'[1, {test_pool}] (image numbers '
                    f'{self.TEST_IMG_MIN}..{self.TEST_IMG_MAX}).'
                )

        # Scan all classes
        class_dirs = sorted([d for d in os.listdir(data_root)
                            if os.path.isdir(os.path.join(data_root, d))])
        skipped_classes = []

        for class_dir in class_dirs:
            class_path = os.path.join(data_root, class_dir)
            missing_required = [
                name for name in ('normal.png', 'mask.png')
                if not os.path.exists(os.path.join(class_path, name))
            ]
            if missing_required:
                skipped_classes.append((class_dir, missing_required))
                continue

            if type == 'test':
                # Multiple non-overlapping windows of size num_images across
                # the test pool (images 11..32), one window per sample.
                test_pool = self.TEST_IMG_MAX - self.TEST_IMG_MIN + 1
                samples_per_class = test_pool // num_images
                for sample_idx in range(samples_per_class):
                    start = self.TEST_IMG_MIN + sample_idx * num_images
                    img_numbers = list(range(start, start + num_images))
                    self.samples.append((class_path, img_numbers))
            else:
                # One sample per class. The number of images and the actual
                # image numbers are sampled fresh at __getitem__ time so that
                # each epoch sees a different distribution / draw.
                self.samples.append((class_path, None))

        if skipped_classes:
            preview = ', '.join(
                f"{name} (missing {', '.join(missing)})"
                for name, missing in skipped_classes[:5]
            )
            suffix = '' if len(skipped_classes) <= 5 else f', ... +{len(skipped_classes) - 5} more'
            print(
                f"[MultiColorPS] Skipped {len(skipped_classes)} class dirs in {data_root} "
                f"without required supervision files: {preview}{suffix}"
            )

        print(f'Total samples: {len(self.samples)}\n')

        if keep_ratio < 1:
            random.shuffle(self.samples)
            keep = max(1, int(len(self.samples) * keep_ratio))
            self.samples = self.samples[:keep]
            print(f'Samples after keep_ratio={keep_ratio}: {len(self.samples)}\n')

    def set_epoch(self, epoch):
        """Update the current training epoch.

        Should be called by the training loop at the start of each epoch
        (before the DataLoader is iterated, so that any spawned workers
        observe the new epoch through the dataset that gets pickled).
        """
        self._current_epoch = int(epoch)

    def _train_distribution(self):
        """Return the (p1, p23, p45, p6_10) distribution for the current
        epoch.

        - If ``num_epochs`` is None, fall back to the static legacy
          distribution (30 / 30 / 20 / 20).
        - Otherwise linearly ramp P(N=1) from 0.0 at epoch 0 up to 0.5 at
          ``num_epochs - 1``. The remaining mass ``1 - P(N=1)`` is split
          across {2,3}, {4,5}, {6..10} in their original 3:2:2 proportion.
        """
        if self.num_epochs is None:
            return 0.30, 0.30, 0.20, 0.20

        denom = max(self.num_epochs - 1, 1)
        progress = min(max(self._current_epoch / denom, 0.0), 1.0)
        p1 = 0.5 * progress
        remain = 1.0 - p1
        # Original ratio among non-1 buckets: 0.30 : 0.20 : 0.20 = 3 : 2 : 2
        p23 = remain * (3.0 / 7.0)
        p45 = remain * (2.0 / 7.0)
        p6_10 = remain * (2.0 / 7.0)
        return p1, p23, p45, p6_10

    def _draw_train_count(self):
        """Draw the number of training images per sample.

        Distribution depends on the current epoch (see ``_train_distribution``).
        The draw is clamped to the configured training pool size
        (``TRAIN_IMG_MAX - TRAIN_IMG_MIN + 1``) so that custom, smaller pools
        never produce an ``n`` larger than the number of available images
        (which would crash ``random.sample`` in ``__getitem__``).
        """
        train_pool = self.TRAIN_IMG_MAX - self.TRAIN_IMG_MIN + 1
        p1, p23, p45, _p6_10 = self._train_distribution()
        r = random.random()
        if r < p1:
            n = 1
        elif r < p1 + p23:
            n = random.randint(2, 3)
        elif r < p1 + p23 + p45:
            n = random.randint(4, 5)
        else:
            n = random.randint(6, 10)
        # Clamp to the configured pool so a small TRAIN_IMG_MAX doesn't
        # produce an n that exceeds the number of available images.
        return max(1, min(n, train_pool))

    def _normalise_shading_process(self, shading_process):
        if shading_process in (None, False):
            return None
        if isinstance(shading_process, str):
            shading_process = shading_process.strip().lower().replace('-', '_')
        if shading_process in ('none', 'clean'):
            return None
        if shading_process is True or shading_process in ('blender_like', 'blender', 'blender_fit'):
            return 'blender_like'
        raise ValueError(
            "Unsupported shading_process {!r}. Use None/False or "
            "'blender_like'.".format(shading_process)
        )

    def _sample_augmentation_params(self):
        """Sample one photometric transform for a multi-shot sample."""
        gain_lo, gain_hi = self.AUG_GAIN_RANGE
        gain = np.float32(np.random.uniform(gain_lo, gain_hi, size=(3,)))

        off_lo, off_hi = self.AUG_CROSSTALK_OFF
        diag_lo, diag_hi = self.AUG_CROSSTALK_DIAG
        crosstalk = np.random.uniform(off_lo, off_hi, size=(3, 3)).astype(np.float32)
        diag = np.random.uniform(diag_lo, diag_hi, size=(3,)).astype(np.float32)
        np.fill_diagonal(crosstalk, diag)

        noise_lo, noise_hi = self.AUG_NOISE_STD_RANGE
        noise_std = float(np.random.uniform(noise_lo, noise_hi))
        return gain, crosstalk, noise_std

    def _apply_augmentation(self, imgs, params):
        """Apply sampled photometric parameters to float32 images.

        ``None`` entries are preserved so callers can keep missing shading
        slots aligned with the corresponding input images.
        """
        gain, crosstalk, noise_std = params
        augmented = []
        for img in imgs:
            if img is None:
                augmented.append(None)
                continue
            # Channel gain: (h, w, 3) * (3,)
            out = img * gain
            # RGB crosstalk: mix channels via 3x3 matrix. Using out @ M.T so
            # that row i of M defines how output channel i is composed from
            # input channels.
            out = out.reshape(-1, 3) @ crosstalk.T
            out = out.reshape(img.shape)
            # Additive Gaussian noise (per-image draw)
            if noise_std > 0:
                out = out + np.random.normal(
                    0.0, noise_std, size=out.shape
                ).astype(np.float32)
            # Keep values in the valid range
            out = np.clip(out, 0.0, 1.0).astype(np.float32)
            augmented.append(out)
        return augmented

    def _process_shading(self, shading):
        if shading is None or self.shading_process is None:
            return shading
        if self.shading_process == 'blender_like':
            gamma = self.SHADING_BLENDER_GAMMA
            bias = self.SHADING_BLENDER_BIAS_OVER_GAIN
            out = np.power(np.clip(shading, 0.0, 1.0), gamma) + bias
            scale = max(1.0 + bias, 1e-6)
            out = out / scale
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        return shading
    
    def _augment_images(self, imgs):
        """Apply random photometric perturbations to a list of float32 images
        normalized to [0, 1].

        The same per-channel gain and RGB crosstalk matrix are applied to all
        N images of the sample (they model camera characteristics, which are
        fixed during a capture session, so the relative photometric
        relationships across the N lights are preserved). Gaussian noise is
        drawn independently for each image (sensor noise varies per shot).

        Args:
            imgs: list of np.float32 arrays of shape (h, w, 3) in [0, 1].
                ``None`` entries are preserved.
        Returns:
            list of np.float32 arrays of shape (h, w, 3), values clipped to
            [0, 1].
        """
        return self._apply_augmentation(imgs, self._sample_augmentation_params())

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Load N images from a single sample from MultiColorPS dataset
        
        Returns:
            I: (3, h, w, N) - RGB images
            N: (3, h, w) - Normal map (unit vectors, [-1, 1])
            M: (1, h, w) - Mask
            S: (3, h, w, N) - Shading maps [0, 1]
            numberOfImages: N
            L: (N, 3, 3) - per-image R/G/B unit light directions parsed
                from ``<class>/lights.txt`` (zero-filled placeholder when
                the file is absent). Suitable to feed into
                ``model_fast``'s ``light_param`` argument (batched to
                ``[B, Nmax, 3, 3]`` by the data loader).
        """
        class_path, img_numbers = self.samples[index]
        # In train mode the sample stores img_numbers=None; the count and the
        # actual image numbers are drawn fresh on each access using a
        # distribution that may depend on the current epoch (see set_epoch /
        # _train_distribution).
        if img_numbers is None:
            n = self._draw_train_count()
            img_numbers = sorted(random.sample(
                range(self.TRAIN_IMG_MIN, self.TRAIN_IMG_MAX + 1),
                n
            ))
        # print(class_path)
        # Read normal and mask (same for all N images)
        normal_path = os.path.join(class_path, 'normal.png')
        mask_path = os.path.join(class_path, 'mask.png')
        
        normal = cv2.cvtColor(
            _imread_required(
                normal_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH, 'normal'
            ),
            cv2.COLOR_BGR2RGB
        )
        mask = _imread_required(
            mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH, 'mask'
        )
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # --- Crop based on mask bounding box ---
        h0, w0 = mask.shape
        rows, cols = np.nonzero(mask)
        if len(rows) > 0 and len(cols) > 0:
            rowmin, rowmax = np.min(rows), np.max(rows)
            colmin, colmax = np.min(cols), np.max(cols)
            row, col = rowmax - rowmin, colmax - colmin
            margin = self.mask_margin

            if row > col:
                r_s, r_e = rowmin - margin, rowmax + margin
                c_s = max(colmin - int(0.5 * (row - col)) - margin, 0)
                c_e = min(colmax + int(0.5 * (row - col)) + margin, w0)
            else:
                r_s = max(rowmin - int(0.5 * (col - row)) - margin, 0)
                r_e = min(rowmax + int(0.5 * (col - row)) + margin, h0)
                c_s, c_e = colmin - margin, colmax + margin

            # Clip to image bounds
            r_s, r_e = max(r_s, 0), min(r_e, h0)
            c_s, c_e = max(c_s, 0), min(c_e, w0)
        else:
            r_s, r_e, c_s, c_e = 0, h0, 0, w0

        # Crop normal and mask
        normal = normal[r_s:r_e, c_s:c_e, :]
        mask = mask[r_s:r_e, c_s:c_e]

        # --- Read the selected images and shading maps ---
        imgs = []
        shadings = []

        for img_num in img_numbers:
            img_path = os.path.join(class_path, f'{img_num}.png')
            shading_path = os.path.join(class_path, f'{img_num}_shading.png')
            
            # Read image (RGB)
            img = cv2.cvtColor(
                _imread_required(
                    img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH, 'input'
                ),
                cv2.COLOR_BGR2RGB
            )
            # # Apply chrominance normalization: r=r/(r+g+b), g=g/(r+g+b), b=b/(r+g+b)
            # img_sum = np.sum(img, axis=2, keepdims=True) + 1e-8
            # img = img / img_sum
            
            # Read shading if file exists; otherwise mark missing
            if os.path.exists(shading_path):
                shading = cv2.cvtColor(
                    _imread_required(
                        shading_path,
                        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
                        'shading',
                    ),
                    cv2.COLOR_BGR2RGB
                )
            else:
                if not self._warned_missing_shading:
                    # print("  Warning: Shading file(s) not found; using normal as shading for missing files.")
                    self._warned_missing_shading = True
                shading = None
            
            # Apply crop
            img = img[r_s:r_e, c_s:c_e, :]
            if shading is not None:
                shading = shading[r_s:r_e, c_s:c_e, :]
            
            imgs.append(img)
            shadings.append(shading)
        sample_num_images = len(imgs)

        # --- Resize to square resolution ---
        h = int(np.floor(max(imgs[0].shape[0], imgs[0].shape[1]) / 256) * 256)
        h = min(max(h, 256), self.max_image_resolution)
        w = h
        
        imgs_resized = []
        shadings_resized = []
        
        for i in range(sample_num_images):
            img_resized = cv2.resize(imgs[i], (w, h), interpolation=cv2.INTER_CUBIC)
            if shadings[i] is not None:
                shading_resized = cv2.resize(shadings[i], (w, h), interpolation=cv2.INTER_CUBIC)
            else:
                shading_resized = None
            imgs_resized.append(img_resized)
            shadings_resized.append(shading_resized)
        
        normal = cv2.resize(normal, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = np.float32(cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC) > 0.5)


        # 分别用各自 dtype 归一化
        imgs_resized = [np.float32(img) / (255.0 if img.dtype == np.uint8 else 65535.0 if img.dtype == np.uint16 else 1.0) for img in imgs_resized]
        shadings_resized = [
            (np.float32(s) / (255.0 if s.dtype == np.uint8 else 65535.0 if s.dtype == np.uint16 else 1.0)) if s is not None else None
            for s in shadings_resized
        ]
        if self.shading_process is not None:
            shadings_resized = [
                self._process_shading(s) for s in shadings_resized
            ]

        if self.augment or self.augment_shading:
            aug_params = self._sample_augmentation_params()
            if self.augment:
                imgs_resized = self._apply_augmentation(imgs_resized, aug_params)
            if self.augment_shading:
                shadings_resized = self._apply_augmentation(shadings_resized, aug_params)

        shadings_resized = [
            s if s is not None else imgs_resized[i]
            for i, s in enumerate(shadings_resized)
        ]

        # normal 用自己的 bit_depth
        normal_bit_depth = 255.0 if normal.dtype == np.uint8 else 65535.0 if normal.dtype == np.uint16 else 1.0
        normal = np.float32(normal) / normal_bit_depth
        normal = 2 * normal - 1
        normal = normal / (np.sqrt(np.sum(normal * normal, axis=2, keepdims=True)) + 1e-8)
        mask = np.float32(mask > 0)

        # Stack N images and shadings
        img_stacked = np.stack(imgs_resized, axis=-1)  # (h, w, 3, N)
        shading_stacked = np.stack(shadings_resized, axis=-1)  # (h, w, 3, N)
        
        # Transpose to (3, h, w, N)
        img_stacked = np.transpose(img_stacked, (2, 0, 1, 3))  # (3, h, w, N)
        normal = np.transpose(normal, (2, 0, 1))  # (3, h, w)
        shading_stacked = np.transpose(shading_stacked, (2, 0, 1, 3))  # (3, h, w, N)
        mask = mask[:, :, np.newaxis]  # (h, w, 1)
        mask = np.transpose(mask, (2, 0, 1))  # (1, h, w)
        
        numberOfImages = sample_num_images

        # --- Light directions (R/G/B unit vectors, one set per image) -----
        # Same convention as ``ColorPSNMPS``: ``lights.txt`` stores
        # world-space point-light *positions*; we L2-normalise them to
        # far-field unit *directions* pointing from the object (assumed
        # at the world origin) towards each light. Layout is
        # ``(image, RGB-channel, xyz)`` so the result reshapes directly
        # into ``model_fast``'s ``[B, Nmax, 3, 3]`` ``light_param``
        # tensor. When ``lights.txt`` is missing for a class we fall
        # back to a zero-filled placeholder of the same shape so the
        # legacy 5-tuple callers and calibration-free architectures keep
        # working unchanged.
        lights_path = os.path.join(class_path, 'lights.txt')
        if os.path.exists(lights_path):
            dirs_all = parse_lights_txt(lights_path)
            num_light_rows = dirs_all.shape[0]
            max_requested = max(img_numbers)
            if max_requested > num_light_rows:
                raise ValueError(
                    f"lights.txt in {class_path!r} has {num_light_rows} rows "
                    f"but image number {max_requested} was requested. Each "
                    f"image k.png must have a corresponding row k in "
                    f"lights.txt."
                )
            sel_idx = [num - 1 for num in img_numbers]
            dirs_sel = dirs_all[sel_idx]                       # (n, 3, 3)
            dirs_sel = _normalise_lights(dirs_sel)
        else:
            dirs_sel = np.zeros(
                (sample_num_images, 3, 3), dtype=np.float32
            )
        L = torch.from_numpy(np.ascontiguousarray(dirs_sel)).float()

        return img_stacked, normal, mask, shading_stacked, numberOfImages, L
