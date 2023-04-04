"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# ==========================================================================================
#
# Adobe’s modifications are Copyright 2023 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
#
# ==========================================================================================

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expansion_utils import io_utils

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class ImageReps:
    name: str
    img: torch.FloatTensor = None

    w_code: torch.FloatTensor = None
    alpha: torch.FloatTensor = None
    deltas: torch.FloatTensor = None

    recon_img: torch.FloatTensor = None
    mask: torch.FloatTensor = None

    def __init__(self, name, img, w_code=None, mask=None):
        self.name = name
        self.img = img
        self.w_code = w_code
        self.mask = mask

    def set(self, latent=None, alpha=None, recon_img=None):
        if latent is not None:
            self.w_code = latent.detach().cpu()
        if alpha is not None:
            self.alpha = alpha.detach().cpu()
        if recon_img is not None:
            self.recon_img = recon_img.detach().cpu()

    @classmethod
    def load_sample(cls, image_path: Path, latent_path: Path = None, mask_path: Path = None, transform=None):
        img = Image.open(image_path).convert('RGB')
        if transform:
            img = transform(img)

        img = img.unsqueeze(0)

        w_code = None
        mask = None

        if latent_path:
            w_code = torch.load(latent_path).cpu()

        return cls(name=image_path.stem, img=img, w_code=w_code, mask=mask)


class PersonalizedDataset(Dataset):
    def __init__(self, images_dir: Path, latent_dir: Path = None, mask_dir: Path = None):
        self.samples = []

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        for img_path in sorted(io_utils.get_images_in_dir(images_dir)):
            latent_path = None
            mask_path = None

            if latent_dir is not None:
                latent_path = latent_dir.joinpath(img_path.stem).with_suffix('.pt')
                if not latent_path.exists():
                    raise ValueError(f'Image {img_path.name} has no latent in {latent_dir}, aborting.')

            if mask_dir is not None:
                mask_path = mask_dir.joinpath(img_path.stem).with_suffix('.jpg')
                if not mask_path.exists():
                    raise ValueError(f'Image {img_path.name} has no mask in {mask_dir}, aborting.')

            sample = ImageReps.load_sample(img_path, latent_path, mask_path, transform)
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def sample_from_P0(anchors: torch.FloatTensor, num_points_to_sample: int, num_anchors_for_sample: int = 3):
    num_anchors = anchors.shape[0]

    scalars = torch.zeros(num_points_to_sample, num_anchors).to(anchors.device)
    tmp_scalars = torch.rand(num_points_to_sample, num_anchors_for_sample).to(anchors.device)
    tmp_scalars = (tmp_scalars / torch.sum(tmp_scalars, axis=-1, keepdims=True))

    indices = np.zeros((num_points_to_sample, num_anchors_for_sample), dtype=int)

    for i in range(num_points_to_sample):
        indices[i, :] = np.random.choice(anchors.shape[0], size=num_anchors_for_sample, replace=False)
        scalars[i, indices[i]] = tmp_scalars[i]

    if anchors.dim() == 4:
        points = torch.einsum('pn, nolk ->  polk', scalars, anchors)
    elif anchors.dim() == 2:
        points = scalars @ anchors
    else:
        raise ValueError('Anchors are in unknown format')

    return points
