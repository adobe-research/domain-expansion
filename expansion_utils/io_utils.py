# Copyright 2023 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
from PIL import Image, ImageDraw, ImageFont
from expansion_utils import consts


def max_num_not_in_list(max_num, lst):
    for i in range(max_num, 0, -1):
        if i not in lst:
            return i


def process_config(expansion_cfg_path):
    with Path(expansion_cfg_path).open() as fp:
        expansion_cfg = json.load(fp)
    assert ("tasks" in expansion_cfg.keys()) and ("tasks_losses" in expansion_cfg.keys())

    curr_max_dim = consts.LATENT_DIM - 1

    used_dims = [x.get("dimension") for x in expansion_cfg["tasks"]]

    for dim in used_dims:
        if dim is not None and used_dims.count(dim) > 1:
            raise ValueError(f"Config tries to repurpose the same dim {dim} more than once, unsupported...")

    for task in expansion_cfg["tasks"]:
        if task.get("dimension") is None:
            curr_max_dim = max_num_not_in_list(curr_max_dim, used_dims)

            if curr_max_dim is None:
                raise ValueError("No available dimension was found")

            task["dimension"] = curr_max_dim
            used_dims.append(curr_max_dim)

    print(f"Parsed config successfuly! Repurposing {len(used_dims)} dims, good luck!")

    return expansion_cfg


def label_image(img: torch.Tensor, label: str = None):
    batch_size = img.shape[0]
    img = torchvision.utils.make_grid(img, batch_size) # concat over W
    if label is not None:
        H, W = img.shape[-2:]
        W = W // (4 * batch_size) 
        H, W = W, H # will be rotated, so H is W
        font = ImageFont.truetype("DejaVuSans.ttf", 60) # TODO: use different font sizes for different resolutions.
        label_img = Image.new('RGB', (W ,H), color='white') 
        draw = ImageDraw.Draw(label_img)
        w, h = draw.textsize(label, font=font)
        draw.text(((W - w) / 2, (H - h) / 2), label, font=font, fill=(0, 0, 0))
        label_img = torchvision.transforms.functional.pil_to_tensor(label_img.rotate(90, expand=True))
        label_img = label_img.to(torch.float32) / 127.5 - 1
        img = torch.cat([label_img, img], dim=-1)
    return img

def save_batched_images(images: torch.Tensor, output_path: Path, labels: list = None, max_row_in_img=5):
    num_rows = images.shape[0]
    
    if labels is not None:
        if num_rows != len(labels):
            raise ValueError('Number of labels should match number of batches')
    else:
        labels = [None] * num_rows

    images = [label_image(image, label) for image, label in zip(images, labels)]
    images = torch.stack(images)

    batched_iter = DataLoader(images, batch_size=max_row_in_img)
    for batch_idx, images_slice in enumerate(batched_iter):
        save_images(
            images_slice,
            output_path.with_name(f"{output_path.stem}_batch_{batch_idx}"),
            1,
        )


def save_images(frames: torch.Tensor, output_path: Path, nrow=None, size=None, separate=False):
    parent_dir = output_path.parent
    parent_dir.mkdir(exist_ok=True, parents=True)

    if size:
        frames = torch.nn.functional.interpolate(frames, size)

    if separate:
        base_name = output_path.stem
        for i, frame in enumerate(frames):
            torchvision.utils.save_image(
                frame,
                output_path.with_name(f"{i:05d}_{base_name}.jpg"),
                nrow=len(frame),
                normalize=True,
                range=(-1, 1),
            )
    else:
        torchvision.utils.save_image(
            frames,
            output_path.with_suffix(".jpg"),
            nrow=nrow if nrow else len(frames),
            normalize=True,
            range=(-1, 1),
        )

