# Copyright 2023 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import os
import sys

sys.path.append('..')

from pathlib import Path
from argparse import ArgumentParser

import dnnlib
import legacy
from expansion_utils import mystyle_data_utils, io_utils

from torchvision.utils import save_image
import torch
import numpy as np

torch.manual_seed(2)
np.random.seed(2)


def synthesize(anchors, generator, output_path, num_points_to_sample):
    latents = mystyle_data_utils.sample_from_P0(anchors, num_points_to_sample).to('cuda')
    batch_size = 5
    i = 0
    while i < latents.shape[0]:
        lats = latents[i: i + batch_size]
        imgs = generator.synthesis(lats.squeeze(1), noise_mode='const')

        for j in range(min(batch_size, num_points_to_sample - i)):
            save_image(imgs[j], output_path.joinpath(f'{i + j}.jpg'), nrow=1, normalize=True, range=(-1, 1))

        del imgs
        i += batch_size


def parse_args(raw_args):
    parser = ArgumentParser()
    parser.add_argument('--anchors_path', required=True, type=Path)
    parser.add_argument('--ckpt', help='Network pickle filename', required=True, type=Path)
    parser.add_argument('--output_path', required=True, type=Path)

    parser.add_argument('--device', default='0')
    parser.add_argument('--num', help='Number of independent samples', type=int, default=10)

    parser.add_argument('--repurposed_dim', type=int, required=True)

    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    args = parse_args(raw_args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.output_path.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda')

    with dnnlib.util.open_url(str(args.ckpt)) as f:
        snapshot_dict = legacy.load_network_pkl(f)
        generator = snapshot_dict['G_ema'].to(device)
    latent_basis = snapshot_dict['latent_basis'].to(device)
    subspace_distance = snapshot_dict['subspace_distance']
    repurposed_dims = snapshot_dict['repurposed_dims'].cpu()

    if args.repurposed_dim not in repurposed_dims:
        print('WARNING: Chosen dimension was not repurposed.')

    anchors = io_utils.load_latents(args.anchors_path)
    personalized_vec = latent_basis[:, args.repurposed_dim]

    # Transport the anchors to the repurposed subspace
    anchors += subspace_distance * personalized_vec.expand(anchors.shape)

    synthesize(anchors, generator, args.output_path, args.num)


if __name__ == '__main__':
    with torch.no_grad():
        main()
