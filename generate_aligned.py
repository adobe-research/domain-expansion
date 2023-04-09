# Copyright 2023 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import sys

sys.path.append('..')

import argparse
from pathlib import Path

import torch

import dnnlib
import legacy
from expansion_utils import io_utils, latent_operations


def generate_images(
        ckpt,
        out_dir,
        num_samples,
        truncation_psi
):
    device = torch.device('cuda')
    with dnnlib.util.open_url(ckpt) as f:
        snapshot_dict = legacy.load_network_pkl(f)
        G = snapshot_dict['G_ema'].to(device)
    latent_basis = snapshot_dict['latent_basis'].to(device)
    subspace_distance = snapshot_dict['subspace_distance']
    repurposed_dims = snapshot_dict['repurposed_dims'].cpu()

    out_dir = Path(out_dir)

    for i in range(num_samples):
        z = torch.randn((1, G.z_dim)).to(device)
        w = G.mapping(z, None, truncation_psi=truncation_psi)
        base_w, edit_ws = latent_operations.project_to_subspaces(w, latent_basis, repurposed_dims,
                                                                 step_size=subspace_distance, mean=G.mapping.w_avg)
        edit_ws = edit_ws[0]  # Single step
        base_img = G.synthesis(base_w, noise_mode='const')
        io_utils.save_images(base_img, out_dir.joinpath('base', f'{i:05d}'))

        for dim_num, edit_w in zip(repurposed_dims, edit_ws):
            dim_out_dir = out_dir.joinpath(f'dim_{dim_num}')

            edit_img = G.synthesis(edit_w, noise_mode='const')
            io_utils.save_images(edit_img, dim_out_dir.joinpath(f'{i:05d}'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', help='Network pickle filename', required=True)
    parser.add_argument('--out_dir', help='Where to save the output images', type=str, required=True, metavar='DIR')
    parser.add_argument('--num', help='Number of independent samples', type=int)
    parser.add_argument('--truncation_psi', help='Coefficient for truncation', type=float, default=1)

    args = parser.parse_args()

    with torch.no_grad():
        generate_images(args.ckpt, args.out_dir, args.num, args.truncation_psi)
