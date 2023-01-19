# Based on a script from https://github.com/rosinality/stylegan2-pytorch

# ==========================================================================================
#
# Adobe’s modifications are Copyright 2023 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
#
# ==========================================================================================


import argparse
import numpy as np
import torch
from pathlib import Path

import dnnlib

import legacy


def factorize(G):
    modulate = {
        k: v
        for k, v in G.named_parameters()
        if ('b4' in k or "torgb" not in k) and ("affine" in k and "weight" in k)
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V

    return eigvec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument("--out", type=str, requited=True, help="path to output file")
    parser.add_argument("ckpt", type=str, help="name of the model checkpoint")

    args = parser.parse_args()
    device = 'cuda'
    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    eigvec = factorize(G)
    torch.save(eigvec, args.out)
