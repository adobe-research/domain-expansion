# Copyright 2023 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import torch


def project_to_subspaces(latent: torch.Tensor, basis: torch.Tensor,
                         repurposed_dims: torch.Tensor, base_dims: torch.Tensor = None,
                         step_size=None, mean=None):
    """
    Project latent on the base subspace (Z_base) - spanned by the base_dims.
    Then, traverses the projected latent along the repurposed directions.
    If the step_size parameter can be interpreted as some 1D structure, 
    then traversal is performed separately for each repurposed dim with these as step sizes.
    Otherwise, it defines a joint traversal of multiple dimensions at once. 
    Usually, it would be 3D, so the output can be visualized in a 2D image grid.

    Returns:
        traversals.shape, 1D case -[num_steps, num_repurposed, shape of input]
        traversals.shape, ND case -[num_steps_1, ..., num_steps_N, shape of input]
    """

    if type(latent) == list:
        if len(latent) != 1:
            raise ValueError('Latent wrapped by list should be of length 1')
        latent = latent[0]

    latent_in_w = False
    if latent.dim() == 2:
        # Lift to W+ just for now
        latent = w_to_wplus(latent)
        latent_in_w = True
    elif latent.dim() != 3:
        raise ValueError('Latent is expected to be 2D (W space) or 3D (W+ space)')

    latent_dim = latent.shape[-1]

    if base_dims is None:
        # Take all non-repurposed dims to span the base subspace -- default mode
        base_dims = torch.Tensor([x for x in range(latent_dim) if x not in repurposed_dims])

    # Use values instead of boolean to change order as needed
    repurposed_directions = basis[:, repurposed_dims.numpy()]
    base_directions = basis[:, base_dims.numpy()]

    projected_latent = latent @ base_directions
    base_latent = projected_latent @ base_directions.T

    if mean is not None:
        base_latent += (mean @ repurposed_directions) @ repurposed_directions.T

    if step_size is None:
        if latent_in_w:
            base_latent = wplus_to_w(base_latent)
        return base_latent, None

    if isinstance(step_size, float) or isinstance(step_size, int):
        step_size = torch.Tensor([step_size]).to(latent.device)

    repurposed_directions = repurposed_directions.T

    num_repurposed = len(repurposed_dims)

    if step_size.dim() == 1:
        # separate same-sized steps on all dims
        num_steps = step_size.shape[0]
        output_shape = [num_steps, num_repurposed, *latent.shape]
        edits = torch.einsum('a, df -> adf', step_size, repurposed_directions)
    elif step_size.dim() == 3:
        # compound steps, on multiple dims
        steps_in_directions = step_size.shape[:-1]
        output_shape = [*steps_in_directions, *latent.shape]

        edits = step_size @ repurposed_directions
    else:
        raise NotImplementedError('Cannot edit with these values')

    edit_latents = base_latent.expand(output_shape) + edits.unsqueeze(2).unsqueeze(2).expand(output_shape)

    if latent_in_w:
        # Bring back to W sapce
        base_latent, edit_latents = wplus_to_w(base_latent), edit_latents[..., 0, :]

    return base_latent, edit_latents


def w_to_wplus(w_latent: torch.Tensor, num_ws=18):
    if w_latent.dim() == 2:
        w_latent.unsqueeze_(1)
    elif w_latent.dim() != 3:
        raise ValueError(f'Input is of unexpected shape {w_latent.shape}')

    return w_latent.repeat([1, num_ws, 1])


def wplus_to_w(latents: torch.Tensor):
    """
    latents is expected to have shape (...,num_ws,512) or 
    """
    with torch.no_grad():
        _, counts = torch.unique(latents, dim=-2, return_counts=True)
    if len(counts) != 1:
        raise ValueError('input latent is not a W code, conversion from W+ is undefined')

    return latents[..., 0, :]
