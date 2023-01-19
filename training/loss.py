# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# ==========================================================================================
#
# Adobe’s modifications are Copyright 2023 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
#
# ==========================================================================================

import copy
import numpy as np
import torch
import dnnlib
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

import lpips
from expansion_utils import latent_operations, consts

# ----------------------------------------------------------------------------


class Loss:
    # to be overridden by subclass
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        raise NotImplementedError()

# ----------------------------------------------------------------------------


class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        raise ValueError('Changes to training_loop.py made this loss unsupported.')
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty(
                        [], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand(
                        [], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(
                        z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # May get synced by Gpl.
                gen_img, _gen_ws = self.run_G(
                    gen_z, gen_c, sync=(sync and not do_Gpl))
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                # -log(sigmoid(gen_logits))
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(
                    gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(
                    gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(
                        gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 +
                 loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                # Gets synced by loss_Dreal.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(
                    gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    # -log(sigmoid(real_logits))
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report(
                        'Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[
                                                       real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal +
                 loss_Dr1).mean().mul(gain).backward()


class DomainExpansionLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, G_synthesis_frozen, D,
                 batch_size, expansion_cfg=None, latent_basis=None, subspace_distance=20,
                 augment_pipe=None, style_mixing_prob=0.9,
                 r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
                 lambda_src=1, lambda_recon_l2=10, lambda_recon_lpips=10, lambda_expand=1
                 ):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.G_synthesis_frozen = G_synthesis_frozen
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.latent_basis = latent_basis.to(device)

        # Domain Expansion additions
        if expansion_cfg is not None:
            self.tasks = self.construct_tasks(copy.deepcopy(expansion_cfg))
            self.repurposed_dims = torch.IntTensor(list(self.tasks.keys()))

        try:
            self.w_avg = self.G_mapping.module.w_avg
            self.num_ws = self.G_mapping.module.num_ws
        except Exception:
            self.w_avg = self.G_mapping.w_avg
            self.num_ws = self.G_mapping.num_ws

        self.batch_size = batch_size
        self.subspace_distance = subspace_distance
        self.lambda_src = lambda_src
        self.lambda_recon_l2 = lambda_recon_l2
        self.lambda_recon_lpips = lambda_recon_lpips
        self.lambda_expand = lambda_expand

        self.mse_loss = torch.nn.MSELoss()
        self.lpips = lpips.LPIPS(net='alex').to(device).eval()

    def construct_tasks(self, expansions):
        tasks = {}

        tasks_spec = expansions['tasks']
        tasks_loss_specs = expansions['tasks_losses']

        w_sampler = WSampler(self.G_mapping, self.device)

        # TODO: let config specify type of sampler
        for task_spec in tasks_spec:
            task_kwargs = task_spec['args']
            task_kwargs['device'] = self.device
            if task_spec['type'] != 'MyStyle':
                task_kwargs['w_sampler'] = w_sampler
            task_kwargs.update(
                tasks_loss_specs[task_spec['type']])

            tasks[task_spec['dimension']] = \
                dnnlib.util.construct_class_by_name(
                    **{'class_name': f'training.adaptation_tasks.{task_spec["type"]}'},
                    **task_kwargs)

        return tasks

    def run_G_from_w(self, G_synthesis, w, sync):
        with misc.ddp_sync(G_synthesis, sync):
            ws = latent_operations.w_to_wplus(w, self.num_ws)
            img = G_synthesis(ws)
        return img

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) <
                     self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['G_adv', 'G_reg', 'G_all', 'D_adv',
                         'D_reg', 'D_all', 'G_expand_and_recon', 'G_expand', 'G_recon']
        do_Gmain = (phase in ['G_adv', 'G_all']) and self.lambda_src > 0
        do_Dmain = (phase in ['D_adv', 'D_all']) and self.lambda_src > 0
        do_Gpl = (phase in ['G_reg', 'G_all']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['D_reg', 'D_all']) and (self.r1_gamma != 0)
        do_Gexpand = (phase in ['G_expand', 'Gall', 'G_expand_and_recon']) and self.lambda_expand > 0
        do_Grecon = (phase in ['G_recon', 'Gall', 'G_expand_and_recon']) and (self.lambda_recon_l2 > 0 or self.lambda_recon_l2 > 0)

        if do_Gexpand or do_Grecon:
            name = 'G_edit_recon' if do_Gexpand and do_Grecon else 'G_edit' if do_Gexpand else 'G_recon'
            with torch.autograd.profiler.record_function(name + '_forward'):
                w, iter_indices, iter_context = self.sample_expansion_data()
                base_w, edit_w = latent_operations.project_to_subspaces(w,
                                                                        self.latent_basis,
                                                                        self.repurposed_dims,
                                                                        step_size=self.subspace_distance,
                                                                        mean=self.w_avg)
                base_imgs = self.run_G_from_w(self.G_synthesis, base_w,
                                              sync=(sync and not do_Gpl))  # May get synced by Gpl.

                loss_G_expand = 0
                if do_Gexpand:
                    all_batch = torch.arange(0, self.batch_size)
                    # Take only latents used in this iter, by index
                    edit_w = edit_w[0, iter_indices, all_batch, :]
                    edit_imgs = self.run_G_from_w(self.G_synthesis, edit_w,
                                                  sync=(sync and not do_Gpl))  # May get synced by Gpl.
                    iter_dims = self.repurposed_dims[iter_indices]
                    loss_funcs = np.array([self.tasks[dim.item()].calc_loss for dim in iter_dims])
                    unique_loss_funcs = list(set(loss_funcs))

                    for loss_func in unique_loss_funcs:
                        batched_indices = all_batch[(loss_funcs == loss_func)]
                        batched_edit_imgs = edit_imgs[batched_indices]
                        batched_base_imgs = base_imgs[batched_indices]
                        batched_context = [iter_context[i] for i in batched_indices]

                        loss_G_expand += loss_func(batched_context,
                                                   batched_base_imgs,
                                                   batched_edit_imgs,
                                                   self.device)

                    # MAKE SURE ALL LOSS FUNCTIONS RETURN SUMS OVER BATCH, SO THEY COULD BE AVERAGED HERE.
                    loss_G_expand = loss_G_expand / self.batch_size
                    training_stats.report('Loss/G/expand_loss', loss_G_expand)

                loss_G_recon = 0
                if do_Grecon:
                    with torch.no_grad():
                        frozen_base_imgs = self.run_G_from_w(self.G_synthesis_frozen, base_w,
                                                            sync=False)  # May get synced by Gpl.

                    loss_G_recon_l2 = self.mse_loss(base_imgs, frozen_base_imgs)
                    loss_G_recon_lpips = self.lpips(base_imgs, frozen_base_imgs).mean()
                    training_stats.report('Loss/G/base_recon_l2', loss_G_recon_l2)
                    training_stats.report('Loss/G/base_recon_lpips', loss_G_recon_lpips)

                    loss_G_recon = self.lambda_recon_l2 * loss_G_recon_l2 + \
                        self.lambda_recon_lpips * loss_G_recon_lpips

            with torch.autograd.profiler.record_function(name + '_backward'):
                (self.lambda_expand * loss_G_expand + loss_G_recon).mean().mul(gain).backward()

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                w = self.G_mapping(gen_z, gen_c)[:, 0]
                if self.lambda_expand > 0:
                    # Apply the original loss (L_src) only on the base subspace.
                    w, _ = latent_operations.project_to_subspaces(w,
                                                                  self.latent_basis,
                                                                  self.repurposed_dims,
                                                                  mean=self.w_avg)

                # May get synced by Gpl.
                gen_img = self.run_G_from_w(self.G_synthesis,
                                            w,
                                            sync=(sync and not do_Gpl))  # May get synced by Gpl.

                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                # -log(sigmoid(gen_logits))
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(
                    gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(
                    gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(
                        gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 +
                 loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                w = self.G_mapping(gen_z, gen_c)[:, 0]
                if self.lambda_expand > 0:
                    # Apply the original loss (L_src) only on the base subspace.
                    w, _ = latent_operations.project_to_subspaces(w,
                                                                  self.latent_basis,
                                                                  self.repurposed_dims,
                                                                  mean=self.w_avg)

                # May get synced by Gpl.
                gen_img = self.run_G_from_w(self.G_synthesis, w, sync=False)

                # Gets synced by loss_Dreal.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(
                    gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    # -log(sigmoid(real_logits))
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report(
                        'Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[
                                                       real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + 
                 loss_Dr1).mean().mul(gain).backward()

    # ----------------

    def sample_expansion_data(self, iter_dims_indices=None):
        """
        Sample data per dimension. 
        Either randomly choose dimensions (training), or use predetermined dimensions (evaluation).
        """

        if iter_dims_indices is None:
            iter_dims_indices = torch.randint(
                len(self.repurposed_dims), (self.batch_size,))

        iter_dims = self.repurposed_dims[iter_dims_indices]
        iter_tasks = [self.tasks[k.item()] for k in iter_dims]

        samples = [task.sample() for task in iter_tasks]

        latents = torch.cat([sample[0] for sample in samples])
        iter_context = [sample[1:] for sample in samples]

        return latents, iter_dims_indices, iter_context

    @torch.no_grad()
    def export_per_direction_images(self):
        all_base_imgs = []
        all_edit_imgs = []

        # TODO: project only once.
        for idx, dim in enumerate(self.repurposed_dims):
            task = self.tasks[dim.item()]
            grid_w = task.const_latent
            base_w, edit_w = latent_operations.project_to_subspaces(grid_w,
                                                                    self.latent_basis,
                                                                    self.repurposed_dims,
                                                                    step_size=self.subspace_distance,
                                                                    mean=self.w_avg)

            edit_w = edit_w[0, idx]  # Take traversal only along current dim

            base_w = latent_operations.w_to_wplus(base_w, self.num_ws).split(self.batch_size)
            edit_w = latent_operations.w_to_wplus(edit_w, self.num_ws).split(self.batch_size)

            dim_base_imgs = torch.cat([self.G_synthesis(w, noise_mode='const').cpu() for w in base_w])
            dim_edit_imgs = torch.cat([self.G_synthesis(w, noise_mode='const').cpu() for w in edit_w])

            all_base_imgs.append(dim_base_imgs)
            all_edit_imgs.append(dim_edit_imgs)

        all_base_imgs = torch.stack(all_base_imgs)
        all_edit_imgs = torch.stack(all_edit_imgs)
    
        return all_base_imgs, all_edit_imgs, [f'Dimension {x}' for x in self.repurposed_dims]

    @torch.no_grad()
    def calc_per_dim_error(self, G_synthesis):
        per_dim_errors = {}

        if self.tasks is None or self.lambda_expand == 0:
            return per_dim_errors

        all_batch = torch.arange(0, self.batch_size)
        for dim_idx, latent_dim in enumerate(self.repurposed_dims):
            latent_dim = latent_dim.item()
            latent_dim = self.repurposed_dims[dim_idx].item()
            task = self.tasks[latent_dim]
            name = f'{latent_dim}: {task}'
            per_dim_errors[name] = []
            iter_dim_indices = torch.full((self.batch_size,), dim_idx)

            for i in range(consts.EVAL_SIZE // self.batch_size):
                w, _, context = self.sample_expansion_data(
                    iter_dims_indices=iter_dim_indices)
                base_w, edit_w = latent_operations.project_to_subspaces(w,
                                                                        self.latent_basis,
                                                                        self.repurposed_dims,
                                                                        step_size=self.subspace_distance,
                                                                        mean=self.w_avg)

                edit_w = edit_w[0, iter_dim_indices, all_batch, :]
                base_imgs = self.run_G_from_w(G_synthesis, base_w, sync=False)
                edit_imgs = self.run_G_from_w(G_synthesis, edit_w, sync=False)
                loss_G_expand = task.calc_loss(
                    context, base_imgs, edit_imgs, self.device) / self.batch_size

                per_dim_errors[name].append(loss_G_expand)

            per_dim_errors[name] = torch.mean(
                torch.tensor(per_dim_errors[name])).item()

        return per_dim_errors


# ----------------------------------------------------------------------------

class WSampler:
    def __init__(self, G_mapping, device):
        self.G_mapping = G_mapping
        self.device = device
        self.const_latent = None

    def sample(self):
        try:
            z_dim = self.G_mapping.module.z_dim
            c_dim = self.G_mapping.module.c_dim
        except:
            z_dim = self.G_mapping.z_dim
            c_dim = self.G_mapping.c_dim

        z = torch.randn([1, z_dim], device=self.device)

        c = torch.zeros((1, c_dim)).to(self.device) if c_dim > 0 else None
        with torch.no_grad():
            w = self.G_mapping(z, c)[:, 0, :]

        return w

    def sample_const(self, size=9):
        if self.const_latent is None:
            self.const_latent = torch.cat([self.sample() for _ in range(size)])

        return self.const_latent
