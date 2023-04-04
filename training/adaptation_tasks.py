# Copyright 2023 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import abc
import random
from pathlib import Path

import lpips
import numpy as np
import torch

from expansion_utils import clip_loss, latent_operations
from expansion_utils.mystyle_data_utils import PersonalizedDataset


class BaseTask(abc.ABC):
    loss_func = None

    def __init__(self, device):
        self.device = device
        self.const_latent = self.sample_const()

    @abc.abstractmethod
    def sample(self):
        """
        Sample a single latent code and any context required by the loss.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_const(self, size=9):
        """
        Sample a set of constant latent codes. Used to track progress of training.
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def calc_loss(context, base_output, edit_output, device):
        """
        Get the sampled context as well as generator output on the base and repurposed subspaces. 
        Calculates the loss. Note that this function must be static as it must be shared for all tasks
        applying the same method.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError()


class NADA(BaseTask):
    clip_models = None
    clip_model_weights = None

    def __init__(self, device, source_text, target_text, w_sampler, clip_models, clip_model_weights):
        self.source_text = source_text
        self.target_text = target_text
        self.w_sampler = w_sampler

        if NADA.clip_models is None and NADA.clip_model_weights is None:
            NADA.clip_models = {
                model_name: clip_loss.CLIPLoss(device, clip_model=model_name)
                for model_name in clip_models
            }
            NADA.clip_model_weights = {
                model_name: weight
                for model_name, weight in zip(clip_models, clip_model_weights)
            }

        super(NADA, self).__init__(device)
        # TODO: check if w_sample_func works with ddp. It needs to get a replica of G.mapping for this GPU.

    def sample(self):
        return self.w_sampler.sample(), self.source_text, self.target_text

    def sample_const(self, size=9):
        return self.w_sampler.sample_const(size)

    def __str__(self):
        return f'{self.source_text} -> {self.target_text}'

    @staticmethod
    def batch(context, base_output, edit_output):
        source_classes = np.array([x[0] for x in context])
        target_classes = np.array([x[1] for x in context])
        return base_output, edit_output, source_classes, target_classes

    @staticmethod
    def calc_loss(context, base_output, edit_output, device):
        ref_img, gen_img, source_classes, target_classes = NADA.batch(context, base_output, edit_output)

        batch_size = gen_img.shape[0]
        # Use the single prompt to all batch samples
        if source_classes.shape[0] == 1:
            source_classes = source_classes.repeat(batch_size)
        if target_classes.shape[0] == 1:
            target_classes = target_classes.repeat(batch_size)

        clip_loss = torch.sum(
            torch.stack(
                [torch.ones((1,)).to(device) * NADA.clip_model_weights[model_name] *
                 NADA.clip_models[model_name](
                     ref_img.detach(), source_classes, gen_img, target_classes)
                 for model_name in NADA.clip_model_weights.keys()]
            )
        )

        return clip_loss


class MyStyle(BaseTask):
    lpips_net = None
    l2_weight = None

    def __init__(self, device, images_dir, anchors_dir, person_name, l2_weight):
        self.dataset = PersonalizedDataset(Path(images_dir), Path(anchors_dir))
        self.set_size = len(self.dataset)
        self.person_name = person_name

        if MyStyle.lpips_net is None:
            MyStyle.lpips_net = lpips.LPIPS(net='alex').to(device).eval()

        if MyStyle.l2_weight is None:
            MyStyle.l2_weight = l2_weight

        super(MyStyle, self).__init__(device)

    def sample(self):
        idx = random.randrange(0, self.set_size)
        sample = self.dataset[idx]
        return latent_operations.wplus_to_w(sample.w_code.unsqueeze(0)).to(self.device), sample.img.to(self.device)

    def sample_const(self, size=9):
        indices = random.sample(range(0, self.set_size), size)
        ws = [latent_operations.wplus_to_w(self.dataset[idx].w_code.unsqueeze(0)) for idx in indices]
        return torch.cat(ws).to(self.device)

    def __str__(self):
        return f'{self.person_name}'

    @staticmethod
    def batch(context, edit_output):
        context = torch.cat([gt[0] for gt in context])

        return context, edit_output

    @staticmethod
    def calc_loss(context, base_output, edit_output, device):
        gt_img, edit_output = MyStyle.batch(context, edit_output)

        lpips_loss = MyStyle.lpips_net(edit_output, gt_img).sum(dim=0).mean()
        l2_loss = (edit_output - gt_img).square().sum(dim=0).mean()

        loss = lpips_loss + MyStyle.l2_weight * l2_loss

        return loss
