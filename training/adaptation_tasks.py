# Copyright 2023 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import abc

import numpy as np
import torch

from expansion_utils import clip_loss

class BaseTask(abc.ABC):
    loss_func = None

    def __init__(self, device):
        self.device = device
        self.const_latent = self.sample_const()

    @abc.abstractmethod
    def sample(self):
        """
        Sample a single latent code and any context reuired by the loss.
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

