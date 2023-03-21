# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the audio adversarial attack on automatic speech recognition systems of Carlini and Wagner
(2018). It generates an adversarial audio example.
| Paper link: https://arxiv.org/abs/1801.01944
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from art.attacks.attack import EvasionAttack
from art.estimators.speech_recognition.wav2vec2ModelWrapper import wav2vec2Model
from art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
import torch.optim as optim
import torch
import torch.nn as nn 
import numpy as np

if TYPE_CHECKING:
    from art.utils import SPEECH_RECOGNIZER_TYPE

logger = logging.getLogger(__name__)


class CarliniWagnerASR(ImperceptibleASR):
    """
    Implementation of the Carlini and Wagner audio adversarial attack against a speech recognition model.
    | Paper link: https://arxiv.org/abs/1801.01944
    """

    attack_params = EvasionAttack.attack_params + [
        "eps",
        "learning_rate",
        "max_iter",
        "batch_size",
        "decrease_factor_eps",
        "num_iter_decrease_eps",
    ]

    def __init__(
        self,
        estimator: wav2vec2Model,
        eps: float = 2000.0,
        max_iter: int = 1000,
        learning_rate: float = 100.0,

        optimizer: Optional["torch.optim.Optimizer"] = None,
        global_max_length: int = 20000,
        initial_rescale: float = 1.0,

        
        decrease_factor_eps: float = 0.8,
        num_iter_decrease_eps: int = 10,
        max_num_decrease_eps: Optional[int] = None,

        targeted: bool = True,
        train_mode_for_backward: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        const: float = 1.0,

        batch_size: int = 1,
    ):
        """
        Create an instance of the :class:`.CarliniWagnerASR`.
        :param estimator: A trained speech recognition estimator.
        :param eps: Initial max norm bound for adversarial perturbation.
        :param learning_rate: Learning rate of attack.
        :param max_iter: Number of iterations.
        :param decrease_factor_eps: Decrease factor for epsilon (Paper default: 0.8).
        :param num_iter_decrease_eps: Iterations after which to decrease epsilon if attack succeeds (Paper default: 10).
        :param batch_size: Batch size.
        """
        # pylint: disable=W0231

        # re-implement init such that inherited methods work
        EvasionAttack.__init__(self, estimator = estimator)

        super(CarliniWagnerASR, self).__init__(
        estimator = estimator,
        eps = eps,
        max_iter_1 = max_iter,
        max_iter_2 = 0,
        learning_rate_1 = learning_rate,
        optimizer_1 = optimizer_1,
        
        targeted = True,
        global_max_length = global_max_length,
        initial_rescale = initial_rescale,
        decrease_factor_eps = decrease_factor_eps,
        num_iter_decrease_eps = num_iter_decrease_eps,
        max_num_decrease_eps = max_num_decrease_eps,
        targeted = targeted,
        train_mode_for_backward = train_mode_for_backward,
        clip_min = clip_min,
        clip_max = clip_max,
        )
        self.rec_const = 1./const if const is not None else 0.

    def _forward_1st_stage(
        self,
        original_input: np.ndarray,
        original_output: np.ndarray,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        real_lengths: np.ndarray,
    ) -> Tuple["torch.Tensor", "torch.Tensor", np.ndarray, "torch.Tensor", "torch.Tensor"]:

        import torch

        # Compute perturbed inputs
        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(self.estimator.device)
        local_delta_rescale *= torch.tensor(rescale).to(self.estimator.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(self.estimator.device)
        masked_adv_input = adv_input * torch.tensor(input_mask).to(self.estimator.device)

        # Compute loss and decoded output
        loss, decoded_output = self.estimator.compute_loss_and_decoded_output(
            masked_adv_input=masked_adv_input,
            original_output=original_output,
            real_lengths=real_lengths,
        )

        loss = loss + self.reg_const * torch.norm(local_delta_rescale)
        self.estimator.eval()
        return loss, local_delta, decoded_output, masked_adv_input, local_delta_rescale