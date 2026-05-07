""" Warmup Cosine Scheduler with Step-based Warmup

Cosine LR schedule with warmup measured in STEPS (batches), not epochs.
This is designed for use cases where warmup is specified in number of
optimizer updates rather than number of epochs.
"""
import math
import torch

from .scheduler import Scheduler


class WarmupCosineLRScheduler(Scheduler):
    """
    Cosine decay with step-based warmup.
    
    Unlike the standard CosineLRScheduler which uses warmup_t in epochs,
    this scheduler uses warmup_t in steps (batches/optimizer updates).
    
    The learning rate starts at warmup_lr_init and linearly increases to
    the base LR over warmup_t steps, then follows a cosine decay schedule.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min: float = 0.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 t_in_epochs=True,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        """Get learning rate at step t."""
        if t < self.warmup_t:
            # Linear warmup
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            # Cosine decay
            t_cosine = t - self.warmup_t
            t_max = self.t_initial - self.warmup_t
            lrs = [
                self.lr_min + 0.5 * (v - self.lr_min) * (1 + math.cos(math.pi * t_cosine / t_max))
                for v in self.base_values
            ]
        return lrs

    def get_epoch_values(self, epoch: int):
        """Not used for step-based scheduler."""
        return None

    def get_update_values(self, num_updates: int):
        """Get learning rate at update step num_updates."""
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def step(self, epoch: int, metric=None):
        """Step the scheduler by one epoch.
        
        For step-based warmup, we need to know how many steps per epoch.
        This is handled by calling step_update() after each optimizer step.
        """
        pass

    def step_update(self, num_updates: int, metric=None):
        """Step the scheduler by one update (optimizer step)."""
        lrs = self._get_lr(num_updates)
        super().update_groups(lrs)

    def get_cycle_length(self, cycles=0):
        """Get the cycle length in steps."""
        return self.t_initial
