"""
Exponential Moving Average (EMA) implementation for model weights.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict


class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights that updates slowly toward current weights.
    At evaluation time, EMA weights are typically less noisy and generalize better.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9995,
        warmup_steps: int = 1000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize EMA.

        Args:
            model: The model whose weights to track
            decay: EMA decay factor (closer to 1 = slower updates)
            warmup_steps: Number of steps to ramp up decay from 0.99
            device: Device to store shadow weights on
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.steps = 0

        # Create shadow copy of floating point parameters
        self.shadow = {}
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                # Always keep EMA weights in fp32 for stability
                self.shadow[name] = param.detach().clone().to(
                    device or param.device
                ).float()

    def get_decay(self) -> float:
        """Get current decay value with warmup."""
        if self.warmup_steps > 0 and self.steps < self.warmup_steps:
            # Ramp from 0.99 to target decay
            warmup_decay = 0.99
            progress = self.steps / self.warmup_steps
            return warmup_decay + (self.decay - warmup_decay) * progress
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update shadow weights with current model weights.
        Call this after each optimizer.step().

        Args:
            model: The model with updated weights
        """
        self.steps += 1
        decay = self.get_decay()

        for name, param in model.state_dict().items():
            if name in self.shadow and param.dtype.is_floating_point:
                # EMA update: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(decay).add_(param, alpha=1 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        Copy shadow weights to model.
        Use this before evaluation to use EMA weights.

        Args:
            model: The model to copy weights to
        """
        model_state = model.state_dict()
        for name, shadow_param in self.shadow.items():
            if name in model_state:
                # Convert back to model's dtype if needed
                target_dtype = model_state[name].dtype
                model_state[name].copy_(shadow_param.to(target_dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """
        Restore original model weights (before copy_to was called).
        Note: This requires saving the original weights first.
        """
        # This is a placeholder - in practice you'd save originals before copy_to
        pass

    def state_dict(self) -> Dict:
        """Get state dict for saving."""
        return {
            'shadow': self.shadow,
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'steps': self.steps
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']
        self.warmup_steps = state_dict['warmup_steps']
        self.steps = state_dict['steps']


class ModelWithEMA:
    """
    Wrapper to manage model with EMA weights for evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        ema: EMA
    ):
        self.model = model
        self.ema = ema
        self.original_state = None

    def __enter__(self):
        """Enter context: switch to EMA weights."""
        # Save original weights
        self.original_state = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        # Apply EMA weights
        self.ema.copy_to(self.model)
        return self.model

    def __exit__(self, *args):
        """Exit context: restore original weights."""
        if self.original_state is not None:
            self.model.load_state_dict(self.original_state)
            self.original_state = None