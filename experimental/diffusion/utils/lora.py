"""
LoRA (Low-Rank Adaptation) utilities for efficient fine-tuning.

This implements LoRA as described in "LoRA: Low-Rank Adaptation of Large Language Models"
(Hu et al., 2021). LoRA freezes pretrained weights and injects trainable low-rank
decomposition matrices into each layer.
"""
import torch
import torch.nn as nn
from typing import Dict, List


class LoRALinear(nn.Linear):
    """
    Linear layer with LoRA adaptation.

    Instead of fine-tuning W, we freeze W and train:
        h = Wx + (BA)x
    where B ∈ R^{d×r}, A ∈ R^{r×k}, and r << min(d,k)

    During inference, we can merge: W' = W + αBA where α = lora_alpha / r
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        # Initialize the base Linear layer
        super().__init__(in_features, out_features, bias=bias, **kwargs)

        self.rank = rank
        self.lora_alpha = lora_alpha

        # Freeze the base weights
        self.weight.requires_grad = False
        if bias and self.bias is not None:
            self.bias.requires_grad = False

        # LoRA matrices
        # A: project down to rank r
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        # B: project back up to out_features
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Scaling factor
        self.scaling = lora_alpha / rank

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Initialize A with kaiming uniform, B with zeros
        # This ensures BA = 0 at initialization (no change to pretrained weights)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Track if weights are merged
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Base output from frozen weights
        result = super().forward(x)

        # Add LoRA adaptation if not merged
        if not self.merged:
            # Compute low-rank update: x @ A^T @ B^T
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_out * self.scaling

        return result

    def merge(self):
        """Merge LoRA weights into base weights for deployment."""
        if not self.merged:
            # W' = W + α * B @ A
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from base weights."""
        if self.merged:
            # W = W' - α * B @ A
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False


def replace_linear_with_lora(
    model: nn.Module,
    rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: List[str] = None
) -> Dict[str, nn.Module]:
    """
    Replace Linear layers in a model with LoRA-adapted versions.

    Args:
        model: The model to modify
        rank: LoRA rank (r)
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module name patterns to target (e.g., ["q_proj", "v_proj"])
                       If None, targets all Linear layers

    Returns:
        Dictionary mapping module names to replaced modules
    """
    replaced = {}

    # First pass: collect all modules to replace
    modules_to_replace = []
    for name, module in model.named_modules():
        # Skip if not a Linear layer
        if not isinstance(module, nn.Linear):
            continue

        # Check if this module should be replaced
        if target_modules is not None:
            # Check if any target pattern matches this module name
            if not any(target in name for target in target_modules):
                continue

        modules_to_replace.append((name, module))

    # Second pass: actually replace them
    for name, module in modules_to_replace:
        # Get parent module and attribute name
        *parent_names, attr_name = name.split('.')
        parent = model
        for parent_name in parent_names:
            parent = getattr(parent, parent_name)

        # Create LoRA replacement
        lora_layer = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=module.bias is not None
        )

        # Copy original weights (LoRALinear inherits from nn.Linear, so weight/bias are direct attributes)
        lora_layer.weight.data = module.weight.data.clone()
        if module.bias is not None:
            lora_layer.bias.data = module.bias.data.clone()

        # Replace module
        setattr(parent, attr_name, lora_layer)
        replaced[name] = lora_layer

    return replaced


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from a model."""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def mark_only_lora_as_trainable(model: nn.Module) -> int:
    """
    Freeze all parameters except LoRA parameters.

    Returns:
        Number of trainable parameters
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze LoRA parameters
    trainable_params = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            trainable_params += module.lora_A.numel() + module.lora_B.numel()

    return trainable_params


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into base weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights from base weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def print_lora_info(model: nn.Module):
    """Print information about LoRA modules in the model."""
    lora_modules = []
    total_lora_params = 0
    total_frozen_params = 0

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params = module.lora_A.numel() + module.lora_B.numel()
            frozen_params = module.weight.numel()
            if module.bias is not None:
                frozen_params += module.bias.numel()

            lora_modules.append({
                'name': name,
                'lora_params': lora_params,
                'frozen_params': frozen_params,
                'rank': module.rank
            })
            total_lora_params += lora_params
            total_frozen_params += frozen_params

    print(f"\n🔧 LoRA Configuration:")
    print(f"  LoRA modules: {len(lora_modules)}")
    print(f"  Trainable LoRA parameters: {total_lora_params:,}")
    print(f"  Frozen parameters in LoRA layers: {total_frozen_params:,}")
    print(f"  Parameter reduction: {100 * (1 - total_lora_params / (total_lora_params + total_frozen_params)):.1f}%")

    if len(lora_modules) > 0:
        print(f"\n  Sample LoRA modules:")
        for info in lora_modules[:5]:
            print(f"    {info['name']}: rank={info['rank']}, lora_params={info['lora_params']:,}")
        if len(lora_modules) > 5:
            print(f"    ... and {len(lora_modules) - 5} more")
