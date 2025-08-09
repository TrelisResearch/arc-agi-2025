"""
CLI: Merge two PEFT LoRA adapters (linear or TIES), save the merged adapter, and
optionally bake into full base weights. Designed to produce a single merged
adapter you can later load into Unsloth.

Recommended workflow:
1) Merge in vanilla Transformers+PEFT using this script.
2) Load the merged adapter (or baked model) with Unsloth for training/serving.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two LoRA adapters with PEFT (linear or TIES).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Base model
    parser.add_argument("--base-id", type=str, default="Qwen/Qwen3-4B", help="Base model repo or path")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"], help="Torch dtype for loading base")
    parser.add_argument("--device-map", type=str, default="auto", help="transformers device_map")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false", help="Disable trust_remote_code when loading base")
    parser.set_defaults(trust_remote_code=True)

    # Adapter 1
    parser.add_argument("--a1-repo", type=str, required=True, help="Adapter 1 repo or path")
    parser.add_argument("--a1-sub", type=str, default=None, help="Adapter 1 subfolder (e.g., checkpoint-XXXX)")
    parser.add_argument("--a1-name", type=str, default="adapter1", help="Adapter 1 logical name")

    # Adapter 2
    parser.add_argument("--a2-repo", type=str, required=True, help="Adapter 2 repo or path")
    parser.add_argument("--a2-sub", type=str, default=None, help="Adapter 2 subfolder (e.g., checkpoint-YYYY)")
    parser.add_argument("--a2-name", type=str, default="adapter2", help="Adapter 2 logical name")

    # Merge config
    parser.add_argument("--method", type=str, default="ties", choices=["linear", "ties"], help="Merge combination type")
    parser.add_argument("--weights", type=float, nargs=2, default=[1.0, 1.0], help="Weights for the two adapters")
    parser.add_argument("--density", type=float, default=0.3, help="Density for TIES (fraction kept, 0<d<=1)")
    parser.add_argument("--sign", type=str, default="total", choices=["frequency", "total"], help="TIES majority sign method")
    parser.add_argument("--merged-name", type=str, default=None, help="Logical adapter name for the merged adapter")

    # Outputs
    parser.add_argument("--out", type=str, default=None, help="Directory to save merged adapter (LoRA files)")
    parser.add_argument("--bake-dir", type=str, default=None, help="Directory to save baked full model (optional)")

    # Hub (optional)
    parser.add_argument("--push-adapter-repo", type=str, default=None, help="HF repo to push merged adapter to (requires token)")
    parser.add_argument("--push-baked-repo", type=str, default=None, help="HF repo to push baked model to (requires token)")
    parser.add_argument("--private", action="store_true", help="Push created repos as private")

    return parser.parse_args()


def str_to_dtype(dtype_str: str) -> torch.dtype | str:
    if dtype_str == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def ensure_dir(path: str | None) -> None:
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


def load_base(base_id: str, torch_dtype: torch.dtype | str, device_map: str, trust_remote_code: bool) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )


def attach_two_adapters(
    base_model: AutoModelForCausalLM,
    a1_repo: str,
    a1_sub: str | None,
    a1_name: str,
    a2_repo: str,
    a2_sub: str | None,
    a2_name: str,
) -> PeftModel:
    model = PeftModel.from_pretrained(
        base_model,
        model_id=a1_repo,
        subfolder=a1_sub,
        adapter_name=a1_name,
    )
    model.load_adapter(
        model_id=a2_repo,
        subfolder=a2_sub,
        adapter_name=a2_name,
    )
    return model


def add_merged_adapter(
    model: PeftModel,
    adapters: List[str],
    weights: List[float],
    merged_name: str,
    method: str,
    density: float,
    majority_sign_method: str,
) -> None:
    kwargs = dict(
        adapters=adapters,
        weights=weights,
        adapter_name=merged_name,
        combination_type=method,
    )
    if method == "ties":
        kwargs.update(density=density, majority_sign_method=majority_sign_method)
    model.add_weighted_adapter(**kwargs)
    model.set_adapter(merged_name)


def save_merged_adapter(model: PeftModel, merged_name: str, out_dir: str | None) -> None:
    if out_dir is None:
        return
    ensure_dir(out_dir)
    model.save_pretrained(out_dir, selected_adapters=[merged_name])
    print(f"Saved merged adapter to: {out_dir}")


def bake_and_save(model: PeftModel, bake_dir: str | None) -> None:
    if bake_dir is None:
        return
    ensure_dir(bake_dir)
    baked = model.merge_and_unload()
    baked.save_pretrained(bake_dir)
    print(f"Saved baked full model to: {bake_dir}")


def maybe_push_adapter(model: PeftModel, repo_id: str | None, private: bool) -> None:
    if not repo_id:
        return
    model.push_to_hub(repo_id, private=private)
    print(f"Pushed merged adapter to: {repo_id}")


def maybe_push_baked(bake_dir: str | None, base_id: str, repo_id: str | None, private: bool) -> None:
    if not repo_id or not bake_dir:
        return
    baked = AutoModelForCausalLM.from_pretrained(bake_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    baked.push_to_hub(repo_id, private=private)
    tok.push_to_hub(repo_id, private=private)
    print(f"Pushed baked full model and tokenizer to: {repo_id}")


def main() -> None:
    args = parse_args()

    merged_name = args.merged_name
    if not merged_name:
        merged_name = f"{args.a1_name}__{args.a2_name}_{args.method}"

    out_dir = args.out or os.path.join("adapters", f"merged_{merged_name}")

    torch_dtype = str_to_dtype(args.dtype)
    base = load_base(args.base_id, torch_dtype=torch_dtype, device_map=args.device_map, trust_remote_code=args.trust_remote_code)

    model = attach_two_adapters(
        base_model=base,
        a1_repo=args.a1_repo,
        a1_sub=args.a1_sub,
        a1_name=args.a1_name,
        a2_repo=args.a2_repo,
        a2_sub=args.a2_sub,
        a2_name=args.a2_name,
    )

    add_merged_adapter(
        model=model,
        adapters=[args.a1_name, args.a2_name],
        weights=args.weights,
        merged_name=merged_name,
        method=args.method,
        density=args.density,
        majority_sign_method=args.sign,
    )

    save_merged_adapter(model, merged_name=merged_name, out_dir=out_dir)

    bake_dir = args.bake_dir
    if bake_dir is None and args.push_baked_repo:
        bake_dir = os.path.join("models", f"baked_{merged_name}")

    bake_and_save(model, bake_dir=bake_dir)

    # Push ops (optional)
    maybe_push_adapter(model, repo_id=args.push_adapter_repo, private=args.private)
    maybe_push_baked(bake_dir=bake_dir, base_id=args.base_id, repo_id=args.push_baked_repo, private=args.private)

    # Final hints for Unsloth users
    print("\nHow to use the merged result with Unsloth:")
    print("- As adapter: load base with FastLanguageModel, then PeftModel.from_pretrained(model, out_dir, adapter_name=\"%s\")" % merged_name)
    if bake_dir:
        print("- As baked full model: FastLanguageModel.from_pretrained(model_name=\"%s\")" % bake_dir)


if __name__ == "__main__":
    main()
